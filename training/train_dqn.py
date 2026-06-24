"""
train_dqn.py — Stage 3: Deep Q-Network Remediation Agent
=========================================================

PURPOSE:
  Trains a DQN agent to select one of 5 remediation actions given the full
  state vector produced by build_dqn_environment.py:

    state = [115 flow features | 1 AE recon error | N attack-type softmax probs
             | 1 max-prob confidence]
    state_dim = 115 + 1 + N + 1  (N loaded from attack_type_label_map.json)

ACTION SPACE:
  0 = Block IP
  1 = Revoke Credentials
  2 = Isolate Server
  3 = Kill Process
  4 = Monitor (no action)

DQN COMPONENTS:
  - Policy network  : state_dim → 128 → 64 → 5  (Q-value head)
  - Target network  : same architecture, weights synced every TARGET_UPDATE steps
  - Experience replay: circular buffer of capacity REPLAY_CAPACITY
  - ε-greedy exploration: ε decays exponentially from 1.0 → EPS_MIN over training
  - Loss: MSE between predicted Q-value and Bellman target Q-value
  - Optimizer: Adam (lr=1e-3)

TRAINING:
  - Total episodes: NUM_EPISODES (each episode = one single-step flow decision)
  - Batch size: BATCH_SIZE (sampled from replay buffer)
  - Warmup: no training until replay buffer has >= REPLAY_START transitions
  - Logging: rolling average reward every LOG_INTERVAL episodes

EVALUATION (held-out test split):
  - Overall action-match accuracy (agent vs optimal action table)
  - Per-class breakdown: which fraction of flows of each attack type get the correct action
  - Average reward per episode on the test split
  - Monitor over-selection check: fraction of non-benign rows where agent chose Monitor

OUTPUTS:
  - models/dqn_agent.pt              — policy network state_dict (best by val reward)
  - models/dqn_training_history.json — rolling avg reward per LOG_INTERVAL episodes

USAGE (from project root):
    python training/train_dqn.py
"""

import os
import sys
import json
import logging
import copy
import random
import shutil
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
sys.path.insert(0, _SCRIPT_DIR)

from dqn_env import RemediationEnv

DATA_DIR  = os.path.join(_PROJECT_ROOT, "data",   "processed")
MODEL_DIR = os.path.join(_PROJECT_ROOT, "models")

# ─────────────────────────────────────────────
# Hyperparameters
# ─────────────────────────────────────────────
HIDDEN_1        = 128
HIDDEN_2        = 64
N_ACTIONS       = 5
BATCH_SIZE      = 64
LR              = 1e-3
GAMMA           = 0.99           # discount factor (episodes are 1-step, so γ barely matters)
REPLAY_CAPACITY = 50_000
REPLAY_START    = 1_000          # min transitions before training starts
TARGET_UPDATE   = 500            # sync target net every N steps
NUM_EPISODES    = 100_000
EPS_START       = 1.0
EPS_MIN         = 0.05
EPS_DECAY       = 0.99995        # multiplicative decay per episode
LOG_INTERVAL    = 1_000          # log rolling avg reward every N episodes
VAL_INTERVAL    = 5_000          # evaluate on val split every N episodes

# ─────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────
LOG_PATH = os.path.join(DATA_DIR, "train_dqn.log")
_sh = logging.StreamHandler()
_sh.stream = open(_sh.stream.fileno(), mode="w", encoding="utf-8", closefd=False, buffering=1)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    handlers=[_sh, logging.FileHandler(LOG_PATH, mode="w", encoding="utf-8")],
)
log = logging.getLogger(__name__)

ACTION_NAMES = {
    0: "Block IP",
    1: "Revoke Credentials",
    2: "Isolate Server",
    3: "Kill Process",
    4: "Monitor",
}


# ═══════════════════════════════════════════════════════════
# MODEL DEFINITION
# ═══════════════════════════════════════════════════════════
class DQNNetwork(nn.Module):
    """
    Deep Q-Network for the 5-action remediation problem.

    Architecture: state_dim → 128 → 64 → 5
    Output: raw Q-values for each of the 5 actions (no activation on output layer).
    Action selection: argmax(Q-values).

    Parameters
    ----------
    state_dim : int  — dimension of the DQN state vector
    n_actions : int  — number of discrete actions (5)
    """

    def __init__(self, state_dim: int, n_actions: int = N_ACTIONS):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, HIDDEN_1),
            nn.ReLU(),
            nn.Linear(HIDDEN_1, HIDDEN_2),
            nn.ReLU(),
            nn.Linear(HIDDEN_2, n_actions),   # linear output — Q-values
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ═══════════════════════════════════════════════════════════
# EXPERIENCE REPLAY BUFFER
# ═══════════════════════════════════════════════════════════
class ReplayBuffer:
    """
    Circular experience replay buffer.

    Stores (state, action, reward, next_state, done) tuples.
    Sampling is uniform random — no prioritization.

    Parameters
    ----------
    capacity : int  — maximum number of transitions to store
    """

    def __init__(self, capacity: int):
        self._buf = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self._buf.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        batch = random.sample(self._buf, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states,      dtype=np.float32),
            np.array(actions,     dtype=np.int64),
            np.array(rewards,     dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones,       dtype=np.float32),
        )

    def __len__(self):
        return len(self._buf)


# ═══════════════════════════════════════════════════════════
# STEP 1 — Load DQN splits
# ═══════════════════════════════════════════════════════════
def load_dqn_data(data_dir: str):
    """Load pre-built DQN state/action/label splits."""
    log.info("[STEP 1] Loading DQN environment arrays ...")

    def _load(prefix):
        s = np.load(os.path.join(data_dir, f"dqn_{prefix}_states.npy"))
        a = np.load(os.path.join(data_dir, f"dqn_{prefix}_actions.npy"))
        l = np.load(os.path.join(data_dir, f"dqn_{prefix}_labels.npy"))
        return s, a, l

    tr = _load("train")
    va = _load("val")
    te = _load("test")

    state_dim = tr[0].shape[1]
    log.info(f"  Train: {tr[0].shape[0]:>7,}  Val: {va[0].shape[0]:>7,}  "
             f"Test: {te[0].shape[0]:>7,}  state_dim={state_dim}")

    # Load label map for reporting
    with open(os.path.join(data_dir, "attack_type_label_map.json")) as f:
        label_map = {int(k): v for k, v in json.load(f).items()}

    return tr, va, te, state_dim, label_map


# ═══════════════════════════════════════════════════════════
# STEP 2 — ε-greedy action selection
# ═══════════════════════════════════════════════════════════
def select_action(policy_net, state_tensor, epsilon: float, device) -> int:
    """ε-greedy: explore with probability ε, exploit otherwise."""
    if random.random() < epsilon:
        return random.randint(0, N_ACTIONS - 1)
    with torch.no_grad():
        q = policy_net(state_tensor.unsqueeze(0).to(device))
        return int(q.argmax(dim=1).item())


# ═══════════════════════════════════════════════════════════
# STEP 3 — One gradient update step
# ═══════════════════════════════════════════════════════════
def optimize(policy_net, target_net, optimizer, replay_buf, device) -> float:
    """
    Sample a mini-batch from the replay buffer and do one gradient step.

    Bellman target: r + γ * max_a' Q_target(s', a')
    (For single-step episodes, done=True always, so the target reduces to r.)
    Loss: MSE(Q_policy(s, a), target)

    Returns
    -------
    loss_val : float
    """
    states, actions, rewards, next_states, dones = replay_buf.sample(BATCH_SIZE)

    states_t      = torch.tensor(states,      device=device)
    actions_t     = torch.tensor(actions,     device=device, dtype=torch.long)
    rewards_t     = torch.tensor(rewards,     device=device)
    next_states_t = torch.tensor(next_states, device=device)
    dones_t       = torch.tensor(dones,       device=device)

    # Q(s, a) from policy net
    q_values  = policy_net(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)

    # Bellman target
    with torch.no_grad():
        next_q    = target_net(next_states_t).max(dim=1).values
        targets   = rewards_t + GAMMA * next_q * (1.0 - dones_t)

    loss = F.mse_loss(q_values, targets)
    optimizer.zero_grad()
    loss.backward()
    # Gradient clipping (prevents exploding gradients on outlier batches)
    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=10.0)
    optimizer.step()
    return loss.item()


# ═══════════════════════════════════════════════════════════
# STEP 4 — Evaluate on a split (val or test)
# ═══════════════════════════════════════════════════════════
def evaluate_split(policy_net, states, actions, labels, device,
                   split_name: str = "test") -> dict:
    """
    Run the greedy (ε=0) policy over the full split and compute:
      - overall action-match accuracy
      - per-class breakdown (fraction correct, confusion over wrong actions)
      - average reward per episode
      - Monitor over-selection rate on non-benign rows

    Parameters
    ----------
    policy_net : DQNNetwork (eval mode)
    states     : np.ndarray  (M, state_dim)
    actions    : np.ndarray  (M,)  int64 — ground-truth optimal actions
    labels     : np.ndarray  (M,)  str   — string attack-type labels
    device     : torch.device
    split_name : str  — used in log messages

    Returns
    -------
    metrics : dict
    """
    policy_net.eval()
    log.info(f"[EVAL] Evaluating on {split_name} split ({len(states):,} rows) ...")

    batch_size = 1024
    all_preds  = []
    with torch.no_grad():
        for start in range(0, len(states), batch_size):
            batch = torch.tensor(states[start:start + batch_size],
                                 dtype=torch.float32, device=device)
            q      = policy_net(batch)
            preds  = q.argmax(dim=1).cpu().numpy()
            all_preds.append(preds)

    preds = np.concatenate(all_preds)

    # ── Reward computation ────────────────────────────────
    rewards = []
    for i in range(len(preds)):
        opt  = int(actions[i])
        pred = int(preds[i])
        lbl  = labels[i]
        is_benign = (lbl == "Benign")
        if pred == opt:
            rewards.append(10.0)
        elif is_benign:
            rewards.append(-3.0 if pred != 4 else 10.0)
        else:
            rewards.append(-5.0)
    rewards = np.array(rewards)

    overall_acc  = float(np.mean(preds == actions))
    avg_reward   = float(np.mean(rewards))

    log.info(f"  Overall action-match accuracy : {overall_acc:.4f}  ({overall_acc*100:.2f}%)")
    log.info(f"  Average reward per episode    : {avg_reward:.4f}")

    # ── Per-class breakdown ───────────────────────────────
    log.info("")
    log.info("  Per-class action accuracy:")
    log.info(f"  {'Class':<30s}  {'Total':>7}  {'Correct':>8}  {'Accuracy':>9}  "
             f"  Action breakdown (wrong)")
    unique_labels = sorted(set(labels.tolist()))
    monitor_on_attack = 0
    total_attack       = 0

    class_results = {}
    for cls in unique_labels:
        mask  = labels == cls
        total = int(mask.sum())
        if total == 0:
            continue
        cls_preds = preds[mask]
        cls_opts  = actions[mask]
        correct   = int((cls_preds == cls_opts).sum())
        acc_cls   = correct / total

        # Wrong-action breakdown
        wrong_mask   = cls_preds != cls_opts
        wrong_preds  = cls_preds[wrong_mask]
        wrong_counts = {}
        for a in range(N_ACTIONS):
            cnt = int(np.sum(wrong_preds == a))
            if cnt > 0:
                wrong_counts[ACTION_NAMES[a]] = cnt

        wrong_str = ", ".join(f"{k}:{v}" for k, v in sorted(wrong_counts.items(),
                                                              key=lambda t: -t[1]))

        log.info(
            f"  {cls:<30s}  {total:>7,}  {correct:>8,}  {acc_cls:>8.4f}"
            + (f"    [{wrong_str}]" if wrong_str else "  [all correct]")
        )
        class_results[cls] = {"total": total, "correct": correct, "accuracy": acc_cls}

        # Monitor-over-selection tracking (non-benign only)
        if cls != "Benign":
            total_attack       += total
            monitor_on_attack  += int(np.sum(cls_preds == 4))

    # ── Monitor over-selection check ──────────────────────
    log.info("")
    if total_attack > 0:
        monitor_rate = monitor_on_attack / total_attack
        flag = "⚠️  MONITOR OVER-SELECTION" if monitor_rate > 0.30 else "✓"
        log.info(
            f"  Monitor-on-Attack rate: {monitor_rate:.4f} ({monitor_rate*100:.2f}%)  {flag}"
        )
        if monitor_rate > 0.30:
            log.warning(
                "  The agent is choosing Monitor on more than 30% of real attack flows. "
                "This is likely a degenerate local optimum — consider extending training "
                "or adjusting the reward structure."
            )
    else:
        log.info("  (No non-benign rows in this split — Monitor rate not computed)")

    policy_net.train()
    return {
        "overall_accuracy": overall_acc,
        "avg_reward": avg_reward,
        "per_class": class_results,
        "monitor_on_attack_rate": monitor_on_attack / max(total_attack, 1),
    }


# ═══════════════════════════════════════════════════════════
# STEP 5 — Training loop
# ═══════════════════════════════════════════════════════════
def train(policy_net, target_net, optimizer, env, replay_buf,
          val_states, val_actions, val_labels, device):
    """
    Main DQN training loop.

    Returns
    -------
    history : list[dict]  rolling avg reward per LOG_INTERVAL episodes
    """
    log.info(f"[STEP 5] Training for {NUM_EPISODES:,} episodes ...")
    log.info(f"  Replay capacity={REPLAY_CAPACITY:,}  warmup={REPLAY_START:,}  "
             f"target_sync={TARGET_UPDATE}")
    log.info(f"  ε: {EPS_START} → {EPS_MIN}  decay={EPS_DECAY}/episode")

    epsilon      = EPS_START
    step_count   = 0          # total optimizer steps taken
    history      = []
    recent_rewards = deque(maxlen=LOG_INTERVAL)
    best_val_acc = -1.0
    best_state   = None

    for episode in range(1, NUM_EPISODES + 1):
        obs, info = env.reset()
        state     = torch.tensor(obs, dtype=torch.float32)

        action = select_action(policy_net, state, epsilon, device)
        next_obs, reward, terminated, _, _ = env.step(action)

        replay_buf.push(obs, action, reward, next_obs, float(terminated))
        recent_rewards.append(reward)

        # ── Gradient update ─────────────────────────────
        if len(replay_buf) >= REPLAY_START:
            optimize(policy_net, target_net, optimizer, replay_buf, device)
            step_count += 1

            # Sync target network
            if step_count % TARGET_UPDATE == 0:
                target_net.load_state_dict(policy_net.state_dict())

        # ── ε decay ─────────────────────────────────────
        epsilon = max(EPS_MIN, epsilon * EPS_DECAY)

        # ── Periodic logging ─────────────────────────────
        if episode % LOG_INTERVAL == 0:
            avg_r = float(np.mean(recent_rewards))
            log.info(
                f"  Episode {episode:>8,}/{NUM_EPISODES:,}  "
                f"ε={epsilon:.4f}  "
                f"replay={len(replay_buf):>6,}  "
                f"avg_reward(last {LOG_INTERVAL})={avg_r:.3f}"
            )
            history.append({"episode": episode, "avg_reward": avg_r, "epsilon": epsilon})

        # ── Periodic val evaluation + best model save ────
        if episode % VAL_INTERVAL == 0 and len(replay_buf) >= REPLAY_START:
            policy_net.eval()
            val_metrics = evaluate_split(
                policy_net, val_states, val_actions, val_labels, device,
                split_name=f"val (ep {episode:,})"
            )
            val_acc = val_metrics["overall_accuracy"]
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state   = copy.deepcopy(policy_net.state_dict())
                log.info(f"  ✓ New best val accuracy: {best_val_acc:.4f}  (weights saved)")
            policy_net.train()

    # Restore best weights
    if best_state is not None:
        policy_net.load_state_dict(best_state)
        log.info(f"  Best val accuracy restored: {best_val_acc:.4f}")

    return history


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════
def main():
    log.info("=" * 70)
    log.info("train_dqn.py — Stage 3: DQN Remediation Agent")
    log.info("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"  Device: {device}")
    if device.type == "cuda":
        log.info(f"  GPU   : {torch.cuda.get_device_name(0)}")

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    # ── Load data ────────────────────────────────────────
    (tr_s, tr_a, tr_l), (va_s, va_a, va_l), (te_s, te_a, te_l), \
        state_dim, label_map = load_dqn_data(DATA_DIR)

    # ── Environment ──────────────────────────────────────
    env = RemediationEnv(tr_s, tr_a, tr_l, seed=42)

    # ── Networks ─────────────────────────────────────────
    log.info("[MODEL] Building DQN networks ...")
    policy_net = DQNNetwork(state_dim).to(device)
    target_net = DQNNetwork(state_dim).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    total_params = sum(p.numel() for p in policy_net.parameters())
    log.info(f"  Architecture : {state_dim} → {HIDDEN_1} → {HIDDEN_2} → {N_ACTIONS}")
    log.info(f"  Total params : {total_params:,}")

    optimizer  = torch.optim.Adam(policy_net.parameters(), lr=LR)
    replay_buf = ReplayBuffer(REPLAY_CAPACITY)

    # ── Train ────────────────────────────────────────────
    history = train(
        policy_net, target_net, optimizer, env, replay_buf,
        va_s, va_a, va_l, device,
    )

    # ── Back up previous model before overwriting ────────
    model_path   = os.path.join(MODEL_DIR, "dqn_agent.pt")
    archive_dir  = os.path.join(MODEL_DIR, "archive")
    os.makedirs(archive_dir, exist_ok=True)
    backup_path  = os.path.join(archive_dir, "dqn_agent_pre_web_merge.pt")
    if os.path.exists(model_path):
        shutil.copy2(model_path, backup_path)
        log.info(f"  Previous model backed up → {backup_path}")
    else:
        log.info("  No previous dqn_agent.pt found — skipping backup.")

    # ── Save new model ────────────────────────────────────
    torch.save(policy_net.state_dict(), model_path)
    log.info(f"  New model saved → {model_path}")

    history_path = os.path.join(MODEL_DIR, "dqn_training_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    log.info(f"  Training history saved → {history_path}")

    # ── Final test evaluation ─────────────────────────────
    log.info("")
    log.info("=" * 70)
    log.info("FINAL EVALUATION ON HELD-OUT TEST SET")
    log.info("=" * 70)
    policy_net.eval()
    test_metrics = evaluate_split(
        policy_net, te_s, te_a, te_l, device, split_name="test"
    )

    log.info("")
    log.info("=" * 70)
    log.info("✓  train_dqn.py complete.")
    log.info(f"   Test action-match accuracy : {test_metrics['overall_accuracy']:.4f}")
    log.info(f"   Test average reward        : {test_metrics['avg_reward']:.4f}")
    log.info(f"   Monitor-on-Attack rate     : "
             f"{test_metrics['monitor_on_attack_rate']:.4f}")
    log.info(f"   Episodes trained           : {NUM_EPISODES:,}")
    log.info(f"   Saved → {model_path}")
    log.info("")
    log.info("   Web_Brute_Force / Web_XSS per-class accuracy (both → Revoke Credentials):")
    for cls in ("Web_Brute_Force", "Web_XSS"):
        if cls in test_metrics.get("per_class", {}):
            r = test_metrics["per_class"][cls]
            log.info(f"     {cls:<20s} : {r['correct']:>5,}/{r['total']:>5,}  "
                     f"({r['accuracy']:.4f})")
        else:
            log.info(f"     {cls:<20s} : not found in per_class results")
    log.info("=" * 70)


if __name__ == "__main__":
    main()
