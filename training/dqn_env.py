"""
dqn_env.py — Gymnasium Environment for DQN Remediation Agent (Stage 3)
=======================================================================

PURPOSE:
  Defines RemediationEnv, a custom Gymnasium environment in which the DQN
  agent learns to choose one of 5 remediation actions for anomalous network
  flows.

DESIGN RATIONALE — SINGLE-STEP EPISODES:
  Each episode consists of exactly ONE step:
    1. reset() → sample a random flow state vector from the training set
    2. step(action) → compute reward, return (next_state, reward, terminated=True, ...)

  Single-step episodes are appropriate here because the remediation decision
  for one network flow is INDEPENDENT of previous decisions:
    - There is no temporal dependency between consecutive flows in the dataset
    - The optimal action for a flow depends only on its own features and attack
      type, not on the history of past actions
    - The dataset does not contain sequential episodes modelling network state
      evolution over time

  A multi-step Markov Decision Process would require a modelled sequence of
  network state transitions (e.g., what happens to the network after we block
  an IP?) which is outside the scope of this dataset-driven simulation.
  Single-step MDP is the correct abstraction for per-flow decision-making.

STATE SPACE:
  Box(state_dim,) where state_dim = 115 + 1 + N + 1  (N = number of attack-type classes)
  Populated from dqn_train_states.npy (or dqn_val/test for evaluation).

ACTION SPACE:
  Discrete(5):
    0 = Block IP
    1 = Revoke Credentials
    2 = Isolate Server
    3 = Kill Process
    4 = Monitor (no action)

REWARD STRUCTURE:
  +10  correct action   (chosen == optimal action for this flow)
  -5   wrong action on a real attack flow (chosen != optimal AND label != Benign)
  -3   any non-Monitor action on a genuinely benign flow (false-alarm cost)
  +10  Monitor on a genuinely benign flow (correct no-action)

USAGE:
  from training.dqn_env import RemediationEnv
  env = RemediationEnv(states, actions, labels)
  obs, info = env.reset()
  obs, reward, terminated, truncated, info = env.step(action)
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces


class RemediationEnv(gym.Env):
    """
    Single-step Gymnasium environment for network intrusion remediation.

    Each episode samples one flow from the supplied state array, presents its
    state vector to the agent, receives one action, and terminates immediately.

    Parameters
    ----------
    states   : np.ndarray  float32  (M × state_dim)
        Pre-computed DQN state vectors (from build_dqn_environment.py).
    actions  : np.ndarray  int64    (M,)
        Ground-truth optimal action per row (used only for reward computation).
        The agent never observes this array directly.
    labels   : np.ndarray  str      (M,)
        String attack-type labels per row (used to distinguish benign vs attack
        in the reward function, and for per-class evaluation after training).
    seed     : int  (optional)
        Random seed for reproducibility.
    """

    metadata = {"render_modes": []}

    def __init__(self, states: np.ndarray, actions: np.ndarray,
                 labels: np.ndarray, seed: int = 42):
        super().__init__()

        self._states  = states.astype(np.float32)
        self._actions = actions.astype(np.int64)
        self._labels  = labels

        state_dim = self._states.shape[1]
        n_actions = 5

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(state_dim,), dtype=np.float32,
        )
        self.action_space = spaces.Discrete(n_actions)

        self._rng = np.random.default_rng(seed=seed)
        self._current_idx: int = 0

    # ─────────────────────────────────────────────
    # Gymnasium API
    # ─────────────────────────────────────────────

    def reset(self, *, seed=None, options=None):
        """
        Sample a random flow from the state array and return its state vector.

        Returns
        -------
        observation : np.ndarray  float32  (state_dim,)
        info        : dict  {"idx": int, "label": str, "optimal_action": int}
        """
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed=seed)

        self._current_idx = int(self._rng.integers(0, len(self._states)))
        obs  = self._states[self._current_idx].copy()
        info = {
            "idx":            self._current_idx,
            "label":          self._labels[self._current_idx],
            "optimal_action": int(self._actions[self._current_idx]),
        }
        return obs, info

    def step(self, action: int):
        """
        Apply the chosen action to the current flow and compute the reward.

        Reward structure
        ----------------
        +10   correct action == optimal action for this flow
        -5    wrong action AND flow is a real attack (not Benign)
        -3    any non-Monitor action on a Benign flow (false-alarm cost)
        +10   Monitor (4) chosen on a Benign flow (correct no-action)

        Each episode is exactly ONE step: terminated is always True.
        Truncated is always False (no time limit).

        Parameters
        ----------
        action : int  in {0, 1, 2, 3, 4}

        Returns
        -------
        observation : np.ndarray  float32  — next state (new sample from reset)
        reward      : float
        terminated  : bool  (always True — single-step episode)
        truncated   : bool  (always False)
        info        : dict
        """
        idx            = self._current_idx
        optimal_action = int(self._actions[idx])
        label          = self._labels[idx]
        is_benign      = (label == "Benign")

        # ── Reward computation ────────────────────────────
        if action == optimal_action:
            reward = 10.0
        elif is_benign:
            # Wrong choice on benign = false alarm
            if action != 4:   # 4 = Monitor
                reward = -3.0
            else:
                reward = 10.0  # Monitor on benign is always correct
        else:
            # Wrong action on a real attack
            reward = -5.0

        # Each episode is one flow decision — terminate immediately
        terminated = True
        truncated  = False

        # Sample next state for the caller's convenience
        # (caller should call reset() for the next episode, but we return a
        # valid obs so step() never returns None)
        next_idx = int(self._rng.integers(0, len(self._states)))
        next_obs = self._states[next_idx].copy()

        info = {
            "idx":            idx,
            "label":          label,
            "optimal_action": optimal_action,
            "chosen_action":  int(action),
            "reward":         reward,
        }
        return next_obs, reward, terminated, truncated, info

    def render(self):
        """No visual rendering — not needed for tabular data."""
        pass

    def close(self):
        pass
