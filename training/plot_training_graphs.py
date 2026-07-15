"""
plot_training_graphs.py - Training Visualisation for Project Report
====================================================================

Reads the saved training history JSON files from models/ and produces
publication-quality matplotlib figures for all three training stages:

  Stage 1: Autoencoder  (training_history.json)
  Stage 2: AttackTypeNN (attack_type_nn_history.json)
  Stage 3: DQN Agent    (dqn_training_history.json)

OUTPUT FIGURES (saved to reports/figures/):
  1. autoencoder_training.png    - Train/Val MSE loss + generalisation gap
  2. attack_type_nn_training.png - Train/Val Cross-Entropy loss + improvement bars
  3. dqn_training.png            - 2x2: rolling reward, epsilon-decay, histogram, scatter
  4. all_stages_summary.png      - unified 2x2 summary (ideal for report intro/conclusion)

USAGE (from project root):
    python training/plot_training_graphs.py
"""

import os, sys, json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator

_SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
MODEL_DIR  = os.path.join(_PROJECT_ROOT, "models")
REPORT_DIR = os.path.join(_PROJECT_ROOT, "reports", "figures")
os.makedirs(REPORT_DIR, exist_ok=True)

plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "font.size":         11,
    "axes.titlesize":    13,
    "axes.labelsize":    11,
    "axes.titleweight":  "bold",
    "axes.grid":         True,
    "grid.alpha":        0.35,
    "grid.linestyle":    "--",
    "grid.linewidth":    0.7,
    "lines.linewidth":   2.2,
    "lines.antialiased": True,
    "legend.framealpha": 0.85,
    "legend.edgecolor":  "#cccccc",
    "figure.dpi":        150,
    "savefig.dpi":       200,
    "savefig.bbox":      "tight",
    "savefig.facecolor": "white",
})

C_TRAIN  = "#2563EB"
C_VAL    = "#DC2626"
C_REWARD = "#16A34A"
C_EPS    = "#9333EA"
C_FILL   = "#93C5FD"
C_BEST   = "#F59E0B"


def _load_json(path, label):
    if not os.path.exists(path):
        print(f"  [WARN] {label} not found -- skipping.", file=sys.stderr)
        return None
    with open(path, "r") as f:
        return json.load(f)


def _best_marker(ax, epochs, values, color=C_BEST, minimize=True):
    idx = int(np.argmin(values) if minimize else np.argmax(values))
    ax.plot(epochs[idx], values[idx], marker="*", markersize=13,
            color=color, zorder=10, label=f"Best ep {epochs[idx]}")
    ax.annotate(f"  {values[idx]:.4f}", xy=(epochs[idx], values[idx]),
                fontsize=9, color=color, va="center", ha="left")


def _smooth(values, window=5):
    if len(values) < window:
        return np.array(values)
    return np.convolve(values, np.ones(window)/window, mode="same")


def _tag(ax, text, color):
    ax.text(0.01, 0.99, text, transform=ax.transAxes,
            fontsize=8, fontweight="bold", color="white", va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.25", facecolor=color, alpha=0.85, linewidth=0))


# =======================================================================
# Figure 1 - Autoencoder
# =======================================================================
def plot_autoencoder(history, save_path):
    tl = history["train_loss"]
    vl = history["val_loss"]
    ep = list(range(1, len(tl)+1))

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        "Stage 1 - Autoencoder Training\n"
        "115->80->48->24->48->80->115  |  MSELoss  |  Adam lr=1e-3  |  Early-stop patience=5",
        fontsize=11, fontweight="bold", y=1.02)

    ax.plot(ep, tl, color=C_TRAIN, label="Train MSE", alpha=0.9)
    ax.plot(ep, vl, color=C_VAL,   label="Val MSE",   alpha=0.9)
    ax.fill_between(ep, tl, vl, alpha=0.12, color=C_FILL, label="Gap")
    _best_marker(ax, ep, vl, minimize=True)
    ax.set_title("MSE Reconstruction Loss per Epoch")
    ax.set_xlabel("Epoch"); ax.set_ylabel("MSE Loss")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.legend(loc="upper right", fontsize=9)
    if len(ep) < 50:
        ax.axvline(len(ep), color="#6B7280", linestyle=":", linewidth=1.5)
        ax.text(len(ep)-0.2, max(tl)*0.97, f"Early stop\n(ep {len(ep)})",
                ha="right", va="top", fontsize=8, color="#6B7280")

    gap = [v-t for t,v in zip(tl, vl)]
    ax2.bar(ep, gap, color=C_FILL, edgecolor=C_VAL, linewidth=0.6, label="Val - Train")
    ax2.axhline(0, color="#374151", linewidth=1.0, linestyle="--")
    ax2.set_title("Generalisation Gap (Val - Train MSE)")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Delta MSE")
    ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax2.legend(fontsize=9)

    fig.tight_layout()
    fig.savefig(save_path)
    print(f"  Saved -> {save_path}")
    plt.close(fig)


# =======================================================================
# Figure 2 - Attack-Type NN
# =======================================================================
def plot_attack_type_nn(history, save_path):
    tl = history["train_loss"]
    vl = history["val_loss"]
    ep = list(range(1, len(tl)+1))

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        "Stage 2 - Attack-Type Classifier Training\n"
        "115->128->64->N  |  Weighted CrossEntropyLoss  |  Adam lr=1e-3  |  Early-stop patience=5",
        fontsize=11, fontweight="bold", y=1.02)

    ax.plot(ep, tl, color=C_TRAIN, label="Train CE", alpha=0.9)
    ax.plot(ep, vl, color=C_VAL,   label="Val CE",   alpha=0.9)
    ax.fill_between(ep, tl, vl, alpha=0.12, color=C_FILL, label="Gap")
    _best_marker(ax, ep, vl, minimize=True)
    ax.set_title("Cross-Entropy Loss per Epoch")
    ax.set_xlabel("Epoch"); ax.set_ylabel("CE Loss")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.legend(loc="upper right", fontsize=9)
    if len(ep) < 50:
        ax.axvline(len(ep), color="#6B7280", linestyle=":", linewidth=1.5)
        ax.text(len(ep)-0.2, max(max(tl), max(vl))*0.97,
                f"Early stop\n(ep {len(ep)})",
                ha="right", va="top", fontsize=8, color="#6B7280")

    improv = [0.0] + [vl[i-1]-vl[i] for i in range(1, len(vl))]
    colors = [C_REWARD if d > 0 else C_VAL for d in improv]
    ax2.bar(ep, improv, color=colors, edgecolor="white", linewidth=0.4)
    ax2.axhline(0, color="#374151", linewidth=1.0, linestyle="--")
    ax2.set_title("Val Loss Change per Epoch\n(green=decrease/improvement, red=increase/worse)")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Delta Val Loss")
    ax2.xaxis.set_major_locator(MaxNLocator(integer=True))

    fig.tight_layout()
    fig.savefig(save_path)
    print(f"  Saved -> {save_path}")
    plt.close(fig)


# =======================================================================
# Figure 3 - DQN Agent (2x2)
# =======================================================================
def plot_dqn(history, save_path):
    episodes   = [h["episode"]    for h in history]
    avg_reward = [h["avg_reward"] for h in history]
    epsilons   = [h["epsilon"]    for h in history]
    ep_k       = [e/1000 for e in episodes]

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle(
        "Stage 3 - DQN Remediation Agent Training\n"
        "state_dim->128->64->5  |  MSE Bellman loss  |  Adam lr=1e-3  |  epsilon-greedy exploration",
        fontsize=11, fontweight="bold", y=1.01)

    smoothed = _smooth(avg_reward, window=7)

    # Top-left: rolling reward
    ax = axes[0,0]
    ax.plot(ep_k, avg_reward, color=C_REWARD, alpha=0.30, linewidth=1.2, label="Raw")
    ax.plot(ep_k, smoothed,   color=C_REWARD, linewidth=2.5, label="Smoothed (7-pt MA)")
    ax.fill_between(ep_k, avg_reward, smoothed, alpha=0.15, color=C_REWARD)
    ax.axhline(10.0, color=C_BEST,    linestyle="--", linewidth=1.2, label="Max reward=10")
    ax.axhline(0.0,  color="#6B7280", linestyle=":",  linewidth=0.8)
    _best_marker(ax, ep_k, avg_reward, minimize=False)
    ax.set_title("Rolling Avg Reward per 1,000 Episodes")
    ax.set_xlabel("Episodes (x1000)"); ax.set_ylabel("Avg Reward")
    ax.legend(fontsize=8.5, loc="lower right")

    # Top-right: epsilon decay
    ax2 = axes[0,1]
    ax2.plot(ep_k, epsilons, color=C_EPS, linewidth=2.2, label="epsilon")
    ax2.fill_between(ep_k, epsilons, 0, alpha=0.15, color=C_EPS)
    ax2.axhline(0.05, color="#6B7280", linestyle="--", linewidth=1.2, label="epsilon_min=0.05")
    ax2.set_title("Exploration Rate (epsilon) Decay")
    ax2.set_xlabel("Episodes (x1000)"); ax2.set_ylabel("epsilon")
    ax2.set_ylim(-0.02, 1.05); ax2.legend(fontsize=9)

    # Bottom-left: reward histogram
    ax3 = axes[1,0]
    ax3.hist(avg_reward, bins=30, color=C_REWARD, edgecolor="white", linewidth=0.5, alpha=0.85)
    ax3.axvline(np.mean(avg_reward),   color=C_BEST, linewidth=2.0, linestyle="--",
                label=f"Mean={np.mean(avg_reward):.2f}")
    ax3.axvline(np.median(avg_reward), color=C_VAL,  linewidth=2.0, linestyle="--",
                label=f"Median={np.median(avg_reward):.2f}")
    ax3.set_title("Distribution of Rolling Avg Rewards")
    ax3.set_xlabel("Avg Reward"); ax3.set_ylabel("Frequency")
    ax3.legend(fontsize=9)

    # Bottom-right: reward vs epsilon scatter
    ax4 = axes[1,1]
    sc = ax4.scatter(epsilons, avg_reward, c=ep_k, cmap="viridis", s=22, alpha=0.75, zorder=5)
    cb = fig.colorbar(sc, ax=ax4, shrink=0.85)
    cb.set_label("Episode (x1000)", fontsize=9)
    ax4.axhline(0, color="#6B7280", linestyle=":", linewidth=0.8)
    ax4.set_title("Reward vs Epsilon (colour=training progress)")
    ax4.set_xlabel("Epsilon"); ax4.set_ylabel("Avg Reward")

    fig.tight_layout()
    fig.savefig(save_path)
    print(f"  Saved -> {save_path}")
    plt.close(fig)


# =======================================================================
# Figure 4 - All-Stages Summary (2x2)
# =======================================================================
def plot_summary(ae_h, nn_h, dqn_h, save_path):
    fig = plt.figure(figsize=(14, 10))
    fig.suptitle(
        "Network Intrusion Remediation System - Training Summary\n"
        "Stage 1: Autoencoder  |  Stage 2: Attack-Type NN  |  Stage 3: DQN Agent",
        fontsize=13, fontweight="bold", y=1.01)
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.40, wspace=0.30)

    # Panel 1 - AE
    ax1 = fig.add_subplot(gs[0,0])
    tl, vl = ae_h["train_loss"], ae_h["val_loss"]
    ep = list(range(1, len(tl)+1))
    ax1.plot(ep, tl, color=C_TRAIN, label="Train MSE", linewidth=2)
    ax1.plot(ep, vl, color=C_VAL,   label="Val MSE",   linewidth=2)
    ax1.fill_between(ep, tl, vl, alpha=0.15, color=C_FILL)
    ax1.axvline(len(ep), color="#9CA3AF", linestyle=":", linewidth=1.0)
    ax1.set_title("Autoencoder - MSE Loss", pad=8)
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("MSE")
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax1.legend(fontsize=8.5)
    _tag(ax1, "Stage 1", "#2563EB")

    # Panel 2 - NN
    ax2 = fig.add_subplot(gs[0,1])
    tl, vl = nn_h["train_loss"], nn_h["val_loss"]
    ep = list(range(1, len(tl)+1))
    ax2.plot(ep, tl, color=C_TRAIN, label="Train CE", linewidth=2)
    ax2.plot(ep, vl, color=C_VAL,   label="Val CE",   linewidth=2)
    ax2.fill_between(ep, tl, vl, alpha=0.15, color=C_FILL)
    ax2.axvline(len(ep), color="#9CA3AF", linestyle=":", linewidth=1.0)
    ax2.set_title("Attack-Type NN - Cross-Entropy Loss", pad=8)
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("CE Loss")
    ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax2.legend(fontsize=8.5)
    _tag(ax2, "Stage 2", "#DC2626")

    # Panel 3 - DQN reward
    ax3 = fig.add_subplot(gs[1,0])
    episodes   = [h["episode"]/1000 for h in dqn_h]
    avg_reward = [h["avg_reward"]    for h in dqn_h]
    smoothed   = _smooth(avg_reward, window=7)
    ax3.plot(episodes, avg_reward, color=C_REWARD, alpha=0.25, linewidth=1.0)
    ax3.plot(episodes, smoothed,   color=C_REWARD, linewidth=2.5, label="Avg Reward (7-pt MA)")
    ax3.axhline(10.0, color=C_BEST, linestyle="--", linewidth=1.2, label="Max reward=10")
    ax3.set_title("DQN Agent - Rolling Avg Reward", pad=8)
    ax3.set_xlabel("Episodes (x1000)"); ax3.set_ylabel("Avg Reward")
    ax3.legend(fontsize=8.5)
    _tag(ax3, "Stage 3", "#16A34A")

    # Panel 4 - DQN epsilon
    ax4 = fig.add_subplot(gs[1,1])
    epsilons = [h["epsilon"] for h in dqn_h]
    ax4.plot(episodes, epsilons, color=C_EPS, linewidth=2.2)
    ax4.fill_between(episodes, epsilons, 0, alpha=0.15, color=C_EPS)
    ax4.axhline(0.05, color="#6B7280", linestyle="--", linewidth=1.2, label="epsilon_min=0.05")
    ax4.set_title("DQN Agent - Epsilon Decay", pad=8)
    ax4.set_xlabel("Episodes (x1000)"); ax4.set_ylabel("Epsilon")
    ax4.set_ylim(-0.02, 1.05)
    ax4.legend(fontsize=8.5)
    _tag(ax4, "Stage 3", "#9333EA")

    fig.savefig(save_path)
    print(f"  Saved -> {save_path}")
    plt.close(fig)


# =======================================================================
# MAIN
# =======================================================================
def main():
    print("=" * 65)
    print("plot_training_graphs.py - Training Visualisation for Report")
    print("=" * 65)
    print(f"  Model dir  : {MODEL_DIR}")
    print(f"  Output dir : {REPORT_DIR}")
    print()

    ae_h  = _load_json(os.path.join(MODEL_DIR, "training_history.json"),       "Autoencoder history")
    nn_h  = _load_json(os.path.join(MODEL_DIR, "attack_type_nn_history.json"), "AttackTypeNN history")
    dqn_h = _load_json(os.path.join(MODEL_DIR, "dqn_training_history.json"),   "DQN history")

    if ae_h:
        print("[1/4] Plotting Autoencoder training curves ...")
        plot_autoencoder(ae_h, os.path.join(REPORT_DIR, "autoencoder_training.png"))

    if nn_h:
        print("[2/4] Plotting Attack-Type NN training curves ...")
        plot_attack_type_nn(nn_h, os.path.join(REPORT_DIR, "attack_type_nn_training.png"))

    if dqn_h:
        print("[3/4] Plotting DQN training curves ...")
        plot_dqn(dqn_h, os.path.join(REPORT_DIR, "dqn_training.png"))

    if ae_h and nn_h and dqn_h:
        print("[4/4] Plotting all-stages summary panel ...")
        plot_summary(ae_h, nn_h, dqn_h, os.path.join(REPORT_DIR, "all_stages_summary.png"))

    print()
    print(f"All figures saved to: {REPORT_DIR}")
    print("=" * 65)


if __name__ == "__main__":
    main()
