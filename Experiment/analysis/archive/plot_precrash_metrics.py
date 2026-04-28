"""Plot training metrics before Judge crash (steps 0-47) with EMA overlay.

Data sourced from W&B run kudzrfba (atts-grpo / 8b-sft-2gpu).
Judge crashed at 11:48 UTC Apr 15 -> val acc zeroed from step 48 onward.
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ---------- Raw data from W&B history (kudzrfba) ----------

steps_val = list(range(0, 76))  # 0..75

acc = [
    0.1625, 0.175, 0.175, 0.2125, 0.15, 0.175, 0.2125, 0.225, 0.15, 0.2125,
    0.15, 0.2375, 0.2125, 0.2375, 0.1875, 0.225, 0.2, 0.225, 0.175, 0.2125,
    0.2375, 0.175, 0.2, 0.2625, 0.1875, 0.2, 0.175, 0.175, 0.25, 0.1375,
    0.2125, 0.1375, 0.2, 0.2125, 0.2, 0.2, 0.1875, 0.15, 0.2125, 0.2,
    0.25, 0.175, 0.2375, 0.1625, 0.125, 0.1875, 0.1875, 0.175,
    # step 48-75: Judge down, all zeros
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
]

discovery = [
    0.375, 0.45, 0.4875, 0.475, 0.4125, 0.425, 0.4875, 0.3875, 0.475, 0.5125,
    0.4625, 0.4625, 0.4875, 0.525, 0.475, 0.45, 0.4625, 0.4875, 0.4625, 0.5375,
    0.475, 0.5625, 0.5, 0.5125, 0.4375, 0.5, 0.45, 0.45, 0.5375, 0.375,
    0.5375, 0.5375, 0.45, 0.4875, 0.4375, 0.4375, 0.3875, 0.3625, 0.425, 0.4875,
    0.4875, 0.5, 0.5125, 0.4875, 0.4, 0.45, 0.425, 0.35,
    0.4375, 0.4375, 0.475, 0.4875, 0.45, 0.45, 0.5375, 0.4375, 0.5375, 0.5125,
    0.5375, 0.525, 0.525, 0.475, 0.45, 0.4875, 0.4875, 0.525, 0.45, 0.475,
    0.4875, 0.4, 0.4875, 0.4125, 0.4625, 0.4875, 0.375, 0.375,
]

has_answer = [
    0.675, 0.6625, 0.7, 0.75, 0.6875, 0.7875, 0.825, 0.775, 0.65, 0.6625,
    0.675, 0.7125, 0.7, 0.6875, 0.6625, 0.7125, 0.7, 0.675, 0.7, 0.6375,
    0.675, 0.625, 0.75, 0.775, 0.725, 0.7125, 0.625, 0.725, 0.725, 0.7625,
    0.6875, 0.725, 0.725, 0.7625, 0.8375, 0.725, 0.7625, 0.75, 0.7375, 0.725,
    0.7625, 0.725, 0.7375, 0.6875, 0.7375, 0.6625, 0.75, 0.725,
    0.725, 0.7125, 0.7375, 0.75, 0.7625, 0.775, 0.725, 0.725, 0.675, 0.65,
    0.7, 0.7, 0.7, 0.7875, 0.775, 0.7375, 0.7, 0.75, 0.7375, 0.7875,
    0.8125, 0.7875, 0.8, 0.7625, 0.8, 0.7375, 0.775, 0.775,
]

steps_reward = list(range(1, 76))
rewards = [
    0.35898, 0.36406, 0.35703, 0.27656, 0.32266, 0.35547, 0.40234, 0.27969,
    0.40039, 0.24180, 0.33477, 0.36992, 0.32344, 0.27383, 0.38633, 0.30156,
    0.35234, 0.34063, 0.26719, 0.38438, 0.32070, 0.27383, 0.28945, 0.48828,
    0.28086, 0.20898, 0.27578, 0.38711, 0.32891, 0.41602, 0.28789, 0.30117,
    0.34297, 0.35156, 0.40430, 0.31992, 0.36758, 0.36211, 0.38594, 0.31445,
    0.45547, 0.31445, 0.21094, 0.34063, 0.39219, 0.42148, 0.24453,
    # step 48-75: Judge down, reduced signal
    0.41484, 0.21523, 0.19805, 0.21250, 0.13438, 0.24023, 0.20586, 0.29531,
    0.24258, 0.26563, 0.19141, 0.16602, 0.22695, 0.22227, 0.23398, 0.19297,
    0.22109, 0.21055, 0.22773, 0.21328, 0.21953, 0.19922, 0.23320, 0.23828,
    0.23125, 0.25117, 0.21563, 0.20352,
]

CRASH_STEP = 48


def ema(values: list[float], alpha: float = 0.2) -> np.ndarray:
    """Exponential moving average. alpha=0.2 -> span ~9 steps."""
    out = np.empty(len(values))
    out[0] = values[0]
    for i in range(1, len(values)):
        out[i] = alpha * values[i] + (1 - alpha) * out[i - 1]
    return out


# ---------- Plot ----------

fig, axes = plt.subplots(2, 2, figsize=(13, 8))
fig.suptitle(
    "8b-sft-2gpu (run kudzrfba) — Judge crashed at step 48 (11:48 UTC)\n"
    "Shaded region = corrupted reward signal (y_T = 0 fallback)",
    fontsize=12,
    fontweight="bold",
)

ALPHA_EMA = 0.2
CRASH_COLOR = "#ffcccc"
CRASH_LINE_COLOR = "#cc0000"

def add_crash_annotation(ax, xmax):
    ax.axvspan(CRASH_STEP, xmax, color=CRASH_COLOR, alpha=0.4, label="Judge down")
    ax.axvline(CRASH_STEP, color=CRASH_LINE_COLOR, linewidth=1.5, linestyle="--")


# --- Panel 1: Training reward ---
ax = axes[0, 0]
ax.scatter(steps_reward, rewards, s=12, color="#888888", alpha=0.5, label="raw")
ax.plot(steps_reward, ema(rewards, ALPHA_EMA), color="#1f77b4", linewidth=2, label=f"EMA(α={ALPHA_EMA})")
add_crash_annotation(ax, max(steps_reward))
ax.set_title("Training reward (critic/rewards/mean)")
ax.set_xlabel("Global step")
ax.set_ylabel("Reward")
ax.legend(fontsize=8)
ax.set_xlim(0, 76)

# --- Panel 2: Val accuracy ---
ax = axes[0, 1]
# Pre-crash only for EMA (post-crash values are 0 by artifact, not learning)
acc_pre = acc[:CRASH_STEP]
steps_acc_pre = steps_val[:CRASH_STEP]
ax.scatter(steps_val, acc, s=12, color="#888888", alpha=0.5, label="raw")
ax.plot(steps_acc_pre, ema(acc_pre, ALPHA_EMA), color="#ff7f0e", linewidth=2, label=f"EMA pre-crash (α={ALPHA_EMA})")
add_crash_annotation(ax, max(steps_val))
ax.set_title("Val accuracy (val-core/acc/mean@4)\nNOTE: post-crash zeros = Judge fallback, NOT model regression")
ax.set_xlabel("Global step")
ax.set_ylabel("Accuracy")
ax.legend(fontsize=8)
ax.set_xlim(0, 76)
ax.set_ylim(-0.02, 0.45)

# --- Panel 3: Val discovery ---
ax = axes[1, 0]
ax.scatter(steps_val, discovery, s=12, color="#888888", alpha=0.5, label="raw")
ax.plot(steps_val, ema(discovery, ALPHA_EMA), color="#2ca02c", linewidth=2, label=f"EMA(α={ALPHA_EMA})")
add_crash_annotation(ax, max(steps_val))
ax.set_title("Val discovery (val-aux/discovery/mean@4)\nUnaffected by Judge crash (from cache)")
ax.set_xlabel("Global step")
ax.set_ylabel("Discovery rate")
ax.legend(fontsize=8)
ax.set_xlim(0, 76)

# --- Panel 4: Has-answer rate ---
ax = axes[1, 1]
ax.scatter(steps_val, has_answer, s=12, color="#888888", alpha=0.5, label="raw")
ax.plot(steps_val, ema(has_answer, ALPHA_EMA), color="#9467bd", linewidth=2, label=f"EMA(α={ALPHA_EMA})")
add_crash_annotation(ax, max(steps_val))
ax.set_title("Val has_answer (val-aux/has_answer/mean@4)\nStructuredOutput emission rate")
ax.set_xlabel("Global step")
ax.set_ylabel("Has-answer rate")
ax.legend(fontsize=8)
ax.set_xlim(0, 76)

plt.tight_layout()
out_path = "/data3/peijia/dr-claw/Explain/Experiment/core_code/tmp/precrash_metrics.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Saved: {out_path}")
