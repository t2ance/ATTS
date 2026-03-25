"""FOK / JOL metacognitive analysis — visualization and statistics."""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import seaborn as sns
from scipy import stats
from sklearn.metrics import roc_auc_score, brier_score_loss
from pathlib import Path

OUT_DIR = Path("figures")
OUT_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

def load_results(path, agent_name):
    rows = []
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            fok = r.get("fok_score", None)
            jol = r.get("jol_score", None)
            if fok is None or jol is None or fok < 0 or jol < 0:
                continue
            rows.append({
                "agent": agent_name,
                "id": r["id"],
                "category": r.get("category", "Unknown"),
                "is_correct": bool(r["is_correct"]),
                "fok": float(fok),
                "jol": float(jol),
                "delta": float(jol) - float(fok),  # JOL - FOK (calibration shift)
            })
    return pd.DataFrame(rows)

df_sonnet = load_results("meta-cognitive/results_metacog_gold.jsonl", "Sonnet 4.6")
df_grok = load_results("langchain-build/results_langchain_gold.jsonl", "Grok 4.1 Fast")
df = pd.concat([df_sonnet, df_grok], ignore_index=True)

print(f"Loaded: Sonnet={len(df_sonnet)}, Grok={len(df_grok)}, Total={len(df)}")

# Plotting style
sns.set_theme(style="whitegrid", font_scale=1.1)
COLORS = {"Sonnet 4.6": "#6366f1", "Grok 4.1 Fast": "#f59e0b"}
CORRECT_COLORS = {True: "#22c55e", False: "#ef4444"}


# ---------------------------------------------------------------------------
# Fig 1: FOK vs JOL scatter, colored by correctness
# ---------------------------------------------------------------------------

fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
for ax, agent in zip(axes, ["Sonnet 4.6", "Grok 4.1 Fast"]):
    sub = df[df["agent"] == agent]
    for correct, color, label in [(True, "#22c55e", "Correct"), (False, "#ef4444", "Incorrect")]:
        mask = sub["is_correct"] == correct
        ax.scatter(sub.loc[mask, "fok"], sub.loc[mask, "jol"],
                   c=color, alpha=0.5, s=40, label=label, edgecolors="white", linewidth=0.3)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3, label="FOK = JOL")
    ax.set_xlabel("FOK (pre-answer)")
    ax.set_ylabel("JOL (post-answer)")
    acc = sub["is_correct"].mean() * 100
    ax.set_title(f"{agent}  (n={len(sub)}, acc={acc:.1f}%)")
    ax.legend(loc="upper left", fontsize=9)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)

fig.suptitle("FOK vs JOL — Colored by Correctness", fontsize=14, fontweight="bold")
fig.tight_layout()
fig.savefig(OUT_DIR / "1_fok_vs_jol_scatter.png", dpi=150)
print("Saved: 1_fok_vs_jol_scatter.png")


# ---------------------------------------------------------------------------
# Fig 2: Calibration curves — binned confidence vs actual accuracy
# ---------------------------------------------------------------------------

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
for ax, (score_name, label) in zip(axes, [("fok", "FOK"), ("jol", "JOL")]):
    for agent, color in COLORS.items():
        sub = df[df["agent"] == agent]
        bins = np.linspace(0, 1, 11)
        bin_centers = []
        bin_accs = []
        bin_counts = []
        for i in range(len(bins) - 1):
            mask = (sub[score_name] >= bins[i]) & (sub[score_name] < bins[i+1])
            if mask.sum() >= 3:
                bin_centers.append((bins[i] + bins[i+1]) / 2)
                bin_accs.append(sub.loc[mask, "is_correct"].mean())
                bin_counts.append(mask.sum())
        ax.plot(bin_centers, bin_accs, "o-", color=color, label=f"{agent} (n={len(sub)})", linewidth=2)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3, label="Perfect calibration")
    ax.set_xlabel(f"{label} Score")
    ax.set_ylabel("Actual Accuracy")
    ax.set_title(f"{label} Calibration Curve")
    ax.legend(fontsize=9)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)

fig.suptitle("Calibration: Confidence vs Actual Accuracy", fontsize=14, fontweight="bold")
fig.tight_layout()
fig.savefig(OUT_DIR / "2_calibration_curves.png", dpi=150)
print("Saved: 2_calibration_curves.png")


# ---------------------------------------------------------------------------
# Fig 3: Delta (JOL - FOK) distribution by correctness
# ---------------------------------------------------------------------------

fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
for ax, agent in zip(axes, ["Sonnet 4.6", "Grok 4.1 Fast"]):
    sub = df[df["agent"] == agent]
    for correct, color, label in [(True, "#22c55e", "Correct"), (False, "#ef4444", "Incorrect")]:
        vals = sub.loc[sub["is_correct"] == correct, "delta"]
        ax.hist(vals, bins=20, alpha=0.6, color=color, label=f"{label} (mean={vals.mean():.3f})", density=True)
    ax.axvline(0, color="black", linestyle="--", alpha=0.3)
    ax.set_xlabel("JOL − FOK (confidence shift)")
    ax.set_ylabel("Density")
    ax.set_title(f"{agent}")
    ax.legend(fontsize=9)

fig.suptitle("Metacognitive Shift: JOL − FOK by Correctness", fontsize=14, fontweight="bold")
fig.tight_layout()
fig.savefig(OUT_DIR / "3_delta_distribution.png", dpi=150)
print("Saved: 3_delta_distribution.png")


# ---------------------------------------------------------------------------
# Fig 4: ROC-like — FOK, JOL, and combined as predictors of correctness
# ---------------------------------------------------------------------------

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
for ax, agent in zip(axes, ["Sonnet 4.6", "Grok 4.1 Fast"]):
    sub = df[df["agent"] == agent]
    y = sub["is_correct"].astype(int).values

    for score_name, color, ls in [("fok", "#3b82f6", "-"), ("jol", "#f97316", "-"), ("delta", "#8b5cf6", "--")]:
        scores = sub[score_name].values
        if len(np.unique(y)) < 2:
            continue
        try:
            auc = roc_auc_score(y, scores)
            # Manual ROC
            thresholds = np.linspace(scores.min(), scores.max(), 200)
            tprs, fprs = [], []
            for t in thresholds:
                pred = scores >= t
                tp = ((pred == 1) & (y == 1)).sum()
                fp = ((pred == 1) & (y == 0)).sum()
                fn = ((pred == 0) & (y == 1)).sum()
                tn = ((pred == 0) & (y == 0)).sum()
                tprs.append(tp / (tp + fn) if (tp + fn) > 0 else 0)
                fprs.append(fp / (fp + tn) if (fp + tn) > 0 else 0)
            ax.plot(fprs, tprs, color=color, linestyle=ls, linewidth=2,
                    label=f"{score_name.upper()} (AUC={auc:.3f})")
        except Exception:
            pass

    ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"{agent}")
    ax.legend(fontsize=9)

fig.suptitle("ROC: FOK, JOL, and Δ(JOL−FOK) as Correctness Predictors", fontsize=14, fontweight="bold")
fig.tight_layout()
fig.savefig(OUT_DIR / "4_roc_curves.png", dpi=150)
print("Saved: 4_roc_curves.png")


# ---------------------------------------------------------------------------
# Fig 5: Category-level FOK/JOL heatmap
# ---------------------------------------------------------------------------

fig, axes = plt.subplots(1, 2, figsize=(16, 8))
for ax, agent in zip(axes, ["Sonnet 4.6", "Grok 4.1 Fast"]):
    sub = df[df["agent"] == agent]
    cats = sub.groupby("category").agg(
        n=("is_correct", "count"),
        accuracy=("is_correct", "mean"),
        avg_fok=("fok", "mean"),
        avg_jol=("jol", "mean"),
        avg_delta=("delta", "mean"),
    ).query("n >= 5").sort_values("accuracy", ascending=True)

    if len(cats) == 0:
        continue

    heat_data = cats[["avg_fok", "avg_jol", "accuracy"]].rename(
        columns={"avg_fok": "FOK", "avg_jol": "JOL", "accuracy": "Accuracy"}
    )
    sns.heatmap(heat_data, annot=True, fmt=".2f", cmap="RdYlGn", ax=ax,
                vmin=0, vmax=1, linewidths=0.5)
    ax.set_title(f"{agent} (categories with n≥5)")
    ax.set_ylabel("")

fig.suptitle("Category-Level: FOK, JOL, Accuracy", fontsize=14, fontweight="bold")
fig.tight_layout()
fig.savefig(OUT_DIR / "5_category_heatmap.png", dpi=150)
print("Saved: 5_category_heatmap.png")


# ---------------------------------------------------------------------------
# Statistical tests
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
print("STATISTICAL ANALYSIS")
print("=" * 60)

for agent in ["Sonnet 4.6", "Grok 4.1 Fast"]:
    sub = df[df["agent"] == agent]
    correct = sub[sub["is_correct"]]
    incorrect = sub[~sub["is_correct"]]
    y = sub["is_correct"].astype(int).values

    print(f"\n{'─' * 50}")
    print(f"  {agent}  (n={len(sub)})")
    print(f"{'─' * 50}")

    # 1. Point-biserial correlation
    for score in ["fok", "jol", "delta"]:
        r, p = stats.pointbiserialr(y, sub[score].values)
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        print(f"  Point-biserial r({score.upper()}, correct): r={r:.4f}, p={p:.4f} {sig}")

    # 2. Mann-Whitney U (correct vs incorrect)
    for score in ["fok", "jol"]:
        if len(correct) > 0 and len(incorrect) > 0:
            u, p = stats.mannwhitneyu(correct[score], incorrect[score], alternative="two-sided")
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
            print(f"  Mann-Whitney U ({score.upper()} correct vs incorrect): U={u:.0f}, p={p:.4f} {sig}")

    # 3. Brier score (lower = better calibrated)
    for score in ["fok", "jol"]:
        brier = brier_score_loss(y, sub[score].values)
        print(f"  Brier score ({score.upper()}): {brier:.4f}")

    # 4. AUC-ROC
    for score in ["fok", "jol", "delta"]:
        try:
            auc = roc_auc_score(y, sub[score].values)
            print(f"  AUC-ROC ({score.upper()}): {auc:.4f}")
        except Exception:
            pass

    # 5. Paired Wilcoxon: FOK vs JOL
    w, p = stats.wilcoxon(sub["fok"], sub["jol"])
    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
    print(f"  Wilcoxon signed-rank (FOK vs JOL): W={w:.0f}, p={p:.6f} {sig}")

    # 6. Gamma correlation (ordinal association)
    # Kendall's tau as proxy
    for score in ["fok", "jol"]:
        tau, p = stats.kendalltau(sub[score], y)
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        print(f"  Kendall τ ({score.upper()}, correct): τ={tau:.4f}, p={p:.4f} {sig}")

print(f"\nAll figures saved to: {OUT_DIR.resolve()}")
