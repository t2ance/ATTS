"""Optimize retry score parameters for FOK/JOL metacognitive signals.

Score_retry = α(1 - JOL) + β·max(FOK - JOL, 0) + γ·max(JOL - FOK, 0) + δ(1 - FOK)

Goal: find (α, β, γ, δ, threshold) per model such that:
  - Score_retry > threshold → predict "will be wrong" (should retry)
  - Score_retry ≤ threshold → predict "will be correct" (no retry needed)

Optimize for best F1 / AUC / accuracy in separating correct vs incorrect.
"""

import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import differential_evolution
from sklearn.metrics import (
    roc_auc_score, f1_score, accuracy_score, precision_score,
    recall_score, confusion_matrix, classification_report, roc_curve
)
from pathlib import Path
from itertools import product

OUT_DIR = Path("figures")
OUT_DIR.mkdir(exist_ok=True)

sns.set_theme(style="whitegrid", font_scale=1.1)


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
            })
    return pd.DataFrame(rows)


df_sonnet = load_results("meta-cognitive/results_metacog_gold.jsonl", "Sonnet 4.6")
df_grok = load_results("langchain-build/results_langchain_gold.jsonl", "Grok 4.1 Fast")

print(f"Loaded: Sonnet={len(df_sonnet)}, Grok={len(df_grok)}")


# ---------------------------------------------------------------------------
# Retry score formula
# ---------------------------------------------------------------------------

def retry_score(fok, jol, alpha, beta, gamma, delta=0.0):
    """Compute retry score. Higher = more likely wrong, should retry."""
    return (alpha * (1 - jol)
            + beta * np.maximum(fok - jol, 0)
            + gamma * np.maximum(jol - fok, 0)
            + delta * (1 - fok))


# ---------------------------------------------------------------------------
# Optimization: find best (α, β, γ, threshold)
# ---------------------------------------------------------------------------

def optimize_for_model(df, model_name):
    """Find optimal parameters using differential evolution."""
    fok = df["fok"].values
    jol = df["jol"].values
    y_wrong = (~df["is_correct"]).astype(int).values  # 1 = wrong (should retry)

    # Objective: maximize F1 for detecting "wrong" answers
    def neg_f1(params):
        alpha, beta, gamma, delta, threshold = params
        scores = retry_score(fok, jol, alpha, beta, gamma, delta)
        preds = (scores > threshold).astype(int)
        # Handle edge cases
        if preds.sum() == 0 or preds.sum() == len(preds):
            return 1.0  # worst
        return -f1_score(y_wrong, preds)

    # Bounds: α, β, γ, δ in [0, 2], threshold in [0, 3]
    bounds = [(0, 2), (0, 2), (0, 2), (0, 2), (0, 3)]
    result = differential_evolution(neg_f1, bounds, seed=42, maxiter=500, tol=1e-8, polish=True)

    alpha, beta, gamma, delta, threshold = result.x
    best_f1 = -result.fun

    scores = retry_score(fok, jol, alpha, beta, gamma, delta)
    preds = (scores > threshold).astype(int)

    # Also compute AUC (threshold-free)
    try:
        auc = roc_auc_score(y_wrong, scores)
    except:
        auc = 0.5

    return {
        "model": model_name,
        "alpha": alpha,
        "beta": beta,
        "gamma": gamma,
        "delta": delta,
        "threshold": threshold,
        "f1": best_f1,
        "auc": auc,
        "accuracy": accuracy_score(y_wrong, preds),
        "precision": precision_score(y_wrong, preds, zero_division=0),
        "recall": recall_score(y_wrong, preds, zero_division=0),
        "scores": scores,
        "preds": preds,
        "y_true": y_wrong,
    }


# ---------------------------------------------------------------------------
# Grid search as sanity check + visualization
# ---------------------------------------------------------------------------

def grid_search(df, model_name, n_grid=20):
    """Coarse grid search to visualize the parameter landscape."""
    fok = df["fok"].values
    jol = df["jol"].values
    y_wrong = (~df["is_correct"]).astype(int).values

    alphas = np.linspace(0, 2, n_grid)
    best_results = []

    for alpha in alphas:
        for beta in np.linspace(0, 2, n_grid):
            for gamma in np.linspace(0, 2, n_grid):
                scores = retry_score(fok, jol, alpha, beta, gamma)
                try:
                    auc = roc_auc_score(y_wrong, scores)
                except:
                    auc = 0.5
                best_results.append({
                    "alpha": alpha, "beta": beta, "gamma": gamma, "auc": auc
                })

    return pd.DataFrame(best_results)


# ---------------------------------------------------------------------------
# Run optimization
# ---------------------------------------------------------------------------

results = {}
for df_model, name in [(df_sonnet, "Sonnet 4.6"), (df_grok, "Grok 4.1 Fast")]:
    print(f"\n{'=' * 60}")
    print(f"Optimizing: {name} (n={len(df_model)})")
    print(f"{'=' * 60}")

    res = optimize_for_model(df_model, name)
    results[name] = res

    print(f"\n  Optimal parameters:")
    print(f"    α (1-JOL weight):          {res['alpha']:.4f}")
    print(f"    β (FOK>JOL penalty):       {res['beta']:.4f}")
    print(f"    γ (JOL>FOK overconfidence): {res['gamma']:.4f}")
    print(f"    δ (1-FOK weight):          {res['delta']:.4f}")
    print(f"    threshold:                  {res['threshold']:.4f}")
    print(f"\n  Performance:")
    print(f"    F1 (detect wrong):  {res['f1']:.4f}")
    print(f"    AUC-ROC:            {res['auc']:.4f}")
    print(f"    Accuracy:           {res['accuracy']:.4f}")
    print(f"    Precision:          {res['precision']:.4f}")
    print(f"    Recall:             {res['recall']:.4f}")

    cm = confusion_matrix(res["y_true"], res["preds"])
    print(f"\n  Confusion matrix (rows=actual, cols=predicted):")
    print(f"    Predict:   No-retry  Retry")
    print(f"    Correct:   {cm[0][0]:>6d}  {cm[0][1]:>6d}")
    print(f"    Wrong:     {cm[1][0]:>6d}  {cm[1][1]:>6d}")

    print(f"\n  Formula:")
    print(f"    Score = {res['alpha']:.3f}·(1-JOL) + {res['beta']:.3f}·max(FOK-JOL,0) + {res['gamma']:.3f}·max(JOL-FOK,0) + {res['delta']:.3f}·(1-FOK)")
    print(f"    Retry if Score > {res['threshold']:.3f}")


# ---------------------------------------------------------------------------
# Fig 6: Retry score distributions (correct vs wrong)
# ---------------------------------------------------------------------------

fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
for ax, name in zip(axes, ["Sonnet 4.6", "Grok 4.1 Fast"]):
    res = results[name]
    scores_correct = res["scores"][res["y_true"] == 0]
    scores_wrong = res["scores"][res["y_true"] == 1]

    ax.hist(scores_correct, bins=25, alpha=0.6, color="#22c55e", label=f"Correct (n={len(scores_correct)})", density=True)
    ax.hist(scores_wrong, bins=25, alpha=0.6, color="#ef4444", label=f"Wrong (n={len(scores_wrong)})", density=True)
    ax.axvline(res["threshold"], color="black", linestyle="--", linewidth=2, label=f"Threshold={res['threshold']:.3f}")
    ax.set_xlabel("Retry Score")
    ax.set_ylabel("Density")
    ax.set_title(f"{name}\nα={res['alpha']:.2f}, β={res['beta']:.2f}, γ={res['gamma']:.2f}, δ={res['delta']:.2f} | F1={res['f1']:.3f}")
    ax.legend(fontsize=9)

fig.suptitle("Retry Score Distribution: Should We Retry?", fontsize=14, fontweight="bold")
fig.tight_layout()
fig.savefig(OUT_DIR / "6_retry_score_distribution.png", dpi=150)
print("\nSaved: 6_retry_score_distribution.png")


# ---------------------------------------------------------------------------
# Fig 7: ROC curves for retry score
# ---------------------------------------------------------------------------

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
for ax, name in zip(axes, ["Sonnet 4.6", "Grok 4.1 Fast"]):
    res = results[name]
    fpr, tpr, thresholds = roc_curve(res["y_true"], res["scores"])
    ax.plot(fpr, tpr, linewidth=2, color="#6366f1", label=f"Retry Score (AUC={res['auc']:.3f})")

    # Also compare with raw JOL, FOK
    df_model = df_sonnet if name == "Sonnet 4.6" else df_grok
    y = (~df_model["is_correct"]).astype(int).values

    # 1-JOL as baseline (simplest predictor of "wrong")
    fpr_jol, tpr_jol, _ = roc_curve(y, 1 - df_model["jol"].values)
    auc_jol = roc_auc_score(y, 1 - df_model["jol"].values)
    ax.plot(fpr_jol, tpr_jol, linewidth=1.5, color="#f97316", linestyle="--", label=f"1−JOL only (AUC={auc_jol:.3f})")

    # FOK as baseline
    fpr_fok, tpr_fok, _ = roc_curve(y, 1 - df_model["fok"].values)
    auc_fok = roc_auc_score(y, 1 - df_model["fok"].values)
    ax.plot(fpr_fok, tpr_fok, linewidth=1.5, color="#3b82f6", linestyle=":", label=f"1−FOK only (AUC={auc_fok:.3f})")

    ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(name)
    ax.legend(fontsize=9)

fig.suptitle("ROC: Retry Score vs Simple Baselines", fontsize=14, fontweight="bold")
fig.tight_layout()
fig.savefig(OUT_DIR / "7_retry_roc.png", dpi=150)
print("Saved: 7_retry_roc.png")


# ---------------------------------------------------------------------------
# Fig 8: Decision boundary in FOK-JOL space
# ---------------------------------------------------------------------------

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
for ax, name in zip(axes, ["Sonnet 4.6", "Grok 4.1 Fast"]):
    res = results[name]
    df_model = df_sonnet if name == "Sonnet 4.6" else df_grok

    # Heatmap of retry score in FOK-JOL space
    fok_grid = np.linspace(0, 1, 200)
    jol_grid = np.linspace(0, 1, 200)
    FOK_G, JOL_G = np.meshgrid(fok_grid, jol_grid)
    SCORE_G = retry_score(FOK_G, JOL_G, res["alpha"], res["beta"], res["gamma"], res["delta"])

    im = ax.contourf(FOK_G, JOL_G, SCORE_G, levels=30, cmap="RdYlGn_r", alpha=0.7)
    ax.contour(FOK_G, JOL_G, SCORE_G, levels=[res["threshold"]], colors="black", linewidths=2, linestyles="--")

    # Scatter actual data points
    for correct, color, marker in [(True, "#22c55e", "o"), (False, "#ef4444", "x")]:
        mask = df_model["is_correct"] == correct
        ax.scatter(df_model.loc[mask, "fok"], df_model.loc[mask, "jol"],
                   c=color, marker=marker, s=30, alpha=0.7, edgecolors="white" if correct else "none",
                   linewidth=0.3, label="Correct" if correct else "Incorrect")

    plt.colorbar(im, ax=ax, label="Retry Score")
    ax.set_xlabel("FOK")
    ax.set_ylabel("JOL")
    ax.set_title(f"{name}\nthreshold={res['threshold']:.3f} (black dashed)")
    ax.legend(fontsize=9, loc="lower right")

fig.suptitle("Decision Boundary in FOK-JOL Space", fontsize=14, fontweight="bold")
fig.tight_layout()
fig.savefig(OUT_DIR / "8_decision_boundary.png", dpi=150)
print("Saved: 8_decision_boundary.png")


# ---------------------------------------------------------------------------
# Fig 9: Parameter sensitivity — α vs F1 (fixing β, γ at optimal)
# ---------------------------------------------------------------------------

fig, axes = plt.subplots(1, 4, figsize=(22, 5))
param_names = ["α (1−JOL)", "β (FOK>JOL)", "γ (JOL>FOK)", "δ (1−FOK)"]
param_keys = ["alpha", "beta", "gamma", "delta"]

for ax, pname, pkey in zip(axes, param_names, param_keys):
    for name, color in [("Sonnet 4.6", "#6366f1"), ("Grok 4.1 Fast", "#f59e0b")]:
        res = results[name]
        df_model = df_sonnet if name == "Sonnet 4.6" else df_grok
        fok = df_model["fok"].values
        jol = df_model["jol"].values
        y_wrong = (~df_model["is_correct"]).astype(int).values

        param_range = np.linspace(0, 2, 50)
        f1s = []
        for pval in param_range:
            params = {"alpha": res["alpha"], "beta": res["beta"], "gamma": res["gamma"], "delta": res["delta"]}
            params[pkey] = pval
            scores = retry_score(fok, jol, **params)
            preds = (scores > res["threshold"]).astype(int)
            if preds.sum() == 0 or preds.sum() == len(preds):
                f1s.append(0)
            else:
                f1s.append(f1_score(y_wrong, preds))
        ax.plot(param_range, f1s, color=color, linewidth=2, label=name)
        ax.axvline(res[pkey], color=color, linestyle="--", alpha=0.5)

    ax.set_xlabel(pname)
    ax.set_ylabel("F1 Score")
    ax.set_title(f"Sensitivity to {pname}")
    ax.legend(fontsize=9)

fig.suptitle("Parameter Sensitivity Analysis", fontsize=14, fontweight="bold")
fig.tight_layout()
fig.savefig(OUT_DIR / "9_parameter_sensitivity.png", dpi=150)
print("Saved: 9_parameter_sensitivity.png")


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

print("\n" + "=" * 70)
print("FINAL OPTIMAL PARAMETERS")
print("=" * 70)
print(f"{'':>25s}  {'Sonnet 4.6':>14s}  {'Grok 4.1 Fast':>14s}")
print(f"{'─' * 60}")
for key in ["alpha", "beta", "gamma", "delta", "threshold"]:
    print(f"  {key:>22s}  {results['Sonnet 4.6'][key]:>14.4f}  {results['Grok 4.1 Fast'][key]:>14.4f}")
print(f"{'─' * 60}")
for key in ["f1", "auc", "accuracy", "precision", "recall"]:
    print(f"  {key:>22s}  {results['Sonnet 4.6'][key]:>14.4f}  {results['Grok 4.1 Fast'][key]:>14.4f}")
print(f"{'─' * 60}")

for name in ["Sonnet 4.6", "Grok 4.1 Fast"]:
    r = results[name]
    print(f"\n  {name}:")
    print(f"    Score = {r['alpha']:.3f}·(1-JOL) + {r['beta']:.3f}·max(FOK-JOL,0) + {r['gamma']:.3f}·max(JOL-FOK,0) + {r['delta']:.3f}·(1-FOK)")
    print(f"    → Retry if Score > {r['threshold']:.3f}")
