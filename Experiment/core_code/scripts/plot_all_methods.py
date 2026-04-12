"""Generate cost-vs-accuracy comparison plots with all TTS methods for each benchmark."""

import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[3]  # dr-claw/Explain/
RUN_DIR = PROJECT_ROOT / "Experiment" / "analysis" / "run"
FIGURES_DIR = PROJECT_ROOT / "Publication" / "paper" / "figures"


def parse_delegated_log(log_path: Path) -> dict | None:
    """Parse a delegated ATTS log for best-of-N curve data."""
    text = log_path.read_text()
    if "EVALUATION COMPLETE" not in text:
        return None

    total_m = re.search(r"^Total:\s+(\d+)", text, re.MULTILINE)
    total = int(total_m.group(1))

    # best-of-N lines (with or without majority column)
    ns = []
    oracle_pcts = []
    costs = []
    for m in re.finditer(r"best-of-(\d+)\s+([\d.]+)%\s+(?:[\d.]+%\s+)?\$([\d.]+)", text):
        n = int(m.group(1))
        oracle = float(m.group(2))
        cost_total = float(m.group(3))
        ns.append(n)
        oracle_pcts.append(oracle)
        costs.append(cost_total / total)

    # +agg line
    agg_m = re.search(r"best-of-\+agg\s+([\d.]+)%\s+(?:[\d.]+%\s+)?\$([\d.]+)", text)
    agg_oracle = float(agg_m.group(1))
    agg_cost = float(agg_m.group(2)) / total

    # integrated accuracy
    int_m = re.search(r"Integrated:\s+(\d+)/(\d+)", text)
    integrated_pct = int(int_m.group(1)) / int(int_m.group(2)) * 100

    return {
        "ns": ns, "oracle_pcts": oracle_pcts, "costs": costs,
        "agg_oracle": agg_oracle, "agg_cost": agg_cost,
        "integrated_pct": integrated_pct, "total": total,
    }


def parse_method_log(log_path: Path) -> dict | None:
    """Parse a method log for accuracy and cost."""
    text = log_path.read_text()
    if "EVALUATION COMPLETE" not in text:
        return None
    int_m = re.search(r"Integrated:\s+(\d+)/(\d+)", text)
    acc = int(int_m.group(1)) / int(int_m.group(2)) * 100
    cost_m = re.search(r"Total\s+\$([\d.]+)\s+\(avg \$([\d.]+)/question\)", text)
    avg_cost = float(cost_m.group(2))
    return {"accuracy": acc, "cost": avg_cost}


def get_rerank_cost(delegated_log: Path, total: int) -> float:
    """Get best-of-8 explore cost per question from delegated log (for rerank methods)."""
    text = delegated_log.read_text()
    m = re.search(r"best-of-8\s+[\d.]+%\s+(?:[\d.]+%\s+)?\$([\d.]+)", text)
    return float(m.group(1)) / total


def read_mm_effort(run_dir: Path) -> dict | None:
    """Read multi-model effort results from progress.json."""
    run_dirs = sorted(Path(run_dir).glob("run_*/progress.json"))
    if not run_dirs:
        return None
    data = json.loads(run_dirs[-1].read_text())
    s = data.get("summary", data)
    total = s["total"]
    return {"accuracy": s["correct"] / total * 100, "cost": s["total_cost_usd"] / total}


BENCHMARKS = {
    "gpqa": {
        "title": "GPQA-Diamond",
        "delegated": RUN_DIR / "gpqa/sonnet/delegated.log",
        "no_integrate": RUN_DIR / "gpqa/sonnet_no_integrate/delegated.log",
        "haiku_orch": RUN_DIR / "gpqa/haiku_orch/delegated.log",
        "opus_orch": RUN_DIR / "gpqa/opus_orch_new/delegated.log",
        "effort_dirs": {
            "Low": RUN_DIR / "gpqa/multi_model_effort_low",
            "Med": RUN_DIR / "gpqa/multi_model_effort_medium",
            "High": RUN_DIR / "gpqa/multi_model_effort_high",
        },
        "methods": {
            "Self-Refine": RUN_DIR / "gpqa/sonnet_self_refine/self_refine.log",
            "Budget Forcing": RUN_DIR / "gpqa/sonnet_budget_forcing/budget_forcing.log",
            "Skywork-Reward-V2": RUN_DIR / "gpqa/sonnet_skywork_rerank/rerank.log",
        },
        "rerank_methods": ["Skywork-Reward-V2"],
        "llm_selection": RUN_DIR / "gpqa/sonnet_standalone_integrator",
    },
    "lcb": {
        "title": "LiveCodeBench",
        "delegated": RUN_DIR / "lcb/sonnet/delegated.log",
        "no_integrate": RUN_DIR / "lcb/sonnet_no_integrate/delegated.log",
        "haiku_orch": RUN_DIR / "lcb/haiku_orch/delegated.log",
        "opus_orch": RUN_DIR / "lcb/opus_orch/delegated.log",
        "effort_dirs": {
            "Low": RUN_DIR / "lcb/multi_model_effort_low",
            "Med": RUN_DIR / "lcb/multi_model_effort_medium",
            "High": RUN_DIR / "lcb/multi_model_effort_high",
        },
        "methods": {
            "Self-Refine": RUN_DIR / "lcb/sonnet_self_refine/self_refine.log",
            "Budget Forcing": RUN_DIR / "lcb/sonnet_budget_forcing/budget_forcing.log",
            "Skywork-Reward-V2": RUN_DIR / "lcb/sonnet_skywork_rerank/rerank2.log",
        },
        "rerank_methods": ["Skywork-Reward-V2"],
        "llm_selection": RUN_DIR / "lcb/sonnet_standalone_integrator",
    },
    "babyvision": {
        "title": "BabyVision",
        "delegated": RUN_DIR / "babyvision/sonnet/delegated.log",
        "no_integrate": RUN_DIR / "babyvision/sonnet_no_integrate/delegated.log",
        "effort_dirs": {
            "Low": RUN_DIR / "babyvision/multi_model_effort_low",
            "Med": RUN_DIR / "babyvision/multi_model_effort_medium",
            "High": RUN_DIR / "babyvision/multi_model_effort_high",
        },
        "methods": {
            "Self-Refine": RUN_DIR / "babyvision/sonnet_self_refine/self_refine.log",
            "Budget Forcing": RUN_DIR / "babyvision/sonnet_budget_forcing/budget_forcing.log",
            "VisualPRM": RUN_DIR / "babyvision/sonnet_visualprm_rerank/rerank.log",
        },
        "rerank_methods": ["VisualPRM"],
        "llm_selection": RUN_DIR / "babyvision/sonnet_standalone_integrator",
    },
    "aime2025": {
        "title": "AIME 2025",
        "delegated": RUN_DIR / "aime2025/sonnet/delegated.log",
        "no_integrate": RUN_DIR / "aime2025/sonnet_no_integrate/delegated.log",
        "methods": {
            "Self-Refine": RUN_DIR / "aime2025/sonnet_self_refine/self_refine.log",
            "Budget Forcing": RUN_DIR / "aime2025/sonnet_budget_forcing/budget_forcing.log",
            "Skywork-Reward-V2": RUN_DIR / "aime2025/sonnet_skywork_rerank/rerank.log",
        },
        "rerank_methods": ["Skywork-Reward-V2"],
    },
    "aime2026": {
        "title": "AIME 2026",
        "delegated": RUN_DIR / "aime2026/sonnet/delegated.log",
        "no_integrate": RUN_DIR / "aime2026/sonnet_no_integrate/delegated.log",
        "methods": {
            "Self-Refine": RUN_DIR / "aime2026/sonnet_self_refine/self_refine.log",
            "Budget Forcing": RUN_DIR / "aime2026/sonnet_budget_forcing/budget_forcing.log",
            "Skywork-Reward-V2": RUN_DIR / "aime2026/sonnet_skywork_rerank/rerank.log",
        },
        "rerank_methods": ["Skywork-Reward-V2"],
    },
    "hle": {
        "title": "HLE-Verified",
        "delegated": RUN_DIR / "hle/sonnet/gold_delegated.log",
        "no_integrate": RUN_DIR / "hle/sonnet_no_integrate/delegated.log",
        "haiku_orch": RUN_DIR / "hle/haiku_orch/delegated.log",
        "opus_orch": RUN_DIR / "hle/opus_orch/delegated.log",
        "effort_dirs": {
            "Low": RUN_DIR / "hle/multi_model_effort_low",
            "Med": RUN_DIR / "hle/multi_model_effort_medium",
            "High": RUN_DIR / "hle/multi_model_effort_high",
        },
        "methods": {
            "Self-Refine": RUN_DIR / "hle/sonnet_self_refine/self_refine.log",
            "Budget Forcing": RUN_DIR / "hle/sonnet_budget_forcing/budget_forcing.log",
            "Skywork-Reward-V2": RUN_DIR / "hle/sonnet_skywork_rerank/rerank_resume2.log",
        },
        "rerank_methods": ["Skywork-Reward-V2"],
        "llm_selection": RUN_DIR / "hle/sonnet_standalone_integrator",
    },
}

# Baselines: all circles, different colors
BASELINE_STYLES = {
    "Self-Refine": {"color": "#4CAF50"},
    "Budget Forcing": {"color": "#9C27B0"},
    "Skywork-Reward-V2": {"color": "#FF5722"},
    "VisualPRM": {"color": "#FF5722"},
    "LLM Selection": {"color": "#FF9800"},
}

OURS_STYLES = {
    "ATTS": {"color": "#E91E63"},
    "ATTS Multi Med": {"color": "#00897B"},
}

# HLE cost correction: 10 compaction-failed explores in hle/sonnet/gold cache produced
# garbage answers at high cost. Their cost_usd is zeroed. Each method consumed a different
# subset depending on stopping behavior. Values verified against results.jsonl round-level
# matching. See main.tex "% Corrected:" comments for full derivation per table cell.
_HLE_COST_CORRECTIONS = {
    # method_key: corrected $/q
    "atts": 1.5879,            # v3 no_integrate: ($190.47 - $31.68) / 100
    "rerank": 3.6133,          # Skywork: ($440.73 - $79.40) / 100, all-8 method
    "llm_selection": 3.7055,   # standalone_integrator: ($449.95 - $79.40) / 100, all-8
    "multi_med": 1.5417,       # effort_medium: ($162.80 - $8.63) / 100
    "multi_low": 1.4355,       # effort_low: ($152.18 - $8.63) / 100
    "haiku_orch": 1.8286,      # ($223.15 - $40.28) / 100, 5 bad consumed
    "opus_orch": 1.4995,       # ($178.71 - $28.75) / 100, 3 bad consumed
    # effort_high: not corrected (0 bad explores consumed)
}


def plot_benchmark(bench_key: str, bench_cfg: dict) -> None:
    """Main figure: baselines + ATTS + Multi Med + BoN curve."""
    delegated = parse_delegated_log(bench_cfg["delegated"])
    if delegated is None:
        print(f"  Skipping {bench_key}: delegated log not complete")
        return

    fig, ax = plt.subplots(figsize=(8, 5))

    all_accs = []
    all_costs = []

    # Parse no_integrate log for ATTS results
    bon_source = bench_cfg.get("no_integrate")
    if bon_source and bon_source.exists():
        ni = parse_delegated_log(bon_source)
    else:
        ni = None
    bon_data = ni if ni is not None else delegated

    # Pass@1 point
    pass1_cost = bon_data["costs"][0]
    pass1_acc = bon_data["oracle_pcts"][0]
    ax.scatter([pass1_cost], [pass1_acc], color="#607D8B", marker="o", s=80, zorder=5,
               label=f"Pass@1 ({pass1_acc:.1f}%)")

    # Baseline methods
    rerank_cost = get_rerank_cost(bench_cfg["delegated"], delegated["total"])
    if bench_key == "hle":
        rerank_cost = _HLE_COST_CORRECTIONS["rerank"]
    for method_name, log_path in bench_cfg["methods"].items():
        if not log_path.exists():
            continue
        data = parse_method_log(log_path)
        if data is None:
            continue
        mcolor = BASELINE_STYLES.get(method_name, {}).get("color", "#607D8B")
        cost = rerank_cost if method_name in bench_cfg.get("rerank_methods", []) else data["cost"]
        ax.scatter([cost], [data["accuracy"]], color=mcolor,
                   marker="o", s=80, zorder=5,
                   label=f"{method_name} ({data['accuracy']:.1f}%)")
        all_accs.append(data["accuracy"])
        all_costs.append(cost)

    # LLM Selection (standalone integrator)
    llm_sel_dir = bench_cfg.get("llm_selection")
    if llm_sel_dir and llm_sel_dir.exists():
        llm_sel = read_mm_effort(llm_sel_dir)
        if llm_sel is not None:
            mcolor = BASELINE_STYLES["LLM Selection"]["color"]
            llm_cost = _HLE_COST_CORRECTIONS["llm_selection"] if bench_key == "hle" else llm_sel["cost"]
            ax.scatter([llm_cost], [llm_sel["accuracy"]], color=mcolor,
                       marker="o", s=80, zorder=5,
                       label=f"LLM Selection ({llm_sel['accuracy']:.1f}%)")
            all_accs.append(llm_sel["accuracy"])
            all_costs.append(llm_cost)

    # ATTS (explore-only, from no_integrate log)
    ours_points = []
    if ni is not None:
        atts_cost = _HLE_COST_CORRECTIONS["atts"] if bench_key == "hle" else ni["agg_cost"]
        ours_points.append(("ATTS", atts_cost, ni["integrated_pct"]))

    # ATTS Multi Med only
    effort_dirs = bench_cfg.get("effort_dirs", {})
    if "Med" in effort_dirs:
        mm = read_mm_effort(effort_dirs["Med"])
        if mm is not None:
            mm_cost = _HLE_COST_CORRECTIONS["multi_med"] if bench_key == "hle" else mm["cost"]
            ours_points.append(("ATTS Multi Med", mm_cost, mm["accuracy"]))

    for label, cost, acc in ours_points:
        mcolor = OURS_STYLES.get(label, {}).get("color", "#E91E63")
        ax.scatter([cost], [acc], color=mcolor, marker="*", s=200, zorder=6,
                   label=f"{label} ({acc:.1f}%)")
        all_accs.append(acc)
        all_costs.append(cost)

    ax.set_xlabel("Avg Cost per Question (USD)", fontsize=16)
    ax.set_ylabel("Accuracy (%)", fontsize=16)
    ax.tick_params(axis="both", labelsize=12)
    ax.grid(True, alpha=0.3)

    min_acc = min(all_accs)
    max_acc = max(all_accs)
    margin = max((max_acc - min_acc) * 0.15, 2.0)
    ax.set_ylim(max(0, min_acc - margin), min(100, max_acc + margin))
    max_cost = max(all_costs)
    ax.set_xlim(0, max_cost * 1.15)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"${v:.2f}"))

    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02),
              ncol=3, fontsize=10, frameon=False)

    out_name = f"{bench_key}_cost_vs_accuracy"
    fig.savefig(FIGURES_DIR / f"{out_name}.pdf", bbox_inches="tight")
    fig.savefig(FIGURES_DIR / f"{out_name}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_name}.pdf")


def plot_effort_ablation() -> None:
    """1x4 row: exploration effort (Low/Med/High) per benchmark, shared legend outside."""
    bench_keys = ["hle", "lcb", "gpqa", "babyvision"]
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.5))
    effort_levels = ["Low", "Med", "High"]
    colors = {"Low": "#80CBC4", "Med": "#00897B", "High": "#004D40"}
    markers = {"Low": "s", "Med": "*", "High": "D"}

    for idx, bench_key in enumerate(bench_keys):
        ax = axes[idx]
        bench_cfg = BENCHMARKS[bench_key]
        effort_dirs = bench_cfg.get("effort_dirs", {})

        pts = {}
        for level in effort_levels:
            if level not in effort_dirs:
                continue
            mm = read_mm_effort(effort_dirs[level])
            if mm is not None:
                hle_key = f"multi_{level.lower()}"
                if bench_key == "hle" and hle_key in _HLE_COST_CORRECTIONS:
                    mm = {**mm, "cost": _HLE_COST_CORRECTIONS[hle_key]}
                pts[level] = mm

        if not pts:
            ax.set_visible(False)
            continue

        levels_present = [l for l in effort_levels if l in pts]
        x_vals = [pts[l]["cost"] for l in levels_present]
        y_vals = [pts[l]["accuracy"] for l in levels_present]
        ax.plot(x_vals, y_vals, "-", color="#00897B", linewidth=2, alpha=0.5, zorder=3)

        for level in levels_present:
            ax.scatter([pts[level]["cost"]], [pts[level]["accuracy"]],
                       color=colors[level], marker=markers[level],
                       s=150 if level == "Med" else 100,
                       zorder=5, label=level)

        ax.set_title(bench_cfg["title"], fontsize=11, fontweight="bold")
        ax.set_xlabel("$/q", fontsize=10)
        if idx == 0:
            ax.set_ylabel("Accuracy (%)", fontsize=10)
        ax.tick_params(axis="both", labelsize=9)
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"${v:.2f}"))

        min_y = min(y_vals)
        max_y = max(y_vals)
        margin_y = max((max_y - min_y) * 0.3, 1.5)
        ax.set_ylim(min_y - margin_y, max_y + margin_y)

    # Shared legend outside, below all subplots
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.02),
               ncol=3, fontsize=10, frameon=False)
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    out = "effort_ablation"
    fig.savefig(FIGURES_DIR / f"{out}.pdf", bbox_inches="tight")
    fig.savefig(FIGURES_DIR / f"{out}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}.pdf")


def plot_orch_ablation() -> None:
    """1x3 grid: orchestrator model (Haiku/Sonnet/Opus) per benchmark, shared legend outside."""
    bench_keys = ["hle", "lcb", "gpqa"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    orch_levels = ["Haiku", "Sonnet", "Opus"]
    colors = {"Haiku": "#795548", "Sonnet": "#E91E63", "Opus": "#1565C0"}
    markers = {"Haiku": "s", "Sonnet": "*", "Opus": "D"}

    for idx, bench_key in enumerate(bench_keys):
        ax = axes[idx]
        bench_cfg = BENCHMARKS[bench_key]

        pts = {}
        # Sonnet = default ATTS (no_integrate)
        if "no_integrate" in bench_cfg and bench_cfg["no_integrate"].exists():
            ni = parse_delegated_log(bench_cfg["no_integrate"])
            if ni is not None:
                cost = _HLE_COST_CORRECTIONS["atts"] if bench_key == "hle" else ni["agg_cost"]
                pts["Sonnet"] = {"accuracy": ni["integrated_pct"], "cost": cost}

        for key, label in [("haiku_orch", "Haiku"), ("opus_orch", "Opus")]:
            if key not in bench_cfg or not bench_cfg[key].exists():
                continue
            vdata = parse_delegated_log(bench_cfg[key])
            if vdata is not None:
                hle_key = f"{key.replace('_orch', '')}_orch"
                cost = _HLE_COST_CORRECTIONS.get(hle_key, vdata["agg_cost"]) if bench_key == "hle" else vdata["agg_cost"]
                pts[label] = {"accuracy": vdata["integrated_pct"], "cost": cost}
            else:
                # Fallback: read from progress.json in the run directory
                run_dirs = sorted(bench_cfg[key].parent.glob("run_*/progress.json"))
                if run_dirs:
                    pdata = json.loads(run_dirs[-1].read_text())
                    s = pdata.get("summary", {})
                    if s.get("total"):
                        pts[label] = {"accuracy": s["correct"] / s["total"] * 100, "cost": s["total_cost_usd"] / s["total"]}

        levels_present = [l for l in orch_levels if l in pts]
        x_vals = [pts[l]["cost"] for l in levels_present]
        y_vals = [pts[l]["accuracy"] for l in levels_present]
        ax.plot(x_vals, y_vals, "-", color="#888888", linewidth=2, alpha=0.4, zorder=3)

        for level in levels_present:
            ax.scatter([pts[level]["cost"]], [pts[level]["accuracy"]],
                       color=colors[level], marker=markers[level],
                       s=150 if level == "Sonnet" else 100,
                       zorder=5, label=level)

        ax.set_title(bench_cfg["title"], fontsize=11, fontweight="bold")
        ax.set_xlabel("$/q", fontsize=10)
        if idx == 0:
            ax.set_ylabel("Accuracy (%)", fontsize=10)
        ax.tick_params(axis="both", labelsize=9)
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"${v:.2f}"))

        min_y = min(y_vals)
        max_y = max(y_vals)
        margin_y = max((max_y - min_y) * 0.3, 1.5)
        ax.set_ylim(min_y - margin_y, max_y + margin_y)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.02),
               ncol=3, fontsize=10, frameon=False)
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    out = "orch_ablation"
    fig.savefig(FIGURES_DIR / f"{out}.pdf", bbox_inches="tight")
    fig.savefig(FIGURES_DIR / f"{out}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}.pdf")


def plot_main_results_combined() -> None:
    """1x4 combined figure for the 4 main benchmarks, shared legend outside."""
    bench_keys = ["hle", "lcb", "gpqa", "babyvision"]
    fig, axes = plt.subplots(1, 4, figsize=(20, 4))

    # Collect all legend entries from first panel
    for idx, bench_key in enumerate(bench_keys):
        ax = axes[idx]
        bench_cfg = BENCHMARKS[bench_key]
        delegated = parse_delegated_log(bench_cfg["delegated"])
        if delegated is None:
            continue

        all_accs = []
        all_costs = []

        bon_source = bench_cfg.get("no_integrate")
        ni = parse_delegated_log(bon_source) if (bon_source and bon_source.exists()) else None
        bon_data = ni if ni is not None else delegated

        # Pass@1
        pass1_cost = bon_data["costs"][0]
        pass1_acc = bon_data["oracle_pcts"][0]
        ax.scatter([pass1_cost], [pass1_acc], color="#607D8B", marker="o", s=100, zorder=5,
                   label="Pass@1" if idx == 0 else "")
        all_accs.append(pass1_acc)
        all_costs.append(pass1_cost)

        # Baselines
        rerank_cost = get_rerank_cost(bench_cfg["delegated"], delegated["total"])
        if bench_key == "hle":
            rerank_cost = _HLE_COST_CORRECTIONS["rerank"]
        for method_name, log_path in bench_cfg["methods"].items():
            if not log_path.exists():
                continue
            data = parse_method_log(log_path)
            if data is None:
                continue
            mcolor = BASELINE_STYLES.get(method_name, {}).get("color", "#607D8B")
            cost = rerank_cost if method_name in bench_cfg.get("rerank_methods", []) else data["cost"]
            ax.scatter([cost], [data["accuracy"]], color=mcolor, marker="o", s=100, zorder=5,
                       label=method_name if idx == 0 else "")
            all_accs.append(data["accuracy"])
            all_costs.append(cost)

        # LLM Selection
        llm_sel_dir = bench_cfg.get("llm_selection")
        if llm_sel_dir and llm_sel_dir.exists():
            llm_sel = read_mm_effort(llm_sel_dir)
            if llm_sel is not None:
                mcolor = BASELINE_STYLES["LLM Selection"]["color"]
                llm_cost = _HLE_COST_CORRECTIONS["llm_selection"] if bench_key == "hle" else llm_sel["cost"]
                ax.scatter([llm_cost], [llm_sel["accuracy"]], color=mcolor,
                           marker="o", s=100, zorder=5,
                           label="LLM Selection" if idx == 0 else "")
                all_accs.append(llm_sel["accuracy"])
                all_costs.append(llm_cost)

        # ATTS
        ours_points = []
        if ni is not None:
            atts_cost = _HLE_COST_CORRECTIONS["atts"] if bench_key == "hle" else ni["agg_cost"]
            ours_points.append(("ATTS", atts_cost, ni["integrated_pct"]))
        effort_dirs = bench_cfg.get("effort_dirs", {})
        if "Med" in effort_dirs:
            mm = read_mm_effort(effort_dirs["Med"])
            if mm is not None:
                mm_cost = _HLE_COST_CORRECTIONS["multi_med"] if bench_key == "hle" else mm["cost"]
                ours_points.append(("ATTS Multi Med", mm_cost, mm["accuracy"]))

        for label, cost, acc in ours_points:
            mcolor = OURS_STYLES.get(label, {}).get("color", "#E91E63")
            ax.scatter([cost], [acc], color=mcolor, marker="*", s=250, zorder=6,
                       label=label if idx == 0 else "")
            all_accs.append(acc)
            all_costs.append(cost)

        ax.set_title(bench_cfg["title"], fontsize=14, fontweight="bold")
        ax.set_xlabel("$/q", fontsize=12)
        if idx == 0:
            ax.set_ylabel("Accuracy (%)", fontsize=12)
        ax.tick_params(axis="both", labelsize=11)
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"${v:.2f}"))

        min_acc = min(all_accs)
        max_acc = max(all_accs)
        margin = max((max_acc - min_acc) * 0.15, 2.0)
        ax.set_ylim(max(0, min_acc - margin), min(100, max_acc + margin))
        ax.set_xlim(0, max(all_costs) * 1.15)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.02),
               ncol=len(labels), fontsize=11, frameon=False)
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    out = "main_cost_vs_accuracy"
    fig.savefig(FIGURES_DIR / f"{out}.pdf", bbox_inches="tight")
    fig.savefig(FIGURES_DIR / f"{out}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}.pdf")


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    for bench_key, bench_cfg in BENCHMARKS.items():
        print(f"Plotting {bench_key}...")
        plot_benchmark(bench_key, bench_cfg)

    print("Plotting main results combined...")
    plot_main_results_combined()
    print("Plotting effort ablation...")
    plot_effort_ablation()
    print("Plotting orchestrator ablation...")
    plot_orch_ablation()


if __name__ == "__main__":
    main()
