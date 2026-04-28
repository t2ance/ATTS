"""Abstract benchmark interface and shared utilities for the TTS agent."""

from __future__ import annotations

import argparse
import base64
import io
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from prompts import format_claude_structured_suffix


# ---------------------------------------------------------------------------
# Shared schemas (answer-based benchmarks)
# ---------------------------------------------------------------------------

_ANSWER_DESCRIPTION = (
    "The final answer only -- a short, direct value "
    "(number, name, letter, formula, etc.) with no preamble or explanation"
)

EXPLORE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "approach": {
            "type": "string",
            "description": "What method/angle you used (one sentence)",
        },
        "reasoning": {
            "type": "string",
            "description": "Detailed step-by-step reasoning",
        },
        "answer": {
            "type": "string",
            "description": _ANSWER_DESCRIPTION,
        },
        "confidence": {
            "type": "number",
            "description": "Your confidence in this answer (0.0 - 1.0)",
        },
    },
    "required": ["approach", "reasoning", "answer", "confidence"],
    "additionalProperties": False,
}

def make_structured_output_function_schema(
    explore_schema: dict[str, Any],
    name: str = "StructuredOutput",
    description: str = "Submit the final answer.",
) -> dict[str, Any]:
    """Wrap an explore_schema into an OpenAI Function tool schema.

    Single bridge between the canonical Python EXPLORE_SCHEMA and any consumer
    that needs the OpenAI Function tool shape (verl tool_config.yaml,
    SFT data builders, etc.).
    """
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": explore_schema,
        },
    }


INTEGRATION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "analysis": {
            "type": "string",
            "description": "Analysis of all candidates (agreements, disagreements, strengths/weaknesses)",
        },
        "final_answer": {
            "type": "string",
            "description": _ANSWER_DESCRIPTION,
        },
        "reasoning": {
            "type": "string",
            "description": "Complete reasoning for the final answer",
        },
    },
    "required": ["analysis", "final_answer", "reasoning"],
    "additionalProperties": False,
}


# ---------------------------------------------------------------------------
# Shared prompts
# ---------------------------------------------------------------------------

ANSWER_FORMAT_RULES = """\
IMPORTANT -- Answer format rules:
- The answer field MUST contain ONLY a short, direct value (a number, a name, a letter, a formula, etc.).
- Do NOT include preambles, explanations, units, or surrounding text in the answer field.
- Do NOT include code, instructions to run code, or "please compute" in the answer field.
- Put all reasoning and analysis in the dedicated reasoning fields, NOT in the answer field."""

EXPLORER_BASE_PROMPT = f"""\
You are an expert problem solver. Solve the given problem step by step.
If you cannot solve it exactly, give your best estimate and set confidence accordingly.

{ANSWER_FORMAT_RULES}
"""

INTEGRATOR_BASE_PROMPT = f"""\
You are an expert answer synthesizer. You will receive a problem and multiple candidate solutions.

Your job:
1. Analyze each candidate's reasoning individually
2. Identify common conclusions and points of disagreement
3. For disagreements, judge which side has more rigorous reasoning
4. If you find logical errors in any candidate, point them out
5. Produce a verified final answer

{ANSWER_FORMAT_RULES}
"""


# ---------------------------------------------------------------------------
# Image utility
# ---------------------------------------------------------------------------

def image_to_data_url(image) -> str:
    """Convert a PIL Image to a base64 data URL."""
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/png;base64,{b64}"


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

Candidate3 = tuple[str, bool, float]  # (norm_answer, is_correct, cost_usd)


def compute_best_of_n(
    candidates_per_question: list[list[Candidate3]],
) -> tuple[dict[int, int], dict[int, int], dict[int, float]]:
    """Compute oracle best-of-n, majority-vote-of-n, and explore cost-of-n."""
    max_n = max((len(c) for c in candidates_per_question), default=0)

    oracle: dict[int, int] = {}
    majority: dict[int, int] = {}
    cost: dict[int, float] = {}

    for n in range(1, max_n + 1):
        oracle_correct = 0
        majority_correct = 0
        total_cost = 0.0
        for cands in candidates_per_question:
            first_n = cands[:n]
            if any(ic for _, ic, _ in first_n):
                oracle_correct += 1
            if first_n:
                counts: dict[str, int] = {}
                first_seen: dict[str, int] = {}
                for i, (norm_ans, _, _) in enumerate(first_n):
                    counts[norm_ans] = counts.get(norm_ans, 0) + 1
                    if norm_ans not in first_seen:
                        first_seen[norm_ans] = i
                winner = max(counts, key=lambda a: (counts[a], -first_seen[a]))
                if next(ic for na, ic, _ in first_n if na == winner):
                    majority_correct += 1
            total_cost += sum(c for _, _, c in first_n)
        oracle[n] = oracle_correct
        majority[n] = majority_correct
        cost[n] = total_cost

    return oracle, majority, cost


def compute_aggregator_stats(
    candidates_per_question: list[list[Candidate3]],
    integrated: list[tuple[str, bool]],
) -> tuple[int, int]:
    """Compute oracle and majority-vote counts including the integrated answer."""
    oracle_correct = 0
    majority_correct = 0
    for cands, (int_ans, int_correct) in zip(candidates_per_question, integrated):
        if any(ic for _, ic, _ in cands) or int_correct:
            oracle_correct += 1
        counts: dict[str, int] = {}
        first_seen: dict[str, int] = {}
        for i, (na, _, _) in enumerate(cands):
            counts[na] = counts.get(na, 0) + 1
            if na not in first_seen:
                first_seen[na] = i
        idx = len(cands)
        counts[int_ans] = counts.get(int_ans, 0) + 1
        if int_ans not in first_seen:
            first_seen[int_ans] = idx
        winner = max(counts, key=lambda a: (counts[a], -first_seen[a]))
        all_correct = {na: ic for na, ic, _ in cands}
        all_correct[int_ans] = all_correct.get(int_ans, False) or int_correct
        if all_correct[winner]:
            majority_correct += 1
    return oracle_correct, majority_correct


MODEL_COLORS = {"haiku": "#4CAF50", "sonnet": "#2196F3", "opus": "#9C27B0"}


def _finalize_plot(fig, ax, all_accs: list[float], max_cost: float, run_dir: Path, ncol: int = 3) -> None:
    """Shared axis formatting and save for cost-vs-accuracy plots."""
    import matplotlib.pyplot as plt

    ax.set_xlabel("Avg Cost per Question (USD)", fontsize=18)
    ax.set_ylabel("Accuracy (%)", fontsize=18)
    ax.tick_params(axis="both", labelsize=14)
    ax.grid(True, alpha=0.3)

    min_acc = min(all_accs)
    max_acc = max(all_accs)
    acc_margin = max((max_acc - min_acc) * 0.15, 2.0)
    ax.set_ylim(max(0, min_acc - acc_margin), min(100, max_acc + acc_margin))
    ax.set_xlim(0, max_cost * 1.15)

    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"${v:.2f}"))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.2f}"))

    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=ncol, fontsize=12, frameon=False)
    fig.savefig(run_dir / "cost_vs_accuracy.png", dpi=150, bbox_inches="tight")
    fig.savefig(run_dir / "cost_vs_accuracy.pdf", bbox_inches="tight")
    plt.close(fig)


def save_cost_accuracy_plot(summary: dict, run_dir: Path, majority_vote: bool = True) -> None:
    """Save cost vs accuracy plot from summary data."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    total = summary["total"]
    per_model = summary.get("per_model_bon")

    if per_model:
        _save_multi_model_plot(plt, summary, run_dir, per_model, total)
    else:
        _save_single_model_plot(plt, summary, run_dir, majority_vote, total)


def _save_single_model_plot(plt, summary: dict, run_dir: Path, majority_vote: bool, total: int) -> None:
    oracle = summary.get("oracle_best_of_n", {})
    majority = summary.get("majority_vote_of_n", {})
    explore_cost = summary.get("explore_cost_of_n", {})
    if not oracle:
        return

    ns = sorted(int(k) for k in oracle)
    oracle_pct = [oracle[str(n)] / total * 100 for n in ns]
    cost_vals = [explore_cost[str(n)] / total for n in ns]

    agg_cost = summary["total_cost_usd"] / total
    agg_oracle_pct = summary.get("aggregator_oracle", 0) / total * 100
    integrated_pct = summary["correct"] / total * 100

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(cost_vals, oracle_pct, "o-", color="#2196F3", linewidth=2, markersize=6, label="Sonnet Oracle BoN", zorder=3)
    ax.scatter([agg_cost], [agg_oracle_pct], color="#2196F3", s=100, zorder=5, marker="D",
               label=f"Oracle (+ integrated) ({agg_oracle_pct:.2f}%)")

    all_accs = oracle_pct + [agg_oracle_pct, integrated_pct]

    if majority_vote and majority:
        majority_pct = [majority[str(n)] / total * 100 for n in ns]
        agg_majority_pct = summary.get("aggregator_majority", 0) / total * 100
        ax.plot(cost_vals, majority_pct, "s-", color="#FF9800", linewidth=2, markersize=6, label="Majority vote best-of-n", zorder=3)
        ax.scatter([agg_cost], [agg_majority_pct], color="#FF9800", s=100, zorder=5, marker="D",
                   label=f"Majority (+ integrated) ({agg_majority_pct:.2f}%)")
        all_accs += majority_pct + [agg_majority_pct]

    ax.scatter([agg_cost], [integrated_pct], color="#E91E63", s=120, zorder=5, marker="*",
               label=f"TTS-Agent ({integrated_pct:.2f}% @ ${agg_cost:.2f})")

    for n, x, y in zip(ns, cost_vals, oracle_pct):
        ax.annotate(f"n={n}", (x, y), textcoords="offset points", xytext=(0, 10), fontsize=10, ha="center", color="#2196F3")
    ax.annotate("+integrated", (agg_cost, agg_oracle_pct), textcoords="offset points", xytext=(0, 10), fontsize=10, ha="center", color="#2196F3")

    _finalize_plot(fig, ax, all_accs, max(cost_vals[-1], agg_cost), run_dir, ncol=3)


def _save_multi_model_plot(plt, summary: dict, run_dir: Path, per_model: dict, total: int) -> None:
    agg_cost = summary["total_cost_usd"] / total
    integrated_pct = summary["correct"] / total * 100

    fig, ax = plt.subplots(figsize=(8, 5))
    all_accs = [integrated_pct]
    all_costs = [agg_cost]

    for model, data in per_model.items():
        oracle = data.get("oracle_best_of_n", {})
        cost_data = data.get("explore_cost_of_n", {})
        if not oracle:
            continue
        ns = sorted(int(k) for k in oracle)
        oracle_pct = [oracle[str(n)] / total * 100 for n in ns]
        cost_vals = [cost_data[str(n)] / total for n in ns]
        color = MODEL_COLORS.get(model, "#666666")

        ax.plot(cost_vals, oracle_pct, "o-", color=color, linewidth=2, markersize=6,
                label=f"{model.capitalize()} Oracle BoN", zorder=3)
        for n, x, y in zip(ns, cost_vals, oracle_pct):
            ax.annotate(f"n={n}", (x, y), textcoords="offset points", xytext=(0, 10),
                        fontsize=8, ha="center", color=color)
        all_accs.extend(oracle_pct)
        all_costs.extend(cost_vals)

    ax.scatter([agg_cost], [integrated_pct], color="#E91E63", s=150, zorder=5, marker="*",
               label=f"Multi-Model ATTS ({integrated_pct:.2f}% @ ${agg_cost:.2f})")

    _finalize_plot(fig, ax, all_accs, max(all_costs), run_dir, ncol=2)


# ---------------------------------------------------------------------------
# Abstract benchmark interface
# ---------------------------------------------------------------------------

class BenchmarkConfig(ABC):
    """Base class for benchmark configurations.

    Subclasses must set class attributes: name, filter_keys, judge_model.
    Override explore_schema / integrate_schema / explorer_base_prompt /
    integrator_base_prompt only when they differ from the defaults.
    """

    name: str
    filter_keys: tuple[str, ...]
    judge_model: str | None
    majority_vote_compatible: bool = True
    explore_schema: dict[str, Any] = EXPLORE_SCHEMA
    integrate_schema: dict[str, Any] = INTEGRATION_SCHEMA
    explorer_base_prompt: str = EXPLORER_BASE_PROMPT
    integrator_base_prompt: str = INTEGRATOR_BASE_PROMPT

    # -- Dataset --

    @abstractmethod
    def load_dataset(self) -> list[dict]:
        ...

    @abstractmethod
    def filter_dataset(self, rows: list[dict], **kwargs) -> list[dict]:
        ...

    @abstractmethod
    def get_question(self, row: dict) -> str:
        ...

    @abstractmethod
    def get_answer(self, row: dict) -> str:
        ...

    @abstractmethod
    def get_id(self, row: dict) -> str:
        ...

    @abstractmethod
    def get_image(self, row: dict) -> str | None:
        ...

    @abstractmethod
    def classify_subset(self, row: dict) -> str:
        ...

    # -- Grading --
    # Each benchmark subclass implements its own async grade(predicted, gold,
    # question, row, backend, out_dir) calling grader.py primitives directly.
    # eval.py:_grade_with_cache reads benchmark.judge_model (declared per subclass)
    # as a cache-invalidation key.

    def normalize_answer(self, text: str) -> str:
        """Normalize answer text for comparison (e.g. majority vote)."""
        from benchmarks.grader import normalize_answer
        return normalize_answer(text)

    # -- Prompts --

    def get_explore_schema(self) -> dict[str, Any]:
        return self.explore_schema

    def get_integrate_schema(self) -> dict[str, Any]:
        return self.integrate_schema

    def get_explorer_system_prompt(self, backend: str) -> str:
        if backend == "claude":
            return self.explorer_base_prompt + format_claude_structured_suffix(self.explore_schema)
        return self.explorer_base_prompt

    def get_integrator_system_prompt(self, backend: str) -> str:
        if backend == "claude":
            return self.integrator_base_prompt + format_claude_structured_suffix(self.integrate_schema)
        return self.integrator_base_prompt

    def build_explorer_message(self, problem: str) -> str:
        return f"## Problem\n\n{problem}\n\nSolve this problem step by step."

    def build_integrator_message(self, problem: str, candidates: list) -> str:
        parts = [f"## Problem\n\n{problem}\n\n## Candidate Solutions\n"]
        for i, c in enumerate(candidates, 1):
            parts.append(
                f"### Candidate {i}\n"
                f"- **Approach**: {c.approach}\n"
                f"- **Answer**: {c.answer}\n"
                f"- **Confidence**: {c.confidence}\n"
                f"- **Reasoning**: {c.reasoning}\n"
            )
        parts.append("\nAnalyze all candidates and produce your final answer.")
        return "\n".join(parts)

    def get_answer_from_explore(self, result: dict) -> str:
        return result["answer"]

    def get_answer_from_integrate(self, result: dict) -> str:
        return result["final_answer"]

    # -- CLI --

    def make_filter_model(self) -> type["BaseModel"]:
        """Return a Pydantic model class describing this benchmark's filter fields.

        Every concrete benchmark MUST override this. Use extra='forbid' on the
        returned model so misspelled filter keys fail loudly.
        """
        raise NotImplementedError(
            f"{type(self).__name__} must override make_filter_model()"
        )

    def add_dataset_args(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--num", type=int, default=None, help="Number of questions")
        parser.add_argument("--skip", type=int, default=0, help="Skip first N questions (applied after filtering, before --num)")
        parser.add_argument("--seed", type=int, default=42)
        parser.add_argument("--shuffle", action="store_true")

    def add_model_args(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--backend", type=str, required=True, choices=["codex", "claude", "vllm"], help="API backend")
        parser.add_argument("--explore-model", type=str, required=True, help="Model for explore sub-model calls")
        parser.add_argument("--budget-tokens", type=int, default=32000, help="Thinking token budget for Claude backend (default: 32000)")
        parser.add_argument("--effort", choices=["low", "medium", "high", "max"], default="low", help="Reasoning effort level (default: low)")
        parser.add_argument("--num-explores", type=int, default=8, help="Explore rounds per question (default: 8)")
        parser.add_argument("--cache-dirs", type=str, default=None,
                            help="Cache directory for explore results. Single path for one model, or model:path pairs for multi-model (e.g. haiku:cache/haiku,sonnet:cache/sonnet)")
        parser.add_argument("--num-workers", type=int, default=1, help="Concurrent workers (default: 1)")
        parser.add_argument("--explore-timeout", type=float, default=1200, help="Timeout in seconds per explore call (default: 1200)")
        parser.add_argument("--max-output-chars", type=int, default=None,
                            help="Max orchestrator output chars (thinking+response). Kills session if exceeded. Default: no limit.")

    # -- Metrics --

    def compute_metrics(
        self,
        candidates_per_question: list[list[Candidate3]],
        integrated: list[tuple[str, bool]],
        subset_labels: list[str] | None = None,
        per_model_candidates: dict[str, list[list[Candidate3]]] | None = None,
    ) -> dict:
        oracle_bon, majority_bon, cost_bon = compute_best_of_n(candidates_per_question)
        agg_oracle, agg_majority = compute_aggregator_stats(candidates_per_question, integrated)
        result = {
            "oracle_best_of_n": {str(k): v for k, v in oracle_bon.items()},
            "explore_cost_of_n": {str(k): v for k, v in cost_bon.items()},
            "aggregator_oracle": agg_oracle,
        }
        if self.majority_vote_compatible:
            result["majority_vote_of_n"] = {str(k): v for k, v in majority_bon.items()}
            result["aggregator_majority"] = agg_majority
        if per_model_candidates:
            per_model_bon = {}
            for model, cands in per_model_candidates.items():
                o, m, c = compute_best_of_n(cands)
                entry: dict[str, Any] = {
                    "oracle_best_of_n": {str(k): v for k, v in o.items()},
                    "explore_cost_of_n": {str(k): v for k, v in c.items()},
                }
                if self.majority_vote_compatible:
                    entry["majority_vote_of_n"] = {str(k): v for k, v in m.items()}
                per_model_bon[model] = entry
            result["per_model_bon"] = per_model_bon
        if subset_labels is not None:
            per_subset: dict[str, dict[str, int]] = {}
            per_subset_single: dict[str, dict[str, int]] = {}
            per_subset_majority: dict[str, dict[str, int]] = {}
            for label, cands, (_, is_correct) in zip(subset_labels, candidates_per_question, integrated):
                # Integrated (TTS-Agent)
                entry = per_subset.setdefault(label, {"correct": 0, "total": 0})
                entry["total"] += 1
                if is_correct:
                    entry["correct"] += 1
                # Single-pass (best-of-1)
                entry_s = per_subset_single.setdefault(label, {"correct": 0, "total": 0})
                entry_s["total"] += 1
                if cands and cands[0][1]:
                    entry_s["correct"] += 1
                # Majority vote (all candidates)
                if self.majority_vote_compatible and cands:
                    entry_m = per_subset_majority.setdefault(label, {"correct": 0, "total": 0})
                    entry_m["total"] += 1
                    counts: dict[str, int] = {}
                    first_seen: dict[str, int] = {}
                    for i, (na, _, _) in enumerate(cands):
                        counts[na] = counts.get(na, 0) + 1
                        if na not in first_seen:
                            first_seen[na] = i
                    winner = max(counts, key=lambda a: (counts[a], -first_seen[a]))
                    if next(ic for na, ic, _ in cands if na == winner):
                        entry_m["correct"] += 1
            result["per_subset"] = per_subset
            result["per_subset_single"] = per_subset_single
            if self.majority_vote_compatible:
                result["per_subset_majority"] = per_subset_majority
        return result

    def print_metrics(self, summary: dict, total: int) -> None:
        per_model = summary.get("per_model_bon")
        if per_model:
            self._print_multi_model_metrics(summary, total, per_model)
            return

        oracle = summary.get("oracle_best_of_n", {})
        majority = summary.get("majority_vote_of_n", {})
        cost_bon = summary.get("explore_cost_of_n", {})
        max_n = max((int(k) for k in oracle), default=0)
        mv = self.majority_vote_compatible
        if max_n > 0 and total > 0:
            header = f"{'':14s} oracle  " + ("majority  " if mv else "") + "explore_cost"
            print(header)
            for n in range(1, max_n + 1):
                o_pct = oracle[str(n)] / total * 100
                e_cost = cost_bon[str(n)]
                line = f"best-of-{n:<6d} {o_pct}%    "
                if mv:
                    m_pct = majority[str(n)] / total * 100
                    line += f"{m_pct}%    "
                line += f"${e_cost}"
                print(line)
            agg_o_pct = summary.get("aggregator_oracle", 0) / total * 100
            total_cost = summary["total_cost_usd"]
            line = f"best-of-{'+agg':<6s} {agg_o_pct}%    "
            if mv:
                agg_m_pct = summary.get("aggregator_majority", 0) / total * 100
                line += f"{agg_m_pct}%    "
            line += f"${total_cost}"
            print(line)
        per_subset = summary.get("per_subset")
        if per_subset:
            ps_single = summary.get("per_subset_single", {})
            ps_majority = summary.get("per_subset_majority", {})
            labels = sorted(per_subset)
            def _pct(d: dict, k: str) -> str:
                e = d.get(k, {})
                if not e or e["total"] == 0:
                    return "---"
                return f"{e['correct']}/{e['total']} ({e['correct']/e['total']*100}%)"
            if mv:
                print(f"Per-subset accuracy:  {'single':>10s} {'majority':>10s} {'integrated':>10s}")
                for label in labels:
                    print(f"  {label:20s} {_pct(ps_single, label):>10s} {_pct(ps_majority, label):>10s} {_pct(per_subset, label):>10s}")
            else:
                print(f"Per-subset accuracy:  {'single':>10s} {'integrated':>10s}")
                for label in labels:
                    print(f"  {label:20s} {_pct(ps_single, label):>10s} {_pct(per_subset, label):>10s}")

    def _print_multi_model_metrics(self, summary: dict, total: int, per_model: dict) -> None:
        if total == 0:
            return
        for model, data in per_model.items():
            oracle = data.get("oracle_best_of_n", {})
            cost_bon = data.get("explore_cost_of_n", {})
            max_n = max((int(k) for k in oracle), default=0)
            if max_n == 0:
                continue
            print(f"  {model.capitalize()} BoN:")
            for n in range(1, max_n + 1):
                o_pct = oracle[str(n)] / total * 100
                e_cost = cost_bon[str(n)]
                print(f"    best-of-{n:<4d} {o_pct:.1f}%    ${e_cost:.2f}")

    def save_plots(self, summary: dict, run_dir: Path) -> None:
        save_cost_accuracy_plot(summary, run_dir, majority_vote=self.majority_vote_compatible)
