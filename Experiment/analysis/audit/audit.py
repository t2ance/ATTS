"""Unified eval audit: 7 atomic analyses behind one fire CLI.

USAGE
    python audit.py <atom> [--key=value ...]

ATOMS (timeout-line atoms 1-6 follow the 'current -> causal -> mechanism ->
label -> global -> counterfactual' research chain; atom 7 is a separate
cache-substrate classifier)

    1. traj_timeout_dist       (bench, method)
       Per-trajectory timeout-ratio histogram.
       "Are timeouts spread thin or concentrated?"

    2. acc_by_timeout          (bench, method)
       Accuracy partitioned by timeout pattern (with-TO / no-TO / all-TO).
       "Does timeout hurt the final answer?"

    3. cross_variant_timeout   (bench, method_a, method_b)
       For each problem where method_b timed out, look at method_a's behavior on
       the same problem. Separates problem-difficulty from setup-induced timeout.

    4. explore_duration_dist   (bench, method, run_name=None)
       Wall-clock duration distribution of explore calls, split by timed_out flag.
       "Did 'timeout' actually hit the 1200s cap or terminate early?"

    5. method_pareto           (bench)
       (cost, accuracy) scatter across all method dirs for one bench, with
       Pareto frontier. "Which methods are dominated?"

    6. skip_topn_traj          (bench, method, n_skip=20)
       Drop the N most expensive problems one at a time (assume wrong);
       plot the (cost, accuracy) trajectory. "Can we earn back Pareto by
       cutting the expensive tail?"

    7. failure_mode_audit      (cache_root=None, run_root=None, out_dir=None)
       Walk the WHOLE cache, classify each cached explore into a failure mode
       (OK / F0_predicted_skip / F1_timeout / F2_compaction / F3_declined /
       F4_early_term / F5_burn), then JOIN with grade.json across run dirs to
       attach the grading method (exact / llm / mixed / ungraded) per cache
       entry. Operates on the cache substrate (not the run substrate), so
       runs once across all benches.

    audit_all                  (bench)
       Convenience: run timeout-line atoms 1-6 on this bench. Atom 7 is
       cache-wide and bench-independent, so it is NOT in audit_all -- run
       separately when needed.

ALL OUTPUTS land under analysis/audit/ with parameterized filenames:
    <atom>_<key1>_<key2>.{json,png,pdf}
    failure_per_explore.jsonl + failure_summary_by_case_mode.{tsv,md}  (atom 7)

DEFAULTS (declared at top of file)
    BENCHMARKS, ATTS_V1, ATTS_V3, TIMEOUT_CAP_SECONDS, MIN_ROWS_FRAC,
    MIN_RUN_COST_USD, N_SKIP_STEPS, SKIP_HIGHLIGHTS, MODE_ORDER, MODE_LEGEND,
    GRADING_STATUSES, KNOWN_MODELS, COMPACTION_*_MARKERS, DECLINED_ANSWER_MARKERS,
    BURN/EARLY_TERM token thresholds.

EXAMPLES
    # timeout-line atoms (per (bench, method))
    python audit.py traj_timeout_dist --bench=lcb --method=sonnet_no_integrate
    python audit.py cross_variant_timeout --bench=lcb \\
            --method_a=sonnet --method_b=sonnet_no_integrate
    python audit.py method_pareto --bench=hle
    python audit.py audit_all --bench=lcb            # 6 timeout atoms

    # cache-wide failure-mode + grading classifier
    python audit.py failure_mode_audit
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import fire
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Constants  (NOT method-specific; all benches share these)
# ---------------------------------------------------------------------------

ANALYSIS_ROOT = Path("/data3/peijia/dr-claw/Explain/Experiment/analysis/run")
OUT_DIR = Path("/data3/peijia/dr-claw/Explain/Experiment/analysis/audit")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Canonical bench list. Kept fixed (rather than discovered) so that audits run
# on a missing bench fail loudly instead of silently shrinking.
BENCHMARKS = ("aime2025", "aime2026", "babyvision", "gpqa", "hle", "lcb", "rbenchv")

# ATTS variants. Used as defaults for atoms 1/2/3/4 and the cross-variant pair.
# v1 = explore + integrate (sonnet), v3 = explore-only (sonnet_no_integrate).
ATTS_V1 = "sonnet"
ATTS_V3 = "sonnet_no_integrate"

# Wall-clock cap of an explore call. Atom 4 reports "fraction of timeouts
# whose duration is within ~1s of this cap" -- if it's 100%, all timeouts
# really did hit the cap.
TIMEOUT_CAP_SECONDS = 1200.0
TIMEOUT_CAP_TOLERANCE = 1.0

# Pareto-scatter filters for atom 5/6:
#  - MIN_ROWS_FRAC: a (method, run) point is dropped if its results.jsonl has
#    fewer rows than this fraction of the bench's max-rows-per-run. This is
#    bench-adaptive (AIME ~30, HLE ~100, GPQA ~200), unlike a fixed threshold.
#  - MIN_RUN_COST_USD: cost-tracking-bug guard (drops $0 runs).
MIN_ROWS_FRAC = 0.5
MIN_RUN_COST_USD = 0.01

# Skip-top-N atom 6: how far to walk and which steps to label on the plot.
N_SKIP_STEPS = 20
SKIP_HIGHLIGHTS = (1, 2, 3, 5, 10)


# ---------------------------------------------------------------------------
# L1 primitives  (data access; method-agnostic)
# ---------------------------------------------------------------------------

def discover_methods(bench: str) -> list[str]:
    """Return method dir names under run/<bench>/ that contain at least one
    `run_*/results.jsonl`. Drops names ending in `.log` (precache stragglers)."""
    bench_dir = ANALYSIS_ROOT / bench
    if not bench_dir.is_dir():
        return []
    out = []
    for md in sorted(bench_dir.iterdir()):
        if not md.is_dir(): continue
        if md.name.endswith(".log"): continue
        if any(md.glob("run_*/results.jsonl")):
            out.append(md.name)
    return out


def latest_run(bench: str, method: str) -> Path | None:
    """Lex-max run_* directory under run/<bench>/<method>/. Names embed timestamps."""
    bench_dir = ANALYSIS_ROOT / bench / method
    if not bench_dir.is_dir():
        return None
    runs = sorted(bench_dir.glob("run_*"))
    return runs[-1] if runs else None


def load_results(bench: str, method: str) -> list[dict]:
    """Concatenate every results.jsonl under run/<bench>/<method>/run_*."""
    bench_dir = ANALYSIS_ROOT / bench / method
    if not bench_dir.is_dir():
        return []
    rows = []
    for run_dir in sorted(bench_dir.glob("run_*")):
        rj = run_dir / "results.jsonl"
        if not rj.is_file():
            continue
        with rj.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                d["_run"] = run_dir.name
                d["_bench"] = bench
                d["_method"] = method
                rows.append(d)
    return rows


def row_timeout_pattern(row: dict) -> tuple[int, int]:
    """For one results.jsonl row, return (n_explores, n_timed_out_explores)."""
    n_exp = n_to = 0
    for r in row.get("rounds", []):
        if r.get("action") != "explore":
            continue
        n_exp += 1
        if r.get("reasoning") == "timed out":
            n_to += 1
    return n_exp, n_to


def walk_trajectories(bench: str, method: str):
    """Yield (run_dir, problem_dir) for every trajectory under (bench, method)."""
    bench_dir = ANALYSIS_ROOT / bench / method
    if not bench_dir.is_dir():
        return
    for run_dir in sorted(bench_dir.glob("run_*")):
        traj_root = run_dir / "trajectories"
        if not traj_root.is_dir():
            continue
        for prob_dir in sorted(traj_root.iterdir()):
            if prob_dir.is_dir():
                yield run_dir, prob_dir


def _save_json(payload, path: Path) -> None:
    path.write_text(json.dumps(payload, indent=2))


# Default cache_root / run_root for atom 7 (failure_mode_audit). The cache
# substrate is `Experiment/analysis/cache/`; ANALYSIS_ROOT (above) is its
# sibling `run/`.
CACHE_ROOT_DEFAULT = ANALYSIS_ROOT.parent / "cache"
RUN_ROOT_DEFAULT = ANALYSIS_ROOT


# ---------------------------------------------------------------------------
# Constants for atom 7  (cache-substrate failure-mode classification)
# ---------------------------------------------------------------------------

COMPACTION_REASONING_MARKERS: tuple[str, ...] = (
    "ran out of context",
    "resumed from a previous session",
    "compacted",
    "stop hooks indicate StructuredOutput",
    "previous conversation context",
)

COMPACTION_ANSWER_MARKERS: tuple[str, ...] = (
    "StructuredOutput called as required",
    "Task completed. The StructuredOutput tool",
    "Task completed - StructuredOutput",
    "Task complete. Please restate your question",
    "Session resumed. Please provide a problem",
    "Task completed. The previous conversation context",
)

DECLINED_ANSWER_MARKERS: tuple[str, ...] = (
    "Unable to determine",
    "I cannot answer",
    "I don't know",
    "I do not know",
    "cannot be determined",
    "insufficient information",
    "I cannot provide",
)

BURN_OUTPUT_TOKEN_THRESHOLD: int = 250_000
EARLY_TERM_OUTPUT_TOKEN_THRESHOLD: int = 200_000
EARLY_TERM_TRAJECTORY_TAIL_BYTES: int = 2_000

# Some benchmarks store the model's response under a non-`answer` key
# (e.g., LCB stores generated code under `code`). The detector tries each
# in order and uses the first non-empty value as the effective answer.
ANSWER_FIELD_CANDIDATES: tuple[str, ...] = ("answer", "code")

# Known model prefixes used to extract the underlying model from a cache
# namespace. Order matters: longer prefixes must come before their shorter
# substrings so `gpt5.2_low` matches before `gpt5.2`. A namespace like
# `sonnet_self_refine` resolves to `sonnet`; `sonnet/gold` (HLE subset path)
# also resolves to `sonnet`.
KNOWN_MODELS: tuple[str, ...] = (
    "gpt5.2_low",
    "gpt5.4",
    "gpt5.2",
    "haiku",
    "sonnet",
    "opus",
)

MODE_ORDER: tuple[str, ...] = (
    "OK",
    "F0_predicted_skip",
    "F1_timeout",
    "F2_compaction",
    "F3_declined",
    "F4_early_term",
    "F5_burn",
)

MODE_LEGEND: dict[str, str] = {
    "OK":                "Model returned a real answer. Usable.",
    "F0_predicted_skip": "We never asked the model. The system pre-decided this call would take too long and skipped it. No answer.",
    "F1_timeout":        "We asked the model, but it did not finish within the time limit. We cut it off. No answer.",
    "F2_compaction":     "The model ran so long that its memory of the question was wiped mid-call. It then returned a meaningless placeholder like \"Task completed.\" instead of a real answer.",
    "F3_declined":       "The model replied but refused to answer. Said things like \"I don't know\" or \"insufficient information\". Often it hallucinated that an image or extra context was missing when it was not. No usable answer.",
    "F4_early_term":     "The model was still thinking when we had to stop the conversation. It never got to write the answer. No answer.",
    "F5_burn":           "The model gave a real answer, but spent ~5x the normal amount of thinking to do so. Answer is usable, the call was just expensive.",
}

GRADING_STATUSES: tuple[str, ...] = (
    "ungraded",   # no run dir has graded this cache entry yet
    "exact",      # graded only by local exact match (judge_model == "none")
    "llm",        # graded only by one or more LLM judges (no exact)
    "mixed",      # graded by both exact and LLM, or by >1 distinct LLM models
)


@dataclass(frozen=True)
class CacheRecord:
    """One classified cache entry. Used only by atom 7."""
    bench: str
    namespace: str  # may include subdir, e.g. "sonnet/gold"
    qid: str
    explore: str    # e.g. "explore_3"
    mode: str       # one of MODE_ORDER
    cost_usd: float
    output_tokens: int
    # Empirically-discovered grading methods (set of judge_model strings).
    # "none" means exact-match. Anything else is the LLM judge model name.
    # Empty frozenset = ungraded. >1 element = mixed.
    grading_methods: frozenset
    grading_status: str  # one of GRADING_STATUSES, derived from grading_methods


# ---------------------------------------------------------------------------
# L1+ primitives for atom 7
# ---------------------------------------------------------------------------

def extract_model(namespace: str) -> str:
    base = namespace.split("/")[0]
    for m in KNOWN_MODELS:
        if base == m or base.startswith(m + "_"):
            return m
    return base


def extract_variant(namespace: str) -> str:
    """Variant suffix after the model prefix; '' if the namespace is just the model."""
    base = namespace.split("/")[0]
    for m in KNOWN_MODELS:
        if base == m:
            return ""
        if base.startswith(m + "_"):
            return base[len(m) + 1:]
    return ""


def _get_answer(d: dict) -> str:
    """First non-empty answer-like field (per ANSWER_FIELD_CANDIDATES)."""
    for key in ANSWER_FIELD_CANDIDATES:
        v = d.get(key)
        if v:
            return v
    return ""


def classify_result(d: dict) -> str:
    """Apply priority-ordered failure-mode detectors to one cached result.json."""
    if d.get("timed_out") is True:
        if d.get("predicted_skip") is True:
            return "F0_predicted_skip"
        return "F1_timeout"

    answer = _get_answer(d)
    reasoning = d.get("reasoning") or ""
    trajectory = d.get("trajectory") or ""
    usage = d.get("usage") or {}
    if not isinstance(usage, dict):
        usage = {}
    output_tokens = usage.get("output_tokens", 0) or 0

    if any(m in reasoning for m in COMPACTION_REASONING_MARKERS):
        return "F2_compaction"
    if any(m in answer for m in COMPACTION_ANSWER_MARKERS):
        return "F2_compaction"

    if answer.strip() == "" or any(m in answer for m in DECLINED_ANSWER_MARKERS):
        return "F3_declined"

    if (
        output_tokens > EARLY_TERM_OUTPUT_TOKEN_THRESHOLD
        and '"answer"' not in trajectory[-EARLY_TERM_TRAJECTORY_TAIL_BYTES:]
    ):
        return "F4_early_term"

    if output_tokens > BURN_OUTPUT_TOKEN_THRESHOLD:
        return "F5_burn"

    return "OK"


def _normalize_cache_dir(cache_dir_str: str) -> tuple[str, str] | None:
    """Map a `run_config.json:cache_dir` string -> (bench, cache_namespace).

    Anchors on the literal `cache/` segment (handles both `../analysis/cache/...`
    and `playground/cache/...` writing styles). Returns None if no `cache/`
    segment with at least bench + 1 more level after it.
    """
    parts = cache_dir_str.replace("\\", "/").strip("/").split("/")
    try:
        i = parts.index("cache")
    except ValueError:
        return None
    rest = parts[i + 1:]
    if len(rest) < 2:
        return None
    return rest[0], "/".join(rest[1:])


def discover_grade_index(run_root: Path) -> dict[tuple[str, str, str, str], frozenset]:
    """Build (bench, cache_namespace, qid, explore_n) -> frozenset of judge_model.

    For each `run_*/run_config.json`, normalize `cache_dir` to (bench, namespace),
    then walk every grade.json under `grading/` AND `trajectories/`. The same
    (bench, ns, qid, explore) key may accumulate values from multiple runs that
    share the cache -- that is how 'mixed' grading is detected.
    """
    accum: dict[tuple[str, str, str, str], set[str]] = defaultdict(set)
    if not run_root.is_dir():
        return {}
    for cfg_path in run_root.rglob("run_config.json"):
        run_dir = cfg_path.parent
        try:
            cfg = json.loads(cfg_path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        ns = _normalize_cache_dir(cfg.get("cache_dir") or "")
        if ns is None:
            continue
        bench, namespace = ns
        for sub in ("grading", "trajectories"):
            base = run_dir / sub
            if not base.is_dir():
                continue
            for gp in base.rglob("grade.json"):
                rel = gp.relative_to(base).parts
                # We want only per-explore grades:
                #   (qid, explore_<n>, "grade.json")                 -- 3 parts
                #   (qid, "rollout_<k>", explore_<n>, "grade.json")  -- 4 parts
                # Skip integrate / top-level grades.
                if len(rel) < 3 or not rel[-2].startswith("explore_"):
                    continue
                qid = rel[0]
                explore = rel[-2]
                try:
                    g = json.loads(gp.read_text())
                except (OSError, json.JSONDecodeError):
                    continue
                jm = g.get("judge_model")
                accum[(bench, namespace, qid, explore)].add(jm if jm else "none")
    return {k: frozenset(v) for k, v in accum.items()}


def _grading_status_label(methods: frozenset) -> str:
    """Collapse a set of judge_model strings into a single 4-way label."""
    if not methods:
        return "ungraded"
    if methods == frozenset({"none"}):
        return "exact"
    if "none" in methods:
        return "mixed"           # at least one exact + at least one LLM
    if len(methods) == 1:
        return "llm"             # single LLM judge
    return "mixed"               # multiple distinct LLM judges


def walk_cache(
    cache_root: Path,
    grade_index: dict[tuple[str, str, str, str], frozenset] | None = None,
) -> Iterator[CacheRecord]:
    """Yield one CacheRecord per cache/<bench>/<ns>/[<sub>/]<qid>/explore_*/result.json.

    Path depth is 5 or 6 (HLE has the extra Gold/Revision/Uncertain subset level).
    If grade_index is given, joins the grading-methods set per cache entry.
    """
    if grade_index is None:
        grade_index = {}
    for path in cache_root.rglob("explore_*/result.json"):
        rel_parts = path.relative_to(cache_root).parts
        if len(rel_parts) == 5:
            bench, namespace, qid, explore, _ = rel_parts
        elif len(rel_parts) == 6:
            bench, ns_top, sub, qid, explore, _ = rel_parts
            namespace = f"{ns_top}/{sub}"
        else:
            continue

        with path.open() as f:
            d = json.load(f)

        mode = classify_result(d)
        usage = d.get("usage") or {}
        if not isinstance(usage, dict):
            usage = {}
        methods = grade_index.get((bench, namespace, qid, explore), frozenset())
        yield CacheRecord(
            bench=bench,
            namespace=namespace,
            qid=qid,
            explore=explore,
            mode=mode,
            cost_usd=float(d.get("cost_usd") or 0.0),
            output_tokens=int(usage.get("output_tokens") or 0),
            grading_methods=methods,
            grading_status=_grading_status_label(methods),
        )


# ---------------------------------------------------------------------------
# L1++ aggregation + writers for atom 7
# ---------------------------------------------------------------------------

def _aggregate_modes(records: list[CacheRecord]) -> dict:
    """(case=(bench, ns), mode) -> {count, qids}."""
    by_case_mode: dict[tuple[str, str], dict[str, dict]] = defaultdict(
        lambda: {m: {"count": 0, "qids": set()} for m in MODE_ORDER}
    )
    for r in records:
        case = (r.bench, r.namespace)
        cell = by_case_mode[case][r.mode]
        cell["count"] += 1
        cell["qids"].add(r.qid)
    return {"cases": sorted(by_case_mode.keys()), "by_case_mode": by_case_mode}


def _aggregate_grading(records: list[CacheRecord]) -> dict:
    """(case=(bench, ns), grading_status) -> {count, qids, judge_models seen}."""
    by_case: dict[tuple[str, str], dict] = defaultdict(
        lambda: {
            "by_status": {s: {"count": 0, "qids": set()} for s in GRADING_STATUSES},
            "judge_models": set(),
        }
    )
    for r in records:
        case = (r.bench, r.namespace)
        cell = by_case[case]
        cell["by_status"][r.grading_status]["count"] += 1
        cell["by_status"][r.grading_status]["qids"].add(r.qid)
        cell["judge_models"].update(r.grading_methods)
    return {"cases": sorted(by_case.keys()), "by_case": by_case}


def _build_rows(agg: dict, sort_by: str) -> list[tuple]:
    rows = []
    for case in agg["cases"]:
        bench, ns = case
        cells = agg["by_case_mode"][case]
        total_cnt = sum(c["count"] for c in cells.values())
        rows.append((bench, ns, total_cnt, cells))
    if sort_by == "benchmark":
        rows.sort(key=lambda r: (r[0], r[1]))
    elif sort_by == "model":
        rows.sort(key=lambda r: (extract_model(r[1]), extract_variant(r[1]), r[0]))
    else:
        raise AssertionError(f"unknown sort_by: {sort_by}")
    return rows


def _write_per_explore_jsonl(records: list[CacheRecord], out_path: Path) -> None:
    with out_path.open("w") as f:
        for r in records:
            f.write(json.dumps({
                "bench": r.bench,
                "namespace": r.namespace,
                "qid": r.qid,
                "explore": r.explore,
                "mode": r.mode,
                "cost_usd": r.cost_usd,
                "output_tokens": r.output_tokens,
                "grading_status": r.grading_status,
                "grading_methods": sorted(r.grading_methods),
            }) + "\n")


def _write_case_mode_tsv(agg: dict, out_path: Path) -> None:
    header = ["bench", "namespace", "total_explores", "total_qids"]
    for m in MODE_ORDER:
        header.append(f"{m}_count")
        header.append(f"{m}_qids")
    lines = ["\t".join(header)]
    for case in agg["cases"]:
        bench, ns = case
        cells = agg["by_case_mode"][case]
        total_cnt = sum(c["count"] for c in cells.values())
        all_qids: set[str] = set()
        for c in cells.values():
            all_qids |= c["qids"]
        row = [bench, ns, str(total_cnt), str(len(all_qids))]
        for m in MODE_ORDER:
            row.append(str(cells[m]["count"]))
            row.append(str(len(cells[m]["qids"])))
        lines.append("\t".join(row))
    out_path.write_text("\n".join(lines) + "\n")


def _write_case_mode_markdown(agg: dict, grading_agg: dict, out_path: Path) -> None:
    """Three tables: by-bench, by-model, and grading-method-per-case."""
    header = ["case", "model", "total"]
    for m in MODE_ORDER:
        header.append(f"{m} (n / q)")
    sep = "|" + "|".join(["---"] * len(header)) + "|"

    sections: list[str] = []
    for sort_by, title in (("benchmark", "Sorted by benchmark"),
                           ("model", "Sorted by model")):
        sections.append(f"## {title}\n")
        sections.append("| " + " | ".join(header) + " |")
        sections.append(sep)
        for bench, ns, total_cnt, cells in _build_rows(agg, sort_by):
            row_str = [f"{bench}/{ns}", extract_model(ns), str(total_cnt)]
            for m in MODE_ORDER:
                row_str.append(f"{cells[m]['count']} / {len(cells[m]['qids'])}")
            sections.append("| " + " | ".join(row_str) + " |")
        sections.append("")

    sections.append("## Grading method per case\n")
    sections.append(
        "Counts are explore files (`n`) and unique qids (`q`). `judge_models` "
        "lists every distinct `judge_model` value found in any grade.json that "
        "touched this case's cache.\n"
    )
    g_header = ["case", "model", "total"]
    for s in GRADING_STATUSES:
        g_header.append(f"{s} (n / q)")
    g_header.append("judge_models")
    sections.append("| " + " | ".join(g_header) + " |")
    sections.append("|" + "|".join(["---"] * len(g_header)) + "|")
    for case in grading_agg["cases"]:
        bench, ns = case
        cell = grading_agg["by_case"][case]
        total_cnt = sum(s["count"] for s in cell["by_status"].values())
        row = [f"{bench}/{ns}", extract_model(ns), str(total_cnt)]
        for s in GRADING_STATUSES:
            sc = cell["by_status"][s]
            row.append(f"{sc['count']} / {len(sc['qids'])}")
        models = sorted(cell["judge_models"]) or ["(none)"]
        row.append(", ".join(models))
        sections.append("| " + " | ".join(row) + " |")
    sections.append("")
    out_path.write_text("\n".join(sections) + "\n")


def _print_console_summary(agg: dict) -> None:
    print("Failure mode legend:")
    print()
    for m in MODE_ORDER:
        print(f"  {m:18s} -- {MODE_LEGEND[m]}")
    print()
    for sort_by, title in (("benchmark", "View 1 -- sorted by benchmark"),
                           ("model", "View 2 -- sorted by model")):
        print(f"=== {title} ===")
        print(f"{'case':45s} {'model':>10s} {'total':>6s}", end="")
        for m in MODE_ORDER:
            print(f" {m[:10]:>10s}", end="")
        print()
        print("-" * (45 + 11 + 7 + 11 * len(MODE_ORDER)))
        for bench, ns, total_cnt, cells in _build_rows(agg, sort_by):
            case_label = f"{bench}/{ns}"
            if len(case_label) > 44:
                case_label = case_label[:41] + "..."
            print(f"{case_label:45s} {extract_model(ns):>10s} {total_cnt:>6d}", end="")
            for m in MODE_ORDER:
                print(f" {cells[m]['count']:>10d}", end="")
            print()
        print()


def _print_grading_summary(grading_agg: dict) -> None:
    print("Grading method per case (counts = explore files):")
    print(f"{'case':45s} {'total':>6s} {'ungrad':>7s} {'exact':>7s} {'llm':>7s} {'mixed':>7s}  judge_models")
    print("-" * 110)
    for case in grading_agg["cases"]:
        bench, ns = case
        cell = grading_agg["by_case"][case]
        total = sum(s["count"] for s in cell["by_status"].values())
        case_label = f"{bench}/{ns}"
        if len(case_label) > 44:
            case_label = case_label[:41] + "..."
        models = sorted(cell["judge_models"]) or ["(none)"]
        print(
            f"{case_label:45s} {total:>6d} "
            f"{cell['by_status']['ungraded']['count']:>7d} "
            f"{cell['by_status']['exact']['count']:>7d} "
            f"{cell['by_status']['llm']['count']:>7d} "
            f"{cell['by_status']['mixed']['count']:>7d}  "
            f"{', '.join(models)}"
        )
    print()


# ---------------------------------------------------------------------------
# L2 atoms  (one method on the Audit class per atom)
# ---------------------------------------------------------------------------

class Audit:
    """Fire-exposed CLI. Each method is one atomic analysis."""

    # ---- atom 1 -----------------------------------------------------------
    def traj_timeout_dist(self, bench: str, method: str = ATTS_V3) -> dict:
        """Per-trajectory timeout-ratio histogram for (bench, method).

        Reads explore_*/result.json under each trajectory; treats a `timed_out`
        flag as the ground truth for "this explore did not finish in time".
        """
        trajectories = []
        for run_dir, prob_dir in walk_trajectories(bench, method):
            total = n_to = 0
            for ed in sorted(prob_dir.glob("explore_*")):
                rj = ed / "result.json"
                if not rj.is_file():
                    continue
                with rj.open() as f:
                    d = json.load(f)
                total += 1
                if d.get("timed_out", False):
                    n_to += 1
            if total == 0:
                continue
            trajectories.append({
                "run": run_dir.name,
                "problem_id": prob_dir.name,
                "total_explores": total,
                "timeout_count": n_to,
                "timeout_ratio": n_to / total,
            })

        n = len(trajectories)
        n_with_any = sum(1 for t in trajectories if t["timeout_count"] > 0)
        n_all_to = sum(1 for t in trajectories if t["timeout_ratio"] == 1.0)
        total_explores = sum(t["total_explores"] for t in trajectories)
        total_timeouts = sum(t["timeout_count"] for t in trajectories)
        summary = {
            "bench": bench, "method": method,
            "n_trajectories": n,
            "n_traj_with_any_timeout": n_with_any,
            "n_traj_all_timeout": n_all_to,
            "total_explores": total_explores,
            "total_timeouts": total_timeouts,
            "traj_timeout_rate": (n_with_any / n) if n else 0.0,
            "explore_timeout_rate": (total_timeouts / total_explores) if total_explores else 0.0,
        }

        # Plot
        ratios = [t["timeout_ratio"] for t in trajectories]
        if ratios:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.hist(ratios, bins=np.linspace(0, 1, 11),
                    edgecolor="black", color="#4c72b0", alpha=0.85)
            ax.set_yscale("log")
            ax.set_xlabel("timeout ratio")
            ax.set_ylabel("# trajectories")
            ax.set_title(f"{bench}/{method}\nN={n}, any-TO={n_with_any}, all-TO={n_all_to}")
            ax.grid(True, alpha=0.3, axis="y")
            fig.tight_layout()
            stem = OUT_DIR / f"traj_timeout_dist_{bench}_{method}"
            fig.savefig(f"{stem}.png", dpi=150, bbox_inches="tight")
            fig.savefig(f"{stem}.pdf", bbox_inches="tight")
            plt.close(fig)

        _save_json(
            {"summary": summary, "trajectories": trajectories},
            OUT_DIR / f"traj_timeout_dist_{bench}_{method}.json",
        )
        print(json.dumps(summary, indent=2))
        return summary

    # ---- atom 2 -----------------------------------------------------------
    def acc_by_timeout(self, bench: str, method: str = ATTS_V3) -> dict:
        """Accuracy partitioned by timeout pattern for (bench, method).

        all_TO requires at least one explore AND every explore timed out.
        """
        rows = load_results(bench, method)
        n = len(rows)
        if n == 0:
            print(f"no rows for {bench}/{method}")
            return {"bench": bench, "method": method, "n_total": 0}

        n_correct = 0
        n_with = c_with = 0
        n_no = c_no = 0
        n_all = c_all = 0
        for r in rows:
            is_corr = bool(r.get("is_correct"))
            if is_corr: n_correct += 1
            n_exp, n_to = row_timeout_pattern(r)
            if n_to > 0:
                n_with += 1
                if is_corr: c_with += 1
                if n_exp > 0 and n_to == n_exp:
                    n_all += 1
                    if is_corr: c_all += 1
            else:
                n_no += 1
                if is_corr: c_no += 1

        def _acc(c, denom):
            return (c / denom) if denom else None

        out = {
            "bench": bench, "method": method,
            "n_total": n, "acc_total": n_correct / n,
            "n_with_timeout": n_with, "acc_with_timeout": _acc(c_with, n_with),
            "n_no_timeout": n_no, "acc_no_timeout": _acc(c_no, n_no),
            "n_all_timeout": n_all, "acc_all_timeout": _acc(c_all, n_all),
        }
        _save_json(out, OUT_DIR / f"acc_by_timeout_{bench}_{method}.json")
        print(json.dumps(out, indent=2))
        return out

    # ---- atom 3 -----------------------------------------------------------
    def cross_variant_timeout(
        self,
        bench: str,
        method_a: str = ATTS_V1,
        method_b: str = ATTS_V3,
    ) -> dict:
        """Cross-variant comparison: for each problem where method_b timed out,
        what did method_a do on the same problem? Separates problem difficulty
        from setup-induced timeout."""
        rows_a = load_results(bench, method_a)
        rows_b = load_results(bench, method_b)
        a_by_id = {r["id"]: r for r in rows_a}
        b_by_id = {r["id"]: r for r in rows_b}
        common_ids = set(a_by_id) & set(b_by_id)

        problems = []
        for pid in sorted(common_ids):
            ra = a_by_id[pid]
            rb = b_by_id[pid]
            a_n_exp, a_n_to = row_timeout_pattern(ra)
            b_n_exp, b_n_to = row_timeout_pattern(rb)
            if b_n_to == 0:
                continue
            problems.append({
                "problem_id": pid,
                f"{method_a}_correct": bool(ra.get("is_correct")),
                f"{method_a}_n_explore": a_n_exp,
                f"{method_a}_n_timeout": a_n_to,
                f"{method_b}_correct": bool(rb.get("is_correct")),
                f"{method_b}_n_explore": b_n_exp,
                f"{method_b}_n_timeout": b_n_to,
                f"{method_b}_all_timeout": b_n_exp > 0 and b_n_to == b_n_exp,
            })

        n = len(problems)
        all_to_b = [p for p in problems if p[f"{method_b}_all_timeout"]]
        partial_to_b = [p for p in problems if not p[f"{method_b}_all_timeout"]]
        a_solved = sum(1 for p in problems if p[f"{method_a}_correct"])
        b_solved = sum(1 for p in problems if p[f"{method_b}_correct"])
        a_solved_all = sum(1 for p in all_to_b if p[f"{method_a}_correct"])
        a_solved_partial = sum(1 for p in partial_to_b if p[f"{method_a}_correct"])

        summary = {
            "bench": bench, "method_a": method_a, "method_b": method_b,
            "n_b_timeout_problems": n,
            "n_a_solved": a_solved,
            "n_b_solved_despite_timeout": b_solved,
            "n_b_all_timeout": len(all_to_b),
            "n_a_solved_of_b_all_timeout": a_solved_all,
            "n_b_partial_timeout": len(partial_to_b),
            "n_a_solved_of_b_partial_timeout": a_solved_partial,
        }
        _save_json(
            {"summary": summary, "problems": problems},
            OUT_DIR / f"cross_variant_{bench}_{method_a}_vs_{method_b}.json",
        )
        print(json.dumps(summary, indent=2))
        if not n:
            print(f"  (no {method_b}-timeout problems on {bench})")
        return summary

    # ---- atom 4 -----------------------------------------------------------
    def explore_duration_dist(
        self,
        bench: str,
        method: str = ATTS_V3,
        run_name: str | None = None,
    ) -> dict:
        """Wall-clock duration of explore calls, split by timed_out flag.

        Default run_name = latest run under (bench, method). The signal we want is
        whether timed-out explores actually hit TIMEOUT_CAP_SECONDS or terminated
        before that.
        """
        if run_name is None:
            run_dir = latest_run(bench, method)
            assert run_dir is not None, f"no runs under {bench}/{method}"
        else:
            run_dir = ANALYSIS_ROOT / bench / method / run_name
        traj_root = run_dir / "trajectories"
        durations = []
        if traj_root.is_dir():
            for prob_dir in sorted(traj_root.iterdir()):
                if not prob_dir.is_dir():
                    continue
                for expl_dir in sorted(prob_dir.glob("explore_*")):
                    rj = expl_dir / "result.json"
                    if not rj.is_file():
                        continue
                    with rj.open() as f:
                        d = json.load(f)
                    durations.append({
                        "trajectory_dir": prob_dir.name,
                        "explore": expl_dir.name,
                        "timed_out": d.get("timed_out", False),
                        "duration_seconds": d.get("duration_seconds", 0.0),
                    })

        timed = [d["duration_seconds"] for d in durations if d["timed_out"]]
        clean = [d["duration_seconds"] for d in durations if not d["timed_out"]]

        def _stats(xs):
            if not xs:
                return None
            xs = sorted(xs)
            return {
                "n": len(xs),
                "min": xs[0],
                "median": xs[len(xs) // 2],
                "p95": xs[int(len(xs) * 0.95)],
                "max": xs[-1],
            }

        near_cap = sum(
            1 for d in timed
            if d >= TIMEOUT_CAP_SECONDS - TIMEOUT_CAP_TOLERANCE
        )
        out = {
            "bench": bench, "method": method, "run": run_dir.name,
            "total_explores": len(durations),
            "timed_out_stats": _stats(timed),
            "clean_stats": _stats(clean),
            "n_timed_out_at_cap": near_cap,
            "frac_timed_out_at_cap": (near_cap / len(timed)) if timed else None,
            "cap_threshold_seconds": TIMEOUT_CAP_SECONDS,
        }
        _save_json(out, OUT_DIR / f"explore_duration_{bench}_{method}.json")
        print(json.dumps(out, indent=2))
        return out

    # ---- atom 5 -----------------------------------------------------------
    def method_pareto(self, bench: str) -> dict:
        """(cost, accuracy) scatter across all methods for one bench, plus
        Pareto frontier (lower-cost-first sweep keeping accuracy improvers).

        Partial-run filter is bench-adaptive: keeps runs with >= MIN_ROWS_FRAC of
        the bench's max-rows-per-run (so 30-question AIME and 100-question HLE
        both work without per-bench thresholds).
        """
        # First pass: collect candidate (method, run, rows). Second pass: filter
        # by max-rows-derived threshold.
        candidates = []
        for method in discover_methods(bench):
            bench_dir = ANALYSIS_ROOT / bench / method
            for rd in sorted(bench_dir.glob("run_*")):
                rj = rd / "results.jsonl"
                if not rj.is_file():
                    continue
                rows = [json.loads(l) for l in rj.open() if l.strip()]
                if not rows:
                    continue
                candidates.append((method, rd.name, rows))

        if not candidates:
            print(f"{bench}: no runs found")
            _save_json({"bench": bench, "n_points": 0, "points": [], "frontier": []},
                       OUT_DIR / f"pareto_{bench}.json")
            return {"bench": bench, "n_points": 0, "n_frontier": 0,
                    "points": [], "frontier": []}

        max_rows = max(len(rows) for _, _, rows in candidates)
        min_rows = max(int(max_rows * MIN_ROWS_FRAC), 1)

        points = []
        for method, run_name, rows in candidates:
            n = len(rows)
            if n < min_rows:
                continue
            c = sum(1 for r in rows if r.get("is_correct"))
            cost = sum(r.get("cost_usd", 0) for r in rows)
            if cost < MIN_RUN_COST_USD:
                continue
            points.append({
                "method": method, "run": run_name,
                "n": n, "acc": c / n, "cost": cost,
            })

        # Pareto: by ascending cost, keep strict acc improvers
        frontier = []
        max_acc = -1.0
        for p in sorted(points, key=lambda p: p["cost"]):
            if p["acc"] > max_acc:
                frontier.append(p)
                max_acc = p["acc"]

        # Plot
        if points:
            fig, ax = plt.subplots(figsize=(10, 7))
            for p in points:
                ax.scatter(p["cost"], p["acc"] * 100, c="lightgray", s=40, zorder=2)
                ax.annotate(p["method"], (p["cost"], p["acc"] * 100),
                            fontsize=6, alpha=0.5,
                            xytext=(3, 3), textcoords="offset points")
            fx = [p["cost"] for p in frontier]
            fy = [p["acc"] * 100 for p in frontier]
            ax.plot(fx, fy, "b-", linewidth=2, alpha=0.7,
                    label="Pareto frontier", zorder=3)
            ax.scatter(fx, fy, c="blue", s=80, zorder=4,
                       edgecolor="black", linewidth=0.8)
            for p in frontier:
                ax.annotate(p["method"], (p["cost"], p["acc"] * 100),
                            fontsize=8, color="blue", fontweight="bold",
                            xytext=(5, -10), textcoords="offset points")
            ax.set_xlabel("Cost ($)")
            ax.set_ylabel("Accuracy (%)")
            ax.set_title(f"{bench}: cost vs accuracy across methods")
            ax.grid(True, alpha=0.3)
            ax.legend()
            fig.tight_layout()
            stem = OUT_DIR / f"pareto_{bench}"
            fig.savefig(f"{stem}.png", dpi=150, bbox_inches="tight")
            fig.savefig(f"{stem}.pdf", bbox_inches="tight")
            plt.close(fig)

        out = {
            "bench": bench,
            "n_points": len(points),
            "n_frontier": len(frontier),
            "points": points,
            "frontier": [p["method"] for p in frontier],
        }
        _save_json(out, OUT_DIR / f"pareto_{bench}.json")
        print(f"{bench}: {len(points)} points, {len(frontier)} on frontier")
        for p in frontier:
            print(f"  frontier: {p['method']:40s} cost=${p['cost']:7.2f} acc={p['acc']*100:5.1f}%")
        return out

    # ---- atom 6 -----------------------------------------------------------
    def skip_topn_traj(
        self,
        bench: str,
        method: str = ATTS_V3,
        n_skip: int = N_SKIP_STEPS,
    ) -> dict:
        """Drop the top-N most expensive problems one at a time (assume wrong);
        report the (cost, acc) trajectory. Counterfactual for 'predictively
        skipping the expensive tail'."""
        run_dir = latest_run(bench, method)
        assert run_dir is not None, f"no runs under {bench}/{method}"
        rj = run_dir / "results.jsonl"
        rows = [json.loads(l) for l in rj.open() if l.strip()]
        assert rows, f"empty results.jsonl in {run_dir}"
        N = len(rows)
        rows_sorted = sorted(rows, key=lambda r: -r["cost_usd"])
        base_cost = sum(r["cost_usd"] for r in rows)
        base_correct = sum(1 for r in rows if r["is_correct"])

        traj = [{"k": 0, "cost": base_cost, "acc": base_correct / N, "label": "baseline"}]
        cum_cost = 0.0
        cum_correct = 0
        for k, r in enumerate(rows_sorted[:n_skip], 1):
            cum_cost += r["cost_usd"]
            cum_correct += int(r["is_correct"])
            traj.append({
                "k": k,
                "cost": base_cost - cum_cost,
                "acc": (base_correct - cum_correct) / N,
                "label": f"-top{k}",
            })

        # Plot trajectory alone (caller can visually compare against pareto_<bench>.png)
        fig, ax = plt.subplots(figsize=(8, 6))
        tx = [t["cost"] for t in traj]
        ty = [t["acc"] * 100 for t in traj]
        ax.plot(tx, ty, "r-o", linewidth=1.5, markersize=5,
                label=f"{method} skip-top-N")
        ax.scatter(tx[0], ty[0], c="red", s=200, marker="*",
                   edgecolor="black", linewidth=1, zorder=6, label="baseline")
        for k_h in SKIP_HIGHLIGHTS:
            if k_h < len(traj):
                t = traj[k_h]
                ax.annotate(t["label"], (t["cost"], t["acc"] * 100),
                            fontsize=8, color="red", fontweight="bold",
                            xytext=(0, 8), textcoords="offset points", ha="center")
        ax.set_xlabel("Cost ($)")
        ax.set_ylabel("Accuracy (%)")
        ax.set_title(f"{bench}/{method}: skip-top-N trajectory")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        stem = OUT_DIR / f"skip_topn_{bench}_{method}"
        fig.savefig(f"{stem}.png", dpi=150, bbox_inches="tight")
        fig.savefig(f"{stem}.pdf", bbox_inches="tight")
        plt.close(fig)

        out = {
            "bench": bench, "method": method, "run": run_dir.name,
            "n_problems": N, "n_skip": n_skip,
            "trajectory": traj,
        }
        _save_json(out, OUT_DIR / f"skip_topn_{bench}_{method}.json")

        # Movement vector
        print(f"=== {bench}/{method} skip-top-N ===")
        print(f"{'step':<6} {'Δcost':>10} {'Δacc':>8} {'$/pp lost':>12}")
        for i in range(1, min(13, len(traj))):
            prev, curr = traj[i - 1], traj[i]
            dc = curr["cost"] - prev["cost"]
            da = (curr["acc"] - prev["acc"]) * 100
            ratio = f"${-dc / da:.2f}" if da < 0 else "FREE" if da == 0 else "BAD"
            print(f"top-{i:<3} ${dc:>8.2f} {da:>+7.1f}pp  {ratio:>12}")
        return out

    # ---- atom 7 -----------------------------------------------------------
    def failure_mode_audit(
        self,
        cache_root: str | None = None,
        run_root: str | None = None,
        out_dir: str | None = None,
    ) -> dict:
        """Walk the WHOLE cache, classify each cached explore by failure mode,
        and JOIN with grade.json across run dirs to attach grading method per
        cache entry.

        This is the only atom that reads the cache substrate (not run/). It is
        bench-independent: one invocation produces a per-(bench, namespace)
        report covering every cache entry.

        Args:
            cache_root: defaults to <ANALYSIS_ROOT>/../cache  (= analysis/cache/)
            run_root:   defaults to ANALYSIS_ROOT             (= analysis/run/)
            out_dir:    defaults to OUT_DIR                   (= analysis/audit/)
        """
        cache_root_p = Path(cache_root) if cache_root else CACHE_ROOT_DEFAULT
        run_root_p = Path(run_root) if run_root else RUN_ROOT_DEFAULT
        out_dir_p = Path(out_dir) if out_dir else OUT_DIR
        assert cache_root_p.is_dir(), f"cache root not found: {cache_root_p}"
        out_dir_p.mkdir(parents=True, exist_ok=True)

        print(f"Indexing grade.json files under {run_root_p}...")
        grade_index = discover_grade_index(run_root_p)
        print(f"  Indexed {len(grade_index)} (bench, ns, qid, explore) keys.")

        records = list(walk_cache(cache_root_p, grade_index))
        assert records, f"no explore results found under {cache_root_p}"
        print(f"Scanned {len(records)} explore results.")

        agg = _aggregate_modes(records)
        grading_agg = _aggregate_grading(records)

        _write_per_explore_jsonl(records, out_dir_p / "failure_per_explore.jsonl")
        _write_case_mode_tsv(agg, out_dir_p / "failure_summary_by_case_mode.tsv")
        _write_case_mode_markdown(
            agg, grading_agg, out_dir_p / "failure_summary_by_case_mode.md"
        )

        _print_console_summary(agg)
        _print_grading_summary(grading_agg)
        print(f"\nWrote:")
        print(f"  {out_dir_p / 'failure_per_explore.jsonl'}")
        print(f"  {out_dir_p / 'failure_summary_by_case_mode.tsv'}")
        print(f"  {out_dir_p / 'failure_summary_by_case_mode.md'}")

        from collections import Counter as _Counter
        return {
            "n_explores": len(records),
            "n_cases": len(agg["cases"]),
            "mode_totals": dict(_Counter(r.mode for r in records)),
            "grading_totals": dict(_Counter(r.grading_status for r in records)),
        }

    # ---- compound: run timeout-line atoms applicable to the bench ---------
    def audit_all(self, bench: str) -> None:
        """Run timeout-line atoms 1-6 on this bench. Atom 7 (failure_mode_audit)
        is cache-wide and bench-independent, so it's NOT in here -- run it
        separately when you need it.

        Defaults:
            atoms 1/2/4: method=ATTS_V3 (where timeouts are most pronounced)
            atom 3:      method_a=ATTS_V1, method_b=ATTS_V3
            atom 5:      bench-only
            atom 6:      method=ATTS_V3 (matches the original v3 skip narrative)
        """
        print(f"\n###### audit_all bench={bench} ######\n")
        print("--- atom 1: traj_timeout_dist ---")
        self.traj_timeout_dist(bench)
        print("\n--- atom 2: acc_by_timeout ---")
        self.acc_by_timeout(bench)
        print("\n--- atom 3: cross_variant_timeout ---")
        self.cross_variant_timeout(bench)
        print("\n--- atom 4: explore_duration_dist ---")
        self.explore_duration_dist(bench)
        print("\n--- atom 5: method_pareto ---")
        self.method_pareto(bench)
        print("\n--- atom 6: skip_topn_traj ---")
        self.skip_topn_traj(bench)


if __name__ == "__main__":
    fire.Fire(Audit)
