"""Unified evaluation: CLI entry point + EvalConfig schema + evaluation loop.

Usage:
    python eval.py --config scripts/<bench>/<model>/<name>.yaml
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import time
from pathlib import Path
import yaml
from pydantic import BaseModel, Field, model_validator

logger = logging.getLogger(__name__)

os.environ.pop("CLAUDECODE", None)

from benchmarks import get_benchmark
from benchmarks.base import BenchmarkConfig, Candidate3, find_cached_judge, judge_label
from benchmarks.specs import BenchmarkSpec
from methods import MethodSpec, get_method
from methods.specs import SamplingConfig
from methods.base import InfraConfig
from multimodal_input import redact_image_for_logs
from logger import RunLogger, now_str, setup_console_logging


# ---------------------------------------------------------------------------
# Configuration schema (formerly eval_config.py)
# ---------------------------------------------------------------------------


class EvalConfig(BaseModel):
    model_config = {"extra": "forbid", "arbitrary_types_allowed": False}

    # Discriminated blocks: each carries its own validator
    benchmark: BenchmarkSpec
    method: MethodSpec

    # Dataset slicing (generic)
    num: int | None = None
    skip: int = 0
    seed: int = 42
    shuffle: bool = False

    num_workers: int = 1

    # Judge retry budget for LLM-based grading. Operational knob, not part of
    # judge identity (find_cached_judge cache key uses only judge_spec dict).
    # Default 3 — applies to claude/codex/vllm judges; rule-based grading
    # paths (LCB code exec, GPQA mc, AIME exact match) bypass judge_answer
    # and are unaffected. Bumping this does NOT invalidate cached bundles.
    judge_max_retries: int = 3

    # Run state
    resume: str | None = None
    log_dir: str = "logs"


def load_config(
    *,
    config_path: Path | str,
    schema: type[BaseModel] = EvalConfig,
) -> BaseModel:
    """Load YAML and validate via the given pydantic schema."""
    with open(config_path, "r") as f:
        yaml_data = yaml.safe_load(f) or {}
    assert isinstance(yaml_data, dict), (
        f"top level of {config_path} must be a mapping, got {type(yaml_data).__name__}"
    )
    return schema.model_validate(yaml_data)


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def _rollout_subpath(base: Path, qid: str, rollout_idx: int | None) -> Path:
    """Return nested path when rollout_idx is set, flat path otherwise.

    K=1 (rollout_idx=None) -> base/<qid>          (old behavior)
    K>1 (rollout_idx=k)    -> base/<qid>/rollout_<k>
    """
    if rollout_idx is None:
        return base / qid
    return base / qid / f"rollout_{rollout_idx}"


# ---------------------------------------------------------------------------
# Grading helpers
# ---------------------------------------------------------------------------

async def _grade_with_cache(
    benchmark: BenchmarkConfig,
    predicted: str, gold: str, question: str, row: dict,
    backend: str, grade_dir: Path,
) -> tuple[bool, float]:
    """Grade an answer, caching the bundle under grade_dir/judges/<label>/.

    grade_dir is the per-explore (cache_base/<qid>/explore_N/) or per-rollout
    integrate (run_dir/grading/<qid>/[/rollout_<r>]/) directory. On cache hit,
    returns the cached verdict with judge_cost=0. On miss, calls benchmark.grade
    (which writes config.json + input.md/output.md/result.json into the bundle)
    and finalizes the bundle with grade.json.

    No-judge benchmarks (LCB/GPQA/AIME -> judge_spec is None) short-circuit
    through benchmark.grade with out_dir=None and write nothing under judges/.
    """
    if benchmark.judge_spec is None:
        return await benchmark.grade(
            predicted, gold, question, row,
            backend=backend, out_dir=None,
        )

    judges_dir = grade_dir / "judges"
    cached = find_cached_judge(judges_dir, benchmark.judge_spec)
    if cached is not None:
        grade_path = cached / "grade.json"
        if grade_path.exists():
            data = json.loads(grade_path.read_text(encoding="utf-8"))
            return data["is_correct"], 0.0
        # config.json present but grade.json missing -> partial bundle, re-run.

    label = judge_label(benchmark.judge_spec)
    bundle_dir = judges_dir / label
    bundle_dir.mkdir(parents=True, exist_ok=True)
    is_correct, judge_cost = await benchmark.grade(
        predicted, gold, question, row,
        backend=backend, out_dir=bundle_dir,
    )
    logger.info(f"  [sub-model] judge: correct={is_correct}, predicted={str(predicted)[:60]}, gold={str(gold)[:60]}, cost=${judge_cost}")
    (bundle_dir / "grade.json").write_text(json.dumps({
        "judge_spec": benchmark.judge_spec,
        "is_correct": is_correct,
        "predicted": predicted,
        "gold": gold,
        "judge_cost_usd": judge_cost,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    return is_correct, judge_cost


async def _grade_cached_explores(
    benchmark: BenchmarkConfig,
    qid: str, gold_answer: str, question: str, row: dict,
    cache_base: Path,
    backend: str,
    grade_qid_dir: Path,
) -> tuple[list[Candidate3], float]:
    """Grade all cached explores for one question from a single cache directory.

    _grade_with_cache writes the bundle into cache_base/<qid>/explore_<idx>/judges/<label>/.
    This makes subsequent runs (same model + judge_spec) hit the cache for free.
    The legacy explicit-cache-hit fast-path is no longer needed; _grade_with_cache
    handles cache lookup internally via find_cached_judge.
    """
    candidates: list[Candidate3] = []
    total_jc = 0.0
    idx = 1
    while True:
        cp = cache_base / qid / f"explore_{idx}" / "result.json"
        if not cp.exists():
            break
        d = json.loads(cp.read_text(encoding="utf-8"))
        if d.get("timed_out"):
            idx += 1
            continue
        ans = benchmark.get_answer_from_explore(d)
        is_correct_exp, jc = await _grade_with_cache(
            benchmark, ans, gold_answer, question, row,
            backend=backend, grade_dir=cache_base / qid / f"explore_{idx}",
        )
        candidates.append((benchmark.normalize_answer(ans), is_correct_exp, d.get("cost_usd", 0.0)))
        total_jc += jc
        idx += 1
    return candidates, total_jc


async def _grade_question_explores(
    benchmark: BenchmarkConfig,
    qid: str, gold_answer: str, question: str, row: dict,
    rounds: list[dict],
    cache_dir: Path | None,
    backend: str,
    traj_base_dir: Path,
    rollout_idx: int | None = None,
) -> tuple[list[Candidate3], float]:
    """Grade all explores for one question. Returns (candidates, judge_cost)."""
    grade_qid_dir = _rollout_subpath(traj_base_dir / "grading", qid, rollout_idx)

    if cache_dir is not None:
        return await _grade_cached_explores(
            benchmark, qid, gold_answer, question, row,
            cache_dir, backend, grade_qid_dir,
        )

    candidates: list[Candidate3] = []
    total_jc = 0.0
    explore_seq = 0
    for r in rounds:
        if r.get("action") == "explore":
            ans = r.get("answer", "")
            explore_seq += 1
            is_correct_exp, jc = await _grade_with_cache(
                benchmark, ans, gold_answer, question, row,
                backend=backend, grade_dir=grade_qid_dir / f"explore_{explore_seq}",
            )
            candidates.append((benchmark.normalize_answer(ans), is_correct_exp, r.get("cost_usd", 0.0)))
            total_jc += jc
    return candidates, total_jc


async def _grade_question_explores_multi(
    benchmark: BenchmarkConfig,
    qid: str, gold_answer: str, question: str, row: dict,
    cache_dirs: dict[str, Path],
    backend: str,
    traj_base_dir: Path,
    rollout_idx: int | None = None,
) -> tuple[dict[str, list[Candidate3]], float]:
    """Grade all cached explores per model for one question."""
    per_model: dict[str, list[Candidate3]] = {}
    total_jc = 0.0
    grade_qid_dir = _rollout_subpath(traj_base_dir / "grading", qid, rollout_idx)
    for model_alias, cache_base in cache_dirs.items():
        cands, jc = await _grade_cached_explores(
            benchmark, qid, gold_answer, question, row,
            cache_base, backend,
            grade_qid_dir / model_alias,
        )
        per_model[model_alias] = cands
        total_jc += jc
    return per_model, total_jc


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

async def evaluate(
    infra: InfraConfig,
    rows: list[dict],
    solve_fn,
    num: int | None = None,
    num_workers: int = 1,
    resume_run_dir: str | None = None,
    log_dir: str = "logs",
    orchestrator_model: str = "gpt-5.2",
    explore_model: str = "gpt-5.2",
    integrate_model: str = "gpt-5.2",
    dataset_config: dict | None = None,
    cache_dirs_multi: dict[str, Path] | None = None,
    sampling: dict | None = None,
) -> dict:
    """Run the TTS agent on dataset rows and record results."""
    # Resume uses composite key (qid, rollout_idx) so K>1 runs with duplicate
    # qids can be resumed correctly. K=1 records persist rollout_idx as null
    # (see line ~476), so dict.get returns None, not the default 0. Coerce
    # None->0 here to match _row_key()'s `... or 0` semantics; without this,
    # done_ids holds (qid, None) while pending checks (qid, 0), and resume
    # silently skips nothing (verified 2026-04-28).
    done_ids: set[tuple[str, int]] = set()
    done_records: list[dict] = []
    if resume_run_dir is not None:
        results_path = Path(resume_run_dir) / "results.jsonl"
        assert results_path.exists(), f"No results.jsonl in {resume_run_dir}"
        with open(results_path) as f:
            for line in f:
                rec = json.loads(line)
                done_ids.add((rec["id"], rec.get("rollout_idx") or 0))
                done_records.append(rec)
        logger.info(f"Resuming {resume_run_dir}: {len(done_ids)} rollouts already completed")

    benchmark = infra.benchmark
    cache_dir = infra.cache_dir
    backend = infra.backend
    num_explores = infra.max_iterations

    def _row_key(r: dict) -> tuple[str, int]:
        return (benchmark.get_id(r), r.get("_rollout_idx", 0) or 0)

    pending = [r for r in rows if _row_key(r) not in done_ids]
    if num is not None:
        remaining = max(0, num - len(done_ids))
        pending = pending[:remaining]

    total = len(pending) + len(done_records)
    correct = 0
    first_correct = 0
    errors = 0

    if resume_run_dir is not None:
        run_logger = RunLogger.resume(resume_run_dir)
    else:
        run_logger = RunLogger.create(
            base_dir=log_dir,
            config={
                "total_questions": total,
                "backend": backend,
                "orchestrator_model": orchestrator_model,
                "explore_model": explore_model,
                "integrate_model": integrate_model,
                "num_explores": num_explores,
                "num_workers": num_workers,
                "cache_dir": str(cache_dir) if cache_dir else None,
                "cache_only": infra.cache_only,
                "judge_spec": benchmark.judge_spec,
                "budget_tokens": infra.budget_tokens,
                "effort": infra.effort,
                **(dataset_config or {}),
            },
        )
    infra.logger = run_logger

    logger.info(f"\n{'=' * 60}")
    logger.info(f"{benchmark.name.upper()} Evaluation")
    logger.info(f"Backend: {backend} | Orchestrator: {orchestrator_model} | Explorer: {explore_model} | Integrator: {integrate_model}")
    logger.info(f"Grading: {benchmark.grading_summary}")
    logger.info(f"Questions to run: {len(pending)} ({len(done_records)} already completed, {total} total)")
    logger.info(f"Max iterations per question: {num_explores} | Workers: {num_workers}")
    logger.info(f"Logs:   {run_logger.run_dir}")
    logger.info(f"{'=' * 60}\n")

    total_cost = 0.0
    total_judge_cost = 0.0
    total_cost_by_component: dict[str, float] = {}
    explore_counts: list[int] = []
    all_records: list[dict] = []

    all_candidates: list[list[Candidate3]] = [[] for _ in done_records]
    all_integrated: list[tuple[str, bool] | None] = [None for _ in done_records]
    all_subset_labels: list[str] = [rec.get("subset", "unknown") for rec in done_records]
    per_model_all_candidates: dict[str, list[list[Candidate3]]] | None = (
        {m: [[] for _ in done_records] for m in cache_dirs_multi} if cache_dirs_multi else None
    )
    bon_judge_cost = 0.0
    # For multi-model grading, use None so grading reads from rounds instead of single cache
    grade_cache_dir = None if cache_dirs_multi else cache_dir

    # Grade done records in parallel
    grade_sem = asyncio.Semaphore(num_workers)
    grade_done_count = 0
    grade_done_lock = asyncio.Lock()

    # Build a lookup from (id, rollout_idx) -> row for grading resumed records
    # rollout_idx=0 for K=1 rows so old resume behavior preserved.
    row_by_id = {(benchmark.get_id(r), r.get("_rollout_idx") or 0): r for r in rows}

    # Count total candidates to grade and how many are already cached.
    # Uses find_cached_judge against the new judges/<label>/ bundle layout.
    # Benchmarks without a judge (judge_spec is None) count as "cached" since
    # they incur zero judge cost regardless.
    def _bundle_cached(parent_dir: Path) -> bool:
        if benchmark.judge_spec is None:
            return True
        try:
            cached = find_cached_judge(parent_dir / "judges", benchmark.judge_spec)
        except RuntimeError:
            return False  # label collision; surface at grade time
        return cached is not None and (cached / "grade.json").exists()

    if done_records:
        total_to_grade = 0
        already_cached = 0
        for rec in done_records:
            qid = rec["id"]
            rec_rollout = rec.get("rollout_idx")  # None = K=1 old behavior
            grade_qid_dir = _rollout_subpath(run_logger.run_dir / "grading", qid, rec_rollout)
            # Count explores from cache_dir (shared cache; always uses real qid)
            if grade_cache_dir is not None:
                idx = 1
                seq = 0
                while (grade_cache_dir / qid / f"explore_{idx}" / "result.json").exists():
                    d = json.loads((grade_cache_dir / qid / f"explore_{idx}" / "result.json").read_text())
                    if not d.get("timed_out"):
                        seq += 1
                        total_to_grade += 1
                        if _bundle_cached(grade_qid_dir / f"explore_{seq}"):
                            already_cached += 1
                    idx += 1
            # Count integrate grade. Write path is grade_qid_dir/judges/<label>/.
            total_to_grade += 1
            if _bundle_cached(grade_qid_dir):
                already_cached += 1
        need_judge = total_to_grade - already_cached
        logger.info(f"Grading {len(done_records)} resumed records ({total_to_grade} candidates, {already_cached} cached, {need_judge} need judge)...")

    async def grade_done_record(idx: int, rec: dict) -> dict:
        nonlocal grade_done_count
        qid = rec["id"]
        rec_rollout = rec.get("rollout_idx")  # None = K=1 old record
        gold_answer = str(rec["gold_answer"])
        question_text = rec.get("question", "")
        predicted = str(rec.get("predicted_answer", ""))
        num_exp = rec.get("num_explores", 0)
        # Use the original row if available, otherwise construct a minimal one
        orig_row = row_by_id.get((qid, rec_rollout or 0), rec)

        async with grade_sem:
            cands, jc = await _grade_question_explores(
                benchmark, qid, gold_answer, question_text, orig_row,
                rec.get("rounds", []), grade_cache_dir, backend, run_logger.run_dir,
                rollout_idx=rec_rollout,
            )
            pm_cands = None
            if cache_dirs_multi:
                pm_cands, pm_jc = await _grade_question_explores_multi(
                    benchmark, qid, gold_answer, question_text, orig_row,
                    cache_dirs_multi, backend, run_logger.run_dir,
                    rollout_idx=rec_rollout,
                )
                jc += pm_jc
            grade_dir = _rollout_subpath(run_logger.run_dir / "grading", qid, rec_rollout)
            # Detect cache-hit BEFORE calling _grade_with_cache so the per-record
            # log line tells the user "previously graded" vs "freshly judged".
            # Uses the new judges/<label>/ layout via find_cached_judge.
            final_cached = _bundle_cached(grade_dir)
            is_correct, jc_int = await _grade_with_cache(
                benchmark, predicted, gold_answer, question_text, orig_row,
                backend=backend, grade_dir=grade_dir,
            )

        async with grade_done_lock:
            grade_done_count += 1
            tag = "[cached]" if final_cached else "[judged]"
            logger.info(f"  Grading resumed records: {grade_done_count}/{len(done_records)} ({qid}) {tag}")

        return {
            "idx": idx, "cands": cands, "bon_jc": jc,
            "is_correct": is_correct, "judge_cost": jc_int,
            "predicted": predicted,
            "first_candidate_correct": cands[0][1] if cands else None,
            "per_model_cands": pm_cands,
        }

    results = await asyncio.gather(*(grade_done_record(i, rec) for i, rec in enumerate(done_records))) if done_records else []

    for rec, gr in zip(done_records, results):
        if str(rec.get("predicted_answer", "")).startswith("ERROR:"):
            errors += 1
        total_cost += rec.get("cost_usd", 0.0)
        explore_counts.append(rec.get("num_explores", 0))
        for comp, comp_cost in rec.get("cost_by_component", {}).items():
            total_cost_by_component[comp] = total_cost_by_component.get(comp, 0.0) + comp_cost

        all_candidates[gr["idx"]] = gr["cands"]
        all_integrated[gr["idx"]] = (benchmark.normalize_answer(gr["predicted"]), gr["is_correct"])
        if per_model_all_candidates and gr.get("per_model_cands"):
            for model, mc in gr["per_model_cands"].items():
                per_model_all_candidates[model][gr["idx"]] = mc
        bon_judge_cost += gr["bon_jc"]
        total_judge_cost += gr["judge_cost"]
        if gr["is_correct"]:
            correct += 1
        if gr["first_candidate_correct"]:
            first_correct += 1
        all_records.append(rec)

    sem = asyncio.Semaphore(num_workers)
    lock = asyncio.Lock()

    async def process_question(i: int, row: dict) -> None:
        nonlocal correct, first_correct, errors, total_cost, total_judge_cost, bon_judge_cost

        qid = benchmark.get_id(row)
        rollout_idx = row.get("_rollout_idx", None)  # None = K=1 old path
        temperature = row.get("_temperature", None)  # None = backend default (0.0)
        question = benchmark.get_question(row)
        gold_answer = benchmark.get_answer(row)
        subset = benchmark.classify_subset(row)
        image_meta = redact_image_for_logs(row)

        qid_label = qid if rollout_idx is None else f"{qid}/rollout_{rollout_idx}"

        async with sem:
            logger.info(f"  [{qid_label}] started")

            t0 = time.time()
            image_data_url = benchmark.get_image(row)

            result = await solve_fn(
                infra=infra,
                problem=question,
                image_data_url=image_data_url,
                question_id=qid,
                orchestrator_model=orchestrator_model,
                explore_model=explore_model,
                integrate_model=integrate_model,
                temperature=temperature,
                rollout_idx=rollout_idx,
                sampling=sampling,
            )
            # Normalize at source: gold_answer is always str, str-ops downstream
            # (logger.startswith, normalize_answer, judge prompts) assume str. HLE
            # has integer-typed answers (e.g. 56) that previously slipped int into
            # the record and crashed logger.py:153 / eval.py:241/725 (2026-04-30).
            predicted = str(result.answer) if result.answer is not None else ""
            question_cost = result.cost.total_cost_usd
            round_logs = [
                {
                    "round": r.round_num,
                    "action": r.action,
                    **r.tool_input,
                }
                for r in result.rounds
            ]

            elapsed = time.time() - t0
            traj_dir = _rollout_subpath(run_logger.run_dir / "trajectories", qid, rollout_idx)
            actual_explores = sum(1 for r in (result.rounds if result else []) if r.action == "explore")
            grade_dir = _rollout_subpath(run_logger.run_dir / "grading", qid, rollout_idx)
            is_correct, judge_cost_1 = await _grade_with_cache(
                benchmark, predicted, gold_answer, question, row,
                backend=backend, grade_dir=grade_dir,
            )

            question_cands, qbon_jc = await _grade_question_explores(
                benchmark, qid, gold_answer, question, row,
                round_logs, grade_cache_dir, backend, run_logger.run_dir,
                rollout_idx=rollout_idx,
            )
            # best-of-1 = first cached explore's grade. orchestrator's run_explore
            # uses cache_key=f"explore_{call_count+1}", so question_cands[0]
            # corresponds to the orchestrator's first explore call. Reusing it
            # avoids a duplicate Haiku judge call per question (~$0.03/q ×
            # batch). Matches the RESUME-path equivalence at line 435.
            first_explore = next(
                (r for r in (result.rounds if result else []) if r.action == "explore"), None
            )
            first_candidate_correct = (
                question_cands[0][1] if (first_explore and question_cands) else None
            )
            question_judge_cost = judge_cost_1
            pm_cands = None
            if cache_dirs_multi:
                pm_cands, pm_jc = await _grade_question_explores_multi(
                    benchmark, qid, gold_answer, question, row,
                    cache_dirs_multi, backend, run_logger.run_dir,
                    rollout_idx=rollout_idx,
                )
                qbon_jc += pm_jc

            category = row.get("category", "Unknown")
            answer_type = row.get("answer_type", "exactMatch")
            record = {
                "id": qid,
                "rollout_idx": rollout_idx,  # None for K=1; resume path uses None→flat, int→rollout_N
                "temperature": temperature if temperature is not None else 0.0,
                "subset": subset,
                "category": category,
                "answer_type": answer_type,
                "question": question,
                "gold_answer": gold_answer,
                "predicted_answer": predicted,
                "is_correct": is_correct,
                "first_candidate_correct": first_candidate_correct,
                "elapsed_seconds": elapsed,
                "cost_usd": question_cost,
                "judge_cost_usd": question_judge_cost,
                "cost_by_component": result.cost.by_component if result else {},
                "num_explores": actual_explores,
                "num_rounds": len(round_logs),
                "rounds": round_logs,
                "explore_candidates": [
                    {"normalized_answer": na, "is_correct": ic, "cost_usd": c}
                    for na, ic, c in question_cands
                ],
                "exit_reason": result.exit_reason if result else "incomplete",
                **image_meta,
            }

            if result is not None:
                result.writer.write_grading(is_correct, predicted, gold_answer, elapsed, question_cost, len(round_logs))
                result.writer.close()

        async with lock:
            if is_correct:
                correct += 1
            if first_candidate_correct:
                first_correct += 1
            total_cost += question_cost
            total_judge_cost += question_judge_cost
            bon_judge_cost += qbon_jc
            explore_counts.append(actual_explores)
            if result:
                for comp, comp_cost in result.cost.by_component.items():
                    total_cost_by_component[comp] = total_cost_by_component.get(comp, 0.0) + comp_cost
            all_candidates.append(question_cands)
            all_integrated.append((benchmark.normalize_answer(predicted), is_correct))
            all_subset_labels.append(subset)
            if per_model_all_candidates and pm_cands:
                for model, mc in pm_cands.items():
                    per_model_all_candidates[model].append(mc)

            all_records.append(record)

            done_so_far = len(all_records)
            status = "correct" if is_correct else "wrong"
            predicted_short = str(predicted).replace("\n", " ")[:80]
            logger.info(f"  [{done_so_far}/{total}] {qid_label}: {status} ({elapsed:.0f}s), predicted={predicted_short}, cost=${question_cost}")

            # Running summary
            metrics = benchmark.compute_metrics(all_candidates, all_integrated, all_subset_labels,
                                                per_model_candidates=per_model_all_candidates)
            explore_dist: dict[int, int] = {}
            for ec in explore_counts:
                explore_dist[ec] = explore_dist.get(ec, 0) + 1
            running_summary = {
                "total": len(all_records),
                "correct": correct,
                "errors": errors,
                "total_cost_usd": total_cost,
                "judge_cost_usd": total_judge_cost + bon_judge_cost,
                "cost_by_component": dict(total_cost_by_component),
                "explore_distribution": {str(k): v for k, v in sorted(explore_dist.items())},
                **metrics,
            }
            run_logger.log_question(record, summary=running_summary)
            benchmark.save_plots(running_summary, run_logger.run_dir)

    await asyncio.gather(*(process_question(i, row) for i, row in enumerate(pending)))

    # Final summary
    metrics = benchmark.compute_metrics(all_candidates, all_integrated, all_subset_labels,
                                        per_model_candidates=per_model_all_candidates)
    explore_dist: dict[int, int] = {}
    for ec in explore_counts:
        explore_dist[ec] = explore_dist.get(ec, 0) + 1
    summary = {
        "total": total,
        "correct": correct,
        "errors": errors,
        "total_cost_usd": total_cost,
        "judge_cost_usd": total_judge_cost + bon_judge_cost,
        "cost_by_component": total_cost_by_component,
        "explore_distribution": {str(k): v for k, v in sorted(explore_dist.items())},
        **metrics,
    }
    run_logger.finalize(summary)
    benchmark.save_plots(summary, run_logger.run_dir)

    logger.info(f"\nEVALUATION COMPLETE")
    logger.info(f"Total:        {total}")
    if total > 0:
        logger.info(f"Integrated:   {correct}/{total} ({correct/total*100}%)")
    logger.info(f"Errors:       {errors}")
    logger.info(f"Cost breakdown:")
    for comp in sorted(total_cost_by_component):
        logger.info(f"  {comp.capitalize():14s} ${total_cost_by_component[comp]}")
    avg_str = f" (avg ${total_cost/total}/question)" if total > 0 else ""
    logger.info(f"  {'Total':14s} ${total_cost}{avg_str}")
    logger.info(f"  {'Judge':14s} ${total_judge_cost + bon_judge_cost} (not included in total)")
    if explore_dist:
        logger.info(f"Explore distribution:")
        for ec in sorted(explore_dist):
            logger.info(f"  {ec} explores: {explore_dist[ec]} questions")
    benchmark.print_metrics(summary, total)
    logger.info(f"Logs:       {run_logger.run_dir}")

    return summary


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

async def async_main() -> None:
    setup_console_logging()
    parser = argparse.ArgumentParser(description="Evaluate TTS agent")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to YAML config")
    args = parser.parse_args()
    cfg = load_config(
        config_path=args.config,
        schema=EvalConfig,
    )
    # Pull `judge` (if present) out of the benchmark spec dump and pass it
    # separately. Spec validation guarantees only HLE/BabyVision/RBenchV carry
    # `judge:`; the remaining benchmarks return judge_spec=None.
    bench_dump = cfg.benchmark.model_dump()
    judge_spec = bench_dump.pop("judge", None)
    benchmark = get_benchmark(
        cfg.benchmark.name,
        judge_spec=judge_spec,
        judge_max_retries=cfg.judge_max_retries,
    )
    bench_filters = cfg.benchmark.model_dump(exclude={"name", "judge"}, exclude_defaults=True)

    method = get_method(cfg.method.name)
    solve = method.build_solve_fn(cfg.method)
    runtime = method.derive_evaluate_args(cfg.method)
    integrate_model = runtime["integrate_model"]
    cache_dirs_multi = runtime["cache_dirs_multi"]
    cache_dir = getattr(cfg.method, "cache_dir", None)
    num_explores = getattr(cfg.method, "num_explores", 8)
    num_rollouts = getattr(cfg.method, "num_rollouts", 1)

    logger.info(f"Loading {benchmark.name.upper()} dataset...")
    all_rows = benchmark.load_dataset()
    logger.info(f"Loaded {len(all_rows)} total questions")

    filtered = benchmark.filter_dataset(all_rows, **bench_filters)
    logger.info(f"Filtered to {len(filtered)} questions")
    if not filtered:
        logger.info("No questions match the filter criteria.")
        return

    if cfg.shuffle:
        import random
        random.seed(cfg.seed)
        random.shuffle(filtered)

    if cfg.skip > 0:
        logger.info(f"Skipping first {cfg.skip} questions")
        filtered = filtered[cfg.skip:]

    # Per-method launch-time hooks: rerank/standalone filter rows by what's
    # cached; tts-agent / self-refine / socratic / budget-forcing fail-fast
    # on missing explore cache entries so the operator catches it in the
    # banner, not hours into a multi-benchmark run.
    filtered = method.filter_rows(filtered, cache_dir, benchmark)
    method.preflight(filtered, cache_dir, num_explores, cfg.num, benchmark)

    if num_rollouts > 1:
        question_rows = filtered if cfg.num is None else filtered[:cfg.num]
        expanded: list[dict] = []
        for row in question_rows:
            for k in range(num_rollouts):
                vrow = dict(row)
                vrow["_rollout_idx"] = k
                vrow["_temperature"] = 0.0 if k == 0 else 0.7
                expanded.append(vrow)
        assert len(expanded) == len(question_rows) * num_rollouts
        filtered = expanded
        effective_num = len(expanded)
        logger.info(f"Rejection sampling: expanded {len(question_rows)} questions x {num_rollouts} rollouts = {effective_num} tasks")
    else:
        effective_num = cfg.num

    # Rerank has no backend (scores cached candidates with a local reward model);
    # for it we fall through to a sentinel backend="" and default knobs --
    # rerank.solve never invokes ctx.call_sub_model so the backend module is unused.
    backend_block = getattr(cfg.method, "backend", None)
    infra = InfraConfig(
        backend=backend_block.name if backend_block else "",
        max_iterations=num_explores,
        cache_dir=cache_dir,
        cache_only=method.cache_only,
        budget_tokens=backend_block.budget_tokens if backend_block else 32000,
        effort=backend_block.effort if backend_block else "low",
        timeout=backend_block.timeout if backend_block else 1200.0,
        benchmark=benchmark,
        logger=None,
        enable_integrate=not getattr(cfg.method, "no_integrate", False),
        max_output_tokens=backend_block.max_output_tokens if backend_block else None,
    )

    sampling = getattr(cfg.method, "sampling", None)
    await evaluate(
        infra=infra,
        rows=filtered,
        solve_fn=solve,
        num=effective_num,
        num_workers=cfg.num_workers,
        resume_run_dir=cfg.resume,
        log_dir=cfg.log_dir,
        orchestrator_model=runtime["orchestrator_model"],
        explore_model=runtime["explore_model"],
        integrate_model=integrate_model,
        dataset_config={
            "benchmark": benchmark.name,
            **bench_filters,
            "seed": cfg.seed,
            "shuffle": cfg.shuffle,
            "num": cfg.num,
            "num_rollouts": num_rollouts,
        },
        cache_dirs_multi=cache_dirs_multi,
        sampling=sampling.model_dump() if sampling else None,
    )


def main() -> None:
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
