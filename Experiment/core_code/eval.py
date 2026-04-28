"""Unified evaluation: CLI entry point + evaluation loop, parameterized by BenchmarkConfig.

Usage:
    python eval.py --benchmark hle --subset gold --num 10 --backend codex ...
    python eval.py --benchmark lcb --difficulty medium --backend codex ...
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from pathlib import Path

os.environ.pop("CLAUDECODE", None)

from benchmarks import get_benchmark
from benchmarks.base import BenchmarkConfig, Candidate3
from methods.base import InfraConfig
from multimodal_input import redact_image_for_logs
from logger import RunLogger


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
    quiet: bool = True,
) -> tuple[bool, float]:
    """Grade an answer, caching the result as grade.json in grade_dir."""
    judge_key = benchmark.judge_model or "none"
    grade_path = grade_dir / "grade.json"
    if grade_path.exists():
        cached = json.loads(grade_path.read_text(encoding="utf-8"))
        if cached.get("judge_model") == judge_key:
            return cached["is_correct"], cached.get("judge_cost_usd", 0.0)

    is_correct, judge_cost = await benchmark.grade(
        predicted, gold, question, row,
        backend=backend, out_dir=grade_dir / "judge",
    )
    if not quiet:
        print(f"  [sub-model] judge: correct={is_correct}, predicted={predicted[:60]}, gold={gold[:60]}, cost=${judge_cost}")
    grade_dir.mkdir(parents=True, exist_ok=True)
    grade_path.write_text(json.dumps({
        "judge_model": judge_key,
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
    quiet: bool = True,
) -> tuple[list[Candidate3], float]:
    """Grade all cached explores for one question from a single cache directory."""
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
        # Explicit cache check: reuse grade from shared cache if available
        cache_grade_path = cache_base / qid / f"explore_{idx}" / "grade.json"
        judge_key = benchmark.judge_model or "none"
        cached_grade = None
        if cache_grade_path.exists():
            cached_grade = json.loads(cache_grade_path.read_text(encoding="utf-8"))
            if cached_grade.get("judge_model") != judge_key:
                cached_grade = None
        if cached_grade is not None:
            is_correct_exp = cached_grade["is_correct"]
            jc = 0.0
        else:
            is_correct_exp, jc = await _grade_with_cache(
                benchmark, ans, gold_answer, question, row,
                backend=backend, grade_dir=grade_qid_dir / f"explore_{idx}", quiet=quiet,
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
    quiet: bool = True,
    rollout_idx: int | None = None,
) -> tuple[list[Candidate3], float]:
    """Grade all explores for one question. Returns (candidates, judge_cost)."""
    grade_qid_dir = _rollout_subpath(traj_base_dir / "grading", qid, rollout_idx)

    if cache_dir is not None:
        return await _grade_cached_explores(
            benchmark, qid, gold_answer, question, row,
            cache_dir, backend, grade_qid_dir, quiet,
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
                backend=backend, grade_dir=grade_qid_dir / f"explore_{explore_seq}", quiet=quiet,
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
    quiet: bool = True,
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
            quiet,
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
    num_rollouts: int = 1,
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
        print(f"Resuming {resume_run_dir}: {len(done_ids)} rollouts already completed")

    benchmark = infra.benchmark
    cache_dir = infra.cache_dir
    backend = infra.backend
    quiet = infra.quiet
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
        logger = RunLogger.resume(resume_run_dir)
    else:
        logger = RunLogger.create(
            base_dir=log_dir,
            config={
                "total_questions": total,
                "backend": backend,
                "orchestrator_model": orchestrator_model,
                "explore_model": explore_model,
                "integrate_model": integrate_model,
                "num_explores": num_explores,
                "num_workers": num_workers,
                "quiet": quiet,
                "cache_dir": str(cache_dir) if cache_dir else None,
                "cache_only": infra.cache_only,
                "judge_model": benchmark.judge_model,
                "budget_tokens": infra.budget_tokens,
                "effort": infra.effort,
                **(dataset_config or {}),
            },
        )
    infra.logger = logger

    print(f"\n{'=' * 60}")
    print(f"{benchmark.name.upper()} Evaluation")
    print(f"Backend: {backend} | Orchestrator: {orchestrator_model} | Explorer: {explore_model} | Integrator: {integrate_model}")
    print(f"Grading: {benchmark.grading_summary}")
    print(f"Questions to run: {len(pending)} ({len(done_records)} already completed, {total} total)")
    print(f"Max iterations per question: {num_explores} | Workers: {num_workers}")
    print(f"Logs:   {logger.run_dir}")
    print(f"{'=' * 60}\n")

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

    # Count total candidates to grade and how many are already cached
    if done_records:
        judge_key = benchmark.judge_model or "none"
        total_to_grade = 0
        already_cached = 0
        for rec in done_records:
            qid = rec["id"]
            rec_rollout = rec.get("rollout_idx")  # None = K=1 old behavior
            grade_qid_dir = _rollout_subpath(logger.run_dir / "grading", qid, rec_rollout)
            # Count explores from cache_dir (shared haiku cache; always uses real qid)
            if grade_cache_dir is not None:
                idx = 1
                seq = 0
                while (grade_cache_dir / qid / f"explore_{idx}" / "result.json").exists():
                    d = json.loads((grade_cache_dir / qid / f"explore_{idx}" / "result.json").read_text())
                    if not d.get("timed_out"):
                        seq += 1
                        total_to_grade += 1
                        gp = grade_qid_dir / f"explore_{seq}" / "grade.json"
                        if gp.exists():
                            c = json.loads(gp.read_text())
                            if c.get("judge_model") == judge_key:
                                already_cached += 1
                    idx += 1
            # Count integrate grade. The actual write path (line ~354) is
            # grade_qid_dir / "grade.json"; an earlier display-only count looked
            # under integrate_{N+1}/ which has never existed on disk, so the
            # banner reported every resumed final-answer grade as 'need judge'
            # while _grade_with_cache silently hit cache. Aligned 2026-04-28.
            total_to_grade += 1
            gp = grade_qid_dir / "grade.json"
            if gp.exists():
                c = json.loads(gp.read_text())
                if c.get("judge_model") == judge_key:
                    already_cached += 1
        need_judge = total_to_grade - already_cached
        print(f"Grading {len(done_records)} resumed records ({total_to_grade} candidates, {already_cached} cached, {need_judge} need judge)...", flush=True)

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
                rec.get("rounds", []), grade_cache_dir, backend, logger.run_dir,
                rollout_idx=rec_rollout,
            )
            pm_cands = None
            if cache_dirs_multi:
                pm_cands, pm_jc = await _grade_question_explores_multi(
                    benchmark, qid, gold_answer, question_text, orig_row,
                    cache_dirs_multi, backend, logger.run_dir,
                    rollout_idx=rec_rollout,
                )
                jc += pm_jc
            grade_dir = _rollout_subpath(logger.run_dir / "grading", qid, rec_rollout)
            is_correct, jc_int = await _grade_with_cache(
                benchmark, predicted, gold_answer, question_text, orig_row,
                backend=backend, grade_dir=grade_dir,
            )

        async with grade_done_lock:
            grade_done_count += 1
            print(f"  Grading resumed records: {grade_done_count}/{len(done_records)} ({qid})", flush=True)

        return {
            "idx": idx, "cands": cands, "bon_jc": jc,
            "is_correct": is_correct, "judge_cost": jc_int,
            "predicted": predicted,
            "first_candidate_correct": cands[0][1] if cands else None,
            "per_model_cands": pm_cands,
        }

    results = await asyncio.gather(*(grade_done_record(i, rec) for i, rec in enumerate(done_records))) if done_records else []

    for rec, gr in zip(done_records, results):
        if rec.get("predicted_answer", "").startswith("ERROR:"):
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
            print(f"  [{qid_label}] started")

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
            )
            predicted = result.answer
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
            traj_dir = _rollout_subpath(logger.run_dir / "trajectories", qid, rollout_idx)
            actual_explores = sum(1 for r in (result.rounds if result else []) if r.action == "explore")
            grade_dir = _rollout_subpath(logger.run_dir / "grading", qid, rollout_idx)
            is_correct, judge_cost_1 = await _grade_with_cache(
                benchmark, predicted, gold_answer, question, row,
                backend=backend, grade_dir=grade_dir, quiet=quiet,
            )

            first_explore = next(
                (r for r in (result.rounds if result else []) if r.action == "explore"), None
            )
            if first_explore:
                first_candidate_correct, judge_cost_2 = await _grade_with_cache(
                    benchmark, first_explore.tool_input.get("answer", ""), gold_answer, question, row,
                    backend=backend, grade_dir=traj_dir / "explore_1", quiet=quiet,
                )
            else:
                first_candidate_correct, judge_cost_2 = None, 0.0
            question_judge_cost = judge_cost_1 + judge_cost_2

            question_cands, qbon_jc = await _grade_question_explores(
                benchmark, qid, gold_answer, question, row,
                round_logs, grade_cache_dir, backend, logger.run_dir,
                quiet=quiet,
                rollout_idx=rollout_idx,
            )
            pm_cands = None
            if cache_dirs_multi:
                pm_cands, pm_jc = await _grade_question_explores_multi(
                    benchmark, qid, gold_answer, question, row,
                    cache_dirs_multi, backend, logger.run_dir,
                    quiet=quiet,
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
                "output_exceeded": result.output_exceeded if result else False,
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
            predicted_short = predicted.replace("\n", " ")[:80]
            print(f"  [{done_so_far}/{total}] {qid_label}: {status}, predicted={predicted_short}, cost=${question_cost}")

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
            logger.log_question(record, summary=running_summary)
            benchmark.save_plots(running_summary, logger.run_dir)

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
    logger.finalize(summary)
    benchmark.save_plots(summary, logger.run_dir)

    print(f"\nEVALUATION COMPLETE")
    print(f"Total:        {total}")
    if total > 0:
        print(f"Integrated:   {correct}/{total} ({correct/total*100}%)")
    print(f"Errors:       {errors}")
    print(f"Cost breakdown:")
    for comp in sorted(total_cost_by_component):
        print(f"  {comp.capitalize():14s} ${total_cost_by_component[comp]}")
    avg_str = f" (avg ${total_cost/total}/question)" if total > 0 else ""
    print(f"  {'Total':14s} ${total_cost}{avg_str}")
    print(f"  {'Judge':14s} ${total_judge_cost + bon_judge_cost} (not included in total)")
    if explore_dist:
        print(f"Explore distribution:")
        for ec in sorted(explore_dist):
            print(f"  {ec} explores: {explore_dist[ec]} questions")
    benchmark.print_metrics(summary, total)
    print(f"Logs:       {logger.run_dir}")

    return summary


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def parse_cli() -> "EvalConfig":
    """Build EvalConfig from CLI: --config FILE.yaml + flat flags + -o key=value overrides.

    Precedence: YAML < flat CLI flags < dot-path overrides. Dict-typed fields
    (cache_dirs, model_budgets, effort_budgets) are NOT exposed as flat flags;
    they must come from --config or -o.
    """
    from eval_config import load_config

    base = argparse.ArgumentParser(add_help=False)
    base.add_argument("--benchmark", type=str, required=True,
                      help="Benchmark name (hle, lcb, gpqa, babyvision, aime2025, aime2026, rbenchv)")
    base.add_argument("--config", type=str, default=None, help="Path to YAML config")
    known, _ = base.parse_known_args()
    benchmark = get_benchmark(known.benchmark)

    parser = argparse.ArgumentParser(
        description=f"Evaluate TTS agent on {benchmark.name.upper()}",
        parents=[base],
    )
    benchmark.add_dataset_args(parser)
    benchmark.add_model_args(parser)
    parser.add_argument("--verbose", action="store_true", default=None)
    parser.add_argument("--resume", type=str, default=None, metavar="RUN_DIR")
    parser.add_argument("--log-dir", type=str, default=None)
    parser.add_argument("--method", type=str, default=None,
                        choices=["tts-agent", "tts-agent-multi", "tts-agent-effort",
                                 "self-refine", "socratic-self-refine", "budget-forcing",
                                 "rerank", "standalone-integrator"])
    parser.add_argument("--reward-model", type=str, default=None)
    parser.add_argument("--orchestrator-model", type=str, default=None)
    parser.add_argument("--integrate-model", type=str, default=None)
    parser.add_argument("--no-cache-only", action="store_true", default=None)
    parser.add_argument("--timeout", type=float, default=None)
    parser.add_argument("--no-integrate", action="store_true", default=None)
    parser.add_argument("--exploration-effort", type=str, default=None,
                        choices=["low", "medium", "high"])
    parser.add_argument("--num-rollouts", type=int, default=None)
    parser.add_argument("-o", "--override", action="append", default=[],
                        help="Dot-path override, e.g. -o model_budgets.haiku=2")

    args = parser.parse_args()

    # Strip None values (so they don't override YAML defaults), and route filter
    # keys + the legacy --cache-dirs flag into per-key dot-path overrides so each
    # CLI value merges with (rather than replaces) the YAML's filters / dict.
    flat: dict[str, object] = {}
    extra_dot: list[str] = []
    for key, val in vars(args).items():
        if key in ("config", "override"):
            continue
        if val is None:
            continue
        if key == "cache_dirs":
            # Legacy CLI flag --cache-dirs is a single-path string; route to
            # cfg.cache_dir (Path | None). Multi-model dict-form must come
            # from YAML or -o cache_dirs.<key>=<val>; parse_cli does not
            # accept the legacy "key:val,key:val" string form.
            assert ":" not in val, (
                f"--cache-dirs accepts only a single path; multi-model cache "
                f"dicts must come from --config FILE.yaml. Got: {val!r}"
            )
            flat["cache_dir"] = val
            continue
        if key in benchmark.filter_keys:
            # Use dot-path routing so each CLI filter flag is merged into the
            # YAML's filters dict instead of replacing it wholesale. Avoids
            # silently erasing YAML-set filter siblings when only some are
            # overridden on the CLI.
            extra_dot.append(f"filters.{key}={val}")
        else:
            flat[key] = val

    return load_config(
        config_path=args.config,
        flat_overrides=flat,
        dot_overrides=list(args.override) + extra_dot,
    )


async def async_main() -> None:
    cfg = parse_cli()
    benchmark = get_benchmark(cfg.benchmark)

    if cfg.method == "self-refine":
        from methods.self_refine import solve
    elif cfg.method == "socratic-self-refine":
        from methods.socratic_self_refine import solve
    elif cfg.method == "budget-forcing":
        from methods.budget_forcing import solve
    elif cfg.method == "rerank":
        import functools
        from methods.reward_rerank import solve as _rerank_solve
        solve = functools.partial(_rerank_solve, reward_model_name=cfg.reward_model)
    elif cfg.method == "standalone-integrator":
        import functools
        from methods.standalone_integrator import solve as _si_solve
        solve = functools.partial(_si_solve, integrate_model=cfg.integrate_model)
    elif cfg.method == "tts-agent-multi":
        import functools
        from methods.tts_agent_multi import solve as _multi_solve
        solve = functools.partial(
            _multi_solve,
            cache_dirs=cfg.cache_dirs,
            model_budgets=cfg.model_budgets,
            exploration_effort=cfg.exploration_effort,
        )
    elif cfg.method == "tts-agent-effort":
        import functools
        from methods.tts_agent_effort import solve as _effort_solve
        solve = functools.partial(
            _effort_solve,
            cache_dirs=cfg.cache_dirs,
            effort_budgets=cfg.effort_budgets,
        )
    else:
        from methods.tts_agent import solve

    # tts-agent-multi and tts-agent-effort reuse orchestrator_model as integrate_model
    # downstream (mirrors the pre-migration in-place mutation of args.integrate_model
    # in the tts-agent-multi and tts-agent-effort branches).
    if cfg.method in ("tts-agent-multi", "tts-agent-effort"):
        integrate_model = cfg.orchestrator_model
        cache_dirs_multi = cfg.cache_dirs
    else:
        integrate_model = cfg.integrate_model
        cache_dirs_multi = None

    print(f"Loading {benchmark.name.upper()} dataset...")
    all_rows = benchmark.load_dataset()
    print(f"Loaded {len(all_rows)} total questions")

    filtered = benchmark.filter_dataset(all_rows, **cfg.filters)
    print(f"Filtered to {len(filtered)} questions")
    if not filtered:
        print("No questions match the filter criteria.")
        return

    if cfg.shuffle:
        import random
        random.seed(cfg.seed)
        random.shuffle(filtered)

    if cfg.skip > 0:
        print(f"Skipping first {cfg.skip} questions")
        filtered = filtered[cfg.skip:]

    # Replaces the legacy `":" in args.cache_dirs` format-detection block.
    # cfg.cache_dir is None for multi/effort methods (validator enforces);
    # single-cache methods set it via YAML or the legacy --cache-dirs CLI flag
    # (parse_cli routes to cfg.cache_dir).
    cache_dir = cfg.cache_dir

    if cfg.method in ("rerank", "standalone-integrator") and cache_dir:
        cached_ids = {p.name for p in cache_dir.iterdir() if p.is_dir() and (p / "explore_1" / "result.json").exists()}
        before = len(filtered)
        filtered = [r for r in filtered if benchmark.get_id(r) in cached_ids]
        print(f"{cfg.method}: {len(filtered)} questions with cache (from {before})")

    if cfg.num_rollouts > 1:
        question_rows = filtered if cfg.num is None else filtered[:cfg.num]
        expanded: list[dict] = []
        for row in question_rows:
            for k in range(cfg.num_rollouts):
                vrow = dict(row)
                vrow["_rollout_idx"] = k
                vrow["_temperature"] = 0.0 if k == 0 else 0.7
                expanded.append(vrow)
        assert len(expanded) == len(question_rows) * cfg.num_rollouts
        filtered = expanded
        effective_num = len(expanded)
        print(f"Rejection sampling: expanded {len(question_rows)} questions x {cfg.num_rollouts} rollouts = {effective_num} tasks")
    else:
        effective_num = cfg.num

    infra = InfraConfig(
        backend=cfg.backend,
        max_iterations=cfg.num_explores,
        cache_dir=cache_dir,
        cache_only=not cfg.no_cache_only,
        budget_tokens=cfg.budget_tokens,
        effort=cfg.effort,
        timeout=cfg.timeout,
        benchmark=benchmark,
        quiet=not cfg.verbose,
        logger=None,
        enable_integrate=not cfg.no_integrate,
        max_output_chars=cfg.max_output_chars,
    )

    await evaluate(
        infra=infra,
        rows=filtered,
        solve_fn=solve,
        num=effective_num,
        num_workers=cfg.num_workers,
        resume_run_dir=cfg.resume,
        log_dir=cfg.log_dir,
        orchestrator_model=cfg.orchestrator_model,
        explore_model=cfg.explore_model,
        integrate_model=integrate_model,
        dataset_config={
            "benchmark": benchmark.name,
            **cfg.filters,
            "seed": cfg.seed,
            "shuffle": cfg.shuffle,
            "num": cfg.num,
            "num_rollouts": cfg.num_rollouts,
        },
        cache_dirs_multi=cache_dirs_multi,
        num_rollouts=cfg.num_rollouts,
    )


def main() -> None:
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
