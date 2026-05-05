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
from benchmarks.base import (
    BenchmarkConfig, Candidate3, find_cached_judge, judge_label,
    summarize_judge_cache,
)
from benchmarks.specs import BenchmarkSpec
from cache_types import JudgeOutcome
from methods import MethodSpec, get_method
from methods.specs import (
    ExploreVariant, SamplingConfig, TTSAgentSpec,
    SelfRefineSpec, SocraticSelfRefineSpec, BudgetForcingSpec,
    StandaloneIntegratorSpec, RerankSpec,
)
from methods.base import InfraConfig
from multimodal_input import redact_image_for_logs
from logger import RunLogger, now_str, setup_console_logging


# ---------------------------------------------------------------------------
# Spec accessor helpers (post-modelconfig refactor 2026-05-04)
# ---------------------------------------------------------------------------

def _primary_backend(spec) -> str:
    """The 'orchestrator-side' backend identifier for a method spec.

    Used for run-banner logging and as the legacy `backend=` kwarg into
    benchmark.grade() (which ignores it; judge backend lives in judge_spec).
    Per spec type:
      TTSAgentSpec       -> spec.orchestrator.backend
      SelfRefine/Socratic/BudgetForcing -> spec.explore.model.backend
      StandaloneIntegrator -> spec.integrate.model.backend
      Rerank             -> "" (no backend dispatch; local PyTorch reward model)
    """
    if isinstance(spec, TTSAgentSpec):
        return spec.orchestrator.backend
    if isinstance(spec, (SelfRefineSpec, SocraticSelfRefineSpec, BudgetForcingSpec)):
        return spec.explore.model.backend
    if isinstance(spec, StandaloneIntegratorSpec):
        return spec.integrate.model.backend
    if isinstance(spec, RerankSpec):
        return ""
    raise AssertionError(f"unknown spec type: {type(spec).__name__}")


def _spec_cache_dir(spec):
    """Single cache_dir for cache-pre-flight + cached-explore reads.

    For multi-variant TTSAgentSpec we use the FIRST variant's cache_dir as
    the pre-flight check target; per-variant grading uses _spec_multi_cache_dirs
    instead. Other method types have one canonical cache_dir.
    """
    if isinstance(spec, TTSAgentSpec):
        return spec.explore[0].cache_dir
    if isinstance(spec, (SelfRefineSpec, SocraticSelfRefineSpec, BudgetForcingSpec)):
        return spec.explore.cache_dir
    if isinstance(spec, StandaloneIntegratorSpec):
        return spec.cache_dir
    if isinstance(spec, RerankSpec):
        return spec.cache_dir
    raise AssertionError(f"unknown spec type: {type(spec).__name__}")


def _spec_multi_cache_dirs(spec) -> dict[str, "Path"] | None:
    """For multi-variant tts-agent: {label: cache_dir} mapping.

    Replaces the pre-refactor TTSAgentMultiMethod / TTSAgentEffortMethod
    derive_evaluate_args which returned spec.cache_dirs. Length-1 single-
    variant returns None (no multi grading needed)."""
    if isinstance(spec, TTSAgentSpec) and len(spec.explore) > 1:
        return {v.label: v.cache_dir for v in spec.explore}
    return None


def _spec_variants(spec) -> list:
    """All ExploreVariant instances in a spec.

    Single-variant specs return [spec.explore]; multi-variant TTSAgentSpec
    returns spec.explore (already a list). Rerank/StandaloneIntegrator have
    no per-variant cache_dir; we synthesize a single ExploreVariant from
    spec.cache_dir at call sites that need one (eval grading walks the
    cache_dir directly via this synthesized variant).
    """
    if isinstance(spec, TTSAgentSpec):
        return list(spec.explore)
    if isinstance(spec, (SelfRefineSpec, SocraticSelfRefineSpec, BudgetForcingSpec)):
        return [spec.explore]
    if isinstance(spec, StandaloneIntegratorSpec):
        # StandaloneIntegrator reads cached explores from a single cache_dir.
        # Synthesize a read-only ExploreVariant for grading-phase iteration.
        # Model fields are placeholders -- only cache_dir is used by
        # get_all_explorations.
        return [ExploreVariant(
            label="default",
            model=spec.integrate.model,  # placeholder; not used for read-only
            cache_dir=spec.cache_dir,
            num_explores=spec.num_explores,
        )]
    if isinstance(spec, RerankSpec):
        # Same synthesis trick.
        from methods.specs import ModelConfig
        return [ExploreVariant(
            label="default",
            model=ModelConfig(backend="claude", model="placeholder"),
            cache_dir=spec.cache_dir,
            num_explores=8,
        )]
    raise AssertionError(f"unknown spec type: {type(spec).__name__}")


class _Grader:
    """Async grader closure: (answer, qid) -> JudgeOutcome.

    Captures benchmark + rows_by_id + judge_spec at construction. Used by
    ExploreVariant.get_all_explorations / get_exploration to attach verdicts
    on cache miss, and by eval.py for the integrated-answer grading step.
    """
    def __init__(self, benchmark, rows_by_id: dict[str, dict], judge_spec: dict | None):
        self.benchmark = benchmark
        self.rows_by_id = rows_by_id
        self.judge_spec = judge_spec

    async def __call__(self, answer: str, qid: str) -> JudgeOutcome:
        row = self.rows_by_id[qid]
        return await self.benchmark.grade(
            predicted=answer,
            gold=str(self.benchmark.get_answer(row)),
            question=self.benchmark.get_question(row),
            row=row,
            backend="",
        )


def _spec_num_explores(spec) -> int:
    if isinstance(spec, TTSAgentSpec):
        return sum(v.num_explores for v in spec.explore)
    if isinstance(spec, (SelfRefineSpec, SocraticSelfRefineSpec, BudgetForcingSpec)):
        return spec.explore.num_explores
    if isinstance(spec, StandaloneIntegratorSpec):
        return spec.num_explores
    if isinstance(spec, RerankSpec):
        return 8  # rerank reads up to 8 cached explores per question by convention
    raise AssertionError(f"unknown spec type: {type(spec).__name__}")


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

# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

async def evaluate(
    infra: InfraConfig,
    rows: list[dict],
    solve_fn,
    spec,                                 # MethodSpec — for logging banner + spec-derived deps
    num: int | None = None,
    num_workers: int = 1,
    resume_run_dir: str | None = None,
    log_dir: str = "logs",
    dataset_config: dict | None = None,
) -> dict:
    """Run the TTS agent on dataset rows and record results.

    `spec` is the method spec (post-modelconfig refactor); the bare
    orchestrator_model/explore_model/integrate_model/sampling kwargs are
    gone. solve_fn is partialed with spec=spec by registry.build_solve_fn,
    so all per-role dispatch flows from spec internally. We keep `spec`
    here for run-banner logging and to derive cache_dirs_multi for the
    multi-variant grading path.
    """
    backend = _primary_backend(spec)
    cache_dirs_multi = _spec_multi_cache_dirs(spec)
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
                # Post-modelconfig-refactor: per-role invocation params live
                # inside the method spec (orchestrator/explore/integrate each
                # carries its own ModelConfig). Dump the whole spec so the run
                # is reproducible from run_config.json alone.
                "method_spec": spec.model_dump(exclude_none=True, mode="json"),
                "num_explores": num_explores,
                "num_workers": num_workers,
                "cache_dir": str(cache_dir) if cache_dir else None,
                "cache_only": infra.cache_only,
                "judge_spec": benchmark.judge_spec,
                **(dataset_config or {}),
            },
        )
    infra.logger = run_logger

    logger.info(f"\n{'=' * 60}")
    logger.info(f"{benchmark.name.upper()} Evaluation")
    logger.info(f"Method: {spec.name} | Primary backend: {backend}")
    logger.info(f"Spec: {json.dumps(spec.model_dump(exclude_none=True, mode='json'))}")
    logger.info(f"Grading: {benchmark.grading_summary}")
    logger.info(f"Questions to run: {len(pending)} ({len(done_records)} already completed, {total} total)")
    logger.info(f"Max iterations per question: {num_explores} | Workers: {num_workers}")
    logger.info(f"Logs:   {run_logger.run_dir}")
    logger.info(f"{'=' * 60}\n")

    # Build grader closure and variants list (used by per-question grading
    # phase + resume-grading path). rows_by_id covers both new pending rows
    # and any rows referenced by done_records.
    rows_by_id_full = {benchmark.get_id(r): r for r in rows}
    grader = _Grader(benchmark, rows_by_id_full, benchmark.judge_spec)
    variants_list = _spec_variants(spec)

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
            primary_explorations = await variants_list[0].get_all_explorations(
                qid, rollout_idx=rec_rollout, grader=grader,
            )
            cands = [
                (benchmark.normalize_answer(e.answer),
                 e.verdict.is_correct if e.verdict else False,
                 e.cost_usd)
                for e in primary_explorations
            ]
            jc = sum((e.verdict.cost_usd if e.verdict else 0.0)
                     for e in primary_explorations)

            pm_cands = None
            if cache_dirs_multi:
                pm_cands = {}
                for v in variants_list:
                    explorations = await v.get_all_explorations(
                        qid, rollout_idx=rec_rollout, grader=grader,
                    )
                    pm_cands[v.label] = [
                        (benchmark.normalize_answer(e.answer),
                         e.verdict.is_correct if e.verdict else False,
                         e.cost_usd)
                        for e in explorations
                    ]
                    jc += sum((e.verdict.cost_usd if e.verdict else 0.0) for e in explorations)

            grade_dir = _rollout_subpath(run_logger.run_dir / "grading", qid, rec_rollout)
            final_cached = _bundle_cached(grade_dir)
            final_label = JudgeOutcome.label_for(benchmark.judge_spec)
            cached_grade_path = (grade_dir / "judges" / final_label / "grade.json"
                                 if final_label else None)
            if cached_grade_path is not None and cached_grade_path.exists():
                gd = json.loads(cached_grade_path.read_text(encoding="utf-8"))
                is_correct = gd["is_correct"]
                jc_int = 0.0
            else:
                final_outcome = await grader(predicted, qid)
                is_correct = final_outcome.is_correct
                jc_int = final_outcome.cost_usd
                if final_outcome.label is not None:
                    final_outcome.persist(grade_dir / "judges" / final_outcome.label)

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
                temperature=temperature,
                rollout_idx=rollout_idx,
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

            # Final-answer grading (integrated answer). Lives under run_dir/grading/<qid>/.
            grade_dir = _rollout_subpath(run_logger.run_dir / "grading", qid, rollout_idx)
            final_label = JudgeOutcome.label_for(benchmark.judge_spec)
            cached_grade_path = (grade_dir / "judges" / final_label / "grade.json"
                                 if final_label else None)
            if cached_grade_path is not None and cached_grade_path.exists():
                gd = json.loads(cached_grade_path.read_text(encoding="utf-8"))
                is_correct = gd["is_correct"]
                judge_cost_1 = 0.0
            else:
                final_outcome = await grader(predicted, qid)
                is_correct = final_outcome.is_correct
                judge_cost_1 = final_outcome.cost_usd
                if final_outcome.label is not None:
                    final_outcome.persist(grade_dir / "judges" / final_outcome.label)
            question_judge_cost = judge_cost_1

            # Per-explore grading via ExploreVariant.get_all_explorations.
            # Single-variant: read from variants[0]; aggregate into question_cands.
            # Multi-variant: also build per-label pm_cands.
            primary_variant = variants_list[0]
            primary_explorations = await primary_variant.get_all_explorations(
                qid, rollout_idx=rollout_idx, grader=grader,
            )
            question_cands = [
                (benchmark.normalize_answer(e.answer),
                 e.verdict.is_correct if e.verdict else False,
                 e.cost_usd)
                for e in primary_explorations
            ]
            qbon_jc = sum((e.verdict.cost_usd if e.verdict else 0.0)
                          for e in primary_explorations)

            first_explore = next(
                (r for r in (result.rounds if result else []) if r.action == "explore"), None
            )
            first_candidate_correct = (
                question_cands[0][1] if (first_explore and question_cands) else None
            )

            pm_cands = None
            if cache_dirs_multi:
                pm_cands = {}
                for v in variants_list:
                    explorations = await v.get_all_explorations(
                        qid, rollout_idx=rollout_idx, grader=grader,
                    )
                    pm_cands[v.label] = [
                        (benchmark.normalize_answer(e.answer),
                         e.verdict.is_correct if e.verdict else False,
                         e.cost_usd)
                        for e in explorations
                    ]
                    qbon_jc += sum((e.verdict.cost_usd if e.verdict else 0.0) for e in explorations)

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
    # P2: aggregated judge cache banner. Suppresses per-call WARNs that would
    # flood the log (800 explore-judge cache hits per HLE run). Only printed
    # when there were any best-effort hits, since exact-only is the steady state.
    judge_cache = summarize_judge_cache()
    if judge_cache["best_effort_hits"]:
        logger.warning(
            f"Judge cache: {judge_cache['exact_hits']} exact hits, "
            f"{judge_cache['best_effort_hits']} best-effort hits "
            f"(legacy bundles trusted under schema evolution; "
            f"new-keys-in-requested seen across run: "
            f"{judge_cache['best_effort_extras']})"
        )
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
    # exclude_none=True drops any None-valued optional fields so that a yaml
    # that does NOT specify e.g. `effort:` produces exactly the same dict as
    # before the field was added -- preserving find_cached_judge exact-match
    # against pre-existing cached config.json bundles.
    bench_dump = cfg.benchmark.model_dump(exclude_none=True)
    judge_spec = bench_dump.pop("judge", None)
    benchmark = get_benchmark(
        cfg.benchmark.name,
        judge_spec=judge_spec,
        judge_max_retries=cfg.judge_max_retries,
    )
    bench_filters = cfg.benchmark.model_dump(exclude={"name", "judge"}, exclude_defaults=True)

    method = get_method(cfg.method.name)
    solve = method.build_solve_fn(cfg.method)
    spec = cfg.method
    # Post-modelconfig-refactor: cache_dir / num_explores live inside the spec
    # at different depths depending on method shape. Centralize lookup via the
    # _spec_* helpers above so InfraConfig stays method-shape-agnostic.
    cache_dir = _spec_cache_dir(spec)
    num_explores = _spec_num_explores(spec)
    num_rollouts = getattr(spec, "num_rollouts", 1)

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

    # Slim InfraConfig: per-role backend/effort/budget/timeout/sampling/provider
    # routing all moved inside the spec's per-role ModelConfig. InfraConfig now
    # only carries cross-role context. enable_integrate is the orchestrator-tool
    # gate consumed by tts_agent.solve; for non-tts-agent specs it has no effect.
    infra = InfraConfig(
        max_iterations=num_explores,
        cache_dir=cache_dir,
        cache_only=method.cache_only,
        benchmark=benchmark,
        logger=None,
        enable_integrate=not getattr(spec, "no_integrate", False),
    )

    # Provider routing was previously injected once at startup via
    # backends.openrouter.set_provider. Post-refactor, every ModelConfig with
    # name="openrouter" carries its own openrouter_provider_order and is
    # threaded per-call into call_sub_model / run_tool_conversation. No
    # module-globals here; nothing to set.

    await evaluate(
        infra=infra,
        rows=filtered,
        solve_fn=solve,
        spec=spec,
        num=effective_num,
        num_workers=cfg.num_workers,
        resume_run_dir=cfg.resume,
        log_dir=cfg.log_dir,
        dataset_config={
            "benchmark": benchmark.name,
            **bench_filters,
            "seed": cfg.seed,
            "shuffle": cfg.shuffle,
            "num": cfg.num,
            "num_rollouts": num_rollouts,
        },
    )


def main() -> None:
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
