"""Unit tests for ExploreVariant cache-I/O helpers and intent-driven methods.

Covers Task 5 (path construction + cache load), Task 6 (get_exploration),
Task 7 (get_all_explorations) of the explore-cache-owner-refactor plan.
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

_CORE_CODE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_CORE_CODE_DIR))

from cache_types import Exploration, JudgeOutcome
from methods.specs import ExploreVariant, ModelConfig


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


def _make_variant(tmp_path: Path) -> ExploreVariant:
    return ExploreVariant(
        label="default",
        model=ModelConfig(backend="claude", model="claude-sonnet-4-6"),
        cache_dir=tmp_path,
        num_explores=8,
    )


# ---- Task 5: path helpers ----------------------------------------------------


def test_explore_dir_no_rollout(tmp_path):
    v = _make_variant(tmp_path)
    assert v._explore_dir("q1", 3) == tmp_path / "q1" / "explore_3"


def test_explore_dir_with_rollout(tmp_path):
    v = _make_variant(tmp_path)
    assert v._explore_dir("q1", 3, rollout_idx=2) == tmp_path / "q1" / "rollout_2" / "explore_3"


def test_judge_dir(tmp_path):
    v = _make_variant(tmp_path)
    label = "claude__claude-haiku-4-5-20251001"
    assert v._judge_dir("q1", 3, label) == tmp_path / "q1" / "explore_3" / "judges" / label


def test_load_explore_returns_none_on_miss(tmp_path):
    v = _make_variant(tmp_path)
    assert v._load_explore("q1", 1) is None


def test_load_explore_reads_persisted(tmp_path):
    v = _make_variant(tmp_path)
    exp = Exploration(
        qid="q1", idx=1, rollout_idx=None,
        answer="42", trajectory="t", cost_usd=0.05, model="m",
    )
    exp.persist(v._explore_dir("q1", 1))
    loaded = v._load_explore("q1", 1)
    assert loaded is not None
    assert loaded.answer == "42"
    assert loaded.cost_usd == 0.05


def test_load_judge_returns_none_on_miss(tmp_path):
    v = _make_variant(tmp_path)
    spec = {"backend": "claude", "model": "claude-haiku-4-5-20251001"}
    assert v._load_judge("q1", 1, spec) is None


def test_load_judge_returns_none_for_rule_based(tmp_path):
    v = _make_variant(tmp_path)
    assert v._load_judge("q1", 1, judge_spec=None) is None


def test_load_judge_exact_match(tmp_path):
    v = _make_variant(tmp_path)
    spec = {"backend": "claude", "model": "claude-haiku-4-5-20251001"}
    outcome = JudgeOutcome(
        is_correct=True, cost_usd=0.001, judge_spec_snapshot=spec,
        input_md="i", output_md="o", result_dict={"correct": True},
    )
    outcome.persist(v._judge_dir("q1", 1, outcome.label))
    loaded = v._load_judge("q1", 1, spec)
    assert loaded is not None
    assert loaded.is_correct is True
    assert loaded.cost_usd == 0.001


def test_load_judge_subset_stored_is_best_effort_hit(tmp_path):
    """Stored is strict subset of requested: legitimate schema evolution case."""
    from cache_types import _JUDGE_CACHE_STATS, reset_judge_cache_stats
    reset_judge_cache_stats()
    v = _make_variant(tmp_path)
    stored = {"backend": "claude", "model": "claude-haiku-4-5-20251001"}
    requested = {"backend": "claude", "model": "claude-haiku-4-5-20251001",
                 "effort": "low"}
    outcome = JudgeOutcome(
        is_correct=True, cost_usd=0.001, judge_spec_snapshot=stored,
        input_md="i", output_md="o", result_dict={"correct": True},
    )
    outcome.persist(v._judge_dir("q1", 1, outcome.label))
    loaded = v._load_judge("q1", 1, requested)
    assert loaded is not None
    assert _JUDGE_CACHE_STATS["best_effort_hits"] == 1


def test_load_judge_conflict_raises(tmp_path):
    v = _make_variant(tmp_path)
    stored = {"backend": "claude", "model": "claude-haiku-4-5-20251001",
              "effort": "low"}
    requested = {"backend": "claude", "model": "claude-haiku-4-5-20251001",
                 "effort": "high"}
    outcome = JudgeOutcome(
        is_correct=True, cost_usd=0.001, judge_spec_snapshot=stored,
        input_md="i", output_md="o", result_dict={"correct": True},
    )
    outcome.persist(v._judge_dir("q1", 1, outcome.label))
    with pytest.raises(RuntimeError, match="conflict"):
        v._load_judge("q1", 1, requested)


# ---- Task 6: get_exploration -------------------------------------------------


def test_get_exploration_cache_hit_does_not_call_generate(tmp_path):
    v = _make_variant(tmp_path)
    cached = Exploration(
        qid="q1", idx=1, rollout_idx=None,
        answer="cached", trajectory="t", cost_usd=0.0, model="m",
    )
    cached.persist(v._explore_dir("q1", 1))

    called = False
    async def gen():
        nonlocal called
        called = True
        raise AssertionError("generate_fn should not be called on cache hit")

    exp = _run(v.get_exploration("q1", 1, generate_fn=gen))
    assert exp.answer == "cached"
    assert called is False


def test_get_exploration_cache_miss_calls_generate_and_persists(tmp_path):
    v = _make_variant(tmp_path)

    async def gen():
        return Exploration(
            qid="q1", idx=1, rollout_idx=None,
            answer="fresh", trajectory="generated", cost_usd=0.05, model="m",
        )

    exp = _run(v.get_exploration("q1", 1, generate_fn=gen))
    assert exp.answer == "fresh"
    assert (v._explore_dir("q1", 1) / "result.json").exists()


def test_get_exploration_with_grader_attaches_verdict(tmp_path):
    class FakeGrader:
        judge_spec = {"backend": "claude", "model": "claude-haiku-4-5-20251001"}
        async def __call__(self, answer, qid):
            return JudgeOutcome(
                is_correct=True, cost_usd=0.001,
                judge_spec_snapshot=self.judge_spec,
                input_md="i", output_md="o", result_dict={"correct": True},
            )

    v = _make_variant(tmp_path)
    async def gen():
        return Exploration(qid="q1", idx=1, rollout_idx=None,
                           answer="42", trajectory="t", cost_usd=0.0, model="m")

    exp = _run(v.get_exploration("q1", 1, generate_fn=gen, grader=FakeGrader()))
    assert exp.verdict is not None
    assert exp.verdict.is_correct is True
    assert (v._judge_dir("q1", 1, exp.verdict.label) / "grade.json").exists()


def test_get_exploration_with_rule_based_grader_skips_judge_persist(tmp_path):
    class RuleBasedGrader:
        judge_spec = None
        async def __call__(self, answer, qid):
            return JudgeOutcome(is_correct=True, cost_usd=0.0,
                                judge_spec_snapshot=None, input_md="", output_md="",
                                result_dict={"correct": True})

    v = _make_variant(tmp_path)
    async def gen():
        return Exploration(qid="q1", idx=1, rollout_idx=None,
                           answer="A", trajectory="", cost_usd=0.0, model="m")

    exp = _run(v.get_exploration("q1", 1, generate_fn=gen, grader=RuleBasedGrader()))
    assert exp.verdict is not None
    assert exp.verdict.is_correct is True
    assert not (v._explore_dir("q1", 1) / "judges").exists()


# ---- Task 7: get_all_explorations -------------------------------------------


def test_get_all_explorations_empty(tmp_path):
    v = _make_variant(tmp_path)
    explorations = _run(v.get_all_explorations("nonexistent_qid"))
    assert explorations == []


def test_get_all_explorations_sorted(tmp_path):
    v = _make_variant(tmp_path)
    for i in [3, 1, 2]:
        Exploration(qid="q1", idx=i, rollout_idx=None,
                    answer=f"ans{i}", trajectory="", cost_usd=0.0, model="m"
                   ).persist(v._explore_dir("q1", i))
    explorations = _run(v.get_all_explorations("q1"))
    assert [e.idx for e in explorations] == [1, 2, 3]
    assert [e.answer for e in explorations] == ["ans1", "ans2", "ans3"]


def test_get_all_explorations_with_grader_attaches_verdicts(tmp_path):
    class FakeGrader:
        judge_spec = {"backend": "claude", "model": "claude-haiku-4-5-20251001"}
        async def __call__(self, answer, qid):
            return JudgeOutcome(is_correct=(answer == "ans1"), cost_usd=0.001,
                                judge_spec_snapshot=self.judge_spec,
                                input_md="i", output_md="o",
                                result_dict={"correct": True})

    v = _make_variant(tmp_path)
    for i in [1, 2]:
        Exploration(qid="q1", idx=i, rollout_idx=None,
                    answer=f"ans{i}", trajectory="", cost_usd=0.0, model="m"
                   ).persist(v._explore_dir("q1", i))

    explorations = _run(v.get_all_explorations("q1", grader=FakeGrader()))
    assert len(explorations) == 2
    assert explorations[0].verdict.is_correct is True
    assert explorations[1].verdict.is_correct is False


def test_get_exploration_isolates_rollouts(tmp_path):
    v = _make_variant(tmp_path)

    async def gen0():
        return Exploration(qid="q1", idx=1, rollout_idx=0,
                           answer="rollout0_ans", trajectory="", cost_usd=0.0, model="m")

    async def gen1():
        return Exploration(qid="q1", idx=1, rollout_idx=1,
                           answer="rollout1_ans", trajectory="", cost_usd=0.0, model="m")

    exp0 = _run(v.get_exploration("q1", 1, rollout_idx=0, generate_fn=gen0))
    exp1 = _run(v.get_exploration("q1", 1, rollout_idx=1, generate_fn=gen1))
    assert exp0.answer == "rollout0_ans"
    assert exp1.answer == "rollout1_ans"
    assert (tmp_path / "q1" / "rollout_0" / "explore_1" / "result.json").exists()
    assert (tmp_path / "q1" / "rollout_1" / "explore_1" / "result.json").exists()


def test_load_judge_narrowing_raises(tmp_path):
    """Stored has key absent from requested: refuse to inherit verdict."""
    v = _make_variant(tmp_path)
    stored = {"backend": "claude", "model": "claude-haiku-4-5-20251001",
              "effort": "low"}
    requested = {"backend": "claude", "model": "claude-haiku-4-5-20251001"}
    outcome = JudgeOutcome(
        is_correct=True, cost_usd=0.001, judge_spec_snapshot=stored,
        input_md="i", output_md="o", result_dict={"correct": True},
    )
    outcome.persist(v._judge_dir("q1", 1, outcome.label))
    with pytest.raises(RuntimeError, match="narrowing"):
        v._load_judge("q1", 1, requested)
