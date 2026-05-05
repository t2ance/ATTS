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
