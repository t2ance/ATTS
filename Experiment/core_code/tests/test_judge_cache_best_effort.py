"""Verify find_cached_judge unidirectional best-effort match policy +
ClaudeJudgeSpec optional `effort` / `budget_tokens` fields, and the
process-level stats counter exposed for eval.py's run-end banner.

Match policy (benchmarks/base.py:find_cached_judge) -- UNIDIRECTIONAL:
  - exact dict equality                    -> hit, exact_hits++
  - stored is a STRICT SUBSET of requested -> hit, best_effort_hits++
  - stored has any key absent from         -> RuntimeError ("spec narrowing")
    requested
  - any shared key disagrees on value      -> RuntimeError ("value conflict")
  - dir or config.json missing             -> None (real miss)
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import TypeAdapter

from benchmarks.base import (
    find_cached_judge, judge_label,
    reset_judge_cache_stats, summarize_judge_cache,
)
from benchmarks.specs import BenchmarkSpec


@pytest.fixture(autouse=True)
def _reset_stats():
    reset_judge_cache_stats()
    yield
    reset_judge_cache_stats()


def _make_judges_dir(tmp_path: Path, label: str, stored: dict) -> Path:
    judges_dir = tmp_path / "judges"
    bundle = judges_dir / label
    bundle.mkdir(parents=True)
    (bundle / "config.json").write_text(
        json.dumps(stored, indent=2, sort_keys=True), encoding="utf-8"
    )
    return judges_dir


# -- spec field tests --

def _make_hle_yaml(judge_extras: dict | None = None) -> dict:
    judge = {"name": "claude", "model": "claude-haiku-4-5-20251001"}
    if judge_extras:
        judge.update(judge_extras)
    return {"name": "hle", "subset": "gold", "text_only": True, "judge": judge}


def test_claude_judge_spec_default_no_extras():
    parsed = TypeAdapter(BenchmarkSpec).validate_python(_make_hle_yaml())
    dump = parsed.model_dump(exclude_none=True)
    # exclude_none drops the optional effort/budget_tokens that defaulted None;
    # this is what eval.py:684 uses to build judge_spec dict for cache lookup.
    assert dump["judge"] == {"name": "claude", "model": "claude-haiku-4-5-20251001"}


def test_claude_judge_spec_with_effort_low():
    parsed = TypeAdapter(BenchmarkSpec).validate_python(
        _make_hle_yaml({"effort": "low"})
    )
    dump = parsed.model_dump(exclude_none=True)
    assert dump["judge"]["effort"] == "low"
    assert "budget_tokens" not in dump["judge"]


def test_claude_judge_spec_with_budget_tokens_only():
    parsed = TypeAdapter(BenchmarkSpec).validate_python(
        _make_hle_yaml({"budget_tokens": 1024})
    )
    dump = parsed.model_dump(exclude_none=True)
    assert dump["judge"]["budget_tokens"] == 1024
    assert "effort" not in dump["judge"]


def test_claude_judge_spec_rejects_unknown_field():
    bad = _make_hle_yaml({"thinking_budget": 1024})  # not a real field
    with pytest.raises(Exception):
        TypeAdapter(BenchmarkSpec).validate_python(bad)


# -- find_cached_judge tests --

def test_exact_match_hits_and_increments_exact_counter(tmp_path: Path):
    stored = {"name": "claude", "model": "claude-haiku-4-5-20251001"}
    requested = dict(stored)
    label = judge_label(requested)
    judges_dir = _make_judges_dir(tmp_path, label, stored)

    hit = find_cached_judge(judges_dir, requested)
    assert hit is not None
    assert hit.name == label
    stats = summarize_judge_cache()
    assert stats["exact_hits"] == 1
    assert stats["best_effort_hits"] == 0
    assert stats["best_effort_extras"] == []


def test_stored_subset_of_requested_hits_with_best_effort_counter(tmp_path: Path):
    """Schema evolution: stored bundle predates a new optional field. The
    legitimate forward-compat case. Counter increments; no per-call warning."""
    stored = {"name": "claude", "model": "claude-haiku-4-5-20251001"}
    requested = {"name": "claude", "model": "claude-haiku-4-5-20251001",
                 "effort": "low"}
    label = judge_label(requested)
    judges_dir = _make_judges_dir(tmp_path, label, stored)

    hit = find_cached_judge(judges_dir, requested)
    assert hit is not None
    stats = summarize_judge_cache()
    assert stats["exact_hits"] == 0
    assert stats["best_effort_hits"] == 1
    assert stats["best_effort_extras"] == ["effort"]


def test_stored_superset_of_requested_raises_spec_narrowing(tmp_path: Path):
    """Spec narrowing: stored bundle has effort=low, requester now omits effort.
    Cached verdict was produced under a non-default config the caller no longer
    specifies — refusing silent reuse is the safety guarantee P1 enforces."""
    stored = {"name": "claude", "model": "claude-haiku-4-5-20251001",
              "effort": "low"}
    requested = {"name": "claude", "model": "claude-haiku-4-5-20251001"}
    label = judge_label(requested)
    judges_dir = _make_judges_dir(tmp_path, label, stored)

    with pytest.raises(RuntimeError, match="spec narrowing"):
        find_cached_judge(judges_dir, requested)


def test_value_conflict_on_shared_key_raises(tmp_path: Path):
    """Both sides set effort but values disagree -> hard error, no soft path."""
    stored = {"name": "claude", "model": "claude-haiku-4-5-20251001",
              "effort": "high"}
    requested = {"name": "claude", "model": "claude-haiku-4-5-20251001",
                 "effort": "low"}
    label = judge_label(requested)
    judges_dir = _make_judges_dir(tmp_path, label, stored)

    with pytest.raises(RuntimeError, match="value conflict"):
        find_cached_judge(judges_dir, requested)


def test_missing_dir_returns_none(tmp_path: Path):
    requested = {"name": "claude", "model": "claude-haiku-4-5-20251001"}
    judges_dir = tmp_path / "judges"  # never created
    assert find_cached_judge(judges_dir, requested) is None


def test_partial_bundle_returns_none(tmp_path: Path):
    """Bundle dir exists but config.json missing -> treat as miss, re-judge."""
    label = "claude__claude-haiku-4-5-20251001"
    bundle = tmp_path / "judges" / label
    bundle.mkdir(parents=True)
    # NOTE: no config.json written
    requested = {"name": "claude", "model": "claude-haiku-4-5-20251001"}
    assert find_cached_judge(tmp_path / "judges", requested) is None


def test_best_effort_extras_aggregate_across_calls(tmp_path: Path):
    """Multiple bundles each adding a different new key -> union accumulated."""
    stored = {"name": "claude", "model": "claude-haiku-4-5-20251001"}
    label = judge_label(stored)
    judges_dir = _make_judges_dir(tmp_path, label, stored)

    # First request adds "effort"; second request adds "budget_tokens" too.
    find_cached_judge(judges_dir, {**stored, "effort": "low"})
    find_cached_judge(judges_dir, {**stored, "effort": "low",
                                   "budget_tokens": 1024})
    stats = summarize_judge_cache()
    assert stats["best_effort_hits"] == 2
    assert stats["best_effort_extras"] == ["budget_tokens", "effort"]
