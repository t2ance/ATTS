from __future__ import annotations

import json

import pytest

from cache_types import Exploration, JudgeOutcome


def test_label_for_with_spec():
    spec = {"backend": "claude", "model": "claude-haiku-4-5-20251001"}
    assert JudgeOutcome.label_for(spec) == "claude__claude-haiku-4-5-20251001"


def test_label_for_none_returns_none():
    assert JudgeOutcome.label_for(None) is None


def test_persist_writes_five_files(tmp_path):
    spec = {"backend": "claude", "model": "claude-haiku-4-5-20251001"}
    outcome = JudgeOutcome(
        is_correct=True,
        cost_usd=0.0012,
        judge_spec_snapshot=spec,
        input_md="judge prompt here",
        output_md="judge response here",
        result_dict={"correct": True, "explanation": "matches"},
    )
    target = tmp_path / "judges" / outcome.label
    outcome.persist(target)

    assert (target / "config.json").exists()
    assert json.loads((target / "config.json").read_text()) == spec
    assert (target / "input.md").read_text() == "judge prompt here"
    assert (target / "output.md").read_text() == "judge response here"
    assert (target / "result.json").exists()
    grade = json.loads((target / "grade.json").read_text())
    assert grade["is_correct"] is True
    assert grade["cost_usd"] == 0.0012
    assert grade["judge_spec"] == spec


def test_persist_asserts_when_rule_based(tmp_path):
    outcome = JudgeOutcome(
        is_correct=True,
        cost_usd=0.0,
        judge_spec_snapshot=None,
        input_md="",
        output_md="",
        result_dict={"correct": True},
    )
    with pytest.raises(AssertionError, match="rule-based grading"):
        outcome.persist(tmp_path / "judges" / "should_not_exist")


def test_exploration_persist_writes_three_files(tmp_path):
    exp = Exploration(
        qid="abc",
        idx=3,
        rollout_idx=None,
        answer="42",
        trajectory="reasoning here",
        cost_usd=0.05,
        model="claude-sonnet-4-6",
        timed_out=False,
        verdict=None,
    )
    exp.persist(tmp_path / "explore_3")
    assert (tmp_path / "explore_3" / "result.json").exists()
    assert (tmp_path / "explore_3" / "input.md").exists()
    assert (tmp_path / "explore_3" / "output.md").read_text() == "reasoning here"
    payload = json.loads((tmp_path / "explore_3" / "result.json").read_text())
    assert payload["answer"] == "42"
    assert payload["cost_usd"] == 0.05
    assert payload["model"] == "claude-sonnet-4-6"
    assert payload.get("timed_out") is False


def test_exploration_with_verdict_field():
    spec = {"backend": "claude", "model": "claude-haiku-4-5-20251001"}
    verdict = JudgeOutcome(
        is_correct=True,
        cost_usd=0.001,
        judge_spec_snapshot=spec,
        input_md="...",
        output_md="...",
        result_dict={"correct": True},
    )
    exp = Exploration(
        qid="q1",
        idx=1,
        rollout_idx=None,
        answer="42",
        trajectory="...",
        cost_usd=0.05,
        model="m",
        verdict=verdict,
    )
    assert exp.verdict is verdict
    assert exp.verdict.is_correct is True
