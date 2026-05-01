from __future__ import annotations

import json

import pytest

from benchmarks.base import find_cached_judge, judge_label


def test_judge_label_claude():
    spec = {"name": "claude", "model": "claude-haiku-4-5-20251001"}
    assert judge_label(spec) == "claude__claude-haiku-4-5-20251001"


def test_judge_label_codex():
    spec = {"name": "codex", "model": "gpt-5-codex-mini"}
    assert judge_label(spec) == "codex__gpt-5-codex-mini"


def test_judge_label_vllm_ignores_sampling():
    spec = {"name": "vllm", "model": "qwen36-35b-a3b-fp8",
            "sampling": {"temperature": 0.6}}
    assert judge_label(spec) == "vllm__qwen36-35b-a3b-fp8"


def test_find_cached_judge_hit(tmp_path):
    judges_dir = tmp_path / "judges"
    spec = {"name": "claude", "model": "claude-haiku-4-5-20251001"}
    bundle = judges_dir / "claude__claude-haiku-4-5-20251001"
    bundle.mkdir(parents=True)
    (bundle / "config.json").write_text(json.dumps(spec))
    found = find_cached_judge(judges_dir, spec)
    assert found == bundle


def test_find_cached_judge_miss_returns_none(tmp_path):
    judges_dir = tmp_path / "judges"
    judges_dir.mkdir()
    spec = {"name": "claude", "model": "claude-haiku-4-5-20251001"}
    assert find_cached_judge(judges_dir, spec) is None


def test_find_cached_judge_no_judges_dir_returns_none(tmp_path):
    """When judges/ does not yet exist, treat as miss; do not raise."""
    spec = {"name": "claude", "model": "x"}
    assert find_cached_judge(tmp_path / "judges", spec) is None


def test_find_cached_judge_label_collision_raises(tmp_path):
    """Two judges sharing the same backend+model label but with different
    sampling configs must surface a clear error, not silently shadow."""
    judges_dir = tmp_path / "judges"
    stored_spec = {"name": "vllm", "model": "qwen36-35b-a3b-fp8",
                   "sampling": {"temperature": 0.6}}
    requested_spec = {"name": "vllm", "model": "qwen36-35b-a3b-fp8",
                      "sampling": {"temperature": 0.0}}
    bundle = judges_dir / "vllm__qwen36-35b-a3b-fp8"
    bundle.mkdir(parents=True)
    (bundle / "config.json").write_text(json.dumps(stored_spec))
    with pytest.raises(RuntimeError, match="Judge label collision"):
        find_cached_judge(judges_dir, requested_spec)


def test_find_cached_judge_partial_bundle_treated_as_miss(tmp_path):
    """A bundle dir with no config.json (e.g. crash mid-write) is a partial
    bundle; treat as miss so the caller can re-grade and overwrite."""
    judges_dir = tmp_path / "judges"
    bundle = judges_dir / "claude__claude-haiku-4-5-20251001"
    bundle.mkdir(parents=True)
    # No config.json written.
    spec = {"name": "claude", "model": "claude-haiku-4-5-20251001"}
    assert find_cached_judge(judges_dir, spec) is None
