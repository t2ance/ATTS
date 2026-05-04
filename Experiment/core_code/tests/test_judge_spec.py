"""Validator tests for benchmark.judge: ModelConfig (post-modelconfig refactor).

The pre-2026-05-04 ClaudeJudgeSpec/CodexJudgeSpec/VllmJudgeSpec discriminated
union was collapsed into a single ModelConfig. CLAUDE.md non-thinking-judge
default lives in ModelConfig.effort default = "low".
"""
from __future__ import annotations

import sys
from pathlib import Path

_CORE_CODE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_CORE_CODE_DIR))

import pytest
from pydantic import BaseModel, ValidationError

from benchmarks.specs import (
    BabyVisionSpec,
    BenchmarkSpec,
    HLESpec,
    RBenchVSpec,
)
from methods.specs import ModelConfig


class _BenchHolder(BaseModel):
    benchmark: BenchmarkSpec


_CLAUDE_JUDGE = {"backend": "claude", "model": "claude-haiku-4-5-20251001"}


def test_hle_with_claude_judge():
    h = _BenchHolder.model_validate({
        "benchmark": {"name": "hle", "subset": "gold", "judge": _CLAUDE_JUDGE}
    })
    assert isinstance(h.benchmark, HLESpec)
    assert isinstance(h.benchmark.judge, ModelConfig)
    assert h.benchmark.judge.backend == "claude"
    assert h.benchmark.judge.model == "claude-haiku-4-5-20251001"
    # CLAUDE.md non-thinking-judge default is encoded as ModelConfig.effort="low"
    assert h.benchmark.judge.effort == "low"


def test_codex_judge():
    h = _BenchHolder.model_validate({
        "benchmark": {"name": "hle", "subset": "gold",
                      "judge": {"backend": "codex", "model": "gpt-5-codex-mini"}}
    })
    assert h.benchmark.judge.backend == "codex"


def test_vllm_judge_with_sampling():
    h = _BenchHolder.model_validate({
        "benchmark": {
            "name": "babyvision",
            "judge": {
                "backend": "vllm",
                "model": "qwen36-35b-a3b-fp8",
                "vllm_sampling": {"temperature": 0.6, "max_tokens": 4096},
            },
        }
    })
    assert isinstance(h.benchmark, BabyVisionSpec)
    assert h.benchmark.judge.vllm_sampling.temperature == 0.6


def test_old_field_name_sampling_rejected_on_vllm_judge():
    """Pre-refactor yamls used `sampling:` (the old VllmJudgeSpec field).

    The new ModelConfig uses `vllm_sampling:` to differentiate from any
    role-level sampling. extra=forbid rejects the old name; the migration
    script renames it before merge."""
    with pytest.raises(ValidationError):
        _BenchHolder.model_validate({
            "benchmark": {
                "name": "babyvision",
                "judge": {
                    "backend": "vllm",
                    "model": "qwen36-35b-a3b-fp8",
                    "sampling": {"temperature": 0.6},
                },
            }
        })


def test_old_field_name_name_rejected():
    """Pre-refactor yamls used `name:` instead of `backend:` for the
    discriminator. extra=forbid rejects the old name; the migration script
    renames it before merge."""
    with pytest.raises(ValidationError):
        _BenchHolder.model_validate({
            "benchmark": {
                "name": "hle", "subset": "gold",
                "judge": {"name": "claude", "model": "claude-haiku-4-5-20251001"},
            }
        })


def test_judge_vllm_sampling_rejected_on_claude_backend():
    """Per ModelConfig validator, vllm_sampling on a non-vllm backend
    fails loud (replaces the prior silent-no-op)."""
    with pytest.raises(ValidationError):
        _BenchHolder.model_validate({
            "benchmark": {
                "name": "hle", "subset": "gold",
                "judge": {
                    "backend": "claude", "model": "claude-haiku-4-5-20251001",
                    "vllm_sampling": {"temperature": 0.5},
                },
            }
        })


def test_judge_unknown_backend_rejected():
    with pytest.raises(ValidationError):
        _BenchHolder.model_validate({
            "benchmark": {
                "name": "hle", "subset": "gold",
                "judge": {"backend": "nope", "model": "x"},
            }
        })


def test_rbenchv_requires_judge():
    with pytest.raises(ValidationError):
        _BenchHolder.model_validate({"benchmark": {"name": "rbenchv"}})


def test_lcb_rejects_judge():
    with pytest.raises(ValidationError):
        _BenchHolder.model_validate({
            "benchmark": {"name": "lcb", "judge": _CLAUDE_JUDGE}
        })


def test_gpqa_rejects_judge():
    with pytest.raises(ValidationError):
        _BenchHolder.model_validate({
            "benchmark": {"name": "gpqa", "judge": _CLAUDE_JUDGE}
        })


def test_aime2025_rejects_judge():
    with pytest.raises(ValidationError):
        _BenchHolder.model_validate({
            "benchmark": {"name": "aime2025", "judge": _CLAUDE_JUDGE}
        })
