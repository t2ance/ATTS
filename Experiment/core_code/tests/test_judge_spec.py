from __future__ import annotations

import pytest
from pydantic import BaseModel, ValidationError

from benchmarks.specs import (
    ClaudeJudgeSpec,
    CodexJudgeSpec,
    JudgeSpec,
    VllmJudgeSpec,
)


class _Holder(BaseModel):
    judge: JudgeSpec


def test_claude_judge_minimal():
    h = _Holder.model_validate({"judge": {"name": "claude", "model": "claude-haiku-4-5-20251001"}})
    assert isinstance(h.judge, ClaudeJudgeSpec)
    assert h.judge.model == "claude-haiku-4-5-20251001"


def test_codex_judge_minimal():
    h = _Holder.model_validate({"judge": {"name": "codex", "model": "gpt-5-codex-mini"}})
    assert isinstance(h.judge, CodexJudgeSpec)
    assert h.judge.model == "gpt-5-codex-mini"


def test_vllm_judge_with_sampling():
    h = _Holder.model_validate({
        "judge": {
            "name": "vllm",
            "model": "qwen36-35b-a3b-fp8",
            "sampling": {"temperature": 0.6, "max_tokens": 4096},
        }
    })
    assert isinstance(h.judge, VllmJudgeSpec)
    assert h.judge.model == "qwen36-35b-a3b-fp8"
    assert h.judge.sampling.temperature == 0.6


def test_vllm_judge_requires_sampling():
    with pytest.raises(ValidationError):
        _Holder.model_validate({"judge": {"name": "vllm", "model": "qwen36-35b-a3b-fp8"}})


def test_claude_judge_rejects_sampling():
    with pytest.raises(ValidationError):
        _Holder.model_validate({
            "judge": {"name": "claude", "model": "x", "sampling": {"temperature": 0.5}}
        })


def test_judge_unknown_name_rejected():
    with pytest.raises(ValidationError):
        _Holder.model_validate({"judge": {"name": "openrouter", "model": "x"}})
