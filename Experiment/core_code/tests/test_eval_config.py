from __future__ import annotations
import sys
from pathlib import Path

_CORE_CODE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_CORE_CODE_DIR))

import pytest
from pydantic import ValidationError
from eval_config import EvalConfig


def _minimal_kwargs(**overrides):
    base = {
        "benchmark": "hle",
        "backend": "claude",
        "explore_model": "claude-sonnet-4-6",
        "method": "self-refine",
    }
    base.update(overrides)
    return base


def test_minimal_config_validates():
    cfg = EvalConfig(**_minimal_kwargs())
    assert cfg.method == "self-refine"
    assert cfg.cache_dirs == {}
    assert cfg.cache_dir is None
    assert cfg.num_rollouts == 1


def test_tts_agent_multi_requires_cache_dirs_and_budgets():
    with pytest.raises(ValidationError, match="tts-agent-multi"):
        EvalConfig(**_minimal_kwargs(method="tts-agent-multi", orchestrator_model="claude-sonnet-4-6"))


def test_tts_agent_multi_happy_path():
    cfg = EvalConfig(**_minimal_kwargs(
        method="tts-agent-multi",
        orchestrator_model="claude-sonnet-4-6",
        cache_dirs={"haiku": "/cache/haiku", "sonnet": "/cache/sonnet"},
        model_budgets={"haiku": 8, "sonnet": 8},
    ))
    assert cfg.cache_dirs["haiku"] == Path("/cache/haiku")
    assert cfg.model_budgets["sonnet"] == 8


def test_cache_dir_and_cache_dirs_mutually_exclusive():
    with pytest.raises(ValidationError, match="self-refine"):
        EvalConfig(**_minimal_kwargs(
            method="self-refine",
            cache_dirs={"x": "/cache/x"},
        ))


def test_num_rollouts_constrained_to_tts_agent_vllm():
    with pytest.raises(ValidationError, match="num_rollouts"):
        EvalConfig(**_minimal_kwargs(num_rollouts=4, method="self-refine"))
    with pytest.raises(ValidationError, match="num_rollouts"):
        EvalConfig(**_minimal_kwargs(
            num_rollouts=4, method="tts-agent",
            orchestrator_model="x", integrate_model="x", backend="claude",
        ))


def test_rerank_requires_reward_model():
    with pytest.raises(ValidationError, match="reward_model"):
        EvalConfig(**_minimal_kwargs(method="rerank"))
