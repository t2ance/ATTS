"""Validator tests for ModelConfig (per-role backend invocation config)."""
from __future__ import annotations

import sys
from pathlib import Path

_CORE_CODE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_CORE_CODE_DIR))

import pytest
from pydantic import ValidationError

from methods.specs import ModelConfig, SamplingConfig


def test_minimal_claude_config_loads():
    cfg = ModelConfig(backend="claude", model="claude-sonnet-4-6")
    assert cfg.budget_tokens == 32000
    assert cfg.effort == "low"
    assert cfg.timeout == 1200.0
    assert cfg.max_output_tokens is None
    assert cfg.vllm_sampling is None
    assert cfg.openrouter_provider_order is None
    assert cfg.openrouter_provider_allow_fallbacks is True


def test_vllm_config_with_sampling():
    cfg = ModelConfig(
        backend="vllm",
        model="qwen36-35b-a3b-fp8",
        vllm_sampling=SamplingConfig(temperature=0.7),
    )
    assert cfg.vllm_sampling.temperature == 0.7


def test_vllm_sampling_rejected_on_non_vllm_backend():
    with pytest.raises(ValidationError) as exc:
        ModelConfig(
            backend="claude",
            model="claude-sonnet-4-6",
            vllm_sampling=SamplingConfig(temperature=0.7),
        )
    assert "vllm_sampling is vllm-only" in str(exc.value)


def test_openrouter_provider_order_rejected_on_non_openrouter_backend():
    with pytest.raises(ValidationError) as exc:
        ModelConfig(
            backend="claude",
            model="claude-sonnet-4-6",
            openrouter_provider_order=["Parasail"],
        )
    assert "openrouter_provider_order is openrouter-only" in str(exc.value)


def test_openrouter_allow_fallbacks_default_true_rejected_when_overridden_on_other_backend():
    with pytest.raises(ValidationError) as exc:
        ModelConfig(
            backend="claude",
            model="claude-sonnet-4-6",
            openrouter_provider_allow_fallbacks=False,
        )
    assert "openrouter_provider_allow_fallbacks is openrouter-only" in str(exc.value)


def test_unknown_backend_rejected():
    with pytest.raises(ValidationError):
        ModelConfig(backend="nope", model="x")


def test_extra_field_rejected():
    with pytest.raises(ValidationError):
        ModelConfig(backend="claude", model="claude-sonnet-4-6", typo_field=1)
