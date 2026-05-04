"""Round-trip tests for the unified TTSAgentSpec (single / multi-model / effort)."""
from __future__ import annotations

import sys
from pathlib import Path

_CORE_CODE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_CORE_CODE_DIR))

import pytest
import yaml
from pydantic import ValidationError, TypeAdapter

from methods.specs import MethodSpec, TTSAgentSpec


_SINGLE_NO_INTEGRATE = """
name: tts-agent
orchestrator_prompt: single
orchestrator:
  backend: openrouter
  model: x-ai/grok-4.1-fast
  effort: high
explore:
  - label: default
    model: {backend: openrouter, model: x-ai/grok-4.1-fast, effort: low}
    cache_dir: /cache/hle/grok/gold
    num_explores: 4
"""

_MULTI_MODEL = """
name: tts-agent
orchestrator_prompt: multi_model
orchestrator: {backend: claude, model: claude-sonnet-4-6}
explore:
  - label: haiku
    model: {backend: claude, model: claude-haiku-4-5-20251001}
    cache_dir: /cache/hle/haiku/gold
    num_explores: 8
  - label: sonnet
    model: {backend: claude, model: claude-sonnet-4-6}
    cache_dir: /cache/hle/sonnet/gold
    num_explores: 8
  - label: opus
    model: {backend: claude, model: claude-opus-4-6}
    cache_dir: /cache/hle/opus/gold
    num_explores: 4
"""

_EFFORT = """
name: tts-agent
orchestrator_prompt: effort
orchestrator: {backend: vllm, model: qwen36-35b-a3b-fp8}
explore:
  - label: low
    model: {backend: vllm, model: qwen36-35b-a3b-fp8, effort: low}
    cache_dir: /cache/hle/qwen_low/gold
    num_explores: 6
  - label: medium
    model: {backend: vllm, model: qwen36-35b-a3b-fp8, effort: medium}
    cache_dir: /cache/hle/qwen_medium/gold
    num_explores: 6
  - label: high
    model: {backend: vllm, model: qwen36-35b-a3b-fp8, effort: high}
    cache_dir: /cache/hle/qwen_high/gold
    num_explores: 6
"""


def test_single_no_integrate_loads():
    spec = TTSAgentSpec.model_validate(yaml.safe_load(_SINGLE_NO_INTEGRATE))
    assert spec.orchestrator_prompt == "single"
    assert len(spec.explore) == 1
    assert spec.integrate is None
    assert spec.orchestrator.effort == "high"
    assert spec.explore[0].model.effort == "low"


def test_multi_model_loads():
    spec = TTSAgentSpec.model_validate(yaml.safe_load(_MULTI_MODEL))
    assert spec.orchestrator_prompt == "multi_model"
    assert len(spec.explore) == 3
    assert {v.label for v in spec.explore} == {"haiku", "sonnet", "opus"}
    assert spec.integrate is None


def test_effort_loads():
    spec = TTSAgentSpec.model_validate(yaml.safe_load(_EFFORT))
    assert spec.orchestrator_prompt == "effort"
    assert {v.label for v in spec.explore} == {"low", "medium", "high"}


def test_single_with_three_variants_rejected():
    bad = yaml.safe_load(_SINGLE_NO_INTEGRATE)
    bad["explore"] = yaml.safe_load(_MULTI_MODEL)["explore"]
    with pytest.raises(ValidationError):
        TTSAgentSpec.model_validate(bad)


def test_multi_model_with_one_variant_rejected():
    bad = yaml.safe_load(_MULTI_MODEL)
    bad["explore"] = bad["explore"][:1]
    with pytest.raises(ValidationError):
        TTSAgentSpec.model_validate(bad)


def test_multi_model_with_wrong_labels_rejected():
    bad = yaml.safe_load(_MULTI_MODEL)
    bad["explore"][0]["label"] = "qwen"
    with pytest.raises(ValidationError):
        TTSAgentSpec.model_validate(bad)


def test_effort_with_wrong_labels_rejected():
    bad = yaml.safe_load(_EFFORT)
    bad["explore"][0]["label"] = "tiny"
    with pytest.raises(ValidationError):
        TTSAgentSpec.model_validate(bad)


def test_multi_model_with_integrate_rejected():
    bad = yaml.safe_load(_MULTI_MODEL)
    bad["integrate"] = {
        "model": {"backend": "claude", "model": "claude-sonnet-4-6"},
        "cache_dir": "/cache/integrate",
    }
    with pytest.raises(ValidationError):
        TTSAgentSpec.model_validate(bad)


def test_duplicate_labels_rejected():
    bad = yaml.safe_load(_MULTI_MODEL)
    bad["explore"][1]["label"] = "haiku"
    with pytest.raises(ValidationError):
        TTSAgentSpec.model_validate(bad)


def test_old_multi_method_name_rejected():
    """The deleted TTSAgentMultiSpec used name: tts-agent-multi.

    TTSAgentSpec.name has Literal["tts-agent"], so direct construction with
    the old name fails. We validate through TTSAgentSpec rather than through
    the union TypeAdapter to avoid pydantic forward-ref rebuild requirements
    (`from __future__ import annotations` in specs.py renders union members
    as string refs at TypeAdapter-build time)."""
    bad = yaml.safe_load(_MULTI_MODEL)
    bad["name"] = "tts-agent-multi"
    with pytest.raises(ValidationError):
        TTSAgentSpec.model_validate(bad)
