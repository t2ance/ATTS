from __future__ import annotations
import sys
import textwrap
from pathlib import Path

_CORE_CODE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_CORE_CODE_DIR))

import pytest
import yaml
from pydantic import ValidationError
from eval import EvalConfig, load_config


# Post-modelconfig-refactor (2026-05-04): judge is a flat ModelConfig dict.
# HLE/BabyVision/RBenchV require it; LCB/GPQA/AIME do not.
_JUDGE = {"backend": "claude", "model": "claude-haiku-4-5-20251001"}

# Reusable explore variant for self-refine etc. — single-cache, label="default".
_DEFAULT_VARIANT = {
    "label": "default",
    "model": {"backend": "claude", "model": "claude-sonnet-4-6"},
    "cache_dir": "/cache/x",
    "num_explores": 8,
}


def _minimal_kwargs(method_block=None, **overrides):
    """Build a minimum valid EvalConfig dict (defaults to self-refine)."""
    base = {
        "benchmark": {"name": "hle", "judge": _JUDGE},
        "method": method_block or {
            "name": "self-refine",
            "explore": _DEFAULT_VARIANT,
        },
    }
    base.update(overrides)
    return base


def _write(tmp_path, name, body):
    p = tmp_path / name
    p.write_text(textwrap.dedent(body))
    return p


# ---------------------------------------------------------------------------
# EvalConfig top-level shape
# ---------------------------------------------------------------------------

def test_minimal_config_validates():
    cfg = EvalConfig(**_minimal_kwargs())
    assert cfg.method.name == "self-refine"
    assert cfg.method.explore.cache_dir == Path("/cache/x")
    assert cfg.method.explore.model.backend == "claude"
    assert cfg.benchmark.name == "hle"


def test_extra_field_forbidden_top_level():
    with pytest.raises(ValidationError, match="typoed_field|extra"):
        EvalConfig(**_minimal_kwargs(typoed_field=True))


def test_top_level_has_no_method_specific_fields():
    """Method-related fields no longer live at top level."""
    for dead in ("orchestrator_model", "explore_model", "integrate_model",
                 "reward_model", "cache_dir", "cache_dirs", "model_budgets",
                 "no_integrate", "num_explores", "num_rollouts",
                 "no_cache_only", "backend", "budget_tokens", "effort",
                 "timeout", "max_output_tokens", "explore_timeout"):
        with pytest.raises(ValidationError, match=dead):
            EvalConfig(**_minimal_kwargs(**{dead: 1}))


# ---------------------------------------------------------------------------
# MethodSpec discriminator + per-method requirements
# ---------------------------------------------------------------------------

def test_tts_agent_single_happy_path():
    cfg = EvalConfig(**_minimal_kwargs(method={
        "name": "tts-agent",
        "orchestrator_prompt": "single",
        "orchestrator": {"backend": "claude", "model": "claude-sonnet-4-6"},
        "explore": [_DEFAULT_VARIANT],
    }))
    assert cfg.method.name == "tts-agent"
    assert cfg.method.orchestrator_prompt == "single"
    assert len(cfg.method.explore) == 1
    assert cfg.method.integrate is None


def test_tts_agent_single_with_integrate():
    cfg = EvalConfig(**_minimal_kwargs(method={
        "name": "tts-agent",
        "orchestrator_prompt": "single",
        "orchestrator": {"backend": "claude", "model": "claude-sonnet-4-6"},
        "explore": [_DEFAULT_VARIANT],
        "integrate": {
            "model": {"backend": "claude", "model": "claude-sonnet-4-6"},
        },
    }))
    assert cfg.method.integrate is not None
    assert cfg.method.integrate.model.model == "claude-sonnet-4-6"


def test_tts_agent_single_rejects_multi_explore():
    """orchestrator_prompt=single requires len(explore)==1."""
    with pytest.raises(ValidationError, match="single"):
        EvalConfig(**_minimal_kwargs(method={
            "name": "tts-agent",
            "orchestrator_prompt": "single",
            "orchestrator": {"backend": "claude", "model": "claude-sonnet-4-6"},
            "explore": [_DEFAULT_VARIANT, {**_DEFAULT_VARIANT, "label": "extra"}],
        }))


def test_tts_agent_multi_model_label_set_enforced():
    """orchestrator_prompt=multi_model hardcodes haiku/sonnet/opus labels."""
    bad = EvalConfig(**_minimal_kwargs(method={
        "name": "tts-agent",
        "orchestrator_prompt": "multi_model",
        "orchestrator": {"backend": "claude", "model": "claude-sonnet-4-6"},
        "explore": [
            {"label": "haiku",  "model": {"backend": "claude", "model": "claude-haiku-4-5-20251001"}, "cache_dir": "/cache/h"},
            {"label": "sonnet", "model": {"backend": "claude", "model": "claude-sonnet-4-6"},        "cache_dir": "/cache/s"},
            {"label": "opus",   "model": {"backend": "claude", "model": "claude-opus-4-6"},          "cache_dir": "/cache/o"},
        ],
    }))
    assert {v.label for v in bad.method.explore} == {"haiku", "sonnet", "opus"}

    with pytest.raises(ValidationError, match="multi_model"):
        EvalConfig(**_minimal_kwargs(method={
            "name": "tts-agent",
            "orchestrator_prompt": "multi_model",
            "orchestrator": {"backend": "claude", "model": "claude-sonnet-4-6"},
            "explore": [
                {"label": "wrong", "model": {"backend": "claude", "model": "x"}, "cache_dir": "/c"},
                {"label": "labels","model": {"backend": "claude", "model": "x"}, "cache_dir": "/c"},
                {"label": "here",  "model": {"backend": "claude", "model": "x"}, "cache_dir": "/c"},
            ],
        }))


def test_self_refine_rejects_orchestrator_model():
    """Dead field for self-refine: orchestrator_model must fail at validation."""
    with pytest.raises(ValidationError, match="orchestrator_model|extra"):
        EvalConfig(**_minimal_kwargs(method={
            "name": "self-refine",
            "explore": _DEFAULT_VARIANT,
            "orchestrator_model": "should not be here",
        }))


def test_rerank_rejects_explore():
    """Dead field for rerank: explore must fail at validation."""
    with pytest.raises(ValidationError, match="explore|extra"):
        EvalConfig(**_minimal_kwargs(method={
            "name": "rerank",
            "reward_model": "rm",
            "cache_dir": "/cache/x",
            "explore": _DEFAULT_VARIANT,
        }))


def test_rerank_requires_reward_model():
    with pytest.raises(ValidationError, match="reward_model"):
        EvalConfig(**_minimal_kwargs(method={
            "name": "rerank",
            "cache_dir": "/cache/x",
        }))


def test_modelconfig_extra_forbidden():
    """ModelConfig has extra:forbid."""
    with pytest.raises(ValidationError, match="extra"):
        EvalConfig(**_minimal_kwargs(method={
            "name": "self-refine",
            "explore": {
                "label": "default",
                "model": {"backend": "claude", "model": "m", "typoed": True},
                "cache_dir": "/cache/x",
            },
        }))


def test_modelconfig_rejects_vllm_sampling_on_non_vllm():
    """vllm_sampling is vllm-only; setting on claude must fail."""
    with pytest.raises(ValidationError, match="vllm_sampling"):
        EvalConfig(**_minimal_kwargs(method={
            "name": "self-refine",
            "explore": {
                "label": "default",
                "model": {
                    "backend": "claude",
                    "model": "m",
                    "vllm_sampling": {"temperature": 0.6},
                },
                "cache_dir": "/cache/x",
            },
        }))


def test_standalone_integrator_requires_integrate():
    with pytest.raises(ValidationError, match="integrate"):
        EvalConfig(**_minimal_kwargs(method={
            "name": "standalone-integrator",
            "num_explores": 8,
        }))


def test_unknown_method_name_rejected():
    with pytest.raises(ValidationError):
        EvalConfig(**_minimal_kwargs(method={"name": "totally-fake-method"}))


# ---------------------------------------------------------------------------
# load_config (YAML loader)
# ---------------------------------------------------------------------------

def test_load_config_yaml(tmp_path):
    yml = _write(tmp_path, "x.yaml", """
        benchmark:
          name: hle
          judge:
            backend: claude
            model: claude-haiku-4-5-20251001
        method:
          name: self-refine
          explore:
            label: default
            model:
              backend: claude
              model: claude-sonnet-4-6
            cache_dir: /cache/single
            num_explores: 8
        num: 50
    """)
    cfg = load_config(config_path=yml, schema=EvalConfig)
    assert cfg.method.explore.cache_dir == Path("/cache/single")
    assert cfg.method.explore.model.backend == "claude"
    assert cfg.num == 50


def test_load_config_tts_agent_multi_model_yaml(tmp_path):
    yml = _write(tmp_path, "x.yaml", """
        benchmark:
          name: hle
          judge:
            backend: claude
            model: claude-haiku-4-5-20251001
        method:
          name: tts-agent
          orchestrator_prompt: multi_model
          orchestrator:
            backend: claude
            model: claude-sonnet-4-6
          explore:
            - label: haiku
              model: {backend: claude, model: claude-haiku-4-5-20251001}
              cache_dir: /cache/haiku
              num_explores: 8
            - label: sonnet
              model: {backend: claude, model: claude-sonnet-4-6}
              cache_dir: /cache/sonnet
              num_explores: 8
            - label: opus
              model: {backend: claude, model: claude-opus-4-6}
              cache_dir: /cache/opus
              num_explores: 4
    """)
    cfg = load_config(config_path=yml, schema=EvalConfig)
    labels = {v.label for v in cfg.method.explore}
    assert labels == {"haiku", "sonnet", "opus"}


# ---------------------------------------------------------------------------
# Benchmark filters (preserved from old test file — schema unchanged)
# ---------------------------------------------------------------------------

def test_hle_filters_validate(tmp_path):
    yml = _write(tmp_path, "x.yaml", """
        benchmark:
          name: hle
          subset: gold
          text_only: true
          judge:
            backend: claude
            model: claude-haiku-4-5-20251001
        method:
          name: self-refine
          explore:
            label: default
            model: {backend: claude, model: claude-sonnet-4-6}
            cache_dir: /cache/x
    """)
    cfg = load_config(config_path=yml, schema=EvalConfig)
    assert cfg.benchmark.subset == "gold"
    assert cfg.benchmark.text_only is True


def test_hle_filters_reject_unknown_field(tmp_path):
    yml = _write(tmp_path, "x.yaml", """
        benchmark:
          name: hle
          domain: physics
        method:
          name: self-refine
          explore:
            label: default
            model: {backend: claude, model: claude-sonnet-4-6}
            cache_dir: /cache/x
    """)
    with pytest.raises(ValidationError, match="domain"):
        load_config(config_path=yml, schema=EvalConfig)


def test_gpqa_filters_validate(tmp_path):
    yml = _write(tmp_path, "x.yaml", """
        benchmark:
          name: gpqa
          domain: Physics
        method:
          name: self-refine
          explore:
            label: default
            model: {backend: claude, model: claude-sonnet-4-6}
            cache_dir: /cache/x
    """)
    cfg = load_config(config_path=yml, schema=EvalConfig)
    assert cfg.benchmark.domain == "Physics"


def test_filters_empty_validates_for_all_benchmarks():
    """Every concrete benchmark must accept its minimal spec.

    HLE/BabyVision/RBenchV require `judge:` per spec design;
    LCB/GPQA/AIME do not carry `judge:` (string-match or code-execution graded).
    """
    judge_required = {"hle", "babyvision", "rbenchv"}
    for name in ("hle", "lcb", "gpqa", "babyvision", "aime2025", "aime2026", "rbenchv"):
        bench: dict = {"name": name}
        if name in judge_required:
            bench["judge"] = _JUDGE
        cfg = EvalConfig(
            benchmark=bench,
            method={
                "name": "self-refine",
                "explore": {
                    "label": "default",
                    "model": {"backend": "claude", "model": "m"},
                    "cache_dir": "/cache/x",
                },
            },
        )
        assert cfg.benchmark.name == name
