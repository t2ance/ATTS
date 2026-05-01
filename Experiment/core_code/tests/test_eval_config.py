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


# HLE/BabyVision/RBenchV require a `judge:` block per JudgeSpec design (2026-05-01).
_JUDGE = {"name": "claude", "model": "claude-haiku-4-5-20251001"}


def _minimal_kwargs(method_block=None, **overrides):
    """Build a minimum valid EvalConfig dict.

    Defaults to a self-refine method block (the simplest single-cache spec
    that requires only backend + explore_model + cache_dir).
    """
    base = {
        "benchmark": {"name": "hle", "judge": _JUDGE},
        "method": method_block or {
            "name": "self-refine",
            "backend": {"name": "claude"},
            "explore_model": "claude-sonnet-4-6",
            "cache_dir": "/cache/x",
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
    assert cfg.method.cache_dir == Path("/cache/x")
    assert cfg.method.backend.name == "claude"
    assert cfg.benchmark.name == "hle"


def test_extra_field_forbidden_top_level():
    with pytest.raises(ValidationError, match="typoed_field|extra"):
        EvalConfig(**_minimal_kwargs(typoed_field=True))


def test_top_level_has_no_method_specific_fields():
    """Method-related fields no longer live at top level. Setting them
    there must fail under extra='forbid'. Includes the 6 newly-moved
    backend fields."""
    string_dead = {"orchestrator_model", "explore_model", "integrate_model",
                   "reward_model", "cache_dir", "backend"}
    for dead in ("orchestrator_model", "explore_model", "integrate_model",
                 "reward_model", "cache_dir", "cache_dirs", "model_budgets",
                 "no_integrate", "num_explores", "num_rollouts",
                 "no_cache_only", "backend", "budget_tokens", "effort",
                 "timeout", "max_output_tokens", "explore_timeout"):
        with pytest.raises(ValidationError, match=dead):
            EvalConfig(**_minimal_kwargs(**{dead: "x" if dead in string_dead else 1}))


# ---------------------------------------------------------------------------
# MethodSpec discriminator + per-method requirements
# ---------------------------------------------------------------------------

def test_tts_agent_happy_path_no_integrate():
    cfg = EvalConfig(**_minimal_kwargs(method={
        "name": "tts-agent",
        "backend": {"name": "claude"},
        "orchestrator_model": "m",
        "explore_model": "m",
        "cache_dir": "/cache/x",
        "no_integrate": True,
    }))
    assert cfg.method.name == "tts-agent"
    assert cfg.method.no_integrate is True
    assert cfg.method.integrate_model is None
    assert cfg.method.backend.name == "claude"


def test_tts_agent_requires_integrate_model_when_no_integrate_false():
    with pytest.raises(ValidationError, match="integrate_model"):
        EvalConfig(**_minimal_kwargs(method={
            "name": "tts-agent",
            "backend": {"name": "claude"},
            "orchestrator_model": "m",
            "explore_model": "m",
            "cache_dir": "/cache/x",
            "no_integrate": False,
        }))


def test_tts_agent_multi_happy_path():
    cfg = EvalConfig(**_minimal_kwargs(method={
        "name": "tts-agent-multi",
        "backend": {"name": "claude"},
        "orchestrator_model": "claude-sonnet-4-6",
        "cache_dirs": {"haiku": "/cache/haiku", "sonnet": "/cache/sonnet"},
        "model_budgets": {"haiku": 8, "sonnet": 8},
    }))
    assert cfg.method.cache_dirs["haiku"] == Path("/cache/haiku")
    assert cfg.method.model_budgets["sonnet"] == 8


def test_tts_agent_multi_missing_required_fields():
    with pytest.raises(ValidationError):
        EvalConfig(**_minimal_kwargs(method={
            "name": "tts-agent-multi",
            "backend": {"name": "claude"},
            "orchestrator_model": "m",
            # missing cache_dirs and model_budgets
        }))


def test_self_refine_rejects_orchestrator_model():
    """Dead field for self-refine: orchestrator_model must fail at validation."""
    with pytest.raises(ValidationError, match="orchestrator_model|extra"):
        EvalConfig(**_minimal_kwargs(method={
            "name": "self-refine",
            "backend": {"name": "claude"},
            "explore_model": "m",
            "cache_dir": "/cache/x",
            "orchestrator_model": "should not be here",
        }))


def test_rerank_rejects_explore_model():
    """Dead field for rerank: explore_model must fail at validation."""
    with pytest.raises(ValidationError, match="explore_model|extra"):
        EvalConfig(**_minimal_kwargs(method={
            "name": "rerank",
            "reward_model": "rm",
            "cache_dir": "/cache/x",
            "explore_model": "should not be here",
        }))


def test_rerank_requires_reward_model():
    with pytest.raises(ValidationError, match="reward_model"):
        EvalConfig(**_minimal_kwargs(method={
            "name": "rerank",
            "cache_dir": "/cache/x",
        }))


def test_rerank_rejects_backend_field():
    """Rerank doesn't have a backend; setting one in the method block must fail."""
    with pytest.raises(ValidationError, match="backend|extra"):
        EvalConfig(**_minimal_kwargs(method={
            "name": "rerank",
            "reward_model": "rm",
            "cache_dir": "/cache/x",
            "backend": {"name": "claude"},
        }))


def test_backend_config_extra_forbidden():
    """BackendConfig itself has extra:forbid."""
    with pytest.raises(ValidationError, match="extra"):
        EvalConfig(**_minimal_kwargs(method={
            "name": "self-refine",
            "backend": {"name": "claude", "typoed": True},
            "explore_model": "m",
            "cache_dir": "/cache/x",
        }))


def test_standalone_integrator_requires_integrate_model():
    with pytest.raises(ValidationError, match="integrate_model"):
        EvalConfig(**_minimal_kwargs(method={
            "name": "standalone-integrator",
            "backend": {"name": "claude"},
            "cache_dir": "/cache/x",
        }))


def test_unknown_method_name_rejected():
    with pytest.raises(ValidationError):
        EvalConfig(**_minimal_kwargs(method={"name": "totally-fake-method"}))


# ---------------------------------------------------------------------------
# load_config (YAML loader, no more dot_overrides)
# ---------------------------------------------------------------------------

def test_load_config_yaml(tmp_path):
    yml = _write(tmp_path, "x.yaml", """
        benchmark:
          name: hle
          judge:
            name: claude
            model: claude-haiku-4-5-20251001
        method:
          name: self-refine
          backend:
            name: claude
          explore_model: claude-sonnet-4-6
          cache_dir: /cache/single
        num: 50
    """)
    cfg = load_config(config_path=yml, schema=EvalConfig)
    assert cfg.method.cache_dir == Path("/cache/single")
    assert cfg.method.backend.name == "claude"
    assert cfg.num == 50


def test_load_config_multi_method_yaml(tmp_path):
    yml = _write(tmp_path, "x.yaml", """
        benchmark:
          name: hle
          judge:
            name: claude
            model: claude-haiku-4-5-20251001
        method:
          name: tts-agent-multi
          backend:
            name: claude
          orchestrator_model: claude-sonnet-4-6
          cache_dirs:
            haiku: /cache/haiku
            sonnet: /cache/sonnet
          model_budgets:
            haiku: 8
            sonnet: 8
    """)
    cfg = load_config(config_path=yml, schema=EvalConfig)
    assert cfg.method.cache_dirs["haiku"] == Path("/cache/haiku")
    assert cfg.method.model_budgets["sonnet"] == 8


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
            name: claude
            model: claude-haiku-4-5-20251001
        method:
          name: self-refine
          backend:
            name: claude
          explore_model: claude-sonnet-4-6
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
          backend:
            name: claude
          explore_model: claude-sonnet-4-6
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
          backend:
            name: claude
          explore_model: claude-sonnet-4-6
          cache_dir: /cache/x
    """)
    cfg = load_config(config_path=yml, schema=EvalConfig)
    assert cfg.benchmark.domain == "Physics"


def test_filters_empty_validates_for_all_benchmarks():
    """Every concrete benchmark must accept its minimal spec.

    HLE/BabyVision/RBenchV require `judge:` per JudgeSpec design (2026-05-01);
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
                "backend": {"name": "claude"},
                "explore_model": "m",
                "cache_dir": "/cache/x",
            },
        )
        assert cfg.benchmark.name == name
