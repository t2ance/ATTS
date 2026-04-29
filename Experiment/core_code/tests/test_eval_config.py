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


def test_extra_field_forbidden():
    with pytest.raises(ValidationError, match="typoed_field|extra"):
        EvalConfig(**_minimal_kwargs(typoed_field=True))


import textwrap
import yaml
from eval_config import load_config, _set_dotpath, EvalConfig


def _write(tmp_path, name, body):
    p = tmp_path / name
    p.write_text(textwrap.dedent(body))
    return p


def test_load_config_yaml_only(tmp_path):
    yml = _write(tmp_path, "x.yaml", """
        benchmark: hle
        backend: claude
        explore_model: claude-sonnet-4-6
        method: self-refine
        cache_dir: /cache/single
        num: 50
    """)
    cfg = load_config(config_path=yml, dot_overrides=[], schema=EvalConfig)
    assert cfg.cache_dir == Path("/cache/single")
    assert cfg.num == 50


def test_dot_overrides_beat_yaml(tmp_path):
    yml = _write(tmp_path, "x.yaml", """
        benchmark: hle
        backend: claude
        explore_model: claude-sonnet-4-6
        method: self-refine
        seed: 13
    """)
    cfg = load_config(config_path=yml, dot_overrides=["seed=7"], schema=EvalConfig)
    assert cfg.seed == 7


def test_dot_overrides_beat_flat(tmp_path):
    yml = _write(tmp_path, "x.yaml", """
        benchmark: hle
        backend: claude
        explore_model: claude-sonnet-4-6
        method: self-refine
    """)
    cfg = load_config(
        config_path=yml,
        dot_overrides=["seed=7", "seed=99"],
        schema=EvalConfig,
    )
    assert cfg.seed == 99


def test_dot_override_dict_field(tmp_path):
    yml = _write(tmp_path, "x.yaml", """
        benchmark: hle
        backend: claude
        explore_model: claude-sonnet-4-6
        method: tts-agent-multi
        orchestrator_model: claude-sonnet-4-6
        cache_dirs:
          haiku: /cache/haiku
          sonnet: /cache/sonnet
        model_budgets:
          haiku: 8
          sonnet: 8
    """)
    cfg = load_config(
        config_path=yml,
        dot_overrides=["model_budgets.haiku=2"],
        schema=EvalConfig,
    )
    assert cfg.model_budgets == {"haiku": 2, "sonnet": 8}


def test_set_dotpath_creates_nested():
    d = {}
    _set_dotpath(d, "a.b.c", "1")
    assert d == {"a": {"b": {"c": 1}}}


def test_set_dotpath_coerces_int_float_bool():
    d = {}
    _set_dotpath(d, "x", "42")
    _set_dotpath(d, "y", "3.5")
    _set_dotpath(d, "z", "true")
    _set_dotpath(d, "s", "hello")
    assert d == {"x": 42, "y": 3.5, "z": True, "s": "hello"}


def test_load_without_yaml(tmp_path):
    cfg = load_config(
        config_path=None,
        dot_overrides=[
            "benchmark=hle", "backend=claude",
            "explore_model=claude-sonnet-4-6", "method=self-refine",
        ],
        schema=EvalConfig,
    )
    assert cfg.benchmark == "hle"


def test_set_dotpath_rejects_non_dict_intermediate():
    d = {"existing": "scalar_value"}
    with pytest.raises(AssertionError, match="cannot descend into non-dict"):
        _set_dotpath(d, "existing.subkey", "5")


def test_hle_filters_validate(tmp_path):
    yml = _write(tmp_path, "x.yaml", """
        benchmark: hle
        backend: claude
        explore_model: claude-sonnet-4-6
        method: self-refine
        filters:
          subset: gold
          text_only: true
    """)
    cfg = load_config(config_path=yml, dot_overrides=[], schema=EvalConfig)
    assert cfg.filters["subset"] == "gold"
    assert cfg.filters["text_only"] is True


def test_hle_filters_reject_unknown_field(tmp_path):
    yml = _write(tmp_path, "x.yaml", """
        benchmark: hle
        backend: claude
        explore_model: claude-sonnet-4-6
        method: self-refine
        filters:
          domain: physics
    """)
    with pytest.raises(ValidationError, match="domain"):
        load_config(config_path=yml, dot_overrides=[], schema=EvalConfig)


def test_gpqa_filters_validate(tmp_path):
    yml = _write(tmp_path, "x.yaml", """
        benchmark: gpqa
        backend: claude
        explore_model: claude-sonnet-4-6
        method: self-refine
        filters:
          domain: Physics
    """)
    cfg = load_config(config_path=yml, dot_overrides=[], schema=EvalConfig)
    assert cfg.filters["domain"] == "Physics"


def test_filters_empty_validates_for_all_benchmarks():
    """Every concrete benchmark must declare a filter model that accepts {}."""
    for name in ("hle", "lcb", "gpqa", "babyvision", "aime", "aime2025", "aime2026", "rbenchv"):
        cfg = EvalConfig(
            benchmark=name,
            backend="claude",
            explore_model="m",
            method="self-refine",
        )
        assert isinstance(cfg.filters, dict)
        # With exclude_defaults=True, an empty input round-trips to empty dict
        assert cfg.filters == {}


def test_parse_cli_only(tmp_path, monkeypatch):
    import importlib
    import eval as eval_mod
    importlib.reload(eval_mod)

    argv = [
        "eval.py",
        "--benchmark", "hle",
        "--backend", "claude",
        "--explore-model", "claude-sonnet-4-6",
        "--method", "self-refine",
        "--num", "20",
    ]
    monkeypatch.setattr("sys.argv", argv)
    cfg = eval_mod.parse_cli()
    assert cfg.benchmark == "hle"
    assert cfg.num == 20
    assert cfg.method == "self-refine"


def test_parse_cli_with_yaml_and_override(tmp_path, monkeypatch):
    import importlib
    import eval as eval_mod
    importlib.reload(eval_mod)

    yml = _write(tmp_path, "x.yaml", """
        benchmark: hle
        backend: claude
        explore_model: claude-sonnet-4-6
        method: tts-agent-multi
        orchestrator_model: claude-sonnet-4-6
        cache_dirs:
          haiku: /cache/haiku
          sonnet: /cache/sonnet
        model_budgets:
          haiku: 8
          sonnet: 8
    """)
    argv = ["eval.py", "--benchmark", "hle", "--config", str(yml), "-o", "model_budgets.haiku=2", "-o", "seed=99"]
    monkeypatch.setattr("sys.argv", argv)
    cfg = eval_mod.parse_cli()
    assert cfg.model_budgets == {"haiku": 2, "sonnet": 8}
    assert cfg.seed == 99


def test_cli_filter_flag_preserves_yaml_filter_siblings(tmp_path, monkeypatch):
    """C1 regression: CLI --category should NOT erase YAML's subset/text_only."""
    import importlib
    import eval as eval_mod
    importlib.reload(eval_mod)

    yml = _write(tmp_path, "x.yaml", """
        benchmark: hle
        backend: claude
        explore_model: claude-sonnet-4-6
        method: self-refine
        filters:
          subset: gold
          text_only: true
    """)
    argv = ["eval.py", "--benchmark", "hle", "--config", str(yml), "--category", "physics"]
    monkeypatch.setattr("sys.argv", argv)
    cfg = eval_mod.parse_cli()
    assert cfg.filters["subset"] == "gold"
    assert cfg.filters["text_only"] is True
    assert cfg.filters["category"] == "physics"


def test_cli_cache_dirs_single_path_routes_to_cache_dir(tmp_path, monkeypatch):
    """C2 regression: legacy --cache-dirs <single-path> still works post-rename-revert."""
    import importlib
    import eval as eval_mod
    importlib.reload(eval_mod)

    argv = [
        "eval.py",
        "--benchmark", "hle",
        "--backend", "claude",
        "--explore-model", "claude-sonnet-4-6",
        "--method", "self-refine",
        "--cache-dirs", "/some/cache/path",
    ]
    monkeypatch.setattr("sys.argv", argv)
    cfg = eval_mod.parse_cli()
    assert cfg.cache_dir == Path("/some/cache/path")
    assert cfg.cache_dirs == {}


def test_cli_cache_dirs_with_colons_rejects(tmp_path, monkeypatch):
    """C2: legacy multi-cache string form must error, point to YAML."""
    import importlib
    import eval as eval_mod
    importlib.reload(eval_mod)

    argv = [
        "eval.py",
        "--benchmark", "hle",
        "--backend", "claude",
        "--explore-model", "claude-sonnet-4-6",
        "--method", "tts-agent-multi",
        "--orchestrator-model", "x",
        "--cache-dirs", "haiku:/cache/a,sonnet:/cache/b",
    ]
    monkeypatch.setattr("sys.argv", argv)
    with pytest.raises(AssertionError, match="multi-model cache dicts must come from"):
        eval_mod.parse_cli()
