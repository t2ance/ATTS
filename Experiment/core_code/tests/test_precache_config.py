from __future__ import annotations
import sys
from pathlib import Path
import textwrap

_CORE_CODE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_CORE_CODE_DIR))

import pytest
import yaml
from pydantic import ValidationError
from eval import load_config
from precache_explores import PrecacheConfig


def _write(tmp_path, name, body):
    p = tmp_path / name
    p.write_text(textwrap.dedent(body))
    return p


def _minimal_kwargs(**overrides):
    base = {
        "benchmark": {"name": "hle"},
        "backend": "claude",
        "explore_model": "claude-sonnet-4-6",
        "cache_dir": "/cache/x",
    }
    base.update(overrides)
    return base


def test_minimal_precache_validates():
    cfg = PrecacheConfig(**_minimal_kwargs())
    assert cfg.benchmark.name == "hle"
    assert cfg.cache_dir == Path("/cache/x")
    assert cfg.num_explores == 8


def test_precache_requires_cache_dir():
    kw = _minimal_kwargs()
    del kw["cache_dir"]
    with pytest.raises(ValidationError, match="cache_dir"):
        PrecacheConfig(**kw)


def test_precache_loader_yaml(tmp_path):
    yml = _write(tmp_path, "p.yaml", """
        benchmark:
          name: hle
          subset: gold
        backend: claude
        explore_model: claude-sonnet-4-6
        cache_dir: /cache/h
        num_explores: 16
    """)
    cfg = load_config(config_path=yml, schema=PrecacheConfig)
    assert cfg.cache_dir == Path("/cache/h")
    assert cfg.num_explores == 16
    assert cfg.benchmark.subset == "gold"


def test_precache_filter_validation(tmp_path):
    yml = _write(tmp_path, "p.yaml", """
        benchmark:
          name: hle
          difficulty: hard
        backend: claude
        explore_model: claude-sonnet-4-6
        cache_dir: /cache/h
    """)
    with pytest.raises(ValidationError, match="difficulty"):
        load_config(config_path=yml, schema=PrecacheConfig)


def test_precache_extra_field_forbidden():
    """A typoed field in PrecacheConfig must fail loudly."""
    with pytest.raises(ValidationError, match="typoed_field|extra"):
        PrecacheConfig(**_minimal_kwargs(typoed_field=True))


def test_precache_rejects_eval_only_keys():
    """A user pasting an EvalConfig YAML into precache must fail at validation."""
    with pytest.raises(ValidationError, match="method|extra"):
        PrecacheConfig(**_minimal_kwargs(method="tts-agent"))
