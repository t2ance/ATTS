"""Validator tests for ExploreVariant (per-variant explore call config)."""
from __future__ import annotations

import sys
from pathlib import Path

_CORE_CODE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_CORE_CODE_DIR))

import pytest
from pydantic import ValidationError

from methods.specs import ExploreVariant, ModelConfig


def test_explore_variant_loads():
    v = ExploreVariant(
        label="haiku",
        model=ModelConfig(backend="claude", model="claude-haiku-4-5-20251001"),
        cache_dir=Path("/tmp/cache/haiku"),
    )
    assert v.label == "haiku"
    assert v.num_explores == 8


def test_explore_variant_label_required():
    with pytest.raises(ValidationError):
        ExploreVariant(
            model=ModelConfig(backend="claude", model="claude-haiku-4-5-20251001"),
            cache_dir=Path("/tmp/cache/haiku"),
        )


def test_explore_variant_rejects_extra_fields():
    with pytest.raises(ValidationError):
        ExploreVariant(
            label="haiku",
            model=ModelConfig(backend="claude", model="claude-haiku-4-5-20251001"),
            cache_dir=Path("/tmp/cache/haiku"),
            unknown_key=1,
        )
