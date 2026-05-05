"""Validator tests for RoleSlot (single-call cacheless role)."""
from __future__ import annotations

import sys
from pathlib import Path

_CORE_CODE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_CORE_CODE_DIR))

import pytest
from pydantic import ValidationError

from methods.specs import ModelConfig, RoleSlot


def test_role_slot_loads():
    slot = RoleSlot(
        model=ModelConfig(backend="claude", model="claude-sonnet-4-6"),
    )
    assert slot.model.model == "claude-sonnet-4-6"


def test_role_slot_rejects_num_explores_typo():
    """RoleSlot is for single-call roles; num_explores would belong on ExploreVariant."""
    with pytest.raises(ValidationError):
        RoleSlot(
            model=ModelConfig(backend="claude", model="claude-sonnet-4-6"),
            num_explores=8,
        )


def test_role_slot_rejects_cache_dir():
    """RoleSlot.cache_dir was deleted in 2026-05-05 explore-cache-owner refactor.
    integrate role is cacheless; explore caching is owned by ExploreVariant."""
    with pytest.raises(ValidationError):
        RoleSlot(
            model=ModelConfig(backend="claude", model="claude-sonnet-4-6"),
            cache_dir=Path("/tmp/x"),
        )
