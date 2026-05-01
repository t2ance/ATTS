"""Unit tests for scripts/migrate_judge_layout.py.

Each test builds a tiny fake cache tree under tmp_path and exercises the
migration script's three phases (dry-run / copy / cleanup) without ever
touching the real analysis/cache/ tree.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
import sys
from pathlib import Path

import pytest

_CORE_CODE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_CORE_CODE_DIR))

# Import the migration script as a module (it lives under scripts/).
SCRIPT_PATH = _CORE_CODE_DIR / "scripts" / "migrate_judge_layout.py"
_spec = importlib.util.spec_from_file_location("migrate_judge_layout", SCRIPT_PATH)
mig = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mig)


# ---------------------------------------------------------------------------
# Test fixture helpers
# ---------------------------------------------------------------------------

def _make_legacy_explore(explore_dir: Path,
                         judge_model: str = "claude-haiku-4-5-20251001",
                         is_correct: bool = True):
    """Create a single legacy-layout explore_N directory with all 7 files."""
    explore_dir.mkdir(parents=True, exist_ok=True)
    (explore_dir / "result.json").write_text(json.dumps({"answer": "D", "cost_usd": 0.0}))
    (explore_dir / "input.md").write_text("question prompt")
    (explore_dir / "output.md").write_text("model output")
    (explore_dir / "grade.json").write_text(json.dumps({
        "judge_model": judge_model,
        "is_correct": is_correct,
        "predicted": "D",
        "gold": "D",
        "judge_cost_usd": 0.0046,
    }))
    judge_dir = explore_dir / "judge"
    judge_dir.mkdir()
    (judge_dir / "input.md").write_text("judge prompt")
    (judge_dir / "output.md").write_text("judge output")
    (judge_dir / "result.json").write_text(json.dumps({"correct": is_correct}))


def _sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# Phase 1: dry-run
# ---------------------------------------------------------------------------

def test_dry_run_counts_eligible(tmp_path, capsys):
    cache = tmp_path / "cache"
    _make_legacy_explore(cache / "qid1" / "explore_1")
    _make_legacy_explore(cache / "qid1" / "explore_2")
    _make_legacy_explore(cache / "qid2" / "explore_1")
    mig.run(cache_root=cache, phase="dry-run", limit=None)
    captured = capsys.readouterr()
    assert "Total: 3 explores eligible" in captured.out
    # Originals untouched.
    assert (cache / "qid1" / "explore_1" / "grade.json").exists()
    assert (cache / "qid1" / "explore_1" / "judge").exists()


def test_dry_run_marks_already_migrated(tmp_path, capsys):
    cache = tmp_path / "cache"
    explore = cache / "qid1" / "explore_1"
    _make_legacy_explore(explore)
    # Pre-create new-layout bundle to simulate already-migrated.
    bundle = explore / "judges" / "claude__claude-haiku-4-5-20251001"
    bundle.mkdir(parents=True)
    (bundle / "config.json").write_text(json.dumps({
        "name": "claude", "model": "claude-haiku-4-5-20251001",
    }))
    (bundle / "grade.json").write_text("{}")
    mig.run(cache_root=cache, phase="dry-run", limit=None)
    captured = capsys.readouterr()
    assert "0 explores eligible" in captured.out
    assert "1 already migrated" in captured.out


def test_dry_run_aborts_on_unknown_judge_model(tmp_path):
    cache = tmp_path / "cache"
    _make_legacy_explore(cache / "qid1" / "explore_1", judge_model="gpt-4-turbo-preview")
    with pytest.raises(SystemExit, match="Unexpected judge_model"):
        mig.run(cache_root=cache, phase="dry-run", limit=None)


def test_dry_run_skips_non_explore_dirs(tmp_path, capsys):
    """Random sibling files/dirs that aren't explore_N must not be counted."""
    cache = tmp_path / "cache"
    _make_legacy_explore(cache / "qid1" / "explore_1")
    (cache / "qid1" / "junk.txt").write_text("hi")
    (cache / "stray_dir").mkdir()
    mig.run(cache_root=cache, phase="dry-run", limit=None)
    captured = capsys.readouterr()
    assert "Total: 1 explores eligible" in captured.out
