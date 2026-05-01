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


# ---------------------------------------------------------------------------
# Phase 2: copy
# ---------------------------------------------------------------------------

def test_copy_creates_full_bundle(tmp_path):
    cache = tmp_path / "cache"
    explore = cache / "qid1" / "explore_1"
    _make_legacy_explore(explore)
    mig.run(cache_root=cache, phase="copy", limit=None)
    bundle = explore / "judges" / "claude__claude-haiku-4-5-20251001"
    assert bundle.exists()
    for fname in ("config.json", "grade.json", "input.md", "output.md", "result.json"):
        assert (bundle / fname).exists(), f"missing {fname}"
    cfg = json.loads((bundle / "config.json").read_text())
    assert cfg == {"name": "claude", "model": "claude-haiku-4-5-20251001"}


def test_copy_leaves_originals_untouched(tmp_path):
    cache = tmp_path / "cache"
    explore = cache / "qid1" / "explore_1"
    _make_legacy_explore(explore)
    src_grade_hash = _sha256(explore / "grade.json")
    src_judge_input_hash = _sha256(explore / "judge" / "input.md")
    mig.run(cache_root=cache, phase="copy", limit=None)
    # Originals still present, identical bytes (copy not move).
    assert (explore / "grade.json").exists()
    assert (explore / "judge" / "input.md").exists()
    assert _sha256(explore / "grade.json") == src_grade_hash
    assert _sha256(explore / "judge" / "input.md") == src_judge_input_hash


def test_copy_byte_for_byte_fidelity(tmp_path):
    cache = tmp_path / "cache"
    explore = cache / "qid1" / "explore_1"
    _make_legacy_explore(explore)
    src_grade = _sha256(explore / "grade.json")
    src_in = _sha256(explore / "judge" / "input.md")
    src_out = _sha256(explore / "judge" / "output.md")
    src_result = _sha256(explore / "judge" / "result.json")
    mig.run(cache_root=cache, phase="copy", limit=None)
    bundle = explore / "judges" / "claude__claude-haiku-4-5-20251001"
    assert _sha256(bundle / "grade.json") == src_grade
    assert _sha256(bundle / "input.md") == src_in
    assert _sha256(bundle / "output.md") == src_out
    assert _sha256(bundle / "result.json") == src_result


def test_copy_idempotent(tmp_path):
    cache = tmp_path / "cache"
    explore = cache / "qid1" / "explore_1"
    _make_legacy_explore(explore)
    mig.run(cache_root=cache, phase="copy", limit=None)
    bundle = explore / "judges" / "claude__claude-haiku-4-5-20251001"
    mtime_first = (bundle / "config.json").stat().st_mtime
    mig.run(cache_root=cache, phase="copy", limit=None)  # second run
    mtime_second = (bundle / "config.json").stat().st_mtime
    assert mtime_first == mtime_second  # already-migrated; skipped


def test_copy_limit(tmp_path):
    cache = tmp_path / "cache"
    for i in range(5):
        _make_legacy_explore(cache / f"qid{i}" / "explore_1")
    mig.run(cache_root=cache, phase="copy", limit=2)
    migrated = list(cache.rglob("config.json"))
    assert len(migrated) == 2


def test_copy_aborts_on_unknown_judge_model(tmp_path):
    cache = tmp_path / "cache"
    _make_legacy_explore(cache / "qid1" / "explore_1", judge_model="gpt-4-turbo-preview")
    with pytest.raises(SystemExit, match="Unexpected judge_model"):
        mig.run(cache_root=cache, phase="copy", limit=None)


def test_copy_processes_many_qids(tmp_path):
    """Confirm we don't only walk the first qid."""
    cache = tmp_path / "cache"
    for q in range(3):
        for e in range(1, 4):  # 3 explores per qid
            _make_legacy_explore(cache / f"qid{q}" / f"explore_{e}")
    mig.run(cache_root=cache, phase="copy", limit=None)
    bundles = list(cache.rglob("judges/claude__claude-haiku-4-5-20251001/grade.json"))
    assert len(bundles) == 9


# ---------------------------------------------------------------------------
# Phase 3: cleanup (only after copy + smoke eval verification pass)
# ---------------------------------------------------------------------------

def test_cleanup_removes_legacy_after_copy(tmp_path):
    cache = tmp_path / "cache"
    explore = cache / "qid1" / "explore_1"
    _make_legacy_explore(explore)
    mig.run(cache_root=cache, phase="copy", limit=None)
    # Pre-cleanup: both legacy and bundle exist.
    assert (explore / "grade.json").exists()
    assert (explore / "judge").exists()
    mig.run(cache_root=cache, phase="cleanup", limit=None)
    # Post-cleanup: legacy removed, bundle intact.
    assert not (explore / "grade.json").exists()
    assert not (explore / "judge").exists()
    bundle = explore / "judges" / "claude__claude-haiku-4-5-20251001"
    assert (bundle / "grade.json").exists()
    assert (bundle / "input.md").exists()
    assert (bundle / "output.md").exists()
    assert (bundle / "result.json").exists()
    assert (bundle / "config.json").exists()


def test_cleanup_skips_explore_with_incomplete_bundle(tmp_path, capsys):
    cache = tmp_path / "cache"
    explore = cache / "qid1" / "explore_1"
    _make_legacy_explore(explore)
    mig.run(cache_root=cache, phase="copy", limit=None)
    # Corrupt the bundle: delete config.json. Cleanup must refuse to delete legacy.
    bundle = explore / "judges" / "claude__claude-haiku-4-5-20251001"
    (bundle / "config.json").unlink()
    mig.run(cache_root=cache, phase="cleanup", limit=None)
    # Legacy preserved for manual inspection.
    assert (explore / "grade.json").exists()
    assert (explore / "judge").exists()
    captured = capsys.readouterr()
    assert "skipped" in captured.out.lower()


def test_cleanup_no_op_if_copy_never_ran(tmp_path):
    """Without phase copy, cleanup must not delete legacy files."""
    cache = tmp_path / "cache"
    explore = cache / "qid1" / "explore_1"
    _make_legacy_explore(explore)
    mig.run(cache_root=cache, phase="cleanup", limit=None)
    assert (explore / "grade.json").exists()
    assert (explore / "judge").exists()


def test_cleanup_skips_when_config_model_mismatches(tmp_path, capsys):
    """If somebody hand-edited config.json to point at a different model
    after copy, cleanup must NOT delete the legacy under that model's
    inferred label (validation must catch the drift)."""
    cache = tmp_path / "cache"
    explore = cache / "qid1" / "explore_1"
    _make_legacy_explore(explore, judge_model="claude-haiku-4-5-20251001")
    mig.run(cache_root=cache, phase="copy", limit=None)
    bundle = explore / "judges" / "claude__claude-haiku-4-5-20251001"
    # Drift the config.json to a different model.
    (bundle / "config.json").write_text(json.dumps({
        "name": "claude", "model": "claude-haiku-4-6-fake",
    }))
    mig.run(cache_root=cache, phase="cleanup", limit=None)
    assert (explore / "grade.json").exists()  # legacy preserved


def test_cleanup_idempotent(tmp_path):
    cache = tmp_path / "cache"
    explore = cache / "qid1" / "explore_1"
    _make_legacy_explore(explore)
    mig.run(cache_root=cache, phase="copy", limit=None)
    mig.run(cache_root=cache, phase="cleanup", limit=None)
    # Second cleanup should not raise; legacy already gone.
    mig.run(cache_root=cache, phase="cleanup", limit=None)


def test_full_lifecycle_dry_copy_smoke_cleanup(tmp_path):
    """End-to-end: dry-run -> copy -> cleanup leaves only the bundle."""
    cache = tmp_path / "cache"
    for q in range(3):
        _make_legacy_explore(cache / f"qid{q}" / "explore_1")
    mig.run(cache_root=cache, phase="dry-run", limit=None)
    mig.run(cache_root=cache, phase="copy", limit=None)
    mig.run(cache_root=cache, phase="cleanup", limit=None)
    for q in range(3):
        explore = cache / f"qid{q}" / "explore_1"
        assert not (explore / "grade.json").exists()
        assert not (explore / "judge").exists()
        bundle = explore / "judges" / "claude__claude-haiku-4-5-20251001"
        assert (bundle / "grade.json").exists()
