"""Migrate legacy grade.json + judge/ layout to judges/<label>/ bundle layout.

Three phases (run in order, one per invocation, each safe to retry):

    --phase dry-run   Walk the cache tree, print eligible counts. Zero I/O.
    --phase copy      Copy each explore's grade.json + judge/* into
                      judges/<label>/, write config.json. Verify sha256 of
                      every copied byte. Originals untouched (safe).
    --phase cleanup   After verification + smoke eval pass, delete the legacy
                      grade.json and judge/ from each fully-migrated explore.

Usage:

    python scripts/migrate_judge_layout.py --phase dry-run \
        --cache-root analysis/cache/hle/qwen36_35b_a3b_fp8/gold

    # Pilot first (recommended): migrate only N explores.
    python scripts/migrate_judge_layout.py --phase copy --limit 5 \
        --cache-root analysis/cache/hle/qwen36_35b_a3b_fp8/gold

    # Full copy after pilot inspection passes.
    python scripts/migrate_judge_layout.py --phase copy \
        --cache-root analysis/cache/hle/qwen36_35b_a3b_fp8/gold

    # Run a smoke eval to confirm $0 judge cost on the migrated cache.
    # Only after smoke passes:
    python scripts/migrate_judge_layout.py --phase cleanup \
        --cache-root analysis/cache/hle/qwen36_35b_a3b_fp8/gold

Design (from docs/superpowers/specs/2026-05-01-judge-block-design.md):
- Use COPY-then-cleanup, not mv. Originals stay until phase 3, so any failure
  in phase 2 leaves the live cache fully intact.
- Phase 2 verifies every copied byte via sha256 before declaring a file done.
- Phase 3 only deletes a legacy file once its destination bundle has all 5
  required files AND its config.json matches the inferred JudgeSpec.
- Idempotent: re-running any phase on an already-processed explore is a no-op.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path


# All legacy grade.json entries observed in cache as of 2026-05-01 use a
# claude-* judge model. Any other model encountered is a hard stop: the user
# must extend this script (and double-check that the new model's bundle layout
# semantics are correct) before continuing.
def _label(judge_model: str) -> str:
    if judge_model.startswith("claude-"):
        return f"claude__{judge_model}"
    raise SystemExit(
        f"Unexpected judge_model {judge_model!r} in legacy grade.json. "
        f"Migration script only handles claude-* models. Extend manually "
        f"(infer the right backend, decide on collision resolution)."
    )


def _explore_dirs(cache_root: Path):
    """Yield every cache_root/.../<qid>/explore_N/ that contains grade.json."""
    for grade in cache_root.rglob("grade.json"):
        explore_dir = grade.parent
        if explore_dir.name.startswith("explore_"):
            yield explore_dir


def _is_already_migrated(explore_dir: Path) -> bool:
    """Returns True if explore_dir/judges/* contains any grade.json."""
    judges_dir = explore_dir / "judges"
    if not judges_dir.exists():
        return False
    for sub in judges_dir.iterdir():
        if sub.is_dir() and (sub / "grade.json").exists():
            return True
    return False


# ---------------------------------------------------------------------------
# Phase 1: dry-run
# ---------------------------------------------------------------------------

def _phase_dry_run(cache_root: Path, limit: int | None) -> None:
    eligible = 0
    already = 0
    samples: list[tuple[Path, str]] = []
    for explore_dir in _explore_dirs(cache_root):
        if _is_already_migrated(explore_dir):
            already += 1
            continue
        # Force the unknown-judge_model check now so the user finds out at
        # dry-run time, not halfway through copy.
        data = json.loads((explore_dir / "grade.json").read_text(encoding="utf-8"))
        label = _label(data["judge_model"])
        eligible += 1
        if len(samples) < 5:
            samples.append((explore_dir, label))

    print(f"Cache root: {cache_root}")
    if samples:
        print("Sample destinations:")
        for explore_dir, label in samples:
            try:
                rel = explore_dir.relative_to(cache_root)
            except ValueError:
                rel = explore_dir
            print(f"  {rel}/  ->  judges/{label}/")
    print(
        f"Total: {eligible} explores eligible for migration; "
        f"{already} already migrated."
    )


# ---------------------------------------------------------------------------
# Phase 2 / 3: stubs (implemented in subsequent commits)
# ---------------------------------------------------------------------------

def _phase_copy(cache_root: Path, limit: int | None) -> None:
    raise NotImplementedError("Implemented in the next commit.")


def _phase_cleanup(cache_root: Path, limit: int | None) -> None:
    raise NotImplementedError("Implemented in the next commit.")


def run(cache_root: Path, phase: str, limit: int | None) -> None:
    if phase == "dry-run":
        return _phase_dry_run(cache_root, limit)
    if phase == "copy":
        return _phase_copy(cache_root, limit)
    if phase == "cleanup":
        return _phase_cleanup(cache_root, limit)
    raise SystemExit(f"Unknown phase: {phase!r}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--cache-root", type=Path, required=True,
                   help="Root of the cache tree to migrate (e.g. analysis/cache/hle/<model>/<filter>).")
    p.add_argument("--phase", choices=["dry-run", "copy", "cleanup"], required=True)
    p.add_argument("--limit", type=int, default=None,
                   help="Process at most N eligible explores (for piloting copy phase).")
    args = p.parse_args()
    run(cache_root=args.cache_root, phase=args.phase, limit=args.limit)


if __name__ == "__main__":
    main()
