"""Migrate legacy grade.json + judge/ layout to judges/<label>/ bundle layout.

Three phases (run in order, one per invocation, each safe to retry):

    --phase dry-run   Walk the cache tree, print eligible counts. Zero I/O.
    --phase copy      Copy each explore's grade.json + judge/* into
                      judges/<label>/, write config.json. Verify sha256 of
                      every copied byte. Originals untouched (safe).
    --phase cleanup   After verification + smoke eval pass, ARCHIVE the legacy
                      grade.json and judge/ to --archive-root (preserving the
                      relative path under cache-root) via verified copy, then
                      delete from cache. Originals are deleted only after sha256
                      verification passes on the archive copy.

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
# Phase 2: copy (verified, never moves; safe to retry; safe to abort)
# ---------------------------------------------------------------------------

def _atomic_write_text(path: Path, content: str) -> None:
    """Write content to path via tempfile + rename (atomic on same fs)."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(content, encoding="utf-8")
    tmp.replace(path)


def _verified_copy(src: Path, dst: Path) -> None:
    """Copy src -> dst and assert byte-for-byte equality via sha256."""
    shutil.copy2(src, dst)
    src_h = hashlib.sha256(src.read_bytes()).hexdigest()
    dst_h = hashlib.sha256(dst.read_bytes()).hexdigest()
    if src_h != dst_h:
        raise SystemExit(
            f"Hash mismatch after copy: {src} -> {dst}\n"
            f"  src sha256: {src_h}\n  dst sha256: {dst_h}\n"
            f"Migration aborted. Original {src} intact; partial bundle at {dst.parent}."
        )


def _phase_copy(cache_root: Path, limit: int | None) -> None:
    processed = 0
    skipped_already = 0

    for explore_dir in _explore_dirs(cache_root):
        if limit is not None and processed >= limit:
            break
        if _is_already_migrated(explore_dir):
            skipped_already += 1
            continue

        legacy_grade = explore_dir / "grade.json"
        data = json.loads(legacy_grade.read_text(encoding="utf-8"))
        judge_model = data["judge_model"]
        label = _label(judge_model)

        bundle = explore_dir / "judges" / label
        bundle.mkdir(parents=True, exist_ok=True)

        # 1) config.json -- atomic, deterministic-key bytes for find_cached_judge.
        config = {"name": "claude", "model": judge_model}
        _atomic_write_text(
            bundle / "config.json",
            json.dumps(config, indent=2, sort_keys=True),
        )

        # 2) grade.json (verified copy from legacy explore_N/grade.json).
        _verified_copy(legacy_grade, bundle / "grade.json")

        # 3) Judge trace files. Two are required (output.md, result.json);
        #    input.md is optional because current grader.py:judge_answer never
        #    writes it (only TrajectoryWriter+output.md and save_sub_model_result
        #    +result.json fire today). Older caches (e.g. 2026-03 opus run) DO
        #    have input.md from a removed code path; copy it when present.
        legacy_judge = explore_dir / "judge"
        for fname in ("output.md", "result.json"):
            src = legacy_judge / fname
            if not src.exists():
                raise SystemExit(
                    f"Required legacy file missing: {src}. Cannot migrate this "
                    f"explore. Inspect manually before retrying."
                )
            _verified_copy(src, bundle / fname)
        # Optional: input.md (only present in older caches).
        opt_in = legacy_judge / "input.md"
        if opt_in.exists():
            _verified_copy(opt_in, bundle / "input.md")

        # 4) Sanity: re-read config.json and confirm content matches.
        stored = json.loads((bundle / "config.json").read_text(encoding="utf-8"))
        if stored != config:
            raise SystemExit(f"config.json content drift at {bundle}")

        processed += 1
        if processed % 50 == 0:
            print(f"  ... copied {processed} explores")

    print(
        f"Phase copy: {processed} explores migrated, "
        f"{skipped_already} already-migrated skipped."
    )
    if processed > 0:
        print("Originals untouched. Run a smoke eval next; only after that passes "
              "should you run --phase cleanup.")


# ---------------------------------------------------------------------------
# Phase 3: cleanup (only after copy + smoke eval verification pass)
# ---------------------------------------------------------------------------

# config.json + grade.json + judge trace (output.md, result.json) are required.
# input.md is optional (only present in older caches that wrote it; current
# grader.py:judge_answer doesn't produce input.md).
_BUNDLE_REQUIRED_FILES = ("config.json", "grade.json", "output.md", "result.json")


def _bundle_complete_and_correct(bundle: Path, expected_model: str) -> bool:
    """Required files present AND config.json matches inferred spec."""
    for fname in _BUNDLE_REQUIRED_FILES:
        if not (bundle / fname).exists():
            return False
    cfg = json.loads((bundle / "config.json").read_text(encoding="utf-8"))
    return cfg.get("name") == "claude" and cfg.get("model") == expected_model


def _archive_then_delete(src: Path, archive_dst: Path) -> None:
    """Verified archive of src to archive_dst, then unlink src.

    File: copy2 + sha256 verify, then unlink.
    Directory: copytree + per-file sha256 verify, then rmtree.
    Cross-filesystem safe (archive may be on a different mount than cache).
    """
    archive_dst.parent.mkdir(parents=True, exist_ok=True)
    if src.is_file():
        shutil.copy2(src, archive_dst)
        if hashlib.sha256(src.read_bytes()).hexdigest() != hashlib.sha256(archive_dst.read_bytes()).hexdigest():
            raise SystemExit(f"sha256 mismatch archiving {src} -> {archive_dst}; src untouched.")
        src.unlink()
    elif src.is_dir():
        # copytree fails if dst exists; if a prior cleanup-attempt was interrupted
        # mid-judge-dir, the archive dir may already exist — drop it first.
        if archive_dst.exists():
            shutil.rmtree(archive_dst)
        shutil.copytree(src, archive_dst)
        # Verify every file copied correctly.
        for src_file in src.rglob("*"):
            if src_file.is_file():
                rel = src_file.relative_to(src)
                dst_file = archive_dst / rel
                if hashlib.sha256(src_file.read_bytes()).hexdigest() != hashlib.sha256(dst_file.read_bytes()).hexdigest():
                    raise SystemExit(f"sha256 mismatch archiving dir-file {src_file} -> {dst_file}; src untouched.")
        shutil.rmtree(src)
    else:
        raise SystemExit(f"Cannot archive: {src} is neither file nor directory.")


def _phase_cleanup(cache_root: Path, archive_root: Path, limit: int | None) -> None:
    """Move legacy grade.json + judge/ to archive_root preserving relative path.

    For each explore_dir at cache_root/<...>/<qid>/explore_N/, the legacy
    grade.json is archived to archive_root/<...>/<qid>/explore_N/grade.json
    (mirroring cache_root structure), then unlinked from cache_root. Same for
    judge/ subdirectory. Originals are deleted ONLY after sha256-verified
    archive copies exist.
    """
    if archive_root.resolve() == cache_root.resolve() or archive_root.resolve().is_relative_to(cache_root.resolve()):
        raise SystemExit(
            f"--archive-root must NOT be inside --cache-root.\n"
            f"  cache-root:   {cache_root}\n"
            f"  archive-root: {archive_root}"
        )

    deleted = 0
    skipped = 0

    for explore_dir in _explore_dirs(cache_root):
        if limit is not None and deleted >= limit:
            break
        legacy_grade = explore_dir / "grade.json"
        if not legacy_grade.exists():
            continue  # already cleaned (or never had grade.json)

        data = json.loads(legacy_grade.read_text(encoding="utf-8"))
        judge_model = data["judge_model"]
        label = _label(judge_model)
        bundle = explore_dir / "judges" / label

        if not bundle.exists() or not _bundle_complete_and_correct(bundle, judge_model):
            print(f"  skipped (incomplete or drifted bundle): {explore_dir}")
            skipped += 1
            continue

        rel = explore_dir.relative_to(cache_root)
        archive_explore_dir = archive_root / rel

        # Archive then delete legacy grade.json.
        _archive_then_delete(legacy_grade, archive_explore_dir / "grade.json")

        # Archive then delete legacy judge/ directory.
        legacy_judge = explore_dir / "judge"
        if legacy_judge.exists():
            _archive_then_delete(legacy_judge, archive_explore_dir / "judge")

        deleted += 1
        if deleted % 50 == 0:
            print(f"  ... archived {deleted} explores")

    print(f"Phase cleanup: {deleted} legacy entries archived to {archive_root}, {skipped} skipped.")


def _phase_archive_only(cache_root: Path, archive_root: Path, limit: int | None) -> None:
    """Archive legacy grade.json + judge/ without writing any new bundle.

    For caches whose legacy grades cannot or should not be migrated to the new
    judges/<label>/ layout (e.g. judge_model is vllm-served and current YAML
    expects a different judge, or the benchmark grades without an LLM judge so
    the legacy grade.json is dead data). Result: cache/<qid>/explore_N/ keeps
    only the explore artifact (result.json etc.); grade.json + judge/ move to
    archive_root preserving cache-root-relative path.
    """
    if archive_root.resolve() == cache_root.resolve() or archive_root.resolve().is_relative_to(cache_root.resolve()):
        raise SystemExit(
            f"--archive-root must NOT be inside --cache-root.\n"
            f"  cache-root:   {cache_root}\n"
            f"  archive-root: {archive_root}"
        )

    archived = 0
    for explore_dir in _explore_dirs(cache_root):
        if limit is not None and archived >= limit:
            break
        legacy_grade = explore_dir / "grade.json"
        legacy_judge = explore_dir / "judge"
        if not legacy_grade.exists() and not legacy_judge.exists():
            continue
        rel = explore_dir.relative_to(cache_root)
        archive_explore_dir = archive_root / rel
        if legacy_grade.exists():
            _archive_then_delete(legacy_grade, archive_explore_dir / "grade.json")
        if legacy_judge.exists():
            _archive_then_delete(legacy_judge, archive_explore_dir / "judge")
        archived += 1
        if archived % 100 == 0:
            print(f"  ... archived {archived} explores")

    print(f"Phase archive-only: {archived} legacy entries archived to {archive_root}.")


def run(cache_root: Path, phase: str, limit: int | None, archive_root: Path | None = None) -> None:
    if phase == "dry-run":
        return _phase_dry_run(cache_root, limit)
    if phase == "copy":
        return _phase_copy(cache_root, limit)
    if phase == "cleanup":
        if archive_root is None:
            raise SystemExit("--archive-root is required for --phase cleanup")
        return _phase_cleanup(cache_root, archive_root, limit)
    if phase == "archive-only":
        if archive_root is None:
            raise SystemExit("--archive-root is required for --phase archive-only")
        return _phase_archive_only(cache_root, archive_root, limit)
    raise SystemExit(f"Unknown phase: {phase!r}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--cache-root", type=Path, required=True,
                   help="Root of the cache tree to migrate (e.g. analysis/cache/hle/<model>/<filter>).")
    p.add_argument("--phase", choices=["dry-run", "copy", "cleanup", "archive-only"], required=True)
    p.add_argument("--archive-root", type=Path, default=None,
                   help="Required for --phase cleanup and --phase archive-only. Legacy grade.json + "
                        "judge/ are moved here, preserving relative path under cache-root. Must be outside cache-root.")
    p.add_argument("--limit", type=int, default=None,
                   help="Process at most N eligible explores (for piloting copy phase).")
    args = p.parse_args()
    run(cache_root=args.cache_root, phase=args.phase, limit=args.limit, archive_root=args.archive_root)


if __name__ == "__main__":
    main()
