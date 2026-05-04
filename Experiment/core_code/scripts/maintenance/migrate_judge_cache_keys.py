#!/usr/bin/env python
"""One-shot data migration: bring every cached judges/<label>/config.json
into the new ModelConfig schema.

Two renames per file (both idempotent):
  - top-level `name`      -> `backend`     (every backend: claude/codex/vllm/openrouter)
  - vllm-only `sampling`  -> `vllm_sampling`

Why: post-modelconfig-refactor (2026-05-04) the in-process judge_spec dict is
a `ModelConfig.model_dump()`, whose discriminator field is `backend` not
`name`. find_cached_judge does dict-set comparison against the on-disk
config.json; a stale `name` key on disk plus a fresh `backend` key in the
request triggers the unidirectional spec-narrowing branch and raises
RuntimeError on the first cache lookup.

Idempotent: re-running on already-migrated configs is a no-op (skipped count
goes up, migrated count stays 0).

Side effect: rewrites file in place via UTF-8 with indent=2.
"""
from __future__ import annotations

import json
from pathlib import Path

ANALYSIS_DIR = Path("/data3/peijia/dr-claw/Explain/Experiment/analysis")


def _migrate_one(p: Path) -> bool:
    """Returns True if a write happened, False if file was already migrated."""
    cfg = json.loads(p.read_text(encoding="utf-8"))
    changed = False
    if "name" in cfg and "backend" not in cfg:
        cfg["backend"] = cfg.pop("name")
        changed = True
    # vllm-only: sampling -> vllm_sampling. Other backends never had a sampling
    # field on disk so the test below is a no-op for them.
    if "sampling" in cfg and "vllm_sampling" not in cfg:
        cfg["vllm_sampling"] = cfg.pop("sampling")
        changed = True
    if changed:
        p.write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")
    return changed


def main() -> None:
    cfg_paths = list(ANALYSIS_DIR.rglob("judges/*/config.json"))
    if not cfg_paths:
        print("No judge config.json files found; nothing to migrate.")
        return

    migrated = 0
    skipped = 0
    for p in cfg_paths:
        if _migrate_one(p):
            migrated += 1
        else:
            skipped += 1
    print(f"Done. migrated={migrated} skipped={skipped} total={len(cfg_paths)}")


if __name__ == "__main__":
    main()
