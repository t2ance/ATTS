"""One-shot YAML migration: drop integrate.cache_dir from method.integrate blocks.

Per the explore-cache-owner-refactor (2026-05-05): integrate role no longer
has a cache_dir field. Removed because cache_key=integrate_<count> only
encodes candidate count, not candidate content or integrate model identity --
integrate cache silently lied when the integrate model changed or candidate
content differed. Re-running integrate is cheap (1 call/question vs 8 explore
calls); resume already skips done records.

Usage: python tools/migrate_yaml_drop_integrate_cache_dir.py
"""
from __future__ import annotations

from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1] / "scripts"


def main() -> None:
    migrated = 0
    for yp in ROOT.rglob("*.yaml"):
        text = yp.read_text()
        try:
            data = yaml.safe_load(text) or {}
        except yaml.YAMLError as e:
            print(f"Skip {yp}: {e}")
            continue
        if not isinstance(data, dict):
            continue
        method = data.get("method")
        if not isinstance(method, dict):
            continue
        integrate = method.get("integrate")
        if not isinstance(integrate, dict):
            continue
        if "cache_dir" not in integrate:
            continue
        del integrate["cache_dir"]
        yp.write_text(yaml.dump(data, default_flow_style=False, sort_keys=False))
        print(f"Migrated: {yp}")
        migrated += 1
    print(f"\nDone. Migrated {migrated} yamls.")


if __name__ == "__main__":
    main()
