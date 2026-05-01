"""One-shot migration: flat YAML -> method-block YAML.

Per method name, the script knows which fields belong INSIDE the method block
vs. which stay at top level. For each YAML in scripts/**/*.yaml that has a
top-level `method:` field, it rewrites the file with:
  method:
    name: <old method value>
    <method-specific fields lifted here>
  <generic fields stay at top level>

Drops these fields outright (they are dead in the new schema):
  - no_cache_only (no longer a user toggle)
  - dead model fields per method (e.g. orchestrator_model in self-refine)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml

# Per-method: which top-level fields move INTO the method block.
METHOD_BLOCK_FIELDS: dict[str, set[str]] = {
    "tts-agent": {
        "orchestrator_model", "explore_model", "integrate_model",
        "cache_dir", "no_integrate", "num_explores", "num_rollouts",
        "sampling",
    },
    "tts-agent-multi": {
        "orchestrator_model", "cache_dirs", "model_budgets",
        "exploration_effort", "num_explores",
    },
    "tts-agent-effort": {
        "orchestrator_model", "explore_model", "cache_dirs",
        "effort_budgets", "num_explores",
    },
    "self-refine": {"explore_model", "cache_dir", "num_explores"},
    "socratic-self-refine": {"explore_model", "cache_dir", "num_explores"},
    "budget-forcing": {"explore_model", "cache_dir", "num_explores"},
    "rerank": {"reward_model", "cache_dir"},
    "standalone-integrator": {"integrate_model", "cache_dir"},
}

# Always-dropped fields (irrespective of method).
ALWAYS_DROP: set[str] = {"no_cache_only"}

# All known method-related fields (across all methods). Used to detect
# fields that look method-related but aren't allowed for THIS method
# (e.g. orchestrator_model in a self-refine YAML) -- those get dropped.
METHOD_RELATED: set[str] = {
    "orchestrator_model", "explore_model", "integrate_model", "reward_model",
    "cache_dir", "cache_dirs", "model_budgets", "effort_budgets",
    "exploration_effort", "no_integrate", "num_explores", "num_rollouts",
    "sampling",
}


def migrate(yaml_path: Path, dry_run: bool = False) -> bool:
    """Rewrite one YAML in-place. Returns True if file changed."""
    with yaml_path.open() as f:
        data = yaml.safe_load(f)
    if data is None:
        return False
    assert isinstance(data, dict), f"{yaml_path}: top-level not a dict"

    if "method" not in data:
        return False  # precache YAMLs etc., not in scope
    if isinstance(data["method"], dict):
        return False  # already migrated

    method_name = data["method"]
    if method_name not in METHOD_BLOCK_FIELDS:
        print(f"  WARN: {yaml_path}: unknown method {method_name!r}, skipping")
        return False
    allowed_inside = METHOD_BLOCK_FIELDS[method_name]

    method_block: dict = {"name": method_name}
    new_top: dict = {}
    dropped: list[str] = []

    for k, v in data.items():
        if k == "method":
            continue
        if k in ALWAYS_DROP:
            dropped.append(k)
            continue
        if k in allowed_inside:
            method_block[k] = v
        else:
            new_top[k] = v

    # Detect dead method-specific fields (e.g. orchestrator_model in self-refine
    # YAMLs) -- these are top-level fields that look method-related but the spec
    # for THIS method does not allow them.
    for k in list(new_top.keys()):
        if k in METHOD_RELATED and k not in allowed_inside:
            dropped.append(k)
            del new_top[k]

    # Reassemble: benchmark first, then backend, then method, then everything else.
    out: dict = {}
    if "benchmark" in new_top:
        out["benchmark"] = new_top.pop("benchmark")
    if "backend" in new_top:
        out["backend"] = new_top.pop("backend")
    out["method"] = method_block
    for k, v in new_top.items():
        out[k] = v

    if dry_run:
        print(f"--- {yaml_path} (dry-run) ---")
        print(yaml.safe_dump(out, sort_keys=False, default_flow_style=False))
        if dropped:
            print(f"  dropped: {dropped}")
        return True

    with yaml_path.open("w") as f:
        yaml.safe_dump(out, f, sort_keys=False, default_flow_style=False)
    if dropped:
        print(f"  {yaml_path}: dropped {dropped}")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", type=Path,
        default=Path("Experiment/core_code/scripts"),
        help="Directory to walk for *.yaml files",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print without writing")
    parser.add_argument("--limit", type=int, default=0, help="Process at most N files")
    args = parser.parse_args()

    yamls = sorted(args.root.rglob("*.yaml"))
    if args.limit:
        yamls = yamls[: args.limit]

    changed = 0
    for p in yamls:
        if migrate(p, dry_run=args.dry_run):
            changed += 1
    print(f"\n{'Would migrate' if args.dry_run else 'Migrated'} {changed}/{len(yamls)} files.")


if __name__ == "__main__":
    main()
