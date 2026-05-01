"""One-shot migration: lift top-level backend fields into method.backend block.

For each YAML in scripts/**/*.yaml that has a top-level `method:` block:

- If method.name == "rerank": delete top-level backend / budget_tokens /
  effort / timeout / max_output_tokens / explore_timeout outright (rerank
  doesn't call any LLM, these were always dead).
- Otherwise: lift those fields into a nested `method.backend:` block:
    method:
      name: tts-agent
      backend:
        name: claude
        budget_tokens: 32000
        effort: low
        timeout: 1200
        max_output_tokens: null
      orchestrator_model: ...

Always drops `explore_timeout` from EvalConfig YAMLs (it was an orphan
field never read by eval.py; only PrecacheConfig uses it, and precache
YAMLs aren't touched).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml

BACKEND_FIELDS: list[str] = [
    "name",            # populated from top-level "backend"
    "budget_tokens",
    "effort",
    "timeout",
    "max_output_tokens",
]

# Top-level keys that the script consumes.
TOP_LEVEL_BACKEND_KEYS: set[str] = {
    "backend", "budget_tokens", "effort", "timeout", "max_output_tokens",
}
ALWAYS_DROP_TOP_LEVEL: set[str] = {"explore_timeout"}


def migrate(yaml_path: Path, dry_run: bool = False) -> bool:
    """Rewrite one YAML in-place. Returns True if file changed."""
    with yaml_path.open() as f:
        data = yaml.safe_load(f)
    if data is None or not isinstance(data, dict):
        return False
    if "method" not in data or not isinstance(data["method"], dict):
        return False  # precache YAML or pre-migration shape; skip

    method_block = data["method"]
    method_name = method_block.get("name")

    # Already migrated? (method.backend already present)
    if "backend" in method_block and isinstance(method_block.get("backend"), dict):
        return False

    is_rerank = method_name == "rerank"

    # Extract top-level backend-related values (if present)
    extracted: dict = {}
    for k in TOP_LEVEL_BACKEND_KEYS:
        if k in data:
            extracted[k] = data.pop(k)
    for k in ALWAYS_DROP_TOP_LEVEL:
        if k in data:
            data.pop(k)

    if is_rerank:
        # Drop entirely. Rerank doesn't have a backend concept.
        action_summary = f"dropped {sorted(extracted.keys())}"
    else:
        # Build method.backend sub-block.
        if "backend" not in extracted:
            print(f"  WARN: {yaml_path}: method={method_name!r} but no top-level backend; cannot lift")
            return False
        backend_block: dict = {"name": extracted.pop("backend")}
        # Carry over the other knobs that were present
        for k in ("budget_tokens", "effort", "timeout", "max_output_tokens"):
            if k in extracted:
                backend_block[k] = extracted.pop(k)
        # Inject backend as the second field of method block (right after name)
        new_method: dict = {}
        for k, v in method_block.items():
            new_method[k] = v
            if k == "name":
                new_method["backend"] = backend_block
        # If method block had no 'name' first, fall back to prepend
        if "backend" not in new_method:
            new_method = {"name": method_name, "backend": backend_block, **method_block}
        data["method"] = new_method
        action_summary = "lifted backend into method.backend"

    # Reassemble: benchmark / method first if present, then everything else.
    out: dict = {}
    if "benchmark" in data:
        out["benchmark"] = data.pop("benchmark")
    if "method" in data:
        out["method"] = data.pop("method")
    for k, v in data.items():
        out[k] = v

    if dry_run:
        print(f"--- {yaml_path} (dry-run, {action_summary}) ---")
        print(yaml.safe_dump(out, sort_keys=False, default_flow_style=False))
        return True

    with yaml_path.open("w") as f:
        yaml.safe_dump(out, f, sort_keys=False, default_flow_style=False)
    print(f"  {yaml_path}: {action_summary}")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", type=Path,
        default=Path("Experiment/core_code/scripts"),
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
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
