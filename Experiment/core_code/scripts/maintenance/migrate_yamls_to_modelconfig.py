#!/usr/bin/env python
"""One-shot yaml migration: rewrite all yamls under scripts/**/*.yaml to the
new ModelConfig-based schema (post-modelconfig-refactor 2026-05-04).

Per-file: detect shape, apply the relevant rewrite, validate via pydantic,
write back (preserving comments/ordering via ruamel.yaml). Idempotent:
already-migrated files are skipped on the basis of marker fields
(`orchestrator_prompt` for tts-agent, presence of `explore.model` for
self-refine etc., presence of `judge.backend` for any benchmark).

Run modes:
  --path <one yaml>             rewrites the single file in place; .bak preserved
  --path <one yaml> --dry-run   prints rewritten yaml to stdout (no write)
  --bulk                        walks scripts/**/*.yaml; rewrites every match in
                                place. On the first validation failure rolls
                                back from .bak and exits non-zero.
"""
from __future__ import annotations

import argparse
import io
import shutil
import sys
import traceback
from pathlib import Path

from ruamel.yaml import YAML

CORE_CODE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CORE_CODE))

from eval import EvalConfig                        # noqa: E402
from precache_explores import PrecacheConfig       # noqa: E402

YAML_RW = YAML(typ="rt")
YAML_RW.preserve_quotes = True
YAML_RW.indent(mapping=2, sequence=4, offset=2)

# Old multi-model alias-to-canonical-model-id mapping (from the deleted
# methods/tts_agent_multi.py:MODEL_ALIASES).
_MODEL_ALIASES = {
    "haiku":  "claude-haiku-4-5-20251001",
    "sonnet": "claude-sonnet-4-6",
    "opus":   "claude-opus-4-6",
}

# Backend-config keys carried over to the new ModelConfig.
_BACKEND_PASSTHROUGH = (
    "budget_tokens", "effort", "timeout", "max_output_tokens",
)


def _build_model_block(old_backend: dict, model_name: str) -> dict:
    """Convert old method.backend dict + bare model string into a new ModelConfig dict."""
    out = {"backend": old_backend["name"], "model": model_name}
    for k in _BACKEND_PASSTHROUGH:
        if k in old_backend:
            out[k] = old_backend[k]
    if old_backend.get("name") == "openrouter":
        if "provider_order" in old_backend and old_backend["provider_order"] is not None:
            out["openrouter_provider_order"] = old_backend["provider_order"]
        if "provider_allow_fallbacks" in old_backend:
            out["openrouter_provider_allow_fallbacks"] = old_backend["provider_allow_fallbacks"]
    return out


def _migrate_judge(bench: dict) -> None:
    """In-place: rename judge.name -> judge.backend, judge.sampling -> judge.vllm_sampling."""
    judge = bench.get("judge")
    if judge is None or "backend" in judge:
        return
    judge["backend"] = judge.pop("name")
    if judge["backend"] == "vllm" and "sampling" in judge:
        judge["vllm_sampling"] = judge.pop("sampling")


def _migrate_tts_agent(method: dict) -> dict:
    backend = method["backend"]
    model_block_orchestrator = _build_model_block(backend, method["orchestrator_model"])
    if method.get("orchestrator_effort") is not None:
        model_block_orchestrator["effort"] = method["orchestrator_effort"]
    explore_variant = {
        "label": "default",
        "model": _build_model_block(backend, method["explore_model"]),
        "cache_dir": method["cache_dir"],
        "num_explores": method.get("num_explores", 8),
    }
    new = {
        "name": "tts-agent",
        "orchestrator_prompt": "single",
        "orchestrator": model_block_orchestrator,
        "explore": [explore_variant],
    }
    if not method.get("no_integrate", False):
        new["integrate"] = {
            "model": _build_model_block(backend, method["integrate_model"]),
            "cache_dir": method["cache_dir"],
        }
    if method.get("num_rollouts", 1) != 1:
        new["num_rollouts"] = method["num_rollouts"]
    if method.get("sampling") is not None:
        new["orchestrator"]["vllm_sampling"] = method["sampling"]
    return new


def _migrate_tts_agent_multi(method: dict) -> dict:
    backend = method["backend"]
    cache_dirs = method["cache_dirs"]
    model_budgets = method["model_budgets"]
    # exploration_effort (old TTSAgentMultiSpec): single effort string applied
    # uniformly to all three explorer variants. None means use the per-variant
    # default (low). After migration, every variant carries its own
    # ModelConfig.effort, so this propagates onto each variant individually.
    exploration_effort = method.get("exploration_effort")
    explore = []
    for alias in ("haiku", "sonnet", "opus"):
        if alias not in cache_dirs:
            continue
        model_block = _build_model_block(backend, _MODEL_ALIASES[alias])
        if exploration_effort is not None:
            model_block["effort"] = exploration_effort
        explore.append({
            "label": alias,
            "model": model_block,
            "cache_dir": cache_dirs[alias],
            "num_explores": model_budgets[alias],
        })
    return {
        "name": "tts-agent",
        "orchestrator_prompt": "multi_model",
        "orchestrator": _build_model_block(backend, method["orchestrator_model"]),
        "explore": explore,
    }


def _migrate_tts_agent_effort(method: dict) -> dict:
    backend = method["backend"]
    cache_dirs = method["cache_dirs"]
    effort_budgets = method["effort_budgets"]
    explore = []
    for level in ("low", "medium", "high"):
        if level not in cache_dirs:
            continue
        model_block = _build_model_block(backend, method["explore_model"])
        model_block["effort"] = level
        explore.append({
            "label": level,
            "model": model_block,
            "cache_dir": cache_dirs[level],
            "num_explores": effort_budgets[level],
        })
    return {
        "name": "tts-agent",
        "orchestrator_prompt": "effort",
        "orchestrator": _build_model_block(backend, method["orchestrator_model"]),
        "explore": explore,
    }


def _migrate_explore_only(method: dict, name: str) -> dict:
    return {
        "name": name,
        "explore": {
            "label": "default",
            "model": _build_model_block(method["backend"], method["explore_model"]),
            "cache_dir": method["cache_dir"],
            "num_explores": method.get("num_explores", 8),
        },
    }


def _migrate_standalone_integrator(method: dict) -> dict:
    return {
        "name": "standalone-integrator",
        "integrate": {
            "model": _build_model_block(method["backend"], method["integrate_model"]),
            "cache_dir": method["cache_dir"],
        },
        "num_explores": method.get("num_explores", 8),
    }


def _migrate_method(method: dict) -> dict:
    n = method["name"]
    if n == "tts-agent":
        return _migrate_tts_agent(method)
    if n == "tts-agent-multi":
        return _migrate_tts_agent_multi(method)
    if n == "tts-agent-effort":
        return _migrate_tts_agent_effort(method)
    if n in ("self-refine", "socratic-self-refine", "budget-forcing"):
        return _migrate_explore_only(method, n)
    if n == "standalone-integrator":
        return _migrate_standalone_integrator(method)
    if n == "rerank":
        return method
    raise ValueError(f"unknown method.name: {n!r}")


def _looks_migrated_eval(data: dict) -> bool:
    """Returns True iff BOTH the method block and the benchmark.judge block
    are already in the post-refactor shape. Rerank's method block is
    structurally unchanged across the refactor, but its benchmark.judge still
    needs the name->backend rename, so we cannot short-circuit on rerank."""
    method = data.get("method", {})
    bench = data.get("benchmark", {})
    judge = bench.get("judge") if isinstance(bench, dict) else None
    judge_migrated = judge is None or "backend" in judge

    n = method.get("name")
    if n == "tts-agent":
        method_migrated = "orchestrator_prompt" in method
    elif n in ("self-refine", "socratic-self-refine", "budget-forcing"):
        explore = method.get("explore", {})
        method_migrated = isinstance(explore, dict) and "model" in explore
    elif n == "standalone-integrator":
        method_migrated = "integrate" in method
    elif n == "rerank":
        method_migrated = True
    else:
        method_migrated = False

    return method_migrated and judge_migrated


def _migrate_precache(data: dict) -> dict:
    backend_name = data["backend"]
    model_block = {
        "backend": backend_name,
        "model": data["explore_model"],
        "budget_tokens": data.get("budget_tokens", 32000),
        "effort": data.get("effort", "low"),
        "timeout": data.get("explore_timeout", 1200.0),
    }
    if backend_name == "vllm" and data.get("sampling") is not None:
        model_block["vllm_sampling"] = data["sampling"]
    if backend_name == "openrouter":
        if data.get("provider_order") is not None:
            model_block["openrouter_provider_order"] = data["provider_order"]
        if "provider_allow_fallbacks" in data:
            model_block["openrouter_provider_allow_fallbacks"] = data["provider_allow_fallbacks"]
    explore = {
        "label": "default",
        "model": model_block,
        "cache_dir": data["cache_dir"],
        "num_explores": data.get("num_explores", 8),
    }
    new = {
        "benchmark": data["benchmark"],
        "explore": explore,
    }
    for k in ("num_workers", "num", "skip", "seed", "shuffle"):
        if k in data:
            new[k] = data[k]
    return new


def _looks_migrated_precache(data: dict) -> bool:
    explore = data.get("explore")
    return isinstance(explore, dict) and "model" in explore and "label" in explore


def _is_precache_shape(data: dict) -> bool:
    """Shape-based detection (filename can lie -- see hle_gemma4_..._smoke_v2.yaml).
    A precache yaml has neither a top-level `method:` block (eval shape) nor a
    post-migration `explore:` ExploreVariant; instead it carries pre-migration
    flat fields like `backend:`, `explore_model:`, `cache_dir:`. After
    migration it has only `explore:` + `benchmark:` at top level.
    """
    if "method" in data:
        return False
    if _looks_migrated_precache(data):
        return True
    return "explore_model" in data or "backend" in data


def _migrate_one(path: Path, *, dry_run: bool) -> str:
    """Returns one of: 'migrated', 'skipped'."""
    raw = YAML_RW.load(path.read_text(encoding="utf-8"))
    if raw is None:
        return "skipped"
    is_precache = _is_precache_shape(raw)
    if is_precache:
        if _looks_migrated_precache(raw):
            return "skipped"
        new = _migrate_precache(raw)
        if isinstance(new.get("benchmark"), dict):
            _migrate_judge(new["benchmark"])
    else:
        method = raw.get("method", {})
        if not method:
            return "skipped"
        if _looks_migrated_eval(raw):
            return "skipped"
        new_method = _migrate_method(method)
        new = dict(raw)
        new["method"] = new_method
        if isinstance(new.get("benchmark"), dict):
            _migrate_judge(new["benchmark"])

    schema = PrecacheConfig if is_precache else EvalConfig
    schema.model_validate(new)

    if dry_run:
        buf = io.StringIO()
        YAML_RW.dump(new, buf)
        print(buf.getvalue())
        return "migrated"

    backup = path.with_suffix(path.suffix + ".bak")
    shutil.copy2(path, backup)
    with path.open("w", encoding="utf-8") as f:
        YAML_RW.dump(new, f)
    return "migrated"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--path", type=Path, help="Single yaml")
    ap.add_argument("--bulk", action="store_true",
                    help="Walk scripts/**/*.yaml under Experiment/core_code/")
    args = ap.parse_args()

    if args.path:
        verdict = _migrate_one(args.path, dry_run=args.dry_run)
        print(f"{verdict}: {args.path}", file=sys.stderr)
        return

    assert args.bulk, "use --path for single-file or --bulk for tree-walk"
    base = Path(__file__).resolve().parents[1]
    paths = sorted(base.glob("**/*.yaml"))
    counts = {"migrated": 0, "skipped": 0}
    for p in paths:
        try:
            verdict = _migrate_one(p, dry_run=False)
        except Exception as e:
            backup = p.with_suffix(p.suffix + ".bak")
            if backup.exists():
                shutil.copy2(backup, p)
                backup.unlink()
            print(f"FAIL {p}: {e}", file=sys.stderr)
            traceback.print_exc()
            sys.exit(1)
        counts[verdict] = counts.get(verdict, 0) + 1
        print(f"{verdict}: {p}")
    print(f"\nDone. {counts}")


if __name__ == "__main__":
    main()
