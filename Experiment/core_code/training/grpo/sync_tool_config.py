"""Keep `training/grpo/tool_config.yaml` in lockstep with `EXPLORE_SCHEMA`.

`tool_config.yaml` is the artifact verl loads at training start. The canonical
StructuredOutput payload schema lives in Python at
`benchmarks/base.py:EXPLORE_SCHEMA`. Two places -> silent drift risk (field
names, descriptions, required list can diverge). This module does two things:

- `verify_tool_config_matches_canonical(yaml_path, explore_schema)`:
  loads the yaml, finds the StructuredOutput tool, compares its
  `function.parameters` dict against `explore_schema`, and raises
  `AssertionError` with a concrete diff on any drift.
- CLI: `python -m training.grpo.sync_tool_config`           -> verify
       `python -m training.grpo.sync_tool_config --write`   -> regenerate
  The verifier default is intentionally verify-only. Rewrite is opt-in to
  keep the yaml a stable, reviewable artifact.

The verifier is wired into `training/grpo/prepare_data_hle.py` so every
training data rebuild fails loud if the yaml has drifted.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml

_CORE_CODE_DIR = Path(__file__).resolve().parent.parent.parent
if str(_CORE_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CORE_CODE_DIR))

from benchmarks.base import EXPLORE_SCHEMA  # noqa: E402

TOOL_CONFIG_PATH = Path(__file__).resolve().parent / "tool_config.yaml"
STRUCTURED_OUTPUT_NAME = "StructuredOutput"


def _load_yaml(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _find_tool(cfg: dict, name: str) -> dict:
    for tool in cfg.get("tools", []):
        schema = tool.get("tool_schema", {})
        if schema.get("function", {}).get("name") == name:
            return tool
    raise AssertionError(
        f"tool_config.yaml has no tool named {name!r}; found: "
        f"{[t.get('tool_schema', {}).get('function', {}).get('name') for t in cfg.get('tools', [])]}"
    )


def verify_tool_config_matches_canonical(
    yaml_path: Path = TOOL_CONFIG_PATH,
    explore_schema: dict[str, Any] = EXPLORE_SCHEMA,
) -> None:
    """Assert the StructuredOutput parameters dict matches `explore_schema`.

    Raises AssertionError with a diff payload on any drift. Compares the
    full `parameters` dict structurally (field names, types, descriptions,
    required list, additionalProperties).
    """
    cfg = _load_yaml(yaml_path)
    tool = _find_tool(cfg, STRUCTURED_OUTPUT_NAME)
    yaml_params = tool["tool_schema"]["function"]["parameters"]
    if yaml_params != explore_schema:
        diff = {
            "yaml": yaml_params,
            "canonical": explore_schema,
        }
        raise AssertionError(
            f"tool_config.yaml StructuredOutput.parameters has drifted from "
            f"EXPLORE_SCHEMA. Run `python -m training.grpo.sync_tool_config --write` "
            f"to regenerate.\n\nDiff:\n{json.dumps(diff, indent=2)}"
        )


def rewrite_tool_config(
    yaml_path: Path = TOOL_CONFIG_PATH,
    explore_schema: dict[str, Any] = EXPLORE_SCHEMA,
) -> None:
    """Overwrite the StructuredOutput parameters block with `explore_schema`.

    Loads yaml, replaces the StructuredOutput tool's function.parameters dict,
    and dumps back. PyYAML's default style loses the original formatting but
    produces valid, deterministic yaml.
    """
    cfg = _load_yaml(yaml_path)
    tool = _find_tool(cfg, STRUCTURED_OUTPUT_NAME)
    tool["tool_schema"]["function"]["parameters"] = explore_schema
    with open(yaml_path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, default_flow_style=False, allow_unicode=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--write",
        action="store_true",
        help="Regenerate tool_config.yaml from EXPLORE_SCHEMA instead of verifying.",
    )
    args = parser.parse_args()
    if args.write:
        rewrite_tool_config()
        print(f"Rewrote {TOOL_CONFIG_PATH} from EXPLORE_SCHEMA.")
        verify_tool_config_matches_canonical()
        print("Post-write verify: OK")
    else:
        verify_tool_config_matches_canonical()
        print(f"OK: {TOOL_CONFIG_PATH} StructuredOutput.parameters matches EXPLORE_SCHEMA.")


if __name__ == "__main__":
    main()
