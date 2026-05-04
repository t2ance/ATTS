"""Verify num_explores plumbing for standalone-integrator (post-modelconfig refactor).

- spec accepts num_explores with default 8
- spec accepts override (e.g. 4)
- spec rejects unknown fields (Pydantic extra=forbid)
- registry's partialed solve carries spec
- load_cached_candidates + slicing produces a kept-cost equal to sum of first N
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

_CORE_CODE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_CORE_CODE_DIR))

import pytest

from methods.specs import StandaloneIntegratorSpec
from methods.registry import METHODS


def _make_spec(num_explores: int | None = None) -> dict:
    spec = {
        "name": "standalone-integrator",
        "integrate": {
            "model": {"backend": "claude", "model": "claude-sonnet-4-6"},
            "cache_dir": "/tmp/fake-cache",
        },
    }
    if num_explores is not None:
        spec["num_explores"] = num_explores
    return spec


def test_spec_default_is_eight():
    parsed = StandaloneIntegratorSpec.model_validate(_make_spec())
    assert parsed.num_explores == 8
    assert parsed.integrate.model.model == "claude-sonnet-4-6"
    assert parsed.integrate.cache_dir == Path("/tmp/fake-cache")


def test_spec_accepts_override():
    parsed = StandaloneIntegratorSpec.model_validate(_make_spec(num_explores=4))
    assert parsed.num_explores == 4


def test_spec_rejects_unknown_field():
    bad = _make_spec()
    bad["explore_model"] = "claude-sonnet-4-6"  # not a standalone-integrator field
    with pytest.raises(Exception):
        StandaloneIntegratorSpec.model_validate(bad)


def test_spec_rejects_old_flat_shape():
    """Old yamls had backend / integrate_model / cache_dir at the method level.
    Migration must transform them; the post-refactor schema rejects them."""
    bad = {
        "name": "standalone-integrator",
        "backend": {"name": "claude"},
        "integrate_model": "claude-sonnet-4-6",
        "cache_dir": "/tmp/fake-cache",
    }
    with pytest.raises(Exception):
        StandaloneIntegratorSpec.model_validate(bad)


def test_registry_partial_carries_spec():
    parsed = StandaloneIntegratorSpec.model_validate(_make_spec(num_explores=4))
    method_cls = METHODS["standalone-integrator"]
    partial = method_cls().build_solve_fn(parsed)
    # build_solve_fn returns functools.partial(solve, spec=parsed); the
    # spec object is the single source of truth carrying integrate role +
    # num_explores into the solver.
    assert partial.keywords["spec"] is parsed
    assert partial.keywords["spec"].num_explores == 4


def test_load_and_truncate_with_fixture_cache(tmp_path: Path):
    """Build a fake 6-explore cache; load_cached_candidates returns 6;
    after slicing [:4] the kept cost equals the sum of first 4 cost_usd."""
    from methods.base import load_cached_candidates

    class _FakeBenchmark:
        @staticmethod
        def get_answer_from_explore(d):
            return d["answer"]

    qid = "qfake1"
    qdir = tmp_path / qid
    costs = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60]
    for i, c in enumerate(costs, 1):
        ed = qdir / f"explore_{i}"
        ed.mkdir(parents=True)
        (ed / "result.json").write_text(json.dumps({
            "answer": f"a{i}", "approach": "", "reasoning": "",
            "confidence": 0.5, "cost_usd": c, "timed_out": False,
        }))

    candidates, total = load_cached_candidates(tmp_path, qid, _FakeBenchmark())
    assert len(candidates) == 6
    assert total == pytest.approx(sum(costs))

    n = 4
    kept = candidates[:n]
    kept_cost = sum(c.cost_usd for c in kept)
    assert len(kept) == 4
    assert kept_cost == pytest.approx(sum(costs[:4]))
