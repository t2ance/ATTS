"""Small-scope unit test for eval._grade_with_cache new bundle layout (T7).

Mocks benchmark.grade() so no real network calls. Verifies:
- Cache miss writes 5-file bundle into grade_dir/judges/<label>/
- Cache hit returns the cached verdict with cost=0 and does NOT call benchmark.grade
- judge_spec=None benchmarks (LCB/GPQA/AIME) short-circuit and write nothing under judges/
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from unittest.mock import patch, AsyncMock

_CORE_CODE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_CORE_CODE_DIR))

from eval import _grade_with_cache
from benchmarks.hle import HLEBenchmark
from benchmarks.gpqa import GPQABenchmark
from cache_types import JudgeOutcome


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


_HAIKU_SPEC = {"backend": "claude", "model": "claude-haiku-4-5-20251001"}


def test_cache_miss_writes_bundle_into_judges_label_dir(tmp_path):
    bench = HLEBenchmark(judge_spec=_HAIKU_SPEC)

    async def fake_grade(self, predicted, gold, question, row, backend):
        return JudgeOutcome(
            is_correct=True, cost_usd=0.012, judge_spec_snapshot=_HAIKU_SPEC,
            input_md="judge prompt", output_md="judge raw output",
            result_dict={"correct": True},
        )

    with patch.object(HLEBenchmark, "grade", new=fake_grade):
        is_correct, judge_cost = _run(_grade_with_cache(
            bench, "x", "y", "q", {"answer_type": "exactMatch"},
            backend="claude", grade_dir=tmp_path,
        ))

    assert is_correct is True
    assert judge_cost == 0.012
    bundle = tmp_path / "judges" / "claude__claude-haiku-4-5-20251001"
    assert (bundle / "config.json").exists()
    assert (bundle / "grade.json").exists()
    assert (bundle / "input.md").exists()
    assert (bundle / "output.md").exists()
    assert (bundle / "result.json").exists()
    grade = json.loads((bundle / "grade.json").read_text())
    assert grade["judge_spec"] == _HAIKU_SPEC
    assert grade["is_correct"] is True


def test_cache_hit_returns_zero_cost_without_calling_judge(tmp_path):
    """Pre-populate the bundle; _grade_with_cache must return cached verdict."""
    bundle = tmp_path / "judges" / "claude__claude-haiku-4-5-20251001"
    bundle.mkdir(parents=True)
    (bundle / "config.json").write_text(json.dumps(_HAIKU_SPEC, sort_keys=True))
    (bundle / "grade.json").write_text(json.dumps({
        "judge_spec": _HAIKU_SPEC,
        "is_correct": False,
        "predicted": "x",
        "gold": "y",
        "judge_cost_usd": 0.012,
    }))
    bench = HLEBenchmark(judge_spec=_HAIKU_SPEC)

    grade_called = AsyncMock()
    with patch.object(HLEBenchmark, "grade", new=grade_called):
        is_correct, judge_cost = _run(_grade_with_cache(
            bench, "x", "y", "q", {"answer_type": "exactMatch"},
            backend="claude", grade_dir=tmp_path,
        ))

    assert is_correct is False
    assert judge_cost == 0.0
    grade_called.assert_not_called()


def test_no_judge_benchmark_skips_judges_dir(tmp_path):
    """GPQA judge_spec=None: no bundle should appear under judges/."""
    bench = GPQABenchmark()  # judge_spec defaults to None

    is_correct, judge_cost = _run(_grade_with_cache(
        bench, "C", "C", "q", {},
        backend="claude", grade_dir=tmp_path,
    ))

    assert is_correct is True
    assert judge_cost == 0.0
    assert not (tmp_path / "judges").exists(), \
        "no-judge benchmarks must not create judges/ directory"
