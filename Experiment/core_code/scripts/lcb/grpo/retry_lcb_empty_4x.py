#!/usr/bin/env python3
"""LCB EMPTY-row retry driver: filter LCB to 12 specific qids and re-run with
4x max_tokens (131072) + per-client timeout 4800s. Created 2026-05-03 for
todo_lcb_empty_retry_4x.md Phase 2.

This script does three things before handing control to eval.main():

  1. Routes the orchestrator_model `qwen36-35b-a3b-fp8` to the GPU-0 vLLM
     serve at port 8002 (default routing in backends/vllm.py points to port
     8000 = gemma4 currently). Does NOT permanently edit vllm.py — the route
     is added at runtime so this driver is the only consumer.

  2. Raises the AsyncOpenAI client timeout from 1800s -> 4800s. The retry's
     128K decode at ~50 tok/s peaks at ~43 min on long-tail problems; 1800s
     is too tight and would replicate the original 64K retry's
     openai.APITimeoutError crash.

  3. Monkey-patches LCBBenchmark.load_dataset to return only the 12 qids in
     tmp/lcb_empty_retry_qids.json (4 confirmed savable + 8 cache-not-graded).
     Asserts the filtered count equals 12.

After patches, hands off to eval.main() with the empty_retry yaml.
"""

import json
import os
import sys
from pathlib import Path

CORE = Path("/data3/peijia/dr-claw/Explain/Experiment/core_code")
sys.path.insert(0, str(CORE))
os.chdir(CORE)

# 0. Repair the lcb_runner editable install that points to a now-deleted
# source dir (/data1/peijia/projects/EXPLaIN/LiveCodeBench). The full source
# still exists at code_references/LiveCodeBench/lcb_runner — repoint the
# editable finder dicts before any import of lcb_runner happens. Without
# this, grade_code (benchmarks/grader.py:248) crashes with
#   ModuleNotFoundError: No module named 'lcb_runner.evaluation'
# because the namespace finder resolves lcb_runner to a non-existent path.
# Site-packages is read-only via Edit tool, so monkey-patch instead.
import __editable___livecodebench_0_1_0_finder as _lcb_finder
_LCB_RUNNER_GOOD = "/data3/peijia/dr-claw/Explain/Experiment/code_references/LiveCodeBench/lcb_runner"
_lcb_finder.MAPPING["lcb_runner"] = _LCB_RUNNER_GOOD
_lcb_finder.NAMESPACES["lcb_runner"] = [_LCB_RUNNER_GOOD]

# 1. Route qwen36-35b-a3b-fp8 -> GPU 0 serve (port 8002), with 4800s timeout.
import backends.vllm as _vllm
_vllm.MODEL_TO_BASE_URL["qwen36-35b-a3b-fp8"] = "http://localhost:8002/v1"

from openai import AsyncOpenAI

def _patched_get_client(model=None):
    base_url = (
        _vllm.MODEL_TO_BASE_URL.get(model, _vllm.DEFAULT_VLLM_BASE_URL)
        if model
        else _vllm.DEFAULT_VLLM_BASE_URL
    )
    if base_url not in _vllm._clients:
        # 4800s = 80 min: 4x the existing 1200s explore_timeout, matches the
        # max_tokens=131072 4x bump. Long-tail LCB problems at 128K decode
        # have wall time ~43 min; 1800s default would crash.
        _vllm._clients[base_url] = AsyncOpenAI(
            base_url=base_url, api_key="not-needed", timeout=4800.0
        )
    return _vllm._clients[base_url]

_vllm._get_client = _patched_get_client

# 2. Filter LCB to the savable qids. The set was reduced from 12 -> 4 on 2026-05-03
# after auditing cache/lcb/sonnet/<qid>/explore_<n>/result.json: 8 of the 12 had
# all 8 explore files marked timed_out=True (Sonnet explorer aborted before
# emitting any candidate), so the orchestrator can never see usable candidates
# regardless of its decode budget. Only the 4 with >=1 non-timed-out cache file
# can possibly benefit from a 4x orchestrator-budget retry.
qid_set = set(json.load(open("tmp/lcb_empty_retry_qids.json")))
assert len(qid_set) == 4, f"Expected 4 qids in tmp/lcb_empty_retry_qids.json; got {len(qid_set)}"
print(f"[retry-driver] Filtering LCB to {len(qid_set)} qids: {sorted(qid_set)}", flush=True)

import benchmarks.lcb as _lcb_mod

# Find the actual benchmark class (file uses class LCBBenchmark or similar).
_bench_cls = getattr(_lcb_mod, "LCBBenchmark", None) or getattr(_lcb_mod, "LCB", None)
if _bench_cls is None:
    for _name in dir(_lcb_mod):
        _obj = getattr(_lcb_mod, _name)
        if isinstance(_obj, type) and "LCB" in _name and hasattr(_obj, "load_dataset"):
            _bench_cls = _obj
            break
assert _bench_cls is not None, "Could not locate LCB benchmark class in benchmarks.lcb"

_orig_load = _bench_cls.load_dataset

def _patched_load(self):
    rows = _orig_load(self)
    filtered = [r for r in rows if self.get_id(r) in qid_set]
    print(
        f"[retry-driver] LCB.load_dataset: {len(rows)} total -> {len(filtered)} after qid filter",
        flush=True,
    )
    assert len(filtered) == 4, (
        f"Expected exactly 4 filtered LCB rows; got {len(filtered)}. "
        f"Missing qids: {qid_set - {self.get_id(r) for r in filtered}}"
    )
    return filtered

_bench_cls.load_dataset = _patched_load

# 3. Hand off to eval.main() with the empty_retry yaml.
sys.argv = [
    "eval.py",
    "--config",
    "scripts/lcb/grpo/lcb_qwen36_35b_a3b_empty_retry.yaml",
]

from eval import main as _eval_main

_eval_main()
