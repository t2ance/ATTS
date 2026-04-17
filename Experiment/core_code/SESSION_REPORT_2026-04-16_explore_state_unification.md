# Session Report: Explore-Loop State Unification (2026-04-16)

Primary objective of this session was to eliminate the structural bug class
behind "Explore budget: 1/8 used, 7 remaining" always rendering in GRPO
training rollouts, and to couple the rendering path to a round-trippable
parser so downstream consumers cannot silently diverge from the format.

Claim labels used below follow the project convention:
- [verified] backed by a file read, edit, command output, or smoke test
- [inferred] derived by logical deduction from verified facts
- [speculation] unverified hypothesis

---

## 1. Root cause identified

[verified] `Experiment/core_code/training/grpo/explore_tool.py` pre-fix held
per-rollout state in `self._instances[instance_id]` keyed on `uuid4()` that
`create()` generated fresh. verl's `ToolAgentLoop._call_tool()` at
`verl/experimental/agent_loop/tool_agent_loop.py:432,447` invokes
`create()` + `release()` around every tool call, not once per rollout.
Result: `call_count` reset to 0 on every call -> rendered budget line was
always "1/8 used, 7 remaining" regardless of progress; `cached_explores`
was re-shuffled per call instead of per rollout.

[verified] Confirmed via the verl source read at that path. Research agent
confirmed `agent_data.extra_fields` is the officially sanctioned
rollout-scoped container for stateful tools (verl docs
`agent_loop.rst`). verl's own example tools (`SandboxFusionTool`,
`SearchTool`, `Geo3kTool`) share the same latent bug class but do not
expose it because their state fields are never read across calls in a
semantically meaningful way.

---

## 2. Fix landed (two-layer single-source-of-truth)

[verified] The architecture now has two canonical modules, symmetric by
design. Both enforce invariants at import time via `_self_check()`:

| Layer            | Module                     | Single source             |
|------------------|----------------------------|---------------------------|
| State transition | `methods/tool_state.py`    | `advance(state)` function |
| String rendering | `methods/tool_io.py`       | `FullRenderer.render/parse` |

`CandidateRecord` is the glue between the two layers.

### 2.1 New module `methods/tool_state.py`

[verified] Defines:
- `ExploreStepState(max_explores: int, call_count: int = 0)` as
  `@dataclass(frozen=True)`. Hand-writing `state.call_count += 1` raises
  `FrozenInstanceError`.
- `advance(state)` -> `ExploreStepState`: increments `call_count` by one;
  asserts if `state.is_exhausted`.
- `_self_check()` runs at module import: transition from 0 to max,
  frozen-mutation rejection, advance-past-cap rejection, negative
  call_count rejection, zero `max_explores` rejection.

### 2.2 `methods/tool_io.py` extended

[verified] Added `FullRenderer.parse(text) -> CandidateRecord` as the
inverse of `FullRenderer.render(record)`. Regex patterns hoisted to
module-level constants `_HEADER_RE`, `_BUDGET_RE`, `_CACHE_RE`,
`_BODY_RE`. `_self_check()` extended with round-trip equality check:
`FullRenderer().parse(FullRenderer().render(rec)) == rec` for both
success and timeout records (test values chosen 2dp-safe for exact
equality).

### 2.3 `methods/base.py::SolvingState` refactored

[verified] Old fields `max_iterations: int = 5` and
`current_iteration: int = 0` deleted. New field
`explore: ExploreStepState` added (required, no default).
`create_solve_context` updated to construct
`ExploreStepState(max_explores=infra.max_iterations)`.

### 2.4 All seven call sites migrated

[verified] Every read/write of the deleted fields was replaced:

| File                                                            | Change                                                                                                        |
|-----------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------|
| `methods/tts_agent.py:75-76, 94, 121, 138, 147, 177, 227, 241, 268` | `state.current_iteration += 1; used = state.current_iteration` -> `state.explore = advance(state.explore); used = state.explore.used`; max_iterations reads swapped to `explore.max_explores` |
| `methods/tts_agent_effort.py:171, 196`                          | same pattern; module-local counters unchanged (flagged as follow-up)                                          |
| `methods/tts_agent_multi.py:179, 204`                           | same pattern; per-model counters unchanged (flagged as follow-up)                                             |
| `methods/self_refine.py:280`                                    | `+= 1` -> `state.explore = advance(state.explore)`                                                            |
| `methods/budget_forcing.py:61`                                  | `= i` -> `state.explore = advance(state.explore)`                                                             |
| `training/grpo/explore_tool.py`                                 | file rewritten; see 2.5                                                                                       |
| `training/sft/build_sft_hle_q101_300_thinking.py`               | `explore_idx += 1` -> `explore_state = advance(explore_state)`                                                |

### 2.5 `training/grpo/explore_tool.py` rewritten as stateless

[verified] Removed `self._instances` dict entirely. `create()` returns
`(str(uuid4()), ToolResponse())` with no side effects. `execute()`
reads state from `agent_data.extra_fields[_STATE_KEY]`; initializes the
bucket (shuffled `cached_explores` tuple + fresh `ExploreStepState`) on
first access per rollout. `release()` is a no-op.

[verified] Shuffle seed changed from `random.Random(hash(request_id))`
to `random.Random(request_id)`. Python's `random.Random` uses SHA-512
on string seeds, which is stable across processes; builtin `hash()` is
randomized per-process under `PYTHONHASHSEED=random`. Smoke-tested
reproducibility: same `request_id` across two fresh `Random` instances
yields identical shuffle `[5, 4, 3, 0, 2, 6, 7, 1]`.

### 2.6 `training/grpo/reward_fn.py` decoupled

[verified] `_extract_explore_tool_responses` now calls
`FullRenderer().parse(body)` instead of the hand-rolled
`re.search(r"- Answer:\s*(.*?)\s*\n- Confidence:")` pattern.
`_EXPLORE_CACHE_ID_RE` deleted. `_TOOL_RESPONSE_RE` added as module
constant. Integration test confirmed: given a synthetic
`<tool_response>...</tool_response>` block produced by
`FullRenderer.render()`, `_extract_explore_tool_responses` returns the
expected `(cache_id, answer)` tuple list; timeout candidates and
candidates without `cache_id` are skipped.

### 2.7 Simplify review applied

[verified] Three review agents ran in parallel (code reuse, code
quality, efficiency) over the diff. Fixes applied:
- Regex constants hoisted in `tool_io.py` (efficiency)
- `hash()` -> direct string seed in `explore_tool.py` (correctness)
- Docstrings in `SolvingState`, `FullRenderer.parse`, `ExploreTool`
  shortened to strip refactor-history narration (style)
- Inline comments in `explore_tool.py::execute` deleted (style)
- Tautological loop-invariant assert in `budget_forcing.py` deleted
  (simplicity)

---

## 3. Verified smoke tests

[verified] After all edits, from `/data3/peijia/dr-claw/Explain/Experiment/core_code`:

```
[1] tool_state._self_check passed at import
[2] all eval modules import OK (base, tts_agent, tts_agent_effort,
    tts_agent_multi, self_refine, budget_forcing)
[3] reward_fn imports OK
[4] SolvingState fields = [problem, explore, candidates,
    final_answer, final_reasoning, final_analysis]
    (no current_iteration, no max_iterations)
[5] advance() sequence 1/8 -> 8/8 matches expected pairs
[6] SolvingState.explore transitions via advance correctly
[7] Hand-mutation of call_count raises FrozenInstanceError
[8] FullRenderer round-trip parse(render(rec)) == rec for success + timeout
[9] reward_fn._extract_explore_tool_responses returns [(cache_id, answer)] pairs
[10] random.Random(request_id) shuffle is deterministic across two fresh instances
[11] regex constants hoisted in tool_io
```

---

## 4. Files touched this session (audit list)

[verified] Via `git diff --stat HEAD` on the listed paths:

NEW:
- `methods/tool_state.py`
- `SESSION_REPORT_2026-04-16_explore_state_unification.md` (this file)

MODIFIED (by this session):
- `methods/tool_io.py`
- `methods/base.py`
- `methods/tts_agent.py`
- `methods/tts_agent_effort.py`
- `methods/tts_agent_multi.py`
- `methods/self_refine.py`
- `methods/budget_forcing.py`
- `training/grpo/explore_tool.py`
- `training/grpo/reward_fn.py`
- `training/sft/build_sft_hle_q101_300_thinking.py`

[verified] Files listed as modified in `git status` at session start
but NOT touched by this session (pre-existing uncommitted work):
- `training/grpo/answer_tool.py`
- `training/grpo/grade_cache.py`
- `training/grpo/prepare_data_hle.py`
- parts of `training/grpo/reward_fn.py` (judge truncation fallback
  `_string_match_grade` predates this session)

---

## 5. Structural invariants now enforced

[verified] The bug class "state counter diverges from UI" is eliminated
by four mechanisms that hold at different layers:

1. **Type-system**: `ExploreStepState` is frozen. `+= 1` is a
   `FrozenInstanceError`. No back door exists.
2. **Import-time**: `tool_state._self_check()` and
   `tool_io._self_check()` run at import. Any schema drift in
   `ExploreStepState` fields or `CandidateRecord` fields that breaks
   transition or round-trip causes `ImportError` before training
   starts.
3. **Call-site uniformity**: all seven places that previously wrote a
   counter now call one function (`advance`). Any new call site that
   hand-writes a counter is detectable by grep on `current_iteration`
   (empty across the project now) or `call_count` (appears only
   inside `tool_state.py`).
4. **Render/parse duality**: `FullRenderer.parse(render(rec)) == rec`
   is asserted at import. Any format-side change in `render()` that
   is not matched in `parse()` fails the import.

---

## 6. Known open items (flagged, not fixed this session)

[verified as flagged in review; not yet addressed]:

### 6.1 Still uses hand-written `+= 1`
- `methods/tts_agent_effort.py:100` — per-effort `effort_explore_counts` dict
- `methods/tts_agent_multi.py:106` — per-model `model_explore_counts` dict

Same bug class as the fixed one; not in the current migration's scope
because they track a different dimension (per-effort, per-model) than
overall explore budget. Next-logical-step migration target.

### 6.2 Weak invariants
- `SolvingState.candidates: list[Candidate]` is mutated in place; no
  assertion binds `len(candidates)` to `state.explore.used`. If a
  future caller appends to `candidates` without calling `advance`, or
  calls `advance` without appending, they silently desync.
- `agent_data.extra_fields[_STATE_KEY]` is a mutable dict bucket. A
  second writer to the same key path would silently overwrite
  `explore_state`.

### 6.3 Leaky fields in `CandidateRecord`
- `extra_budget_text` is populated only by `tts_agent_effort` and
  `tts_agent_multi`; it threads a per-caller display suffix through a
  generic record. Architecturally cleaner if the caller appended to
  the rendered text in its own wrapper.

### 6.4 Pre-existing `reward_fn.py` no-fallback violation
[verified] `training/grpo/reward_fn.py:89-127` contains
`_judge_remote` which, on `finish_reason == "length"`, falls back to
`_string_match_grade`. This pattern violates CLAUDE.md's "no defensive
programming / no fallback" rule. Pre-existing code not touched by
this session. Fix direction: increase `max_tokens` so truncation
cannot occur, and assert on `finish_reason == "stop"`.

### 6.5 sys.path bootstrap duplicated
[verified by grep] Six files under `training/` and `methods/` all
contain the same 3-line `_CORE_CODE_DIR = Path(__file__)...; if str
not in sys.path: sys.path.insert(0, ...)` pattern. Could be extracted
into `training/_bootstrap.py`. Out of scope this session.

---

## 7. Training state at end of session

[verified by `ls checkpoints/atts-grpo/8b-sft-2gpu-bs96/` and Weave
probe]:

- Experiment: `8b-sft-2gpu-bs96` (2 GPUs, CUDA_VISIBLE_DEVICES=0,1)
- Latest saved checkpoint: `global_step_3`
- Latest Weave trace observed: `step=4`, i.e. training is mid-step-4
  to step-5 at report write time
- All step-0 through step-3 weights were trained under the buggy
  `explore_tool.py` that always rendered "1/8 used, 7 remaining";
  policy learned during those steps has seen a budget-blind signal

[verified] Training script: `training/scripts/train_grpo_vllm_8b_sft_2gpu.sh`
- `data.train_batch_size=96`, `actor.ppo_mini_batch_size=16`,
  `ppo_micro_batch_size_per_gpu=1`
- `rollout.n=8`, `rollout.temperature=1.0`, `rollout.repetition_penalty`
  unset (defaults to 1.0 per Weave schema probe)
- `max_assistant_turns=9`, `response_length=16384`,
  `max_model_len=24576`, `gpu_memory_utilization=0.55`
- `trainer.save_freq=3`, `trainer.test_freq=1`, `trainer.total_epochs=15`

[verified] Per-step wall-clock time on prior runs of this config
(v2 log, pre-fix): step ~410s. Breakdown: rollout gen ~72s,
`old_log_prob` ~48s, `ref` ~69s, `update_actor` ~187s,
`update_weights` ~5s.

---

## 8. Weave pathology scan (inconclusive)

[verified as null result, with explicit sample-composition caveat]:

User reported seeing malformed tool-call output with repeated
`"arguments": {}` keys and NSFW-token tails. Scanned 300
`HermesToolParser.extract_tool_calls` traces across three time windows
(2026-04-09 step=6 val, 2026-04-14 step=33 val, 2026-04-16 step=4
training) from project `pqin/atts-grpo`:

| Signal                                    | Count       |
|-------------------------------------------|-------------|
| NSFW word-bounded tokens                  | 0 / 300     |
| `"arguments":{},"arguments":` duplicate   | 0 / 300     |
| 4+ n-gram repetition                      | 0 / 300     |
| Non-`<|im_end|>` tail closure             | 0 / 300     |
| `exception` field non-null (50-sample)    | 0 / 50      |

[speculation, not verified] The user's reported pattern most plausibly
lives inside the `<tool_call>` JSON body (the `FunctionCall` arguments
field). The current Weave MCP returns `FunctionCall` objects as refs
that this session's tool access did not expand. `ToolAgentLoop.run`
root traces have ~100K-token outputs that the Weave server L2-drops
on fetch. Therefore the proportion of body-level malformation is NOT
measured. User elected to stop the investigation and defer. Possible
follow-up paths: (a) ask user for a specific Weave trace URL; (b)
decode `inputs.responses_ids` via Qwen tokenizer locally; (c) try to
expand `FunctionCall` refs via alternate MCP surface.

[verified] Training-side sampling config that would make pathology
possible even after an SFT-init: `temperature=1.0` with
`repetition_penalty=1.0` (no penalty). Qwen3's 150K+ vocab contains
rarely-seen tokens that can surface in degenerate sampling. No
guided-decoding / JSON-schema constraint is active under hermes mode.

---

## 9. How to resume this session in a fresh context

Read this report. Then:

1. `git diff --stat HEAD -- methods training/grpo training/sft` shows
   exactly the files this session changed.
2. `methods/tool_state.py` is the canonical new module; read it first.
3. `python -c "from methods import tool_state, tool_io"` from
   `Experiment/core_code` validates both self-checks.
4. The training run `atts-grpo/8b-sft-2gpu-bs96` may still be in
   flight or may have been restarted; check with `ps -ef | grep
   main_ppo`, then `tail tmp/grpo_8b_sft_2gpu_bs96.log`.
5. Open items in section 6 are the next-session candidates. User has
   not yet decided between (a) discarding `global_step_3` checkpoint
   and restarting from SFT init with fixed code, and (b) resuming
   from `global_step_3` under fixed code.
