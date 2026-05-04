# TODO: Gemma-4 thinking_token_budget validation — recover HLE timeout rate

> **STATUS: PAUSED 2026-05-03 evening** — Gemma path deferred per user directive "先放弃吧，暂时不考虑 gemma 了".
>
> **Why this todo is also paused**: `thinking_token_budget` would only matter if the upstream Gemma-4 thinking pipeline worked end-to-end. The 2026-05-03 HLE 16-explore smoke (post Layer-1+Layer-2 fix) showed only 3/16 explores actually carried `<think>` blocks; 13/16 failed via two distinct failure modes (mid-thinking timeout + empty-reasoning skip). Until the upstream channel-marker bugs are fixed, validating `thinking_token_budget` is meaningless — the parameter only kicks in after thinking emits properly.
>
> **Resume condition**: vllm#38855 closed upstream + fresh ≥14/16 smoke pass on the parent Gemma todo (`todo_inf_gemma_A_exp_orch.md` Phase 5 item 14) BEFORE attempting any item below. Until then **all items below are frozen**.

## What this is

Empirical validation of vllm 0.20.0's `thinking_token_budget` parameter (PR #20859, merged 2026-03-24) on Gemma-4-26B-A4B-it served with `--structured-outputs-config '{"backend":"xgrammar","reasoning_parser":"gemma4"}'`. The PR explicitly tested DeepSeek-R1 / Qwen3 / Nemotron-3; gemma4 reasoning parser is registered in `vllm/reasoning/gemma4_reasoning_parser.py` but not in the PR's test matrix — so empirical verification is required.

**Why this matters**: prior HLE precache attempt on Gemma-4 hit **77.8% timeout rate** (249/324 result.json with `timed_out=true` at the kill point). Apples-to-apples Qwen3.6-35B-A3B-FP8 hit **4.4% timeout rate** on the same 100Q × 8 explores. Token-distribution analysis showed Gemma's median output is **5.75× longer than Qwen's** (11,090 vs 1,929 tokens), and p99 hits 25K. Root cause is Gemma's natural thinking length, not vllm throughput. `thinking_token_budget` directly addresses this by force-injecting `<|/think|>` end-of-thinking token via logits processor when the cap is reached.

**User-approved path**: this is NOT in the user's forbidden-fallback list (the 5 forbidden are: disable thinking entirely / drop the row / restructure prompt / multi-step JSON / shrink max_tokens). `thinking_token_budget` is a graduated control on thinking length, not a binary on/off — it's the closest analog to "reasoning_effort=low" for Gemma which doesn't exist natively.

## Output target

Decide a production thinking_token_budget value (or reject the approach if it breaks JSON quality), then re-run HLE precache with the chosen value to feed `tab:backbone-ablation` Gemma row in `Publication/paper/main.tex`. Downstream ripple: GPQA / LCB precache YAMLs adopt the same budget setting if HLE validation passes.

## Discipline

Every event below has explicit Gates with checkboxes. An event flips from `☐` to `✓` only after **all** its Gates pass AND each Gate's `Evidence ·` line is filled with the actual measurement. Soft-Gates require Justification with concrete qid + line-number evidence — never narrative claims like "looks fine". No silent skipping, no marking done before evidence is recorded.

## State at start of this todo (2026-05-03 00:57)

- vllm 0.20.0 in `grpo_vllm` conda env (verified `pip show vllm | grep Version` returned `Version: 0.20.0`)
- vllm serve still alive: PID 4068133 (parent) / 4068576 (child) on GPU 0,1,2,3 with `gemma4-26b-a4b-it` alias, `reasoning_parser=gemma4` set
- Existing Gemma HLE cache at `../analysis/cache/hle/gemma4_26b_a4b_it/gold/`: **75 useful + 249 timed_out = 324 result.json**
- precache process killed (PID 1146021 stopped); monitor `banh3eudt` stopped
- companion todos: `todo_inf_gemma_A_exp_orch.md` (Item 14 ✓ smoke; Item 15 HLE precache currently held pending this validation outcome)

## Resume / restart procedure

`thinking_token_budget` is a per-request sampling parameter, NOT a server-side flag. So:
- vllm serve does NOT need restart — the existing serve already has `--structured-outputs-config '{"backend":"xgrammar","reasoning_parser":"gemma4"}'` which is what `thinking_token_budget` requires (the parser is what detects `<|think|>...<|/think|>` boundaries)
- Per-request: client passes `thinking_token_budget` via `sampling.extra_body` in the OpenAI chat completion call

If a smoke run dies mid-way: re-run the same `precache_explores.py` command — `cache_dir` auto-skips already-cached `(qid, explore_idx)` pairs. Verify banner `Tasks: K to run, J already cached` with J>0.

## Risk register (known failure modes for this validation)

| # | Failure | Root cause | Defense in this todo |
|---|---|---|---|
| R1 | `thinking_token_budget` not honored by gemma4 parser → output_tokens still hit 60K | gemma4 parser may not properly emit `<|/think|>` token id needed by logits processor | Phase 3 item 03 G2 (assert max(output_tokens) ≤ budget × 1.1) |
| R2 | Budget enforced but post-think output is empty / parse_failed → JSON quality collapses | logits processor forces end-of-think too aggressively; model has no time to produce valid JSON | Phase 3 item 03 G3 + Phase 4 item 04 G3 (parse_failed rate ≤ 10% of non-timed_out) |
| R3 | `extra_body` field name mismatch — vllm OpenAI server may use snake_case `thinking_token_budget` while reqs send camelCase or alternate name | naming inconsistency between PR description, code, and OpenAI compat layer | Phase 2 item 02 G1 (single curl with both field-name candidates; assert one accepted, one rejected/ignored) |
| R4 | SamplingConfig schema rejects extra_body or doesn't pass-through to vllm backend | `methods/specs.py:SamplingConfig` has `extra="forbid"` and may not list `thinking_token_budget` | Phase 2 item 02 G2 (smoke yaml parses without ValidationError) |
| R5 | Throughput unchanged — fewer tokens generated per request but same per-token wall time → timeout rate doesn't drop | thinking_token_budget reduces token volume but if compute throughput is the bottleneck, won't help (Falsified: Qwen comparison shows tokens are the bottleneck, not throughput) | Phase 4 item 04 G2 (timed_out rate ≤ 30% — direct measurement) |
| R6 | Cache contamination — old timed_out result.json files (without budget) get conflated with new budgeted runs | `cache_dir` skip-check is file-existence only, doesn't compare config | Phase 5 item 06 (delete only `timed_out=true` result.json files; keep 75 useful) |

## Co-monitor — log paths

| Phase | Run log | Power log |
|---|---|---|
| 03 budget smoke (num=2, budget=15K) | `tmp/precache_hle_gemma_budget15k_smoke.log` | (use vllm metrics endpoint instead of nvidia-smi log) |
| 04 budget scan (num=10, budgets {10K, 15K, 20K}) | `tmp/precache_hle_gemma_budget_scan.log` | n/a |
| 06 production HLE precache (resume) | `tmp/precache_hle_gemma_budgeted_full.log` | n/a |

User can `tail -f /data3/peijia/dr-claw/Explain/Experiment/core_code/<path>` for any of these.

## Phase 1 — Pre-flight + feature confirmation [1/1 ✓]

01 ✓ Confirm `thinking_token_budget` is callable via the existing serve
   ├ G1 ✓ Gate · `grep thinking_token_budget /home/peijia/miniconda3/envs/grpo_vllm/lib/python3.11/site-packages/vllm/sampling_params.py` returns ≥1 line
   │      Evidence · line 302 `thinking_token_budget: int | None = None`; also lines 327/368/898 (constructor + repr) → field present in installed wheel
   ├ G2 ✓ Gate · `ls vllm/reasoning/gemma4_reasoning_parser.py` exists in installed package
   │      Evidence · `/home/peijia/miniconda3/envs/grpo_vllm/lib/python3.11/site-packages/vllm/reasoning/gemma4_reasoning_parser.py` resolves
   ├ G3 ✓ Gate · vllm serve still alive: `curl http://localhost:8000/v1/models` returns 200 with `gemma4-26b-a4b-it` alias; serve cmdline contains `--structured-outputs-config` with `reasoning_parser=gemma4`
   │      Evidence · `/v1/models` → `['gemma4-26b-a4b-it']`; ps cmdline of PID 4068133/4068576 contains `--structured-outputs-config {"backend":"xgrammar","reasoning_parser":"gemma4"}` and DP=4
   ├ G4 ✓ Gate · GPU 0,1,2,3 still allocated to this Gemma serve (4× ~70GB used per `nvidia-smi`); no other model squatting
   │      Evidence · nvidia-smi: GPU 0=71177 MiB, GPU 1=69839 MiB, GPU 2=69839 MiB, GPU 3=69839 MiB used (DP=4 replicas, ~70GB each, matches Gemma-4-26B-A4B-it config)
   └ How  · the four commands above, copy outputs into Evidence

## Phase 2 — Backend wiring [1/1 ✓]

02 ✓ Verify `thinking_token_budget` plumbs through SamplingConfig → backends/vllm.py → vllm OpenAI server
   ├ G1 ✓ Gate · single curl test returns 200 with bounded completion_tokens
   │      Evidence · First attempt (pre-restart) HTTP 400 "thinking_token_budget is set but reasoning_config is not configured" — ROOT CAUSE found: vllm 0.20.0 only initializes ReasoningConfig when top-level `--reasoning-parser` is set (arg_utils.py:2332-2337), NOT when `reasoning_parser` is only inside `--structured-outputs-config`. Edited serve_gemma4_26b_a4b_dp4.sh to add `--reasoning-parser gemma4` + drop the duplicate inside `--structured-outputs-config`, restarted serve (PID 2850904). Post-restart curl: HTTP 200, completion_tokens=60 ≤ 100 ✓ on `What is 2+2?` with budget=50, finish_reason=stop, content="2 + 2 = 4". Note: Gemma4 thinking parser uses `<|channel>...<channel|>` delimiters (not `<|think|>` per docstring); simple prompts skip thinking entirely → reasoning_content empty, budget moot. Hard prompt (modular arithmetic proof) with budget=200 also did NOT enter thinking phase (reasoning_content_len=0, completion_tokens=1176) — confirms budget is per-thinking-phase, not per-output. HLE-class prompts that historically hit p99=25K tokens are where budget matters; verified mechanism in Phase 3 below.
   ├ G2 ✓ Gate · `thinking_token_budget` added to SamplingConfig + routed through `_split_sampling_kwargs`
   │      Evidence · methods/specs.py:46-54 added `thinking_token_budget: int | None = None` with comment on the top-level `--reasoning-parser` requirement. backends/vllm.py:111 added `thinking_token_budget` to the `extra_body` route (alongside top_k/min_p/repetition_penalty). Smoke: `SamplingConfig(thinking_token_budget=15000, ...)` parses; `_split_sampling_kwargs` returns `extra={'thinking_token_budget': 15000, 'chat_template_kwargs': {'enable_thinking': True}}`; `extra="forbid"` still rejects unknown fields (ValidationError on `foo_bar=1`).
   ├ G3 ✓ Gate · smoke yaml parses through PrecacheConfig
   │      Evidence · scripts/hle/grpo/hle_gemma4_26b_a4b_precache_budget15k_smoke.yaml created (num=2, num_workers=16, num_explores=8, cache_dir=`../analysis/cache/hle/gemma4_26b_a4b_it_budget15k_smoke/gold`, sampling.thinking_token_budget=15000, sampling.max_tokens=60000). `load_config(config_path=..., schema=PrecacheConfig)` returns parsed config with `thinking_token_budget=15000` visible at `c.sampling.thinking_token_budget`.
   └ How  · curl for G1; read `methods/specs.py` + `backends/vllm.py:call_sub_model` for G2; minimal yaml + precache_explores.py for G3

## Phase 3 — Mechanism smoke (1 budget value, num=2 = 16 explores) [STOP — budget NOT enforced for gemma4]

03 ☐ Run HLE smoke with `thinking_token_budget=15000` on the same 2 questions used by prior smoke (qids `668825f80a642802bdfeadfa` and `66b827b9b64deaedfbb997a2` — first 2 of HLE-gold-text-only at seed=42)
   ├ G1 ☐ Gate · 16/16 result.json written; runs to completion in wall ≤ 12 min (vs 20+min when budget absent)
   │      Evidence · 16/16 written but wall=17:00 (started 01:14:20, last cache write 01:31:16) — exceeds 12 min. 3/16 finished at ~2-4 min (the natural-short ones), 13/16 ran to the explore_timeout=1015s wall (~17 min). Banner: `Done. 16 cached, 0 skipped (already existed).` FAIL on wall budget.
   ├ G2 ☐ Gate · max `usage.output_tokens` across 16 result.json ≤ **16500** (15K budget + ~1.5K post-think JSON headroom). If max > 16500 → budget NOT enforced for gemma4 parser → STOP, report to user, do NOT proceed to Phase 4. If max ≤ 16500 → budget mechanism works.
   │      Evidence · **STOP-the-run.** 13/16 explores hit `completion_tokens=60000 finish_reason=length` (max_tokens cap, NOT budget cap) → effective max output_tokens = 60000 ≫ 16500 threshold. Of the 3 non-timed-out: 8774, 8811, 16707 — these were the natural-short ones that would have finished anyway. Budget mechanism did NOT fire on the 13 long ones. Root cause confirmed by web search (vllm#38855 + #39103 + parser inspection 2026-05-03): Gemma4ReasoningParser uses text-level `<|channel>...<channel|>` matching, but vLLM's V1 engine strips special tokens before the parser sees them at the streaming/serving path; AND for HLE-style structured-output prompts Gemma4 does not emit `<|channel>` open at all — "thinking" instead lands inline in the JSON `approach` field. PR #20859's logits processor depends on `is_reasoning_end()` returning a meaningful state, which it cannot when the parser never sees the open token. Sources: https://github.com/vllm-project/vllm/issues/38855 (open, vllm 0.18.2rc1+main affected; suggested fix not yet merged), https://github.com/vllm-project/vllm/issues/39103 (analogous "thinking unbounded" symptom on Nemotron v3 with `--reasoning-config`).
   ├ G3 ☐ Gate · `parse_failed=true` count ≤ 2/16 (≤12.5%); `answer` field non-empty in ≥14/16 (orchestrator can still produce structured output after forced thinking-end)
   │      Evidence · `parse_failed=True` in 13/16 (81.25%) — every timed_out row carries it. Non-empty `answer` in 3/16 (the 3 natural-short rows: `C, C, 89`); 13/16 have `answer=None` because JSON was truncated at completion_tokens=60000 before the schema closed. FAIL by ~7×.
   ├ G4 ☐ Gate · zero token-repetition loops in any output.md (grep ≥10 identical adjacent tokens like `"step-step-step..."`); should not be present given xgrammar+reasoning_parser=gemma4 already verified clean in prior smoke
   │      Evidence · No `output.md` files written in this cache (precache_explores.py only writes result.json + duration). Truncated JSON fragments inspected via log: e.g. explore_3 starts `{"approach": "The problem asks which condition of Arrhenius's sixth impossibility theorem ..."` — coherent text, NO repetition collapse pattern (`step-step-step` etc.) → reasoning_parser+xgrammar chain still keeps Gemma JSON clean; the 60K is genuine model verbosity, not a vllm#40080 loop. PASS on this gate (the only one).
   └ How  · clone `hle_gemma4_26b_a4b_precache_smoke.yaml` to `_budget15k.yaml`, set sampling block with thinking_token_budget=15000 (via whichever wiring Phase 2 found), separate `cache_dir: ../analysis/cache/hle/gemma4_26b_a4b_it_budget15k_smoke/gold`, run `precache_explores.py`

## Phase 4 — Budget scan (3 values, num=10 = 80 explores each) [0/1]

04 ☐ Sweep `thinking_token_budget ∈ {10000, 15000, 20000}` on the same 10 HLE questions; pick the budget value with best (low timeout rate) × (high parse success) × (high non-empty answer rate)
   ├ G1 ☐ Gate · 3 separate cache_dirs (`_budget10k_scan/_budget15k_scan/_budget20k_scan`), 80 result.json each, all 3 runs complete (no Traceback)
   │      Evidence · 
   ├ G2 ☐ Gate · timed_out rate per budget ≤ 30% (vs 78% baseline); Δ between best and worst ≥ 10pp (signal must be larger than noise on 80-sample size)
   │      Evidence · 
   ├ G3 ☐ Gate · parse_failed rate per budget ≤ 15% (degraded from prior 0% for non-timed_out cases is acceptable but cap at 15%)
   │      Evidence · 
   ├ G4 ☐ Gate · empty `answer` field rate per budget ≤ 10% across non-timed_out non-parse_failed rows
   │      Evidence · 
   ├ G5 ☐ Soft-Gate · **Quality sanity (Claude self-conducted)** — sample 5 random qids per budget value, for each verify:
   │      (a) `answer` is structurally plausible for the question (e.g., HLE physics question gets a number/expression, not "I don't know")
   │      (b) `reasoning` field shows `<|think|>` was forced to close around the budget value (visible truncation pattern)
   │      (c) `confidence` field is in [0, 1] and not always 0 / always 1 (suggests judgment was preserved post-truncation)
   │      Justification (required) · 1-2 sentences per (a)-(c) per budget value, with qid + result.json path + concrete excerpt; do NOT just say "looks fine".
   │      Evidence · 
   ├ G6 ☐ Gate · **decision recorded** — picked budget = X based on: lowest timed_out + parse_failed combined, with answer-quality soft-gate passing. Document the trade-off: if 10K cuts timeouts most but parse_failed > 15K's parse_failed by >5pp, prefer 15K.
   │      Evidence · 
   └ How  · 3 yaml clones; sequential or parallel runs (parallel needs different cache_dir, OK); python script to count timed_out/parse_failed/empty per cache_dir; sample-5 review

## Phase 5 — Production rollout decision [STOP — short-circuited at Phase 3]

05 ✓ Pick path forward — short-circuit STOP fired at Phase 3 (mechanism failed before scan)
   ├ G1 ☐ Gate · IF Phase 4 G2 passed (timeout rate ≤ 30% achievable at some budget value): proceed to Phase 6, embed budget in production yamls
   │      Evidence · n/a — Phase 4 not entered.
   ├ G2 ✓ Gate · IF Phase 4 G2 failed (no budget value gets timeout rate below 30%): STOP and report to user. Original 78% timeout rate stands as Gemma's model-physics ceiling on HLE. Document that thinking_token_budget was tried + insufficient.
   │      Evidence · STOP triggered at Phase 3 G2 (budget mechanism does not fire on Gemma4 — root cause: Gemma4ReasoningParser does not detect `<|channel>` boundaries in the V1 streaming path, AND Gemma4 inlines its chain-of-thought into the JSON `approach` field rather than emitting a `<|channel>...<channel|>` thinking block on HLE-style prompts → PR #20859's logits processor has nothing to clamp). Phase 3 smoke timeout rate 81.25% (13/16) ≈ baseline 78% (249/324). Approach abandoned. Phase 4 budget scan skipped. Production HLE Gemma row reverts to baseline strategy: keep the 75 useful explores cached, accept the model-physics ceiling, and either (a) run the paper row on the sparse 75-explore subset with disclosure, or (b) drop Gemma from `tab:backbone-ablation` HLE column entirely. **Decision deferred to user.**
   ├ G3 ☐ Gate · IF G3 (parse_failed) or G4 (empty answer) blew past threshold: STOP — budget kills JSON quality faster than it saves timeouts. Net is worse than baseline. Report to user.
   │      Evidence · n/a — different failure mode (budget never fired; parse_failed is a downstream of max_tokens cap, not budget killing JSON).
   └ How  · summary table of Phase 4 metrics; pattern-match to G1/G2/G3 decision tree; report decision to user before launching production

## Phase 6 — Production HLE precache rerun (resume cache) [SKIPPED — budget approach abandoned]

Skipped: thinking_token_budget does not enforce on Gemma4 (Phase 3 G2 STOP). Pivoting strategy belongs to companion `todo_inf_gemma_A_exp_orch.md` Item 15, which retains the 75-useful cached subset and the unchanged 78% timeout ceiling.

## Phase 7 — Roll out to GPQA + LCB if HLE validates [SKIPPED — HLE never validated]

Skipped for the same reason as Phase 6.

<!-- Original Phase 6 + Phase 7 items (06-09) deleted 2026-05-03: Phase 3 G2 STOP fired, budget mechanism does not apply to Gemma4 → no production budget value to roll out. -->

## Sharper diagnosis (appended 2026-05-03 04:40 after Gemma DP=1 + tool_chat_template_gemma4.jinja experiment)

Earlier Phase 3 evidence concluded "Gemma4 inlines its chain-of-thought into the JSON `approach` field". That diagnosis was incomplete. Today's experiment with the official `tool_chat_template_gemma4.jinja` (vendored to `scripts/gpqa/grpo/tool_chat_template_gemma4.jinja`) and a clean DP=1 serve disambiguated the layers:

- **Parser layer is fine.** `Gemma4ReasoningParser` already exposes `start_token_id=100` (`<|channel>`) and `end_token_id=101` (`<channel|>`) — auto-derived in `BaseThinkingReasoningParser.__init__` from the `start_token`/`end_token` properties. The vllm#38855 patch the user referenced is structurally already in 0.20.0. `adjust_request` also sets `skip_special_tokens=False` so the parser sees boundary tokens.
- **Chat template layer is fine.** `--chat-template tool_chat_template_gemma4.jinja` static rendering matches the model's default tokenizer template **byte-for-byte** for `enable_thinking=True` (both produce `<bos><|turn>system\n<|think|>\n<turn|>\n<|turn>user\n...<turn|>\n<|turn>model\n`). Restarting Gemma DP=1 with `--chat-template` did not change observed behavior.
- **Endpoint layer is the real issue.** Decisive A/B at `temperature=0` (deterministic):
  - `/v1/completions` with `prompt_token_ids = [2, 105, 9731, 107, 98, 107, 106, 107, 105, 2364, 107, 3689, ...]` (the chat-rendered IDs): **3/3 trials → output starts with `"thought\n..."`** (channel block opens correctly).
  - `/v1/chat/completions` with the same logical message + `chat_template_kwargs:{enable_thinking:true}` at top-level (NOT in `extra_body` — that location silently strips it, dropping prompt_tokens 28 → 20 with `<|think|>` lost): **5/5 trials → `reasoning_content=""`, `content="To find the sum..."`** (channel block never opens).
  - With `thinking_token_budget=200` added: **5/5 trials → `rc=0`, `ct≈170-420`, `finish=stop` well below `max_tokens=600`.** Budget never fires because model never opens channel.
- **Recommendation handover.** Three production options, each with cost:
  - **A.** Patch `backends/vllm.py` to call `/v1/completions` with manually rendered chat template + post-process raw text through the reasoning parser. ~50 LoC, breaks OpenAI-compat for Gemma only.
  - **B.** (recommended) Drop `thinking_token_budget` for Gemma; cap with `max_tokens=4000` instead. Existing 75 useful Gemma trajectories stay valid for `tab:backbone-ablation` HLE column with disclosure footnote.
  - **C.** Wait for vllm > 0.20.0 (no known fix PR; 0.20.0 is current latest as of 2026-05-03).

Decision deferred to user. The ATTS pipeline currently consumes Gemma via chat completions through `backends/vllm.py:run_tool_conversation`; switching to raw completions requires breaking that abstraction or paying option A's engineering tax.

Sources cross-checked today:
- vllm#38855 (open, no merged fix) https://github.com/vllm-project/vllm/issues/38855
- vllm#37112 (reasoning_budget feature, merged) https://github.com/vllm-project/vllm/pull/37112
- vLLM Recipes Gemma 4 Usage Guide https://docs.vllm.ai/projects/recipes/en/latest/Google/Gemma4.html
- vllm#39697 (Qwen3.5 budget leaks reasoning_end_str into content — analogous fragility) https://github.com/vllm-project/vllm/issues/39697

## Update 2026-05-03 evening — channel-open root cause fixed; this todo is revisitable

The 04:40 diagnosis above concluded "Gemma4 inlines its chain-of-thought into the JSON `approach` field" and "model never opens `<|channel>`". Today's evening session (2026-05-03 ~18:00-19:00) refined the root cause and produced a working fix:

- **Root cause refined (Layer 1)**: HF default `chat_template.jinja` line 348-352 (and vLLM upstream `examples/tool_chat_template_gemma4.jinja` line 324-330) only prefill `<|channel>thought\n<channel|>` (closed empty thinking block) when `enable_thinking=false`; for `enable_thinking=true` they leave the prompt at bare `<|turn>model\n` and rely on the model to open `<|channel>` itself. The IT-tuned weights' first-token logit at `<|turn>model\n` prefers a normal text token over `<|channel>` (token 100) — verified at temperature=0 and 1.0 (3 seeds), with system-prompt nudges, even with `<|channel>thought\n` prefill via `/v1/completions` without the system-turn `<|think|>`.
- **Fix (Layer 1)**: vendored `chat_template.jinja` to `scripts/gpqa/grpo/tool_chat_template_gemma4_fixed.jinja`, patched the `add_generation_prompt` block to also prefill `<|channel>thought\n` (channel OPEN, no closing token) when `enable_thinking=true`. Wired into serve via `--chat-template scripts/gpqa/grpo/tool_chat_template_gemma4_fixed.jinja` flag in `scripts/gpqa/grpo/serve_gemma4_26b_a4b_dp4.sh`.
- **Root cause refined (Layer 2)**: even with channel open, vLLM 0.20.0 routes the parsed thinking into `message.reasoning` (matching OpenAI o1-series schema), NOT `message.reasoning_content` (the older field name some vLLM docs and parser comments still reference). Pre-fix `backends/vllm.py:call_sub_model` read only `reasoning_content` → silently dropped thinking.
- **Fix (Layer 2)**: `backends/vllm.py` now reads `getattr(msg, "reasoning", None) or getattr(msg, "reasoning_content", None)` (forward-compat); `run_tool_conversation` mirrors.
- **Validation evidence**: real e2e through `backends/vllm.py:call_sub_model` at `tmp/case_demo_real/output.md` — 2688 chars: `<think>` block (1679 chars natural CoT including self-correction `"Wait, should I list it as '1' or 'x=1'?"` and verification `1 - 4 + 6 - 4 + 1 = 0`) followed by schema-valid JSON answer. `usage.completion_tokens=944`.

**Implications for this todo:**
- Phase 3 G2 STOP is now logically obsolete — channel DOES open, thinking DOES flow into a separate field. The "thinking_token_budget mechanism does not fire on Gemma4" conclusion was based on the channel-never-opens premise; that premise is now falsified.
- The `thinking_token_budget` approach can therefore be retried. Phase 3 / Phase 4 budget-scan items can be re-attempted with the fix in place.
- However, this todo is currently NOT in the user's active workstream — Variant A `_exp_orch` and Variant B `_sonnet_orch` are. Re-opening this todo is only justified if HLE-class long-tail timeouts re-emerge AT scale on the post-fix pipeline (Variant A item 14 smoke + item 15 full HLE precache will measure).
- If timeouts re-emerge, the next attempt should: (a) NOT clone the pre-fix budget-scan caches (they're archived at `analysis/archive/gemma_pre_thinking_fix_2026-05-03/cache/hle/gemma4_26b_a4b_it_budget15k_smoke/`); (b) start with `thinking_token_budget=15000` on a fresh `cache_dir`; (c) verify `usage.output_tokens` actually clamps near budget × 1.1 (the original Phase 3 G2 threshold).

Cross-reference: full Gemma-4 thinking double-bug entry in vllm skill at `~/.claude/dream-clone/plugins/memory-recall/skills/vllm/references/troubleshooting.md` (commit `47b978b vllm skill: document Gemma-4 thinking double-bug on vLLM 0.20.x`).

