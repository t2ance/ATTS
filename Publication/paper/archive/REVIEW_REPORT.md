# Consolidated Peer Review: ATTS Paper
Generated: 2026-03-26

## CRITICAL — Factual Errors in Text

| # | Issue | Location | Status |
|---|-------|----------|--------|
| C1 | Abstract: "degrades accuracy on **three of four** benchmarks" — actual: 2 degrade (LCB, GPQA), 1 improve (HLE), 1 tie (BBV) | Abstract L60, Conclusion L329 | TODO |
| C2 | Abstract: "outperforms or matches **all** baselines" — false: LLM Selection beats ATTS on HLE (58% vs 56%) | Abstract L60 | TODO |
| C3 | BabyVision Majority Voting overall=22.42% contradicts its own per-subset numbers (which compute to 23.20%) | Table 1d | TODO |
| C4 | GPQA Pass@1=74.24% comes from the **with-integrator** run, not the default ATTS run (should be 74.75%) | Table 1c | TODO |
| C5 | GPQA Majority Voting=74.24% (same as Pass@1) — data shows 73.74%. Looks like Pass@1 was copy-pasted | Table 1c | TODO |
| C6 | "88% of questions use exactly two explore calls" — from the wrong run. Default ATTS: 82.3% | Sections 5,6 | TODO |
| C7 | HLE LLM Selection=58.00% — data shows 57/99 completed = 57.58% (or 57/100 = 57.00%) | Table 1a | TODO |

## HIGH — Numerical Errors in Tables

| # | Issue | Location | Status |
|---|-------|----------|--------|
| H1 | HLE Skywork cost $2.22 — data shows $4.41/q (8 candidates, not 4) | Table 1a | TODO |
| H2 | HLE Pass@1 cost $0.28 — data shows $0.50/q | Table 1a | TODO |
| H3 | BabyVision +Integrator cost $0.34 — data shows $0.378/q | Table 3 | TODO |
| H4 | Section "4.4" reference doesn't exist (should be 4.3 or use \ref) | Appendix A, L342 | TODO |

## HIGH — Methodological Gaps

| # | Issue | Status |
|---|-------|--------|
| M1 | **No error bars or variance on any result.** HLE=100 questions, GPQA=198, AIME=30. A 2pp diff on HLE = 2 questions. All single-run. BabyVision official protocol reports std but paper drops it. | TODO |
| M2 | **Single model family (Claude only).** Orchestrator, explorer, all baselines = Claude. No GPT, Gemini, DeepSeek, or open-weight. Generality claim unsupported. | TODO |
| M3 | **"Principles over instructions" (Sec 3.3) unsubstantiated.** Key design claim supported only by anecdote, no ablation comparing prescriptive vs principles-based prompts. | TODO |
| M4 | **Cost accounting potentially unfair.** Reward model compute (Skywork, VisualPRM) runs locally = $0 in cost. ATTS includes orchestrator API cost. Apples-to-oranges. No wall-clock latency reported. | TODO |
| M5 | **LLM Selection comparison mixed but underplayed.** ATTS loses on HLE, ties on BBV, wins on LCB+GPQA. Paper frames as universal win. | TODO |

## HIGH — Missing Related Work

| Paper | Why critical | Status |
|-------|-------------|--------|
| Brown et al. 2024, "Large Language Monkeys" (2407.21787) | Coverage scales with repeated sampling; selection bottleneck is ATTS's premise | TODO |
| Chen et al. 2023, "Universal Self-Consistency" (2311.17311) | LLM selects among candidates for free-form answers — ATTS's synthesis step | TODO |
| Manvi et al. 2024, "Adaptive Inference-Time Compute" (2410.02725) | LLM predicts if restarting helps; adaptive sample count — same stopping idea | TODO |
| RASC (2408.17017) / ReASC (2601.02970) | Self-consistency with adaptive stopping based on reasoning quality | TODO |
| De Sabbata et al. 2024, "Rational Metareasoning for LLMs" (2410.05563) | Actually implements Russell & Wefald VOC for LLMs — paper cites R&W but not this | TODO |
| Bilal et al. 2026, "Adaptive TTC Allocation" (2602.01070) | Contemporary work doing PRM-guided adaptive allocation | TODO |

## MEDIUM — Structural / Presentation Issues

| # | Issue | Location | Status |
|---|-------|----------|--------|
| S1 | NeurIPS checklist **entirely unfilled** (all \answerNA). Submission blocker. | L596-672 | TODO |
| S2 | Style file says NeurIPS **2024** (38th), needs 2026 (40th) | main.sty | TODO |
| S3 | Rational metareasoning framing (Sec 3.4) is window dressing — no belief state, no VOC, no formal stopping. Either formalize or remove. | Sec 3.4 | TODO |
| S4 | Notation S_t, b_t, c_i, a_i, r_i, p_i, s_i introduced then never used again | Sec 3.1 | TODO |
| S5 | No Algorithm pseudocode box for the ATTS loop | Sec 3 | TODO |
| S6 | BabyVision text: "ATTS **leads**" — actually ties with LLM Selection at 23.20% | L262 | TODO |
| S7 | Conclusion: "**consistently** degrades accuracy" — same error as C1 | L329 | TODO |
| S8 | Bitter Lesson citation is a stretch — ATTS is fully prompt-engineered, opposite of Sutton's argument | L69 | TODO |
| S9 | No code or data release mentioned anywhere | Throughout | TODO |

## MEDIUM — Missing Ablations

| Ablation | Why needed | Status |
|----------|-----------|--------|
| Exploration budget T (T=2,4,8,16) | Paper uses T=8 without justification; most stop at N=2 | TODO |
| Temperature / sampling diversity | Baselines use default temp; optimized temp might close the gap | TODO |
| Prescriptive vs principles-based prompts | Sec 3.3 claim requires this | TODO |
| Equal-compute comparison | ATTS often stops early = fewer candidates = lower cost. Is it just using less compute? | TODO |

## LOW — Writing Nits

| # | Issue | Location | Status |
|---|-------|----------|--------|
| W1 | "makes clear" -> "demonstrates" | L65 | TODO |
| W2 | 5-line sentence, consider breaking | L67 | TODO |
| W3 | Parallel Scaling paragraph too dense (8 citations in one block) | L80 | TODO |
| W4 | "information gradient step" metaphor never formalized | L113 | TODO |
| W5 | "the paper" -> "our" (avoid self-referential) | L274 | TODO |
| W6 | "arose from empirical observation" — informal for Methods | L107 | TODO |
| W7 | Unused listing label lst:multi-model-prompt (defined but never \ref'd) | Appendix | TODO |
| W8 | Several unused \label definitions | Throughout | TODO |
