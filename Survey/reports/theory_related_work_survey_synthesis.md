# Theory Related Work Survey: Synthesis
Date: 2026-03-29
Sources: 3 parallel research agents (metareasoning, adaptive-TTS, RL-orchestration)

## Core Finding

**No existing work combines all four of ATTS's defining properties:**
1. Rational metareasoning / VOC theoretical framework
2. Candidate-level (inter-generation) decisions
3. Agentic orchestrator (separate LLM reading reasoning quality)
4. Sequential closed-loop (observe each candidate before deciding next)

## Closest Competitors (by dimension)

### Theoretical Framework: Rational Metareasoning + LLM
- **De Sabbata et al. (2024)** — ONLY paper connecting metareasoning/VOC to LLMs
  - But: TOKEN-level (intra-generation), not candidate-level
  - They explicitly call for agentic extension as future work (Section 7)

### Candidate-Level Stopping with Formal Framework
- **Kalayci et al. (2025)** — Pandora's Box for BoN stopping
  - Has formal framework (Weitzman index) at candidate level
  - But: uses reward model thresholds, not LLM orchestrator
  - Different theoretical lineage (Pandora's Box, not metalevel MDP)
  - No mention of Russell & Wefald or Hay et al.

### Learned Adaptive Compute Allocation
- **Damani et al. (ICLR 2025 Oral)** — Learned marginal benefit predictor
  - Conceptually close to VOC (predicting value of additional computation)
  - But: no formal metareasoning connection, one-shot allocation (not sequential)

### RL-Trained Multi-LLM Orchestrator
- **Router-R1 (NeurIPS 2025)** — GRPO-trained orchestrator, multi-round dispatch
  - Closest SYSTEM: separate orchestrator LLM, multi-round, GRPO-trained
  - But: routes to DIFFERENT models, not same model explored multiple times
  - No formal stopping theory, no VOC framework

- **xRouter (Salesforce, 2025)** — RL-trained cost-aware routing
  - Similar: separate orchestrator, cost-aware RL
  - But: routes to different models, not sequential explore-aggregate

### Sequential Multi-Turn RL (Structural Analog)
- **Search-R1 (NeurIPS 2025)** — RL-trained search-interleaved reasoning
  - Structural analog: "issue search query -> receive results -> decide again"
  - Similar to ATTS: "issue explore call -> receive candidate -> decide again"
  - But: search tool, not candidate generation; no metareasoning framework

## Gap Analysis Summary

| Dimension | De Sabbata | Kalayci | Damani | Router-R1 | Search-R1 | ATTS |
|---|---|---|---|---|---|---|
| Metareasoning/VOC theory | YES | No (Pandora) | No (learned) | No | No | YES |
| Candidate-level | No (token) | YES | YES | YES | N/A | YES |
| LLM reads reasoning | No | No | No | Yes (aggregates) | No | YES |
| Sequential closed-loop | No | YES | No (one-shot) | YES | YES | YES |
| Agentic orchestrator | No | No | Separate predictor | YES | No (same LLM) | YES |
| Effort/model selection | No | No | Model routing | Model routing | No | YES |
| GRPO trainable | Expert Iteration | No | Supervised | YES (GRPO) | YES (GRPO) | YES (planned) |

## Novelty Assessment

ATTS's positioning as "rational metareasoning lifted from token-level to candidate-level with an agentic orchestrator" is NOVEL. Specifically:

1. De Sabbata did metareasoning + LLM at token level -> we extend to candidate level
2. Kalayci did candidate-level optimal stopping -> we add LLM orchestrator + metareasoning theory
3. Router-R1 did RL-trained orchestrator -> we add formal VOC framework + same-model exploration
4. Search-R1 did sequential multi-turn RL -> we add metareasoning theory + candidate aggregation

## Key Papers to Cite

### Must-Cite
1. Russell & Wefald (1991) — Rational metareasoning foundation
2. Hay et al. (2012) — Metalevel MDP, selecting computations
3. De Sabbata et al. (2024) — Only metareasoning+LLM paper (token-level)
4. Kalayci et al. (2025) — Pandora's Box for candidate stopping
5. Damani et al. (ICLR 2025) — Learned adaptive compute allocation
6. Router-R1 (NeurIPS 2025) — GRPO-trained multi-round orchestrator
7. Snell et al. (ICLR 2025) — Compute-optimal TTS

### Should-Cite
8. Callaway et al. (2017) — Learning to select computations (cog sci)
9. BEST-Route (ICML 2025) — Joint model+N selection
10. Search-R1 (NeurIPS 2025) — Multi-turn RL with external calls
11. xRouter (2025) — RL-trained cost-aware routing
12. Agent-R1 (2025) — GRPO for multi-turn agentic RL

### Context-Cite
13. Adaptive-Consistency (EMNLP 2023) — Early statistical stopping
14. BEACON (2025) — Bayesian optimal stopping
15. ConSol (2025) — SPRT for self-consistency
16. Setlur et al. (CMU 2025) — Meta-RL framing of TTS
17. Ares (2026) — Per-step effort selection for agents
