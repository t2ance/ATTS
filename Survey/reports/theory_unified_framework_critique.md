# Unified Meta-MDP Framework: Critical Analysis
Date: 2026-03-29
Source: framework-critic agent

## Coverage Assessment

| Method Family | Fits? | Why / Why Not |
|---|---|---|
| Parallel sampling (SC, BoN, weighted voting) | Yes, cleanly | Independent complete candidates, episode-level reward |
| Adaptive stopping (BEACON, ConSol, ATTS) | Yes, cleanly | Stopping as action in the MDP |
| Debate | Yes, with caveat | Symmetric constraint; but violates i.i.d. if agents update |
| Tree-of-Thought / MCTS | No | Step-level granularity, tree-structured state, expansions not i.i.d. |
| Refinement / Self-Refine | No | Violates i.i.d. — output conditioned on previous round |
| Process Reward Models | No | Step-level supervision, granularity mismatch |
| Speculative decoding | No | Latency optimization, not reasoning strategy |
| Training-time methods (R1, o1) | No | Training-time, not inference-time compute allocation |

## Spectrum Analysis

- Smooth interpolation WITHIN families: voting weights, stopping thresholds
- Discrete jumps BETWEEN families: fixed->adaptive, frequency->reasoning
- More of a partial order than a linear spectrum

## Predictive Power

The framework predicts: unrestricted optimal >= restricted optimal. This is tautologically true.

It does NOT predict:
- Ordering of IMPLEMENTED (suboptimal) methods
- When removing a restriction helps vs hurts
- Which new restriction combinations yield disproportionate gains

Paper's own data shows reversals:
- BabyVision: MV = ATTS (23.20%) — orchestrator adds cost, not accuracy
- HLE: Fixed-N LLM Selection (58%) > Adaptive ATTS (56%)
- Opus is worst orchestrator on GPQA despite being strongest model

## Extensibility

Each extension (refine, verify, human feedback) works by expanding state/action space, but each expansion sacrifices the structural properties (i.i.d., candidate-level granularity) that make the framework tractable and informative.

## Falsifiability

As stated ("every TTS method is a restricted policy"), unfalsifiable — can always expand state/action space.

Falsifiable version: "within sample-and-aggregate methods, adaptive orchestration over independent explores improves cost-accuracy tradeoff."

Falsified by: systematic superiority of fixed-N methods, or training-time methods dominating inference-time methods.

## Recommendation

Frame as "unifying perspective for sample-and-aggregate TTS methods" — NOT "all TTS methods."

Explicitly scope: methods where explores produce complete, independent candidates, and the orchestrator's role is sequential stopping and synthesis.

Acknowledge boundary: tree-search, refinement, step-level, and training-time methods lie outside this scope.
