# Competitor Analysis: metacog-tts vs meta-cognitive vs ATTS
Date: 2026-03-29

## Two Projects (Same Author, Different Versions)

### meta-cognitive (simpler version)
- Pipeline: FOK → Solve+JOL → done (single attempt)
- Phase 1: FOK (Feeling of Knowing) — quick gut-feeling assessment (0-1 score + domain)
- Phase 2: Solve + JOL — solve the problem, report Judgment of Learning (0-1 confidence)
- NO retry loop. Single attempt only.
- Tools: fok, solve_with_jol
- Same model does both assessment and solving

### metacog-tts (extended version with retry)
- Pipeline: FOK → Solve+JOL → Retry loop → Summary
- Phase 1: FOK — same as above
- Phase 2: Initial Solve+JOL — same as above
- Phase 3: RETRY LOOP based on formula:
  Score_retry = alpha*(1-JOL) + beta*max(FOK-JOL,0) + gamma*max(JOL-FOK,0) + delta*(1-FOK)
  Default params: alpha=1.561, beta=0.386, gamma=0.054, delta=0.174, threshold=0.751
  If score > threshold → retry (with history of previous attempts)
  Up to max_retries (default 3)
- Phase 4: Summary — synthesize all attempts into final answer
- Tools: fok, solve_with_jol, summarize

## Key Architectural Differences: metacog-tts vs ATTS

| Dimension | metacog-tts | ATTS |
|---|---|---|
| Metacognitive signal | FOK (pre-solve) + JOL (post-solve) — explicit scalar scores | Orchestrator's free-form assessment of candidate pool |
| Retry decision | Formula-based: alpha*(1-JOL) + ... > threshold | LLM orchestrator judges VOC implicitly |
| Solver independence | Same model retries with history (NOT independent) | Independent fresh solvers (stateless, never see prior candidates) |
| Retry mechanism | Sequential refinement — solver sees previous attempts | Parallel sampling — each explore is fresh |
| Stopping criterion | Parametric formula with fitted coefficients | LLM orchestrator's judgment (no explicit formula) |
| Action space | Single: retry(same model, with history) | Extensible: explore(effort), explore(model), stop(answer) |
| Synthesis | Separate summary phase after all retries | Orchestrator synthesizes throughout, informed by full history |
| Theory | Metacognition (FOK/JOL from cognitive science) | Rational metareasoning (VOC from Russell & Wefald) |

## Critical Difference: Independent vs Dependent Retries

**metacog-tts**: Each retry SEES the previous attempts ("try a DIFFERENT approach and address the concerns raised in previous JOL reasons"). This means retries are CORRELATED — each one is conditioned on the full history. This is self-refinement, not independent sampling.

**ATTS**: Each explore is STATELESS — the explorer never sees previous candidates. This guarantees conditional independence. The orchestrator accumulates evidence, but the evidence sources are independent.

This is the fundamental architectural distinction. metacog-tts is a SEQUENTIAL REFINEMENT system with metacognitive monitoring. ATTS is a PARALLEL SAMPLING system with agentic orchestration.

## How to Differentiate in Paper

1. metacog-tts uses EXPLICIT metacognitive signals (FOK, JOL) as scalar scores, then a FORMULA decides retry. ATTS uses the LLM's full reasoning assessment with NO explicit formula.

2. metacog-tts's retries are DEPENDENT (solver sees history) — this is self-correction, which Huang et al. (2024) showed LLMs cannot reliably do. ATTS's explores are INDEPENDENT — diversity comes from fresh starts, not from "try differently."

3. metacog-tts has a FIXED action space (retry or stop, always same model). ATTS has an EXTENSIBLE action space (different efforts, different models, etc.).

4. metacog-tts's stopping is PARAMETRIC (fitted formula). ATTS's stopping is EMERGENT (orchestrator's judgment, no fitted parameters).

## Framing for Two Papers

**Their paper (metacog-tts)**: "Metacognitive monitoring (FOK/JOL) enables LLMs to know what they know and retry when uncertain."
- Core contribution: FOK/JOL as structured metacognitive signals for self-assessment
- Theoretical grounding: Cognitive science metacognition literature
- Mechanism: Self-refinement with metacognitive monitoring

**Our paper (ATTS)**: "Rational metareasoning enables an orchestrator to manage independent explorations as a sequential resource allocation problem."
- Core contribution: Agentic orchestration with extensible action space
- Theoretical grounding: Rational metareasoning (Russell & Wefald, Hay et al.)
- Mechanism: Independent candidate pool with delegated evaluation

The key differentiator: metacog-tts asks "does the model KNOW it doesn't know?" (metacognition). ATTS asks "should the orchestrator SPEND more compute?" (metareasoning / resource allocation). Same family, different questions.
