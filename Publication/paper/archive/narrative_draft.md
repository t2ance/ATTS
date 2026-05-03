# ATTS Paper Narrative Draft
Date: 2026-03-29
Purpose: Complete story outline for teammate review before rewriting Methodology

## One-sentence pitch
ATTS is a delegated sequential test-time scaling framework where a lightweight orchestrator adaptively controls independent explorations, treating compute allocation as closed-loop decision-making over an extensible action space.

## Narrative threads (must be consistent throughout)
1. **Delegated sequential** — control separated from solving, orchestrator never solves
2. **Fresh independent explores** — stateless solvers, diversity by design
3. **Extensible action space** — richness of decisions = value of framework
4. **Stateful evidence accumulation** — orchestrator accumulates multi-turn context
5. **Cost-aware stopping** — each explore has a cost, stopping saves money

## Section-by-section story

### Abstract (no change needed)
Already covers: delegated, explore tool, adaptive stopping, extensible action space, results, integrator ablation.

### Introduction
Para 1: Test-time scaling is important. Three axes (sequential, parallel, hybrid). No single strategy is optimal [Snell]. Naive scaling can degrade [BoN paper]. Central question: HOW to allocate adaptively.

Para 2: Current approaches are either open-loop heuristics (predict before generation) or heavy micro-level search (tree expansion, step-level verification). Agentic TTS methods add variants but still use statistical rules. **Underexplored alternative**: delegate to an LLM orchestrator doing closed-loop sequential decisions over candidate hypotheses.

Para 3: Introduce ATTS. Separate control from solution generation. Orchestrator manages candidate pool via explore(). Extensible action space. We evaluate simplest instantiation + multi-model variant.

Para 4: Evaluation summary. 6 benchmarks. Results.

**Proposed addition**: One sentence in Para 2 or 3 connecting to rational metareasoning:
> "This design reflects the principle of rational metareasoning [Russell & Wefald 1991]: an agent should reason about whether additional computation is worth its cost before committing to an answer."

### Related Work (no structural change, minor edits)
4 paragraphs as currently written:
1. Test-Time Scaling (sequential/parallel/hybrid) — ATTS is distinct (delegated independent solves)
2. Adaptive Compute Allocation — statistical stopping, Pandora's Box, learned allocators; ATTS uses LLM reading full traces
3. Agentic Orchestration — meta-controllers, RL routers, debate; ATTS = asymmetric (fresh independent + accumulating orchestrator)
4. Cost-Aware Agent Training — CATP-LLM, CTA, ToolRL; ATTS targets candidate-level TTS specifically

**Move De Sabbata differentiation here**: Add to paragraph 2, after discussing Hay et al.:
> "De Sabbata et al. [2024] operationalize rational metareasoning at the token level, training models to skip unnecessary reasoning steps. ATTS pursues the same principle at a coarser candidate level, where each explore is the unit of computation — the agentic extension they identify as future work."

### Methodology (THE MAIN REWRITE)

**Opening paragraph** (~4 sentences):
- Test-time scaling as resource allocation: given a budget of model calls, how to decide what to compute next and when to stop?
- Brief metareasoning connection (1 sentence): "In the rational metareasoning framework [Russell & Wefald 1991, Hay et al. 2012], an agent should compute if and only if the expected benefit of the computation exceeds its cost. ATTS operationalizes this principle for test-time scaling:"
- Core design in one sentence: a lightweight orchestrator manages a growing pool of candidate solutions through a single tool, explore(), which dispatches a fresh independent solver.
- When the orchestrator judges additional exploration unlikely to change the outcome enough to justify its cost, it stops and synthesizes the final answer.

**3.1 The ATTS Loop**
- **Explore**: Each explore() dispatches fresh solver on original problem. Solver is stateless (sees only problem, never prior candidates). Returns structured candidate (answer, reasoning, approach, confidence). Appended to pool C_t.
  - Interpretive comment: "This fresh-start design guarantees conditional independence: the orchestrator can assess the marginal value of each additional explore in isolation, without modeling interactions between future computations."
- **Stopping**: Orchestrator can stop at any point before budget T. Decision based on its assessment of the candidate pool. No external threshold imposed. Prompt encodes principles for judgment (convergence, repeated failure, cost justification).
  - Interpretive comment: "The stopping judgment is analogous to the value-of-computation criterion [Hay et al. 2012]: the orchestrator implicitly assesses whether one more explore is likely to produce enough new information to justify its cost."
- **Synthesis**: When orchestrator stops, it synthesizes final answer from C_t. Has observed full multi-turn history — richer context than a single-call aggregator.
- Algorithm box (same as current)
- Implementation note: control logic is benchmark-agnostic, only explorer prompt changes.

**3.2 Extensible Action Space** (keep, minor edits)
- The framework's value scales with the richness of the orchestrator's action space.
- Simplest: single action "explore once more with same model" — only decides WHEN to stop.
- With heterogeneous computations: also decides WHAT to compute (effort, model, strategy).
- "This positions ATTS as a framework for selecting computations [Hay et al. 2012] at test time."
- We evaluate: single model uniform effort + multi-model dispatch + (future) adaptive effort.

**3.3 Multi-Model Extension** (keep, reframe effort)
- Model selection added to action space: Haiku/Sonnet/Opus pool.
- Model profiles inform dispatch: start cheap, escalate on disagreement, cross-model agreement = stronger evidence.
- **Effort as exploration floor** (reframed):
  > "An exploration effort parameter sets a minimum number of explores and model diversity before the orchestrator's stopping judgment takes effect. Low effort permits stopping after any two candidates converge; Medium requires at least three explores across two models; High requires at least five explores using all three models. Above these floors, the orchestrator remains free to stop or continue based on its assessment of the candidate pool."

### Experiments (no change)
### Limitations (no change)
### Conclusion (no change)
### Appendix Theory (no change — submartingale + replication theorems stay)

## Potential challenges from reviewers

1. "If the orchestrator is just following prompt instructions, how is this different from prompt engineering?"
   → Answer: The contribution is the ARCHITECTURE (separated control, independent explores, extensible action space), not the prompt. The prompt encodes principles; the framework ensures they have the right information to work with.

2. "VOC is never computed — how can you claim metareasoning?"
   → Answer: We claim the DESIGN is motivated by metareasoning, not that the orchestrator formally computes VOC. Just as a human doesn't compute Bayesian updates but their behavior can be described by Bayesian models.

3. "Effort levels are hardcoded — this contradicts adaptive allocation"
   → Answer: Effort sets a floor (minimum exploration). Above the floor, stopping is fully adaptive. This is analogous to setting a minimum sample size in hypothesis testing before applying a stopping rule.

4. "Opus worse than Sonnet on GPQA — why?"
   → Answer: Orchestrator quality depends on model-domain calibration, not raw capability. This is an empirical finding, and the mechanism remains open (acknowledged in paper).

5. "82% of questions use exactly 2 explores — is the orchestrator even doing anything interesting?"
   → Answer: Yes, and this is both a strength and a limitation. On GPQA, most questions are answerable with 2 high-quality explores when they converge. The conservative stopping is noted as a limitation with open research direction (GRPO training for better stopping).
