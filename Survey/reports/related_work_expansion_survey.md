# Related Work Expansion Survey
Date: 2026-03-28

## Must-Cite Papers (gap if missing)

### 1. "Debate or Vote: Which Yields Better Decisions in Multi-Agent LLMs?" (NeurIPS 2025 Spotlight)
- Authors: Choi, Zhu et al.
- Key finding: Majority voting alone accounts for most gains attributed to multi-agent debate. Debate induces a martingale over belief trajectories — no expected improvement across rounds.
- ATTS connection: Must address why ATTS's sequential observation is not subject to the martingale limitation. Key difference: ATTS is asymmetric (orchestrator accumulates, candidates are frozen), debate is symmetric (all agents update).

### 2. Damani et al., "Learning How Hard to Think: Input-Adaptive Allocation of LM Computation" (ICLR 2025 Oral)
- Key finding: Learns to predict reward distribution given input + compute budget, allocates more compute where it helps most. 50% less compute, same performance.
- ATTS connection: Closest "learned" analog to ATTS's "zero-shot" adaptive allocation. ATTS uses the LLM itself as predictor instead of a learned one.

### 3. "Large Language Models Cannot Self-Correct Reasoning Yet" (ICLR 2024)
- Authors: Huang, Shi, Liu, Welleck et al.
- Key finding: LLMs cannot improve reasoning through intrinsic self-correction (prompting alone). Post-correction accuracy drops.
- ATTS connection: Justifies ATTS's design of separating orchestrator from explorer. Self-correction fails; external evaluation succeeds.

### 4. DeepSeek-R1: Incentivizing Reasoning via RL (Nature 2025)
- Key finding: Pure RL incentivizes self-verification, reflection, long CoT. Matches o1.
- ATTS connection: Sequential scaling via training vs ATTS's agentic approach at inference time. Complementary paradigms.

## Should-Cite Papers

### 5. BEACON: Bayesian Optimal Stopping for Efficient LLM Sampling (Oct 2025)
- Key finding: Sequential search with Bayesian learning, stops when marginal utility < cost. 80% sampling reduction.
- ATTS connection: Principled version of ATTS's stopping. BEACON needs external reward model; ATTS doesn't.

### 6. Li et al., "LLMs Can Generate a Better Answer by Aggregating Their Own Responses" (GSA, Mar 2025)
- Key finding: Sample diverse responses, use as context for improved generation. Works for open-ended tasks.
- ATTS connection: GSA is ATTS's integrator variant (batch aggregation). ATTS's orchestrator (sequential) beats integrator → sequential > batch for synthesis.

### 7. Yan et al., "Position: LLMs Need a Bayesian Meta-Reasoning Framework" (ICML 2025 Position)
- Key finding: LLMs should monitor/evaluate their own reasoning via Bayesian meta-reasoning.
- ATTS connection: Direct theoretical support for "LLM as meta-reasoner" framing. ATTS empirically tests this thesis.

### 8. AB-MCTS: Adaptive Branching in Test-Time Compute (NeurIPS 2025 Spotlight)
- Authors: Sakana AI
- Key finding: Dynamically decides wider (new candidates) vs deeper (refine existing). Outperforms both.
- ATTS connection: Wider/deeper tradeoff maps to ATTS's explore-count decision.

### 9. "Can LLM Agents Really Debate?" (2025)
- Key finding: Intrinsic reasoning strength and group diversity are dominant; structural parameters offer limited gains. Majority pressure suppresses independent correction.
- ATTS connection: Validates independent explores over debate-style interaction.

### 10. "Don't Overthink It" (2025)
- Key finding: Shorter reasoning chains up to 34.5% more accurate than longest. Short-m@k: stop when first m complete.
- ATTS connection: Supports ATTS's adaptive effort mechanism and early stopping.

### 11. ThinkPRM: Process Reward Models That Think (2025)
- Key finding: Generative PRM with CoT verification. 8% better than discriminative PRMs on GPQA-Diamond.
- ATTS connection: Tested on same benchmarks. ATTS's orchestrator plays a similar evaluative role.

### 12. SCoRe: Self-Correct via RL (ICLR 2025)
- Key finding: Multi-turn online RL trains self-correction. 15.6% gain on MATH.
- ATTS connection: Training-time alternative to ATTS's inference-time orchestration.

### 13. Inference Scaling Laws (ICLR 2025)
- Authors: Wu et al.
- Key finding: Smaller models + advanced inference offer Pareto-optimal tradeoffs. Llemma-7B + tree search > Llemma-34B.
- ATTS connection: Supports "small model + more TTC" paradigm.

### 14. Team of Thoughts (2026)
- Key finding: Orchestrator + specialized agents. 96.67% on AIME24.
- ATTS connection: Closest conceptual cousin — orchestrator delegates to specialists.

### 15. "Reasoning on a Budget: Survey of Adaptive TTC" (TMLR 2025)
- Key finding: L1 (fixed) vs L2 (adaptive) controllability taxonomy.
- ATTS connection: ATTS is L2-adaptive. Good for positioning.

### 16. MASTER: Multi-Agent MCTS (NAACL 2025)
- Key finding: Dynamic agent recruitment based on task complexity.
- ATTS connection: Analogous to ATTS's adaptive explore count.

## Context Papers (completeness)

### 17. Self-Correction Survey (TACL 2024)
- Confirms no successful intrinsic self-correction with prompted feedback.

### 18. "Stop Overthinking" Survey (TMLR 2025)
- Efficient reasoning taxonomy. ATTS fits "input-based efficient reasoning."

### 19. MAD / Degeneration-of-Thought (EMNLP 2024)
- ATTS avoids DoT by using independent solvers.

### 20. DMAD / Breaking Mental Set (ICLR 2025)
- Diversity of reasoning approach matters more than number of agents.

### 21. "It's Not That Simple" (2025)
- Naive TTS (append "Wait") doesn't truly scale. Agentic TTS genuinely expands solution space.

### 22. Optimal Bayesian Stopping (Feb 2026)
- L=3 answer tracking is asymptotically optimal. Cuts LLM calls by 50%.

### 23. Adaptive-Consistency (EMNLP 2023)
- Statistical stopping rule (Beta/Dirichlet). ATTS replaces with LLM judgment.

### 24. BEST-Route (ICML 2025)
- Joint model selection + sample count. Learned version of ATTS's multi-model variant.
