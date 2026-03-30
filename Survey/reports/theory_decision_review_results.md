# Theory Proposal: Decision Review Results
Date: 2026-03-29
Method: 4-round Socratic questioning (first principles, Occam's razor, Socratic questioning)

## Claims Tested and Outcomes

| Original Claim | Status | Reason |
|---|---|---|
| Implicit posterior maintenance | Dropped | Unfalsifiable (revealed preference — any consistent behavior can be rationalized by some belief function) |
| Exchangeability test | Dropped | Self-defeating (LLMs are architecturally order-dependent due to causal attention) |
| Non-degradation unique to ATTS | Dropped | Applies to any adaptive stopping method, not ATTS-specific |
| "Between MV and ceiling" prediction | Dropped | Vacuous — any method using more info than frequency counts falls in this band |
| Quantitative savings formula | Dropped | Tautological — savings from adaptivity apply to any adaptive method |
| Minority extraction as theorem | Dropped | Formalizes the obvious ("more info -> better decisions"); needs statistical validation of denominator |
| Meta-MDP formalization | SURVIVES | As framework for positioning, not novel theorem |
| Submartingale remark | SURVIVES | As remark contrasting Choi et al.'s debate martingale. Near-trivial but clean |
| Existing Blackwell proofs (appendix) | SURVIVES | Simple, correct, directly connected to integrator ablation |
| GRPO training target | SURVIVES | Forward-looking engineering value — meta-MDP defines reward and optimal policy |

## Key Insights from Review

1. "Behaves as if it has a posterior" is trivially true by revealed preference — unfalsifiable
2. "LLM beats majority vote" is trivially true by data processing inequality — no theory needed
3. Most theoretical predictions (non-degradation, savings, between-curves) are properties of ANY adaptive method, not specific to ATTS
4. The paper is a SYSTEMS contribution with theoretical grounding, not a theory paper
5. Theory section should FRAME the results within known theory, not prove novel theorems

## Recommended Theory Section Structure

1-page "Theoretical Framework" subsection:
- Map ATTS to meta-MDP (state, action, reward formalization)
- Connect to Wald/SPRT (optimal stopping), Huang et al. (non-degradation of adaptive methods), Choi et al. (asymmetric vs symmetric debate)
- Submartingale observation as a remark (3 lines, not main theorem)
- Existing Blackwell proofs in appendix (keep as-is)

## Action Items

1. Compute minority extraction denominator: how many GPQA problems had correct answer as minority among explores? Is 4/N statistically significant?
2. Rewrite theory section from "we prove X" to "ATTS maps to the meta-MDP framework, connecting to these established results"
3. Keep submartingale as remark, not theorem
4. Frame GRPO training as future work enabled by the meta-MDP formalization
