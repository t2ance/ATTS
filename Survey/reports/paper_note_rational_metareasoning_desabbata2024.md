# Paper Note: Rational Metareasoning for Large Language Models
Authors: De Sabbata, Sumers, AlKhamissi, Bosselut, Griffiths (Princeton, EPFL, Anthropic)
ArXiv: 2410.05563v3, June 2025
Status: Under review

## Main Contribution

RaM (Rational Metareasoning): trains LLMs to use intermediate reasoning steps SELECTIVELY when they are likely to be beneficial. Uses a VOC-inspired reward function within Expert Iteration.

## Key Formulation

### VOC-Inspired Reward
R(x, y, z) = U(z | x, y) - C(z)

where:
- x = input (question)
- y = target answer
- z = chain of thought (reasoning chain)
- U(z | x, y) = log pi(y | z, x) - log pi(y | x)  (utility = log-prob improvement from CoT)
- C(z) = gamma * l(z)  (cost = proportional to token count)

### Key Properties
- Utility = how much the reasoning chain improves the model's probability of the correct answer
- Cost = proportional to reasoning chain length (in tokens)
- Reward is self-referential: parameterized by the same model weights theta (no external reward model needed)
- VOC > 0 means reasoning helps; VOC < 0 means reasoning is wasteful

### Training: Expert Iteration
1. Generate K reasoning chains per problem using current policy
2. Score with R(x, y, z), compute advantage: a_{i,k} = r_{i,k} - mean(r)
3. Rejection sampling: keep chains with positive advantage
4. Fine-tune on selected chains
5. Repeat N iterations

## Granularity

**TOKEN-LEVEL within a single generation.** Each "computation" = one reasoning token. The model decides whether to generate reasoning tokens before answering. NOT candidate-level.

This is fundamentally different from ATTS:
- RaM: single model, decides how much to reason WITHIN a single generation
- ATTS: orchestrator model, decides how many SEPARATE candidate generations to request

## Results

- 23-45% fewer tokens generated vs CoT/STaR
- Matching or improved accuracy
- Adaptive: hard problems get more reasoning tokens, easy problems get fewer
- Models: Llama-3.2-3B, Llama-3.1-8B
- Benchmarks: ARC, CommonsenseQA, GSM8K, ProofWriter, MMLU-CF

## Connection to Russell & Wefald / Hay et al.

- Directly builds on rational metareasoning (Russell & Wefald 1991)
- VOC formula is a direct instantiation of the classical VOC
- Computation c = individual reasoning token
- Agent's belief state b = model's probability distribution over answers
- VOC(z) = improvement in belief quality - cost of computing z

## Key Differences from ATTS

| Aspect | RaM (De Sabbata et al.) | ATTS (ours) |
|---|---|---|
| Granularity | Token-level (within single generation) | Candidate-level (across multiple generations) |
| Architecture | Single model, self-modulated reasoning | Orchestrator + independent explorers |
| Computation type | Reasoning tokens (think more/less) | Separate API calls (explore more/fewer) |
| VOC computation | Explicit (log-prob improvement) | Implicit (orchestrator judges quality) |
| Training | Expert Iteration with VOC reward | Currently zero-shot; future GRPO |
| Action space | Length of reasoning chain | {explore(effort), stop(answer)} |
| Independence | NOT independent (tokens are sequential) | Independent explores (stateless explorers) |

## Critical Observation for ATTS Paper

RaM is "rational metareasoning for INTRA-generation reasoning" (how much to think within one call).
ATTS is "rational metareasoning for INTER-generation resource allocation" (how many calls to make, what kind).

Both are instantiations of Russell & Wefald's framework but at DIFFERENT GRANULARITIES:
- RaM: metalevel MDP over reasoning tokens
- ATTS: metalevel MDP over candidate generations

This means we should NOT claim to be "the first to apply rational metareasoning to LLMs" — RaM already did that. Our claim should be: "we extend rational metareasoning from intra-generation (token-level) to inter-generation (candidate-level) with heterogeneous computation types."

## Limitation They Acknowledge (relevant to us)

From Section 7: "One particularly relevant example is the agentic setting, where LLMs act autonomously in complex digital environments. Adapting our method to this context would require incorporating the cost of tool use (e.g., API calls) into the reward function."

This is EXACTLY what ATTS does. We are the agentic extension they call for as future work.
