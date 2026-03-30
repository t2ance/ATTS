# ATTS Orchestrator: GRPO Training Design
Date: 2026-03-29
Source: rl-training-expert agent

## SFT Phase

Three-tier trajectory selection:
- Efficient-correct (60%): correct + low cost. Teach "stop when good enough"
- Persistent-correct (25%): correct + high cost. Hard problems, persistence paid off
- Informative-failures (15%): incorrect but clear stopping decisions

Behavioral stratification: easy -> 1-2 explores, medium -> 2-3, hard -> 3-5

Format: Full trajectory with orchestrator <think> at each decision point.

## GRPO Phase

### Reward Function (recommended)
R(tau) = 1[correct] - lambda * cost(tau) / C_ref
where C_ref = cost of N_max high-effort explores. Normalizes R in [-lambda, 1].

Why not alternatives:
- 1[correct] - lambda*num_explores: treats low/high effort identically (wrong incentive)
- 1[correct] * (1-cost/budget): multiplicative kills gradient
- Shaped intermediate rewards: circular (requires defining "good candidate")

### GRPO Setup
- K=16 trajectories per problem at T=0.9
- A_i = (r_i - mean(r)) / (std(r) + eps)
- loss = -E[A_i * log pi(tau_i | q)] + beta * KL(pi || pi_ref)
- beta=0.01, sweep lambda in {0.05, 0.1, 0.2}
- CRITICAL: advantage normalization must be per-problem

## What the Trained Policy Encodes

NOT a discriminator. The policy encodes 2D information:
- P(correct | stop here)
- E[P(correct | continue)]

It makes the differential comparison -- sequential decision-making under uncertainty.

| Behavior | What it actually is |
|----------|---------------------|
| Stop after consensus | P*(q, C_t) is high, VOI is low |
| Persist after disagreement | E[V*(s_{t+1})] - cost is still positive |
| Choose high-effort for ambiguous | Delta_P/cost comparison across effort levels |
| Final synthesis | Posterior aggregation: argmax_a P(correct | a, C_t) |

## Why ATTS > BoN (Formal Argument)

By Jensen's inequality on concave accuracy-cost frontier:
  E_q[acc(cost_q)] >= acc(E_q[cost_q])

ATTS achieves left side (adaptive). BoN achieves right side. Gap = Jensen gap, nonzero whenever difficulty is heterogeneous.

Savings factor examples:
- alpha=0.5 easy, N_max=4: 1.6x cost savings
- alpha=0.7 easy, N_max=4: 2.1x cost savings

## Convergence

- GRPO convergence to stationary point: O(1/sqrt(T))
- Suboptimality: E[R(pi_hat)] >= E[R(pi*)] - O(beta*KL + 1/sqrt(T))
- Practical: T=200-500 steps, N_p=3000-10000 problems, K=16

## Open Design Choices

| Decision | Recommendation | Sensitivity |
|----------|---------------|-------------|
| lambda | Sweep {0.05, 0.1, 0.2} | High |
| K | 16 | Medium |
| beta | 0.01 | Medium |
| SFT ratio | 60/25/15 | Low |
| Temperature | 0.9 | Medium |
