# Theoretical Backing Analysis: Online Learning / Bandits / Regret Bounds
Date: 2026-03-29
Source: bandit-theorist agent

## Executive Summary

Recommended structure: Main Theorem = ATTS achieves O(sqrt(C* * epsilon'^2)) regret per problem, independent of N (Framework B). Key Corollary = sequential ATTS regret is O(epsilon' * sqrt(log k)) while batch integrator regret is Omega(epsilon * sqrt(log N)) with N >> k. Efficiency Theorem = over heterogeneous problems, ATTS total budget is O(Sum 1/Delta_i) vs BoN's Theta(N_prob / min Delta_i). Supporting Lemma = ATTS's stopping rule implements approximately SPRT-optimal sequential test.

Most novel angle: Generative BAI -- no existing paper has formalized best-arm identification over a generative arm distribution.

---

## Framework A: Best-Arm Identification (BAI)

### The Mapping Problem

The naive BAI mapping is wrong. Standard BAI assumes fixed stochastic arms. ATTS violates this -- the orchestrator never re-pulls the same candidate; it generates NEW candidates each time.

The correct mapping is Generative BAI:
- Each explore draws (answer, reasoning, confidence) ~ Pi(. | problem x, effort e_k), a fresh sample
- The orchestrator maintains a posterior P(correct | observed candidates 1:k)
- Stopping: posterior of MAP answer exceeds threshold 1-delta

This maps precisely to Wald's SPRT.

### Theorems

**Theorem A1 (SPRT-Optimal Stopping):** Let p = P(correct in one explore | x). The SPRT stopping rule achieves:

    E[K_SPRT] = O(log(1/delta) / D_KL(correct || incorrect))

Tight by Wald-Wolfowitz.

**Theorem A2 (Aggregate Efficiency):** Over N_prob problems with heterogeneous p_1, ..., p_{N_prob}:

    E[total explores, ATTS] = O(Sum_i log(1/delta) / p_i)
    B_BoN = N_prob * log(1/delta) / log(1/(1-min_i p_i))

Efficiency ratio approaches N_prob when hardness is highly heterogeneous.

### Assessment

Good supporting lemma. The genuinely novel angle: Generative BAI has not been formalized. Each pull creates a new candidate from the LLM's distribution -- new theoretical setting distinct from standard BAI.

---

## Framework B: Coverage Coefficient / Regret (Huang et al. ICML 2025)

### The Mapping

This is the strongest framework for the paper's narrative.

- pi* = optimal policy (oracle picks correct answer)
- pi_base = base LLM (single forward pass)
- Coverage coefficient C*(x) = max_y [pi*(y|x) / pi_base(y|x)]
- Verifier error epsilon: |r(y,x) - 1[y correct]| <= epsilon

Huang et al. show:
- BoN: Regret >= Omega(sqrt(N * epsilon^2)) -- grows without bound as N -> infinity
- ITP: Regret <= O(sqrt(C* * epsilon^2)) -- minimax optimal, independent of N

ATTS mapping:
- The "verifier" is the orchestrator's judgment -- an ENRICHED verifier seeing reasoning chains and confidence
- Enriched verifier error epsilon' where epsilon' <= epsilon (richer signal => lower error)
- Adaptive stopping prevents verifier exploitation

### Theorems

**Theorem B1 (ATTS Regret Under Enriched Verifier):**

    Regret_ATTS <= O(sqrt(C* * (epsilon')^2))

Since epsilon' <= epsilon and bound is independent of N:

    Regret_ATTS <= Regret_ITP <= Regret_BoN (for large N)

**Theorem B2 (Why Sequential Beats Batch) -- most important:**

Batch integrator = BoN with fixed aggregator. Expected regret of maximum of N i.i.d. noisy signals grows as O(epsilon * sqrt(log N)) -- diverges with N.

Sequential ATTS stops when running posterior exceeds 1-delta, NOT on maximum over all N. Self-regularizing.

    E[Regret_batch] >= Omega(epsilon * sqrt(log N))
    E[Regret_ATTS] <= O(sqrt(C* * epsilon^2))

### Strength

Strongest result because:
1. Connects to published ICML 2025 lower bound for BoN
2. Formally explains sequential > batch from first principles
3. Coverage coefficient C* well-studied, connects to offline RL literature

### Critical Assumption

The entire framework rests on: the orchestrator functions as a better-than-scalar verifier. Empirically testable: measure orchestrator selection accuracy. If ~85%, then epsilon' ~ 0.15.

---

## Framework C: Adaptive Budget Allocation (Zuo & Zhu 2025)

### The Mapping

- Delta_i = P(correct in one explore | problem x_i)
- ATTS adaptively allocates K_i explores based on observed difficulty
- Uniform BoN allocates exactly N to each problem

### Theorem

**Theorem C1 (Adaptive Efficiency):** ATTS achieves accuracy >= 1-delta with expected total explores:

    E[B_ATTS] = O(Sum_i log(1/delta) / log(1/(1-Delta_i)))

vs BoN-uniform:

    B_BoN = N_prob * log(1/delta) / log(1/(1-min_i Delta_i))

Efficiency ratio B_BoN / E[B_ATTS] >= 1 (by AM-HM inequality), equality only when all problems have equal hardness.

Example with half-easy (Delta=0.9), half-hard (Delta=0.1): gain ~1.8x. With highly skewed hardness: gain approaches O(N_prob).

### Narrative Fit

Excellent for cost-efficiency claims. Directly explains "ATTS achieves higher accuracy than BoN at lower cost."

---

## Framework D: Contextual Bandit for Effort/Model Selection

### The Mapping

- Context: problem embedding x
- Action space: {(effort, model)} pairs
- Reward: P(correct) / cost
- Policy: pi(context, history) -> next action

Under linear contextual bandit: Regret(T) = O_tilde(sqrt(T * |A| * d))

### Assessment

Standard result, not novel. Linearity assumption likely false for LLMs. Useful as supporting remark, not main theorem.

---

## Framework E: Online Convex Optimization

Not recommended. Loss function not convex in tau. OCO is adversarial but ATTS problems are i.i.d. Oracle comparison (fixed tau) is weak baseline. Skip entirely.

---

## Summary Table

| Framework | Result Quality | Novelty | Assumption Realism | Narrative Fit | Recommendation |
|---|---|---|---|---|---|
| A: Generative BAI (SPRT) | Tight optimal | High | Moderate (i.i.d. fragile) | Good | Supporting lemma |
| B: Coverage/Regret (Huang) | Strong | Medium | Moderate (orch quality) | Excellent | **Main theorem** |
| C: Adaptive Budget | Moderate | Medium | Good | Excellent | **Efficiency corollary** |
| D: Contextual Bandit | Standard | Low | Moderate | Moderate | Appendix/remark |
| E: OCO | Weak | Low | Poor | Poor | Skip |

---

## Open Questions

1. What exactly does the batch integrator do -- majority vote, weighted scoring, or LLM synthesis?
2. Have you measured orchestrator selection accuracy? This gives concrete epsilon'.
3. Does "extensible action space" mean (a) add new models/tools at deployment, or (b) adaptive selection over fixed set?
4. Do you want to prove ATTS is optimal, or just better than BoN?
