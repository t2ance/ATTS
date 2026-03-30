# Theoretical Backing Analysis: Decision Theory / Optimal Stopping / Rational Metareasoning
Date: 2026-03-29
Source: decision-theorist agent

## Executive Summary

The best theoretical story combines Metalevel MDP as the primary framework (it provides the complete justification and directly models the architecture), Pandora's Box as a cleaner stopping sub-result, and Chernoff as a supporting argument for effort selection. The metalevel MDP framework has a critical structural advantage: ATTS's independent-explore design satisfies the blinkered independence assumption *exactly*, making the VOC computation not an approximation but the exact optimal criterion.

The most NeurIPS-compelling single theorem is an Adaptive-vs-Batch dominance result that directly explains the empirical integrator inferiority and follows cleanly from the metalevel MDP structure.

---

## Framework A: Pandora's Box (Weitzman 1979 / Kalayci et al. 2025)

### Mathematical Mapping

The Pandora's Box problem: N boxes with costs {c_i} and rewards {v_i ~ F_i}. The decision maker opens boxes sequentially, paying c_i to reveal v_i, and keeps the best reward found. Weitzman's reservation price sigma_i satisfies:

    E[(v_i - sigma_i)^+] = c_i

Optimal policy: open boxes in decreasing order of sigma_i; stop when the best observed reward exceeds the next sigma.

**ATTS mapping (homogeneous case, single effort level):**
- Each explore = one box
- c = API cost per explore
- v_i = quality of candidate answer from explore i (unobservable true quality)
- Observed signal s_i = (answer + reasoning + confidence) is a noisy proxy for v_i
- F = distribution over answer quality induced by the model

Under i.i.d. explores (same model, same problem), all reservation prices are equal. The single reservation price sigma satisfies E[(v - sigma)^+] = c. Optimal stopping: stop when observed quality exceeds sigma.

**ATTS mapping (heterogeneous effort levels):**
Each effort level e in {low, mid, high} defines a distinct "box type" with:
- Cost: c_e (API cost, higher for high-effort)
- Distribution: F_e (higher effort -> higher mean quality, lower variance)
- Reservation price: sigma_e, where E[(v - sigma_e)^+] = c_e

Under binary correct/incorrect with P_e = P(correct | effort e):

    sigma_e = 1 - c_e / P_e

Optimal effort selection: use effort e* = argmax_e sigma_e = argmax_e (P_e - c_e).

### Theorems

**Theorem A1 (Weitzman Effort Selection):** Under the Pandora's Box model with K effort levels, each with known accuracy P_e and cost c_e, the optimal ATTS effort policy uses effort e* = argmax_e (P_e - c_e) at every step, and stops when the orchestrator's confidence in the best answer exceeds sigma_{e*}. This policy is optimal among all sequential stopping policies in expected net value.

**Theorem A2 (Kalayci UCB):** When (P_e, c_e) are unknown but stationary, an ATTS orchestrator using UCB-based effort selection achieves regret O(sqrt(KT log T)) against the oracle that knows (P_e, c_e), where T is the number of explores and K is the number of effort levels. This is tight (matching lower bound from Kalayci et al., 2025).

### Strength and Tightness

- Theorem A1: Tight, direct corollary of Weitzman's classical result. Clean and simple.
- Theorem A2: Tight in the minimax sense (Kalayci et al. proved the matching lower bound).
- The binary utility simplification (correct/incorrect) is lossy but gives clean closed-form sigma_e.

### Assumptions and Realism

| Assumption | Status |
|------------|--------|
| Explores are i.i.d. draws from F_e | TRUE by ATTS architecture (each explore is an independent API call) |
| Distribution F_e is stationary across explores | TRUE for fixed model + problem |
| Orchestrator observes a signal monotone in v_i | PLAUSIBLE but not verified |
| Opening order is unrestricted (no memory effects) | TRUE -- independent explores have no ordering effects |

### Narrative Fit

Moderate. Pandora's Box cleanly supports "optimal sequential resource allocation" and gives the reservation price as an explicit formula. However, it is inherently a rule-based framework -- you precompute sigma offline and follow a threshold rule. This misses the richness of "agentic closed-loop decision-making." Additionally, does not explain why the integrator is worse.

---

## Framework B: Metalevel MDP / Rational Metareasoning (Russell & Wefald 1991 / Hay et al. 2012)

### Mathematical Mapping

- State: (q, H_t) where q = problem, H_t = {(a_i, r_i, conf_i)}_{i=1}^t = history of t explore results
- Base-level belief: b_t(a) = P(a is correct | q, H_t)
- Current best: U_t = max_a b_t(a)
- Actions: {explore(e) | e in E} union {stop(a) | a in candidates}
- Utility: U(stop(a)) = 1[a correct], cost c_e per explore(e)

**Value of Computation:**

    VOC(e, b_t) = E_{s ~ p(.|q,e)}[max(U_t, quality(s))] - U_t - c_e

**Critical structural observation:**

Since each ATTS explore is a fresh independent API call, the result of the next explore is statistically independent of H_t conditioned on q:

    p(s_{t+1} | q, H_t, e) = p(s_{t+1} | q, e)

This is the blinkered independence assumption from Hay et al. (2012), and for ATTS it holds EXACTLY (not as an approximation), because of the architectural guarantee that explores are independent.

Consequence: VOC decomposes into a single-step calculation:

    VOC(e, b_t) = E_v[(v - U_t)^+] - c_e

This is identical in form to Weitzman's equation. The VOC equals the expected gain from opening one more box of type e, minus cost.

### Theorems

**Theorem B1 (Exact VOC):** Under ATTS's independent-explore architecture, the blinkered VOC is the exact (not approximate) optimal metalevel decision criterion. The orchestrator should explore with effort e iff VOC(e, b_t) > 0:

    Explore iff P_e / c_e > 1 / (1 - U_t)

i.e., the information-cost ratio exceeds the inverse confidence gap.

**Theorem B2 (Expected Explores Bound):** Under the optimal ATTS policy:

    E[N*] <= (1 - P_1) / c

where P_1 is single-explore accuracy. (From Hay et al.'s general bound E[N*] <= (E[U*] - U_0) / c.)

**Theorem B3 (Orchestrator Strictly Dominates Integrator):** For any accuracy target alpha and any problem with P(first explore correct) > 0:

    Expected_cost(pi_seq, target=alpha) < Expected_cost(pi_batch(N), accuracy=alpha)

Proof sketch:
1. pi_seq is the optimal metalevel MDP policy (by Theorem B1)
2. pi_batch(N) is a feasible but non-adaptive policy in the same MDP
3. Standard MDP theory: adaptive policies weakly dominate non-adaptive ones
4. Strict dominance holds whenever early stopping has positive probability

### Strength and Tightness

- Theorem B1: Exact (not a bound) given the independence structure
- Theorem B2: Not tight, but gives right order of magnitude and is directly testable
- Theorem B3: Dominance direction is tight; magnitude depends on problem distribution

### Assumptions and Realism

| Assumption | Status |
|------------|--------|
| Explores are conditionally independent given q | EXACTLY TRUE (architectural guarantee) |
| Utility is binary (correct/incorrect) | Simplified but clean |
| VOC-based stopping is implemented by the LLM orchestrator | ASSUMPTION -- we cannot prove the LLM computes VOC |
| Cost is deterministic per effort level | TRUE -- API pricing is fixed |
| Orchestrator beliefs are calibrated | WEAKENED: result holds under monotone ranking |

### Narrative Fit

Excellent. Metalevel MDP is the most direct mathematical instantiation of "agentic closed-loop decision-making":
- "Closed-loop" = orchestrator observes results and updates beliefs before next action
- "Decision-making" = orchestrator evaluates VOC to decide whether to continue
- "Extensible action space" = new effort levels, model choices, tools are new meta-actions with their own VOC

---

## Framework C: Chernoff Sequential Experiment Design (1959)

### Mathematical Mapping

Chernoff's setup: unknown state theta in Theta, experiments e in E with results X ~ p(X | theta, e). Goal: identify theta with P(error) <= delta using minimum expected experiments.

Chernoff's 1959 result -- asymptotically optimal adaptive policy:

    e_t* = argmax_e Sum_{i!=j} pi_i pi_j E[KL(p(X|theta_i,e) || p(X|theta_j,e))]

Expected cost: E[T*] = (1 + o(1)) * H(theta) / C*

**ATTS mapping (multiple-choice, e.g. GPQA):**
- theta = correct answer option (A/B/C/D)
- e = effort level (low/mid/high)
- X = probability distribution over options from explore
- Goal: identify theta with minimum expected explores

### Theorem

**Theorem C1:** For multiple-choice problems with K options, ATTS selecting effort e* = argmax_e C*(e, posterior_t) is asymptotically Bayes-optimal. E[T*] <= (1+o(1)) * H(answer) / C*(e*) as delta -> 0.

### Strength

- Asymptotically tight (Chernoff proved matching lower bound)
- Finite-sample behavior not characterized -- for 2-10 explores, asymptotic guarantee says little
- Only applicable to multiple-choice, not open-ended tasks

### Narrative Fit

Moderate. Clean information-theoretic story for effort selection specifically, but does not address stopping criteria, sequential vs. batch comparison, or open-ended tasks.

---

## Synthesis: Recommended Theoretical Package

### The Unified Story

Level 1 (Metalevel MDP): Orchestrator is a rational metareasoner. Evaluates VOC for each effort level, stops when VOC <= 0.

Level 2 (Pandora's Box): VOC stopping reduces to Weitzman's reservation price. The two frameworks agree exactly on the stopping rule.

Level 3 (Chernoff): For multiple-choice benchmarks, effort selection within each step maximizes expected KL divergence -- Chernoff-optimal experiment design.

### The Elegant Combined Theorem

**Theorem (ATTS Optimality):** Consider ATTS with K effort levels, costs {c_e}, and conditionally independent explores:
1. Selects effort e* = argmax_e VOC(e, b_t), stopping when max_e VOC(e, b_t) <= 0
2. Stopping threshold reduces to Weitzman reservation price
3. For multiple-choice, effort selection is asymptotically Chernoff-optimal

This policy strictly dominates any non-adaptive (batch) policy.

### What a NeurIPS Reviewer Finds Compelling

The most compelling result is Theorem B3 (Orchestrator > Integrator):
1. Directly connected to an empirical finding in the paper
2. Not a trivial consequence of a well-known result -- requires the specific ATTS structure
3. Clean and short proof
4. Makes a falsifiable prediction

### What NOT to Claim

- Do NOT claim the LLM orchestrator provably computes VOC
- Do NOT apply Chernoff to open-ended tasks
- Do NOT use Theorem B2 as a tight bound
- Present Pandora's Box and Metalevel MDP as the same result from two viewpoints
