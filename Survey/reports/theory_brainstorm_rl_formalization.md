# ATTS as Meta-MDP: RL Formalization
Date: 2026-03-29
Source: rl-formalist agent

## Meta-MDP Definition

State: s_t = (x, H_t) where H_t = {(e_i, r_i)}_{i=1}^{t-1}
Action: A = {explore(e) : e in E} union {stop(a) : a in A}
Transition: T(s_{t+1} | s_t, explore(e)) = delta_x * P_exp(r | x, e)  (explorer independent given x,e)
Reward: Design-dependent (see below)

## Reward Designs

### Design A: R = 1[correct] - lambda * n
- Optimal policy: threshold rule in posterior space
- Stop when P(y* | H_t) >= tau*(lambda)
- lambda -> 0: always explore; lambda -> inf: stop immediately

### Design C: R = 1[correct] - lambda * sum cost(e_t)
- Most economically complete
- Optimal effort selection: e* = argmax_e E[Delta V | H_t, e] / cost(e), explore iff ratio >= lambda
- Induces adaptive effort: cheap explores early, expensive explores late

### Design D: GRPO relative reward
- Same fixed point as A/C in limit K -> inf
- Advantage: variance reduction without learned value function

## Emergent Capabilities (from reward, not assumed)

### 1. Implicit Posterior Maintenance
Bellman equation depends on H_t only through sufficient statistic T(H_t) = P(y* | H_t).
By Neyman-Pearson sufficiency theorem: any policy not computing T(H_t) is strictly suboptimal.
Therefore optimal policy MUST develop internal posterior representation.

### 2. Log-Likelihood Ratio Aggregation
Optimal aggregation: log P(a=y* | H_t) proportional to sum_{i:a_i=a} log P(r_i | y*=a, e_i) / P(r_i | y*!=a, e_i)
This is Condorcet-weighted voting generalized to heterogeneous effort levels.

### 3. Stopping Rule = Reservation Price (Chow-Robbins 1961)
tau* satisfies: tau* = tau* - lambda + E[max(tau*, P(y*|H_{t+1})) | P(y*|H_t) = tau*]
Continue exploring iff expected posterior gain > lambda.

### 4. VOI Computation
VOI(e | H_t) = E[V*(H_{t+1}) | H_t, e] - V_stop(H_t)
Explore iff VOI(e) > cost(e) for some e. This is EVSI (DeGroot 1962).

## Central Theorem

Under reward Design C with penalty lambda, optimal policy pi* satisfies:
1. Sufficiency: pi* depends on H_t only through posterior P(y* | H_t)
2. Threshold: pi* stops at first t where P(y* | H_t) >= tau*(lambda)
3. Pareto optimality: any policy with Accuracy >= Accuracy(pi*) has Cost >= Cost(pi*)
4. SPRT equivalence: for binary problems with i.i.d. explores, pi* = Wald's SPRT

Corollary (GRPO): pi_theta from GRPO with convergence epsilon satisfies:
  Accuracy(pi_theta) >= tau*(lambda) - epsilon
  E[n_{pi_theta}] <= tau*(lambda) / lambda + O(epsilon/lambda)

## Theoretical Chain

1. Reward -> Optimal Policy: Bellman theorem (strong, established)
2. Optimal Policy -> GRPO Convergence: moderate gap (no tight LLM finite-sample bound)
3. GRPO Convergence -> Inference Bound: direct (given convergence)

## Key Citations
- Wald (1945): Sequential tests of statistical hypotheses
- Chow & Robbins (1961): Optimal stopping
- Puterman (1994): Markov Decision Processes
- DeGroot (1962): Uncertainty and sequential experiments
- Agarwal et al. (2021): Theory of policy gradient methods
