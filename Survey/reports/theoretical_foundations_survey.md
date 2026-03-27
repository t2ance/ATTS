# Theoretical Foundations for ATTS: Literature Survey
Date: 2026-03-27

## 1. Rational Metareasoning (Russell & Wefald 1991)

**Core framework:** Value of Computation (VOC) = E[utility improvement from computation] - cost. Agent stops computing when no computation has positive VOC. Formalized as a metalevel MDP.

**Key formula:**
```
VOC(c, b) = E_{P(b'|c)}[ max_a E[U(a)|b'] ] - max_a E[U(a)|b] - cost(c)
```

**Hay et al. (UAI 2012) "Selecting Computations"** — most rigorous formal treatment:
- Metalevel MDP: M = (S, s_0, A_s, T, R) where actions = {computations} ∪ {stop}
- **Theorem (Finite Sampling Bound):** E[N | optimal policy] <= (1/c)(E[max U_i] - max μ_i)
- **Theorem 7 (Myopic-Optimal):** If myopic policy computes in state s, optimal does too
- Blinkered VOC approximation decomposes multi-action problem into independent subproblems

**De Sabbata et al. (arXiv 2410.05563)** — VOC for LLM token-level reasoning:
- R(x,y,z) = U(z|x,y) - C(z), utility = log-prob improvement, cost = γ·token_length
- Empirical only (no theorems), 20-37% token reduction
- Operates at token granularity vs ATTS at candidate granularity

**Connection to ATTS:** ATTS is a metalevel MDP where state = (problem, candidate pool), actions = {explore, stop}, cost = API cost per explore. Hay's finite sampling bound directly applies.

---

## 2. Optimal Stopping for LLM Inference

### 2a. Pandora's Box (Kalayci et al., arXiv 2510.01394)
- Each LLM generation = opening a box with cost c and random reward
- **Weitzman reservation price τ:** E[(v-τ)+] = c. Stop when max reward >= τ
- **Theorem 5 (UCB regret bound):** Explicit bound for unknown distributions
- Achieves same accuracy as Best-of-N with **15-35% fewer generations**

### 2b. BEACON: Bayesian Optimal Stopping (arXiv 2510.15945)
- **Theorem 1:** Continue sampling iff h_{n,k}(ẑ_k) > c/σ_k
- Sufficient statistics: (best reward, posterior mean, posterior std)
- Normal-Inverse-Gamma conjugate prior

### 2c. Certified Self-Consistency / MMC (arXiv 2510.17472)
- **Theorem 2.2:** P[majority wrong] <= (k-1)·exp(-n·δ²/2), δ = mode margin
- **Theorem 3.1:** MMC e-process is test supermartingale; anytime-valid by Ville's inequality
- **Expected sample size:** N ~ 2(p_c+p_j)/(p_c-p_j)² · log(1/ε)
- Most rigorous LLM stopping formalization

### 2d. ConSol: SPRT for LLM Consistency (arXiv 2503.17587)
- Applies Wald's SPRT to test dominant answer existence
- H0: p'=0.5 vs H1: p'>0.5 using likelihood ratio
- 63.9-88.7% token reduction over self-consistency@40

### 2e. Optimal Bayesian Stopping (Huang et al., arXiv 2602.05395)
- L-aggregated stopping policy tracking L-1 most frequent answer counts
- L=3 suffices for asymptotic optimality
- Up to 50% reduction in LLM calls

---

## 3. Regret Bounds for Adaptive Compute

### 3a. Huang et al. (ICML 2025) "Is Best-of-N the Best of Them?"
- **Coverage Coefficient:** C*(x) = E[π*(y|x)/π_ref(y|x)]
- **Theorem 3.1 (BoN upper bound):** Regret <= R_max·C*·log(R_max/ε_RM)/N + √(N·ε_RM²)
- **Theorem 3.2 (BoN degrades):** Regret >= Ω(√(N·ε_RM²)) — BoN provably fails with more samples
- **Theorem 4.1 (Optimal):** InferenceTimePessimism achieves √(C*·ε_RM²) — matches minimax lower bound
- **Proposition 2.1 (Lower bound):** Any algorithm has regret >= (1/4)√(C*·ε_RM²)

### 3b. Zuo & Zhu (arXiv 2506.12721) "Strategic Scaling via Bandits"
- **Theorem 1:** Adaptive needs Õ(Σ 1/Δ_x) budget; uniform needs Θ̃(|S|/max Δ_x)
- Adaptive can be up to **|S| times more efficient**

### 3c. Hashimoto (arXiv 2502.17578) "Power Laws for Large Language Monkeys"
- **Theorem 3.1:** p_D(p) ~ C·p^{b-1} near 0 => -log(pass@k) ~ C·Γ(b)·k^{-b}
- **Theorem 3.2 (Converse):** Power-law scaling => power-law difficulty distribution
- Formally explains why coverage scales as power law

### 3d. Best-of-Majority (arXiv 2510.03199)
- **Theorem 4.1:** Majority voting can have Ω(1) constant regret regardless of N
- **Theorem 4.2:** BoN has regret >= Ω(min{1, √(N·ε_RM²/k)}) — grows with N
- **Theorem 5.1:** Best-of-Majority achieves regret ε_opt + O(√(C*·ε_RM²/k))

---

## 4. Sequential vs Batch Observation Theory

### 4a. Wald-Wolfowitz SPRT Optimality (1948)
- Among all sequential tests with type-I error <= α and type-II error <= β, SPRT minimizes E[N] under both hypotheses simultaneously
- Relative efficiency: SPRT saves **36-75%** samples vs best fixed-sample test
- f(0+) = 1/4 (75% savings at extreme α), f(1/2-) = 2/π ≈ 0.64 (36% savings)

### 4b. Blackwell's Informativeness Theorem (1953)
- σ' is garbling of σ => W(σ') <= W(σ) for every decision problem
- **Directly explains integrator ablation:** integrator receives garbled (lossy) version of orchestrator's full history

### 4c. Data Processing Inequality
- X → Y → Z Markov chain => I(X;Y) >= I(X;Z)
- Integrator ablation: (problem, full history) → (candidate pool summary) → (integrator output)
- Information bottleneck provably limits integrator quality

### 4d. Circuit Complexity of Chain-of-Thought (Li et al., ICLR 2024; Merrill & Sabharwal, ICLR 2024)
- Without CoT: constant-depth transformers solve only AC⁰
- With T steps CoT: solve circuits of size T
- With polynomial steps: solve all of P
- Multi-turn orchestrator = more computational depth than single-turn integrator

### 4e. Chernoff's Sequential Experiment Design (1959)
- Adaptive experiment selection based on current posterior is asymptotically optimal in Bayes risk
- Analogous to ATTS choosing exploration strategy based on accumulated evidence

---

## 5. Summary: Applicable Results for ATTS Paper

### Explaining "adaptive stopping > fixed-N"
| Result | Source | Type |
|--------|--------|------|
| SPRT minimizes expected samples | Wald-Wolfowitz 1948 | Proved |
| Pandora's Box reservation price | Kalayci et al. 2025 | Proved (UCB bound) |
| Adaptive budget O(Σ1/Δ) vs uniform O(n/maxΔ) | Zuo & Zhu 2025 | Proved |
| Metalevel MDP finite sampling bound | Hay et al. 2012 | Proved |
| BoN provably degrades | Huang et al. ICML 2025 | Proved |

### Explaining "orchestrator synthesis > separate integrator"
| Result | Source | Type |
|--------|--------|------|
| Blackwell informativeness (garbling loses info) | Blackwell 1953 | Proved |
| Data Processing Inequality | Shannon | Proved |
| CoT computational depth (AC⁰ vs P) | Li et al. ICLR 2024 | Proved |

### Explaining "ATTS doesn't degrade with more compute"
| Result | Source | Type |
|--------|--------|------|
| BoN regret ~ √(N·ε²) grows with N | Huang et al. 2025 | Proved |
| InferenceTimePessimism is scaling-monotonic | Huang et al. 2025 | Proved |
| MMC anytime-valid stopping | Cordero-Encinar 2025 | Proved |

---

## 6. Recommendation for Paper

**Route A (practical, 1-2 days):** Replace trivial coverage theorems in appendix with "Theoretical Connections" section citing Wald, Blackwell, Hay, Huang as formal grounding for ATTS's empirical claims. No new proofs needed.

**Route B (ambitious, 1-2 weeks):** Prove an ATTS-specific stopping guarantee using MMC framework: adaptive stopping achieves accuracy 1-ε with expected explores N ~ O(log(1/ε)/KL(p_mode||p_runner-up)).

**Route C (Lean formalization, weeks-months):** Formalize SPRT or metalevel MDP in Lean 4 — novel formalization contribution but high effort.

## 7. Key References (BibTeX keys to add)

- hay_selecting_computations_2012
- kalayci_optimal_stopping_bon_2025
- beacon_bayesian_optimal_stopping_2025
- cordero_certified_self_consistency_2025
- consol_sprt_llm_2025
- zuo_strategic_scaling_bandits_2025
- hashimoto_power_laws_monkeys_2025
- best_of_majority_2025
- huang_optimal_bayesian_stopping_2026
- li_cot_circuit_complexity_2024
- merrill_expressive_power_cot_2024
- wang_dora_rollout_allocation_2025
