/-
  ATTS Coverage Guarantee

  Theorem: If each explore independently succeeds with probability p,
  then after n >= 2 independent explores, the probability of at least
  one success is at least 1 - (1-p)^2.

  This formalizes the key property that even 2 explores suffice to
  substantially improve over Pass@1 (a single explore).
-/
import Mathlib.Analysis.SpecialFunctions.Pow.Real
import Mathlib.Tactic

/-- (1-p)^n is non-increasing in n when 0 <= p <= 1. -/
theorem coverage_monotone (p : ℝ) (hp0 : 0 ≤ p) (hp1 : p ≤ 1) (n : ℕ) (hn : 2 ≤ n) :
    (1 - p) ^ n ≤ (1 - p) ^ 2 := by
  have h0 : 0 ≤ 1 - p := by linarith
  have h1 : 1 - p ≤ 1 := by linarith
  exact pow_le_pow_of_le_one h0 h1 hn

/-- P(at least one success in n trials) >= P(at least one success in 2 trials) for n >= 2. -/
theorem coverage_guarantee (p : ℝ) (hp0 : 0 ≤ p) (hp1 : p ≤ 1) (n : ℕ) (hn : 2 ≤ n) :
    1 - (1 - p) ^ 2 ≤ 1 - (1 - p) ^ n := by
  linarith [coverage_monotone p hp0 hp1 n hn]

/-- The lower bound improves over Pass@1: 1-(1-p)^2 = p(2-p) >= p for p in [0,1]. -/
theorem coverage_improves_over_pass1 (p : ℝ) (hp0 : 0 ≤ p) (hp1 : p ≤ 1) :
    p ≤ 1 - (1 - p) ^ 2 := by
  nlinarith [sq_nonneg p]
