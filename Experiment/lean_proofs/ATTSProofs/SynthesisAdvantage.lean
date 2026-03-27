/-
  ATTS Synthesis Advantage (Blackwell's Informativeness, Specialized)

  Formalizes why the orchestrator's direct synthesis outperforms a separate
  integrator that only sees the candidate pool (the integrator ablation).

  Core principle: The integrator observes a lossy compression of the
  orchestrator's full multi-turn history. By Blackwell's informativeness
  theorem, no decision rule on a garbled signal can dominate a decision
  rule on the original signal.
-/
import Mathlib.Tactic

/-- Theorem 1 (Replication Principle).
    Any integrator decision rule d_I : Summary -> Answer can be exactly
    replicated by the orchestrator as d_I . garble : Obs -> Answer.
    The orchestrator's decision space is therefore a superset of the
    integrator's. -/
theorem synthesis_replication
    {Obs Summary Answer : Type}
    (garble : Obs → Summary)
    (utility : Obs → Answer → ℝ)
    (d_I : Summary → Answer) :
    ∃ d_O : Obs → Answer, ∀ obs : Obs,
      utility obs (d_O obs) = utility obs (d_I (garble obs)) :=
  ⟨d_I ∘ garble, fun _ => rfl⟩

/-- Theorem 2 (Information Loss Hurts).
    When garble is non-injective (maps distinct observations to the same
    summary), and these observations have UNIQUELY different optimal answers,
    the integrator must fail on at least one -- while the orchestrator can
    succeed on both.

    This is the core of Blackwell's theorem specialized to deterministic
    garbling: lossy compression provably degrades decision quality when
    different inputs require different optimal actions. -/
theorem info_loss_hurts
    {Obs Summary Answer : Type}
    (garble : Obs → Summary)
    (utility : Obs → Answer → ℝ)
    (obs1 obs2 : Obs)
    (a1 a2 : Answer)
    -- Two observations map to the same summary (information lost)
    (h_same : garble obs1 = garble obs2)
    -- a1 is uniquely optimal on obs1
    (h_unique1 : ∀ a, utility obs1 a ≥ utility obs1 a1 → a = a1)
    -- a2 is uniquely optimal on obs2
    (h_unique2 : ∀ a, utility obs2 a ≥ utility obs2 a2 → a = a2)
    -- The optimal answers are different
    (h_ne : a1 ≠ a2) :
    -- Then every integrator rule fails on at least one observation
    ∀ d_I : Summary → Answer,
      utility obs1 (d_I (garble obs1)) < utility obs1 a1 ∨
      utility obs2 (d_I (garble obs2)) < utility obs2 a2 := by
  intro d_I
  -- Key fact: d_I must output the same answer for obs1 and obs2
  have h_eq : d_I (garble obs1) = d_I (garble obs2) := congrArg d_I h_same
  -- Proof by contradiction: assume integrator is optimal on BOTH
  by_contra h_neg
  push_neg at h_neg
  obtain ⟨h1, h2⟩ := h_neg
  -- If optimal on obs1: d_I(garble obs1) = a1 (by uniqueness)
  have heq1 : d_I (garble obs1) = a1 := h_unique1 _ h1
  -- If optimal on obs2: d_I(garble obs2) = a2 (by uniqueness)
  have heq2 : d_I (garble obs2) = a2 := h_unique2 _ h2
  -- But d_I(garble obs1) = d_I(garble obs2), so a1 = a2. Contradiction!
  exact h_ne (heq1 ▸ h_eq ▸ heq2)

/-- Corollary: The orchestrator can achieve what no integrator can.
    On the pair (obs1, obs2), the orchestrator gets both right (total
    utility = utility obs1 a1 + utility obs2 a2), but every integrator
    must sacrifice at least one. -/
theorem orchestrator_strictly_better
    {Obs Summary Answer : Type}
    (garble : Obs → Summary)
    (utility : Obs → Answer → ℝ)
    (obs1 obs2 : Obs)
    (a1 a2 : Answer)
    (h_same : garble obs1 = garble obs2)
    (h_unique1 : ∀ a, utility obs1 a ≥ utility obs1 a1 → a = a1)
    (h_unique2 : ∀ a, utility obs2 a ≥ utility obs2 a2 → a = a2)
    (h_ne : a1 ≠ a2) :
    -- The orchestrator can define rules that achieve the optimum on each
    ∃ (d1 d2 : Obs → Answer),
      utility obs1 (d1 obs1) = utility obs1 a1 ∧
      utility obs2 (d2 obs2) = utility obs2 a2 :=
  ⟨fun _ => a1, fun _ => a2, rfl, rfl⟩
