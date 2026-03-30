# Style Analysis: Top ML Papers vs. Our ATTS Paper

Analysis of ~57 arXiv LaTeX sources focusing on five high-profile papers:
- **DeepSeek-R1** (2501.12948)
- **GPQA** (2311.12022)
- **Large Language Monkeys** (2407.21787)
- **Rational Metareasoning** (2410.05563, RaM)
- **RSA** (2509.26626, Recursive Self-Aggregation)

---

## 1. Emphasis Patterns (Bold/Italic in Body Text)

### Findings

**`\emph{}` is the dominant emphasis tool in body text.** Across all five papers, `\emph{}` is used sparingly but consistently for:
- **Italicizing key technical concepts on first or defining use**: RSA uses `\emph{sequential}`, `\emph{parallel}`, `\emph{hybrid}`, `\emph{self-aggregation}`, `\emph{aggregation-aware}`. RaM uses `\emph{rational metareasoning}`, `\emph{adaptive}`, `\emph{and the cost}`. Our paper already does this well.
- **Italicizing a word for rhetorical emphasis**: RSA: "can result in \emph{worse} performance"; RaM: "metareasoning solves the problem of \emph{how} to solve a problem"; Large Language Monkeys: "the \textit{success rate:} the fraction of problems..."

**`\textbf{}` in body text is reserved for specific structural purposes:**
- **Enumerated list items / named categories**: Large Language Monkeys: `\textbf{Coverage:}`, `\textbf{Precision:}`, `\textbf{GSM8K:}`, `\textbf{SWE-bench Lite:}`. DeepSeek-R1: `\textbf{Accuracy rewards}`, `\textbf{Format rewards}`, `\textbf{Token efficiency:}`.
- **Method name introduction in abstract**: RaM: `\textbf{\method} (\textbf{Ra}tional \textbf{M}etareasoning)`.
- **Bold is almost never used for inline emphasis in running prose.** No paper bolds key findings or results within sentences.

**`\paragraph{}` is heavily used as inline section headers**, especially in Results and Related Work:
- GPQA uses it extensively: `\paragraph{Question Writing}`, `\paragraph{Expert Accuracy}`, `\paragraph{Discussion}`.
- Self-Consistency (2203.11171) uses it for sub-results: `\paragraph{Arithmetic Reasoning}`, `\paragraph{Comparison to Beam Search}`.
- RSA uses it for contribution enumeration in the intro: `\paragraph{Self-aggregation.}`, `\paragraph{Recursive self-aggregation.}`, `\paragraph{Aggregation-aware RL.}`.
- Our paper uses `\paragraph{}` well (35 instances), consistent with top papers.

### Comparison with Our Paper

Our paper's emphasis usage is largely consistent with top papers. We use `\emph{}` for key concepts (e.g., `\emph{control}`, `\emph{solution generation}`, `\emph{when}`, `\emph{what}`) and `\textbf{}` only in tables for best-result highlighting. This is correct practice.

**One gap**: We don't use `\emph{}` to highlight contrast or surprise in the way RSA and RaM do. See recommendation 5.1 below.

---

## 2. Narrative/Storytelling Techniques

### How Top Papers Open Their Introductions

| Paper | Opening Strategy |
|-------|-----------------|
| DeepSeek-R1 | **Grand challenge + capability gap**: "Reasoning capability, the cornerstone of human intelligence..." then immediately identifies the limitation of current approaches (dependence on human annotations). |
| GPQA | **Dataset motivation via scalable oversight**: Opens with the benchmark's purpose, immediately establishing the "Google-proof" difficulty framing. |
| Large Language Monkeys | **Contrast between training and inference scaling**: "Scaling the amount of compute used to train language models has dramatically improved their capabilities. However, when it comes to inference, we often limit models to making only one attempt." This creates a tension the paper resolves. |
| RaM | **Cost tension + cognitive science analogy**: Opens with the cost-performance tradeoff, then draws the human cognition parallel ("In stark contrast to this static tradeoff, humans are able to adaptively allocate..."). |
| RSA | **Scaling law + gap identification**: Opens with training compute scaling, then identifies the gap in test-time scaling methods ("a universal and effective test-time-scaling method that allows reuse of promising fragments from multiple candidate solutions is lacking"). |

**Common pattern**: All five papers use a 2-3 sentence setup that creates a **tension** or **gap**, then introduce their contribution as the resolution. None opens with a literature taxonomy or survey-style enumeration.

### How They Frame Contributions

- **Large Language Monkeys**: Uses a numbered `\begin{enumerate}` list at the end of the introduction with 3 concise items. Each item starts with "We demonstrate/show that..."
- **RaM**: Uses a numbered list with 3 items, each starting with "We employ/formalize/demonstrate..."
- **RSA**: Uses `\paragraph{}` headers for each of 3 contributions in the intro body itself, with a brief summary paragraph at the end pointing to sections.
- **DeepSeek-R1**: No explicit contribution list. Instead, the intro is structured as a narrative with "we show/introduce/enable" statements woven into the text.

### Roadmap Paragraphs

- Most papers do NOT have an explicit "The rest of this paper is organized as follows..." paragraph.
- RSA weaves section references naturally: "...described in the following paragraphs", then uses `(\Cref{sec:rsa})` inline.
- RaM ends the intro with forward references embedded in result claims: "We evaluated...across a diverse set of tasks" then immediately lists contributions.

### Section Transitions

Top papers generally do NOT use transitional paragraphs between sections. Each section starts directly with its own content. When there is a transition, it's a single sentence at the end of the preceding section.

### Comparison with Our Paper

**Our introduction (lines 69-77) has a significant structural weakness.** It opens with a literature-survey-style taxonomy:

> "The allocation of inference-time compute has emerged as a central frontier... Recent work on test-time compute is commonly organized around sequential scaling, parallel scaling, and hybrid search."

This is an academic mapping, not a tension. The reader doesn't feel a problem that needs solving until paragraph 2. Compare with Large Language Monkeys' immediate contrast: "Scaling compute for training has been dramatic... However, when it comes to inference, we often limit models to one attempt."

**Our contributions are not explicitly listed.** The intro states what ATTS is and what we evaluate, but there's no clear numbered list of contributions. Top papers almost universally enumerate 2-4 specific claims.

---

## 3. Experiment Section Structure

### Subsection Patterns

| Paper | Experiment Structure |
|-------|---------------------|
| RSA | `\section{Experiments}` > `\subsection{RSA Matches Human Performance on ARC-AGI-2}` > `\subsection{RSA Outperforms Other Test-Time Scaling Methods}` > `\subsection{RSA Yields Consistent Gains across Models}` > `\subsection{Effect of RSA Hyperparameters}` > `\subsection{RSA Improves with Aggregation-Aware RL}` |
| RaM | `\section{Experiments}` (with Datasets, Baselines, Training Details subsections) > `\section{Results}` (separate!) > `\subsection{Performance vs Cost}` > `\subsection{Adaptive computation}` > `\subsection{Generalization}` |
| Large Language Monkeys | `\section{Scaling Repeated Sampling}` > `\subsection{Repeated Sampling is Effective Across Tasks}` > `\subsection{Repeated Sampling is Effective Across Model Sizes}` > `\subsection{Repeated Sampling Can Help Balance Performance and Cost}` |
| DeepSeek-R1 | `\section{Experiment}` (flat, with subsections in appendix) |
| GPQA | Not an experiment paper; uses `\paragraph{}` throughout for results |

**Key pattern**: RSA and Large Language Monkeys use **claim-as-subsection-title** style. Each subsection title IS a finding ("RSA Outperforms Other Test-Time Scaling Methods", "Repeated Sampling is Effective Across Tasks"). This is dramatically more readable than generic titles like "Main Results" or "Ablation Studies."

### Analysis/Discussion Subsections

- None of the five papers have an explicit "Discussion" subsection within Experiments.
- Analysis is woven into result paragraphs. RSA has a dedicated `\subsection{Effect of RSA Hyperparameters}` which serves as both ablation and analysis.
- RaM has `\subsection{Adaptive computation}` which analyzes behavior rather than reporting numbers.

### Ablation Presentation

- RSA presents ablations under "Effect of RSA Hyperparameters" with specific parameter sweeps.
- Our paper uses `\paragraph{Exploration effort.}`, `\paragraph{Adding a separate integrator.}`, `\paragraph{Orchestrator model selection.}`, `\paragraph{Backbone model family.}` -- all under a single `\subsection{Ablation Studies}`.
- This is functional but less readable than claim-titled subsections.

### Case Studies / Qualitative Analysis

- RSA includes a full appendix section `\section{Qualitative Example}` with detailed examples.
- RaM includes `\section{Qualitative examples}` in appendix.
- Large Language Monkeys includes `\subsection{Verifiers and Software Tasks: Two Cautionary Tales}` in the main text -- a qualitative narrative section that's particularly effective.
- Our paper has `\section{Qualitative Analysis}` in appendix, which is fine.

### Comparison with Our Paper

Our `\subsection{Main Results}` and `\subsection{Ablation Studies}` use generic titles. The content is organized by `\paragraph{}` headers per benchmark, which is functional but doesn't guide the reader toward takeaways. RSA's approach of using finding-as-title is substantially more readable. See recommendation 3.1.

---

## 4. Listing/Code Formatting

### Findings

Papers that include prompts or code use varied approaches:
- **DeepSeek-R1**: Defines two `lstdefinestyle`: `code` (gray background, single-line frame) and `data` (top-bottom frame only). Uses `\footnotesize\ttfamily`.
- **Large Language Monkeys**: Defines `mystyle` with numbered lines, light background (`backcolour`).
- **RSA**: Uses `tcolorbox` with `listings` library for styled prompt boxes (`expblock` environment) -- more visually polished.
- **RaM**: No code listings; prompts are shown in appendix as plain text/tables.

### Comparison with Our Paper

Our paper uses `lstlisting` with `frame=shadowbox`, `backgroundcolor=gray!5`, `rulecolor=blue!40!black`. This is functional and not far from top papers. The shadow box with blue rule is slightly more ornate than the typical gray-frame approach.

One issue: our listings (lines 396-468) are long and take up significant space. RSA's `tcolorbox` approach with custom titled boxes would be slightly cleaner, but this is a minor stylistic difference.

---

## 5. Specific Recommendations for Our Paper

### 5.1. Introduction: Create Tension, Not Taxonomy (Lines 71-73)

**Current** (line 71):
> "The allocation of inference-time compute has emerged as a central frontier for improving the reasoning capabilities of large language models. Recent work on test-time compute is commonly organized around sequential scaling, parallel scaling, and hybrid search."

**Problem**: Opens with a literature taxonomy. The reader has no reason to care yet.

**Recommended rewrite**:
> "Scaling inference-time compute has become a key lever for improving large language model reasoning. Yet existing methods face a persistent tension: open-loop heuristics cannot react to evidence that emerges during generation, while closed-loop search methods require explicit tree expansion, step-level verifiers, or statistical stopping rules. An underexplored alternative is to delegate the allocation decision entirely to an LLM orchestrator that reasons about when to stop and what to conclude."

This follows the Large Language Monkeys / RaM pattern: **establish a tension in 2 sentences, then introduce the resolution**.

### 5.2. Introduction: Add an Explicit Contribution List (After Line 77)

Top papers nearly universally enumerate contributions. Add after the evaluation paragraph:

```latex
Our contributions are:
\begin{enumerate}
    \item We introduce \our{}, a delegated sequential test-time scaling framework where a lightweight orchestrator adaptively controls a pool of independent solvers through a single action (\explore{}).
    \item We show that even this minimal configuration achieves competitive or superior accuracy to fixed-budget alternatives while using substantially fewer API calls across six benchmarks.
    \item We provide theoretical analysis showing that the orchestrator's information advantage over a separate aggregator is provable (Blackwell replication) and that sequential accumulation yields a submartingale guarantee on accuracy.
\end{enumerate}
```

### 5.3. Experiment Subsection Titles: Use Claims Instead of Generic Labels

**Current structure**:
```
\subsection{Main Results}
\subsection{Ablation Studies}
```

**Recommended structure** (following RSA / Large Language Monkeys):
```
\subsection{ATTS Achieves Competitive Accuracy at Lower Cost}
\subsection{Medium Exploration Effort is the Best Operating Point}
\subsection{Stateful Orchestration Outperforms Separate Integration}
\subsection{The Framework Generalizes Across Model Families}
```

Each title previews the finding, making the paper skimmable and the claims immediately visible.

### 5.4. "Overall trends" Paragraph is Too Hedging (Lines 294-297)

**Current** (line 295):
> "\our{} is intended to occupy a middle ground between fixed-budget sequential refinement and purely parallel sampling."

**Problem**: "is intended to" is hedging language. The results show it DOES occupy this middle ground.

**Recommended rewrite**:
> "\our{} occupies a middle ground between fixed-budget sequential refinement and purely parallel sampling."

Similarly, line 296:
> "Direct one-shot generation is often already strong on easier exact-match tasks, but it can leave performance on the table..."

This is vague. Say what the data shows:
> "Direct one-shot generation achieves near-ceiling accuracy on easy subtasks (100\% Easy on LiveCodeBench) but leaves significant headroom on hard problems (60\% Hard)."

### 5.5. Use `\emph{}` for Rhetorical Contrast (Throughout)

Top papers use italics to highlight surprising or contrasting words. Our paper is somewhat flat in emphasis. Examples:

**Line 334** (integrator ablation):
> "The degradation likely stems from the integrator having less context for synthesis"

Could be:
> "The degradation likely stems from the integrator having \emph{less} context for synthesis"

**Line 339** (orchestrator ablation):
> "the strongest model is the best orchestrator on some benchmarks but the worst on others"

Could be:
> "the strongest model is the best orchestrator on some benchmarks but the \emph{worst} on others"

### 5.6. Related Work: Consider \paragraph{} Subgrouping with Transition Sentences

Our Related Work (lines 79-88) is a single dense section with 4 `\paragraph{}` blocks. This is consistent with top papers. However, each paragraph currently reads as an independent survey with no connecting argument. Consider adding a single sentence at the end of each paragraph that positions ATTS relative to the category, e.g.:

After the "Test-Time Scaling" paragraph:
> "\our{} is neither a single-trajectory method nor a tree-search method..." (this sentence already exists and is good)

After the "Adaptive Compute Allocation" paragraph:
> "\our{} uses an LLM orchestrator that reads full reasoning traces..." (this also exists and is good)

The Related Work is actually well-structured. No major changes needed.

### 5.7. Abstract: Lead with the Problem, Not the Solution (Line 66)

**Current opening**:
> "Test-time scaling has become a key mechanism for improving large language model reasoning, but existing approaches often rely either on rigid heuristic allocation rules or on explicit fine-grained search..."

This is good but could be tightened. The "but" clause is the real hook. Consider:

> "Existing test-time scaling approaches rely on rigid heuristic allocation or explicit search over intermediate steps, missing an alternative: delegating compute allocation to an LLM that reasons about when additional exploration justifies its cost."

### 5.8. Methodology Section Opening: Strengthen the Framing (Line 93)

**Current** (line 93):
> "Test-time scaling can be viewed as a resource allocation problem: given a budget of model calls, how should an agent decide what to compute next and when to stop?"

This is a good opening. But the next sentence is weak:
> "Drawing on the principle of rational metareasoning..."

**Problem**: "Drawing on" is hedging. From our memory, the framing is "draws on, not operationalizes." But "drawing on" still sounds passive. Consider:

> "The principle of rational metareasoning---that an agent should reason about whether additional computation justifies its cost---motivates our design: \our{} as a delegated sequential framework..."

---

## 6. Example Rewrites

### Rewrite 1: Introduction Opening (Lines 71-75)

**Before**:
> The allocation of inference-time compute has emerged as a central frontier for improving the reasoning capabilities of large language models. Recent work on test-time compute is commonly organized around sequential scaling, parallel scaling, and hybrid search. The compute-allocation perspective of \citet{snell_test_time_scaling_paper} demonstrates that no single strategy is uniformly optimal across prompts of varying difficulty. At the same time, simply allocating more compute is not enough: naive scaling can saturate, and in sampling-based settings can even degrade through poor coverage and reward hacking. The central question is therefore not whether to scale test-time compute, but how to allocate it adaptively and efficiently.

**After**:
> Scaling inference-time compute improves large language model reasoning, but how to allocate that compute remains an open problem. No single strategy---sequential, parallel, or hybrid---is uniformly optimal across problems of varying difficulty \citep{snell_test_time_scaling_paper}, and naive scaling can saturate or even degrade accuracy through poor coverage and reward hacking \citep{best_of_n_optimality_paper}. The central question is not \emph{whether} to scale, but \emph{how} to allocate adaptively: reacting to evidence as it appears, stopping when further exploration cannot justify its cost.

**Why**: Removes the taxonomy-first structure. Opens with the tension (scaling helps, but allocation is unsolved). Uses `\emph{}` for the "whether/how" contrast, following the RaM pattern.

### Rewrite 2: Overall Trends Paragraph (Lines 294-297)

**Before**:
> Across benchmarks, \our{} is intended to occupy a middle ground between fixed-budget sequential refinement and purely parallel sampling. Direct one-shot generation is often already strong on easier exact-match tasks, but it can leave performance on the table once problems require either broader hypothesis coverage or evidence reconciliation across partially correct candidates.

**After**:
> Across benchmarks, \our{} occupies a distinct operating point between fixed-budget refinement and purely parallel sampling. One-shot generation reaches near-ceiling accuracy on easy subtasks (100\% Easy on LiveCodeBench, 74.24\% Pass@1 on GPQA) but leaves substantial headroom on hard problems. Majority Voting increases candidate diversity yet reduces the final decision to selection without reasoning about the evidence. \our{} uses the same test-time budget more adaptively: exploring when the current pool is incomplete, and synthesizing when the candidates contain sufficient evidence.

**Why**: Removes hedging ("is intended to"), replaces vague claims with specific numbers, sharpens the contrast.

### Rewrite 3: GPQA Results Paragraph (Lines 281-283)

**Before**:
> On GPQA-Diamond, \our{} achieves the highest accuracy among all test-time scaling methods (84.77\%) at \$0.36/q, substantially outperforming Budget Forcing (77.16\%) and Skywork-Reward-V2 (76.14\%). Self-Refine (73.10\%) falls below Pass@1 (74.24\%), consistent with the observation that its feedback model frequently accepts the initial draft without revision, and occasionally introduces errors when it does revise.

**After**:
> On GPQA-Diamond, \our{} achieves the highest accuracy among all test-time scaling methods: 84.77\% at \$0.36/q, a 7.6-point gain over both Budget Forcing and LLM Selection (77.16\%). Self-Refine (73.10\%) falls \emph{below} Pass@1 (74.24\%), consistent with the known limitation that its feedback model frequently accepts the initial draft, and occasionally introduces errors during revision \citep{cannot_self_correct_paper}.

**Why**: Adds the delta (7.6 points) for immediate impact assessment. Uses `\emph{below}` for the surprising finding that Self-Refine hurts. Tightens the Self-Refine explanation.

### Rewrite 4: Methodology Opening (Lines 93-94)

**Before**:
> Test-time scaling can be viewed as a resource allocation problem: given a budget of model calls, how should an agent decide what to compute next and when to stop? Drawing on the principle of rational metareasoning~\citep{russell1991right,hay_selecting_computations}---that an agent should reason about whether additional computation justifies its cost---we design \our{} as a delegated sequential framework in which a lightweight orchestrator manages a growing pool of candidate solutions, decides when to stop exploring, and synthesizes the final answer from the accumulated evidence.

**After**:
> Test-time scaling is a resource allocation problem: given a budget of model calls, how should an agent decide what to compute next and when to stop? Rational metareasoning~\citep{russell1991right,hay_selecting_computations} provides the guiding principle: additional computation is warranted only when its expected information gain justifies the cost. We instantiate this principle in \our{}, a delegated sequential framework in which a lightweight orchestrator manages a growing pool of candidate solutions, decides when to stop exploring, and synthesizes the final answer from the accumulated evidence.

**Why**: Removes "can be viewed as" (hedging). Breaks into three sentences: problem statement, principle, instantiation. More direct.

### Rewrite 5: Limitations Opening (Lines 372-374)

**Before**:
> \our{} inherits several limitations from the underlying language model and from its own orchestrator design. First, the method can only synthesize evidence that is already present in the candidate pool, so if all explored candidates share the same misconception, the orchestrator has little opportunity to recover.

**After**:
> \our{} has four principal limitations. First, it can only synthesize evidence present in the candidate pool: if all candidates share the same misconception, the orchestrator cannot recover.

**Why**: Top papers are direct about limitations. "inherits several limitations from the underlying language model and from its own orchestrator design" is throat-clearing.

---

## Summary of Priorities

| Priority | Recommendation | Impact |
|----------|---------------|--------|
| High | Rewrite intro opening to create tension, not taxonomy (5.1) | First impression; skimmability |
| High | Add explicit contribution list (5.2) | Reviewer convenience |
| High | Use claim-as-title for experiment subsections (5.3) | Skimmability; findings stand out |
| Medium | Remove hedging language ("is intended to", "can be viewed as") (5.4, 5.8) | Authority and clarity |
| Medium | Add `\emph{}` for rhetorical contrast on surprising findings (5.5) | Readability |
| Low | Tighten abstract opening (5.7) | Minor polish |
| Low | Tighten limitations opening (Rewrite 5) | Minor polish |
