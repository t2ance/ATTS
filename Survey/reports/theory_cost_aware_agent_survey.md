# Cost-Aware Agent Training & Tool-Use RL: Survey
Date: 2026-03-29
Source: User-provided papers + web research

## Core Question
Has anyone unified RaM-style VOC reward with general agent tool-use trajectories?
Answer: Not yet. Pieces exist but no unified framework.

## Tier 1: Closest to "VOC for Agents"

### Calibrate-Then-Act (CTA) — Feb 2026
- arXiv: 2602.16699
- Authors: (unknown from abstract)
- What: Formalizes cost-uncertainty tradeoff for LLM agents. Agent decides whether additional exploration (tool calls) is worth the cost vs. committing with current uncertainty.
- Framework: Sequential decision-making under uncertainty with cost-benefit tradeoffs made explicit. Supplies LLM with prior + cost context.
- Tasks: Information-seeking QA, simplified coding (test-if-uncertain)
- Results: Improvement preserved even under RL training
- Connection to ATTS: **Very close conceptually.** Their "continue exploring vs commit" is our "explore more vs stop." Their cost-uncertainty tradeoff is essentially VOC. But they don't explicitly cite metareasoning/VOC.
- Key difference: General agent tasks (retrieval, coding), not candidate-level sampling. No explicit metalevel MDP formalization.

### CATP-LLM — Nov 2024
- arXiv: 2411.16313
- Venue: ICCV 2025
- What: Cost-aware tool PLANNING. LLM schedules external tools considering execution costs (time). Uses cost-aware offline RL.
- Training: Offline RL with performance-cost tradeoff objective
- Results: Outperforms GPT-4 with Llama2-7B, 1.5-93.9% plan quality improvement
- Connection to ATTS: Most direct on "tool cost in reward." But focuses on tool PLANNING (which tools to call in what order), not sequential exploration with stopping.
- Key difference: Tool planning ≠ sequential explore-and-stop. Different problem structure.

## Tier 2: Tool-Use Reward Design

### ToolRL — Apr 2025
- arXiv: 2504.13958
- What: Systematic study of reward design for tool-use RL. Studies reward type, scale, granularity, temporal dynamics.
- Training: GRPO with designed rewards
- Results: 17% improvement over base, 15% over SFT
- Connection to ATTS: Relevant methodology for future GRPO training of orchestrator. Shows reward design matters greatly.
- Key difference: Studies reward engineering, not cost-aware stopping.

### Tool-call Reward Model (TRM) — ICLR 2026
- OpenReview: LnBEASInVr
- What: Process reward model specifically for tool calls. Evaluates each tool call individually, not just final outcome.
- Integration: Plugs into PPO/GRPO for step-level credit assignment
- Connection to ATTS: Solves the credit assignment problem we'd face in GRPO training. If we train orchestrator to decide explore/stop, we need per-step credit.
- Key difference: Evaluates tool calls, doesn't model tool costs.

### StepTool — Oct 2024
- arXiv: 2410.07745
- Venue: CIKM 2025
- What: Step-grained RL for multi-step tool use. Assigns rewards per tool interaction based on invocation success + contribution to task completion.
- Connection to ATTS: Similar step-level reward granularity. Each explore call could get its own reward signal.

## Tier 3: Outcome-Only Agentic RL (Not Cost-Aware)

### Search-R1 — NeurIPS 2025
- Structurally similar to ATTS (multi-turn search-reason-search-reason)
- BUT: outcome-only reward, no cost modeling

### ReTool, ToRL — 2025
- Learn strategic tool invocation from outcome feedback
- No explicit cost in reward

## Assessment

The landscape in March 2026:
- Cost-aware planning: CATP-LLM (tool scheduling with cost)
- Cost-aware exploration: CTA (explore vs commit with cost-uncertainty tradeoff)
- Tool-use reward design: ToolRL, TRM, StepTool (how to reward tool calls)
- Outcome-only agentic RL: Search-R1, ReTool, ToRL (learn tool use without cost modeling)

**Gap: No one has unified VOC/metareasoning + candidate-level TTS + agentic orchestrator + cost-aware RL training.**

ATTS occupies a unique position:
1. We formalize the explore-stop problem as rational metareasoning (VOC, metalevel MDP)
2. We operate at candidate level (independent explores, not tool planning)
3. We have an agentic orchestrator (not self-evaluation)
4. Future GRPO training can directly use R = 1[correct] - lambda*cost

CTA is the closest conceptual cousin — they should be cited and differentiated.
CATP-LLM is closest on cost-in-reward — but different problem structure (planning ≠ stopping).
ToolRL/TRM/StepTool provide the reward engineering methodology for our future training.
