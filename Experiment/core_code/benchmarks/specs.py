"""Benchmark spec sub-schemas: discriminated union over benchmark name.

Replaces both EvalConfig.benchmark (str) and EvalConfig.filters (dict). Each
benchmark gets one Pydantic class enumerating exactly its valid filter keys.
"""
from __future__ import annotations

from typing import Annotated, Literal, Union

from pydantic import BaseModel, Field

from methods.specs import SamplingConfig


class _Spec(BaseModel):
    model_config = {"extra": "forbid"}


# JudgeSpec: per-judge config block nested inside benchmarks that grade with
# an LLM judge (HLE / BabyVision-blank / RBenchV). Discriminated union over
# `name`. claude/codex carry only `model`; vllm carries both `model` and a
# required `sampling` block matching the SamplingConfig used by explorer /
# orchestrator. Benchmarks that grade without a judge (LCB by code execution,
# GPQA by multipleChoice, AIME by exactMatch) do not carry `judge:` and reject
# it via the `_Spec.extra="forbid"` policy.
class _JudgeSpec(BaseModel):
    model_config = {"extra": "forbid"}


class ClaudeJudgeSpec(_JudgeSpec):
    name: Literal["claude"]
    model: str
    # Thinking-budget controls for the judge call. Default `effort: "low"`
    # enforces CLAUDE.md global policy "For judge of answer, ALWAYS use
    # non-thinking to save money" — without this default, ~60 yamls that
    # don't set effort: explicitly would fall back to library defaults
    # (`thinking.enabled=true, budget_tokens=32000` → ~1.8K avg output
    # tokens / verdict → ~$0.013 per call → $10+ per 800-grade eval).
    # Setting effort=low caps thinking and brings cost to ~$2-3 per
    # 800-grade eval (verified 2026-05-04 against gpt-oss-20b LOW
    # cache: 135 Haiku grades sum to $1.81). Override here in the YAML
    # explicitly (`effort: "high"` / `budget_tokens: 32000`) ONLY if a
    # specific judging task needs more reasoning depth (none currently
    # in scripts/ as of 2026-05-04 — see grep `effort:` under judge:).
    # Cache coupling: judge_label includes only (name, model), NOT effort,
    # so existing caches built before this default change still hit; new
    # cache misses produce verdicts under the new low-effort default.
    # find_cached_judge logs a best-effort warning if stored config lacks
    # `effort` keys.
    effort: Literal["low", "medium", "high", "max"] | None = "low"
    budget_tokens: int | None = None


class CodexJudgeSpec(_JudgeSpec):
    name: Literal["codex"]
    model: str


class VllmJudgeSpec(_JudgeSpec):
    name: Literal["vllm"]
    model: str
    sampling: SamplingConfig


JudgeSpec = Annotated[
    Union[ClaudeJudgeSpec, CodexJudgeSpec, VllmJudgeSpec],
    Field(discriminator="name"),
]


class HLESpec(_Spec):
    name: Literal["hle"]
    subset: Literal["gold", "revision", "uncertain"] | None = None
    category: str | None = None
    text_only: bool = False
    judge: "JudgeSpec"  # required; explicit YAML, no implicit class-attr default


class GPQASpec(_Spec):
    name: Literal["gpqa"]
    domain: str | None = None
    # No `judge:` — multipleChoice grading is purely string match. extra="forbid" rejects it.


class LCBSpec(_Spec):
    name: Literal["lcb"]
    difficulty: str | None = None
    # No `judge:` — code execution grading via lcb_runner. extra="forbid" rejects it.


class BabyVisionSpec(_Spec):
    name: Literal["babyvision"]
    type: str | None = None
    subtype: str | None = None
    judge: "JudgeSpec"  # required; choice questions short-circuit before invoking judge


class RBenchVSpec(_Spec):
    name: Literal["rbenchv"]
    category: str | None = None
    judge: "JudgeSpec"  # required; visual reasoning answers need semantic equivalence


class AIME2025Spec(_Spec):
    name: Literal["aime2025"]
    year: int | None = None


class AIME2026Spec(_Spec):
    name: Literal["aime2026"]
    year: int | None = None


BenchmarkSpec = Annotated[
    Union[
        HLESpec, GPQASpec, LCBSpec, BabyVisionSpec, RBenchVSpec,
        AIME2025Spec, AIME2026Spec,
    ],
    Field(discriminator="name"),
]
