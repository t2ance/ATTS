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


class GPQASpec(_Spec):
    name: Literal["gpqa"]
    domain: str | None = None


class LCBSpec(_Spec):
    name: Literal["lcb"]
    difficulty: str | None = None


class BabyVisionSpec(_Spec):
    name: Literal["babyvision"]
    type: str | None = None
    subtype: str | None = None


class RBenchVSpec(_Spec):
    name: Literal["rbenchv"]
    category: str | None = None


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
