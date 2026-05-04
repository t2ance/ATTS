"""Benchmark spec sub-schemas: discriminated union over benchmark name.

Replaces both EvalConfig.benchmark (str) and EvalConfig.filters (dict). Each
benchmark gets one Pydantic class enumerating exactly its valid filter keys.

Judge invocation block: HLE / BabyVision / RBenchV carry `judge: ModelConfig`.
The previous Claude/Codex/Vllm-discriminated JudgeSpec union was collapsed
2026-05-04 into a single ModelConfig (per-role backend invocation type).
The CLAUDE.md non-thinking-judge default is encoded as ModelConfig.effort
default = "low".
"""
from __future__ import annotations

from typing import Annotated, Literal, Union

from pydantic import BaseModel, Field

from methods.specs import ModelConfig


class _Spec(BaseModel):
    model_config = {"extra": "forbid"}


class HLESpec(_Spec):
    name: Literal["hle"]
    subset: Literal["gold", "revision", "uncertain"] | None = None
    category: str | None = None
    text_only: bool = False
    judge: ModelConfig  # required; explicit YAML, no implicit class-attr default


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
    judge: ModelConfig  # required; choice questions short-circuit before invoking judge


class RBenchVSpec(_Spec):
    name: Literal["rbenchv"]
    category: str | None = None
    judge: ModelConfig  # required; visual reasoning answers need semantic equivalence


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
