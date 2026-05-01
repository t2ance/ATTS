"""Method spec sub-schemas: discriminated union over method name.

Mirror of benchmarks/specs.py. Each method gets one Pydantic class enumerating
exactly its valid fields. extra="forbid" rejects unknown YAML keys, so dead
fields (e.g. orchestrator_model in self-refine YAMLs) fail validation instead
of being silently ignored.

Pairs with methods/registry.py (the runtime behavior layer).
"""
from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal, Union

from pydantic import BaseModel, Field, model_validator


class SamplingConfig(BaseModel):
    """Per-call sampling knobs for the orchestrator (vLLM backend only).

    None for any field means "use the upstream library default" -- the
    backend's _split_sampling_kwargs drops None entries before calling
    client.chat.completions.create. OpenAI-native fields (temperature,
    top_p, presence_penalty, max_tokens) are passed directly; vLLM
    extensions (top_k, min_p, repetition_penalty, enable_thinking) are
    routed through extra_body.

    `max_tokens` is the per-turn output cap (NOT the cumulative cap across
    turns -- that lives at EvalConfig.max_output_tokens).
    """
    model_config = {"extra": "forbid"}
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    min_p: float | None = None
    presence_penalty: float | None = None
    repetition_penalty: float | None = None
    enable_thinking: bool | None = None
    max_tokens: int | None = None


class _MethodSpec(BaseModel):
    model_config = {"extra": "forbid"}


class TTSAgentSpec(_MethodSpec):
    name: Literal["tts-agent"]
    orchestrator_model: str
    explore_model: str
    integrate_model: str | None = None
    cache_dir: Path
    no_integrate: bool = False
    num_explores: int = 8
    num_rollouts: int = 1
    sampling: SamplingConfig | None = None

    @model_validator(mode="after")
    def _check_integrate(self):
        if not self.no_integrate:
            assert self.integrate_model, (
                "tts-agent requires integrate_model unless no_integrate=true"
            )
        return self


class TTSAgentMultiSpec(_MethodSpec):
    name: Literal["tts-agent-multi"]
    orchestrator_model: str
    cache_dirs: dict[str, Path]
    model_budgets: dict[str, int]
    exploration_effort: Literal["low", "medium", "high"] | None = None
    num_explores: int = 8


class TTSAgentEffortSpec(_MethodSpec):
    name: Literal["tts-agent-effort"]
    orchestrator_model: str
    explore_model: str
    cache_dirs: dict[str, Path]
    effort_budgets: dict[str, int]
    num_explores: int = 8


class SelfRefineSpec(_MethodSpec):
    name: Literal["self-refine"]
    explore_model: str
    cache_dir: Path
    num_explores: int = 8


class SocraticSelfRefineSpec(_MethodSpec):
    name: Literal["socratic-self-refine"]
    explore_model: str
    cache_dir: Path
    num_explores: int = 8


class BudgetForcingSpec(_MethodSpec):
    name: Literal["budget-forcing"]
    explore_model: str
    cache_dir: Path
    num_explores: int = 8


class RerankSpec(_MethodSpec):
    name: Literal["rerank"]
    reward_model: str
    cache_dir: Path
    # No explore_model / integrate_model / num_explores: rerank reads cached
    # explores and scores them with a reward model, no LLM call, no new explore.


class StandaloneIntegratorSpec(_MethodSpec):
    name: Literal["standalone-integrator"]
    integrate_model: str
    cache_dir: Path
    # No explore_model / num_explores: reads cached explores, single integrate call.


MethodSpec = Annotated[
    Union[
        TTSAgentSpec, TTSAgentMultiSpec, TTSAgentEffortSpec,
        SelfRefineSpec, SocraticSelfRefineSpec, BudgetForcingSpec,
        RerankSpec, StandaloneIntegratorSpec,
    ],
    Field(discriminator="name"),
]
