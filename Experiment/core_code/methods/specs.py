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
    # Per-request cap on thinking tokens (vLLM PR #20859, ships in 0.20.0).
    # Forces `<|/think|>` injection via logits processor when the cap is hit.
    # Default None: unbounded thinking. Requires the serve to be started with
    # top-level `--reasoning-parser <name>` (sets up vLLM's ReasoningConfig);
    # putting `reasoning_parser` only inside `--structured-outputs-config` is
    # NOT sufficient — see arg_utils.py:2332-2337 + input_processor.py:101-109
    # which raises HTTP 400 "thinking_token_budget is set but reasoning_config
    # is not configured" otherwise.
    thinking_token_budget: int | None = None
    max_tokens: int | None = None
    # Backend-side flag: when True, drop `response_format=json_schema` and
    # inject the schema as text instructions in the user message; rely on
    # the model's natural JSON-emission ability + post-hoc json.loads.
    # Required for Gemma-4 (vllm#40080: xgrammar guided JSON triggers
    # deterministic repetition loops). Read by backends/vllm.py:call_sub_model.
    disable_response_format: bool | None = None


class ModelConfig(BaseModel):
    """Per-role backend invocation config.

    Replaces the shared method-level BackendConfig + bare-string model name
    pattern. Each model role (orchestrator, explore variant, integrate, judge)
    carries its own complete invocation parameters, including optional
    backend-specific blocks (vllm_sampling for vllm, openrouter_provider_*
    for openrouter). The model_validator hard-fails when a backend-specific
    field is set on the wrong backend, replacing the prior silent-no-op
    behavior of BackendConfig.
    """
    model_config = {"extra": "forbid"}

    backend: Literal["codex", "claude", "vllm", "openrouter"]
    model: str
    budget_tokens: int = 32000
    effort: Literal["low", "medium", "high", "max"] = "low"
    timeout: float = 1200.0
    max_output_tokens: int | None = None

    vllm_sampling: SamplingConfig | None = None

    openrouter_provider_order: list[str] | None = None
    openrouter_provider_allow_fallbacks: bool = True

    @model_validator(mode="after")
    def _check_backend_specific(self):
        if self.backend != "vllm":
            assert self.vllm_sampling is None, (
                f"vllm_sampling is vllm-only but backend={self.backend!r}; remove from yaml"
            )
        if self.backend != "openrouter":
            assert self.openrouter_provider_order is None, (
                f"openrouter_provider_order is openrouter-only but backend={self.backend!r}"
            )
            assert self.openrouter_provider_allow_fallbacks is True, (
                f"openrouter_provider_allow_fallbacks is openrouter-only but "
                f"backend={self.backend!r}"
            )
        return self


class RoleSlot(BaseModel):
    """A model invocation + the cache_dir its outputs land in.

    Used by single-call cached roles (e.g. integrate). Pydantic extra="forbid"
    rejects num_explores or other ExploreVariant-shaped fields at config-load.
    """
    model_config = {"extra": "forbid"}
    model: ModelConfig
    cache_dir: Path


class ExploreVariant(BaseModel):
    """One explore-pass: model + its cache_dir + how many candidates to draw.

    The label is required and serves as the orchestrator-visible identifier
    (the unified explore tool exposes `variant: enum[<labels>]` when there
    is more than one variant in a TTSAgentSpec.explore list). Cache layout
    on disk is per-variant via cache_dir, so cache_keys remain
    `f"explore_{idx}"` within each variant.
    """
    model_config = {"extra": "forbid"}
    label: str
    model: ModelConfig
    cache_dir: Path
    num_explores: int = 8


class _MethodSpec(BaseModel):
    model_config = {"extra": "forbid"}


class TTSAgentSpec(_MethodSpec):
    """Unified spec covering single, multi-model, and multi-effort runs.

    `explore: list[ExploreVariant]` length encodes the operating mode:
    - length 1 with `orchestrator_prompt: single`: standard ATTS.
    - length 3 with `orchestrator_prompt: multi_model` and labels
      {haiku, sonnet, opus}: replaces the old TTSAgentMultiSpec.
    - length 3 with `orchestrator_prompt: effort` and labels
      {low, medium, high}: replaces the old TTSAgentEffortSpec.

    The orchestrator's explore tool exposes `variant: enum[<labels>]` when
    `len(explore) > 1`, no parameter when `len(explore) == 1`. The function
    `prompts.select_orchestrator_prompt(spec)` routes
    (orchestrator_prompt, integrate is None) to the matching system prompt;
    the multi_model and effort prompts hardcode their label sets, so the
    validator below enforces label-set match.
    """
    name: Literal["tts-agent"]
    orchestrator: ModelConfig
    explore: list[ExploreVariant]
    integrate: RoleSlot | None = None
    orchestrator_prompt: Literal["single", "multi_model", "effort"]
    num_rollouts: int = 1

    @model_validator(mode="after")
    def _check_consistency(self):
        n = len(self.explore)
        p = self.orchestrator_prompt
        assert (p == "single") == (n == 1), (
            f"orchestrator_prompt={p!r} requires "
            f"{'len(explore)==1' if p == 'single' else 'len(explore)>1'}, got {n}"
        )
        labels = [v.label for v in self.explore]
        assert len(set(labels)) == len(labels), (
            f"duplicate labels in explore list: {labels}"
        )
        if p == "multi_model":
            assert set(labels) == {"haiku", "sonnet", "opus"}, (
                f"orchestrator_prompt=multi_model hardcodes haiku/sonnet/opus; "
                f"yaml has labels={set(labels)}"
            )
        if p == "effort":
            assert set(labels) == {"low", "medium", "high"}, (
                f"orchestrator_prompt=effort hardcodes low/medium/high; "
                f"yaml has labels={set(labels)}"
            )
        if p in ("multi_model", "effort"):
            assert self.integrate is None, (
                f"orchestrator_prompt={p!r} assumes orchestrator self-synthesizes; "
                f"integrate must be None"
            )
        return self


class SelfRefineSpec(_MethodSpec):
    name: Literal["self-refine"]
    explore: ExploreVariant


class SocraticSelfRefineSpec(_MethodSpec):
    name: Literal["socratic-self-refine"]
    explore: ExploreVariant


class BudgetForcingSpec(_MethodSpec):
    name: Literal["budget-forcing"]
    explore: ExploreVariant


class RerankSpec(_MethodSpec):
    name: Literal["rerank"]
    reward_model: str
    cache_dir: Path
    # Kept asymmetric: reward_model is a HuggingFace model id loaded by
    # transformers, not a backend-routed remote call. Forcing through
    # ModelConfig would require adding a "local" backend Literal which
    # mixes "remote backend" and "local PyTorch" semantics.


class StandaloneIntegratorSpec(_MethodSpec):
    name: Literal["standalone-integrator"]
    integrate: RoleSlot
    # Default 8 reproduces the paper's "LLM Selection (N=8)" baseline (main.tex
    # tab:main-results). Override e.g. `num_explores: 4` in yaml to take only
    # the first N cached candidates per question -- enables cost-vs-accuracy
    # Pareto sweeps on the same fixed cache without regenerating explores.
    num_explores: int = 8


MethodSpec = Annotated[
    Union[
        TTSAgentSpec,
        SelfRefineSpec, SocraticSelfRefineSpec, BudgetForcingSpec,
        RerankSpec, StandaloneIntegratorSpec,
    ],
    Field(discriminator="name"),
]
