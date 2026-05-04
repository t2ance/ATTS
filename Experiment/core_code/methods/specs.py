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


class BackendConfig(BaseModel):
    model_config = {"extra": "forbid"}
    # "openrouter" added 2026-05-03: dispatches via the OpenRouter Anthropic-Skin
    # endpoint (ANTHROPIC_BASE_URL=https://openrouter.ai/api). Reuses backends/claude.py
    # transport — backend module name "openrouter" maps to importing claude.py via
    # a thin alias. See todo_openrouter_via_claude.md for the viability proof.
    name: Literal["codex", "claude", "vllm", "openrouter"]
    budget_tokens: int = 32000
    effort: Literal["low", "medium", "high", "max"] = "low"
    timeout: float = 1200.0
    max_output_tokens: int | None = None
    # OpenRouter-only: pin upstream provider routing. See
    # backends/openrouter.py:set_provider for rationale (2026-05-04
    # deepseek-v4-flash incident: ~33% of routes 400'd on forced tool_choice
    # because some upstream providers like DeepSeek's deepseek-reasoner endpoint
    # reject `tool_choice={"type":"function",...}`). When set, eval.py calls
    # backends.openrouter.set_provider(...) at startup; both call_sub_model and
    # run_tool_conversation inject the block into extra_body.provider on every
    # request. Ignored silently by non-openrouter backends.
    provider_order: list[str] | None = None
    provider_allow_fallbacks: bool = True


class _MethodSpec(BaseModel):
    model_config = {"extra": "forbid"}


class TTSAgentSpec(_MethodSpec):
    name: Literal["tts-agent"]
    backend: BackendConfig
    orchestrator_model: str
    explore_model: str
    integrate_model: str | None = None
    cache_dir: Path
    no_integrate: bool = False
    num_explores: int = 8
    num_rollouts: int = 1
    sampling: SamplingConfig | None = None
    # orchestrator_effort: per-role override of backend.effort, applied ONLY
    #   to the orchestrator turn (run_tool_conversation in tts_agent.py:~280),
    #   NOT to explore tool calls. Use case: cached explores at effort=low
    #   are reused; orchestrator at effort=high reasons more deeply when
    #   selecting/integrating among them. None = inherit backend.effort.
    #   Coupling: cache_only=True (registry.py:94) prevents new explore calls,
    #   so cache effort consistency is preserved regardless of this setting.
    orchestrator_effort: Literal["low", "medium", "high", "max"] | None = None

    @model_validator(mode="after")
    def _check_integrate(self):
        if not self.no_integrate:
            assert self.integrate_model, (
                "tts-agent requires integrate_model unless no_integrate=true"
            )
        return self


class TTSAgentMultiSpec(_MethodSpec):
    name: Literal["tts-agent-multi"]
    backend: BackendConfig
    orchestrator_model: str
    cache_dirs: dict[str, Path]
    model_budgets: dict[str, int]
    exploration_effort: Literal["low", "medium", "high"] | None = None
    num_explores: int = 8


class TTSAgentEffortSpec(_MethodSpec):
    name: Literal["tts-agent-effort"]
    backend: BackendConfig
    orchestrator_model: str
    explore_model: str
    cache_dirs: dict[str, Path]
    effort_budgets: dict[str, int]
    num_explores: int = 8


class SelfRefineSpec(_MethodSpec):
    name: Literal["self-refine"]
    backend: BackendConfig
    explore_model: str
    cache_dir: Path
    num_explores: int = 8


class SocraticSelfRefineSpec(_MethodSpec):
    name: Literal["socratic-self-refine"]
    backend: BackendConfig
    explore_model: str
    cache_dir: Path
    num_explores: int = 8


class BudgetForcingSpec(_MethodSpec):
    name: Literal["budget-forcing"]
    backend: BackendConfig
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
    backend: BackendConfig
    integrate_model: str
    cache_dir: Path
    # Default 8 reproduces the paper's "LLM Selection (N=8)" baseline (main.tex
    # tab:main-results). Override e.g. `num_explores: 4` in yaml to take only
    # the first N cached candidates per question -- enables cost-vs-accuracy
    # Pareto sweeps on the same fixed cache without regenerating explores.
    # Coupling: integrator response cache key is `integrate_standalone_{N}` so
    # different N values do NOT collide on the cached integrator output.
    num_explores: int = 8


MethodSpec = Annotated[
    Union[
        TTSAgentSpec, TTSAgentMultiSpec, TTSAgentEffortSpec,
        SelfRefineSpec, SocraticSelfRefineSpec, BudgetForcingSpec,
        RerankSpec, StandaloneIntegratorSpec,
    ],
    Field(discriminator="name"),
]
