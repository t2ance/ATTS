"""Method spec sub-schemas: discriminated union over method name.

Mirror of benchmarks/specs.py. Each method gets one Pydantic class enumerating
exactly its valid fields. extra="forbid" rejects unknown YAML keys, so dead
fields (e.g. orchestrator_model in self-refine YAMLs) fail validation instead
of being silently ignored.

Pairs with methods/registry.py (the runtime behavior layer).
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Annotated, Literal, Union, TYPE_CHECKING

from pydantic import BaseModel, Field, model_validator

if TYPE_CHECKING:
    from cache_types import Exploration, JudgeOutcome

logger = logging.getLogger(__name__)


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
    # effort default went from "low" to None on 2026-05-05. Rationale: yaml
    # users wanted a way to say "don't override the API/backend's own default".
    # None is the only value Claude Agent SDK / Codex / OpenRouter / vLLM
    # all interpret as "omit the reasoning.effort param entirely" — claude
    # backend skips `--effort` flag (claude_agent_sdk subprocess_cli.py:315),
    # codex/openrouter skip the `extra_body['reasoning']` block (their
    # `if effort:` guards), vllm just doesn't pass it. With effort=None,
    # the Anthropic API server applies its own model-builtin default = "high"
    # for Opus 4.6/4.7 and Sonnet 4.6 (verified 2026-05-05 via official docs:
    # https://platform.claude.com/docs/en/build-with-claude/effort —
    # "By default, Claude uses high effort"; "Setting effort to high produces
    # exactly the same behavior as omitting the effort parameter entirely").
    # Codex/OpenRouter/vLLM defaults are model-dependent. Warning is emitted
    # at validate-time so the user cannot quietly miss the silent jump from
    # the historical "low" to the API-default "high" (~2-5x token cost).
    effort: Literal["low", "medium", "high", "max"] | None = None
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
        if self.effort is None:
            logger.warning(
                f"ModelConfig(backend={self.backend!r}, model={self.model!r}): "
                f"effort=None (yaml omits the field or sets it to null). "
                f"None is passed through unchanged: claude backend skips "
                f"--effort CLI flag, codex/openrouter skip "
                f"extra_body['reasoning']['effort'], vllm ignores it. "
                f"For claude backend on Sonnet 4.6 / Opus 4.6 / 4.7, the API "
                f"server then applies its model-builtin default = 'high' "
                f"(see https://platform.claude.com/docs/en/build-with-claude/effort). "
                f"This is ~2-5x the token cost of 'low'; if you want the "
                f"old 'low' default, write effort: low explicitly."
            )
        return self


class RoleSlot(BaseModel):
    """A model invocation. Used by single-call cacheless roles (e.g. integrate).

    No cache_dir: integrate input is candidate content, which is not encoded
    in any cache_key; caching by (qid, count) is content-blind and unsafe.
    Pydantic extra="forbid" rejects num_explores or other ExploreVariant-shaped
    fields at config-load.
    """
    model_config = {"extra": "forbid"}
    model: ModelConfig


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

    # ---- Internal helpers (path construction + atomic I/O) ----

    def _explore_dir(self, qid: str, idx: int, rollout_idx: int | None = None) -> Path:
        """Path to one explore's bundle directory.
        K=1 (rollout_idx=None) -> cache_dir/<qid>/explore_<idx>
        K>1 (rollout_idx=k)    -> cache_dir/<qid>/rollout_<k>/explore_<idx>
        """
        base = self.cache_dir / qid
        if rollout_idx is not None:
            base = base / f"rollout_{rollout_idx}"
        return base / f"explore_{idx}"

    def _judge_dir(self, qid: str, idx: int, label: str,
                   rollout_idx: int | None = None) -> Path:
        return self._explore_dir(qid, idx, rollout_idx) / "judges" / label

    def _has_explore(self, qid: str, idx: int, rollout_idx: int | None = None) -> bool:
        return (self._explore_dir(qid, idx, rollout_idx) / "result.json").exists()

    def _load_explore(self, qid: str, idx: int,
                      rollout_idx: int | None = None) -> "Exploration | None":
        """Load Exploration from disk; None if result.json missing."""
        from cache_types import Exploration
        d = self._explore_dir(qid, idx, rollout_idx)
        rp = d / "result.json"
        if not rp.exists():
            return None
        payload = json.loads(rp.read_text(encoding="utf-8"))
        traj = (d / "output.md").read_text(encoding="utf-8") if (d / "output.md").exists() else ""
        reserved = {"answer", "cost_usd", "model", "timed_out"}
        return Exploration(
            qid=qid, idx=idx, rollout_idx=rollout_idx,
            answer=payload.get("answer", ""),
            trajectory=traj,
            cost_usd=payload.get("cost_usd", 0.0),
            model=payload.get("model", ""),
            timed_out=payload.get("timed_out", False),
            extra={k: v for k, v in payload.items() if k not in reserved},
        )

    def _load_judge(self, qid: str, idx: int, judge_spec: dict | None,
                    rollout_idx: int | None = None) -> "JudgeOutcome | None":
        """Look up the cached judge bundle for (qid, idx) under this variant.

        Match policy is UNIDIRECTIONAL by design (preserved verbatim from the
        pre-refactor `benchmarks.base.find_cached_judge`):
          - stored == spec                  -> exact hit
          - stored is strict subset of spec -> best-effort hit (legitimate
              schema evolution: cache predates a new optional field)
          - shared key with disagreeing val -> RuntimeError (true conflict)
          - stored has key absent from spec -> RuntimeError (cache was made
              under stricter spec; refusing to inherit its verdict for a
              less-specific request)
          - config / grade missing          -> None (real cache miss)
        """
        from cache_types import JudgeOutcome, _JUDGE_CACHE_STATS
        label = JudgeOutcome.label_for(judge_spec)
        if label is None:
            return None
        jd = self._judge_dir(qid, idx, label, rollout_idx)
        gp = jd / "grade.json"
        cp = jd / "config.json"
        rp = jd / "result.json"
        if not (gp.exists() and cp.exists()):
            return None
        stored = json.loads(cp.read_text(encoding="utf-8"))

        if stored == judge_spec:
            _JUDGE_CACHE_STATS["exact_hits"] += 1
        else:
            shared = set(stored) & set(judge_spec)
            conflicts = {k: (stored[k], judge_spec[k]) for k in shared
                         if stored[k] != judge_spec[k]}
            if conflicts:
                raise RuntimeError(
                    f"Judge config value conflict at {jd}.\n"
                    f"  Conflicting keys (stored vs requested): {conflicts}\n"
                    f"  Stored:    {stored}\n"
                    f"  Requested: {judge_spec}\n"
                    f"Wipe the bundle or rename the label before re-running."
                )
            only_stored = sorted(set(stored) - set(judge_spec))
            if only_stored:
                raise RuntimeError(
                    f"Judge cache spec narrowing at {jd}.\n"
                    f"  Stored has keys absent from requested: {only_stored}\n"
                    f"  Stored:    {stored}\n"
                    f"  Requested: {judge_spec}\n"
                    f"Cached verdict was produced under a non-default spec; "
                    f"refusing to reuse it for a less-specific request."
                )
            only_requested = sorted(set(judge_spec) - set(stored))
            _JUDGE_CACHE_STATS["best_effort_hits"] += 1
            _JUDGE_CACHE_STATS["best_effort_extras"].update(only_requested)

        grade = json.loads(gp.read_text(encoding="utf-8"))
        return JudgeOutcome(
            is_correct=grade["is_correct"],
            cost_usd=grade["cost_usd"],
            judge_spec_snapshot=stored,
            input_md=(jd / "input.md").read_text(encoding="utf-8") if (jd / "input.md").exists() else "",
            output_md=(jd / "output.md").read_text(encoding="utf-8") if (jd / "output.md").exists() else "",
            result_dict=json.loads(rp.read_text(encoding="utf-8")) if rp.exists() else {},
        )

    # ---- Public intent-driven API --------------------------------------

    async def get_exploration(
        self,
        qid: str,
        idx: int,
        *,
        rollout_idx: int | None = None,
        generate_fn,
        grader=None,
    ) -> "Exploration":
        """Cache hit -> return cached Exploration. Cache miss -> call generate_fn,
        persist, return. Optional grader -> also produce + persist verdict.

        Always returns Exploration. Never None. Cache miss with a generate_fn
        that fails (raises) propagates that failure -- there is no silent
        degradation.
        """
        from cache_types import Exploration
        exp = self._load_explore(qid, idx, rollout_idx)
        # Surfaced to eval.log so users can audit hit/miss without inspecting
        # the cache directory mtimes. variant=self.label disambiguates multi-
        # variant runs (multi_model / effort modes); rollout_idx is included
        # for K>1 num_rollouts setups.
        rollout_tag = f" rollout={rollout_idx}" if rollout_idx is not None else ""
        if exp is None:
            logger.info(
                f"[cache miss] explore qid={qid} idx={idx} "
                f"variant={self.label}{rollout_tag} -> calling generate_fn"
            )
            exp = await generate_fn()
            assert isinstance(exp, Exploration), (
                f"generate_fn must return Exploration, got {type(exp).__name__}"
            )
            exp.qid, exp.idx, exp.rollout_idx = qid, idx, rollout_idx
            exp.persist(self._explore_dir(qid, idx, rollout_idx))
            logger.info(
                f"[cache writeback] explore qid={qid} idx={idx} "
                f"variant={self.label}{rollout_tag} -> {self._explore_dir(qid, idx, rollout_idx)}"
            )
        else:
            logger.info(
                f"[cache hit] explore qid={qid} idx={idx} "
                f"variant={self.label}{rollout_tag} timed_out={exp.timed_out}"
            )

        if grader is not None:
            cached_outcome = self._load_judge(qid, idx, grader.judge_spec, rollout_idx)
            if cached_outcome is not None:
                exp.verdict = cached_outcome
            else:
                outcome = await grader(exp.answer, qid)
                exp.verdict = outcome
                if outcome.label is not None:
                    outcome.persist(self._judge_dir(qid, idx, outcome.label, rollout_idx))
        return exp

    async def get_all_explorations(
        self,
        qid: str,
        *,
        rollout_idx: int | None = None,
        grader=None,
    ) -> list["Exploration"]:
        """Pure read of cached explorations under (qid, rollout_idx).

        Walks the directory, returns Explorations in ascending idx order.
        Empty list is a legitimate result, not an error.

        If grader provided, attach verdict to each Exploration (using judge
        cache when present; calling grader on miss).
        """
        from cache_types import Exploration

        base = self.cache_dir / qid
        if rollout_idx is not None:
            base = base / f"rollout_{rollout_idx}"
        if not base.exists():
            return []
        found: list[int] = []
        for p in base.iterdir():
            if not (p.is_dir() and p.name.startswith("explore_")):
                continue
            if not (p / "result.json").exists():
                continue
            try:
                found.append(int(p.name.split("_", 1)[1]))
            except ValueError:
                continue
        found.sort()

        explorations: list[Exploration] = []
        for idx in found:
            exp = self._load_explore(qid, idx, rollout_idx)
            assert exp is not None
            if grader is not None:
                cached_outcome = self._load_judge(qid, idx, grader.judge_spec, rollout_idx)
                if cached_outcome is not None:
                    exp.verdict = cached_outcome
                else:
                    outcome = await grader(exp.answer, qid)
                    exp.verdict = outcome
                    if outcome.label is not None:
                        outcome.persist(self._judge_dir(qid, idx, outcome.label, rollout_idx))
            explorations.append(exp)
        return explorations


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
    # Where to read cached explore candidates from. Top-level (not on RoleSlot)
    # because the integrate role is cacheless after 2026-05-05 refactor; this
    # field points at the explore cache produced by precache.
    cache_dir: Path
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
