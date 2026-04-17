"""Level-3 reward function for ATTS GRPO.

Level 3 reward (Howard step-value decomposition under Principle P6):
    R(tau) = gamma * V_star_N + rho * y_T - lambda * N

where
    V_star_N = max_t y_t   (oracle discovery indicator over all explore steps)
    y_t      = grade(explorer's self-reported answer at step t, ground_truth)
    y_T      = grade(final StructuredOutput answer, ground_truth)
    N        = clamped explore count, capped at N_MAX

Coefficients (from principle, not tuned):
    gamma = 0.5   (discovery credit)
    rho   = 0.5   (integration credit; gamma + rho = 1 keeps bounded upper at 1)
    lambda= 0.05  (cost coefficient, same as L0)
    N_MAX = 9     (matches rollout loop assistant-turn budget minus StructuredOutput turn)

Grader:
    - Explore steps: use pre-computed grade from cached_explores[*].is_correct (no live call)
    - Final answer: call deployed model at JUDGE_URL; fallback to 0.0 on error (with log)
"""

from __future__ import annotations

import json
import logging
import re
import sys
from pathlib import Path
from typing import Any

from openai import OpenAI

# Reuse the canonical FullRenderer.parse so reward_fn is not coupled to the
# rendered tool-response format. See methods/tool_io.py for the contract.
_CORE_CODE_DIR = Path(__file__).resolve().parent.parent.parent
if str(_CORE_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CORE_CODE_DIR))

from methods.tool_io import FullRenderer  # noqa: E402

_RENDERER = FullRenderer()


# ---------- Hyperparameters (L3, from principles) ----------
GAMMA = 0.5
RHO = 0.5
LAMBDA_COST = 0.05
N_MAX = 9
HAS_ANSWER_BONUS = 0.1  # reward for emitting StructuredOutput even when answer wrong

# ---------- Judge server config ----------
JUDGE_URL = "http://127.0.0.1:8000/v1"
JUDGE_MODEL = "Qwen/Qwen3-8B"
JUDGE_TIMEOUT_S = 300.0

_JUDGE_SYSTEM = """\
Judge whether the following [response] to [question] is correct based on the [correct_answer].

Your judgement must follow these criteria:
- extracted_final_answer: The final exact answer extracted from the [response]. Put "None" if there is no exact final answer to extract.
- reasoning: Explain why the extracted answer is correct or incorrect based on [correct_answer]. Focus only on whether there are meaningful differences. Do not comment on background, do not attempt to solve the problem, do not argue for any answer different than [correct_answer].
- correct: true if extracted_final_answer matches [correct_answer], or is within a small margin of error for numerical problems. Consider mathematical equivalence (1/2 = 0.5), notation differences (LaTeX vs Unicode), and formatting differences that don't change meaning. false otherwise."""

_JUDGE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "extracted_final_answer": {"type": "string"},
        "reasoning": {"type": "string"},
        "correct": {"type": "boolean"},
    },
    "required": ["extracted_final_answer", "reasoning", "correct"],
    "additionalProperties": False,
}

_JUDGE_CLIENT: OpenAI | None = None
_LOGGER = logging.getLogger(__name__)


def _get_judge_client() -> OpenAI:
    global _JUDGE_CLIENT
    if _JUDGE_CLIENT is None:
        _JUDGE_CLIENT = OpenAI(base_url=JUDGE_URL, api_key="dummy", timeout=JUDGE_TIMEOUT_S)
    return _JUDGE_CLIENT


# ---------- LLM judge ----------


def _string_match_grade(predicted: str, ground_truth: str) -> float:
    """Fallback when judge truncates: normalized exact string match."""
    def _norm(s: str) -> str:
        s = str(s).strip()
        s = re.sub(r"^\$+|\$+$", "", s)
        s = re.sub(r"^\\boxed\{(.+)\}$", r"\1", s)
        s = re.sub(r"\s+", " ", s).strip()
        return s.lower()
    return 1.0 if _norm(predicted) == _norm(ground_truth) else 0.0


def _judge_remote(predicted: str, ground_truth: str, question: str) -> float:
    """Call deployed model judge; returns 1.0 / 0.0. Raises on any failure."""
    client = _get_judge_client()
    user = f"[question]: {question}\n[response]: {predicted}\n[correct_answer]: {ground_truth}"
    resp = client.chat.completions.create(
        model=JUDGE_MODEL,
        messages=[
            {"role": "system", "content": _JUDGE_SYSTEM},
            {"role": "user", "content": user},
        ],
        temperature=0.0,
        max_tokens=4096,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "judge_result",
                "schema": _JUDGE_SCHEMA,
                "strict": True,
            },
        },
        extra_body={
            "chat_template_kwargs": {"enable_thinking": False},
            "repetition_penalty": 1.1,
        },
    )
    content = resp.choices[0].message.content or ""
    finish_reason = resp.choices[0].finish_reason
    assert content.strip(), f"judge returned empty content, finish={finish_reason}"
    if finish_reason == "length":
        score = _string_match_grade(predicted, ground_truth)
        _LOGGER.warning(
            "judge truncated (max_tokens=4096), falling back to string match: "
            "score=%s pred=%r gold=%r", score, predicted[:100], ground_truth[:100],
        )
        return score
    assert finish_reason == "stop", f"judge unexpected finish_reason={finish_reason!r}"
    result = json.loads(content)
    assert isinstance(result.get("correct"), bool), f"judge bad correct field: {result}"
    return 1.0 if result["correct"] else 0.0


# ---------- Per-step parsers for L3 ----------


_EXPLORE_TOOL_CALL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
_STRUCT_OUTPUT_ANSWER_RE = re.compile(
    r'"name"\s*:\s*"StructuredOutput".*?"answer"\s*:\s*"([^"]*)"', re.DOTALL
)
_TOOL_RESPONSE_RE = re.compile(r"<tool_response>(.*?)</tool_response>", re.DOTALL)
_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def _strip_think_blocks(text: str) -> str:
    """Remove <think>...</think> blocks to prevent hallucinated content
    inside thinking from being parsed as real tool calls/responses."""
    return _THINK_BLOCK_RE.sub("", text)


def _extract_final_answer(solution_str: str) -> str:
    """Last StructuredOutput tool call's `answer` field (unchanged from v1)."""
    matches = _STRUCT_OUTPUT_ANSWER_RE.findall(solution_str)
    return matches[-1] if matches else ""


def _extract_explore_tool_responses(solution_str: str) -> list[tuple[int, str]]:
    """Extract explorer (idx, self-reported answer) pairs from each
    <tool_response>...</tool_response> block.

    Returns 1-based Candidate #N idx instead of cache_id. Caller indexes
    cached_explores[idx-1] directly (offline-shuffled order from parquet).
    Timeout candidates and empty-answer responses are skipped.
    """
    responses: list[tuple[int, str]] = []
    for m in _TOOL_RESPONSE_RE.finditer(solution_str):
        body = m.group(1).strip()
        if not body.startswith("Candidate #"):
            continue
        rec = _RENDERER.parse(body)
        if rec.answer:
            responses.append((rec.idx, rec.answer))
    return responses


def _count_explore_tool_calls(solution_str: str) -> int:
    """Count complete <tool_call>...</tool_call> blocks that invoke explore,
    capped at N_MAX (Principle P4: boundedness)."""
    blocks = _EXPLORE_TOOL_CALL_RE.findall(solution_str)
    count = sum(1 for b in blocks if '"name"' in b and '"explore"' in b)
    return min(count, N_MAX)


# ---------- Main entry: compute_score ----------


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict | None = None,
    **kwargs,
) -> dict:
    """Compute Level-3 trajectory reward + auxiliary metrics for ATTS GRPO.

    Returns a dict consumed by verl's reward manager:
        score:           gamma*V_star_N + rho*y_T - lambda*N  (goes to gradient)
        acc:             raw correctness of final answer (val-core key)
        num_explores:    explore tool call count
        has_answer:      1.0 if model emitted StructuredOutput, else 0.0
        penalty:         lambda * N
        discovery:       V_star_N  (any explorer found correct answer?)
        integration:     y_T given V_star_N = 1 (final synthesis preserved it?)
                         NOTE: integration is only meaningful when discovery = 1;
                         when discovery = 0 we still log 0 for schema uniformity.
        grader_gap:      |V_star_N - y_T| (diagnostic: grader disagreement signal)
    """
    # Pull question for judge. Extra_info may be None if verl forgot to pass it;
    # we fall back to empty string in that case (Principle P4: never crash
    # inside the loss loop from missing metadata, but log it so we notice).
    question = ""
    if extra_info is not None:
        question = extra_info.get("question", "") or ""

    tools_kwargs = (extra_info or {}).get("tools_kwargs") or {}
    explore_kwargs = ((tools_kwargs.get("explore") or {}).get("create_kwargs") or {})
    cached_explores_ordered = explore_kwargs.get("cached_explores") or []
    assert cached_explores_ordered, (
        "extra_info.tools_kwargs.explore.create_kwargs.cached_explores missing; "
        "explore grading requires offline-prepared cached explores"
    )

    # Strip <think> blocks before parsing to prevent hallucinated tool
    # calls/responses inside thinking from being matched by regexes.
    solution_str = _strip_think_blocks(solution_str)

    # Final answer correctness y_T via deployed judge
    final_answer = _extract_final_answer(solution_str)
    y_T = _judge_remote(final_answer, ground_truth, question) if final_answer else 0.0

    # Per-step explorer answers and y_{1:N}. Consume pre-computed is_correct
    # grades by positional index into the offline-shuffled cached_explores.
    explore_answers = _extract_explore_tool_responses(solution_str)
    y_per_step = []
    for idx, answer in explore_answers:
        if idx < 1 or idx > len(cached_explores_ordered):
            raise ValueError(
                f"Candidate #{idx} out of range; "
                f"cached_explores has {len(cached_explores_ordered)} entries"
            )
        is_correct = cached_explores_ordered[idx - 1].get("is_correct")
        if not isinstance(is_correct, bool):
            raise ValueError(
                f"cached_explores[{idx-1}] missing bool is_correct: "
                f"{cached_explores_ordered[idx-1]}"
            )
        y_per_step.append(1.0 if (is_correct and answer) else 0.0)
    V_star_N = max(y_per_step) if y_per_step else 0.0

    # Explore count N (for cost)
    num_explores = _count_explore_tool_calls(solution_str)
    penalty = LAMBDA_COST * num_explores

    # has_answer bonus: encourage StructuredOutput emission when answer is wrong
    has_answer_bonus = HAS_ANSWER_BONUS if (final_answer and y_T < 1.0) else 0.0

    # Level-3 reward
    reward = GAMMA * V_star_N + RHO * y_T + has_answer_bonus - penalty

    return {
        "score": reward,
        "acc": y_T,  # primary val-core metric = final answer correctness
        "num_explores": float(num_explores),
        "has_answer": 1.0 if final_answer else 0.0,
        "penalty": penalty,
        "discovery": V_star_N,
        "integration": y_T if V_star_N >= 1.0 else 0.0,
        "grader_gap": abs(V_star_N - y_T),
    }
