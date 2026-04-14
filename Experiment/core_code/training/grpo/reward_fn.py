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
from typing import Any

from openai import OpenAI


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
_JUDGE_ERROR_COUNT = 0
_LOGGER = logging.getLogger(__name__)


def _get_judge_client() -> OpenAI:
    global _JUDGE_CLIENT
    if _JUDGE_CLIENT is None:
        _JUDGE_CLIENT = OpenAI(base_url=JUDGE_URL, api_key="dummy", timeout=JUDGE_TIMEOUT_S)
    return _JUDGE_CLIENT


# ---------- LLM judge ----------


def _judge_remote(predicted: str, ground_truth: str, question: str) -> float:
    """Call deployed model judge; returns 1.0 / 0.0. Falls back to 0.0 with error log on failure."""
    global _JUDGE_ERROR_COUNT
    try:
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
        assert finish_reason == "stop", (
            f"judge truncated (finish={finish_reason}, max_tokens=4096). "
            f"reasoning field likely too long for budget. content[:500]={content[:500]!r}"
        )
        result = json.loads(content)
        assert isinstance(result.get("correct"), bool), f"judge bad correct field: {result}"
        return 1.0 if result["correct"] else 0.0
    except Exception as exc:
        _JUDGE_ERROR_COUNT += 1
        _LOGGER.exception(
            "[reward_fn] judge failed; fallback score=0.0 (count=%d): %s",
            _JUDGE_ERROR_COUNT,
            exc,
        )
        return 0.0


# ---------- Per-step parsers for L3 ----------


_EXPLORE_TOOL_CALL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
_STRUCT_OUTPUT_ANSWER_RE = re.compile(
    r'"name"\s*:\s*"StructuredOutput".*?"answer"\s*:\s*"([^"]*)"', re.DOTALL
)
_EXPLORE_CACHE_ID_RE = re.compile(r"- Cache ID:\s*([^\n]+)")


def _extract_final_answer(solution_str: str) -> str:
    """Last StructuredOutput tool call's `answer` field (unchanged from v1)."""
    matches = _STRUCT_OUTPUT_ANSWER_RE.findall(solution_str)
    return matches[-1] if matches else ""


def _extract_explore_tool_responses(solution_str: str) -> list[tuple[str, str]]:
    """Extract explorer cache ids + self-reported answers from tool RESPONSES.

    ExploreTool.execute (training/grpo/explore_tool.py) returns plain text:
        Candidate #N recorded.
        - Answer: <answer>
        - Confidence: <confidence>
        - Approach: <approach>
        - Reasoning: <reasoning>

    verl's ToolAgentLoop wraps that as {"role": "tool", "content": <text>} and
    Qwen's chat template emits it inside <tool_response>...</tool_response>.
    We walk each <tool_response> block and pull the text between
    `- Answer: ` and the next `\\n- Confidence:` line (the answer itself is
    emitted on one line by the f-string, but we DOTALL-match for safety).
    """
    responses: list[tuple[str, str]] = []
    for m in re.finditer(r"<tool_response>(.*?)</tool_response>", solution_str, re.DOTALL):
        body = m.group(1)
        cache_match = _EXPLORE_CACHE_ID_RE.search(body)
        ans_match = re.search(r"- Answer:\s*(.*?)\s*\n- Confidence:", body, re.DOTALL)
        if ans_match and cache_match:
            responses.append((cache_match.group(1).strip(), ans_match.group(1).strip()))
    return responses


def _extract_cached_explore_grades(extra_info: dict | None) -> dict[str, float]:
    """Build strict cache-id -> correctness map from exported cached explores.

    If cached explores are present, every entry must carry both `cache_id` and
    `is_correct`; otherwise we raise loudly rather than silently re-grade.
    """
    if extra_info is None:
        return {}
    tools_kwargs = extra_info.get("tools_kwargs") or {}
    explore_kwargs = ((tools_kwargs.get("explore") or {}).get("create_kwargs") or {})
    cached_explores = explore_kwargs.get("cached_explores") or []
    if not cached_explores:
        return {}

    grade_by_cache_id: dict[str, float] = {}
    for idx, explore in enumerate(cached_explores, start=1):
        cache_id = str(explore.get("cache_id", "")).strip()
        if not cache_id:
            raise ValueError(f"cached explore #{idx} missing cache_id")
        is_correct = explore.get("is_correct")
        if not isinstance(is_correct, bool):
            raise ValueError(f"cached explore {cache_id} missing bool is_correct")
        grade_by_cache_id[cache_id] = 1.0 if is_correct else 0.0
    return grade_by_cache_id


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
    cached_explore_grades = _extract_cached_explore_grades(extra_info)

    # Final answer correctness y_T via deployed judge
    final_answer = _extract_final_answer(solution_str)
    y_T = _judge_remote(final_answer, ground_truth, question) if final_answer else 0.0

    # Per-step explorer answers and y_{1:N}. When cached explores are present,
    # consume their exported grades directly instead of re-grading.
    explore_answers = _extract_explore_tool_responses(solution_str)
    assert cached_explore_grades, "no cached_explore_grades: explore grading requires pre-computed cache"
    y_per_step = []
    for cache_id, answer in explore_answers:
        if cache_id not in cached_explore_grades:
            raise ValueError(f"tool response referenced unknown cached explore {cache_id}")
        y_per_step.append(cached_explore_grades[cache_id] if answer else 0.0)
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
