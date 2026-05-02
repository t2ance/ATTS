"""Reward Model Reranking baseline.

Reads pre-generated candidates from a cache directory, scores each with a
reward model (Skywork-Reward-V2 for text, VisualPRM for image benchmarks),
and picks the highest-scored candidate. No API calls -- only local GPU inference.

Skywork-Reward-V2 is an ORM (outcome reward model): one score per response.
VisualPRM is a PRM (process reward model): per-step scores, aggregated via
its built-in select_best_response() method.
"""

from __future__ import annotations

import base64
import io
import json

import torch
from PIL import Image

from dataclasses import replace
from methods.base import InfraConfig, create_solve_context, load_cached_candidates
from trajectory import CostTracker, RoundLog, SolveResult

# ---------------------------------------------------------------------------
# Lazy-loaded reward model singletons
# ---------------------------------------------------------------------------

_model = None
_tokenizer = None
_model_name = None


def _is_visualprm(model_name: str) -> bool:
    return "VisualPRM" in model_name


def _load_model(model_name: str):
    global _model, _tokenizer, _model_name
    if _model is not None and _model_name == model_name:
        return _model, _tokenizer

    if _is_visualprm(model_name):
        from transformers import AutoModel, AutoTokenizer
        _tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True, use_fast=False,
        )
        _model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            torch_dtype=torch.bfloat16,
        ).eval().cuda()
    else:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        _tokenizer = AutoTokenizer.from_pretrained(model_name)
        _model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            num_labels=1,
        )
        _model.eval()

    _model_name = model_name
    return _model, _tokenizer


# ---------------------------------------------------------------------------
# VisualPRM image preprocessing (from HuggingFace model card)
# ---------------------------------------------------------------------------

import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _build_transform(input_size: int):
    return T.Compose([
        T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def _dynamic_preprocess(image: Image.Image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height
    target_ratios = sorted(set(
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if min_num <= i * j <= max_num
    ), key=lambda x: x[0] * x[1])

    # Find closest aspect ratio
    best_ratio = (1, 1)
    best_diff = float("inf")
    area = orig_width * orig_height
    for ratio in target_ratios:
        diff = abs(aspect_ratio - ratio[0] / ratio[1])
        if diff < best_diff:
            best_diff = diff
            best_ratio = ratio
        elif diff == best_diff and area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
            best_ratio = ratio

    target_w = image_size * best_ratio[0]
    target_h = image_size * best_ratio[1]
    resized = image.resize((target_w, target_h))
    blocks = best_ratio[0] * best_ratio[1]
    processed = []
    for i in range(blocks):
        box = (
            (i % (target_w // image_size)) * image_size,
            (i // (target_w // image_size)) * image_size,
            ((i % (target_w // image_size)) + 1) * image_size,
            ((i // (target_w // image_size)) + 1) * image_size,
        )
        processed.append(resized.crop(box))
    if use_thumbnail and len(processed) != 1:
        processed.append(image.resize((image_size, image_size)))
    return processed


def _data_url_to_pixel_values(data_url: str) -> torch.Tensor:
    """Convert a base64 data URL to VisualPRM pixel_values tensor."""
    header, b64_data = data_url.split(",", 1)
    raw = base64.b64decode(b64_data)
    pil_image = Image.open(io.BytesIO(raw)).convert("RGB")
    transform = _build_transform(448)
    patches = _dynamic_preprocess(pil_image, image_size=448, use_thumbnail=True, max_num=12)
    pixel_values = torch.stack([transform(p) for p in patches])
    return pixel_values


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def _score_skywork(model, tokenizer, problem: str, reasoning: str, answer: str) -> float:
    """Score a candidate using Skywork-Reward-V2 chat format (ORM: one score)."""
    conversation = [
        {"role": "user", "content": problem},
        {"role": "assistant", "content": f"{reasoning}\n\nAnswer: {answer}"},
    ]
    input_ids = tokenizer.apply_chat_template(
        conversation, tokenize=True, return_tensors="pt",
    ).to(model.device)
    with torch.no_grad():
        output = model(input_ids=input_ids)
    return output.logits[0][0].item()


def _score_visualprm_batch(
    model, tokenizer, problem: str, candidates: list[dict], pixel_values: torch.Tensor,
) -> None:
    """Score all candidates using VisualPRM (PRM: per-step scores, aggregated).

    Uses the model's built-in select_best_response() which returns
    [(response, score), ...] sorted by score descending.
    Mutates each candidate dict to add 'reward_score'.
    """
    # Deduplicate: append index to make each response unique, so identical
    # responses don't collapse in the score mapping.
    response_list = [
        f"{c['reasoning']}\n\nAnswer: {c['answer']}\n[candidate {i}]"
        for i, c in enumerate(candidates)
    ]
    sorted_results = model.select_best_response(
        tokenizer=tokenizer,
        question=problem,
        response_list=response_list,
        pixel_values=pixel_values,
        return_scores=True,
    )
    score_by_response = {resp: score for resp, score in sorted_results}
    for cand, resp in zip(candidates, response_list):
        cand["reward_score"] = score_by_response[resp]


# ---------------------------------------------------------------------------
# Solve
# ---------------------------------------------------------------------------

async def solve(
    infra: InfraConfig,
    problem: str,
    image_data_url: str | None = None,
    question_id: str | None = None,
    reward_model_name: str | None = None,
    **_extra,
) -> SolveResult:
    """Rerank cached candidates using a reward model."""
    assert infra.cache_dir is not None, "cache_dir is required for rerank"
    assert question_id is not None, "question_id is required for rerank"
    assert reward_model_name is not None, "--reward-model is required for rerank"

    use_visualprm = _is_visualprm(reward_model_name)
    if use_visualprm:
        assert image_data_url is not None, "VisualPRM requires an image"

    ctx = create_solve_context(
        infra=replace(infra, timeout=None),
        problem=problem, image_data_url=image_data_url,
        question_id=question_id,
        writer_system_prompt="(reward model reranking -- no system prompt)",
        writer_user_message=problem,
        writer_header_lines=[
            f"**Reward Model**: {reward_model_name}",
            f"**Method**: rerank",
        ],
        writer_title_suffix="(rerank)",
    )

    # -- Load all cached candidates --
    question_cache = infra.cache_dir / question_id
    candidates = []
    idx = 1
    while True:
        result_path = question_cache / f"explore_{idx}" / "result.json"
        if not result_path.exists():
            break
        d = json.loads(result_path.read_text(encoding="utf-8"))
        if not d.get("timed_out"):
            answer = ctx.benchmark.get_answer_from_explore(d)
            candidates.append({
                "idx": idx,
                "answer": answer,
                "reasoning": d.get("reasoning", ""),
                "approach": d.get("approach", ""),
                "confidence": d.get("confidence", 0.0),
                "cost_usd": d.get("cost_usd", 0.0),
                "result": d,
            })
        idx += 1

    if len(candidates) == 0:
        print(f"  [rerank] No valid candidates for {question_id} -- returning empty answer")
        return SolveResult(answer="", cost=CostTracker(), rounds=[], writer=ctx.writer)

    # -- Score candidates --
    model, tokenizer = _load_model(reward_model_name)

    if use_visualprm:
        pixel_values = _data_url_to_pixel_values(image_data_url).to(torch.bfloat16).cuda()
        _score_visualprm_batch(model, tokenizer, problem, candidates, pixel_values)
    else:
        for cand in candidates:
            cand["reward_score"] = _score_skywork(
                model, tokenizer, problem, cand["reasoning"], cand["answer"],
            )

    # -- Pick best --
    best = max(candidates, key=lambda c: c["reward_score"])

    # -- Log all candidates with scores --
    ctx.writer.write_text(f"## Candidates ({len(candidates)} total)\n")
    has_image = image_data_url is not None
    for cand in candidates:
        selected = " **[SELECTED]**" if cand is best else ""
        image_section = "[Image]\n(attached)\n\n" if has_image else ""
        reward_input = (
            f"[Question]\n{problem}\n\n"
            f"{image_section}"
            f"[Response]\n{cand['reasoning']}\n\nAnswer: {cand['answer']}"
        )
        ctx.writer.write_text(
            f"### Explore {cand['idx']}{selected}\n"
            f"- **Answer**: {cand['answer']}\n"
            f"- **Reward Score**: {cand['reward_score']:.4f}\n"
            f"- **Confidence**: {cand['confidence']}\n\n"
            f"<details>\n<summary>Reward model input</summary>\n\n"
            f"```\n{reward_input}\n```\n\n</details>"
        )

    # -- Build rounds for grading compatibility --
    for cand in candidates:
        ctx.rounds.append(RoundLog(
            round_num=cand["idx"],
            action="explore",
            tool_input={
                "answer": cand["answer"],
                "cost_usd": cand["cost_usd"],
                "reward_score": cand["reward_score"],
                "selected": cand is best,
            },
        ))

    print(f"  [rerank] {len(candidates)} candidates scored, best=explore_{best['idx']} "
          f"(score={best['reward_score']:.4f}, answer={best['answer']})")

    return ctx.result(best["answer"])
