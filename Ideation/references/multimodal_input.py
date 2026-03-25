"""Shared helpers for text+image inputs across evaluation pipelines."""

from __future__ import annotations

from collections.abc import AsyncIterable
from typing import Any


def has_image(row: dict[str, Any]) -> bool:
    """Return True when a dataset row has a non-empty image field."""
    image = row.get("image")
    return isinstance(image, str) and bool(image.strip())


def normalize_image_data_url(image_field: str) -> str:
    """Validate and normalize a data URL image field."""
    if not isinstance(image_field, str):
        raise ValueError("image field must be a string data URL")

    value = image_field.strip()
    if not value:
        raise ValueError("image field is empty")
    if not value.startswith("data:") or ";base64," not in value:
        raise ValueError("image field must be in data:<media_type>;base64,<data> format")

    header, data = value.split(",", 1)
    media_type = header.removeprefix("data:").split(";", 1)[0]

    if not media_type:
        raise ValueError("image media type is missing")
    if not media_type.startswith("image/"):
        raise ValueError(f"unsupported media type: {media_type}")

    data = "".join(data.split())
    if not data:
        raise ValueError("image base64 payload is empty")

    return f"{header},{data}"


def extract_media_type(image_data_url: str) -> str:
    """Extract media type from a validated data URL."""
    header = image_data_url.split(",", 1)[0]
    media_type = header.removeprefix("data:").split(";", 1)[0]
    return media_type or "unknown"


def build_claude_content_blocks(text: str, image_data_url: str | None = None) -> list[dict[str, Any]]:
    """Build Claude message content blocks for text-only or text+image."""
    content: list[dict[str, Any]] = [{"type": "text", "text": text}]

    if image_data_url:
        media_type = extract_media_type(image_data_url)
        _, b64 = image_data_url.split(",", 1)
        content.append(
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": media_type,
                    "data": b64,
                },
            }
        )

    return content


def build_claude_prompt_events(
    text: str,
    image_data_url: str | None = None,
) -> AsyncIterable[dict[str, Any]]:
    """Build a one-message async prompt event stream for Claude Agent SDK."""

    async def _events() -> AsyncIterable[dict[str, Any]]:
        yield {
            "type": "user",
            "message": {
                "role": "user",
                "content": build_claude_content_blocks(text, image_data_url),
            },
        }

    return _events()


def build_openai_content(text: str, image_data_url: str | None = None) -> str | list[dict[str, Any]]:
    """Build OpenAI-compatible message content for ChatOpenAI."""
    if not image_data_url:
        return text

    return [
        {"type": "text", "text": text},
        {"type": "image_url", "image_url": {"url": image_data_url}},
    ]


def redact_image_for_logs(row: dict[str, Any]) -> dict[str, Any]:
    """Return lightweight image metadata safe for logs/results."""
    out: dict[str, Any] = {"has_image": has_image(row)}
    if not out["has_image"]:
        return out

    try:
        image_data_url = normalize_image_data_url(str(row["image"]))
        out["image_media_type"] = extract_media_type(image_data_url)
    except Exception:
        out["image_media_type"] = "invalid"
    return out
