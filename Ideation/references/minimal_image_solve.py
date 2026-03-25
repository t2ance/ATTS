"""Minimal fixed-sample image solve using Claude Agent SDK.

This uses streaming input mode (AsyncIterable prompt), which supports image uploads.
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
from pathlib import Path
import zipfile

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ResultMessage,
    query,
)

# Allow running inside a Claude Code session
os.environ.pop("CLAUDECODE", None)

SAMPLE_ID = "6687ffb1091058ff19128813"
QUESTION = (
    "Black to move. Without moving the black queens, which sequence is mate in 2 for black, "
    "regardless of what white does? Use standard chess notation, leaving out the white move."
)
IMAGE_PATH = Path(__file__).resolve().parents[1] / "tmp_images" / f"{SAMPLE_ID}.jpg"
HLE_CACHE_DIR = Path.home() / ".cache/huggingface/hub/datasets--skylenage--HLE-Verified"


def _image_content_block(path: Path) -> dict:
    data = base64.b64encode(path.read_bytes()).decode("utf-8")
    return {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": "image/jpeg",
            "data": data,
        },
    }


def _ensure_image_file() -> None:
    if IMAGE_PATH.exists():
        return

    snapshots = HLE_CACHE_DIR / "snapshots"
    if not snapshots.exists():
        raise FileNotFoundError(
            f"Image not found at {IMAGE_PATH}, and HLE cache missing: {snapshots}"
        )

    IMAGE_PATH.parent.mkdir(parents=True, exist_ok=True)

    for snap in snapshots.iterdir():
        data_dir = snap / "data"
        if not data_dir.exists():
            continue
        for zpath in sorted(data_dir.glob("*.zip")):
            with zipfile.ZipFile(zpath) as zf:
                for name in zf.namelist():
                    if not name.endswith(".jsonl"):
                        continue
                    with zf.open(name) as f:
                        for line in f:
                            row = json.loads(line)
                            if row.get("id") != SAMPLE_ID:
                                continue
                            image = row.get("image", "")
                            if not isinstance(image, str) or "," not in image:
                                raise ValueError(
                                    f"Sample {SAMPLE_ID} found but image field is invalid."
                                )
                            _, b64 = image.split(",", 1)
                            IMAGE_PATH.write_bytes(base64.b64decode(b64))
                            return

    raise FileNotFoundError(
        f"Could not find sample id={SAMPLE_ID} with image in local HLE cache."
    )


async def _prompt_events():
    yield {
        "type": "user",
        "message": {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        f"Solve this HLE-Verified sample (id={SAMPLE_ID}).\n\n"
                        f"Question:\n{QUESTION}\n\n"
                        "Return only the final mating sequence and one short justification."
                    ),
                },
                _image_content_block(IMAGE_PATH),
            ],
        },
    }


def _assistant_text(msg: AssistantMessage) -> str:
    if isinstance(msg.content, str):
        return msg.content
    texts = []
    for block in msg.content:
        text = getattr(block, "text", None)
        if text:
            texts.append(text)
    return "\n".join(texts)


async def main() -> None:
    _ensure_image_file()

    opts = ClaudeAgentOptions(
        model="claude-sonnet-4-6",
        allowed_tools=[],
        max_turns=3,
    )

    print(f"SAMPLE_ID: {SAMPLE_ID}")
    print(f"IMAGE:     {IMAGE_PATH}")
    print("Running...\n")

    async for message in query(prompt=_prompt_events(), options=opts):
        if isinstance(message, AssistantMessage):
            text = _assistant_text(message).strip()
            if text:
                print("[assistant]")
                print(text)
                print()
        elif isinstance(message, ResultMessage):
            print("[result]")
            print(f"stop_reason: {message.subtype}")
            print(f"cost_usd:    {message.total_cost_usd}")
            print(f"usage:       {message.usage}")


if __name__ == "__main__":
    asyncio.run(main())
