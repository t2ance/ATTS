"""HLE-Verified benchmark configuration."""

from __future__ import annotations

import json
import logging
import os
import zipfile
from pathlib import Path

logger = logging.getLogger(__name__)

from benchmarks.base import BenchmarkConfig
from benchmarks.grader import check_answer, judge_answer
from multimodal_input import has_image, normalize_image_data_url


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

_HF_HUB = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"
DATA_DIR = _HF_HUB / "datasets--skylenage--HLE-Verified"


def _load_via_datasets() -> list[dict] | None:
    """Load via datasets library (supports parquet cache). Returns None if not installed."""
    try:
        from datasets import load_dataset
    except ImportError:
        return None
    ds = load_dataset("skylenage/HLE-Verified")
    split = "train" if "train" in ds else list(ds.keys())[0]
    return list(ds[split])


def _find_data_dir() -> Path | None:
    """Find the snapshot data directory (zip format). Returns None if not found."""
    snapshots = DATA_DIR / "snapshots"
    if not snapshots.exists():
        return None
    dirs = list(snapshots.iterdir())
    if not dirs:
        return None
    data_dir = dirs[0] / "data"
    zips = list(data_dir.glob("*.zip")) if data_dir.exists() else []
    return data_dir if zips else None


def _load_hle_dataset() -> list[dict]:
    """Load all rows from the HLE-Verified dataset."""
    rows = _load_via_datasets()
    if rows is not None:
        return rows
    data_dir = _find_data_dir()
    if data_dir is None:
        logger.info("Dataset not found. Downloading via datasets library...")
        rows = _load_via_datasets()
        if rows is not None:
            return rows
        raise FileNotFoundError(
            "Could not load HLE-Verified. Try: python -c \"from datasets import load_dataset; "
            "load_dataset('skylenage/HLE-Verified')\""
        )
    all_rows = []
    for zpath in sorted(data_dir.glob("*.zip")):
        with zipfile.ZipFile(zpath) as zf:
            for name in zf.namelist():
                if name.endswith(".jsonl"):
                    with zf.open(name) as f:
                        for line in f:
                            all_rows.append(json.loads(line))
    return all_rows


def _classify_subset(row: dict) -> str:
    """Classify a row into Gold / Revision / Uncertain."""
    vc = row.get("Verified_Classes", "")
    if "Gold" in vc:
        return "gold"
    if "Revision" in vc:
        return "revision"
    if "Uncertain" in vc:
        return "uncertain"
    has_original = any(k in row for k in ("original_question", "original_answer", "original_rationale"))
    if has_original:
        return "revision"
    if "verify_meta_info" in row:
        return "gold"
    return "uncertain"


def _filter_dataset(
    rows: list[dict],
    subset: str | None = None,
    category: str | None = None,
    text_only: bool = False,
) -> list[dict]:
    """Filter dataset by subset and/or category."""
    filtered = []
    for r in rows:
        if subset and _classify_subset(r) != subset.lower():
            continue
        if category and r.get("category", "").lower() != category.lower():
            continue
        if text_only and r.get("image"):
            continue
        filtered.append(r)
    return filtered


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

class HLEBenchmark(BenchmarkConfig):
    name = "hle"
    majority_vote_compatible = False
    # 2026-05-01: judge_model class attribute removed. Judge identity now lives
    # in YAML (benchmark.judge: {name, model, ...}) and arrives via __init__'s
    # judge_spec. The bespoke "vllm -> claude" and "codex -> gpt-5-codex-mini"
    # routing that used to live here is also gone: backend mapping was a
    # workaround for the fact that orchestrator backend leaked into judge
    # selection. Now the user picks the judge backend explicitly in YAML.
    grading_summary = (
        "LLM judge per YAML benchmark.judge block; "
        "multipleChoice rows fall through to string match"
    )

    def load_dataset(self) -> list[dict]:
        return _load_hle_dataset()

    def filter_dataset(self, rows: list[dict], **kwargs) -> list[dict]:
        return _filter_dataset(rows, **kwargs)

    def get_question(self, row: dict) -> str:
        return row["question"]

    def get_answer(self, row: dict) -> str:
        return str(row["answer"])

    def get_id(self, row: dict) -> str:
        return row["id"]

    def get_image(self, row: dict) -> str | None:
        if has_image(row):
            return normalize_image_data_url(row["image"])
        return None

    def classify_subset(self, row: dict) -> str:
        return _classify_subset(row)

    async def grade(self, predicted, gold, question, row, backend, out_dir=None):
        # HLE rows carry per-row answer_type; multipleChoice rows skip the LLM judge.
        answer_type = row.get("answer_type", "exactMatch")
        if answer_type == "multipleChoice":
            return check_answer(predicted, gold, "multipleChoice"), 0.0
        # `backend` (orchestrator backend) is intentionally ignored here: the
        # judge backend lives entirely in self.judge_spec["name"] per YAML.
        return await judge_answer(
            predicted, gold, question, self.judge_spec,
            max_retries=self.judge_max_retries,
            out_dir=out_dir,
        )

