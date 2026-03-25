"""HLE-Verified benchmark configuration."""

from __future__ import annotations

import argparse
import json
import os
import zipfile
from pathlib import Path

from benchmarks.base import BenchmarkConfig
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
        print("Dataset not found. Downloading via datasets library...")
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
    filter_keys = ("subset", "category", "text_only")
    majority_vote_compatible = False
    judge_model = "claude-haiku-4-5-20251001"

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

    def get_answer_type(self, row: dict) -> str:
        return row.get("answer_type", "exactMatch")

    def add_dataset_args(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--subset", choices=["gold", "revision", "uncertain"], default=None)
        parser.add_argument("--category", type=str, default=None)
        parser.add_argument("--text-only", action="store_true")
        super().add_dataset_args(parser)
