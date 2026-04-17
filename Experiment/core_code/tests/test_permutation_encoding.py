from __future__ import annotations

import itertools
import random
import sys
from collections import Counter
from pathlib import Path

_CORE_CODE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_CORE_CODE_DIR))

from training.grpo.prepare_data_hle import (
    TRAIN_PERMUTATIONS_PER_QID,
    build_permutation_id,
    split_permutation_id,
)

N_EXPLORES = 8
ALL_PERMS = list(itertools.permutations(range(N_EXPLORES)))


def _sample_perms(qid: str, n: int) -> list[tuple[int, ...]]:
    return random.Random(f"train:{qid}").sample(ALL_PERMS, n)


def test_encode_decode_round_trip():
    perm = [5, 0, 7, 3, 1, 4, 2, 6]
    pid = build_permutation_id("hle_q0123", perm)
    assert pid == "hle_q0123#5_0_7_3_1_4_2_6"
    qid, decoded = split_permutation_id(pid)
    assert qid == "hle_q0123"
    assert decoded == perm


def test_qid_with_underscores_splits_cleanly():
    perm = [0, 1, 2, 3, 4, 5, 6, 7]
    pid = build_permutation_id("hle_q_0_1_2", perm)
    qid, decoded = split_permutation_id(pid)
    assert qid == "hle_q_0_1_2"
    assert decoded == perm


def test_5_samples_are_unique_per_qid():
    sampled = _sample_perms("hle_q0123", TRAIN_PERMUTATIONS_PER_QID)
    assert len(set(sampled)) == TRAIN_PERMUTATIONS_PER_QID, (
        f"duplicate permutations in {sampled}"
    )


def test_position0_uniform_across_800_qids():
    pos0 = Counter(
        _sample_perms(f"hle_q{i:04d}", 1)[0][0] for i in range(800)
    )
    for idx in range(N_EXPLORES):
        assert 60 <= pos0[idx] <= 140, f"pos0 bucket {idx}: {pos0[idx]}"
