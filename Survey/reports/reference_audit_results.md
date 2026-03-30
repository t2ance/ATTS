# Bibliography Audit Report

**Date**: 2026-03-29
**File**: `/data3/peijia/dr-claw/Explain/Publication/paper/main.bib`
**Purpose**: Upgrade arXiv-only citations to published venue versions for NeurIPS 2026 submission.

---

## Entries Upgraded to Published Venues

### 1. `cannot_self_correct_paper`
- **Before**: `@misc` with booktitle field (invalid BibTeX)
- **After**: `@inproceedings` at **ICLR 2024**
- Added `eprint` and `archiveprefix` fields

### 2. `deepseek_r1_paper`
- **Before**: `@misc` with arXiv eprint 2501.12948
- **After**: `@article` in **Nature**, volume 645, pages 633--638, 2025
- DOI: `10.1038/s41586-025-09422-z`

### 3. `debate_or_vote_paper`
- **Before**: `@misc` with booktitle field (invalid BibTeX)
- **After**: `@inproceedings` at **NeurIPS 2025** (Spotlight)
- Updated arXiv ID from 2412.14245 to the published version 2508.17536
- Updated author names to match published version (Hyeong Kyu Choi)

### 4. `inference_scaling_laws_paper`
- **Before**: `@misc` with booktitle field (invalid BibTeX)
- **After**: `@inproceedings` at **ICLR 2025**
- Added `eprint` and `archiveprefix` fields

### 5. `meta_reasoning_position_paper`
- **Before**: `@misc` with booktitle field (invalid BibTeX)
- **After**: `@inproceedings` at **ICML 2025**
- Updated first author name to Yan, Hanqi (verified from OpenReview/ICML)
- Added `eprint` and `archiveprefix` fields

### 6. `router_r1_paper`
- **Before**: `@misc` with booktitle field (invalid BibTeX)
- **After**: `@inproceedings` at **NeurIPS 2025**
- Added `eprint` and `archiveprefix` fields

### 7. `adaptive_consistency_paper`
- **Before**: `@misc` with booktitle field (invalid BibTeX)
- **After**: `@inproceedings` at **EMNLP 2023**, pages 12375--12396
- URL updated to ACL Anthology

### 8. `rasc_paper`
- **Before**: `@misc` with arXiv eprint 2408.17017
- **After**: `@inproceedings` at **NAACL 2025**, pages 3613--3635
- URL updated to ACL Anthology

### 9. `rational_metareasoning_llm_paper`
- **Before**: `@misc` with arXiv eprint 2410.05563
- **After**: `@inproceedings` at **NeurIPS 2024**
- Removed `primaryclass`

### 10. `adaptive_inference_time_paper`
- **Before**: `@misc` with arXiv eprint 2410.02725
- **After**: `@inproceedings` at **ICLR 2025**
- Year updated from 2024 to 2025
- Removed `primaryclass`

### 11. `correctbench_paper`
- **Before**: `@misc` with arXiv eprint 2510.16062
- **After**: `@inproceedings` at **NeurIPS 2025 (Datasets and Benchmarks Track)**
- Removed `primaryclass`

### 12. `diffadapt_paper`
- **Before**: `@misc` with arXiv eprint 2510.19669
- **After**: `@inproceedings` at **ICLR 2026**
- Year updated from 2025 to 2026
- Removed `primaryclass`

### 13. `steptool_paper`
- **Before**: `@misc` with booktitle "CIKM" (invalid BibTeX), author "Anonymous"
- **After**: `@inproceedings` at **CIKM 2025** (34th ACM International Conference on Information and Knowledge Management)
- DOI: `10.1145/3746252.3761391`
- Updated title to published version, added real author names

### 14. `catp_llm_paper`
- **Before**: `@misc` with no venue, author "Shao, Yuling"
- **After**: `@inproceedings` at **ICCV 2025**, pages 8699--8709
- Updated author names to match published version (Wu, Duo et al.)

### 15. `toolrl_paper`
- **Before**: `@misc` with author "Anonymous"
- **After**: `@inproceedings` at **NeurIPS 2025**
- Added real author names and arXiv fields

### 16. `skywork_reward_v2_paper`
- **Before**: `@misc` with arXiv eprint 2507.01352
- **After**: `@inproceedings` at **ICLR 2026**
- Removed `primaryclass`

### 17. `callaway_learning_select`
- **Before**: `@misc` with URL only
- **After**: `@inproceedings` at **UAI 2018**, pages 776--785
- Added `eprint`, `archiveprefix`, `publisher` fields

### 18. `orch_paper`
- **Before**: `@misc` with PMC URL, author "Anonymous"
- **After**: `@article` in **Frontiers in Artificial Intelligence**, volume 9, 2026
- DOI: `10.3389/frai.2026.1748735`
- Added real author names (Zhou, Hanlin and Chan, Huah Yong)

### 19. `universal_self_consistency_paper`
- **Before**: `@misc` with arXiv eprint 2311.17311
- **After**: `@inproceedings` at **ICML 2024 Workshop on In-Context Learning**
- Note: This is a workshop paper, not main ICML conference. Year updated from 2023 to 2024.
- Removed `primaryclass`

---

## Additional Corrections (not venue upgrades)

### `gsa_paper`
- Fixed wrong arXiv eprint from `2503.00000` to `2503.04104`
- Updated first author from "Li, Yufan" to "Li, Zichong" (verified via arXiv)
- Added URL field

### `kalayci_optimal_stopping_bon`
- Fixed author first names: "Sefa" -> "Yusuf", "Anirudh" -> "Vinod" (verified via arXiv)
- Added `eprint` and `archiveprefix` fields

### `consol_sprt_paper`
- Updated author names to match arXiv (Lee, Jaeyeon et al.)
- Added `eprint` and `archiveprefix` fields

### `search_r1_paper`
- Removed invalid `booktitle` field from `@misc` entry (NeurIPS 2025 acceptance could not be confirmed)
- Added `eprint` and `archiveprefix` fields

---

## Entries Remaining as arXiv Preprints

The following entries were verified to still be arXiv-only as of 2026-03-29:

| Entry Key | arXiv ID | Notes |
|---|---|---|
| `hle_verified_paper` | 2602.13964 | Feb 2026 preprint |
| `babyvision_paper` | 2601.06521 | Jan 2026 preprint |
| `visualprm_paper` | 2503.10291 | Mar 2025 preprint |
| `mcts_reasoning_paper` | 2405.00451 | May 2024 preprint |
| `rsa_paper` | 2509.26626 | Sep 2025 preprint, on OpenReview |
| `catts_webagents_paper` | 2602.12276 | Feb 2026 preprint |
| `general_agentbench_tts_paper` | 2602.18998 | Feb 2026 preprint |
| `tumix_paper` | 2510.01279 | Oct 2025 preprint, on OpenReview |
| `large_language_monkeys_paper` | 2407.21787 | Jul 2024 preprint, on OpenReview for ICLR 2025 but main conference acceptance unconfirmed |
| `adaptive_ttc_bilal_paper` | 2602.01070 | Feb 2026 preprint |
| `beacon_stopping_paper` | 2510.15945 | Oct 2025 preprint, under review at ARR |
| `gsa_paper` | 2503.04104 | Mar 2025 preprint |
| `can_llms_debate_paper` | 2511.07784 | Nov 2025 preprint |
| `dont_overthink_paper` | 2505.17813 | May 2025 preprint, on OpenReview |
| `kalayci_optimal_stopping_bon` | 2510.01394 | Oct 2025 preprint, under review at ICLR 2026 |
| `consol_sprt_paper` | 2503.17587 | Mar 2025 preprint |
| `search_r1_paper` | 2503.09516 | Mar 2025 preprint, on OpenReview |
| `optimal_bayesian_stopping_paper` | 2602.05395 | Feb 2026 preprint |
| `certified_self_consistency_paper` | 2510.17472 | Oct 2025 preprint |
| `xrouter_paper` | 2510.08439 | Oct 2025 preprint |
| `msv_paper` | 2603.03417 | Mar 2026 preprint |
| `corefine_paper` | 2602.08948 | Feb 2026 preprint |
| `meta_reasoner_paper` | 2502.19918 | Feb 2025 preprint, on OpenReview |
| `calibrate_then_act_paper` | 2602.16699 | Feb 2026 preprint |

---

## Entries Not Requiring Changes

The following entries were already correctly formatted with their published venues:

| Entry Key | Venue |
|---|---|
| `blackwell1953equivalent` | Annals of Mathematical Statistics (1953) |
| `russell1991right` | MIT Press book (1991) |
| `hle_paper_nature` | Nature (2026) |
| `livecodebench_paper` | ICLR (2025) |
| `gpqa_paper` | COLM (2024) |
| `snell_test_time_scaling_paper` | ICLR (2025) |
| `self_consistency_paper` | ICLR (2023) |
| `self_refine_paper` | NeurIPS (2023) |
| `s1_budget_forcing_paper` | EMNLP (2025) |
| `best_of_n_optimality_paper` | ICML (2025) |
| `limiting_confidence_paper` | NeurIPS (2025) |
| `tree_of_thoughts_paper` | NeurIPS (2023) |
| `math_shepherd_paper` | ACL (2024) |
| `damani_adaptive_allocation_paper` | ICLR (2025) |
| `hay_selecting_computations` | UAI (2012) |
| `best_route_paper` | ICML (2025) |
| `ab_mcts_paper` | NeurIPS (2025) |
| `bitter_lesson_paper` | Blog post (2019) |

---

## Entries Under Review (Not Published)

| Entry Key | Submission Venue | Notes |
|---|---|---|
| `tool_call_rm_paper` | ICLR 2026 | Under review, `@misc` with booktitle left as-is |

---

## Notes

- All venue information was verified via WebSearch (OpenReview, conference websites, ACL Anthology, ACM DL, Nature, Frontiers).
- No venue information was fabricated.
- `eprint` fields were preserved or added for all entries that have arXiv versions.
- `primaryclass` fields were removed when upgrading from `@misc` to `@inproceedings`/`@article`.
- Misc resource entries (HuggingFace dataset cards, leaderboards, SDK docs, model cards, MathArena) were not modified as they are not academic papers.
