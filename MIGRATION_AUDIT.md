Migration Audit: tts-agent -> dr-claw/Explain
Audit date: 2026-03-25
Source: /data1/peijia/projects/EXPLaIN/
Destination: /data3/peijia/dr-claw/Explain/

---

## Overall Verdict

The core migration is structurally complete and functionally correct. All Python source files,
shell scripts (with correct new paths), paper assets, and run/cache data are present in the
destination. However, there are several specific issues documented below.

---

## Issue 1 - BLOCKER: sync_playground.sh references a non-existent path

File: /data3/peijia/dr-claw/Explain/Experiment/core_code/scripts/sync_playground.sh

The script uploads folder_path='playground' but the working directory is
/data3/peijia/dr-claw/Explain/Experiment/core_code, which has no playground/ subdirectory.
The data is now in ../analysis/ (i.e. /data3/peijia/dr-claw/Explain/Experiment/analysis/).

Source version (what it pointed to):
  folder_path='playground'
  (ran from tts-agent/ which DID have a playground/ dir)

Current destination version still uses:
  folder_path='playground'
  (runs from core_code/ which has NO playground/ dir -- it would fail or upload nothing)

The script should either point to '../analysis' or be updated to reflect the new structure.

---

## Issue 2 - HIGH: README.md contains fully stale path documentation

File: /data3/peijia/dr-claw/Explain/Experiment/core_code/README.md

The README still describes the old playground/ layout and old usage examples:

  Line 47: sync_playground.sh     # Sync playground/ to HuggingFace (see below)
  Line 66: ## Output Structure (`playground/`)
  Line 68: All outputs (caches and run results) are stored under `playground/`:
  Line 71: playground/
  Line 122: --cache-dir playground/cache/hle/opus/gold
  Line 143: --log-dir playground/run/hle/opus
  Line 147: --cache-dir playground/cache/hle/opus/gold
  Line 151: ### Sync playground to HuggingFace
  Line 153: Upload the entire `playground/` directory to HuggingFace...
  Line 156: bash scripts/sync_playground.sh
  Line 159: This syncs `playground/` to the `t2ance/tts-agent-playground` dataset repo...

All of these should refer to ../analysis/ (relative to where scripts are run from core_code/).
The README is the primary documentation for how to run experiments and is now misleading.

---

## Issue 3 - HIGH: 8 EXPLaIN root-level files were not migrated anywhere

These files exist in the source at /data1/peijia/projects/EXPLaIN/ but are absent from
ALL subdirectories of /data3/peijia/dr-claw/Explain/:

  analysis.py          (10,861 bytes, Feb 21)
  api.md               (10,435 bytes, Mar 5)
  literature.md        (22,573 bytes, Mar 5)
  minimal_image_solve.py (4,400 bytes, Feb 21)
  multimodal_input.py  (3,494 bytes, Feb 21)
  optimize_retry.py    (14,335 bytes, Feb 21)
  plan.md              (26,421 bytes, Mar 5)
  results_summary.csv  (277,992 bytes, Feb 21)

These were at the EXPLaIN repo root (not inside tts-agent/) and the migration spec defined
no destination for EXPLaIN-root files (only "other subdirs" -> code_references/).
It is unclear whether these were intentionally excluded or accidentally omitted.

The migration spec says:
  "EXPLaIN/ other subdirs (claude-relay-service, LiveCodeBench, etc.) -> Experiment/code_references/"

These are files, not subdirs -- so the spec has a gap. They are still physically present at
/data1/peijia/projects/EXPLaIN/ (not deleted from source) and were tracked in git HEAD.

---

## Issue 4 - HIGH: .claude/settings.local.json not migrated

File in source: /data1/peijia/projects/EXPLaIN/.claude/settings.local.json
Status in destination: absent from all of /data3/peijia/dr-claw/Explain/

This file contains Claude Code project-level settings (permissions, hooks, etc.) and was
tracked in git. It was not migrated to the new location. If the new project root is
/data3/peijia/dr-claw/Explain/, a .claude/settings.local.json may need to be recreated there.

---

## Issue 5 - MED: .pipeline/ directory not migrated

Source: /data1/peijia/projects/EXPLaIN/tts-agent/.pipeline/ (config.json, tasks/tasks.json, docs/)
Destination: absent from /data3/peijia/dr-claw/Explain/Experiment/core_code/

The .pipeline/ directory was committed in git HEAD. It was not copied to core_code/.
Whether this is intentional (pipeline config is environment-specific) or an omission is unclear.

---

## Issue 6 - MED: playground/.cache/huggingface (244MB upload cache) not migrated

Source: /data1/peijia/projects/EXPLaIN/tts-agent/playground/.cache/huggingface/upload/
Size: 244MB, 46,113 files
Content: HuggingFace upload cache (cache/ and run/ subdirs mirroring the actual data)
Status: still exists only at source, not present at /data3/peijia/dr-claw/Explain/

This is an upload progress cache used by huggingface_hub.upload_large_folder() to avoid
re-uploading already-transferred files. If sync_playground.sh is run again from the new
location (after fixing Issue 1), it will not find this cache and will re-upload everything.
The 244MB must be copied to the new location for efficient re-syncing, OR the sync will
restart from scratch (functionally correct but wastes time/bandwidth).

Expected destination if migrated: /data3/peijia/dr-claw/Explain/Experiment/analysis/.cache/huggingface/

---

## Issue 7 - LOW: 16 new scripts in core_code/ that don't exist in the source git commit

These scripts exist in core_code/scripts/ but were NOT in the last committed source
(tts-agent git HEAD at 321f955). They appear to have been created post-migration:

  scripts/babyvision/haiku/run_precache.sh
  scripts/babyvision/multi_model_run_effort_high.sh
  scripts/babyvision/multi_model_run_effort_low.sh
  scripts/babyvision/multi_model_run_effort_medium.sh
  scripts/babyvision/opus/run_precache.sh
  scripts/gpqa/multi_model/run_delegated_effort_high.sh
  scripts/gpqa/multi_model/run_delegated_v2.sh
  scripts/gpqa/multi_model/run_effort_high.sh
  scripts/gpqa/multi_model/run_effort_low.sh
  scripts/gpqa/multi_model/run_effort_medium.sh
  scripts/hle/multi_model/run_effort_high.sh
  scripts/hle/multi_model/run_effort_low.sh
  scripts/hle/multi_model/run_effort_medium.sh
  scripts/lcb/multi_model/run_effort_high.sh
  scripts/lcb/multi_model/run_effort_low.sh
  scripts/lcb/multi_model/run_effort_medium.sh

These are NOT a problem (they appear intentional additions in the new location), but they
confirm the source git repo was never updated to reflect these new scripts -- they exist
only in the destination.

---

## Issue 8 - LOW: Destination paper/ has new files not in source git

These files exist in /data3/peijia/dr-claw/Explain/Publication/paper/ but were not in
the source tts-agent/paper/ git commit:

  paper/figures/effort_ablation.pdf   (new figure)
  paper/figures/effort_ablation.png   (new figure)
  paper/figures/orch_ablation.pdf     (new figure)
  paper/figures/orch_ablation.png     (new figure)
  paper/ref/agentic_reasoning_2601.12538.pdf   (reference paper)
  paper/build/main.aux, .bbl, .blg, .fdb_latexmk, .fls, .log, .out  (build artifacts)

The two new figure pairs are post-migration additions. The ref/ file and build artifacts
are new structure. This is not a problem -- the destination has more than the source, which
is expected for ongoing work.

---

## Issue 9 - LOW: rbenchv benchmark has no coverage in plot_all_methods.py

The analysis/run/rbenchv/ and analysis/cache/rbenchv/ directories are fully populated, and
scripts exist for rbenchv, but scripts/plot_all_methods.py has no rbenchv benchmark entry.

This means rbenchv results are never plotted for the paper. This may be intentional (rbenchv
was excluded from the paper) but is flagged for awareness.

---

## Issue 10 - LOW: rbenchv/cache missing sonnet_visualprm_rerank subdirectory

The run_visualprm_rerank.sh script for rbenchv references cache/rbenchv/sonnet (not
sonnet_visualprm_rerank) so this is not a breakage, but the run dir
analysis/run/rbenchv/sonnet_visualprm_rerank/ exists while no corresponding dedicated
cache dir exists. Consistent with the other rerank scripts.

---

## What Is Confirmed Correct (Summary)

1. All source Python modules (backends/, benchmarks/, methods/, eval.py, logger.py,
   precache_explores.py, prompts.py, trajectory.py, pyproject.toml) are present in
   core_code/ with identical content to git HEAD.

2. All shell scripts in core_code/scripts/ correctly use:
   - cd /data3/peijia/dr-claw/Explain/Experiment/core_code  (new absolute path)
   - ../analysis/run/  (correct relative path to run data)
   - ../analysis/cache/  (correct relative path to cache data)
   No scripts contain /home/peijia or /data1/peijia references.

3. The --resume flags in scripts use relative paths pointing to ../analysis/run/ correctly.

4. All tts-agent/paper/ git-committed files are present in Publication/paper/ (plus new additions).

5. The playground run/ and cache/ data was correctly migrated to analysis/run/ and analysis/cache/.

6. All analysis/run/ subdirectories referenced by plot_all_methods.py exist.

7. CLAUDE.md has been updated to reference ../../Publication/paper/ correctly.

8. EXPLaIN other subdirs (claude-agent-sdk-python, claude-code-openai-wrapper, claude-relay-service,
   langchain-build, LiveCodeBench, meta-cognitive, metacog-tts) are present in code_references/.

9. EXPLaIN-level figures/ (9 PNG files) are present in code_references/figures/.

10. plot_all_methods.py uses PROJECT_ROOT = Path(__file__).resolve().parents[3] which correctly
    resolves to /data3/peijia/dr-claw/Explain/ from its location in core_code/scripts/.

---

## Action Items Priority List

MUST FIX:
  1. Fix sync_playground.sh: change folder_path='playground' to folder_path='../analysis'
     (relative to core_code/) or use the absolute path to analysis/.

SHOULD FIX:
  2. Update README.md: replace all playground/ path examples with ../analysis/
  3. Decide fate of 8 EXPLaIN root-level files (analysis.py, plan.md, etc.) -- migrate or discard
  4. Copy playground/.cache/huggingface/ to analysis/.cache/huggingface/ if re-sync efficiency matters

INVESTIGATE:
  5. .pipeline/ directory -- intentionally excluded or accidentally omitted?
  6. .claude/settings.local.json -- recreate at new project root if Claude Code hooks were configured
