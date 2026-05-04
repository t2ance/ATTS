"""Find "elimination synthesis" trajectories: orchestrator deduced the correct answer
WITHOUT any explorer providing it.

Definition (clean, benchmark-agnostic):
  - The orchestrator saw explorers e_1..e_{t*} during the run.
  - Among those t* explorers, ZERO answered correctly (sum(is_correct)=0).
  - Yet the orchestrator's final answer is correct.

For GPQA, we additionally check letter-level: the orchestrator's chosen letter
was NEVER suggested by any of the seen explorers (e.g. all explorers said A/B/C
but the orchestrator picked D, the correct option).

Why this matters: this is the strongest C3 evidence — orchestrator is doing
quality reasoning that goes beyond synthesizing from the explorer pool.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from benchmarks.grader import _extract_mc_letter, normalize_answer  # noqa: E402

ANOMALOUS_GPQA = {"recK9F5aqdaybl8bb", "recRgabRzMaEoBRcM", "recZ13cwgDQf9jRd9"}

BENCH_RUNS = {
    "gpqa": [Path(f"/data3/peijia/dr-claw/Explain/Experiment/analysis/run/gpqa/sonnet_no_integrate_run{i}/run_20260328_060344/results.jsonl") for i in range(1, 6)],
    "hle": [Path("/data3/peijia/dr-claw/Explain/Experiment/analysis/run/hle/sonnet_no_integrate/run_20260319_003712/results.jsonl")],
    "lcb": [Path("/data3/peijia/dr-claw/Explain/Experiment/analysis/run/lcb/sonnet/run_20260308_230222/results.jsonl")],
    "babyvision": [Path("/data3/peijia/dr-claw/Explain/Experiment/analysis/run/babyvision/sonnet_no_integrate/run_20260319_021914/results.jsonl")],
}


def load_jsonl(path: Path):
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def to_letter(text):
    if text is None:
        return None
    return _extract_mc_letter(normalize_answer(str(text)))


def analyze(bench: str, paths) -> dict:
    cohort = {"total": 0, "no_correct_seen_AND_final_correct": [], "no_correct_seen_AND_final_wrong": 0,
              "letter_not_in_seen_AND_correct_GPQA_only": []}
    for p in paths:
        for r in load_jsonl(p):
            if bench == "gpqa" and r["id"] in ANOMALOUS_GPQA:
                continue
            ec = r.get("explore_candidates") or []
            t_star = r["num_explores"]
            if t_star < 1 or t_star > len(ec):
                continue
            cohort["total"] += 1
            seen = ec[:t_star]
            n_correct_seen = sum(1 for c in seen if c.get("is_correct"))
            final_correct = bool(r["is_correct"])
            if n_correct_seen == 0:
                if final_correct:
                    case = {
                        "run": p.parent.parent.name,
                        "qid": r["id"],
                        "t_star": t_star,
                        "n_explore_candidates_total": len(ec),
                        "seen_normalized_answers": [c.get("normalized_answer") for c in seen],
                        "predicted_answer": r["predicted_answer"],
                        "gold_answer": r["gold_answer"],
                    }
                    if bench == "gpqa":
                        seen_letters = set(filter(None, (to_letter(c.get("normalized_answer")) for c in seen)))
                        final_letter = to_letter(r["predicted_answer"])
                        gold_letter = to_letter(r["gold_answer"])
                        case["seen_letters"] = sorted(seen_letters)
                        case["final_letter"] = final_letter
                        case["gold_letter"] = gold_letter
                        if final_letter not in seen_letters:
                            cohort["letter_not_in_seen_AND_correct_GPQA_only"].append(case)
                    cohort["no_correct_seen_AND_final_correct"].append(case)
                else:
                    cohort["no_correct_seen_AND_final_wrong"] += 1
    return cohort


def main():
    out_path = Path("/data3/peijia/dr-claw/Explain/Experiment/analysis/orch_evidence/elimination_synthesis.json")
    summary = {}
    for bench, paths in BENCH_RUNS.items():
        c = analyze(bench, paths)
        n_total = c["total"]
        n_elim = len(c["no_correct_seen_AND_final_correct"])
        n_dead = c["no_correct_seen_AND_final_wrong"]
        n_letter_not_in_seen = len(c["letter_not_in_seen_AND_correct_GPQA_only"]) if bench == "gpqa" else None
        print(f"\n=== {bench} ===")
        print(f"  total trajectories: {n_total}")
        print(f"  no-correct-explorer cohort: {n_elim + n_dead}")
        print(f"    of which orchestrator answered correctly: {n_elim} ({n_elim / n_total * 100:.1f}% of all)")
        print(f"    of which orchestrator answered wrong: {n_dead}")
        if bench == "gpqa":
            print(f"  GPQA letter-not-in-seen (strongest form, picked unsuggested option): {n_letter_not_in_seen}")
        summary[bench] = {
            "total_trajectories": n_total,
            "n_no_correct_seen_total": n_elim + n_dead,
            "n_elimination_synthesis": n_elim,
            "n_no_correct_seen_final_wrong": n_dead,
            "elimination_rate_overall_pct": round(n_elim / n_total * 100, 2),
            "elimination_rate_within_no_correct_seen_pct": round(n_elim / (n_elim + n_dead) * 100, 2) if (n_elim + n_dead) > 0 else None,
            "n_letter_not_in_seen_GPQA_only": n_letter_not_in_seen,
            "sample_cases": c["no_correct_seen_AND_final_correct"][:5],
        }
        if bench == "gpqa":
            summary[bench]["letter_not_in_seen_cases_sample"] = c["letter_not_in_seen_AND_correct_GPQA_only"][:5]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2, default=str))
    print(f"\n\nWrote {out_path}")

    # Print sample GPQA cases for inspection
    gpqa_cases = summary["gpqa"]["letter_not_in_seen_cases_sample"]
    if gpqa_cases:
        print(f"\n=== Sample GPQA cases where orchestrator picked an UNSUGGESTED letter (and it was correct) ===")
        for case in gpqa_cases:
            print(f"\n  qid={case['qid']} run={case['run']}: t*={case['t_star']}, seen letters={case['seen_letters']}, picked={case['final_letter']} (gold={case['gold_letter']})")
            for i, ans in enumerate(case["seen_normalized_answers"], 1):
                a = (ans or "")[:50]
                print(f"    explore_{i}: {a}")
            print(f"    final: {(case['predicted_answer'] or '')[:50]}")


if __name__ == "__main__":
    main()
