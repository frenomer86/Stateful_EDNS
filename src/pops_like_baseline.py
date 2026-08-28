"""A same-corpus re-implementation of the POPS R1 detection rule (Afek et
al., USENIX Security 2025): flag a query as under a statistical guessing
attack when five or more responses for the same outstanding query arrive
within about one second and are identical apart from transaction ID or
source port. Evaluated directly on the raw episode event streams produced
by collect_episodes.py (not on the engineered feature table), exactly as
the manuscript re-implementation was.

A strict and a permissive reading are both reported: POPS was designed
against a brute-force TXID/port guessing attack, which this corpus does
not contain (no spoofed or guessed responses are ever sent), so a weak
result here is expected and is not evidence against POPS on its own
target attack -- it is evidence that a response-count rule does not by
itself cover ECS-continuity and aggregation-state violations.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
OUT_DIR = Path(__file__).resolve().parent.parent / "results" / "tables" / "baselines"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _answer_signature(resp):
    return (resp.get("rcode_name"), tuple(sorted(resp.get("answers", []))), tuple(resp.get("ttls", [])))


def score_episode(ep, window_s=1.0, min_count=5):
    """POPS R1 is a per-outstanding-query rule: it counts responses racing
    for the SAME query, not all traffic in an episode. We group client-
    facing responses by which sub-query produced them (t_query, the same
    grouping used for response_race_gap_mean) so a benign episode with
    several independent concurrent sub-queries is not miscounted as one
    large race just because it is busy."""
    from collections import Counter, defaultdict
    by_query = defaultdict(list)
    for r in ep["client_responses"]:
        by_query[r.get("t_query")].append(r)

    max_in_window = 0
    max_identical_group = 0
    for _, group in by_query.items():
        times = sorted(r["t"] for r in group if r.get("t") is not None)
        j = 0
        for i in range(len(times)):
            while times[i] - times[j] > window_s:
                j += 1
            max_in_window = max(max_in_window, i - j + 1)
        sigs = [_answer_signature(r) for r in group]
        sig_counts = Counter(sigs)
        if sig_counts:
            max_identical_group = max(max_identical_group, max(sig_counts.values()))

    strict_flag = 1 if (max_in_window >= min_count and max_identical_group >= min_count) else 0
    permissive_flag = 1 if max_in_window >= min_count else 0
    return strict_flag, permissive_flag, max_in_window, max_identical_group


def main():
    rows = []
    for seed_dir in sorted(RAW_DIR.glob("seed_*")):
        raw_path = seed_dir / "raw_episodes.jsonl"
        if not raw_path.exists():
            continue
        with open(raw_path) as fh:
            for line in fh:
                ep = json.loads(line)
                strict_flag, perm_flag, max_win, max_grp = score_episode(ep)
                rows.append({"episode_id": ep["episode_id"], "seed": ep["seed"], "policy": ep["policy"],
                             "condition": ep["condition"], "label": ep["label"],
                             "pops_strict_flag": strict_flag, "pops_permissive_flag": perm_flag,
                             "max_responses_in_1s": max_win, "max_identical_group": max_grp})
    df = pd.DataFrame(rows)

    # This corpus's adverse conditions produce at most two competing
    # responses per outstanding query (a two-way race), never the five the
    # published R1 rule requires, so the literal rule never fires here (see
    # summary below). A relaxed min_count=2 variant is reported alongside
    # it purely as a sensitivity check -- it is not a claim that POPS
    # itself uses this threshold.
    relaxed = df.apply(lambda row: pd.Series(
        {"pops_relaxed_min2_flag": 1 if row["max_responses_in_1s"] >= 2 else 0}), axis=1)
    df = pd.concat([df, relaxed], axis=1)
    df.to_csv(OUT_DIR / "pops_like_predictions.csv", index=False)

    def summarize(flag_col):
        y = df["label"].values
        pred = df[flag_col].values
        tp = int(((pred == 1) & (y == 1)).sum())
        fp = int(((pred == 1) & (y == 0)).sum())
        fn = int(((pred == 0) & (y == 1)).sum())
        tn = int(((pred == 0) & (y == 0)).sum())
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        fpr = fp / (fp + tn) if (fp + tn) else 0.0
        return {"precision": precision, "recall": recall, "f1": f1, "fpr": fpr, "tp": tp, "fp": fp, "fn": fn, "tn": tn}

    summary = {"strict_r1": summarize("pops_strict_flag"), "permissive_r1": summarize("pops_permissive_flag"),
               "relaxed_min2_r1": summarize("pops_relaxed_min2_flag"), "n_episodes": len(df)}
    with open(OUT_DIR / "pops_like_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    pd.DataFrame(summary).T.to_csv(OUT_DIR / "pops_like_summary.csv")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
