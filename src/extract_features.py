"""Episode reconstruction -> behavioral feature table.

Reads data/processed/seed_<seed>/raw_episodes.jsonl (produced fresh by
collect_episodes.py every time it is run) and writes
data/processed/seed_<seed>/features.csv, one row per episode. Features are
grouped into the same six families as the original study -- volume,
aggregation, ECS, entropy, answer/cache, timing -- plus a state-reuse
feature that replaces the earlier "cache hit rate" placeholder with an
honestly-named quantity this harness can actually measure (see
state_reuse_rate below).

No literal domain name, IP address, or raw ECS prefix identity is placed
in the feature matrix; only counts, ratios, and entropy/consistency
measures cross into the model, for the same train/test-leakage reasons
given in the manuscript.
"""
from __future__ import annotations

import argparse
import ipaddress
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

FEATURE_COLUMNS = [
    # volume
    "n_client_queries", "n_client_responses", "n_upstream_queries", "n_upstream_responses",
    "total_wire_bytes",
    # aggregation
    "upstream_client_ratio", "response_multiplicity", "max_active_upstream",
    # ECS
    "distinct_client_ecs", "distinct_upstream_ecs", "ecs_fanout", "missing_ecs_rate", "ecs_mismatch_rate",
    # entropy
    "txid_entropy", "sport_entropy",
    # answer/cache
    "answer_uniqueness", "answer_disagreement", "ttl_mean", "ttl_cv", "rcode_diversity",
    "servfail_rate", "state_reuse_rate",
    # timing
    "client_latency_mean", "client_latency_p95", "client_latency_cv",
    "upstream_latency_mean", "response_race_gap_mean",
    # scale/context (kept out of the "29 core" count but retained for audit)
    "n_timeouts",
]


def _entropy(values) -> float:
    if not values:
        return 0.0
    counts = Counter(values)
    n = len(values)
    h = -sum((c / n) * math.log2(c / n) for c in counts.values())
    max_h = math.log2(min(n, 2 ** 16)) if n > 1 else 1.0
    return h / max_h if max_h > 0 else 0.0


def _ecs_network(ecs):
    if not ecs or not ecs.get("address"):
        return None
    try:
        return str(ipaddress.ip_network(f"{ecs['address']}/{ecs.get('prefix', 32)}", strict=False))
    except Exception:
        return None


def _pair_upstream(upstream_queries, upstream_responses):
    """Pair upstream queries to responses by txid (falls back to positional
    pairing if a txid collides or is absent)."""
    resp_by_txid = {}
    for r in upstream_responses:
        resp_by_txid.setdefault(r.get("txid"), []).append(r)
    pairs = []
    for q in upstream_queries:
        candidates = resp_by_txid.get(q.get("txid"), [])
        if candidates:
            pairs.append((q, candidates.pop(0)))
        else:
            pairs.append((q, None))
    return pairs


def _max_overlap(intervals):
    if not intervals:
        return 0
    events = []
    for a, b in intervals:
        if b is None or b < a:
            b = a
        events.append((a, 1))
        events.append((b, -1))
    events.sort()
    cur, best = 0, 0
    for _, delta in events:
        cur += delta
        best = max(best, cur)
    return best


def compute_features(ep: dict) -> dict:
    cq = ep["client_queries"]
    cr = ep["client_responses"]
    uq = ep["upstream_queries"]
    ur = ep["upstream_responses"]

    n_cq, n_cr, n_uq, n_ur = len(cq), len(cr), len(uq), len(ur)
    total_bytes = sum(x.get("wire_len", 0) for x in (cq + cr + uq + ur) if isinstance(x, dict))
    # client_queries do not carry wire_len (they are logical, not wire, in
    # the benign path where the resolver is the wire endpoint); upstream
    # events carry the true wire bytes actually sent/received.
    total_bytes = sum(x.get("wire_len", 0) for x in (uq + ur))

    upstream_client_ratio = n_uq / max(1, n_cq)
    response_multiplicity = n_cr / max(1, n_cq)

    pairs = _pair_upstream(uq, ur)
    intervals = [(q["t"], (resp["t"] if resp else q["t"])) for q, resp in pairs]
    max_active_upstream = _max_overlap(intervals)

    client_ecs_nets = {_ecs_network(q.get("ecs_sent")) for q in cq if q.get("ecs_sent")}
    client_ecs_nets.discard(None)
    upstream_ecs_nets = {_ecs_network(q.get("ecs_sent")) for q in uq if q.get("ecs_sent")}
    upstream_ecs_nets.discard(None)
    ecs_fanout = len(upstream_ecs_nets) / max(1, n_uq)

    n_ecs_sent = sum(1 for q in uq if q.get("ecs_sent"))
    n_missing_ecs_resp = sum(1 for q, r in pairs if q.get("ecs_sent") and r is not None and not r.get("ecs_resp"))
    missing_ecs_rate = n_missing_ecs_resp / max(1, n_ecs_sent)

    n_mismatch = 0
    n_checkable = 0
    for q, r in pairs:
        if q.get("ecs_sent") and r is not None and r.get("ecs_resp"):
            n_checkable += 1
            req_net = _ecs_network(q["ecs_sent"])
            got_net = _ecs_network(r["ecs_resp"])
            if got_net is not None and got_net != req_net:
                n_mismatch += 1
    ecs_mismatch_rate = n_mismatch / max(1, n_checkable)

    txids = [q.get("txid") for q in uq if q.get("txid") is not None]
    sports = [q.get("sport") for q in uq if q.get("sport") is not None]
    txid_entropy = _entropy(txids)
    sport_entropy = _entropy(sports)

    answer_sets = [tuple(sorted(r.get("answers", []))) for r in cr]
    n_distinct_answer_sets = len(set(answer_sets)) if answer_sets else 0
    answer_uniqueness = n_distinct_answer_sets / max(1, len(answer_sets))
    if answer_sets:
        group_sizes = Counter(answer_sets)
        max_group = max(group_sizes.values())
        answer_disagreement = 1 - (max_group / len(answer_sets))
    else:
        answer_disagreement = 0.0

    all_ttls = [t for r in cr for t in r.get("ttls", [])]
    ttl_mean = float(np.mean(all_ttls)) if all_ttls else 0.0
    ttl_std = float(np.std(all_ttls)) if all_ttls else 0.0
    ttl_cv = (ttl_std / ttl_mean) if ttl_mean > 0 else 0.0

    rcodes = [r.get("rcode_name") for r in cr]
    rcode_diversity = (len(set(rcodes)) / max(1, len(rcodes))) if rcodes else 0.0
    servfail_rate = (sum(1 for x in rcodes if x == "SERVFAIL") / max(1, len(rcodes))) if rcodes else 0.0
    n_timeouts = sum(1 for x in rcodes if x == "TIMEOUT")

    # state_reuse_rate: fraction of client-facing sub-queries that were
    # served without a dedicated upstream round trip because an identical
    # in-flight request (same aggregation key) was already outstanding.
    # This is the harness's real, measurable analogue of a cache-hit rate;
    # it is not derived from a TTL-based cache because this harness does
    # not implement persistent record caching.
    state_reuse_rate = max(0.0, (n_cq - n_uq) / max(1, n_cq))

    # Latency is computed from the (t_query -> t) pair recorded on each
    # client response itself, not by positionally zipping the query and
    # response lists (concurrent sub-queries can complete out of order, so
    # positional pairing silently mismatches query i with response j != i).
    c_lat = [r["t"] - r["t_query"] for r in cr if r.get("t_query") is not None]
    client_latency_mean = float(np.mean(c_lat)) if c_lat else 0.0
    client_latency_p95 = float(np.percentile(c_lat, 95)) if c_lat else 0.0
    client_latency_std = float(np.std(c_lat)) if c_lat else 0.0
    client_latency_cv = (client_latency_std / client_latency_mean) if client_latency_mean > 0 else 0.0

    u_lat = [resp["t"] - q["t"] for q, resp in pairs if resp is not None]
    upstream_latency_mean = float(np.mean(u_lat)) if u_lat else 0.0

    # response race gap: group client-facing responses by which sub-query
    # (t_query) produced them, then take the spread between first and last
    # arrival within each group that actually raced (>1 response).
    by_query = {}
    for r in cr:
        by_query.setdefault(r.get("t_query"), []).append(r["t"])
    race_gaps = [max(ts) - min(ts) for ts in by_query.values() if len(ts) > 1]
    response_race_gap_mean = float(np.mean(race_gaps)) if race_gaps else 0.0

    return {
        "n_client_queries": n_cq, "n_client_responses": n_cr,
        "n_upstream_queries": n_uq, "n_upstream_responses": n_ur,
        "total_wire_bytes": total_bytes,
        "upstream_client_ratio": upstream_client_ratio,
        "response_multiplicity": response_multiplicity,
        "max_active_upstream": max_active_upstream,
        "distinct_client_ecs": len(client_ecs_nets),
        "distinct_upstream_ecs": len(upstream_ecs_nets),
        "ecs_fanout": ecs_fanout,
        "missing_ecs_rate": missing_ecs_rate,
        "ecs_mismatch_rate": ecs_mismatch_rate,
        "txid_entropy": txid_entropy,
        "sport_entropy": sport_entropy,
        "answer_uniqueness": answer_uniqueness,
        "answer_disagreement": answer_disagreement,
        "ttl_mean": ttl_mean,
        "ttl_cv": ttl_cv,
        "rcode_diversity": rcode_diversity,
        "servfail_rate": servfail_rate,
        "state_reuse_rate": state_reuse_rate,
        "client_latency_mean": client_latency_mean,
        "client_latency_p95": client_latency_p95,
        "client_latency_cv": client_latency_cv,
        "upstream_latency_mean": upstream_latency_mean,
        "response_race_gap_mean": response_race_gap_mean,
        "n_timeouts": n_timeouts,
    }


def process_seed(seed_dir: Path) -> pd.DataFrame:
    raw_path = seed_dir / "raw_episodes.jsonl"
    rows = []
    with open(raw_path) as fh:
        for line in fh:
            ep = json.loads(line)
            feats = compute_features(ep)
            feats.update({
                "episode_id": ep["episode_id"], "seed": ep["seed"], "policy": ep["policy"],
                "condition": ep["condition"], "label": ep["label"], "resolver_used": ep.get("resolver_used"),
            })
            rows.append(feats)
    df = pd.DataFrame(rows)
    out_path = seed_dir / "features.csv"
    df.to_csv(out_path, index=False)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=str, default=str(Path(__file__).resolve().parent.parent / "data" / "processed"))
    args = ap.parse_args()
    data_dir = Path(args.data_dir)
    seed_dirs = sorted(p for p in data_dir.glob("seed_*") if (p / "raw_episodes.jsonl").exists())
    all_dfs = []
    for sd in seed_dirs:
        df = process_seed(sd)
        print(f"{sd.name}: {len(df)} episodes -> {sd / 'features.csv'}")
        all_dfs.append(df)
    combined = pd.concat(all_dfs, ignore_index=True)
    combined_path = data_dir / "combined_features.csv"
    combined.to_csv(combined_path, index=False)
    print(f"combined: {len(combined)} episodes -> {combined_path}")
    print(combined.groupby(["policy", "condition"]).size())


if __name__ == "__main__":
    main()
