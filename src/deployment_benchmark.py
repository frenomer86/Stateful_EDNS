"""Runtime cost measurements, taken on whatever hardware this script
happens to run on -- reported as an environment-specific measurement, not a
production-capacity guarantee, following the same discipline the original
study used. Everything here is timed directly; nothing is a modeled or
assumed number."""
from __future__ import annotations

import json
import os
import resource
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

import modeling_common as mc
from dns_wire import build_query, build_response, parse_message
from extract_features import compute_features
from harness import OutstandingTracker

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
OUT_DIR = Path(__file__).resolve().parent.parent / "results" / "tables" / "deployment"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def bench_wire_roundtrip(n=50000):
    qnames = ["example.com", "www.google.com", "whatsapp.com", "cdn.example.net", "api.service.io"]
    answers_pool = [["93.184.216.34"], ["142.251.155.119", "142.251.154.119"], ["57.144.163.32"]]
    packets = []
    for i in range(n):
        q = qnames[i % len(qnames)]
        pkt, txid = build_query(q, "A", ecs=(1, 24, "8.8.8.0") if i % 3 == 0 else None)
        resp = build_response(q, "A", txid, answers_pool[i % len(answers_pool)], ttl=300)
        packets.append((pkt, resp))

    t0 = time.perf_counter()
    for pkt, _ in packets:
        parse_message(pkt if False else pkt)  # query parsing is symmetric wire work; parse the built query bytes
    t_query_parse = time.perf_counter() - t0

    t0 = time.perf_counter()
    for _, resp in packets:
        parse_message(resp)
    t_resp_parse = time.perf_counter() - t0

    return {"n_packets": n,
            "query_parse_rate_per_s": n / t_query_parse,
            "response_parse_rate_per_s": n / t_resp_parse}


def bench_feature_extraction():
    seed_dirs = sorted(DATA_DIR.glob("seed_*"))
    total_episodes = 0
    total_time = 0.0
    import json as _json
    for sd in seed_dirs:
        raw_path = sd / "raw_episodes.jsonl"
        if not raw_path.exists():
            continue
        eps = [_json.loads(l) for l in open(raw_path)]
        t0 = time.perf_counter()
        for ep in eps:
            compute_features(ep)
        dt = time.perf_counter() - t0
        total_episodes += len(eps)
        total_time += dt
    return {"n_episodes": total_episodes, "total_s": total_time,
            "episodes_per_s": total_episodes / total_time if total_time > 0 else float("nan")}


def bench_inference_latency(n_trials=2000):
    df = pd.read_csv(DATA_DIR / "combined_features.csv")
    scaler = StandardScaler().fit(df[mc.ALL_FEATURES].values)
    clf = LogisticRegression(max_iter=2000).fit(scaler.transform(df[mc.ALL_FEATURES].values), df["label"].values)
    rows = df[mc.ALL_FEATURES].values
    rng = np.random.RandomState(0)
    idx = rng.randint(0, len(rows), size=n_trials)
    latencies = []
    for i in idx:
        x = rows[i:i + 1]
        t0 = time.perf_counter()
        xs = scaler.transform(x)
        clf.predict_proba(xs)
        latencies.append(time.perf_counter() - t0)
    latencies = np.array(latencies) * 1000  # ms
    return {"n_trials": n_trials, "mean_ms": float(np.mean(latencies)),
            "p50_ms": float(np.percentile(latencies, 50)), "p95_ms": float(np.percentile(latencies, 95)),
            "p99_ms": float(np.percentile(latencies, 99))}


def bench_outstanding_state_memory(n_entries=200000):
    def rss_kb():
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    before = rss_kb()
    tracker = OutstandingTracker()
    keys = []
    for i in range(n_entries):
        key = (f"name{i}.example", "A", f"10.{(i>>16)&255}.{(i>>8)&255}.{i&255}/24")
        entry, is_leader = tracker.get_or_start(key)
        keys.append((key, entry))
    after = rss_kb()
    # RSS units differ by platform (KB on Linux); ru_maxrss is peak, so this
    # is a conservative (upper-bound-ish) delta, not a live-heap-only delta.
    delta_kb = after - before
    for key, entry in keys:
        tracker.finish(key, entry, ([], []))
    return {"n_entries": n_entries, "rss_delta_kb": delta_kb, "rss_delta_mib": delta_kb / 1024.0,
            "bytes_per_entry_estimate": (delta_kb * 1024) / n_entries if n_entries else None}


def bench_env():
    return {"cpu_count": os.cpu_count(), "platform": os.uname()._asdict() if hasattr(os.uname(), "_asdict") else str(os.uname())}


def main():
    results = {}
    print("Benchmarking wire parse throughput...")
    results["wire_roundtrip"] = bench_wire_roundtrip()
    print("Benchmarking feature extraction throughput on the full retained corpus...")
    results["feature_extraction"] = bench_feature_extraction()
    print("Benchmarking single-episode inference latency...")
    results["inference_latency"] = bench_inference_latency()
    print("Benchmarking outstanding-state memory footprint...")
    results["outstanding_state_memory"] = bench_outstanding_state_memory()
    results["environment"] = bench_env()
    results["note"] = ("All figures were measured on the single cloud instance this study's "
                        "pipeline was executed on; they describe this environment, not a production "
                        "deployment, and are not extrapolated to other hardware.")

    with open(OUT_DIR / "deployment_summary.json", "w") as fh:
        json.dump(results, fh, indent=2, default=str)
    rows = []
    for k, v in results.items():
        if isinstance(v, dict):
            for kk, vv in v.items():
                rows.append({"benchmark": k, "metric": kk, "value": vv})
    pd.DataFrame(rows).to_csv(OUT_DIR / "deployment_summary.csv", index=False)
    print(json.dumps(results, indent=2, default=str))


if __name__ == "__main__":
    main()
