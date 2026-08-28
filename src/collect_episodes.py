"""Data collection entry point.

Runs the harness across every (seed, resolver policy, condition) cell and
writes one JSON object per resolution episode to
data/processed/seed_<seed>/raw_episodes.jsonl. Every benign episode in that
file is the product of a live query against real internet DNS
infrastructure made while this script was running; nothing is replayed
from a cached corpus. Re-running this script performs a fresh round of
real queries and will not reproduce byte-identical output (answers, TTLs,
and timings depend on the live state of the internet at run time), which
is the expected and correct behavior for a measurement study of this kind.

Usage:
    python3 collect_episodes.py --reps 90 --seeds 20260814 20260815 20260816 \
        --workers 24 --out ../data/processed
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import random
import time
from pathlib import Path

import harness
import domain_pool


def _episode_to_record(ep: dict) -> dict:
    def strip(evlist):
        return evlist
    return {
        "episode_id": ep["episode_id"], "seed": ep["seed"], "policy": ep["policy"],
        "condition": ep["condition"], "label": ep["label"],
        "t_start": ep["t_start"], "t_end": ep["t_end"], "resolver_used": ep.get("resolver_used"),
        "client_queries": ep["client_queries"], "client_responses": ep["client_responses"],
        "upstream_queries": ep["upstream_queries"], "upstream_responses": ep["upstream_responses"],
    }


def collect_for_seed(seed: int, reps: int, plain_domains, ecs_domains, out_dir: Path, workers: int):
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"seed_{seed}" / "raw_episodes.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    jobs = []
    for policy in harness.POLICIES:
        for condition in harness.BENIGN_CONDITIONS + harness.ADVERSE_CONDITIONS:
            for rep in range(reps):
                jobs.append((policy, condition, rep))

    results = []
    t0 = time.time()

    def _worker(job):
        policy, condition, rep = job
        rng = random.Random(f"{seed}|{policy}|{condition}|{rep}")
        tracker = harness.OutstandingTracker()
        ep = harness.run_episode(seed, policy, condition, plain_domains, ecs_domains, rng, tracker)
        return ep

    with cf.ThreadPoolExecutor(max_workers=workers) as ex, open(out_path, "w") as fh:
        futs = [ex.submit(_worker, j) for j in jobs]
        n_done = 0
        for fut in cf.as_completed(futs):
            ep = fut.result()
            fh.write(json.dumps(_episode_to_record(ep)) + "\n")
            n_done += 1
            if n_done % 200 == 0:
                print(f"  seed={seed} {n_done}/{len(jobs)} episodes ({time.time()-t0:.1f}s elapsed)")
    print(f"seed={seed}: wrote {len(jobs)} episodes to {out_path} in {time.time()-t0:.1f}s")
    return out_path, len(jobs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=90)
    ap.add_argument("--seeds", type=int, nargs="+", default=[20260814, 20260815, 20260816])
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--out", type=str, default=str(Path(__file__).resolve().parent.parent / "data" / "processed"))
    ap.add_argument("--rebuild-domain-pool", action="store_true")
    args = ap.parse_args()

    if args.rebuild_domain_pool or not domain_pool.DOMAIN_POOL_PATH.exists():
        print("Building live domain pool (real queries against 8.8.8.8)...")
        domain_pool.build_pool()

    plain_domains, ecs_domains = harness.load_domain_pool()
    print(f"domain pool: {len(plain_domains)} live domains, {len(ecs_domains)} ECS-sensitive")

    print("Calibrating local-authority reply timing from real network round trips...")
    pool = harness.calibrate_latency_pool(n=60)
    print(f"  latency pool: n={len(pool)} median={sorted(pool)[len(pool)//2]*1000:.2f}ms "
          f"min={min(pool)*1000:.2f}ms max={max(pool)*1000:.2f}ms")

    out_dir = Path(args.out)
    total = 0
    for seed in args.seeds:
        _, n = collect_for_seed(seed, args.reps, plain_domains, ecs_domains, out_dir, args.workers)
        total += n
    print(f"TOTAL episodes collected: {total}")


if __name__ == "__main__":
    main()
