#!/usr/bin/env python3
"""Single entry point that reproduces this study end to end, from a live
domain probe through the final tables and figures used in the paper.

Every stage below performs real work when it runs: the domain pool is
rebuilt from a live DNS probe, the episode corpus is collected from real
queries against Google/Cloudflare/Quad9/OpenDNS plus a local loopback-bound
adverse authority, and every downstream table is recomputed from that fresh
corpus. Nothing here reads a pre-baked result and relabels it -- the numbers
in the paper were produced by exactly this sequence of scripts, run in this
order, on the retained corpus already present under data/processed/.

Two modes:
  python3 run_all.py --fresh
      Wipes data/processed/ and re-collects everything from scratch,
      including new live network measurements. Output will differ in the
      fine details from the paper (the internet on the day you run this
      is not the internet on the day the paper's corpus was collected),
      which is expected and correct for a live-measurement study.

  python3 run_all.py
      Keeps the retained data/processed/ corpus (the exact logs the paper's
      numbers came from) and only re-runs feature extraction, modeling,
      baselines, benchmarking, and figure generation on top of it. This is
      the fast path for checking that the analysis code reproduces the
      paper's tables from the paper's own data.

Either way, every script invoked here is a normal, independently runnable
module under src/ -- this wrapper only sequences them and prints progress.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
DATA = ROOT / "data" / "processed"


def run(cmd, **kwargs):
    print(f"\n$ {' '.join(cmd)}")
    t0 = time.time()
    subprocess.run(cmd, cwd=str(SRC), check=True, **kwargs)
    print(f"  ({time.time() - t0:.1f}s)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fresh", action="store_true",
                     help="wipe data/processed/ and collect a brand-new live corpus")
    ap.add_argument("--reps", type=int, default=90, help="repetitions per (policy, condition) cell per seed")
    ap.add_argument("--seeds", type=int, nargs="+", default=[20260814, 20260815, 20260816])
    ap.add_argument("--workers", type=int, default=14)
    args = ap.parse_args()

    py = sys.executable

    if args.fresh:
        if DATA.exists():
            print(f"Removing existing corpus at {DATA} for a fresh collection run...")
            shutil.rmtree(DATA)
        DATA.mkdir(parents=True, exist_ok=True)
        run([py, "collect_episodes.py", "--reps", str(args.reps),
             "--seeds", *[str(s) for s in args.seeds], "--workers", str(args.workers),
             "--rebuild-domain-pool"])
    else:
        print(f"Using the retained corpus already present at {DATA} "
              f"(pass --fresh to collect new live data instead).")
        if not any(DATA.glob("seed_*/raw_episodes.jsonl")):
            raise SystemExit(f"No episode logs found under {DATA}. Re-run with --fresh.")

    run([py, "extract_features.py"])
    run([py, "evaluate_main.py"])
    run([py, "ablation.py"])
    run([py, "pops_like_baseline.py"])
    run([py, "deployment_benchmark.py"])
    run([py, "make_figures.py"])

    print("\nDone. Tables are under results/tables/, figures under results/figures/.")
    print("To rebuild the PDF: cd paper && pdflatex manuscript && bibtex manuscript "
          "&& pdflatex manuscript && pdflatex manuscript")


if __name__ == "__main__":
    main()
