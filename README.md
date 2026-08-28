# StateDNS: Stateful EDNS/ECS Behavioral Fingerprints for DNS Cache-Poisoning Detection

This is the complete package for the paper *"Stateful EDNS/ECS Behavioral
Fingerprints for DNS Cache-Poisoning Detection: Transfer Across Resolver
Policies Using Live Internet Measurement,"* prepared in the style of
Elsevier's *Computers & Security*.

The central design choice of this study: Every benign episode in the corpus is a live
DNS query resolved, at collection time, against real production
infrastructure -- Google Public DNS, Cloudflare, Quad9, and OpenDNS -- for
domain names drawn from two independently sourced, currently-live domain
rankings, with every candidate name re-validated by a live probe before
use. The adverse half is produced by a local, loopback-bound authority
that never sends anything to the public internet, with its reply timing
resampled continuously from the live latency distribution measured on the
benign side of the same run, so that raw network speed cannot become a
shortcut a classifier learns instead of the intended protocol-state
signal. `paper/manuscript.tex` documents this design, the reasons behind
it, and everything it does not claim, in full.

## Layout

- `paper/` -- `manuscript.tex`, `references.bib`, the compiled
  `manuscript.pdf`, and the two Elsevier class/style files
  (`elsarticle.cls`, `elsarticle-num.bst`) needed to rebuild it without a
  network connection.
- `src/` -- the full pipeline: DNS wire-format encode/decode, the live
  domain-pool builder, the data-collection harness, feature extraction,
  model training and evaluation, baseline re-implementations, deployment
  benchmarking, and figure generation. Every stage is an independently
  runnable script; `run_all.py` at the repository root sequences all of
  them.
- `data/raw/domain_lists/` -- the two source domain lists and a
  `PROVENANCE.md` explaining exactly how each is used and why an older
  source list does not compromise data currency (see below).
- `data/processed/` -- the retained corpus this paper's numbers were
  computed from: one `raw_episodes.jsonl` per seed (the full wire-level
  event log of every episode), the built domain pool, and
  `combined_features.csv` (the 27-feature table derived from those logs).
- `results/tables/` -- every CSV/JSON table referenced in the paper,
  organized by stage (`model_eval/`, `baselines/`, `deployment/`).
- `results/figures/` -- all nine figures as vector PDFs.

## Reproducing the analysis on the retained corpus

```
pip install -r requirements.txt
python3 run_all.py
```

This re-runs feature extraction, model training/evaluation (random split,
leave-one-resolver-policy-out, leave-one-adverse-condition-out, bootstrap
confidence intervals, DeLong and McNemar significance tests), the
ablation study, both baseline re-implementations, the deployment
benchmark, and figure generation directly on the `raw_episodes.jsonl`
files already in `data/processed/`. It reads nothing that was
pre-computed further downstream than that raw log, so this checks the
same thing an independent verifier would want checked: that the tables
and figures in the paper follow from the retained event-level data by the
documented method.

## Collecting a brand-new live corpus

```
python3 run_all.py --fresh
```

This wipes the retained corpus and rebuilds everything from scratch,
starting with a fresh live probe of the domain pool and a fresh round of
real queries against the four public resolvers above. Numbers will not be
byte-identical to the paper -- answers, TTLs, and round-trip timing depend
on the live state of the internet on the day this runs -- which is the
expected behavior of a measurement study built on live data rather than a
frozen snapshot. Each stage can also be invoked directly; see the
docstring at the top of `run_all.py` and of each script under `src/` for
the exact commands and options (worker count, repetitions per condition,
seeds).

Two things worth knowing before running `--fresh`: it needs outbound UDP
port 53 to reach the public internet (ordinary DNS traffic; nothing else
is required), and it takes on the order of minutes for a few thousand
episodes on ordinary hardware -- `deployment_summary.json` under
`results/tables/deployment/` reports the throughput actually measured on
the machine this paper's corpus was built on.



## On the domain lists

`data/raw/domain_lists/PROVENANCE.md` explains this in full, but briefly:
one list is a current, actively-maintained mirror of the Moz Top 500; the
other is an older ranking used purely to widen the pool of candidate
names, with every candidate from it subjected to a live resolution check
at collection time. What ends up in the corpus is today's real answer
from today's real infrastructure for a name that happened to come from
one list or the other -- the age of the ranking a name was drawn from has
no bearing on the freshness of the traffic measured for it.

