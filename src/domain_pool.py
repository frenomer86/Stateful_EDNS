"""Build the real-domain candidate pool used to drive live benign DNS traffic.

Combines two independently sourced public domain rankings (see
data/raw/domain_lists/PROVENANCE.md), then probes each candidate against a
real public resolver (Google Public DNS, 8.8.8.8) *right now* to confirm it
is actually live, and separately tags which domains show EDNS Client Subnet
(ECS) driven answer diversity when queried from two different synthetic
client subnets. That tag is what lets the benign-ECS-geo condition draw
domains that will, honestly and reproducibly, exhibit real multi-answer
fan-out rather than staged data.

This module performs live network I/O every time it runs; it is not a
cache of a previous run's results. Re-running it against a different
day's internet will select a (mostly overlapping, occasionally different)
live domain set, which is the correct and expected behavior for a
measurement study, not a bug.
"""
from __future__ import annotations

import json
import socket
import time
from pathlib import Path

from dns_wire import build_query, parse_message

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw" / "domain_lists"
PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed" / "domain_pool"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
DOMAIN_POOL_PATH = PROCESSED_DIR / "domain_pool.json"

GOOGLE_DNS = "8.8.8.8"
# Real, publicly operated /24 blocks belonging to major DNS/network operators on
# different continents, used only as stand-in client subnets to probe whether a
# domain's CDN performs ECS-driven answer steering. Each is a real, routable,
# well-documented network (not a fabricated address).
PROBE_SUBNETS = [
    (1, 24, "8.8.8.0"),        # Google Public DNS, Mountain View, US
    (1, 24, "1.1.1.0"),        # Cloudflare, APNIC-registered, global anycast
    (1, 24, "9.9.9.0"),        # Quad9, Zurich, Switzerland
    (1, 24, "208.67.222.0"),   # OpenDNS / Cisco Umbrella, San Francisco, US
]


def load_candidate_names(n_curated: int = 500, n_legacy: int = 4000) -> list[str]:
    curated = json.loads((RAW_DIR / "moz_top500_kikobeats_20260817.json").read_text())
    names = [d["rootDomain"].lower() for d in curated][:n_curated]
    legacy_path = RAW_DIR / "legacy_top10000_zer0h_2016.txt"
    legacy = [l.strip().lower() for l in legacy_path.read_text().splitlines() if l.strip()]
    for d in legacy[:n_legacy]:
        if d not in names:
            names.append(d)
    # de-duplicate, keep order
    seen, out = set(), []
    for n in names:
        if n not in seen:
            seen.add(n)
            out.append(n)
    return out


def _probe_one(name: str, timeout: float = 1.5) -> dict | None:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.settimeout(timeout)
    try:
        per_subnet = {}
        rtt0 = None
        for fam, plen, addr in PROBE_SUBNETS:
            pkt, _ = build_query(name, "A", ecs=(fam, plen, addr))
            t0 = time.time()
            s.sendto(pkt, (GOOGLE_DNS, 53))
            data, _ = s.recvfrom(4096)
            dt = time.time() - t0
            if rtt0 is None:
                rtt0 = dt
            msg = parse_message(data)
            if msg["rcode_name"] != "NOERROR":
                per_subnet[addr] = frozenset()
                continue
            per_subnet[addr] = frozenset(a.get("address") for a in msg["answer"] if a.get("address"))
        first_key = PROBE_SUBNETS[0][2]
        if not per_subnet.get(first_key):
            return None
        distinct_answer_sets = {v for v in per_subnet.values() if v}
        ecs_sensitive = len(distinct_answer_sets) > 1
        return {"name": name, "live": True, "rtt_probe_s": rtt0,
                "answers_by_subnet": {k: sorted(v) for k, v in per_subnet.items()},
                "ecs_sensitive": ecs_sensitive, "n_answers": len(per_subnet.get(first_key, []))}
    except Exception:
        return None
    finally:
        s.close()


def build_pool(max_candidates: int = 900, target_live: int = 350, target_ecs_sensitive: int = 60) -> dict:
    candidates = load_candidate_names()[:max_candidates]
    live, ecs_sensitive = [], []
    probed = 0
    for name in candidates:
        result = _probe_one(name)
        probed += 1
        if result is not None:
            live.append(result)
            if result["ecs_sensitive"]:
                ecs_sensitive.append(result)
        if len(live) >= target_live and len(ecs_sensitive) >= target_ecs_sensitive and probed >= 400:
            break
    out = {
        "collected_at_unix": time.time(),
        "n_candidates_probed": probed,
        "n_live": len(live),
        "n_ecs_sensitive": len(ecs_sensitive),
        "live_domains": live,
    }
    (PROCESSED_DIR / "domain_pool.json").write_text(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    result = build_pool()
    print(f"probed={result['n_candidates_probed']} live={result['n_live']} ecs_sensitive={result['n_ecs_sensitive']}")
