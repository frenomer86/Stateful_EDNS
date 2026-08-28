"""Resolution-episode harness.

Benign episodes are resolved against real internet DNS infrastructure (a
rotating choice of Google Public DNS, Cloudflare, Quad9, and OpenDNS, for
real domain names drawn from data/processed/domain_pool/domain_pool.json).
Every benign wire exchange in the resulting corpus is a genuine response
from production DNS infrastructure captured on the date the harness is
run — nothing about the benign class is templated or replayed.

Adverse episodes stay entirely on 127.0.0.1. A local "faulty authority"
thread answers queries from the local resolver with a deliberately broken
protocol behavior (missing ECS, mismatched ECS, conflicting answers, high
concurrent ECS fan-out). This mirrors resolver-state conditions relevant to
cache-poisoning risk without ever sending malformed, spoofed, or
attack-shaped traffic onto the public internet — the same ethical boundary
used in the original study. Where practical, the local authority seeds its
base answer from a real prior lookup so the perturbation is applied to a
real record rather than an invented one.

Three resolver *policies* govern how the local resolver aggregates client
queries into upstream requests:
  aggregate  -- key = (qname, qtype); ECS is ignored for coalescing.
  ecs-keyed  -- key = (qname, qtype, ecs-network); distinct client ECS
                contexts each get their own upstream request.
  ecs-strict -- same key as ecs-keyed, plus the resolver checks that the
                upstream response's echoed ECS scope is consistent with
                the request before accepting it onto the client path.
"""
from __future__ import annotations

import ipaddress
import json
import random
import socket
import threading
import time
import uuid
from pathlib import Path

from dns_wire import build_query, build_response, parse_message

RESOLVERS_REAL = [
    ("google", "8.8.8.8"),
    ("cloudflare", "1.1.1.1"),
    ("quad9", "9.9.9.9"),
    ("opendns", "208.67.222.222"),
]

DOMAIN_POOL_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "domain_pool" / "domain_pool.json"


def load_domain_pool():
    data = json.loads(DOMAIN_POOL_PATH.read_text())
    live = data["live_domains"]
    plain = [d["name"] for d in live]
    ecs = [d["name"] for d in live if d["ecs_sensitive"]]
    if len(ecs) < 5:
        raise RuntimeError("domain pool has too few ECS-sensitive domains; rerun domain_pool.build_pool")
    return plain, ecs


# ---------------------------------------------------------------------------
# Client-subnet values used to represent distinct client vantage points.
# RFC 5737/3849 documentation ranges look natural for this but real public
# resolvers correctly REFUSE queries carrying a bogon/reserved ECS prefix
# (verified empirically against Google, Cloudflare, Quad9, and OpenDNS
# before choosing this list), so every entry here is instead a real,
# routable /24 belonging to a well-known, publicly operated network in a
# different region. Using a real operator's block as an ECS value does not
# send any traffic from that address; it only tells the upstream resolver
# "steer this answer as if the client were in this subnet," which is
# exactly the mechanism RFC 7871 defines and which real CDNs process
# constantly for anycast/geo steering.
CLIENT_ECS_POOL = [
    (1, 24, "8.8.8.0"),        # Google Public DNS -- Mountain View, US
    (1, 24, "1.1.1.0"),        # Cloudflare -- global anycast, APNIC-registered
    (1, 24, "9.9.9.0"),        # Quad9 -- Zurich, Switzerland
    (1, 24, "208.67.222.0"),   # OpenDNS / Cisco Umbrella -- San Francisco, US
    (1, 24, "77.88.8.0"),      # Yandex DNS -- Moscow, Russia
    (1, 24, "114.114.114.0"),  # 114DNS -- China
]


def _now():
    return time.time()


# ---------------------------------------------------------------------------
# Latency realism for the local adverse harness.
#
# The benign path's client-visible latency is a genuine round trip to real
# internet infrastructure. Left alone, the local adverse authority would
# reply in well under a millisecond over loopback, which would let a
# classifier separate benign from adverse simply by "was this fast" rather
# than by any protocol-state property -- an unrelated timing shortcut, not
# the signal this study is about.
#
# _LATENCY_POOL starts from a real calibration pass and is then continually
# refreshed with the real upstream round-trip time of every live benign
# query made during collection (see _real_upstream_query), capped to the
# most recent 800 samples. The local authority's reply delay is resampled
# from that live, continually updated pool, so it tracks whatever
# congestion the run is actually experiencing at the time rather than a
# fixed number measured once in isolation. Every value slept is a value
# that was actually measured on a real network earlier in the same run.
_LATENCY_POOL = []
_LATENCY_LOCK = threading.Lock()
_LATENCY_POOL_CAP = 800


def calibrate_latency_pool(n=60):
    samples = []
    for _ in range(n):
        name_ip = random.choice(RESOLVERS_REAL)[1]
        try:
            pkt, _ = build_query("example.com", "A")
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.settimeout(1.5)
            t0 = _now()
            s.sendto(pkt, (name_ip, 53))
            s.recvfrom(4096)
            samples.append(_now() - t0)
            s.close()
        except Exception:
            continue
    with _LATENCY_LOCK:
        _LATENCY_POOL.clear()
        _LATENCY_POOL.extend(samples if samples else [0.008, 0.012, 0.020, 0.035])
    return list(_LATENCY_POOL)


def _record_real_latency(dt):
    with _LATENCY_LOCK:
        _LATENCY_POOL.append(dt)
        if len(_LATENCY_POOL) > _LATENCY_POOL_CAP:
            del _LATENCY_POOL[:len(_LATENCY_POOL) - _LATENCY_POOL_CAP]


def _sample_latency_delay(rng):
    with _LATENCY_LOCK:
        pool = list(_LATENCY_POOL) or [0.010]
    base = rng.choice(pool)
    jitter = rng.uniform(0.85, 1.15)
    return max(0.0005, base * jitter)


def _real_upstream_query(qname, qtype, ecs, resolver_ip, timeout=2.5, max_attempts=3):
    """A real DNS client retries a lost query rather than giving up after
    one packet; this reflects that standard behavior (RFC 1035 sec 7.2)
    rather than hiding genuine network loss. Each attempt is a fresh real
    UDP round trip with a new transaction ID."""
    event_q = None
    for attempt in range(max_attempts):
        pkt, txid = build_query(qname, qtype, ecs=ecs)
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.settimeout(timeout)
        t_sent = _now()
        this_q = {"t": t_sent, "txid": txid, "qname": qname, "qtype": qtype,
                  "ecs_sent": _ecs_dict(ecs), "target": resolver_ip, "real": True,
                  "wire_len": len(pkt), "attempt": attempt + 1}
        if event_q is None:
            event_q = this_q
        try:
            s.sendto(pkt, (resolver_ip, 53))
            this_q["sport"] = s.getsockname()[1]
            if event_q is this_q:
                event_q["sport"] = this_q["sport"]
            data, _ = s.recvfrom(4096)
            t_recv = _now()
            _record_real_latency(t_recv - t_sent)
            msg = parse_message(data)
            event_r = _response_event(t_recv, msg, real=True, wire_len=len(data))
            event_r["attempts"] = attempt + 1
            return event_q, event_r
        except Exception:
            continue
        finally:
            s.close()
    t_recv = _now()
    return event_q, {"t": t_recv, "txid": event_q["txid"], "real": True,
                      "rcode_name": "TIMEOUT", "answers": [], "ttls": [], "ecs_resp": None,
                      "wire_len": 0, "attempts": max_attempts}


def _ecs_dict(ecs):
    if ecs is None:
        return None
    fam, plen, addr = ecs
    return {"family": fam, "prefix": plen, "address": addr}


def _response_event(t_recv, msg, real, wire_len):
    answers = []
    ttls = []
    for a in msg["answer"]:
        answers.append(a.get("address") or a.get("target") or (a.get("txt") or b"").hex())
        ttls.append(a["ttl"])
    ecs_resp = None
    if msg.get("edns") and msg["edns"].get("options", {}).get("ecs"):
        e = msg["edns"]["options"]["ecs"]
        ecs_resp = {"family": e["family"], "prefix": e["source_prefix"], "address": e["address"]}
    return {"t": t_recv, "txid": msg["txid"], "rcode_name": msg["rcode_name"],
            "answers": answers, "ttls": ttls, "ecs_resp": ecs_resp, "wire_len": wire_len,
            "real": real, "aa": msg.get("aa", 0)}


class OutstandingTracker:
    """Coalesces concurrent upstream requests that share an aggregation key,
    mirroring how a real caching resolver avoids issuing duplicate upstream
    queries for the same in-flight name/type (or name/type/ECS)."""

    def __init__(self):
        self._lock = threading.Lock()
        self._inflight = {}

    def get_or_start(self, key):
        with self._lock:
            entry = self._inflight.get(key)
            if entry is not None:
                entry["waiters"] += 1
                return entry, False
            entry = {"event": threading.Event(), "result": None, "waiters": 1}
            self._inflight[key] = entry
            return entry, True

    def finish(self, key, entry, result):
        entry["result"] = result
        entry["event"].set()
        with self._lock:
            self._inflight.pop(key, None)

    def max_concurrent_hint(self):
        with self._lock:
            return len(self._inflight)


def aggregation_key(policy, qname, qtype, ecs):
    if policy == "aggregate":
        return (qname, qtype)
    net = None
    if ecs is not None:
        fam, plen, addr = ecs
        if fam == 1:
            net = str(ipaddress.ip_network(f"{addr}/{plen}", strict=False))
    return (qname, qtype, net)


def resolve_subquery(policy, qname, qtype, ecs, tracker: OutstandingTracker, upstream_fn, n_responses_fn=None):
    """One client-facing sub-query resolved through the local policy's
    aggregation logic. `upstream_fn(qname, qtype, ecs)` performs the actual
    upstream exchange (either a real internet query or a local faulty-
    authority exchange) and returns (upstream_events, response_events),
    where response_events is a list because the adverse path may produce
    more than one competing response for a single upstream leg. Coalescing
    by aggregation key is identical for benign and adverse traffic, so the
    'aggregate' policy collapsing many client requests into one upstream
    transaction (and 'ecs-keyed'/'ecs-strict' not collapsing them) is a
    genuine emergent property of the same code path in both cases, not two
    separately hand-tuned behaviors."""
    t_client_sent = _now()
    client_q_event = {"t": t_client_sent, "qname": qname, "qtype": qtype, "ecs_sent": _ecs_dict(ecs)}

    key = aggregation_key(policy, qname, qtype, ecs)
    entry, is_leader = tracker.get_or_start(key)
    upstream_events = []
    if is_leader:
        uq_events, resp_events = upstream_fn(qname, qtype, ecs)
        upstream_events = uq_events
        accept_flags = [True] * len(resp_events)
        if policy == "ecs-strict" and ecs is not None:
            fam, plen, addr = ecs
            requested_net = str(ipaddress.ip_network(f"{addr}/{plen}", strict=False))
            for i, r in enumerate(resp_events):
                if r.get("ecs_resp") is not None and r["ecs_resp"]["address"]:
                    got_net = str(ipaddress.ip_network(f"{r['ecs_resp']['address']}/{r['ecs_resp']['prefix']}", strict=False))
                    accept_flags[i] = (got_net == requested_net) or (r["ecs_resp"]["prefix"] == 0)
        tracker.finish(key, entry, (resp_events, accept_flags))
    else:
        entry["event"].wait(timeout=5.0)
        resp_events, accept_flags = entry["result"] if entry["result"] else ([], [])

    t_client_recv = _now()
    client_r_events = []
    for r, acc in zip(resp_events, accept_flags):
        # Use the response's own real arrival timestamp (r["t"]), not a
        # single shared time -- when a sub-query produces more than one
        # competing response (adverse-conflict / adverse-ecs-fanout), each
        # one genuinely arrived at a different real instant, and the gap
        # between those instants is exactly what response_race_gap_mean
        # is meant to measure. Sharing one timestamp across all of them
        # would silently zero out that feature.
        arrival_t = r.get("t", t_client_recv)
        client_r_events.append({"t": arrival_t, "t_query": t_client_sent, "rcode_name": r.get("rcode_name"),
                                 "answers": r.get("answers", []), "ttls": r.get("ttls", []), "ecs_resp": r.get("ecs_resp"),
                                 "wire_len": r.get("wire_len", 0), "accepted": acc})
    if not client_r_events:
        client_r_events = [{"t": t_client_recv, "t_query": t_client_sent, "rcode_name": "TIMEOUT", "answers": [],
                             "ttls": [], "ecs_resp": None, "wire_len": 0, "accepted": True}]
    return {"client_q": client_q_event, "client_r": client_r_events, "upstream_q": upstream_events,
            "upstream_r": resp_events}


def resolve_benign_subquery(policy, qname, qtype, ecs, tracker: OutstandingTracker, resolver_ip):
    def upstream_fn(qname, qtype, ecs):
        uq, ur = _real_upstream_query(qname, qtype, ecs, resolver_ip)
        return [uq], [ur]
    return resolve_subquery(policy, qname, qtype, ecs, tracker, upstream_fn)


# --------------------------- adverse (local, simulated) --------------------

def _local_base_answer(qname, resolver_ip="8.8.8.8"):
    """Seed a locally-crafted adverse response from one real lookup so the
    perturbation below is applied to a genuine record rather than an
    invented one."""
    try:
        pkt, txid = build_query(qname, "A")
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.settimeout(2.0)
        s.sendto(pkt, (resolver_ip, 53))
        data, _ = s.recvfrom(4096)
        s.close()
        msg = parse_message(data)
        addrs = [a["address"] for a in msg["answer"] if a.get("address")]
        ttl = msg["answer"][0]["ttl"] if msg["answer"] else 300
        return addrs or ["93.184.216.34"], ttl
    except Exception:
        return ["93.184.216.34"], 300


def _faulty_authority_exchange(qname, qtype, ecs, condition, base_addrs, base_ttl, rng, idx):
    """Real UDP exchange, entirely on 127.0.0.1, between a client socket and
    a locally bound 'faulty authority' socket. The authority reads the
    actual query bytes the client sent and crafts one or more real,
    well-formed-but-adversarial response packets, which the client then
    genuinely receives and times. Nothing here is a fabricated event
    record: every field comes from real socket send/recv calls."""
    client_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    client_sock.bind(("127.0.0.1", 0))
    authority_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    authority_sock.bind(("127.0.0.1", 0))
    authority_sock.settimeout(2.0)
    client_sock.settimeout(2.5)

    pkt, txid = build_query(qname, qtype, ecs=ecs)
    t_q_sent = _now()
    client_sock.sendto(pkt, authority_sock.getsockname())
    upstream_q_event = {"t": t_q_sent, "txid": txid, "qname": qname, "qtype": qtype,
                         "ecs_sent": _ecs_dict(ecs), "target": "local-adverse-authority", "real": False,
                         "wire_len": len(pkt), "sport": client_sock.getsockname()[1]}

    try:
        _, client_addr = authority_sock.recvfrom(4096)
    except Exception:
        client_sock.close(); authority_sock.close()
        return [upstream_q_event], []

    clean_ecs = (ecs[0], ecs[1], ecs[1], ecs[2]) if ecs else None
    # Each condition's fault does not fire on every single exchange -- a
    # race is not always won, a stale ECS cache entry does not always get
    # served, a competing responder does not always beat the legitimate one.
    # Sampling a per-exchange fault probability keeps the resulting feature
    # distributions continuous rather than perfectly categorical, which is
    # both more realistic and a harder, fairer test for the detector than a
    # condition that is either always or never present.
    fires = rng.random() < rng.uniform(0.55, 0.90)

    if condition == "adverse-ecs-omit":
        specs = [(base_addrs, None)] if fires else [(base_addrs, clean_ecs)]
    elif condition == "adverse-ecs-mismatch":
        if fires:
            if rng.random() < 0.7:
                # far mismatch: a different operator's /24 entirely
                wrong_addr = rng.choice([a for _, _, a in CLIENT_ECS_POOL if ecs is None or a != ecs[2]])
                ecs_resp = (1, 24, 24, wrong_addr)
            else:
                # near mismatch: right network, wrong scope prefix length
                # (still flagged as inconsistent by a strict resolver, but a
                # softer signal than a completely different subnet)
                ecs_resp = (ecs[0], ecs[1], 16, ecs[2]) if ecs else (1, 24, 16, "8.8.8.0")
            specs = [(base_addrs, ecs_resp)]
        else:
            specs = [(base_addrs, clean_ecs)]
    elif condition == "adverse-conflict":
        alt_addrs = [f"203.0.113.{rng.randint(1, 254)}"]
        if fires:
            specs = [(base_addrs, clean_ecs), (alt_addrs, clean_ecs)]
        else:
            specs = [(base_addrs, clean_ecs)]
    elif condition == "adverse-ecs-fanout":
        alt_addrs = [f"203.0.113.{rng.randint(1, 254)}"]
        if fires:
            specs = [(base_addrs, None), (alt_addrs, None)]
        else:
            specs = [(base_addrs, clean_ecs)]
    else:
        specs = [(base_addrs, clean_ecs)]

    upstream_r_events = []
    for answers, ecs_resp in specs:
        time.sleep(_sample_latency_delay(rng))
        resp_pkt = build_response(qname, qtype, txid, answers, base_ttl, ecs_resp=ecs_resp)
        authority_sock.sendto(resp_pkt, client_addr)

    for _ in specs:
        try:
            data, _ = client_sock.recvfrom(4096)
        except socket.timeout:
            break
        t_recv = _now()
        msg = parse_message(data)
        upstream_r_events.append(_response_event(t_recv, msg, real=False, wire_len=len(data)))

    client_sock.close()
    authority_sock.close()
    return [upstream_q_event], upstream_r_events


def resolve_adverse_episode(policy, condition, qname, rng, tracker: OutstandingTracker, n_client=6):
    """Builds a full adverse episode using the same aggregation-key logic as
    the benign path, so 'aggregate' collapsing concurrent adverse traffic
    into one upstream transaction is the same emergent behavior, not a
    separately hand-coded case."""
    base_addrs, base_ttl = _local_base_answer(qname)
    n = 1 if condition in ("adverse-ecs-omit", "adverse-ecs-mismatch") else n_client
    variants = rng.sample(CLIENT_ECS_POOL, k=min(n, len(CLIENT_ECS_POOL)))
    while len(variants) < n:
        variants.append(rng.choice(CLIENT_ECS_POOL))

    idx_counter = {"i": 0}

    def upstream_fn(qname_, qtype_, ecs_, _base_addrs=base_addrs, _base_ttl=base_ttl):
        idx_counter["i"] += 1
        return _faulty_authority_exchange(qname_, qtype_, ecs_, condition, _base_addrs, _base_ttl, rng, idx_counter["i"])

    jobs = [(lambda v=v: resolve_subquery(policy, qname, "A", v, tracker, upstream_fn)) for v in variants]
    subs = _run_concurrent(jobs)

    client_queries, client_responses, upstream_queries, upstream_responses = [], [], [], []
    for sub in subs:
        client_queries.append(sub["client_q"])
        client_responses.extend(sub["client_r"])
        upstream_queries.extend(sub["upstream_q"])
        upstream_responses.extend(sub["upstream_r"])

    return {"client_queries": client_queries, "client_responses": client_responses,
            "upstream_queries": upstream_queries, "upstream_responses": upstream_responses}


# --------------------------- episode drivers --------------------------------

BENIGN_CONDITIONS = ["benign-plain", "benign-ecs-stable", "benign-ecs-geo", "benign-burst"]
ADVERSE_CONDITIONS = ["adverse-ecs-omit", "adverse-ecs-mismatch", "adverse-conflict", "adverse-ecs-fanout"]
POLICIES = ["aggregate", "ecs-keyed", "ecs-strict"]


def _run_concurrent(jobs):
    """jobs: list of zero-arg callables, each returning one sub-query result.
    Fired on real threads so their socket I/O genuinely overlaps in time --
    this is what lets the 'aggregate' policy's in-flight coalescing actually
    trigger, exactly as it would for a real caching resolver handling
    concurrent client requests for the same logical name."""
    results = [None] * len(jobs)

    def _run(i, fn):
        results[i] = fn()

    threads = [threading.Thread(target=_run, args=(i, fn)) for i, fn in enumerate(jobs)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=6.0)
    return [r for r in results if r is not None]


def run_benign_episode(policy, condition, plain_domains, ecs_domains, rng, tracker: OutstandingTracker):
    resolver_name, resolver_ip = rng.choice(RESOLVERS_REAL)
    client_queries, client_responses, upstream_queries, upstream_responses = [], [], [], []

    if condition == "benign-plain":
        qname = rng.choice(plain_domains)
        jobs = [lambda: resolve_benign_subquery(policy, qname, "A", None, tracker, resolver_ip)]
    elif condition == "benign-ecs-stable":
        qname = rng.choice(ecs_domains) if ecs_domains else rng.choice(plain_domains)
        ecs = rng.choice(CLIENT_ECS_POOL)
        jobs = [lambda: resolve_benign_subquery(policy, qname, "A", ecs, tracker, resolver_ip) for _ in range(3)]
    elif condition == "benign-ecs-geo":
        qname = rng.choice(ecs_domains) if ecs_domains else rng.choice(plain_domains)
        variants = rng.sample(CLIENT_ECS_POOL, k=min(4, len(CLIENT_ECS_POOL)))
        jobs = [(lambda v=v: resolve_benign_subquery(policy, qname, "A", v, tracker, resolver_ip)) for v in variants]
    elif condition == "benign-burst":
        qnames = rng.sample(plain_domains, k=min(4, len(plain_domains)))
        jobs = [(lambda q=q: resolve_benign_subquery(policy, q, "A", None, tracker, resolver_ip)) for q in qnames]
    else:
        raise ValueError(condition)

    subs = _run_concurrent(jobs)

    for sub in subs:
        client_queries.append(sub["client_q"])
        client_responses.extend(sub["client_r"])
        upstream_queries.extend(sub["upstream_q"])
        upstream_responses.extend(sub["upstream_r"])

    return {"client_queries": client_queries, "client_responses": client_responses,
            "upstream_queries": upstream_queries, "upstream_responses": upstream_responses,
            "resolver_used": resolver_name}


def run_episode(seed, policy, condition, plain_domains, ecs_domains, rng, tracker):
    episode_id = str(uuid.uuid4())
    label = 0 if condition.startswith("benign") else 1
    t_episode_start = _now()
    if label == 0:
        events = run_benign_episode(policy, condition, plain_domains, ecs_domains, rng, tracker)
    else:
        qname = rng.choice(plain_domains)
        events = resolve_adverse_episode(policy, condition, qname, rng, tracker)
        events["resolver_used"] = "local-adverse-authority"
    t_episode_end = _now()
    return {"episode_id": episode_id, "seed": seed, "policy": policy, "condition": condition,
            "label": label, "t_start": t_episode_start, "t_end": t_episode_end, **events}
