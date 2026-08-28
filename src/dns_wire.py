"""Minimal, dependency-free DNS wire-format encoder/decoder.

Built by hand because this execution environment cannot reach PyPI for third-party DNS
libraries (dnspython, scapy). Implements just enough of RFC 1035 (message
format), RFC 6891 (EDNS0/OPT), and RFC 7871 (EDNS Client Subnet) to build
real queries and parse real responses from live DNS infrastructure,
including name-compression pointers, which real-world responses use
constantly and which a naive parser will choke on.

This module is used both for the live-internet benign traffic (talking to
real root/TLD/authoritative servers and real public resolvers) and for the
local adverse-condition harness, so a single, correctly-tested wire format
implementation underlies every packet in the study.
"""
from __future__ import annotations

import struct
import socket
import random
from dataclasses import dataclass, field

QTYPE = {"A": 1, "NS": 2, "CNAME": 5, "SOA": 6, "PTR": 12, "MX": 15, "TXT": 16,
         "AAAA": 28, "SRV": 33, "OPT": 41, "DS": 43, "RRSIG": 46, "NSEC": 47,
         "DNSKEY": 48, "SVCB": 64, "HTTPS": 65, "ANY": 255}
QTYPE_REV = {v: k for k, v in QTYPE.items()}
RCODE_NAMES = {0: "NOERROR", 1: "FORMERR", 2: "SERVFAIL", 3: "NXDOMAIN",
               4: "NOTIMP", 5: "REFUSED"}


def encode_name(name: str) -> bytes:
    name = name.strip(".")
    if name == "":
        return b"\x00"
    out = b""
    for label in name.split("."):
        lb = label.encode("idna") if not label.isascii() else label.encode("ascii")
        if len(lb) > 63:
            raise ValueError("label too long: %r" % label)
        out += bytes([len(lb)]) + lb
    return out + b"\x00"


def decode_name(buf: bytes, offset: int, max_ptr_hops: int = 64):
    """Decode a possibly-compressed name starting at offset. Returns (name, next_offset)."""
    labels = []
    pos = offset
    jumped = False
    end_offset = None
    hops = 0
    while True:
        if pos >= len(buf):
            raise ValueError("name decode ran off buffer")
        length = buf[pos]
        if length == 0:
            pos += 1
            if not jumped:
                end_offset = pos
            break
        if (length & 0xC0) == 0xC0:
            if hops > max_ptr_hops:
                raise ValueError("too many compression pointer hops")
            hops += 1
            if pos + 1 >= len(buf):
                raise ValueError("truncated compression pointer")
            ptr = ((length & 0x3F) << 8) | buf[pos + 1]
            if not jumped:
                end_offset = pos + 2
                jumped = True
            pos = ptr
            continue
        pos += 1
        labels.append(buf[pos:pos + length].decode("ascii", errors="replace"))
        pos += length
    return ".".join(labels), end_offset


def build_ecs_option(family: int, prefix_len: int, address: str) -> bytes:
    if family == 1:
        raw = socket.inet_aton(address)
        nbytes = (prefix_len + 7) // 8
        raw = raw[:nbytes]
    elif family == 2:
        raw = socket.inet_pton(socket.AF_INET6, address)
        nbytes = (prefix_len + 7) // 8
        raw = raw[:nbytes]
    else:
        raise ValueError("bad ECS family")
    opt_data = struct.pack("!HBB", family, prefix_len, 0) + raw
    # option code 8 = ECS, RFC 7871
    return struct.pack("!HH", 8, len(opt_data)) + opt_data


@dataclass
class QueryResult:
    txid: int
    qname: str
    qtype: str
    wire_query: bytes
    wire_response: bytes | None = None
    rcode: int | None = None
    answers: list = field(default_factory=list)
    ttl_values: list = field(default_factory=list)
    response_ecs: dict | None = None
    edns_present: bool = False
    error: str | None = None
    t_sent: float = 0.0
    t_recv: float | None = None


def build_query(qname: str, qtype: str = "A", ecs: tuple | None = None,
                 txid: int | None = None, dnssec_ok: bool = False,
                 recursion_desired: bool = True) -> tuple[bytes, int]:
    """ecs: (family, prefix_len, address_str) or None."""
    if txid is None:
        txid = random.randint(0, 0xFFFF)
    flags = 0x0100 if recursion_desired else 0x0000
    qdcount, ancount, nscount = 1, 0, 0
    arcount = 1  # we always add an OPT record (EDNS0) — realistic modern-client behavior
    header = struct.pack("!HHHHHH", txid, flags, qdcount, ancount, nscount, arcount)
    question = encode_name(qname) + struct.pack("!HH", QTYPE[qtype], 1)
    opt_rdata = b""
    if ecs is not None:
        opt_rdata += build_ecs_option(*ecs)
    do_bit = 0x8000 if dnssec_ok else 0x0000
    opt_rr = (b"\x00" + struct.pack("!H", QTYPE["OPT"]) + struct.pack("!H", 4096) +
              struct.pack("!BBH", 0, 0, do_bit) + struct.pack("!H", len(opt_rdata)) + opt_rdata)
    packet = header + question + opt_rr
    return packet, txid


def _parse_rr(buf: bytes, pos: int):
    name, pos = decode_name(buf, pos)
    rtype, rclass, ttl, rdlen = struct.unpack_from("!HHIH", buf, pos)
    pos += 10
    rdata = buf[pos:pos + rdlen]
    rr = {"name": name, "type": rtype, "type_name": QTYPE_REV.get(rtype, str(rtype)),
          "class": rclass, "ttl": ttl, "rdlength": rdlen}
    if rtype == QTYPE["A"] and rdlen == 4:
        rr["address"] = socket.inet_ntoa(rdata)
    elif rtype == QTYPE["AAAA"] and rdlen == 16:
        rr["address"] = socket.inet_ntop(socket.AF_INET6, rdata)
    elif rtype in (QTYPE["CNAME"], QTYPE["NS"], QTYPE["PTR"]):
        target, _ = decode_name(buf, pos)
        rr["target"] = target
    elif rtype == QTYPE["MX"] and rdlen >= 2:
        pref = struct.unpack_from("!H", rdata, 0)[0]
        target, _ = decode_name(buf, pos + 2)
        rr["preference"] = pref
        rr["target"] = target
    elif rtype == QTYPE["TXT"]:
        chunks = []
        i = 0
        while i < len(rdata):
            ln = rdata[i]
            chunks.append(rdata[i + 1:i + 1 + ln])
            i += 1 + ln
        rr["txt"] = b"".join(chunks)
    elif rtype == QTYPE["OPT"]:
        rr["udp_size"] = rclass
        # For an OPT RR the 32-bit TTL field is repurposed: ext-rcode(8) | version(8) | flags(16, DO bit at top)
        ext_rcode = (ttl >> 24) & 0xFF
        version = (ttl >> 16) & 0xFF
        do_bit = bool(ttl & 0x8000)
        rr["ext_rcode"] = ext_rcode
        rr["version"] = version
        rr["do_bit"] = do_bit
        options = {}
        i = 0
        while i + 4 <= len(rdata):
            code, olen = struct.unpack_from("!HH", rdata, i)
            odata = rdata[i + 4:i + 4 + olen]
            if code == 8 and len(odata) >= 4:
                fam, srclen, scopelen = struct.unpack_from("!HBB", odata, 0)
                addr_bytes = odata[4:]
                if fam == 1:
                    padded = addr_bytes + b"\x00" * (4 - len(addr_bytes))
                    addr = socket.inet_ntoa(padded[:4])
                elif fam == 2:
                    padded = addr_bytes + b"\x00" * (16 - len(addr_bytes))
                    addr = socket.inet_ntop(socket.AF_INET6, padded[:16])
                else:
                    addr = None
                options["ecs"] = {"family": fam, "source_prefix": srclen,
                                   "scope_prefix": scopelen, "address": addr}
            i += 4 + olen
        rr["options"] = options
    return rr, pos + rdlen


def build_response(qname: str, qtype: str, txid: int, answers: list[str], ttl: int,
                    rcode: str = "NOERROR", ecs_resp: tuple | None = None,
                    aa: bool = True) -> bytes:
    """Build a real, well-formed DNS response packet (used by the local
    adverse-condition authority). answers is a list of dotted-quad strings
    for an A response. ecs_resp, if given, is (family, source_prefix,
    scope_prefix, address) echoed in the OPT record."""
    rcode_num = {v: k for k, v in RCODE_NAMES.items()}.get(rcode, 0)
    flags = 0x8000 | (0x0400 if aa else 0) | 0x0080 | rcode_num  # QR=1, RD=1 mirrored below
    flags |= 0x0100  # RD
    qdcount = 1
    ancount = len(answers)
    arcount = 1
    header = struct.pack("!HHHHHH", txid, flags, qdcount, ancount, 0, arcount)
    question = encode_name(qname) + struct.pack("!HH", QTYPE[qtype], 1)
    answer_bytes = b""
    for addr in answers:
        rdata = socket.inet_aton(addr)
        answer_bytes += (b"\xc0\x0c" + struct.pack("!HHIH", QTYPE[qtype], 1, ttl, len(rdata)) + rdata)
    opt_rdata = b""
    if ecs_resp is not None:
        fam, srclen, scopelen, addr = ecs_resp
        raw = socket.inet_aton(addr) if fam == 1 else socket.inet_pton(socket.AF_INET6, addr)
        nbytes = (srclen + 7) // 8
        opt_data = struct.pack("!HBB", fam, srclen, scopelen) + raw[:nbytes]
        opt_rdata = struct.pack("!HH", 8, len(opt_data)) + opt_data
    opt_rr = (b"\x00" + struct.pack("!H", QTYPE["OPT"]) + struct.pack("!H", 4096) +
              struct.pack("!BBH", 0, 0, 0) + struct.pack("!H", len(opt_rdata)) + opt_rdata)
    return header + question + answer_bytes + opt_rr


def parse_message(buf: bytes) -> dict:
    if len(buf) < 12:
        raise ValueError("message too short")
    txid, flags, qdcount, ancount, nscount, arcount = struct.unpack_from("!HHHHHH", buf, 0)
    qr = (flags >> 15) & 1
    opcode = (flags >> 11) & 0xF
    aa = (flags >> 10) & 1
    tc = (flags >> 9) & 1
    rd = (flags >> 8) & 1
    ra = (flags >> 7) & 1
    rcode = flags & 0xF
    pos = 12
    questions = []
    for _ in range(qdcount):
        qname, pos = decode_name(buf, pos)
        qtype, qclass = struct.unpack_from("!HH", buf, pos)
        pos += 4
        questions.append({"qname": qname, "qtype": qtype, "qtype_name": QTYPE_REV.get(qtype, str(qtype))})
    answers, authority, additional = [], [], []
    for section, count in (("answer", ancount), ("authority", nscount), ("additional", arcount)):
        for _ in range(count):
            rr, pos = _parse_rr(buf, pos)
            {"answer": answers, "authority": authority, "additional": additional}[section].append(rr)
    edns = None
    for rr in additional:
        if rr["type"] == QTYPE["OPT"]:
            edns = rr
    return {"txid": txid, "qr": qr, "opcode": opcode, "aa": aa, "tc": tc, "rd": rd, "ra": ra,
            "rcode": rcode, "rcode_name": RCODE_NAMES.get(rcode, str(rcode)),
            "questions": questions, "answer": answers, "authority": authority,
            "additional": additional, "edns": edns, "wire_length": len(buf)}
