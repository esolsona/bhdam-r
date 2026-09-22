#!/usr/bin/env python3
"""
BHDAM-R comparative evaluation against baselines (paper revision).

Adds the comparative analysis requested at review: the overhead BHDAM-R pays
over a standard encrypted channel (TLS), the availability it buys under channel
failure that TLS-with-retry cannot match, and how it differs from a classical
dispersed-storage baseline (AONT-RS). Results are reported throughput-first so
the trade-off is honest rather than a raw speed race.

Subcommands
  tls-overhead   BHDAM-R sealing+dispersal vs a TLS-like AES-256-GCM record
                 baseline, over sizes x profiles. Reports MB/s + ratio.
  tls-real       Transfer the payload over a real TLS 1.3 loopback socket, to
                 validate the synthetic baseline against a real TLS stack.
  failure        Completion probability + transmission cost under per-channel
                 failure: BHDAM-R k-of-n vs TLS single channel with full
                 retransmission, incl. an asynchronous physical channel.
  dispersed      Cost of BHDAM-R directed-transfer assurance over a bare
                 AONT-RS dispersed-storage core (same erasure primitives).
  parity         Capability comparison table (what plain TLS would need to add).
  all-compare    Run everything; write CSVs + figures.

Usage (in Docker, as the rest of the artefact):
  docker run --rm -v "$(pwd)/results:/app/results" bhdam-r \
      python compare.py all-compare --out /app/results \
      --sizes 1 10 100 500 --reps 20 --warmup 3 --trials 5000
"""
from __future__ import annotations

import argparse
import os
import socket
import ssl
import tempfile
import threading
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

import bhdam_r  # noqa: F401  (ensures the library is importable in-container)
from bhdam_r import Sender, Recipient  # noqa: F401
import datasets
from experiments import parties, _fresh_ev, _channels, _agg, _write_csv, _save

PROFILES = [(4, 5), (4, 6), (6, 9), (7, 10)]


def _raw_bytes(files: dict) -> bytes:
    """Concatenate the byte values of a datasets payload dict (flat baselines)."""
    return b"".join(files.values())


# --------------------------------------------------------------------------- #
# TLS-like synthetic baseline: AES-256-GCM over the payload (TLS 1.3 record).
# --------------------------------------------------------------------------- #
# TLS 1.3 protects data in fixed-size records (max 2^14 payload bytes each);
# AES-GCM itself is capped at 2^31-1 bytes per invocation. We therefore seal the
# payload in successive chunks, which is both correct (mirrors a real TLS record
# layer) and free of the single-call size limit. A 16 KiB record is the TLS
# maximum; we use a larger 4 MiB chunk to keep per-call overhead negligible while
# staying well under the AES-GCM limit, since we are measuring bulk AEAD cost.
_TLS_CHUNK = 4 * 1024 * 1024  # 4 MiB

def tls_like_seal(payload: bytes, key: bytes) -> list[tuple[bytes, bytes]]:
    aead = AESGCM(key)
    out = []
    for off in range(0, len(payload), _TLS_CHUNK):
        nonce = os.urandom(12)
        out.append((nonce, aead.encrypt(nonce, payload[off:off + _TLS_CHUNK], None)))
    return out


def tls_like_open(records: list[tuple[bytes, bytes]], key: bytes) -> int:
    aead = AESGCM(key)
    total = 0
    for nonce, ct in records:
        total += len(aead.decrypt(nonce, ct, None))
    return total


def cmd_tls_overhead(args):
    s, spub, r, rpub = parties()
    key = os.urandom(32)
    reps, warmup = args.reps, args.warmup
    rows = []
    print(f"tls-overhead: sizes={args.sizes} MB, profiles={args.profiles_parsed}, "
          f"reps={reps} (+{warmup} warmup)")
    for mb in args.sizes:
        payload = datasets.sized_blob(mb)
        raw = _raw_bytes(payload)

        for _ in range(warmup):
            recs = tls_like_seal(raw, key); tls_like_open(recs, key)
        base_send = []
        for _ in range(reps):
            t0 = time.perf_counter()
            recs = tls_like_seal(raw, key)
            base_send.append((time.perf_counter() - t0) * 1000)
        b_send, _ = _agg(base_send)
        tls_thr = mb / (b_send / 1000) if b_send else 0

        for (k, nn) in args.profiles_parsed:
            ch = _channels(nn)
            for aont in (False, True):
                try:
                    for _ in range(warmup):
                        s.build_transfer(payload, r.recipient_id, rpub, k, nn, ch, use_aont=aont)
                    snd = []
                    for _ in range(reps):
                        t0 = time.perf_counter()
                        s.build_transfer(payload, r.recipient_id, rpub, k, nn, ch, use_aont=aont)
                        snd.append((time.perf_counter() - t0) * 1000)
                    b_snd, b_snd_sd = _agg(snd)
                    m2, d2, sh2, sig2, ev2 = s.build_transfer(
                        payload, r.recipient_id, rpub, k, nn, ch, use_aont=aont)
                    bh_thr = mb / (b_snd / 1000) if b_snd else 0
                    data_ovh = sum(len(x) for x in sh2) / (mb * 1024 * 1024)
                    rows.append({
                        "size_mb": mb, "profile": f"{k}-of-{nn}", "aont": aont,
                        "tls_throughput_MBps": round(tls_thr, 1),
                        "bhdamr_send_ms": round(b_snd, 2), "bhdamr_send_sd": round(b_snd_sd, 2),
                        "bhdamr_throughput_MBps": round(bh_thr, 1),
                        "seal_cost_x": round(b_snd / (mb / tls_thr * 1000), 1) if tls_thr else 0,
                        "data_overhead_pct": round((data_ovh - 1) * 100, 1),
                        "status": "ok",
                    })
                    print(f"  {mb:>4}MB {f'{k}-of-{nn}':>8} aont={str(aont):<5} "
                          f"TLS={tls_thr:7.0f}MB/s  BHDAM-R={bh_thr:6.0f}MB/s  "
                          f"(+{(data_ovh-1)*100:.0f}% data)")
                except (MemoryError, Exception) as e:
                    rows.append({
                        "size_mb": mb, "profile": f"{k}-of-{nn}", "aont": aont,
                        "tls_throughput_MBps": round(tls_thr, 1),
                        "bhdamr_throughput_MBps": None, "status": f"failed: {type(e).__name__}",
                    })
                    print(f"  {mb:>4}MB {f'{k}-of-{nn}':>8} aont={str(aont):<5} "
                          f"FAILED ({type(e).__name__}) - recorded and skipped")
                # incremental save after every (size,profile,aont): a later crash
                # never loses results already computed.
                _write_csv(os.path.join(args.out, "compare_tls_overhead.csv"), rows)
    try:
        _fig_tls_overhead([x for x in rows if x.get("status") == "ok"], args.out)
    except Exception as e:
        print(f"  (figure skipped: {e})")
    return rows


# --------------------------------------------------------------------------- #
# Real TLS 1.3 loopback transfer (validation of the synthetic baseline).
# --------------------------------------------------------------------------- #
def _make_selfsigned():
    from cryptography import x509
    from cryptography.x509.oid import NameOID
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from datetime import datetime, timedelta, timezone
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "localhost")])
    cert = (x509.CertificateBuilder().subject_name(name).issuer_name(name)
            .public_key(key.public_key()).serial_number(x509.random_serial_number())
            .not_valid_before(datetime.now(timezone.utc) - timedelta(days=1))
            .not_valid_after(datetime.now(timezone.utc) + timedelta(days=1))
            .add_extension(x509.SubjectAlternativeName([x509.DNSName("localhost")]), False)
            .sign(key, hashes.SHA256()))
    d = tempfile.mkdtemp()
    cpath, kpath = os.path.join(d, "c.pem"), os.path.join(d, "k.pem")
    with open(cpath, "wb") as f:
        f.write(cert.public_bytes(serialization.Encoding.PEM))
    with open(kpath, "wb") as f:
        f.write(key.private_bytes(serialization.Encoding.PEM,
                                  serialization.PrivateFormat.TraditionalOpenSSL,
                                  serialization.NoEncryption()))
    return cpath, kpath


def _tls_roundtrip_once(payload: bytes, cpath: str, kpath: str) -> float:
    sctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    sctx.load_cert_chain(cpath, kpath)
    sctx.minimum_version = ssl.TLSVersion.TLSv1_3
    lsock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    lsock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    lsock.bind(("127.0.0.1", 0)); lsock.listen(1)
    port = lsock.getsockname()[1]
    received = {}

    def server():
        conn, _ = lsock.accept()
        with sctx.wrap_socket(conn, server_side=True) as ss:
            buf = bytearray()
            while len(buf) < len(payload):
                chunk = ss.recv(1 << 20)
                if not chunk:
                    break
                buf += chunk
            received["n"] = len(buf)
            try:
                ss.sendall(b"OK")
            except Exception:
                pass

    th = threading.Thread(target=server); th.start()
    cctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    cctx.check_hostname = False
    cctx.verify_mode = ssl.CERT_NONE
    cctx.minimum_version = ssl.TLSVersion.TLSv1_3
    t0 = time.perf_counter()
    with socket.create_connection(("127.0.0.1", port)) as raw:
        with cctx.wrap_socket(raw, server_hostname="localhost") as cs:
            cs.sendall(payload)
            cs.recv(2)
    th.join()
    dt = (time.perf_counter() - t0) * 1000
    lsock.close()
    assert received.get("n") == len(payload)
    return dt


def cmd_tls_real(args):
    cpath, kpath = _make_selfsigned()
    reps, warmup = args.reps, args.warmup
    rows = []
    print(f"tls-real: real TLS 1.3 loopback, sizes={args.sizes} MB, reps={reps}")
    for mb in args.sizes:
        payload = _raw_bytes(datasets.sized_blob(mb))
        for _ in range(warmup):
            _tls_roundtrip_once(payload, cpath, kpath)
        times = [_tls_roundtrip_once(payload, cpath, kpath) for _ in range(reps)]
        mean, sd = _agg(times)
        thr = mb / (mean / 1000) if mean else 0
        rows.append({"size_mb": mb, "tls_real_ms_mean": round(mean, 2),
                     "tls_real_ms_std": round(sd, 2), "throughput_MBps": round(thr, 1)})
        print(f"  {mb:>4}MB  TLS1.3 loopback  {mean:7.1f}±{sd:<4.1f} ms  ({thr:.0f} MB/s)")
    _write_csv(os.path.join(args.out, "compare_tls_real.csv"), rows)
    return rows


# --------------------------------------------------------------------------- #
# Channel failure: BHDAM-R k-of-n vs TLS single channel w/ full retransmission.
# --------------------------------------------------------------------------- #
def cmd_failure(args):
    rng = np.random.default_rng(20260710)
    trials = args.trials
    p_grid = [round(x, 3) for x in np.linspace(0.0, 0.5, 11)]
    profiles = args.profiles_parsed
    rows = []
    print(f"failure: {trials} trials/point; BHDAM-R k-of-n vs TLS retransmission")
    max_attempts = 5
    for p in p_grid:
        tls_complete = 1 - p ** max_attempts
        exp_tx = sum(i * (p ** (i - 1)) * (1 - p) for i in range(1, max_attempts + 1)) \
                 + max_attempts * (p ** max_attempts)
        row = {"p_fail": p, "tls_complete": round(tls_complete, 4),
               "tls_exp_transmissions": round(exp_tx, 3)}
        for (k, n) in profiles:
            succ = sum(1 for _ in range(trials) if int(rng.binomial(n, 1 - p)) >= k)
            row[f"bhdamr_{k}of{n}_complete"] = round(succ / trials, 4)
            row[f"bhdamr_{k}of{n}_tx"] = round(n / k, 3)
        rows.append(row)
        pretty = "  ".join(f"{k}of{n}={row[f'bhdamr_{k}of{n}_complete']:.3f}"
                           for (k, n) in profiles)
        print(f"  p={p:.2f}  TLS={tls_complete:.3f}(x{exp_tx:.2f} tx)  {pretty}")
    _write_csv(os.path.join(args.out, "compare_failure.csv"), rows)
    _fig_failure(rows, p_grid, profiles, args.out)
    _failure_physical(args, rng)
    return rows


def _failure_physical(args, rng):
    trials = args.trials
    k, n = 4, 6
    p_net, p_phys = 0.15, 0.01
    succ_mixed = succ_netonly = 0
    for _ in range(trials):
        net = rng.random(n - 1) > p_net
        phys = rng.random() > p_phys
        if int(net.sum()) + int(phys) >= k:
            succ_mixed += 1
        if int(net.sum()) >= k:
            succ_netonly += 1
    rows = [{"scenario": "5 network + 1 physical (BHDAM-R)",
             "p_network": p_net, "p_physical": p_phys,
             "complete_prob": round(succ_mixed / trials, 4)},
            {"scenario": "5 network only (TLS cannot use physical)",
             "p_network": p_net, "p_physical": None,
             "complete_prob": round(succ_netonly / trials, 4)}]
    _write_csv(os.path.join(args.out, "compare_failure_physical.csv"), rows)
    print(f"  physical-channel (4-of-6): network+physical={rows[0]['complete_prob']:.3f} "
          f"vs network-only={rows[1]['complete_prob']:.3f}")
    return rows


# --------------------------------------------------------------------------- #
# Dispersed-storage baseline: bare AONT-RS core vs full BHDAM-R.
# --------------------------------------------------------------------------- #
def cmd_dispersed(args):
    s, spub, r, rpub = parties()
    reps, warmup = args.reps, args.warmup
    rows = []
    print(f"dispersed: AONT-RS core vs full BHDAM-R, sizes={args.sizes} MB")
    for mb in args.sizes:
        payload = datasets.sized_blob(mb)
        k, n = 4, 6
        ch = _channels(n)
        for _ in range(warmup):
            s.build_transfer(payload, r.recipient_id, rpub, k, n, ch, use_aont=True)
        base = []
        for _ in range(reps):
            ts = {}
            s.build_transfer(payload, r.recipient_id, rpub, k, n, ch,
                             use_aont=True, timings=ts)
            core = (ts.get("aont_ms", 0) + ts.get("erasure_encode_ms", 0)
                    + ts.get("shard_hash_ms", 0))
            base.append(core)
        base_mean, base_sd = _agg(base)
        full = []
        for _ in range(reps):
            t0 = time.perf_counter()
            s.build_transfer(payload, r.recipient_id, rpub, k, n, ch, use_aont=True)
            full.append((time.perf_counter() - t0) * 1000)
        full_mean, full_sd = _agg(full)
        rows.append({
            "size_mb": mb, "profile": f"{k}-of-{n}",
            "dispersed_core_ms": round(base_mean, 2), "dispersed_core_sd": round(base_sd, 2),
            "bhdamr_full_ms": round(full_mean, 2), "bhdamr_full_sd": round(full_sd, 2),
            "assurance_overhead_pct": round((full_mean / base_mean - 1) * 100, 1) if base_mean else 0,
        })
        print(f"  {mb:>4}MB  dispersed-core={base_mean:7.1f}ms  full={full_mean:7.1f}ms  "
              f"(+{(full_mean/base_mean-1)*100:.0f}% assurance)")
    _write_csv(os.path.join(args.out, "compare_dispersed.csv"), rows)
    return rows


# --------------------------------------------------------------------------- #
def cmd_parity(args):
    caps = [
        ("Confidentiality in transit", "Yes", "Yes",
         "Both AES-256-GCM; TLS 1.3 record layer vs BHDAM-R DEK."),
        ("Integrity in transit", "Yes", "Yes",
         "TLS MAC per record; BHDAM-R AEAD tag + signed manifest."),
        ("Survives loss of a channel", "No", "Yes",
         "TLS drops the connection and retransmits in full; BHDAM-R reconstructs from any k of n."),
        ("Split custody across intermediaries", "No", "Yes",
         "A TLS proxy/endpoint sees the whole stream; no single BHDAM-R channel holds enough shards."),
        ("Persistent tamper-evidence after delivery", "No", "Yes",
         "TLS protection ends with the session; BHDAM-R retains a signed manifest + evidence (R0-R5)."),
        ("Asynchronous / physical channels", "No", "Yes",
         "TLS needs a live connection; BHDAM-R can use a courier or portable disk as one channel."),
        ("Recipient-bound reconstruction", "Partial", "Yes",
         "TLS authenticates the endpoint for the session; BHDAM-R binds the DEK to the recipient key."),
    ]
    rows = [{"capability": c, "plain_TLS": t, "BHDAM_R": b, "notes": n}
            for (c, t, b, n) in caps]
    _write_csv(os.path.join(args.out, "compare_parity.csv"), rows)
    print("Capability comparison (plain TLS vs BHDAM-R):")
    for c, t, b, _ in caps:
        print(f"  {c:<44} TLS={t:<8} BHDAM-R={b}")
    return rows


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def _fig_tls_overhead(rows, out):
    sizes = sorted({r["size_mb"] for r in rows})
    fig, ax = plt.subplots(figsize=(8, 5))
    for (k, n) in PROFILES:
        prof = f"{k}-of-{n}"
        ys = [next((r["bhdamr_throughput_MBps"] for r in rows
                    if r["size_mb"] == mb and r["profile"] == prof and r["aont"] is True), None)
              for mb in sizes]
        ax.plot(sizes, ys, marker="o", label=f"{prof} (AONT-RS)")
    tls = [next((r["tls_throughput_MBps"] for r in rows if r["size_mb"] == mb), None)
           for mb in sizes]
    ax.plot(sizes, tls, marker="s", color="gray", ls="--", label="TLS-like AEAD baseline")
    ax.set_xlabel("payload size (MB)"); ax.set_ylabel("throughput (MB/s)")
    ax.set_yscale("log")
    ax.set_title("Sealing throughput: BHDAM-R vs a TLS-like AEAD baseline")
    ax.legend(fontsize=8); ax.grid(alpha=.3, which="both")
    _save(fig, os.path.join(out, "fig_tls_overhead"))


def _fig_failure(rows, p_grid, profiles, out):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(p_grid, [r["tls_complete"] for r in rows], marker="s",
            color="tab:blue", label="TLS single channel (<=5 retries)")
    for (k, n) in profiles:
        ax.plot(p_grid, [r[f"bhdamr_{k}of{n}_complete"] for r in rows],
                marker="o", label=f"BHDAM-R {k}-of-{n}")
    ax.set_xlabel("per-channel failure probability p")
    ax.set_ylabel("probability of completing the transfer")
    ax.set_title("Transfer completion: BHDAM-R vs TLS with retransmission")
    ax.legend(fontsize=8); ax.grid(alpha=.3)
    _save(fig, os.path.join(out, "fig_failure_vs_tls"))


# --------------------------------------------------------------------------- #
def _add_common(p):
    p.add_argument("--out", default="out")
    p.add_argument("--sizes", type=int, nargs="+", default=[1, 10, 100])
    p.add_argument("--reps", type=int, default=20)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--trials", type=int, default=5000)
    p.add_argument("--profiles", type=str, nargs="+", default=None)


def _parse_profiles(args):
    if getattr(args, "profiles", None):
        args.profiles_parsed = [tuple(int(x) for x in tok.lower().split("-of-"))
                                for tok in args.profiles]
    else:
        args.profiles_parsed = PROFILES


def main():
    ap = argparse.ArgumentParser(description="BHDAM-R comparative evaluation")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("tls-overhead", "tls-real", "failure", "dispersed", "parity", "all-compare"):
        _add_common(sub.add_parser(name))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    _parse_profiles(args)
    if args.cmd == "tls-overhead":
        cmd_tls_overhead(args)
    elif args.cmd == "tls-real":
        cmd_tls_real(args)
    elif args.cmd == "failure":
        cmd_failure(args)
    elif args.cmd == "dispersed":
        cmd_dispersed(args)
    elif args.cmd == "parity":
        cmd_parity(args)
    elif args.cmd == "all-compare":
        cmd_tls_overhead(args)
        cmd_tls_real(args)
        cmd_failure(args)
        cmd_dispersed(args)
        cmd_parity(args)
        print("\nAll comparative experiments complete. Artefacts in:", args.out)


if __name__ == "__main__":
    main()
