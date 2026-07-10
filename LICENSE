#!/usr/bin/env python3
"""
BHDAM-R experiment harness (paper artefact).

Subcommands (see `python experiments.py --help`):
  test       Security test suite (paper Section 14). Exit code 0 iff all pass.
  bench      Reproducible per-stage timing benchmark (warmup + repetitions,
             mean/std) over sizes x profiles x {AONT off/on} (Section 16).
  overhead   Storage overhead by k-of-n profile (Section 8.4).
  recover    Monte-Carlo recovery vs single-channel baseline, validated against
             the exact binomial model, over channel-failure probabilities.
  realistic  End-to-end run on representative imaging/omics/clinical datasets.
  all        Run everything, write figures + environment.json + summary.

Outputs (default ./out): CSV tables, PNG+PDF figures, environment.json.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import statistics
import sys
import time
from dataclasses import asdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey

import bhdam_r
from bhdam_r import Sender, Recipient, Evidence, ReconstructionError, erasure_decode
import datasets

PROFILES = [(4, 5), (4, 6), (6, 9), (7, 10)]
OUTDIR = "out"


# --------------------------------------------------------------------------- #
def parties():
    sk = Ed25519PrivateKey.generate()
    kem = X25519PrivateKey.generate()
    return (Sender("Hospital-Source", sk), sk.public_key(),
            Recipient("Partner-Lab", kem), kem.public_key())


def _fresh_ev(tid):
    return Evidence(transfer_id=tid, trust_state="R3")


def _channels(n):
    return [f"ch{i}" for i in range(n)]


# --------------------------------------------------------------------------- #
# Environment capture (reproducibility)
# --------------------------------------------------------------------------- #
def environment() -> dict:
    import cryptography
    import zfec
    return {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "cpu_count": os.cpu_count(),
        "libraries": {
            "cryptography": cryptography.__version__,
            "zfec": getattr(zfec, "__version__", "unknown"),
            "numpy": np.__version__,
            "matplotlib": matplotlib.__version__,
        },
        "primitives": {
            "aead": "AES-256-GCM (SP 800-38D)",
            "signature": "Ed25519 (RFC 8032)",
            "kem": "X25519 + HKDF-SHA256 (RFC 7748/5869)",
            "erasure": "Reed-Solomon via zfec",
            "aont": "AONT-RS (Rivest 1997; Resch & Plank FAST 2011)",
        },
    }


# --------------------------------------------------------------------------- #
# test  (Section 14)
# --------------------------------------------------------------------------- #
def cmd_test(args) -> bool:
    s, spub, r, rpub = parties()
    files = datasets.sized_blob(1)
    ch = _channels(6)

    def fresh(aont=True):
        return s.build_transfer(files, r.recipient_id, rpub, 4, 6, ch, use_aont=aont)

    results = []

    def check(name, ok):
        results.append((name, ok))
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")

    m, d, sh, sig, ev = fresh()
    try:
        r.receive(m, sig, spub, [(i, sh[i]) for i in (0, 1, 2, 3)], ev)
        check("reconstruct with exactly k valid shards", ev.trust_state == "R5")
    except ReconstructionError:
        check("reconstruct with exactly k valid shards", False)

    m, d, sh, sig, ev = fresh()
    r.receive(m, sig, spub, list(enumerate(sh)), ev)
    check("reconstruct with all n shards", ev.trust_state == "R5")

    m, d, sh, sig, ev = fresh()
    try:
        r.receive(m, sig, spub, [(i, sh[i]) for i in (0, 1, 2)], ev)
        check("fail when fewer than k shards", False)
    except ReconstructionError:
        check("fail when fewer than k shards", True)

    m, d, sh, sig, ev = fresh()
    t = bytearray(sh[1]); t[0] ^= 0xFF
    r.receive(m, sig, spub, [(0, sh[0]), (1, bytes(t)), (2, sh[2]),
                             (3, sh[3]), (4, sh[4])], ev)
    check("detect a modified shard (quarantined)",
          any(e.get("reason") == "hash_mismatch_tamper" for e in ev.events)
          and ev.trust_state == "R5")

    m1, d1, sh1, sig1, _ = fresh()
    m2, d2, sh2, sig2, _ = fresh()
    ev = _fresh_ev(m1.transfer_id)
    try:
        r.receive(m1, sig1, spub, [(0, sh1[0]), (1, sh1[1]), (2, sh1[2]),
                                   (3, sh2[3])], ev)
        check("detect replayed shard from another transfer", False)
    except ReconstructionError:
        check("detect replayed shard from another transfer",
              any(e.get("reason") in ("hash_mismatch_tamper",
                                      "replay_wrong_transfer") for e in ev.events))

    wrong = Recipient("attacker", X25519PrivateKey.generate())
    m, d, sh, sig, ev = fresh()
    try:
        wrong.receive(m, sig, spub, [(i, sh[i]) for i in (0, 1, 2, 3)], ev)
        check("fail when recipient key does not match", False)
    except ReconstructionError:
        check("fail when recipient key does not match", True)

    other = Ed25519PrivateKey.generate().public_key()
    m, d, sh, sig, ev = fresh()
    try:
        r.receive(m, sig, other, [(i, sh[i]) for i in (0, 1, 2, 3)], ev)
        check("reject manifest under wrong sender key", False)
    except ReconstructionError:
        check("reject manifest under wrong sender key", True)

    m, d, sh, sig, ev = fresh()
    try:
        erasure_decode([sh[0], sh[1], sh[2]], [0, 1, 2], m.k, m.n, m.orig_package_len)
        check("AONT/threshold: k-1 shards insufficient for package", False)
    except ValueError:
        check("AONT/threshold: k-1 shards insufficient for package", True)

    passed = sum(ok for _, ok in results)
    print(f"\n  {passed}/{len(results)} tests passed")
    _write_csv(os.path.join(args.out, "test_results.csv"),
               [{"test": n, "passed": ok} for n, ok in results])
    return passed == len(results)


# --------------------------------------------------------------------------- #
# bench  (Section 16) -- warmup + reps, per-stage mean/std
# --------------------------------------------------------------------------- #
STAGES_SEND = ["package_ms", "encrypt_ms", "aont_ms", "erasure_encode_ms",
               "shard_hash_ms"]
STAGES_RECV = ["shard_validate_hash_ms", "erasure_decode_ms", "aont_decode_ms",
               "decrypt_ms"]


def _agg(values):
    return (round(statistics.mean(values), 3),
            round(statistics.pstdev(values), 3) if len(values) > 1 else 0.0)


def _select_profiles(args):
    chosen = getattr(args, "profiles", None)
    if not chosen:
        return PROFILES
    out = []
    for spec in chosen:
        try:
            k, n = spec.lower().split("-of-")
            out.append((int(k), int(n)))
        except ValueError:
            raise SystemExit(f"bad --profiles value '{spec}', expected e.g. 4-of-6")
    return out


def _select_aont(args):
    mode = getattr(args, "aont", "both")
    return {"both": (False, True), "on": (True,), "off": (False,)}[mode]


def cmd_bench(args):
    s, spub, r, rpub = parties()
    sizes = args.sizes
    reps, warmup = args.reps, args.warmup
    profiles = _select_profiles(args)
    aont_modes = _select_aont(args)
    workers = getattr(args, "workers", 1)
    rows = []
    print(f"benchmark: sizes={sizes} MB, profiles={profiles}, "
          f"aont={aont_modes}, workers={workers}, reps={reps} (+{warmup} warmup)")
    for mb in sizes:
        payload = datasets.sized_blob(mb)
        for (k, n) in profiles:
            ch = _channels(n)
            for aont in aont_modes:
                # warmup
                for _ in range(warmup):
                    m, d, sh, sig, ev = s.build_transfer(
                        payload, r.recipient_id, rpub, k, n, ch,
                        use_aont=aont, workers=workers)
                    r.receive(m, sig, spub, [(i, sh[i]) for i in range(k)],
                              _fresh_ev(m.transfer_id), workers=workers)
                # measured
                acc = {st: [] for st in STAGES_SEND + STAGES_RECV}
                total_send, total_recv, overhead = [], [], None
                for _ in range(reps):
                    ts = {}
                    t0 = time.perf_counter()
                    m, d, sh, sig, ev = s.build_transfer(
                        payload, r.recipient_id, rpub, k, n, ch,
                        use_aont=aont, timings=ts, workers=workers)
                    total_send.append((time.perf_counter() - t0) * 1000)
                    tr = {}
                    t1 = time.perf_counter()
                    r.receive(m, sig, spub, [(i, sh[i]) for i in range(k)],
                              _fresh_ev(m.transfer_id), timings=tr, workers=workers)
                    total_recv.append((time.perf_counter() - t1) * 1000)
                    for st in STAGES_SEND:
                        acc[st].append(ts.get(st, 0.0))
                    for st in STAGES_RECV:
                        acc[st].append(tr.get(st, 0.0))
                    overhead = sum(len(x) for x in sh) / (mb * 1024 * 1024)
                row = {"size_mb": mb, "profile": f"{k}-of-{n}", "k": k, "n": n,
                       "aont": aont, "overhead_ratio": round(overhead, 3)}
                for st in STAGES_SEND + STAGES_RECV:
                    mean, sd = _agg(acc[st])
                    row[st + "_mean"] = mean
                    row[st + "_std"] = sd
                sm, ss = _agg(total_send); rm, rs = _agg(total_recv)
                row.update({"send_total_ms_mean": sm, "send_total_ms_std": ss,
                            "recv_total_ms_mean": rm, "recv_total_ms_std": rs,
                            "throughput_MBps_send": round(mb / (sm / 1000), 2)})
                rows.append(row)
                print(f"  {mb:>4}MB {f'{k}-of-{n}':>8} aont={str(aont):<5} "
                      f"send={sm:7.1f}±{ss:<5.1f}ms  recv={rm:6.1f}±{rs:<4.1f}ms  "
                      f"ovh={overhead:.2f}")
    _write_csv(os.path.join(args.out, "benchmark.csv"), rows)
    _fig_stage_breakdown(rows, args.out)
    _fig_throughput(rows, args.out)
    return rows


# --------------------------------------------------------------------------- #
# overhead  (deterministic)
# --------------------------------------------------------------------------- #
def cmd_overhead(args):
    rows = []
    for (k, n) in PROFILES:
        rows.append({"profile": f"{k}-of-{n}", "k": k, "n": n,
                     "tolerated_loss": n - k,
                     "overhead_ratio": round(n / k, 3),
                     "overhead_pct": round((n / k - 1) * 100, 1)})
    for row in rows:
        print(f"  {row['profile']:>8}  tolerate {row['tolerated_loss']} loss  "
              f"overhead {row['overhead_pct']:.1f}%")
    _write_csv(os.path.join(args.out, "overhead.csv"), rows)
    return rows


# --------------------------------------------------------------------------- #
# recover  -- Monte-Carlo vs single-channel, validated vs binomial
# --------------------------------------------------------------------------- #
def _binom_at_least_k(n, k, p_fail):
    """Exact P(at least k of n channels deliver), each fails independently p."""
    q = 1 - p_fail
    return sum(math.comb(n, i) * q ** i * p_fail ** (n - i) for i in range(k, n + 1))


def cmd_recover(args):
    s, spub, r, rpub = parties()
    files = datasets.sized_blob(1)   # small, many reconstructions
    rng = np.random.default_rng(20260709)
    p_grid = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]
    M = args.trials
    rows = []
    schemes = [("single-channel", 1, 1)] + [(f"{k}-of-{n}", k, n) for (k, n) in PROFILES]

    print(f"recovery Monte-Carlo: {M} trials/point, validated vs binomial")
    for label, k, n in schemes:
        ch = _channels(n)
        for p in p_grid:
            if n == 1:
                # single-channel baseline: one route, no erasure coding.
                # Delivery succeeds iff the single channel does not fail.
                success = int(np.sum(rng.random(M) >= p))
            else:
                m, d, sh, sig, ev = s.build_transfer(
                    files, r.recipient_id, rpub, k, n, ch, use_aont=True)
                success = 0
                for _ in range(M):
                    fail_mask = rng.random(n) < p
                    survivors = [i for i in range(n) if not fail_mask[i]]
                    if len(survivors) >= k:
                        arriving = [(i, sh[i]) for i in survivors[:k]]
                        try:
                            r.receive(m, sig, spub, arriving,
                                      _fresh_ev(m.transfer_id))
                            success += 1
                        except ReconstructionError:
                            pass  # would indicate a real guarantee failure
                    # else: < k survivors -> genuine unrecoverable transfer
            emp = success / M
            analytic = 1 - p if n == 1 else _binom_at_least_k(n, k, p)
            rows.append({"scheme": label, "k": k, "n": n, "p_fail": p,
                         "empirical_recovery": round(emp, 4),
                         "analytic_recovery": round(analytic, 4),
                         "abs_error": round(abs(emp - analytic), 4)})
        print(f"  {label:>14}: max|emp-analytic| = "
              f"{max(x['abs_error'] for x in rows if x['scheme']==label):.3f}")
    _write_csv(os.path.join(args.out, "recovery.csv"), rows)
    _fig_recovery(rows, p_grid, schemes, args.out)
    return rows


# --------------------------------------------------------------------------- #
# realistic  -- headline end-to-end numbers on representative datasets
# --------------------------------------------------------------------------- #
def cmd_realistic(args):
    s, spub, r, rpub = parties()
    k, n = 4, 6
    ch = _channels(n)
    rows = []
    print(f"realistic datasets, {k}-of-{n}, AONT-RS on, {args.reps} reps:")
    for label, files in datasets.realistic_datasets().items():
        size_mb = sum(len(v) for v in files.values()) / 1e6
        for _ in range(args.warmup):
            m, d, sh, sig, ev = s.build_transfer(files, r.recipient_id, rpub,
                                                 k, n, ch, use_aont=True)
        sends, recvs = [], []
        for _ in range(args.reps):
            ts, tr = {}, {}
            t0 = time.perf_counter()
            m, d, sh, sig, ev = s.build_transfer(files, r.recipient_id, rpub,
                                                 k, n, ch, use_aont=True, timings=ts)
            sends.append((time.perf_counter() - t0) * 1000)
            # simulate 2 lost channels (Section 15): drop shards 3 and 4
            arriving = [(i, sh[i]) for i in (0, 1, 2, 5)]
            t1 = time.perf_counter()
            r.receive(m, sig, spub, arriving, _fresh_ev(m.transfer_id), timings=tr)
            recvs.append((time.perf_counter() - t1) * 1000)
        sm, ss = _agg(sends); rm, rs = _agg(recvs)
        row = {"dataset": label, "size_mb": round(size_mb, 2),
               "send_ms_mean": sm, "send_ms_std": ss,
               "recv_ms_mean": rm, "recv_ms_std": rs,
               "send_throughput_MBps": round(size_mb / (sm / 1000), 1),
               "recovered_from": "4 of 6 channels (2 lost)"}
        rows.append(row)
        print(f"  {label:>14} {size_mb:6.2f}MB  send {sm:7.1f}±{ss:<5.1f}ms "
              f"({row['send_throughput_MBps']:.1f} MB/s)  recv {rm:6.1f}ms")
        # emit the R5 evidence package for the first (imaging) dataset
        if not os.path.exists(os.path.join(args.out, "evidence_package.json")):
            evx = _fresh_ev(m.transfer_id)
            r.receive(m, sig, spub, [(i, sh[i]) for i in (0, 1, 2, 5)], evx)
            with open(os.path.join(args.out, "evidence_package.json"), "w") as f:
                json.dump({"manifest": json.loads(m.canonical_bytes()),
                           "signature_hex": sig.hex(),
                           "evidence": evx.__dict__}, f, indent=2)
            print("    wrote " + os.path.join(args.out, "evidence_package.json"))
    _write_csv(os.path.join(args.out, "realistic.csv"), rows)
    return rows


# --------------------------------------------------------------------------- #
# scaling  -- shard-hashing speedup vs worker threads
# --------------------------------------------------------------------------- #
def cmd_scaling(args):
    s, spub, r, rpub = parties()
    k, n = args.k, args.n
    ch = _channels(n)
    mb = args.size
    payload = datasets.sized_blob(mb)
    worker_grid = args.workers_grid
    rows = []
    print(f"scaling: {mb} MB, {k}-of-{n}, AONT on, "
          f"workers={worker_grid}, reps={args.reps}")
    for w in worker_grid:
        # warmup
        for _ in range(args.warmup):
            m, d, sh, sig, ev = s.build_transfer(payload, r.recipient_id, rpub,
                                                 k, n, ch, use_aont=True, workers=w)
            r.receive(m, sig, spub, [(i, sh[i]) for i in range(k)],
                      _fresh_ev(m.transfer_id), workers=w)
        hsh, vld = [], []
        for _ in range(args.reps):
            ts, tr = {}, {}
            m, d, sh, sig, ev = s.build_transfer(payload, r.recipient_id, rpub,
                                                 k, n, ch, use_aont=True,
                                                 timings=ts, workers=w)
            r.receive(m, sig, spub, [(i, sh[i]) for i in range(k)],
                      _fresh_ev(m.transfer_id), timings=tr, workers=w)
            hsh.append(ts.get("shard_hash_ms", 0.0))
            vld.append(tr.get("shard_validate_hash_ms", 0.0))
        hm, hs = _agg(hsh); vm, vs = _agg(vld)
        rows.append({"workers": w, "shard_hash_ms_mean": hm, "shard_hash_ms_std": hs,
                     "validate_hash_ms_mean": vm, "validate_hash_ms_std": vs})
        print(f"  workers={w:>2}  shard_hash={hm:7.1f}±{hs:<4.1f}ms  "
              f"validate_hash={vm:6.1f}±{vs:<4.1f}ms")
    # speedup vs workers=1
    base = rows[0]["shard_hash_ms_mean"] or 1.0
    for row in rows:
        row["shard_hash_speedup"] = round(base / (row["shard_hash_ms_mean"] or 1e-9), 2)
    _write_csv(os.path.join(args.out, "scaling.csv"), rows)
    _fig_scaling(rows, mb, k, n, args.out)
    return rows


def _fig_scaling(rows, mb, k, n, out):
    ws = [r["workers"] for r in rows]
    hh = [r["shard_hash_ms_mean"] for r in rows]
    vv = [r["validate_hash_ms_mean"] for r in rows]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.plot(ws, hh, marker="o", label="sender shard hashing")
    ax1.plot(ws, vv, marker="s", label="recipient validation hashing")
    ax1.set_xlabel("worker threads"); ax1.set_ylabel("time (ms)")
    ax1.set_title(f"Shard-hashing latency ({mb} MB, {k}-of-{n})")
    ax1.legend(fontsize=8); ax1.grid(True, alpha=0.3)
    sp = [r["shard_hash_speedup"] for r in rows]
    ax2.plot(ws, sp, marker="o", color="tab:green", label="measured speedup")
    ax2.plot(ws, ws, linestyle="--", color="grey", label="ideal (linear)")
    ax2.set_xlabel("worker threads"); ax2.set_ylabel("speedup vs 1 thread")
    ax2.set_title("Sender shard-hashing speedup")
    ax2.legend(fontsize=8); ax2.grid(True, alpha=0.3)
    _save(fig, os.path.join(out, "fig_scaling"))


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def _save(fig, path_noext):
    fig.savefig(path_noext + ".png", dpi=150, bbox_inches="tight")
    fig.savefig(path_noext + ".pdf", bbox_inches="tight")
    plt.close(fig)


def _fig_stage_breakdown(rows, out):
    # pick largest size, AONT on, stacked bar per profile
    mb = max(r["size_mb"] for r in rows)
    sub = [r for r in rows if r["size_mb"] == mb and r["aont"]]
    sub.sort(key=lambda r: r["n"])
    if not sub:
        return  # no AONT runs to break down
    labels = [r["profile"] for r in sub]
    stages = STAGES_SEND + STAGES_RECV
    nice = ["package", "encrypt", "AONT", "RS encode", "shard hash",
            "validate hash", "RS decode", "AONT decode", "decrypt"]
    fig, ax = plt.subplots(figsize=(7, 4))
    bottom = np.zeros(len(sub))
    for st, name in zip(stages, nice):
        vals = np.array([r[st + "_mean"] for r in sub])
        ax.bar(labels, vals, bottom=bottom, label=name)
        bottom += vals
    ax.set_ylabel("time (ms)")
    ax.set_xlabel("k-of-n profile")
    ax.set_title(f"Per-stage cost, {mb} MB payload (AONT-RS on)")
    ax.legend(fontsize=8, ncol=2)
    _save(fig, os.path.join(out, "fig_stage_breakdown"))


def _fig_throughput(rows, out):
    fig, ax = plt.subplots(figsize=(7, 4))
    cmap = plt.get_cmap("tab10")
    for idx, (k, n) in enumerate(PROFILES):
        color = cmap(idx)
        for aont in (False, True):
            pts = sorted([r for r in rows if r["k"] == k and r["n"] == n
                          and r["aont"] == aont], key=lambda r: r["size_mb"])
            if not pts:
                continue
            xs = [r["size_mb"] for r in pts]
            ys = [r["throughput_MBps_send"] for r in pts]
            ax.plot(xs, ys, marker="o", color=color,
                    linestyle="-" if aont else "--",
                    label=f"{k}-of-{n} {'AONT' if aont else 'base'}")
    ax.set_xlabel("payload size (MB)")
    ax.set_ylabel("sender throughput (MB/s)")
    ax.set_title("Sender-side throughput vs payload size "
                 "(solid = AONT-RS, dashed = base)")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)
    _save(fig, os.path.join(out, "fig_throughput"))


def _fig_recovery(rows, p_grid, schemes, out):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for label, k, n in schemes:
        emp = [next(r["empirical_recovery"] for r in rows
                    if r["scheme"] == label and r["p_fail"] == p) for p in p_grid]
        ana = [next(r["analytic_recovery"] for r in rows
                    if r["scheme"] == label and r["p_fail"] == p) for p in p_grid]
        line, = ax.plot(p_grid, ana, linestyle="-", linewidth=1.5,
                        label=f"{label} (model)")
        ax.plot(p_grid, emp, linestyle="none", marker="o", markersize=4,
                color=line.get_color())
    ax.set_xlabel("per-channel failure probability p")
    ax.set_ylabel("transfer recovery probability")
    ax.set_title("Recovery vs single-channel (lines: binomial model; dots: Monte-Carlo)")
    ax.set_ylim(-0.02, 1.02)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    _save(fig, os.path.join(out, "fig_recovery"))


# --------------------------------------------------------------------------- #
def _write_csv(path, rows):
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader(); w.writerows(rows)
    print(f"    wrote {path}")


def cmd_all(args):
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "environment.json"), "w") as f:
        json.dump(environment(), f, indent=2)
    print("== TEST =="); ok = cmd_test(args)
    print("\n== OVERHEAD =="); cmd_overhead(args)
    print("\n== BENCH =="); cmd_bench(args)
    print("\n== RECOVER =="); cmd_recover(args)
    print("\n== REALISTIC =="); cmd_realistic(args)
    print("\nAll experiments complete. Artefacts in:", args.out)
    return ok


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description="BHDAM-R experiment harness")
    ap.add_argument("--out", default=OUTDIR, help="output directory")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("test")
    b = sub.add_parser("bench")
    b.add_argument("--sizes", type=int, nargs="+", default=[1, 5, 25, 100])
    b.add_argument("--reps", type=int, default=7)
    b.add_argument("--warmup", type=int, default=2)
    b.add_argument("--profiles", nargs="+", default=None,
                   help="e.g. --profiles 4-of-6 6-of-9 (default: all)")
    b.add_argument("--aont", choices=["both", "on", "off"], default="both")
    b.add_argument("--workers", type=int, default=1,
                   help="threads for per-shard hashing (default 1)")
    sc = sub.add_parser("scaling")
    sc.add_argument("--size", type=int, default=256, help="payload MB")
    sc.add_argument("--k", type=int, default=6)
    sc.add_argument("--n", type=int, default=9)
    sc.add_argument("--workers-grid", type=int, nargs="+",
                    default=[1, 2, 4, 8], dest="workers_grid")
    sc.add_argument("--reps", type=int, default=5)
    sc.add_argument("--warmup", type=int, default=1)
    sub.add_parser("overhead")
    rec = sub.add_parser("recover")
    rec.add_argument("--trials", type=int, default=200)
    rl = sub.add_parser("realistic")
    rl.add_argument("--reps", type=int, default=5)
    rl.add_argument("--warmup", type=int, default=1)
    a = sub.add_parser("all")
    a.add_argument("--sizes", type=int, nargs="+", default=[1, 5, 25, 100])
    a.add_argument("--reps", type=int, default=7)
    a.add_argument("--warmup", type=int, default=2)
    a.add_argument("--trials", type=int, default=200)

    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    # defaults for cross-command attrs
    for attr, val in (("reps", 5), ("warmup", 1), ("trials", 200),
                      ("sizes", [1, 5, 25, 100])):
        if not hasattr(args, attr):
            setattr(args, attr, val)

    if args.cmd == "test":
        sys.exit(0 if cmd_test(args) else 1)
    elif args.cmd == "bench":
        cmd_bench(args)
    elif args.cmd == "overhead":
        cmd_overhead(args)
    elif args.cmd == "recover":
        cmd_recover(args)
    elif args.cmd == "scaling":
        cmd_scaling(args)
    elif args.cmd == "realistic":
        cmd_realistic(args)
    elif args.cmd == "all":
        sys.exit(0 if cmd_all(args) else 1)


if __name__ == "__main__":
    main()
