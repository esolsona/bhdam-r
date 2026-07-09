#!/usr/bin/env python3
"""
Synthetic-but-representative health datasets for BHDAM-R benchmarking.

These are NOT real patient data. They reproduce the size, shape and structure of
common regulated health-data payloads so that timing and overhead figures are
credible, without distributing protected data. Generation is deterministic
(fixed seed) for reproducibility. Payload entropy does not affect the pipeline
(no compression is applied), so structured phantoms and real scans yield
equivalent cryptographic/coding cost at equal size.
"""
from __future__ import annotations

import io
import json
import numpy as np

_SEED = 20260709


def imaging_volume(slices: int = 40, dim: int = 512) -> bytes:
    """A CT/MRI-like int16 volume (slices x dim x dim). 40x512x512 ~= 20 MB.

    A simple analytic phantom (nested ellipsoids + gradient + mild noise) so the
    bytes resemble a reconstructed scan rather than white noise.
    """
    rng = np.random.default_rng(_SEED)
    yy, xx = np.mgrid[0:dim, 0:dim].astype(np.float32)
    cx = cy = dim / 2.0
    r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    base = np.clip(1200 - (r / dim) * 2400, -1000, 1200)  # HU-like range
    vol = np.empty((slices, dim, dim), dtype=np.int16)
    for z in range(slices):
        ell = 300 * np.sin(2 * np.pi * z / max(slices, 1)) * np.exp(-(r / (dim / 3)) ** 2)
        noise = rng.normal(0, 15, size=(dim, dim))
        vol[z] = np.clip(base + ell + noise, -1024, 3071).astype(np.int16)
    return vol.tobytes()


def omics_matrix(genes: int = 20000, samples: int = 60) -> bytes:
    """Gene-expression count matrix as TSV. 20000 x 60 ~= 8-10 MB of text."""
    rng = np.random.default_rng(_SEED + 1)
    counts = rng.negative_binomial(5, 0.3, size=(genes, samples))
    buf = io.StringIO()
    buf.write("gene_id\t" + "\t".join(f"S{i:03d}" for i in range(samples)) + "\n")
    # write in blocks for speed
    for g in range(genes):
        buf.write(f"ENSG{g:011d}\t" + "\t".join(map(str, counts[g].tolist())) + "\n")
    return buf.getvalue().encode()


def clinical_bundle(subjects: int = 500) -> bytes:
    """A clinical records bundle as JSON (labs, vitals, visits)."""
    rng = np.random.default_rng(_SEED + 2)
    records = []
    for s in range(subjects):
        records.append({
            "subject_id": f"SUBJ-{s:05d}",
            "arm": ["A", "B", "placebo"][s % 3],
            "vitals": {"hr": int(rng.integers(55, 95)),
                       "sbp": int(rng.integers(100, 150)),
                       "dbp": int(rng.integers(60, 95))},
            "labs": {k: round(float(rng.normal(m, sd)), 2) for k, m, sd in
                     [("alt", 25, 8), ("ast", 24, 7), ("crp", 3, 2),
                      ("hgb", 14, 1.5), ("wbc", 7, 2)]},
            "visits": [{"day": d, "dose_mg": int(rng.integers(0, 200))}
                       for d in range(0, 84, 7)],
        })
    return json.dumps({"study": "SYNTH-001", "records": records}).encode()


def realistic_datasets() -> dict[str, dict[str, bytes]]:
    """Return the three headline datasets, labelled by their actual size."""
    raw = {
        "imaging": ("scan_volume.raw", imaging_volume(slices=40, dim=512)),
        "omics": ("expression_matrix.tsv", omics_matrix(genes=20000, samples=160)),
        "clinical": ("clinical_bundle.json", clinical_bundle(subjects=3800)),
    }
    out = {}
    for kind, (fname, data) in raw.items():
        mb = len(data) / 1e6
        out[f"{kind}_{mb:.0f}MB"] = {fname: data}
    return out


def sized_blob(mb: int) -> dict[str, bytes]:
    """A structured payload of ~mb megabytes for the size-scaling benchmark."""
    rng = np.random.default_rng(_SEED + mb)
    n = mb * 1024 * 1024
    # low-entropy-ish structured bytes (repeated pattern + noise) at target size
    arr = (np.arange(n, dtype=np.uint8) + rng.integers(0, 32, size=n, dtype=np.uint8))
    return {f"payload_{mb}MB.bin": arr.tobytes()}


if __name__ == "__main__":
    for label, files in realistic_datasets().items():
        size = sum(len(v) for v in files.values())
        print(f"{label:16s} {size/1e6:7.2f} MB  ({list(files)[0]})")
