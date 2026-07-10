"""BHDAM-R visual demonstration.

Transfers a synthetic CT-like image (Shepp-Logan phantom) with a 4-of-6
profile and AONT-RS enabled, loses two channels in transit, reconstructs the
image byte-for-byte from the four surviving shards, verifies SHA-256 equality,
shows that three shards are insufficient, and emits the signed R5 evidence
package.  Output: demo_figure.png/.pdf + demo_evidence.json.

Run:  python demo_image_transfer.py [outdir]
"""
import io, json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey
from bhdam_r import Sender, Recipient, Evidence, sha256

OUT = sys.argv[1] if len(sys.argv) > 1 else "demo_out"
os.makedirs(OUT, exist_ok=True)
K, N = 4, 6
LOST = {1, 4}                       # channels that never arrive
SURVIVING = [i for i in range(N) if i not in LOST]


def shepp_logan(size=512):
    """Standard Shepp-Logan head phantom (synthetic CT slice)."""
    E = [  # intensity, a, b, x0, y0, phi(deg)
        (1.00, .69, .92, 0, 0, 0), (-.80, .6624, .8740, 0, -.0184, 0),
        (-.20, .1100, .3100, .22, 0, -18), (-.20, .1600, .4100, -.22, 0, 18),
        (.10, .2100, .2500, 0, .35, 0), (.10, .0460, .0460, 0, .1, 0),
        (.10, .0460, .0460, 0, -.1, 0), (.10, .0460, .0230, -.08, -.605, 0),
        (.10, .0230, .0230, 0, -.606, 0), (.10, .0230, .0460, .06, -.605, 0),
    ]
    y, x = np.mgrid[1:-1:size * 1j, -1:1:size * 1j]
    img = np.zeros((size, size))
    for A, a, b, x0, y0, phi in E:
        t = np.deg2rad(phi)
        xr = (x - x0) * np.cos(t) + (y - y0) * np.sin(t)
        yr = -(x - x0) * np.sin(t) + (y - y0) * np.cos(t)
        img[(xr / a) ** 2 + (yr / b) ** 2 <= 1] += A
    img = np.clip(img, 0, 1)
    rng = np.random.default_rng(20260709)          # realistic acquisition noise
    img = np.clip(img + rng.normal(0, .02, img.shape), 0, 1)
    return (img * 255).astype(np.uint8)


def png_bytes(arr):
    buf = io.BytesIO(); Image.fromarray(arr, mode="L").save(buf, "PNG")
    return buf.getvalue()


def shard_tile(shard, size=64):
    """Render the first size*size bytes of a shard as a grayscale noise tile."""
    need = size * size
    raw = (shard * (need // len(shard) + 1))[:need]
    return np.frombuffer(raw, dtype=np.uint8).reshape(size, size)


def main():
    # ---- the "dataset": one synthetic CT slice ----
    ct = shepp_logan()
    original = png_bytes(ct)
    files = {"ct_slice.png": original}
    h_orig = sha256(original)
    print(f"original : ct_slice.png  {len(original):,} bytes  sha256={h_orig[:16]}…")

    # ---- sender: seal + disperse over 6 channels ----
    sk, kem = Ed25519PrivateKey.generate(), X25519PrivateKey.generate()
    sender = Sender("Hospital-Source", sk)
    recipient = Recipient("Biotech-Lab", kem)
    manifest, descriptors, shards, sig, _ = sender.build_transfer(
        files, recipient.recipient_id, kem.public_key(), K, N,
        [f"ch{i}" for i in range(N)], use_aont=True)
    print(f"dispersed: {N} shards of {len(shards[0]):,} bytes "
          f"(k={K}, AONT-RS on) — channels ch1 and ch4 lost in transit")

    # ---- recipient: reconstruct from the 4 surviving shards ----
    ev = Evidence(transfer_id=manifest.transfer_id, trust_state="R3")
    plaintext = recipient.receive(manifest, sig, sk.public_key(),
                                  [(i, shards[i]) for i in SURVIVING], ev)
    recovered = plaintext[8:]                     # strip 8-byte length prefix
    h_rec = sha256(recovered)
    assert recovered == original and ev.trust_state == "R5"
    print(f"recovered: {len(recovered):,} bytes from {len(SURVIVING)}/{N} shards  "
          f"sha256={h_rec[:16]}…  -> BYTE-IDENTICAL, trust state {ev.trust_state}")

    # ---- negative proof: k-1 shards must fail ----
    try:
        recipient.receive(manifest, sig, sk.public_key(),
                          [(i, shards[i]) for i in SURVIVING[:K - 1]],
                          Evidence(transfer_id=manifest.transfer_id, trust_state="R3"))
        raise SystemExit("ERROR: reconstruction with k-1 shards should fail")
    except Exception as e:
        print(f"negative : {K - 1} shards -> reconstruction FAILED as designed ({e})")

    # ---- signed evidence package ----
    evidence = {"manifest": json.loads(manifest.canonical_bytes()),
                "signature_hex": sig.hex(), "evidence": ev.__dict__}
    with open(f"{OUT}/demo_evidence.json", "w") as f:
        json.dump(evidence, f, indent=2)
    Image.fromarray(ct).save(f"{OUT}/ct_original.png")
    with open(f"{OUT}/ct_recovered.png", "wb") as f:
        f.write(recovered)

    # ---- figure ----
    fig = plt.figure(figsize=(13, 5.6))
    gs = fig.add_gridspec(2, 5, width_ratios=[2.2, 1, 1, 1, 2.2],
                          hspace=.35, wspace=.15)
    axo = fig.add_subplot(gs[:, 0])
    axo.imshow(ct, cmap="gray"); axo.set_title("1. Original dataset\n(synthetic CT slice, "
                                               f"{len(original)//1024} kB)", fontsize=10)
    axo.axis("off")
    for j in range(N):
        ax = fig.add_subplot(gs[j // 3, 1 + j % 3])
        ax.imshow(shard_tile(shards[j]), cmap="gray")
        ax.set_xticks([]); ax.set_yticks([])
        if j in LOST:
            ax.plot([0, 63], [0, 63], "r-", lw=3); ax.plot([0, 63], [63, 0], "r-", lw=3)
            ax.set_title(f"ch{j} — LOST", fontsize=9, color="crimson")
            for s in ax.spines.values(): s.set_edgecolor("crimson"); s.set_linewidth(2)
        else:
            ax.set_title(f"ch{j}", fontsize=9)
    axr = fig.add_subplot(gs[:, 4])
    axr.imshow(np.array(Image.open(io.BytesIO(recovered))), cmap="gray")
    axr.set_title(f"3. Reconstructed from {len(SURVIVING)} of {N} shards\n"
                  "SHA-256 identical — trust state R5", fontsize=10, color="darkgreen")
    axr.axis("off")
    fig.text(.5, .965, "BHDAM-R proof of transfer — 4-of-6 erasure-coded, AES-256-GCM + AONT-RS",
             ha="center", fontsize=12, weight="bold")
    fig.text(.435, .5, "2. Each channel carries one shard —\nindistinguishable from random noise",
             ha="center", fontsize=9)
    fig.text(.5, .045, f"sha256(original)  = {h_orig}\nsha256(recovered) = {h_rec}",
             ha="center", fontsize=8, family="monospace")
    fig.text(.5, .005, "with only 3 shards (k−1), reconstruction fails — verified",
             ha="center", fontsize=9, style="italic")
    fig.savefig(f"{OUT}/demo_figure.png", dpi=150, bbox_inches="tight")
    fig.savefig(f"{OUT}/demo_figure.pdf", bbox_inches="tight")
    print(f"wrote     {OUT}/demo_figure.png|pdf, demo_evidence.json, ct_original.png, ct_recovered.png")


if __name__ == "__main__":
    main()
