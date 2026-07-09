#!/usr/bin/env python3
"""
BHDAM-R -- Biotech Health Data Resilient Replication Model
Proof-of-Concept reference implementation.

This module implements the transfer lifecycle described in the BHDAM-R paper
(Sections 8.1-8.6), the transfer trust states R0-R5 (Section 9), the security
test suite (Section 14) and the evaluation metrics (Section 16). It is meant to
accompany the paper as reproducible artefact evidence, not as production code.

Cryptographic composition (Section 8.3 / 10.1):

    dataset
      -> canonicalise + per-file SHA-256 + Merkle root      (manifest, Sec 8.2)
      -> AES-256-GCM with fresh random DEK                   (AEAD, recipient-only)
      -> [optional] AONT package transform over ciphertext   (Sec 3 / point 3)
      -> k-of-n Reed-Solomon erasure coding (zfec)           (Sec 8.4)
      -> Ed25519 signature over manifest + shard descriptors (Sec 8.5)
      -> DEK wrapped for recipient via X25519 ECIES          (Sec 8.3)

Threat-model note (point 2): confidentiality holds for any adversary controlling
t <= n channels, because dispersal is applied to AEAD ciphertext under a fresh
random DEK; the threshold k governs *availability*, not confidentiality. When the
AONT stage is enabled (AONT-RS, Resch & Plank, FAST 2011), the threshold ALSO
becomes a confidentiality boundary: k-1 shards reveal nothing, even about the
ciphertext, reducing dependence on DEK secrecy.

Primitives / references:
  AES-GCM            NIST SP 800-38D
  Ed25519            RFC 8032
  X25519 + HKDF      RFC 7748 / RFC 5869
  Reed-Solomon       Reed & Solomon 1960; zfec (Tahoe-LAFS)
  AONT / package     Rivest 1997; AONT-RS: Resch & Plank, FAST 2011
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, asdict
from typing import Optional

import zfec
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)
from cryptography.hazmat.primitives.asymmetric.x25519 import (
    X25519PrivateKey,
    X25519PublicKey,
)
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives import hashes, serialization


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_many(items: list[bytes], workers: int = 1) -> list[str]:
    """Hash a list of byte strings, optionally across threads.

    hashlib releases the GIL for inputs larger than a few KB, so threads give
    real parallelism on multiple cores without copying shard bytes between
    processes. Order of results matches the input order.
    """
    if workers <= 1 or len(items) <= 1:
        return [sha256(b) for b in items]
    with ThreadPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(sha256, items))


def merkle_root(leaves: list[bytes]) -> str:
    """Binary Merkle root over pre-hashed leaves (Section 8.2)."""
    if not leaves:
        return sha256(b"")
    layer = [hashlib.sha256(leaf).digest() for leaf in leaves]
    while len(layer) > 1:
        if len(layer) % 2 == 1:
            layer.append(layer[-1])  # duplicate last (standard padding)
        layer = [
            hashlib.sha256(layer[i] + layer[i + 1]).digest()
            for i in range(0, len(layer), 2)
        ]
    return layer[0].hex()


class _Timer:
    """Lightweight per-stage timer. If `sink` is None, timing is a no-op."""
    def __init__(self, sink: Optional[dict]):
        self.sink = sink
        self._t0 = None

    def start(self):
        if self.sink is not None:
            self._t0 = time.perf_counter()

    def stop(self, key: str):
        if self.sink is not None and self._t0 is not None:
            self.sink[key] = self.sink.get(key, 0.0) + (
                time.perf_counter() - self._t0) * 1000.0


# --------------------------------------------------------------------------- #
# AONT package transform (canonical AONT-RS, Resch & Plank 2011)
# --------------------------------------------------------------------------- #
def aont_encode(data: bytes) -> bytes:
    """
    All-or-nothing transform over `data`.
    Package = AES-CTR(K, data) || (K XOR SHA256(ciphertext)).
    Recovering K requires the ENTIRE ciphertext, so any missing block => nothing.
    K is internal and keyless from the caller's perspective.
    """
    k_internal = os.urandom(32)
    # AES-CTR via AESGCM is not CTR; use AES in CTR mode explicitly:
    from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

    nonce = b"\x00" * 16  # deterministic; K is fresh so (K, nonce) never repeats
    enc = Cipher(algorithms.AES(k_internal), modes.CTR(nonce)).encryptor()
    ct = enc.update(data) + enc.finalize()
    digest = hashlib.sha256(ct).digest()
    difference = bytes(a ^ b for a, b in zip(k_internal, digest))
    return ct + difference  # last 32 bytes = difference block


def aont_decode(package: bytes) -> bytes:
    from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

    ct, difference = package[:-32], package[-32:]
    digest = hashlib.sha256(ct).digest()
    k_internal = bytes(a ^ b for a, b in zip(difference, digest))
    nonce = b"\x00" * 16
    dec = Cipher(algorithms.AES(k_internal), modes.CTR(nonce)).decryptor()
    return dec.update(ct) + dec.finalize()


# --------------------------------------------------------------------------- #
# Erasure coding (k-of-n, systematic Reed-Solomon via zfec)  -- Section 8.4
# --------------------------------------------------------------------------- #
def erasure_encode(data: bytes, k: int, n: int) -> tuple[list[bytes], int]:
    """Return (n shares, original_length). Any k shares reconstruct `data`."""
    orig_len = len(data)
    pad = (-orig_len) % k
    padded = data + b"\x00" * pad
    block = len(padded) // k
    blocks = [padded[i * block:(i + 1) * block] for i in range(k)]
    shares = zfec.Encoder(k, n).encode(blocks)  # list of n byte strings
    return shares, orig_len


def erasure_decode(
    shares: list[bytes], sharenums: list[int], k: int, n: int, orig_len: int
) -> bytes:
    if len(shares) < k:
        raise ValueError(f"need {k} shares, got {len(shares)}")
    dec = zfec.Decoder(k, n)
    blocks = dec.decode(shares[:k], sharenums[:k])
    return b"".join(blocks)[:orig_len]


# --------------------------------------------------------------------------- #
# Data classes
# --------------------------------------------------------------------------- #
@dataclass
class ShardDescriptor:
    shard_id: str
    transfer_id: str
    sharenum: int
    k: int
    n: int
    size: int
    sha256: str
    channel: str


@dataclass
class Manifest:
    transfer_id: str
    dataset_version: str
    sender_id: str
    recipient_id: str
    created_utc: float
    files: list[dict]              # inventory: name, size, sha256
    merkle_root: str
    k: int
    n: int
    aont: bool
    package_sha256: str           # hash of the (optionally AONT'd) ciphertext package
    orig_package_len: int
    aead_alg: str
    # recipient-only DEK protection (X25519 ECIES):
    dek_wrap: dict                # ephemeral_pub, nonce, ciphertext (all hex)
    aead_nonce: str              # hex
    shards: list[dict]           # ShardDescriptor dicts

    def canonical_bytes(self) -> bytes:
        return json.dumps(asdict(self), sort_keys=True, separators=(",", ":")).encode()


@dataclass
class Evidence:
    """R5 evidence package (Section 9 / 13)."""
    transfer_id: str
    trust_state: str
    events: list[dict] = field(default_factory=list)

    def log(self, event: str, **kw):
        self.events.append({"t": round(time.time(), 3), "event": event, **kw})


# --------------------------------------------------------------------------- #
# Sender  (R0 -> R3)
# --------------------------------------------------------------------------- #
class Sender:
    def __init__(self, sender_id: str, signing_key: Ed25519PrivateKey):
        self.sender_id = sender_id
        self.signing_key = signing_key

    def build_transfer(
        self,
        files: dict[str, bytes],      # filename -> content
        recipient_id: str,
        recipient_pub: X25519PublicKey,
        k: int,
        n: int,
        channels: list[str],
        dataset_version: str = "1.0",
        use_aont: bool = False,
        timings: Optional[dict] = None,
        workers: int = 1,
    ) -> tuple[Manifest, list[ShardDescriptor], list[bytes], bytes, Evidence]:
        assert n <= 256 and 0 < k < n, "zfec requires 0<k<n<=256"
        assert len(channels) == n, "one channel per shard"
        _t = _Timer(timings)
        tid = str(uuid.uuid4())
        ev = Evidence(transfer_id=tid, trust_state="R0")

        # R1: canonicalise + manifest (Section 8.2)
        _t.start()
        inventory, leaves = [], []
        for name in sorted(files):
            content = files[name]
            h = sha256(content)
            inventory.append({"name": name, "size": len(content), "sha256": h})
            leaves.append(bytes.fromhex(h))
        # deterministic package: length-prefixed sorted files
        package = b"".join(
            len(files[n_]).to_bytes(8, "big") + files[n_] for n_ in sorted(files)
        )
        mroot = merkle_root(leaves)
        _t.stop("package_ms")
        ev.trust_state = "R1"; ev.log("packaged", files=len(files), merkle_root=mroot)

        # R2: encrypt with fresh DEK (AES-256-GCM), then optional AONT
        _t.start()
        dek = AESGCM.generate_key(bit_length=256)
        aead_nonce = os.urandom(12)
        ciphertext = AESGCM(dek).encrypt(aead_nonce, package, tid.encode())
        _t.stop("encrypt_ms")
        _t.start()
        transfer_package = aont_encode(ciphertext) if use_aont else ciphertext
        _t.stop("aont_ms")
        pkg_hash = sha256(transfer_package)

        # wrap DEK for recipient (X25519 ECIES): recipient-only decryption
        dek_wrap = self._wrap_dek(dek, recipient_pub)
        ev.trust_state = "R2"; ev.log("sealed", aont=use_aont, aead="AES-256-GCM")

        # R3: erasure-code the package into n shards (Section 8.4)
        _t.start()
        shares, orig_len = erasure_encode(transfer_package, k, n)
        _t.stop("erasure_encode_ms")
        # Per-shard hashing is embarrassingly parallel across the n shards.
        _t.start()
        share_hashes = _sha256_many(shares, workers)
        _t.stop("shard_hash_ms")
        descriptors, shard_dicts = [], []
        for i, share in enumerate(shares):
            d = ShardDescriptor(
                shard_id=f"{tid}:{i}",
                transfer_id=tid,
                sharenum=i,
                k=k, n=n,
                size=len(share),
                sha256=share_hashes[i],
                channel=channels[i],
            )
            descriptors.append(d)
            shard_dicts.append(asdict(d))

        manifest = Manifest(
            transfer_id=tid,
            dataset_version=dataset_version,
            sender_id=self.sender_id,
            recipient_id=recipient_id,
            created_utc=time.time(),
            files=inventory,
            merkle_root=mroot,
            k=k, n=n,
            aont=use_aont,
            package_sha256=pkg_hash,
            orig_package_len=orig_len,
            aead_alg="AES-256-GCM",
            dek_wrap=dek_wrap,
            aead_nonce=aead_nonce.hex(),
            shards=shard_dicts,
        )
        # Sign the manifest (descriptors are inside it -> single signature covers all)
        signature = self.signing_key.sign(manifest.canonical_bytes())
        ev.trust_state = "R3"
        ev.log("dispersed", n=n, k=k, channels=channels)
        return manifest, descriptors, shares, signature, ev

    @staticmethod
    def _wrap_dek(dek: bytes, recipient_pub: X25519PublicKey) -> dict:
        eph = X25519PrivateKey.generate()
        shared = eph.exchange(recipient_pub)
        kek = HKDF(algorithm=hashes.SHA256(), length=32, salt=None,
                   info=b"BHDAM-R-dek-wrap").derive(shared)
        nonce = os.urandom(12)
        wrapped = AESGCM(kek).encrypt(nonce, dek, b"dek")
        eph_pub = eph.public_key().public_bytes(
            serialization.Encoding.Raw, serialization.PublicFormat.Raw)
        return {"ephemeral_pub": eph_pub.hex(), "nonce": nonce.hex(),
                "ciphertext": wrapped.hex()}


# --------------------------------------------------------------------------- #
# Recipient  (R3 -> R5)
# --------------------------------------------------------------------------- #
class ReconstructionError(Exception):
    pass


class Recipient:
    def __init__(self, recipient_id: str, kem_key: X25519PrivateKey):
        self.recipient_id = recipient_id
        self.kem_key = kem_key

    def receive(
        self,
        manifest: Manifest,
        signature: bytes,
        sender_pub: Ed25519PublicKey,
        arriving: list[tuple[int, bytes]],   # (sharenum, bytes) as they arrive
        ev: Evidence,
        timings: Optional[dict] = None,
        workers: int = 1,
    ) -> bytes:
        _t = _Timer(timings)
        # 1. Verify sender signature over the manifest (Section 10.2)
        try:
            sender_pub.verify(signature, manifest.canonical_bytes())
        except Exception as e:
            raise ReconstructionError(f"manifest signature invalid: {e}")
        ev.log("manifest_verified", sender=manifest.sender_id)

        expected = {s["sharenum"]: s for s in manifest.shards}

        # 2. Validate each arriving shard against its signed descriptor (Section 8.6).
        # Per-shard hashing is parallel across the arriving shards.
        _t.start()
        arriving_hashes = _sha256_many([blob for _, blob in arriving], workers)
        _t.stop("shard_validate_hash_ms")
        valid_shares, valid_nums = [], []
        for (num, blob), blob_hash in zip(arriving, arriving_hashes):
            desc = expected.get(num)
            if desc is None:
                ev.log("quarantine", sharenum=num, reason="unknown_shard"); continue
            if desc["transfer_id"] != manifest.transfer_id:
                ev.log("quarantine", sharenum=num, reason="replay_wrong_transfer"); continue
            if blob_hash != desc["sha256"]:
                ev.log("quarantine", sharenum=num, reason="hash_mismatch_tamper"); continue
            if num in valid_nums:
                ev.log("quarantine", sharenum=num, reason="duplicate"); continue
            valid_shares.append(blob); valid_nums.append(num)
            ev.log("shard_accepted", sharenum=num)

        # 3. Threshold check
        if len(valid_shares) < manifest.k:
            ev.trust_state = "R3"
            raise ReconstructionError(
                f"insufficient shards: {len(valid_shares)}/{manifest.k}")

        # 4. Reconstruct package (any k valid shards)
        _t.start()
        pkg = erasure_decode(valid_shares, valid_nums, manifest.k, manifest.n,
                             manifest.orig_package_len)
        _t.stop("erasure_decode_ms")
        if sha256(pkg) != manifest.package_sha256:
            raise ReconstructionError("package hash mismatch after decode")
        ev.log("package_reconstructed", used=len(valid_shares))

        # 5. Reverse AONT if used
        _t.start()
        ciphertext = aont_decode(pkg) if manifest.aont else pkg
        _t.stop("aont_decode_ms")

        # 6. Unwrap DEK (recipient-only) and decrypt (AEAD) -- R4
        dek = self._unwrap_dek(manifest.dek_wrap)   # raises on wrong key
        _t.start()
        try:
            plaintext = AESGCM(dek).decrypt(
                bytes.fromhex(manifest.aead_nonce), ciphertext,
                manifest.transfer_id.encode())
        except Exception as e:
            raise ReconstructionError(f"AEAD decryption failed: {e}")
        _t.stop("decrypt_ms")
        ev.trust_state = "R4"; ev.log("decrypted", aead_ok=True)

        # 7. Validate reconstructed files against manifest -- R5
        # Package is length-prefixed in sorted-name order; manifest.files is in the
        # same sorted order, so map blocks positionally back to names.
        blocks = self._unpack(plaintext)
        if len(blocks) != len(manifest.files):
            raise ReconstructionError("file count mismatch")
        files = {}
        for entry, block in zip(manifest.files, blocks):
            if sha256(block) != entry["sha256"]:
                raise ReconstructionError(f"file hash mismatch {entry['name']}")
            files[entry["name"]] = block
        ev.trust_state = "R5"; ev.log("evidence_ready", files=len(files))
        return plaintext

    def _unwrap_dek(self, wrap: dict) -> bytes:
        eph_pub = X25519PublicKey.from_public_bytes(bytes.fromhex(wrap["ephemeral_pub"]))
        shared = self.kem_key.exchange(eph_pub)
        kek = HKDF(algorithm=hashes.SHA256(), length=32, salt=None,
                   info=b"BHDAM-R-dek-wrap").derive(shared)
        try:
            return AESGCM(kek).decrypt(bytes.fromhex(wrap["nonce"]),
                                       bytes.fromhex(wrap["ciphertext"]), b"dek")
        except Exception as e:
            raise ReconstructionError(f"DEK unwrap failed (wrong recipient key?): {e}")

    @staticmethod
    def _unpack(package: bytes) -> list[bytes]:
        """Inverse of the length-prefixed concatenation -> ordered list of blocks."""
        out, off = [], 0
        while off < len(package):
            ln = int.from_bytes(package[off:off + 8], "big"); off += 8
            out.append(package[off:off + ln]); off += ln
        return out
