#!/usr/bin/env python3
"""
BHDAM-R multi-channel deployment validation (Experiment B).

NOT a performance benchmark. It demonstrates that the channel-independence
model of BHDAM-R operates over genuinely heterogeneous, real transports: each
of the n shards is dispatched over a *distinct* endpoint spanning several
technologies and providers -- Amazon S3, Azure Blob Storage, SFTP (SSH) and
FTPS, plus a local disk standing in for a physical/courier channel. Some
channels are then deliberately made unavailable and the dataset is
reconstructed from the surviving shards with the full R5 evidence package.

The point is qualitative: many independent, heterogeneous custodians including
an offline one, with a real cryptographic reconstruction and signed evidence.
Absolute timings include uncontrolled network latency and are context only.

Endpoints are supplied as environment variables holding COMMA-SEPARATED lists,
so one shard maps to one real endpoint. Credentials are read from the
environment and never written to the result files (which record only generic
transport labels, never bucket names, hosts or account IDs).

  S3     AWS_S3_BUCKETS="bucket-a,bucket-b,..."  (+ AWS credentials in env/role)
         optional AWS_S3_PREFIX (default bhdamr/)
  Azure  AZURE_SAS_URLS="https://...c1?sig=..,https://...c2?sig=.."
  SFTP   SFTP_ENDPOINTS="host1|user1|/keyfile1,host2|user2|/keyfile2,..."
         each entry host|user|keyfile[|port][|dir]  (port default 22, dir /tmp)
  FTPS   FTPS_ENDPOINTS="host1|user1|pass1,host2|user2|pass2,..."
         each entry host|user|password[|port]       (port default 21, explicit TLS)
  Disk   always added last as the physical channel

Run:
  python experiment_b_channels.py --k 10 --n 15 --lose 2 7 9 12 14 --size-mb 21 --out results_b
"""
from __future__ import annotations

import argparse
import ftplib
import hashlib
import io
import json
import os
import shutil
import ssl
import time

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey

from bhdam_r import Sender, Recipient, Evidence, sha256
import datasets


class LocalDiskChannel:
    def __init__(self, idx=0):
        self.name = "local-disk (physical)"
        self.dir = os.environ.get("BHDAMR_DISK_DIR", "./_physical_channel")
        os.makedirs(self.dir, exist_ok=True)

    def put(self, key, data):
        with open(os.path.join(self.dir, key), "wb") as f:
            f.write(data)

    def get(self, key):
        with open(os.path.join(self.dir, key), "rb") as f:
            return f.read()


class S3Channel:
    def __init__(self, bucket, idx):
        import boto3
        self.bucket = bucket
        self.prefix = os.environ.get("AWS_S3_PREFIX", "bhdamr/")
        self.cli = boto3.client("s3")
        self.name = f"amazon-s3 #{idx}"

    def put(self, key, data):
        self.cli.put_object(Bucket=self.bucket, Key=self.prefix + key, Body=data)

    def get(self, key):
        return self.cli.get_object(Bucket=self.bucket, Key=self.prefix + key)["Body"].read()


class AzureBlobChannel:
    def __init__(self, sas_url, idx):
        from azure.storage.blob import ContainerClient
        self.container_client = ContainerClient.from_container_url(sas_url)
        self.name = f"azure-blob #{idx}"

    def put(self, key, data):
        self.container_client.get_blob_client(key).upload_blob(data, overwrite=True)

    def get(self, key):
        return self.container_client.get_blob_client(key).download_blob().readall()


class SFTPChannel:
    def __init__(self, spec, idx):
        import paramiko
        parts = spec.split("|")
        host, user, keyfile = parts[0], parts[1], parts[2]
        port = int(parts[3]) if len(parts) > 3 and parts[3] else 22
        self.dir = parts[4] if len(parts) > 4 and parts[4] else "/tmp"
        self.t = paramiko.Transport((host, port))
        pkey = None
        for loader in (paramiko.RSAKey, paramiko.Ed25519Key, paramiko.ECDSAKey):
            try:
                pkey = loader.from_private_key_file(keyfile); break
            except Exception:
                continue
        if pkey is None:
            raise RuntimeError(f"could not load SFTP key {keyfile}")
        self.t.connect(username=user, pkey=pkey)
        self.sftp = paramiko.SFTPClient.from_transport(self.t)
        self.name = f"sftp-ssh #{idx}"

    def put(self, key, data):
        with self.sftp.open(f"{self.dir}/{key}", "wb") as f:
            f.write(data)

    def get(self, key):
        with self.sftp.open(f"{self.dir}/{key}", "rb") as f:
            return f.read()

    def close(self):
        try:
            self.sftp.close(); self.t.close()
        except Exception:
            pass


class _ReusedSessionFTP_TLS(ftplib.FTP_TLS):
    """FTP_TLS that reuses the control-channel TLS session for the data channel.
    Required by AWS Transfer Family FTPS, which rejects a fresh data-channel
    session with '522 Data connection must use cached TLS session'."""
    def ntransfercmd(self, cmd, rest=None):
        conn, size = ftplib.FTP.ntransfercmd(self, cmd, rest)
        if self._prot_p:
            conn = self.context.wrap_socket(
                conn, server_hostname=self.host,
                session=self.sock.session)  # <-- reuse control session
        return conn, size


class FTPSChannel:
    def __init__(self, spec, idx):
        parts = spec.split("|")
        self.host, self.user, self.password = parts[0], parts[1], parts[2]
        self.port = int(parts[3]) if len(parts) > 3 and parts[3] else 21
        self.dir = os.environ.get("FTPS_DIR", "")
        self.name = f"ftps #{idx}"
        self._connect().quit()  # validate connectivity now

    def _connect(self):
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE  # data-plane validation; custom-CA endpoints
        ftps = _ReusedSessionFTP_TLS(context=ctx)
        ftps.connect(self.host, self.port, timeout=30)
        ftps.login(self.user, self.password)
        ftps.prot_p()
        return ftps

    def put(self, key, data):
        ftps = self._connect()
        try:
            ftps.storbinary(f"STOR {key}", io.BytesIO(data))
        finally:
            try: ftps.quit()
            except Exception: pass

    def get(self, key):
        ftps = self._connect()
        buf = io.BytesIO()
        try:
            ftps.retrbinary(f"RETR {key}", buf.write)
        finally:
            try: ftps.quit()
            except Exception: pass
        return buf.getvalue()


def _csv(env):
    return [x.strip() for x in os.environ.get(env, "").split(",") if x.strip()]


def available_channels():
    chans = []
    for i, b in enumerate(_csv("AWS_S3_BUCKETS"), 1):
        try: chans.append(S3Channel(b, i))
        except Exception as e: print(f"  [skip] S3 #{i}: {e}")
    for i, u in enumerate(_csv("AZURE_SAS_URLS"), 1):
        try: chans.append(AzureBlobChannel(u, i))
        except Exception as e: print(f"  [skip] Azure #{i}: {e}")
    for i, s in enumerate(_csv("SFTP_ENDPOINTS"), 1):
        try: chans.append(SFTPChannel(s, i))
        except Exception as e: print(f"  [skip] SFTP #{i}: {e}")
    for i, s in enumerate(_csv("FTPS_ENDPOINTS"), 1):
        try: chans.append(FTPSChannel(s, i))
        except Exception as e: print(f"  [skip] FTPS #{i}: {e}")
    chans.append(LocalDiskChannel())
    return chans


def main():
    ap = argparse.ArgumentParser(description="BHDAM-R real multi-channel validation")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--n", type=int, default=15)
    ap.add_argument("--lose", type=int, nargs="*", default=[2, 7, 9, 12, 14])
    ap.add_argument("--size-mb", type=int, default=21)
    ap.add_argument("--aont", action="store_true", default=True)
    ap.add_argument("--out", default="results_b")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    print("BHDAM-R multi-channel deployment validation")
    print(f"  profile {args.k}-of-{args.n}, AONT-RS={'on' if args.aont else 'off'}, "
          f"losing shards {args.lose}")

    chans = available_channels()
    print(f"  {len(chans)} real endpoints configured")
    if 0 < len(chans) < args.n:
        print(f"  note: {len(chans)} endpoints for {args.n} shards; assigning round-robin "
              f"(each upload/download is still real).")
    if len(chans) == 0:
        raise SystemExit("no transports configured")
    assignment = [chans[i % len(chans)] for i in range(args.n)]
    print("  channel assignment:")
    for i, c in enumerate(assignment):
        tag = "  <-- LOST in transit" if i in args.lose else ""
        print(f"    shard {i:>2} -> {c.name}{tag}")

    sk, kem = Ed25519PrivateKey.generate(), X25519PrivateKey.generate()
    sender = Sender("Hospital-Source", sk)
    recipient = Recipient("Biotech-Lab", kem)
    files = {"payload.bin": datasets.sized_blob(args.size_mb)[f"payload_{args.size_mb}MB.bin"]}
    original = files["payload.bin"]
    h_orig = sha256(original)
    ch_names = [f"{assignment[i].name}:{i}" for i in range(args.n)]
    manifest, desc, shards, sig, _ = sender.build_transfer(
        files, recipient.recipient_id, kem.public_key(),
        args.k, args.n, ch_names, use_aont=args.aont)
    print(f"  sealed: {len(original):,} B -> {args.n} shards of {len(shards[0]):,} B "
          f"(sha256 {h_orig[:16]}...)")

    up_ms = {}
    for i, (c, sh) in enumerate(zip(assignment, shards)):
        key = f"{manifest.transfer_id}.shard{i}"
        t0 = time.perf_counter()
        c.put(key, sh)
        up_ms[i] = round((time.perf_counter() - t0) * 1000, 1)
        print(f"    up   shard {i:>2} -> {c.name:<20} {up_ms[i]:>9.1f} ms")

    arriving, down_ms = [], {}
    for i, c in enumerate(assignment):
        if i in args.lose:
            continue
        key = f"{manifest.transfer_id}.shard{i}"
        t0 = time.perf_counter()
        data = c.get(key)
        down_ms[i] = round((time.perf_counter() - t0) * 1000, 1)
        assert hashlib.sha256(data).hexdigest() == desc[i].sha256, \
            f"shard {i} integrity check failed after transport"
        arriving.append((i, data))
        print(f"    down shard {i:>2} <- {c.name:<20} {down_ms[i]:>9.1f} ms")

    ev = Evidence(transfer_id=manifest.transfer_id, trust_state="R3")
    plaintext = recipient.receive(manifest, sig, sk.public_key(), arriving, ev)
    recovered = plaintext[8:]
    h_rec = sha256(recovered)
    ok = (recovered == original and ev.trust_state == "R5")
    print(f"  reconstructed from {len(arriving)}/{args.n} shards "
          f"(lost {len(args.lose)}): sha256 {h_rec[:16]}... trust {ev.trust_state} -> "
          f"{'BYTE-IDENTICAL' if ok else 'MISMATCH'}")
    assert ok, "reconstruction or evidence verification failed"

    transports = sorted({c.name.split(' #')[0].split(' (')[0] for c in assignment})
    result = {
        "profile": f"{args.k}-of-{args.n}", "aont": args.aont,
        "payload_bytes": len(original),
        "sha256_original": h_orig, "sha256_recovered": h_rec,
        "byte_identical": ok, "trust_state": ev.trust_state,
        "n_endpoints": len(chans), "lost_shards": args.lose,
        "channel_types": {i: assignment[i].name for i in range(args.n)},
        "distinct_transport_technologies": transports,
        "upload_ms": up_ms, "download_ms": down_ms,
        "note": ("Deployment validation over real heterogeneous transports; timings "
                 "include network latency and are context only, not throughput."),
    }
    with open(os.path.join(args.out, "multichannel_result.json"), "w") as f:
        json.dump(result, f, indent=2, default=str)
    with open(os.path.join(args.out, "multichannel_evidence.json"), "w") as f:
        json.dump({"manifest": json.loads(manifest.canonical_bytes()),
                   "signature_hex": sig.hex(), "evidence": ev.__dict__}, f, indent=2, default=str)

    for c in assignment:
        if isinstance(c, LocalDiskChannel):
            shutil.rmtree(c.dir, ignore_errors=True); break
    for c in assignment:
        if isinstance(c, SFTPChannel):
            c.close()

    print(f"  wrote {args.out}/multichannel_result.json + multichannel_evidence.json")
    print("  technologies exercised:", ", ".join(transports))


if __name__ == "__main__":
    main()
