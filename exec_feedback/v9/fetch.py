"""Fetch a registered BIRD archive with parallel byte ranges before the main environment exists (stdlib only), then
extract it exactly as `python -m exec_feedback fetch-bird` does (that command finds the verified archive in place).

v0.8 spent about 19 minutes on BIRD over one connection. A server that ignores ranges, or any failure here, leaves no
archive behind and the run script falls back to the single-connection fetch.

python -m exec_feedback.v9.fetch --part train --sha256 <registered sha256> --out data/raw --workers 16
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import os
import sys
import time
import urllib.request
from pathlib import Path

from ..downloads import SOURCES, fetch_bird
from ..v8.download import RangeIgnored, fetch_range

PART = 256 << 20
CHUNK = 8 << 20


def content_length(url: str) -> int:
    request = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "efa-v09-fetch"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return int(response.headers["Content-Length"])


def download(url: str, target: Path, sha256: str, workers: int, log) -> None:
    size = content_length(url)
    parts = [(start, min(size, start + PART) - 1) for start in range(0, size, PART)]
    paths = [target.with_name(f"{target.name}.range{i:03d}") for i in range(len(parts))]

    def one(index: int) -> None:
        start, end = parts[index]
        for attempt in range(10):
            try:
                fetch_range(url, paths[index], start, end, timeout=120)
                return
            except RangeIgnored:
                raise
            except Exception as exc:  # noqa: BLE001 - network errors of any kind: retry
                log(f"retry range {index}: {type(exc).__name__}: {str(exc)[:160]}")
                time.sleep(min(60, 2 ** attempt))
        raise RuntimeError(f"range {index} failed ten times")

    started = time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(workers) as pool:
        list(pool.map(one, range(len(parts))))
    tmp, digest = target.with_name(target.name + ".part"), hashlib.sha256()
    with tmp.open("wb") as out:
        for path in paths:
            with path.open("rb") as f:
                for block in iter(lambda: f.read(CHUNK), b""):
                    digest.update(block)
                    out.write(block)
    for path in paths:
        path.unlink()
    if tmp.stat().st_size != size or digest.hexdigest() != sha256:
        tmp.unlink()
        raise RuntimeError(f"assembled archive differs from the registered sha256 ({digest.hexdigest()})")
    os.replace(tmp, target)
    log(f"ok {target.name}: {size / 1e9:.2f} GB in {len(parts)} ranges, "
        f"{size / max(1e-6, time.monotonic() - started) / 1e6:.0f} MB/s")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", required=True, choices=sorted(SOURCES))
    ap.add_argument("--sha256", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    def log(text: str) -> None:
        print(f"{time.strftime('%H:%M:%S')} BIRD {args.part} {text}", flush=True)

    if (args.out / f"{args.part}.source.json").exists():
        log("already fetched and extracted")
        return 0
    archive = args.out / ".downloads" / f"{args.part}.zip"
    archive.parent.mkdir(parents=True, exist_ok=True)
    if not archive.exists():
        try:
            download(SOURCES[args.part], archive, args.sha256, args.workers, log)
        except Exception as exc:  # noqa: BLE001 - leave nothing behind; the single-connection fetch takes over
            log(f"parallel download failed ({type(exc).__name__}: {str(exc)[:200]}); falling back")
            for leftover in archive.parent.glob(f"{archive.name}*"):
                leftover.unlink()
            return 1
    started = time.monotonic()
    manifest = fetch_bird(args.part, args.out, accept_terms=True)
    log(f"extracted {manifest['sqlite_databases']} databases in {(time.monotonic() - started) / 60:.1f} min")
    return 0


if __name__ == "__main__":
    sys.exit(main())
