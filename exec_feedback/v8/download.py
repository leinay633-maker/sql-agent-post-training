"""Download one pinned model snapshot from ModelScope or Hugging Face; every file must match the manifest sha256.

v0.7 measured 5-10 MB/s per connection from ModelScope, so a 236 GB snapshot needs many connections: files larger
than two parts are fetched as 1 GiB byte ranges in parallel and assembled (a server that ignores ranges falls back to
whole-file download). Resumable; the verification marker is written only when the whole snapshot is complete.

python -m exec_feedback.v8.download --manifest configs/v08-models.json --model <repo> --out models/teacher --workers 32
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import sys
import threading
import time
import urllib.request
from pathlib import Path

from ..storage import read_json
from ..v7.download import SOURCES, one
from ..v7.freeze import mark_verified

PART = 1 << 30
CHUNK = 8 << 20


class RangeIgnored(Exception):
    pass


def fetch_range(url: str, path: Path, start: int, end: int, timeout: float) -> None:
    """Bytes [start, end] into path, resuming a partial part."""
    want = end - start + 1
    have = path.stat().st_size if path.exists() else 0
    if have > want:
        path.unlink()
        have = 0
    if have == want:
        return
    request = urllib.request.Request(url, headers={"User-Agent": "efa-v08-download",
                                                   "Range": f"bytes={start + have}-{end}"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        if response.status != 206:
            raise RangeIgnored(f"HTTP {response.status} to a range request")
        with path.open("ab") as out:
            while True:
                block = response.read(CHUNK)
                if not block:
                    break
                out.write(block)
    if path.stat().st_size != want:
        raise OSError(f"part {path.name}: {path.stat().st_size} of {want} bytes")


def ranges(size: int) -> list[tuple[int, int]]:
    return [(start, min(size, start + PART) - 1) for start in range(0, size, PART)]


class Snapshot:
    def __init__(self, repo: str, revision: str, files: dict, out: Path, sources: list[str], log) -> None:
        self.repo, self.revision, self.files, self.out, self.sources, self.log = repo, revision, files, out, sources, log
        self.lock = threading.Lock()
        self.left: dict[str, int] = {}
        self.started: dict[str, float] = {}

    def url(self, source: str, name: str) -> str:
        return SOURCES[source].format(repo=self.repo, revision=self.revision, name=name)

    def part(self, name: str, index: int) -> str:
        meta, (start, end) = self.files[name], ranges(self.files[name]["size"])[index]
        path = self.out / f"{name}.range{index:03d}"
        errors = []
        self.started.setdefault(name, time.monotonic())
        # A failed teacher download costs the whole teacher arm, so parts keep retrying for up to ~25 minutes
        # (throttling, resets); the first host gets three tries before hosts alternate.
        for attempt in range(30):
            source = self.sources[0] if attempt < 3 else self.sources[attempt % len(self.sources)]
            try:
                fetch_range(self.url(source, name), path, start, end, timeout=120)
                break
            except RangeIgnored:
                raise
            except Exception as exc:  # noqa: BLE001 - network errors of any kind: retry
                errors.append(f"{source}: {type(exc).__name__}: {str(exc)[:160]}")
                self.log(f"retry {name} part {index}: {errors[-1]}")
                time.sleep(min(60, 2 ** attempt))
        else:
            raise RuntimeError(f"{name} part {index}: " + " | ".join(errors[-4:]))
        with self.lock:
            self.left[name] -= 1
            last = self.left[name] == 0
        return self.assemble(name, meta) if last else f"part {name} {index}"

    def assemble(self, name: str, meta: dict) -> str:
        parts = [self.out / f"{name}.range{i:03d}" for i in range(len(ranges(meta["size"])))]
        tmp, h = self.out / (name + ".part"), hashlib.sha256()
        with tmp.open("wb") as out:
            for p in parts:
                with p.open("rb") as f:
                    for block in iter(lambda: f.read(CHUNK), b""):
                        h.update(block)
                        out.write(block)
        if tmp.stat().st_size != meta["size"] or h.hexdigest() != meta["sha256"]:
            for p in [tmp, *parts]:
                p.unlink(missing_ok=True)
            raise OSError(f"{name}: assembled file differs from the manifest; parts discarded")
        tmp.replace(self.out / name)
        for p in parts:
            p.unlink()
        rate = meta["size"] / max(1e-6, time.monotonic() - self.started[name]) / 1e6
        self.log(f"ok {name} ({meta['size'] / 1e9:.2f} GB in {len(parts)} ranges, {rate:.0f} MB/s)")
        return f"ok {name}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--sources", default="modelscope,huggingface")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    entry = read_json(args.manifest)[args.model]
    files, revision = entry["files"], entry["revision"]
    args.out.mkdir(parents=True, exist_ok=True)
    sources = [s for s in args.sources.split(",") if s]
    if not sources or any(s not in SOURCES for s in sources):
        raise SystemExit(f"Unknown source in {args.sources}")

    def log(text: str) -> None:
        print(f"{time.strftime('%H:%M:%S')} {args.model} {text}", flush=True)

    started = time.monotonic()
    snap = Snapshot(args.model, revision, files, args.out, sources, log)
    big = sorted((n for n in files if files[n]["size"] > 2 * PART), key=lambda n: -files[n]["size"])
    small = [n for n in files if n not in big]
    failed: list[BaseException] = []
    with concurrent.futures.ThreadPoolExecutor(args.workers) as pool:
        jobs = {}
        for name in big:
            target = args.out / name
            if target.is_file() and target.stat().st_size == files[name]["size"]:
                small.append(name)  # complete from an earlier attempt: hash-checked by one()
                continue
            snap.left[name] = len(ranges(files[name]["size"]))
            for i in range(snap.left[name]):
                jobs[pool.submit(snap.part, name, i)] = name
        for name in small:
            jobs[pool.submit(one, args.model, revision, name, files[name], args.out, sources, log)] = name
        fallback = set()
        for future in concurrent.futures.as_completed(jobs):
            exc = future.exception()
            if isinstance(exc, RangeIgnored):
                fallback.add(jobs[future])
            elif exc is not None:
                failed.append(exc)
    for name in sorted(fallback):  # server ignored ranges: whole-file download, as in v0.7
        log(f"ranges ignored for {name}; whole-file download")
        for p in args.out.glob(f"{name}.range*"):
            p.unlink()
        try:
            one(args.model, revision, name, files[name], args.out, sources, log)
        except Exception as exc:  # noqa: BLE001
            failed.append(exc)
    for exc in failed:
        log(f"FAILED {exc}")
    if failed:
        return 1
    mark_verified(args.out, files)  # every file was sha256-checked (assemble() or one())
    total = sum(f["size"] for f in files.values())
    log(f"verified {len(files)} files, {total / 1e9:.1f} GB in {(time.monotonic() - started) / 60:.1f} min")
    return 0


if __name__ == "__main__":
    sys.exit(main())
