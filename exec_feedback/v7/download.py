"""Download one pinned model snapshot, file by file, from ModelScope or Hugging Face; every file must match the
manifest sha256 (the two hosts serve byte-identical files for these revisions). Resumable; writes the verification
marker only when the whole snapshot is complete.

python -m exec_feedback.v7.download --manifest configs/v07-models.json --model Qwen/Qwen3-14B --out models/scale
"""
from __future__ import annotations

import argparse
import concurrent.futures
import sys
import time
import urllib.request
from pathlib import Path

from ..storage import read_json
from .freeze import file_sha256, mark_verified

SOURCES = {
    "modelscope": "https://modelscope.cn/models/{repo}/resolve/master/{name}",
    "huggingface": "https://huggingface.co/{repo}/resolve/{revision}/{name}",
}
CHUNK = 8 << 20


def fetch(url: str, part: Path, size: int, timeout: float) -> None:
    have = part.stat().st_size if part.exists() else 0
    if have > size:
        part.unlink()
        have = 0
    if have == size:
        return
    request = urllib.request.Request(url, headers={"User-Agent": "efa-v07-download"})
    if have:
        request.add_header("Range", f"bytes={have}-")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        if have and response.status != 206:  # server ignored the range: start over
            have = 0
        with part.open("ab" if have else "wb") as out:
            while True:
                block = response.read(CHUNK)
                if not block:
                    break
                out.write(block)


def one(repo: str, revision: str, name: str, meta: dict, out: Path, sources: list[str], log) -> str:
    target = out / name
    if target.is_file() and target.stat().st_size == meta["size"] and file_sha256(target) == meta["sha256"]:
        return f"have {name}"
    part = out / (name + ".part")
    errors = []
    for attempt in range(6):
        source = sources[attempt % len(sources)]
        url = SOURCES[source].format(repo=repo, revision=revision, name=name)
        started = time.monotonic()
        try:
            fetch(url, part, meta["size"], timeout=120)
            if part.stat().st_size != meta["size"]:
                raise OSError(f"size {part.stat().st_size} != {meta['size']}")
            if file_sha256(part) != meta["sha256"]:
                part.unlink()
                raise OSError("sha256 mismatch; partial file discarded")
            part.replace(target)
            rate = meta["size"] / max(1e-6, time.monotonic() - started) / 1e6
            log(f"ok {name} from {source} ({meta['size'] / 1e9:.2f} GB, {rate:.0f} MB/s)")
            return f"ok {name}"
        except Exception as exc:  # noqa: BLE001 - network errors of any kind: retry with the other host
            errors.append(f"{source}: {type(exc).__name__}: {str(exc)[:200]}")
            log(f"retry {name}: {errors[-1]}")
            time.sleep(min(30, 2 ** attempt))
    raise RuntimeError(f"{name}: " + " | ".join(errors))


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
    # Largest files first so the long transfers overlap.
    order = sorted(files, key=lambda n: -files[n]["size"])
    with concurrent.futures.ThreadPoolExecutor(args.workers) as pool:
        futures = [pool.submit(one, args.model, revision, n, files[n], args.out, sources, log) for n in order]
        failed = [f for f in futures if f.exception() is not None]
    for f in failed:
        log(f"FAILED {f.exception()}")
    if failed:
        return 1
    mark_verified(args.out, files)  # every file was sha256-checked in one()
    total = sum(f["size"] for f in files.values())
    log(f"verified {len(files)} files, {total / 1e9:.1f} GB in {(time.monotonic() - started) / 60:.1f} min")
    return 0


if __name__ == "__main__":
    sys.exit(main())
