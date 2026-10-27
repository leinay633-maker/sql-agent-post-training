"""Explicit public BIRD download and bounded ZIP extraction. Never runs by import."""
from __future__ import annotations

import os
import shutil
import stat
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any

from .core import digest
from .storage import atomic_json, read_json

SOURCES = {"train": "https://bird-bench.oss-cn-beijing.aliyuncs.com/train.zip",
           "dev": "https://bird-bench.oss-cn-beijing.aliyuncs.com/dev.zip"}


def safe_extract(archive: Path, destination: Path, *, max_bytes: int = 80 * 1024**3) -> dict[str, Any]:
    if destination.exists():
        raise ValueError("Extraction target already exists")
    with zipfile.ZipFile(archive) as z:
        members = z.infolist()
        total = sum(i.file_size for i in members)
        if total > max_bytes or len(members) > 100000:
            raise ValueError("Archive exceeds extraction limits")
        for info in members:
            name = PurePosixPath(info.filename)
            mode = info.external_attr >> 16
            if name.is_absolute() or ".." in name.parts or "\\" in info.filename or ":" in info.filename or stat.S_ISLNK(mode):
                raise ValueError("Unsafe archive path or symlink")
            if info.file_size and info.compress_size == 0 or info.file_size / max(1, info.compress_size) > 10000:
                raise ValueError("Suspicious archive expansion ratio")
        parent = destination.parent
        parent.mkdir(parents=True, exist_ok=True)
        if shutil.disk_usage(parent).free < total:
            raise ValueError("Insufficient disk space for archive")
        destination.mkdir()
        for info in members:
            target = destination.joinpath(*PurePosixPath(info.filename).parts)
            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with z.open(info) as source, target.open("xb") as sink:
                shutil.copyfileobj(source, sink, length=1024*1024)
    return {"files": len(members), "uncompressed_bytes": total}


def fetch_bird(part: str, output: Path, *, accept_terms: bool = False,
               max_download_gb: int = 40) -> dict[str, Any]:
    if not accept_terms:
        raise ValueError("Read BIRD's CC BY-SA 4.0 data terms and pass --accept-terms")
    if part not in SOURCES or not 1 <= max_download_gb <= 100:
        raise ValueError("Invalid data part or download limit")
    output.mkdir(parents=True, exist_ok=True)
    cached = output / ".downloads"
    cached.mkdir(exist_ok=True)
    archive = cached / f"{part}.zip"
    target = output / part
    manifest_path = output / f"{part}.source.json"
    if manifest_path.exists():
        saved = read_json(manifest_path)
        if not archive.exists() or digest(archive) != saved["sha256"] or not target.exists():
            raise ValueError("Downloaded source was changed or removed")
        return saved
    if not archive.exists():
        temporary = archive.with_suffix(".zip.part")
        request = urllib.request.Request(SOURCES[part], headers={"User-Agent": "exec-feedback-agent/0.2"})
        with urllib.request.urlopen(request, timeout=60) as response, temporary.open("wb") as sink:
            total = 0
            while block := response.read(1024*1024):
                total += len(block)
                if total > max_download_gb * 1024**3:
                    raise ValueError("Download byte budget exceeded")
                sink.write(block)
        os.replace(temporary, archive)
    if target.exists():
        raise ValueError("Partial extraction directory exists; inspect it or use a new data root")
    extracted = safe_extract(archive, target)
    remaining = 80*1024**3 - extracted["uncompressed_bytes"]
    # BIRD's archive carries macOS AppleDouble files (__MACOSX/._x.zip) that
    # are metadata, not ZIPs.
    def appledouble(p: Path) -> bool:
        return "__MACOSX" in p.parts and p.name.startswith("._")
    nested = [p for p in target.rglob("*.zip") if not appledouble(p)]
    for inner in nested:
        result = safe_extract(inner, inner.with_suffix(""), max_bytes=remaining)
        remaining -= result["uncompressed_bytes"]
    if any(p.suffix == ".zip" and not appledouble(p)
           for inner in nested for p in inner.with_suffix("").rglob("*")):
        raise ValueError("Unexpected third-level nested ZIP; inspect source before continuing")
    manifest = {"part": part, "url": SOURCES[part], "sha256": digest(archive),
                "data_terms": "CC BY-SA 4.0; retain attribution; review database-specific rights",
                "integrity_policy": "first-download hash frozen; not an upstream-signed checksum",
                "json_candidates": [p.relative_to(output).as_posix() for p in target.rglob("*.json")],
                "sqlite_databases": len(list(target.rglob("*.sqlite"))), "target": str(target)}
    atomic_json(manifest_path, manifest)
    return manifest
