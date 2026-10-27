"""The final model of a finished main online run, packed for the trip home: one tar cut into 1 GiB parts in the exports
directory, each part hashed, and a manifest written last. Every booking is a fresh host that is recycled when it ends,
and the results leave through python's http.server, which cannot resume a broken download, so a part that arrives
damaged is fetched again on its own. unpack() rebuilds the directory at home and checks it against the run's milestone
record."""
from __future__ import annotations

import datetime as dt
import hashlib
import shutil
import tarfile
from pathlib import Path
from typing import Any, BinaryIO

from ..storage import atomic_json, read_json
from .freeze import load
from .weights import MARKER, dir_identity

PART_BYTES = 1 << 30


def pack_stem(identity: str, name: str, step: int) -> str:
    return f"v10-model-{identity[:8]}-{name}-step{step:04d}"


class Parts:
    """A write-only stream cut into numbered files of `size` bytes, hashing each part."""

    def __init__(self, stem: Path, size: int) -> None:
        self.stem, self.size = stem, size
        self.parts: list[dict[str, Any]] = []
        self.fh: BinaryIO | None = None
        self.hash = hashlib.sha256()
        self.left = 0

    def write(self, data: bytes) -> int:
        view = memoryview(data)
        while view:
            if self.fh is None or self.left == 0:
                self._next()
            chunk = view[:self.left]
            self.fh.write(chunk)
            self.hash.update(chunk)
            self.parts[-1]["bytes"] += len(chunk)
            self.left -= len(chunk)
            view = view[len(chunk):]
        return len(data)

    def _next(self) -> None:
        self.close()
        path = self.stem.with_name(f"{self.stem.name}.tar.part-{len(self.parts):03d}")
        self.fh, self.hash, self.left = path.open("wb"), hashlib.sha256(), self.size
        self.parts.append({"file": path.name, "bytes": 0})

    def close(self) -> None:
        if self.fh is not None:
            self.fh.close()
            self.parts[-1]["sha256"] = self.hash.hexdigest()
            self.fh = None


def pack(run: Path, *, name: str, exports: Path, part_bytes: int = PART_BYTES) -> dict[str, Any]:
    """Parts and manifest of `name`'s last milestone (step K of a finished run), checked against its record first."""
    frozen, _ = load(run)
    folder = run / "train" / name
    result = read_json(folder / "result.json") if (folder / "result.json").exists() else {}
    if not result.get("finished"):
        raise ValueError(f"{name} has not finished")
    step = result["last_milestone"]
    target = folder / "milestones" / f"step-{step:04d}"
    identity = read_json(folder / "milestones.json").get(str(step)) if (folder / "milestones.json").exists() else None
    if identity is None or dir_identity(target) != identity:
        raise ValueError(f"{name} milestone {step} is missing or changed on disk")
    exports.mkdir(parents=True, exist_ok=True)
    stem = exports / pack_stem(frozen["identity"], name, step)
    manifest_path = stem.with_name(stem.name + ".json")
    if manifest_path.exists():
        return read_json(manifest_path)
    for old in exports.glob(f"{stem.name}.tar.part-*"):  # an attempt that died before its manifest
        old.unlink()
    parts = Parts(stem, part_bytes)
    root = f"{name}-step{step:04d}"
    with tarfile.open(fileobj=parts, mode="w|", format=tarfile.PAX_FORMAT) as tar:
        for path in sorted(p for p in target.rglob("*") if p.is_file() and p.name != MARKER):
            tar.add(path, arcname=f"{root}/{path.relative_to(target).as_posix()}", recursive=False)
    parts.close()
    manifest = {"name": name, "step": step, "run": frozen["identity"], "root": root, "milestone_identity": identity,
                "bytes": sum(p["bytes"] for p in parts.parts), "parts": parts.parts,
                "created": dt.datetime.now(dt.timezone.utc).isoformat()}
    atomic_json(manifest_path, manifest)
    return manifest


class _Chain:
    """The parts read back as one stream, each checked against its hash as it is read."""

    def __init__(self, folder: Path, parts: list[dict[str, Any]]) -> None:
        self.folder, self.todo, self.fh, self.hash, self.part = folder, list(parts), None, None, None

    def read(self, size: int = -1) -> bytes:
        while True:
            if self.fh is None:
                if not self.todo:
                    return b""
                self.part = self.todo.pop(0)
                self.fh, self.hash = (self.folder / self.part["file"]).open("rb"), hashlib.sha256()
            data = self.fh.read(size if size and size > 0 else 1 << 22)
            if data:
                self.hash.update(data)
                return data
            self.fh.close()
            self.fh = None
            if self.hash.hexdigest() != self.part["sha256"]:
                raise ValueError(f"{self.part['file']} differs from its manifest")


def unpack(manifest_path: Path, dest: Path) -> Path:
    """dest/<root>, rebuilt from the parts beside the manifest and checked against the milestone identity."""
    manifest = read_json(manifest_path)
    out = dest / manifest["root"]
    if out.exists() and dir_identity(out) == manifest["milestone_identity"]:
        return out
    tmp = dest / f".{manifest['root']}.tmp"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    chain = _Chain(manifest_path.parent, manifest["parts"])
    with tarfile.open(fileobj=chain, mode="r|") as tar:
        tar.extractall(tmp, filter="data")
    while chain.read(1 << 22):  # the end-of-archive padding too, so every part is checked to its last byte
        pass
    if dir_identity(tmp / manifest["root"]) != manifest["milestone_identity"]:
        raise ValueError(f"Unpacked {manifest['root']} is not the run's milestone")
    if out.exists():
        shutil.rmtree(out)
    (tmp / manifest["root"]).rename(out)
    shutil.rmtree(tmp)
    return out
