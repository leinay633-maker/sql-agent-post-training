"""Credential-free code handoff and verified experiment-result archives.

Git is used only on the developer's computer when packaging tracked source.
The GPU only writes local exports; nothing is pushed or uploaded automatically.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import re
import shutil
import subprocess
import time
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any

from .core import digest
from .storage import atomic_json, exclusive_lock, read_json


def _safe_name(name: str) -> None:
    path = PurePosixPath(name)
    if not name or path.is_absolute() or '..' in path.parts or '\\' in name or ':' in name:
        raise ValueError("Unsafe archive member")


def _write_bundle(files: list[tuple[Path, str]], out: Path, metadata: dict[str, Any]) -> dict[str, Any]:
    if out.exists():
        raise FileExistsError("Do not overwrite a handoff archive")
    out.parent.mkdir(parents=True, exist_ok=True)
    temp = out.with_name(out.name + '.tmp')
    if temp.exists():
        raise FileExistsError("Previous partial bundle exists; use a new output name")
    entries = {}
    try:
        with zipfile.ZipFile(temp, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
            for path, name in files:
                _safe_name(name)
                if name in entries or name == 'BUNDLE.json' or path.is_symlink():
                    raise ValueError("Duplicate/reserved name or symlink in bundle")
                before = path.stat()
                h = hashlib.sha256()
                with path.open('rb') as source, archive.open(name, 'w', force_zip64=True) as target:
                    while block := source.read(1024 * 1024):
                        h.update(block); target.write(block)
                after = path.stat()
                if (before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (after.st_size, after.st_mtime_ns, after.st_ctime_ns):
                    raise ValueError(f"File changed during export: {name}; stop writers before exporting")
                entries[name] = {"sha256": h.hexdigest(), "bytes": before.st_size}
            manifest = {"format": "efa-handoff-v1", "created_unix": time.time(), **metadata, "files": entries}
            archive.writestr('BUNDLE.json', json.dumps(manifest, indent=2, ensure_ascii=False))
        os.replace(temp, out)
    finally:
        temp.unlink(missing_ok=True)
    result = {"file": out.name, "sha256": digest(out), "bytes": out.stat().st_size,
              "files": len(entries), "kind": metadata["kind"]}
    atomic_json(out.with_suffix(out.suffix + '.sha256.json'), result)
    return result


def package_code(root: Path, out: Path) -> dict[str, Any]:
    root = root.resolve(strict=True)
    changed = subprocess.check_output(['git', 'status', '--porcelain', '--', '.'], cwd=root, text=True, encoding='utf-8')
    if changed.strip():
        raise ValueError("Commit project changes before packaging; only committed source is exported")
    names = subprocess.check_output(['git', 'ls-files', '-z', '--', '.'], cwd=root).decode('utf-8').split('\0')
    files = []
    for name in filter(None, names):
        _safe_name(name)
        path = PurePosixPath(name)
        if any(part in {'.git', '.venv', '.runtime', 'data', 'outputs', 'exports', '.ssh', '.hf', '.cache'} for part in path.parts):
            raise ValueError(f"Runtime data or credentials unexpectedly tracked: {name}")
        if path.name == 'local.json' or path.name.startswith('.env') and path.name != '.env.example' or path.suffix in {'.pem', '.key'}:
            raise ValueError(f"Possible secret/config tracked: {name}")
        files.append((root / name, 'exec-feedback-agent/' + name))
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True, encoding='utf-8').strip()
    return _write_bundle(files, out, {"kind": "source", "git_commit": commit, "credentials_included": False})


def export_results(study: Path, destination: Path, *, label: str = 'manual', lock: bool = True,
                   skip: set[str] = frozenset()) -> dict[str, Any]:
    study = study.resolve(strict=True)
    destination = destination.resolve()
    if destination.is_relative_to(study) or not re.fullmatch(r'[A-Za-z0-9_-]{1,80}', label):
        raise ValueError("Export destination must be outside the study; use a simple label")
    with exclusive_lock(study / '.study.lock') if lock else contextlib.nullcontext():
        state = read_json(study / 'study.json')
        files = []
        for path in sorted(study.rglob('*')):
            if not path.is_file(): continue
            rel = path.relative_to(study)
            if any(part.startswith('.') or '.tmp-' in part or part.endswith('.tmp') for part in rel.parts): continue
            if rel.parts[0] in skip: continue  # stages still being written by another process
            if path.suffix in {'.sqlite', '.zip'}: continue
            if path.name.startswith('checkpoint-'): continue
            # Half-written checkpoint directories are never restorable evidence.
            parent_checkpoint = next((parent for parent in path.parents if parent.name.startswith('checkpoint-')), None)
            if parent_checkpoint and not (parent_checkpoint / 'complete.json').exists(): continue
            files.append((path, study.name + '/' + rel.as_posix()))
        # Include setup and pilot environment evidence, not caches/weights/data.
        project = Path(__file__).resolve().parents[1]
        for folder in (project / 'outputs' / 'environment', project / 'outputs' / 'pilot'):
            if folder.exists():
                for path in sorted(folder.rglob('*')):
                    if path.is_file() and path.suffix in {'.json', '.jsonl', '.txt', '.log', '.xml'}:
                        files.append((path, 'runtime-evidence/' + path.relative_to(project / 'outputs').as_posix()))
        name = f'{study.name}-{label}-{time.time_ns()}.zip'
        result = _write_bundle(files, destination / name, {"kind": "experiment-results", "study": study.name,
                    "study_status": state["status"], "code_signature": state.get('signature'),
                    "base_weights_included": False, "databases_included": False,
                    "contains_checkpoints": any('checkpoint-' in n for _, n in files)})
        atomic_json(destination / 'latest.json', result)
        return result


def verify_bundle(path: Path, expected_sha256: str | None = None) -> dict[str, Any]:
    if expected_sha256 and digest(path) != expected_sha256:
        raise ValueError("Bundle SHA-256 mismatch")
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if len(set(names)) != len(names): raise ValueError("Duplicate archive entries")
        for name in names: _safe_name(name)
        meta = json.loads(archive.read('BUNDLE.json'))
        if meta.get('format') != 'efa-handoff-v1' or set(names) != set(meta['files']) | {'BUNDLE.json'}:
            raise ValueError("Bundle manifest differs from archive")
        for name, expected in meta['files'].items():
            h, size = hashlib.sha256(), 0
            with archive.open(name) as source:
                while block := source.read(1024 * 1024): h.update(block); size += len(block)
            if h.hexdigest() != expected['sha256'] or size != expected['bytes']:
                raise ValueError(f"Corrupt handoff entry: {name}")
    return {"verified": True, "kind": meta['kind'], "files": len(meta['files']), "sha256": digest(path)}


def pull_latest(host: str, remote: str, local: Path, port: int, once: bool, interval: float) -> None:
    """Run on Windows/local machine, using its existing SSH auth; no Git auth on GPU."""
    if not re.fullmatch(r'[A-Za-z0-9_.@-]+', host) or host.startswith('-'):
        raise ValueError("Use a plain SSH host alias or user@hostname")
    if not re.fullmatch(r'/[A-Za-z0-9_./-]+', remote) or '..' in PurePosixPath(remote).parts:
        raise ValueError("Remote export path must be absolute with no spaces or traversal")
    if not 1 <= port <= 65535 or interval < 10: raise ValueError("Invalid SSH port or interval")
    local.mkdir(parents=True, exist_ok=True)
    # -O: legacy SCP protocol. OpenSSH 9 defaults to SFTP, which GPU hosts
    # without an sshd sftp Subsystem refuse.
    scp = ['scp', '-O', '-P', str(port)]
    while True:
        meta = local / 'remote-latest.tmp.json'
        proc = subprocess.run([*scp, '-q', f'{host}:{remote}/latest.json', str(meta)], check=False)
        if proc.returncode == 0:
            info = read_json(meta); name = info['file']
            if PurePosixPath(name).name != name or not re.fullmatch(r'[A-Za-z0-9_.-]+\.zip', name):
                raise ValueError("Unsafe remote bundle name")
            target = local / name
            if not target.exists():
                temp = target.with_suffix('.zip.part')
                subprocess.run([*scp, f'{host}:{remote}/{name}', str(temp)], check=True)
                verified = verify_bundle(temp, info['sha256'])
                os.replace(temp, target)
                atomic_json(target.with_suffix('.receipt.json'), {**info, **verified})
                print(json.dumps({"saved": str(target), "verified": True}), flush=True)
            elif digest(target) != info['sha256']:
                raise ValueError("Existing local archive checksum differs")
        elif once:
            raise RuntimeError("No remote export yet or SCP failed")
        if once: return
        time.sleep(interval)


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='command', required=True)
    code = sub.add_parser('code'); code.add_argument('--out', type=Path, required=True)
    results = sub.add_parser('results'); results.add_argument('--study', type=Path, required=True); results.add_argument('--out', type=Path, default=Path('exports'))
    check = sub.add_parser('verify'); check.add_argument('path', type=Path); check.add_argument('--sha256')
    pull = sub.add_parser('pull'); pull.add_argument('--host', required=True); pull.add_argument('--remote', required=True)
    pull.add_argument('--out', type=Path, required=True); pull.add_argument('--port', type=int, default=22)
    pull.add_argument('--once', action='store_true'); pull.add_argument('--interval', type=float, default=300)
    args = parser.parse_args()
    if args.command == 'code': result = package_code(Path.cwd(), args.out)
    elif args.command == 'results': result = export_results(args.study, args.out)
    elif args.command == 'verify': result = verify_bundle(args.path, args.sha256)
    else:
        pull_latest(args.host, args.remote, args.out, args.port, args.once, args.interval); return
    print(json.dumps(result, indent=2))


if __name__ == '__main__': main()
