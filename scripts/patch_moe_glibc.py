#!/usr/bin/env python3
"""Make vLLM 0.11.2's MoE kernels loadable on glibc 2.28 (the H20 image).

vllm/_moe_C.abi3.so is the only extension in the 0.11.2 wheel that needs a glibc newer than 2.14, and only for one
symbol: log2@GLIBC_2.29 (glibc 2.29 re-versioned log2 when it dropped the SVID error wrapper). glibc 2.28 exports the
same function as log2@GLIBC_2.2.5, so the version requirement is rewritten to GLIBC_2.2.5. Refuses to touch anything
else: any other symbol bound to the version, a missing GLIBC_2.2.5 string, or an unexpected file. Idempotent; keeps
<file>.orig. Without this, dense models run and every MoE model fails at import.

python scripts/patch_moe_glibc.py <path to _moe_C.abi3.so>    (exit 0: patched or nothing to do)
"""
from __future__ import annotations

import shutil
import struct
import sys
from pathlib import Path

OLD, NEW = b"GLIBC_2.29", b"GLIBC_2.2.5"
ALLOWED = {b"log2"}


def elf_hash(name: bytes) -> int:
    h = 0
    for c in name:
        h = ((h << 4) + c) & 0xFFFFFFFF
        g = h & 0xF0000000
        if g:
            h ^= g >> 24
        h &= ~g & 0xFFFFFFFF
    return h


def sections(data: bytes) -> dict[str, dict[str, int]]:
    if data[:4] != b"\x7fELF" or data[4] != 2 or data[5] != 1:
        raise ValueError("not a 64-bit little-endian ELF file")
    shoff, = struct.unpack_from("<Q", data, 0x28)
    shentsize, shnum, shstrndx = struct.unpack_from("<HHH", data, 0x3A)
    raw = [struct.unpack_from("<IIQQQQIIQQ", data, shoff + i * shentsize) for i in range(shnum)]
    names_off = raw[shstrndx][4]
    out = {}
    for name, typ, _, _, off, size, link, info, _, _ in raw:
        out[cstr(data, names_off + name).decode()] = {"off": off, "size": size, "link": link, "info": info}
    return out


def cstr(data: bytes, off: int) -> bytes:
    return bytes(data[off:data.index(b"\0", off)])


def find_string(data: bytes, start: int, size: int, text: bytes) -> int | None:
    """Offset (relative to the string table) of `text` as a whole NUL-terminated string."""
    table = data[start:start + size]
    pos = table.find(b"\0" + text + b"\0")
    return None if pos < 0 else pos + 1


def patch(path: Path) -> str:
    data = bytearray(path.read_bytes())
    secs = sections(data)
    verneed, dynstr, dynsym, versym = (secs.get(n) for n in (".gnu.version_r", ".dynstr", ".dynsym", ".gnu.version"))
    if not all((verneed, dynstr, dynsym, versym)):
        raise ValueError("no symbol version tables")
    new_name = find_string(data, dynstr["off"], dynstr["size"], NEW)
    targets, off = [], verneed["off"]
    for _ in range(verneed["info"]):
        vn_version, vn_cnt, vn_file, vn_aux, vn_next = struct.unpack_from("<HHIII", data, off)
        aux = off + vn_aux
        for _ in range(vn_cnt):
            vna_hash, vna_flags, vna_other, vna_name, vna_next = struct.unpack_from("<IHHII", data, aux)
            if cstr(data, dynstr["off"] + vna_name) == OLD:
                targets.append((aux, vna_other))
            if not vna_next:
                break
            aux += vna_next
        if not vn_next:
            break
        off += vn_next
    if not targets:
        return "nothing to do (no GLIBC_2.29 requirement)"
    if new_name is None:
        raise ValueError("GLIBC_2.2.5 is not in .dynstr; cannot re-point the requirement")
    indices = {index for _, index in targets}
    bound = set()
    for i in range(dynsym["size"] // 24):
        st_name, = struct.unpack_from("<I", data, dynsym["off"] + i * 24)
        ndx, = struct.unpack_from("<H", data, versym["off"] + i * 2)
        if ndx & 0x7FFF in indices:
            bound.add(cstr(data, dynstr["off"] + st_name))
    if not bound <= ALLOWED:
        raise ValueError(f"GLIBC_2.29 also binds {sorted(bound - ALLOWED)}; refusing to patch")
    backup = path.with_name(path.name + ".orig")
    if not backup.exists():
        shutil.copy2(path, backup)
    for aux, _ in targets:
        struct.pack_into("<I", data, aux, elf_hash(NEW))
        struct.pack_into("<I", data, aux + 8, new_name)
    tmp = path.with_name(path.name + ".patching")
    tmp.write_bytes(bytes(data))
    shutil.copymode(path, tmp)
    tmp.replace(path)
    return f"patched {sorted(b.decode() for b in bound)}: GLIBC_2.29 -> GLIBC_2.2.5"


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    print(patch(Path(sys.argv[1])))
