"""Data contracts, bounded read-only SQLite execution, and independent grading."""
from __future__ import annotations

import collections
import dataclasses
import hashlib
import functools
import json
import math
import re
import sqlite3
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


@functools.lru_cache(maxsize=512)
def _snapshot_digest(path: str, stat_key: tuple[int, ...]) -> str:
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def digest(path: Path) -> str:
    path = path.resolve(strict=True)
    before = path.stat()
    key = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns)
    value = _snapshot_digest(str(path), key)
    after = path.stat()
    if key != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns):
        raise ValueError("File changed during hashing")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


@dataclasses.dataclass(frozen=True)
class Task:
    id: str
    db_id: str
    split: str
    question: str
    gold_sql: str
    db_file: str
    db_sha256: str
    source: str
    source_revision: str
    evidence: str = ""
    ordered: bool = False
    fixture: bool = False
    difficulty: str = "unknown"

    def __post_init__(self) -> None:
        for name in ("id", "db_id", "question", "gold_sql", "db_file", "source", "source_revision"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name).strip():
                raise ValueError(f"Invalid task field: {name}")
        if self.split not in {"train", "validation", "test"}:
            raise ValueError("split must be train, validation, or test")
        if not re.fullmatch(r"[0-9a-f]{64}", self.db_sha256):
            raise ValueError("Invalid database SHA-256")
        if type(self.ordered) is not bool or type(self.fixture) is not bool:
            raise ValueError("ordered and fixture must be booleans")
        if not isinstance(self.evidence, str):
            raise ValueError("evidence must be a string")
        if "mini-dev" in self.source.lower() and self.split != "test":
            raise ValueError("BIRD Mini-Dev is held-out test data in this project")

    def database(self, root: Path) -> Path:
        root = root.resolve(strict=True)
        if Path(self.db_file).is_absolute():
            raise ValueError("Database path must be relative")
        path = (root / self.db_file).resolve(strict=True)
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError("Database escapes the configured root")
        for suffix in ("-wal", "-journal"):
            sidecar = Path(str(path) + suffix)
            if sidecar.exists() and sidecar.stat().st_size:
                raise ValueError("Use a closed, checkpointed database snapshot")
        if digest(path) != self.db_sha256:
            raise ValueError(f"Database hash mismatch: {self.db_id}")
        return path


def audit(tasks: list[Task], root: Path) -> dict[str, Any]:
    if not tasks:
        raise ValueError("Dataset is empty")
    ids: set[str] = set()
    seen: dict[tuple[str, str], str] = {}
    db_versions: dict[str, str] = {}
    for task in tasks:
        if task.id in ids:
            raise ValueError(f"Duplicate task id: {task.id}")
        ids.add(task.id)
        task.database(root)
        if db_versions.setdefault(task.db_id, task.db_sha256) != task.db_sha256:
            raise ValueError(f"Conflicting snapshots for database: {task.db_id}")
        # SQL text is not normalized by lowercasing: string literal case matters.
        qa = hashlib.sha256((task.question.strip() + "\0" + task.gold_sql.strip()).encode()).hexdigest()
        for key in (("database_id", task.db_id), ("database_bytes", task.db_sha256), ("qa", qa)):
            previous = seen.setdefault(key, task.split)
            if previous != task.split:
                raise ValueError(f"Cross-split leakage: {key[0]}")
    return {"tasks": len(tasks), "databases": len(db_versions),
            "splits": dict(collections.Counter(t.split for t in tasks)),
            "fixtures": sum(t.fixture for t in tasks)}


@dataclasses.dataclass(frozen=True)
class Limits:
    seconds: float = 3.0
    max_rows: int = 1000
    max_bytes: int = 524288
    vm_steps: int = 5_000_000

    def __post_init__(self) -> None:
        if not 0 < self.seconds <= 60 or not 0 < self.max_rows <= 10000:
            raise ValueError("Invalid SQL time/row limit")
        if not 1024 <= self.max_bytes <= 8_388_608 or not 1000 <= self.vm_steps <= 2_000_000_000:
            raise ValueError("Invalid SQL byte/instruction limit")


SAFE_FUNCTIONS = set("""abs avg ceil ceiling char coalesce concat concat_ws count date datetime
floor format glob group_concat hex ifnull iif instr julianday length like likelihood likely
ln log log10 log2 lower ltrim max min mod nullif octet_length pow power printf quote replace
round rtrim sign sqrt strftime substr substring sum time timediff total trim typeof unicode
unixepoch unlikely upper json json_array json_array_length json_extract json_object json_quote
json_type json_valid row_number rank dense_rank percent_rank cume_dist ntile lag lead
first_value last_value nth_value""".split())
SCHEMA_SQL = "SELECT name, sql FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"


def _worker(request: dict[str, Any]) -> dict[str, Any]:
    """One disposable worker. Not a replacement for OS/container isolation."""
    limits = Limits(**request["limits"])
    sql = request["sql"]
    if not isinstance(sql, str) or len(sql.encode()) > 65536:
        return {"status": "rejected", "error": "SQL must be a string of at most 64 KiB"}
    prefix = re.sub(r"\A(?:\s|--[^\n]*(?:\n|$)|/\*.*?\*/)*", "", sql, flags=re.S)
    if not re.match(r"(?i)(SELECT|WITH)\b", prefix):
        return {"status": "rejected", "error": "Only a single SELECT/CTE is allowed"}
    started = time.monotonic()
    calls = 0
    def progress() -> int:
        nonlocal calls
        calls += 1000
        return int(calls >= limits.vm_steps or time.monotonic() - started > limits.seconds)
    def authorize(code: int, a: str | None, b: str | None, db: str | None, origin: str | None) -> int:
        if code == sqlite3.SQLITE_FUNCTION:
            return sqlite3.SQLITE_OK if (b or "").lower() in SAFE_FUNCTIONS else sqlite3.SQLITE_DENY
        if code in (sqlite3.SQLITE_SELECT, sqlite3.SQLITE_READ, sqlite3.SQLITE_RECURSIVE):
            return sqlite3.SQLITE_OK
        return sqlite3.SQLITE_DENY
    con = None
    try:
        path = Path(request["database"]).resolve(strict=True)
        con = sqlite3.connect(path.as_uri() + "?mode=ro", uri=True, timeout=0.2)
        con.execute("PRAGMA query_only=ON")
        con.execute("PRAGMA trusted_schema=OFF")
        con.setlimit(sqlite3.SQLITE_LIMIT_LENGTH, limits.max_bytes)
        con.setlimit(sqlite3.SQLITE_LIMIT_SQL_LENGTH, 65536)
        con.setlimit(sqlite3.SQLITE_LIMIT_COLUMN, 128)
        con.set_authorizer(authorize)
        con.set_progress_handler(progress, 1000)
        cur = con.execute(sql)
        rows: list[list[Any]] = []
        size = 0
        for row in cur:
            if len(rows) >= limits.max_rows:
                return {"status": "result_limit", "error": "Row limit; never grade a truncated result"}
            values = [{"$blob": x.hex()} if isinstance(x, bytes) else x for x in row]
            size += len(json.dumps(values, ensure_ascii=False, allow_nan=False).encode())
            if size > limits.max_bytes:
                return {"status": "result_limit", "error": "Byte limit; never grade a truncated result"}
            rows.append(values)
        return {"status": "ok", "columns": [c[0] for c in cur.description or []], "rows": rows}
    except (sqlite3.Error, ValueError, OSError) as exc:
        return {"status": "timeout" if "interrupted" in str(exc) else "sql_error", "error": str(exc)[:500]}
    finally:
        if con is not None:
            con.close()


def execute(database: Path, sql: str, limits: Limits = Limits()) -> dict[str, Any]:
    request = {"database": str(database.resolve()), "sql": sql, "limits": dataclasses.asdict(limits)}
    try:
        proc = subprocess.run([sys.executable, "-m", "exec_feedback.core", "--worker"],
                              input=json.dumps(request), text=True, encoding="utf-8",
                              capture_output=True, timeout=limits.seconds + 1, check=False)
    except subprocess.TimeoutExpired:
        return {"status": "timeout", "error": "Worker terminated at wall-clock limit"}
    if proc.returncode:
        return {"status": "worker_error", "error": proc.stderr[-500:]}
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError:
        return {"status": "worker_error", "error": "Invalid worker response"}


def cell_equal(a: Any, b: Any) -> bool:
    if a is None or b is None:
        return a is b
    if type(a) in (int, float) and type(b) in (int, float):
        if not math.isfinite(a) or not math.isfinite(b):
            return False
        if type(a) is int or type(b) is int or (a == int(a) and b == int(b)):
            return a == b  # Never use relative tolerance for integer IDs/counts.
        return math.isclose(a, b, rel_tol=1e-7, abs_tol=1e-6)
    return type(a) is type(b) and a == b


def rows_equal(a: list[Any], b: list[Any]) -> bool:
    return len(a) == len(b) and all(cell_equal(x, y) for x, y in zip(a, b))


def compare(pred: dict[str, Any], gold: dict[str, Any], ordered: bool = False) -> dict[str, Any]:
    if gold["status"] != "ok":
        return {"status": "invalid_gold", "correct": False, "reason": gold["status"]}
    if pred["status"] != "ok":
        return {"status": "execution_failed", "correct": False, "reason": pred["status"]}
    a, b = pred["rows"], gold["rows"]
    if len(pred["columns"]) != len(gold["columns"]) or len(a) != len(b):
        ok = False
    elif ordered:
        ok = all(rows_equal(x, y) for x, y in zip(a, b))
    elif collections.Counter(json.dumps(x, sort_keys=True) for x in a) == collections.Counter(json.dumps(x, sort_keys=True) for x in b):
        ok = True
    else:
        # Maximum bipartite matching avoids greedy tolerance-matching errors.
        if len(a) > 200:
            return {"status": "grader_limit", "correct": False, "reason": "Approximate bag matching exceeds 200 rows"}
        adj = [[j for j, y in enumerate(b) if rows_equal(x, y)] for x in a]
        matched: dict[int, int] = {}
        def augment(i: int, seen: set[int]) -> bool:
            for j in adj[i]:
                if j not in seen:
                    seen.add(j)
                    if j not in matched or augment(matched[j], seen):
                        matched[j] = i
                        return True
            return False
        ok = all(augment(i, set()) for i in range(len(a)))
    return {"status": "correct" if ok else "incorrect", "correct": ok}


if __name__ == "__main__":
    if sys.argv[1:] != ["--worker"]:
        raise SystemExit("Use the project CLI")
    try:
        import resource
        resource.setrlimit(resource.RLIMIT_AS, (512 * 1024 * 1024,) * 2)
    except (ImportError, ValueError, OSError):
        pass  # Windows: parent wall-clock limit still applies; use container isolation for untrusted DBs.
    print(json.dumps(_worker(json.load(sys.stdin)), ensure_ascii=False, allow_nan=False))
