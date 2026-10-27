"""CPU-first entry points. No network call occurs without the explicit eval command."""
from __future__ import annotations

import argparse
import dataclasses
import importlib.util
import json
import platform
import re
import sqlite3
import sys
from pathlib import Path

from .core import Task, audit, digest, read_jsonl, write_jsonl
from .pipeline import FixturePolicy, HTTPPolicy, build_sft, evaluate


def load_tasks(paths: list[Path]) -> list[Task]:
    return [Task(**row) for path in paths for row in read_jsonl(path)]


def make_fixture(root: Path) -> Task:
    root.mkdir(parents=True, exist_ok=False)
    db = root / "shop.sqlite"
    with sqlite3.connect(db) as con:
        con.executescript("""CREATE TABLE orders(id INTEGER PRIMARY KEY, customer_id INTEGER, amount REAL, status TEXT);
INSERT INTO orders VALUES (1,1,12.5,'paid'),(2,1,12.5,'paid'),(3,2,20,'cancelled'),(4,3,15,'paid'),(5,3,NULL,'paid');""")
    return Task(id="fixture-paid-revenue", db_id="fixture-shop", split="train",
                question="What is the total amount of paid orders?", gold_sql="SELECT SUM(amount) FROM orders WHERE status='paid'",
                db_file="shop.sqlite", db_sha256=digest(db), source="engineering-fixture",
                source_revision="efa-fixture-v1", fixture=True)


def import_bird(args: argparse.Namespace) -> dict:
    actual = digest(args.input)
    if actual != args.expected_sha256:
        raise ValueError(f"Source hash mismatch; actual SHA-256: {actual}")
    if not args.ack_data_terms:
        raise ValueError("Review upstream data terms and pass --ack-data-terms")
    if args.kind == "bird-mini-dev" and args.split != "test":
        raise ValueError("Mini-Dev must remain test-only")
    if args.kind == "bird-train" and args.split == "test":
        raise ValueError("Use an explicitly named evaluation source for test data")
    raw = json.loads(args.input.read_text(encoding="utf-8"))
    if not isinstance(raw, list) or not raw:
        raise ValueError("Expected a nonempty BIRD JSON array")
    tasks = []
    for i, row in enumerate(raw):
        db_id = row["db_id"]
        if not isinstance(db_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", db_id):
            raise ValueError("Unsafe database id")
        sql = row["SQL"]
        prefix = re.sub(r"\A(?:\s|--[^\n]*(?:\n|$)|/\*.*?\*/)*", "", sql, flags=re.S)
        if not re.match(r"(?i)(SELECT|WITH)\b", prefix):
            raise ValueError(f"Non-SELECT record {i}; explicitly curate a SELECT-only input, never silently drop it")
        rel = f"{db_id}/{db_id}.sqlite"
        db = (args.db_root / rel).resolve(strict=True)
        if not db.is_relative_to(args.db_root.resolve(strict=True)):
            raise ValueError("Database path escapes root")
        task = Task(id=f"{args.kind}:{row.get('question_id', i)}", db_id=db_id,
                    split=args.split, question=row["question"], gold_sql=sql,
                    db_file=rel, db_sha256=digest(db), source=args.kind,
                    source_revision=args.source_revision, evidence=row.get("evidence") or "")
        tasks.append(task)
    report = audit(tasks, args.db_root)
    write_jsonl(args.output, [dataclasses.asdict(t) for t in tasks])
    manifest = {"source_sha256": actual, "source_revision": args.source_revision, "kind": args.kind,
                "terms_acknowledged": True, "output_sha256": digest(args.output),
                "order_policy": "bag default; manually annotate order-sensitive tasks before freezing", **report}
    with args.output.with_suffix(".manifest.json").open("x", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return manifest


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="efa")
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("doctor")
    demo = sub.add_parser("demo")
    demo.add_argument("--out", type=Path, required=True)
    sha = sub.add_parser("hash")
    sha.add_argument("path", type=Path)
    imp = sub.add_parser("import-bird")
    for key in ("input", "db-root", "output"):
        imp.add_argument("--" + key, type=Path, required=True)
    imp.add_argument("--expected-sha256", required=True)
    imp.add_argument("--source-revision", required=True)
    imp.add_argument("--kind", choices=["bird-train", "bird-mini-dev"], required=True)
    imp.add_argument("--split", choices=["train", "validation", "test"], required=True)
    imp.add_argument("--ack-data-terms", action="store_true")
    for command in ("audit", "eval", "sft"):
        cmd = sub.add_parser(command)
        cmd.add_argument("--tasks", type=Path, nargs="+", required=True)
        cmd.add_argument("--db-root", type=Path, required=True)
        if command == "eval":
            cmd.add_argument("--out", type=Path, required=True)
            cmd.add_argument("--endpoint", default="http://127.0.0.1:8000/v1")
            cmd.add_argument("--model", required=True)
            cmd.add_argument("--split", choices=["train", "validation", "test"], default="test")
            cmd.add_argument("--max-steps", type=int, default=4)
            cmd.add_argument("--max-tokens", type=int, default=512)
            cmd.add_argument("--direct", action="store_true")
            cmd.add_argument("--allow-remote", action="store_true")
        if command == "sft":
            cmd.add_argument("--out", type=Path, required=True)
            cmd.add_argument("--mode", choices=["success", "repair"], required=True)
            cmd.add_argument("--traces", type=Path)
            cmd.add_argument("--allow-fixture", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.command == "doctor":
            result = {"python": sys.version.split()[0], "platform": platform.platform(),
                      "sqlite": sqlite3.sqlite_version, "gpu_checked": False,
                      "optional_packages": {p: importlib.util.find_spec(p) is not None for p in ("torch", "transformers", "peft", "bitsandbytes")},
                      "spending_authorized": 0, "network_requests": 0}
        elif args.command == "hash":
            result = {"sha256": digest(args.path)}
        elif args.command == "demo":
            args.out.mkdir(parents=True, exist_ok=False)
            task = make_fixture(args.out / "db")
            write_jsonl(args.out / "tasks.jsonl", [dataclasses.asdict(task)])
            policy = FixturePolicy([{"action": "query", "sql": "SELECT SUM(gross_sales) FROM orders"},
                                    {"action": "finish", "sql": task.gold_sql}])
            result = evaluate([task], args.out / "db", policy, args.out / "run")
            traces = read_jsonl(args.out / "run" / "episodes.jsonl")
            for mode in ("success", "repair"):
                rows = build_sft([task], args.out / "db", mode, traces, allow_fixture=True)
                write_jsonl(args.out / f"{mode}.jsonl", rows)
        elif args.command == "import-bird":
            result = import_bird(args)
        else:
            tasks = load_tasks(args.tasks)
            checked = audit(tasks, args.db_root)  # Audit all provided splits BEFORE selecting one.
            if args.command == "audit":
                result = checked
            elif args.command == "eval":
                selected = [t for t in tasks if t.split == args.split]
                policy = HTTPPolicy(args.endpoint, args.model, allow_remote=args.allow_remote, max_tokens=args.max_tokens)
                result = evaluate(selected, args.db_root, policy, args.out,
                                  max_steps=1 if args.direct else args.max_steps, feedback=not args.direct)
            else:
                selected = [t for t in tasks if t.split == "train"]
                traces = read_jsonl(args.traces) if args.traces else None
                rows = build_sft(selected, args.db_root, args.mode, traces, allow_fixture=args.allow_fixture)
                write_jsonl(args.out, rows)
                result = {"rows": len(rows), "mode": args.mode, "sha256": digest(args.out),
                          "fixture_rows": sum(r["fixture"] for r in rows)}
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0
    except (ValueError, OSError, KeyError, TypeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
