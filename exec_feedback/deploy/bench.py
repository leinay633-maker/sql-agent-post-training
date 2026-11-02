"""Load test and demo: the analysis service (whole agent dialogues, POST /api/analyze) in front of vLLM's OpenAI-
compatible server, one deployment at a time on the whole GPU.

Per deployment and attempt (bench/<deployment>/attempt-<n>/, never overwritten): a warm-up on test tasks outside the
fixed set, then the fixed bench tasks at each concurrency level, the prefix cache reset before each level once vLLM shows
no request running or waiting (the dev-mode endpoint answers 200 whether or not it reset), so no level reuses another's
prompts. Per request, appended as it completes: the latency seen by the client, the outcome and each model call's own
usage. Per level: wall time, vLLM's token counters before and after, and the peak KV cache use, running requests and GPU
memory sampled while it ran. A level takes no new request after the deadline, or once its infrastructure failures (no
HTTP 200, or the service's model or database did not answer) pass the allowance; the deployment then has no result.
Correctness is scored afterwards, from the final SQL, by the report.

While servers run, their process groups are listed in <run>/.servers/<owner pid>.json, so the driver can stop them if
this process is killed before its own clean-up.
"""
from __future__ import annotations

import contextlib
import datetime as dt
import http.client
import json
import os
import re
import secrets
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable, Iterator

from ..storage import append_record, atomic_json, read_json
from ..v6.rollouts import versions
from . import SERVED
from .evaluate import engine_for
from .freeze import bench_tasks, load, stop_ids

OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))
COUNTERS = re.compile(r"^(vllm:[a-z_]+?)(_total)?(\{[^}]*\})? ([0-9.eE+-]+)$")
# the service's own answer when its model or database did not answer: infrastructure, not a model outcome
INFRA = {"provider_error", "schema_error"}
# without these the load test measures nothing: the deployment fails before its warm-up
REQUIRED = ("vllm:prompt_tokens", "vllm:generation_tokens", "vllm:num_requests_running", "vllm:num_requests_waiting")
SERVERS = ".servers"


def failed(row: dict[str, Any]) -> bool:
    """An infrastructure failure: no HTTP 200, or the service reports that its model or database did not answer."""
    return row.get("code") != 200 or row.get("status") in INFRA


def allowance(requests: int) -> int:
    """Infrastructure failures a level tolerates: 2% of its requests (none on the smoke run's 8)."""
    return requests // 50


def vllm_command(python: str, run: Path, frozen: dict[str, Any], condition: str, *, host: str = "127.0.0.1"
                 ) -> list[str]:
    """vLLM's OpenAI-compatible server for a deployment, with the evaluation engines' memory-independent settings:
    bf16, the context limit, the offline engine's token budget per step (the server's own default is half of it), no
    model-card sampling defaults (--generation-config vllm), prefix caching."""
    s = frozen["settings"]
    spec, adapter, quantization, _ = engine_for(run, frozen, condition)
    cmd = [python, "-m", "vllm.entrypoints.openai.api_server", "--model", spec["local_path"], "--dtype", "bfloat16",
           "--max-model-len", str(s["max_context"]), "--gpu-memory-utilization", str(s["serve_gpu_memory"]),
           "--max-num-seqs", str(s["serve_max_num_seqs"]), "--max-num-batched-tokens", str(s["serve_batched_tokens"]),
           "--generation-config", "vllm", "--enable-prefix-caching", "--seed", "20261029", "--host", host,
           "--port", str(s["vllm_port"])]
    if adapter is not None:  # requests name the LoRA module; the base is served under another name
        cmd += ["--served-model-name", "base", "--enable-lora", "--max-lora-rank", str(s["lora_r"]), "--max-loras", "1",
                "--lora-modules", f"{SERVED}={adapter}"]
    else:
        cmd += ["--served-model-name", SERVED]
    if quantization:
        cmd += ["--quantization", quantization]
    return cmd


def service_config(run: Path, frozen: dict[str, Any], *, host: str, port: int, results: Path,
                   concurrent: int) -> dict[str, Any]:
    """The analysis service in front of the vLLM server: the evaluation protocol (one result review, fenced JSON
    accepted), greedy, the evaluation engines' stop ids, every test database of the run."""
    s = frozen["settings"]
    _, tasks = load(run, "test", check_code=False)
    roots = {Path(t.database(Path(frozen["raw_root"]))).parent.parent for t in tasks}
    if len(roots) != 1:
        raise ValueError(f"The test databases are not under one folder: {sorted(map(str, roots))}")
    return {"model": {"backend": "http", "id": SERVED, "endpoint": f"http://127.0.0.1:{s['vllm_port']}/v1",
                      "max_tokens": s["max_tokens"], "temperature": 0.0, "seed": 42, "timeout": s["request_timeout"],
                      "extra_body": {"stop_token_ids": stop_ids(frozen["models"]["student"]["spec"])}},
            "service": {"host": host, "port": port, "database_root": str(roots.pop()), "results_root": str(results),
                        "adapter": None, "max_steps": s["max_steps"], "max_concurrent": concurrent,
                        "review_success": True, "fenced_json": True},
            "limits": s["limits"], "resources": {"max_model_calls": 1_000_000, "max_active_seconds": 604800}}


def get(url: str, timeout: float = 10, key: str = "") -> tuple[int | None, bytes]:
    request = urllib.request.Request(url, headers={"Authorization": "Bearer " + key} if key else {})
    try:
        with OPENER.open(request, timeout=timeout) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()
    except (OSError, ValueError):
        return None, b""


def post(url: str, body: dict[str, Any] | None, timeout: float, key: str = "") -> tuple[int | None, dict[str, Any]]:
    data = json.dumps(body or {}).encode()
    headers = {"Content-Type": "application/json", **({"Authorization": "Bearer " + key} if key else {})}
    try:
        with OPENER.open(urllib.request.Request(url, data=data, headers=headers), timeout=timeout) as response:
            raw = response.read()
            return response.status, json.loads(raw) if raw.strip() else {}
    except urllib.error.HTTPError as exc:
        try:
            return exc.code, json.loads(exc.read() or b"{}")
        except (ValueError, OSError, http.client.HTTPException):
            return exc.code, {}
    except (OSError, ValueError, http.client.HTTPException) as exc:  # e.g. a response cut off mid-read
        return None, {"error": type(exc).__name__}


def port_free(port: int) -> bool:
    """Nothing answers on the loopback port (a server left from an earlier attempt would be taken for this one)."""
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=1):
            return False
    except OSError:
        return True


def wait_ready(url: str, procs: list[subprocess.Popen], timeout: float, key: str = "") -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if any(p.poll() is not None for p in procs):
            raise RuntimeError(f"A server exited while starting ({url})")
        if get(url, key=key)[0] == 200:
            return
        time.sleep(2)
    raise RuntimeError(f"Not ready within {timeout:.0f} s: {url}")


def metrics(base: str) -> dict[str, float]:
    """vLLM's Prometheus counters and gauges, summed over labels (histogram parts left out)."""
    code, raw = get(base + "/metrics")
    out: dict[str, float] = {}
    if code != 200:
        return out
    for line in raw.decode("utf-8", "replace").splitlines():
        match = COUNTERS.match(line.strip())
        if match and not match.group(1).endswith(("_bucket", "_sum", "_count", "_created")):
            out[match.group(1)] = out.get(match.group(1), 0.0) + float(match.group(4))
    return out


def reset_prefix_cache(vllm: str, wait: float = 120.0) -> dict[str, Any]:
    """vLLM resets its prefix cache only while no request holds cache blocks, and answers 200 either way: wait (at most
    `wait` seconds) until its gauges show nothing running or waiting, then reset; the record says whether it was
    drained. The caller takes anything but a drained 200 as a failed reset."""
    deadline, drained = time.monotonic() + wait, False
    while True:
        m = metrics(vllm)
        drained = m.get("vllm:num_requests_running") == 0 and m.get("vllm:num_requests_waiting") == 0
        if drained or time.monotonic() > deadline:
            break
        time.sleep(1)
    return {"status": post(vllm + "/reset_prefix_cache", None, 60)[0], "drained": drained}


def gpu_memory_mib() -> int | None:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")[0].strip()
    cmd = ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"] + (["-i", visible] if visible else [])
    try:
        values = subprocess.run(cmd, capture_output=True, text=True, timeout=10).stdout.split()
        return max(int(v) for v in values) if values else None
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


class Peaks(threading.Thread):
    """From the start of a level and then every 2 s: KV cache use and running requests (vLLM gauges) and GPU memory."""

    def __init__(self, base: str) -> None:
        super().__init__(daemon=True)
        self.base, self.halt, self.lock, self.peak, self.samples = base, threading.Event(), threading.Lock(), {}, 0

    def sample(self) -> None:
        seen = {k: v for k, v in metrics(self.base).items()
                if k in ("vllm:kv_cache_usage_perc", "vllm:gpu_cache_usage_perc", "vllm:num_requests_running",
                         "vllm:num_requests_waiting")}
        memory = gpu_memory_mib()
        if memory is not None:
            seen["gpu_memory_mib"] = memory
        with self.lock:
            for k, v in seen.items():
                self.peak[k] = max(self.peak.get(k, v), v)
            self.samples += 1

    def run(self) -> None:
        while True:
            with contextlib.suppress(Exception):  # a failed sample is skipped, the next one tried
                self.sample()
            if self.halt.wait(2.0):
                return

    def finish(self) -> dict[str, Any]:
        """Stop sampling; the peaks so far and the number of samples (a sample can take up to 20 s)."""
        self.halt.set()
        self.join(25)
        with self.lock:
            return {**self.peak, "samples": self.samples}


def request(service: str, task: Any, timeout: float) -> dict[str, Any]:
    """One analysis through the service; whatever goes wrong on the way is a failed request, never a lost one."""
    started = time.monotonic()
    try:
        code, payload = post(service + "/api/analyze", {"database": task.db_id, "question": task.question,
                                                         "evidence": task.evidence}, timeout)
    except Exception as exc:  # noqa: BLE001 - recorded as an infrastructure failure of this request
        code, payload = None, {"error": f"{type(exc).__name__}: {exc}"[:300]}
    return {"task_id": task.id, "db_id": task.db_id, "code": code, "seconds": time.monotonic() - started,
            "status": payload.get("status"), "reason": payload.get("reason"), "final_sql": payload.get("final_sql"),
            "turns": len(payload.get("steps") or []), "calls": payload.get("calls"), "usage": payload.get("usage"),
            "run_id": payload.get("run_id"), "error": payload.get("error") or payload.get("detail")}


def covered(rows: list[dict[str, Any]], tasks: list[Any]) -> bool:
    """Every task answered exactly once."""
    ids = [r["task_id"] for r in rows]
    return len(ids) == len(set(ids)) == len(tasks) and set(ids) == {t.id for t in tasks}


def level(service: str, tasks: list[Any], concurrency: int, timeout: float, *, journal: Path | None = None,
          end: float | None = None, allowed: int | None = None) -> tuple[list[dict[str, Any]], float, str]:
    """The tasks through the service, `concurrency` clients taking them in the fixed order, each row appended to
    `journal` as it completes. No new request after `end` (epoch seconds), once more than `allowed` infrastructure
    failures happened, or once the journal could not be written. Returns the rows in task order, the wall time and why
    it stopped: done, deadline, failures or "journal: <error>"."""
    pending, rows, lock = list(enumerate(tasks)), {}, threading.Lock()
    state = {"stopped": "done", "failures": 0}

    def take() -> tuple[int, Any] | None:
        with lock:
            if state["stopped"] != "done" or not pending:
                return None
            if end is not None and time.time() > end:
                state["stopped"] = "deadline"
                return None
            return pending.pop(0)

    def client() -> None:
        while (item := take()) is not None:
            row = request(service, item[1], timeout)
            with lock:
                rows[item[0]] = row
                try:
                    if journal is not None:
                        append_record(journal, row)
                except OSError as exc:  # a row the journal lost: the level stops and its attempt fails
                    state["stopped"] = f"journal: {type(exc).__name__}: {exc}"[:300]
                if failed(row):
                    state["failures"] += 1
                    if allowed is not None and state["failures"] > allowed and state["stopped"] == "done":
                        state["stopped"] = "failures"

    started = time.monotonic()
    clients = [threading.Thread(target=client, daemon=True) for _ in range(max(1, min(concurrency, len(tasks))))]
    for thread in clients:
        thread.start()
    for thread in clients:  # daemon threads: an interrupt here leaves them to the servers' shutdown
        thread.join()
    return [rows[i] for i in sorted(rows)], time.monotonic() - started, state["stopped"]


def proc_state(pid: int) -> tuple[str, int] | None:
    """(state letter, start time in clock ticks) of a process from Linux's /proc; None if there is no such process."""
    try:
        raw = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    fields = raw[raw.rindex(")") + 2:].split()  # after "pid (comm) ": state is field 3, pgrp 5, starttime 22
    return fields[0], int(fields[19])


def owner_alive(record: dict[str, Any]) -> bool:
    """A ledger's owner still runs: it exists, is not a zombie (killed, never reaped), and is the same process (its
    start time, where recorded, rules out a reused pid)."""
    state = proc_state(record["owner"])
    return state is not None and state[0] != "Z" and record.get("owner_start") in (None, state[1])


def alive_group(group: int) -> bool:
    """A process group still has a live member; zombies (dead, waiting to be reaped) do not count."""
    proc = Path("/proc")
    if not proc.is_dir():
        try:
            os.killpg(group, 0)
            return True
        except OSError:  # gone, or not ours
            return False
    for entry in proc.iterdir():
        if entry.name.isdigit():
            try:
                raw = (entry / "stat").read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            fields = raw[raw.rindex(")") + 2:].split()
            if int(fields[2]) == group and fields[0] != "Z":
                return True
    return False


def stop_groups(groups: list[int], procs: list[subprocess.Popen] = (), grace: float = 20.0) -> None:
    """SIGTERM every process group, SIGKILL whatever is left after `grace` seconds; own children are reaped. Elsewhere
    (no process groups) the processes themselves are terminated."""
    if not hasattr(os, "killpg"):
        for proc in procs:
            with contextlib.suppress(OSError):
                proc.terminate()
        for proc in procs:
            with contextlib.suppress(subprocess.TimeoutExpired):
                proc.wait(grace)
        return
    for sig, wait in ((signal.SIGTERM, grace), (signal.SIGKILL, 10.0)):
        for group in groups:
            with contextlib.suppress(OSError):
                os.killpg(group, sig)
        deadline = time.monotonic() + wait
        while time.monotonic() < deadline:
            for proc in procs:
                proc.poll()  # reap own children
            if not any(alive_group(g) for g in groups):
                return
            time.sleep(0.5)


def reap_servers(run: Path) -> list[str]:
    """Stop the server process groups of a bench or demo that was killed before its own clean-up (its owner is gone or
    a zombie). A ledger is removed only once none of its groups has a live process. Linux only, where the servers run
    (elsewhere os.kill(pid, 0) would end the process instead of probing it)."""
    if not hasattr(os, "killpg") or not Path("/proc").is_dir():
        return []
    notes = []
    for path in sorted((run / SERVERS).glob("*.json")):
        record = read_json(path)
        if owner_alive(record):
            continue  # its servers are its own business
        left = [g for g in record["groups"] if alive_group(g)]
        if left:
            stop_groups(left)
            still = [g for g in left if alive_group(g)]
            if still:
                notes.append(f"WARNING: server process groups {still} left by process {record['owner']} survive SIGKILL")
                continue
            notes.append(f"stopped server process groups {left} left by process {record['owner']}")
        path.unlink()
    return notes


@contextlib.contextmanager
def servers(run: Path, frozen: dict[str, Any], condition: str, *, vllm_python: str, main_python: str, logs: Path,
            host: str, port: int, results: Path, concurrent: int, key: str = "") -> Iterator[tuple[str, str]]:
    """vLLM's server and the analysis service in their own process groups, listed in the run's server ledger while they
    live, both stopped on the way out. With a key both require it: vLLM through VLLM_API_KEY, the service through
    EFA_SERVER_KEY, and the service sends it to vLLM (EFA_API_KEY); never on a command line."""
    s = frozen["settings"]
    busy = [p for p in (s["vllm_port"], port) if not port_free(p)]
    if busy:
        raise RuntimeError(f"Ports {busy} are in use: a server from an earlier attempt may still run")
    logs.mkdir(parents=True, exist_ok=True)
    config = logs / f"service-{condition}.json"
    atomic_json(config, service_config(run, frozen, host=host, port=port, results=results, concurrent=concurrent))
    ledger = run / SERVERS / f"{os.getpid()}.json"
    ledger.parent.mkdir(exist_ok=True)
    procs: list[subprocess.Popen] = []

    me = proc_state(os.getpid())

    def start(cmd: list[str], env: dict[str, str], log: Path) -> None:
        with log.open("ab") as fh:
            procs.append(subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT, env=env, start_new_session=True))
        atomic_json(ledger, {"owner": os.getpid(), "owner_start": me[1] if me else None,
                             "groups": [p.pid for p in procs], "condition": condition})

    try:
        secret = {"VLLM_API_KEY": key} if key else {}
        start(vllm_command(vllm_python, run, frozen, condition), {**os.environ, "VLLM_SERVER_DEV_MODE": "1", **secret},
              logs / f"vllm-{condition}.log")  # dev mode: /reset_prefix_cache
        vllm = f"http://127.0.0.1:{s['vllm_port']}"
        wait_ready(vllm + "/v1/models", procs, 1200, key)
        secret = {"EFA_SERVER_KEY": key, "EFA_API_KEY": key} if key else {}
        start([main_python, "-m", "exec_feedback.deploy", "api", "--config", str(config)], {**os.environ, **secret},
              logs / f"service-{condition}.log")
        service = f"http://127.0.0.1:{port}"
        wait_ready(service + "/healthz", procs, 120)
        yield vllm, service
    finally:
        stop_groups([p.pid for p in procs], procs)
        with contextlib.suppress(FileNotFoundError):
            ledger.unlink()


def bench(run: Path, *, condition: str, vllm_python: str = sys.executable, main_python: str = sys.executable,
          until: str | None = None, launch: Callable[..., Any] = servers) -> dict[str, Any]:
    frozen, test = load(run, "test")
    s = frozen["settings"]
    out = run / "bench" / condition
    if (out / "result.json").exists():
        return read_json(out / "result.json")
    out.mkdir(parents=True, exist_ok=True)
    attempt = 1 + max((int(p.name.split("-")[1]) for p in out.glob("attempt-*")), default=0)
    here = out / f"attempt-{attempt}"
    here.mkdir()
    ordered = bench_tasks(test, s["salt"], len(test))
    fixed, warm = ordered[:s["bench_tasks"]], ordered[s["bench_tasks"]:s["bench_tasks"] + s["warmup_tasks"]]
    if [t.id for t in fixed] != frozen["bench"]["tasks"]:
        raise ValueError("The load test's tasks differ from the frozen ones")
    end = dt.datetime.fromisoformat(until).timestamp() if until else None
    timeout = float(s["request_timeout"]) * (s["max_steps"] + 1)
    started = time.monotonic()
    levels: dict[str, Any] = {}
    failure = stopped = loaded = None
    warmup: dict[str, Any] = {"tasks": len(warm)}
    record = {"run": frozen["identity"], "condition": condition, "attempt": attempt, "tasks": len(fixed),
              "status": "running", "started": dt.datetime.now(dt.timezone.utc).isoformat(),
              "gpu": os.environ.get("CUDA_VISIBLE_DEVICES")}
    atomic_json(here / "attempt.json", record)  # stays "running" only if the process was killed outright
    try:
        with launch(run, frozen, condition, vllm_python=vllm_python, main_python=main_python, logs=here / "logs",
                    host="127.0.0.1", port=s["service_port"], results=here / "service",
                    concurrent=s["concurrency"][-1]) as (vllm, service):
            loaded = time.monotonic() - started
            missing = [k for k in REQUIRED if k not in metrics(vllm)]
            if missing:
                failure = f"vLLM's metrics lack {missing}"
            else:
                rows, wall, why = level(service, warm, min(len(warm), s["concurrency"][-1]), timeout,
                                        journal=here / "warmup.jsonl", end=end, allowed=0)
                warmup.update(wall_seconds=wall, answered=len(rows), failures=sum(failed(r) for r in rows))
                if why == "deadline":
                    stopped = "deadline"
                elif warmup["failures"] or why != "done" or not covered(rows, warm):
                    failure = (f"the warm-up answered {len(rows)} of {len(warm)} tasks, {warmup['failures']} with an "
                               f"infrastructure failure ({why})")
            for concurrency in s["concurrency"]:
                if failure or stopped:
                    break
                if end is not None and time.time() > end:
                    stopped = "deadline"
                    break
                reset = reset_prefix_cache(vllm, 120.0 if end is None else max(5.0, min(120.0, end - time.time())))
                if reset["status"] != 200 or not reset["drained"]:  # a warm cache would carry the last level's prompts
                    failure = f"the prefix cache was not reset before concurrency {concurrency}: {reset}"
                    break
                before, peaks = metrics(vllm), Peaks(vllm)
                peaks.start()
                try:
                    rows, wall, why = level(service, fixed, concurrency, timeout,
                                            journal=here / f"level-{concurrency}.jsonl", end=end,
                                            allowed=allowance(len(fixed)))
                finally:
                    peak = peaks.finish()
                if why == "done" and not covered(rows, fixed):
                    why = "incomplete"
                data = {"concurrency": concurrency, "wall_seconds": wall, "prefix_cache_reset": reset,
                        "metrics_before": before, "metrics_after": metrics(vllm), "peaks": peak, "stopped": why,
                        "rows": rows}
                atomic_json(here / f"level-{concurrency}.json", data)
                if why == "done":
                    levels[str(concurrency)] = data
                elif why == "deadline":
                    stopped = "deadline"
                elif why == "failures":
                    failure = f"{sum(failed(r) for r in rows)} infrastructure failures at concurrency {concurrency}"
                elif why == "incomplete":
                    failure = f"concurrency {concurrency} answered {len(rows)} of {len(fixed)} tasks"
                else:
                    failure = f"concurrency {concurrency} stopped: {why}"
    except BaseException as exc:  # servers that never came up, an interrupt, a crash: the attempt still ends recorded
        atomic_json(here / "attempt.json", {
            **record, "status": "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed", "complete": False,
            "failure": f"{type(exc).__name__}: {exc}"[:500], "load_seconds": loaded, "warmup": warmup,
            "levels": sorted(levels, key=int)})
        raise
    complete = failure is None and stopped is None and len(levels) == len(s["concurrency"])
    result = {**record, "status": "complete" if complete else "stopped" if stopped else "failed",
              "load_seconds": loaded, "warmup": warmup, "levels": sorted(levels, key=int), "complete": complete,
              "failure": failure, "stopped": stopped, "packages": versions()}
    atomic_json(here / "attempt.json", result)
    if complete:
        atomic_json(out / "result.json", result)
    return result


def write_secret(path: Path, value: str) -> None:
    """Created readable by the owner only, also while it is written; an earlier file or link there is replaced, never
    followed."""
    with contextlib.suppress(FileNotFoundError):
        path.unlink()
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.write(value + "\n")


def demo(run: Path, *, condition: str, minutes: float, port: int, vllm_python: str, main_python: str,
         key_file: Path) -> dict[str, Any]:
    """The service on every interface for a screen recording: a fresh key (at least 24 characters, never logged) in
    `key_file`, readable by the owner only, required by both servers; both stop after `minutes`."""
    frozen, _ = load(run)
    key = os.environ.get("EFA_SERVER_KEY") or secrets.token_urlsafe(24)
    write_secret(key_file, key)
    out = run / "demo" / condition
    started = time.time()
    with servers(run, frozen, condition, vllm_python=vllm_python, main_python=main_python, logs=out / "logs",
                 host="0.0.0.0", port=port, results=out / "service", concurrent=4, key=key):
        atomic_json(out / "open.json", {"port": port, "opened": dt.datetime.now(dt.timezone.utc).isoformat(),
                                        "minutes": minutes})
        time.sleep(max(0.0, minutes * 60 - (time.time() - started)))
    return {"condition": condition, "port": port, "minutes": minutes,
            "requests": len(list((out / "service").glob("*.json"))) if (out / "service").exists() else 0}
