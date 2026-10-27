"""v0.12 teacher scoring: v0.8's FP8 235B in vLLM over teacher_gpus GPUs, loaded once and shared by every OPD run.

score(sequences) returns, for each token-id sequence, the teacher's log-probability of every token given the tokens
before it (vLLM prompt log-probs of the raw logits; position 0 has none). The ids go in as they are: student and teacher
share the tokenizer files (checked at the freeze), and vLLM's echo of the prompt ids is compared with what was sent.

The service is a local HTTP endpoint around one engine, which stays on the thread that built it: runs post their
dialogues, the HTTP threads queue them by priority, and that thread scores one piece at a time, so a lower-priority
request (queued in pieces) holds up a higher-priority one by at most one piece. Before it serves, the gate has the
teacher sample continuations of real first-turn prompts and score them again: the two log-probs of each sampled token
must agree (mean |difference| below teacher_gate_max_gap), which catches misaligned positions, a wrong template or
broken prompt log-probs, not kernel noise; a restart after a passed gate keeps that result. Records: gate.json,
serving.json (port) while serving, requests.jsonl, served.json; `over`, written by the scheduler, once no service will
come back.
"""
from __future__ import annotations

import contextlib
import hashlib
import itertools
import json
import math
import os
import queue
import subprocess
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable

from ..core import Task
from ..storage import append_record, atomic_json, fingerprint, read_json

LOW_PRIORITY_CHUNK = 32   # a lower-priority request is scored in pieces, so a queued main request waits at most one


class TeacherMismatch(ValueError):
    """The teacher returned log-probs for other tokens than the ones sent, or not one per position."""


class TeacherUnavailable(RuntimeError):
    """No teacher answered within the wait budget, or it stopped serving."""


class EngineFailure(RuntimeError):
    """The engine raised while scoring: the service answers 503 and ends, and the scheduler starts a fresh one."""


class ServiceClosed(RuntimeError):
    """The service ended while the request was queued: the caller retries against the next one."""


def identity_of(frozen: dict[str, Any]) -> str:
    return fingerprint(frozen["models"]["teacher"]["identity"])


def gpu_used_mib() -> list[int] | None:
    """Memory in use on this process's visible GPUs, all processes counted (the tensor-parallel workers hold the
    teacher, so this process's own allocator cannot tell); None when nvidia-smi cannot answer."""
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if not visible:
        return None
    try:
        done = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits", "-i", visible],
                              capture_output=True, text=True, timeout=20)
        return [int(v) for v in done.stdout.split()] if done.returncode == 0 else None
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def token_logprob(entry: Any, token: int) -> float | None:
    """vLLM keeps a dict per position (token id -> Logprob) unless VLLM_FLAT_LOGPROBS is set; both answer .get()."""
    item = entry.get(token) if entry is not None and hasattr(entry, "get") else None
    return float(item.logprob) if item is not None else None


def aligned(sent: list[int], returned: list[int] | None, prompt_logprobs: Any) -> list[float | None]:
    if returned is not None and list(returned) != list(sent):
        raise TeacherMismatch("The teacher scored other token ids than the ones sent")
    if prompt_logprobs is None or len(prompt_logprobs) != len(sent):
        raise TeacherMismatch("The teacher did not return one log-prob per position")
    return [None] + [token_logprob(prompt_logprobs[p], sent[p]) for p in range(1, len(sent))]


class TeacherEngine:
    """The 235B as v0.8 ran it (4-way tensor parallel, piecewise CUDA graphs), sized for scoring whole dialogues."""

    def __init__(self, spec: dict[str, Any], settings: dict[str, Any], *, seed: int = 20261022) -> None:
        from vllm import LLM

        visible = [g for g in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if g.strip()]
        if len(visible) != settings["teacher_gpus"]:
            raise RuntimeError(f"The teacher needs exactly {settings['teacher_gpus']} visible GPUs, got {visible}")
        self.llm = LLM(model=spec["local_path"], tokenizer=spec["local_path"], dtype="bfloat16",
                       max_model_len=settings["teacher_score_context"],
                       gpu_memory_utilization=settings["teacher_score_gpu_memory"], enable_prefix_caching=True,
                       seed=seed, generation_config="vllm", max_num_seqs=settings["teacher_score_max_num_seqs"],
                       max_num_batched_tokens=settings["teacher_score_batched_tokens"], trust_remote_code=False,
                       tensor_parallel_size=settings["teacher_gpus"],
                       # v0.8 (2026-10-12 H20): the full decode-graph capture ran out of memory
                       compilation_config={"cudagraph_mode": "PIECEWISE"})

    def score(self, sequences: list[list[int]]) -> list[list[float | None]]:
        from vllm import SamplingParams
        from vllm.inputs import TokensPrompt

        # prompt log-probs make vLLM skip reading the prefix cache, so every position is computed from its prefix
        params = SamplingParams(n=1, temperature=0.0, max_tokens=1, prompt_logprobs=0, detokenize=False)
        outputs = self.llm.generate([TokensPrompt(prompt_token_ids=list(s)) for s in sequences], params,
                                    use_tqdm=False)
        if len(outputs) != len(sequences):
            raise TeacherMismatch("The teacher returned a different number of sequences")
        rows = []
        for s, o in zip(sequences, outputs):
            try:
                rows.append(aligned(s, o.prompt_token_ids, o.prompt_logprobs))
            except TeacherMismatch:  # preregistered: that dialogue stays out of the gradient and is counted
                rows.append([None] * len(s))
        return rows

    def sample(self, prompts: list[list[int]], *, max_tokens: int, stop_ids: list[int]) -> list[dict[str, Any]]:
        from vllm import SamplingParams
        from vllm.inputs import TokensPrompt

        params = SamplingParams(n=1, temperature=1.0, top_p=1.0, max_tokens=max_tokens, stop_token_ids=stop_ids,
                                logprobs=0, detokenize=False)
        outputs = self.llm.generate([TokensPrompt(prompt_token_ids=p) for p in prompts], params, use_tqdm=False)
        samples = []
        for out in outputs:
            choice = out.outputs[0]
            tokens = list(choice.token_ids)
            logprobs = choice.logprobs or []
            samples.append({"tokens": tokens,
                            "logprobs": [token_logprob(logprobs[i] if i < len(logprobs) else None, t)
                                         for i, t in enumerate(tokens)]})
        return samples


# ------------------------------------------------------------------------------------------------ the gate
def gate_prompts(tasks: list[Task], root: Path, tokenizer: Any, settings: dict[str, Any], count: int) -> list[list[int]]:
    """First-turn prompts of pool tasks, as the runs render them: half of them the longest that fit the student's
    context (so prefills run over several chunks), half in a fixed hash order."""
    from ..agent import dialogue_stream
    from ..core import Limits

    limits = Limits(**settings["limits"])
    order = sorted(tasks, key=lambda t: hashlib.sha256(f"gate:{t.id}".encode()).digest())[:max(256, count)]
    prompts = []
    for task in order:
        stream = dialogue_stream(task.database(root), task.question, task.evidence, max_steps=settings["max_steps"],
                                 feedback=True, limits=limits, review_success=True, fenced_json=True)
        try:
            messages = next(stream)
        finally:
            stream.close()
        ids = list(tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True))
        if len(ids) + settings["max_tokens"] <= settings["max_context"]:
            prompts.append(ids)
    longest = sorted(prompts, key=len, reverse=True)[:count // 2]
    rest = [p for p in prompts if not any(p is q for q in longest)][:count - len(longest)]
    return longest + rest


def gate(engine: Any, prompts: list[list[int]], *, max_tokens: int, stop_ids: list[int],
         max_gap: float) -> dict[str, Any]:
    """The teacher samples a continuation of every prompt (temperature 1, no truncation, its own log-probs kept), then
    the scoring path scores prompt + continuation; the sampled tokens' two log-probs must agree."""
    started = time.monotonic()
    samples = engine.sample(prompts, max_tokens=max_tokens, stop_ids=stop_ids)
    sampled = time.monotonic()
    sequences = [list(p) + list(s["tokens"]) for p, s in zip(prompts, samples)]
    scored = engine.score(sequences)
    seconds = time.monotonic() - sampled
    gaps: list[float] = []
    missing = 0
    for prompt, sample, row in zip(prompts, samples, scored):
        for i, value in enumerate(sample["logprobs"]):
            other = row[len(prompt) + i]
            if value is None or other is None or not math.isfinite(value) or not math.isfinite(other):
                missing += 1
            else:
                gaps.append(abs(value - other))
    mean_gap = sum(gaps) / len(gaps) if gaps else None  # None, not inf: the record must stay valid JSON
    passed = bool(gaps) and not missing and mean_gap < max_gap
    return {"passed": passed, "sequences": len(sequences), "sampled_tokens": len(gaps) + missing, "missing": missing,
            "mean_abs_gap": mean_gap, "max_abs_gap": max(gaps) if gaps else None, "max_gap": max_gap,
            "prompt_tokens": [min(map(len, prompts)), max(map(len, prompts))] if prompts else None,
            "score_tokens": sum(map(len, sequences)), "score_seconds": seconds,
            "sample_seconds": sampled - started}


# ------------------------------------------------------------------------------------------------ the service
class Work:
    """One scoring call: an HTTP thread queues it and waits; the engine's thread runs it."""

    def __init__(self, sequences: list[list[int]]) -> None:
        self.sequences = sequences
        self.done = threading.Event()
        self.rows: list[list[float | None]] | None = None
        self.error: BaseException | None = None
        self.seconds = 0.0


class Service:
    """vLLM's client keeps its sockets on the thread that built the engine, so every engine call runs on that thread
    (run()); the HTTP threads only queue work and wait. Lower numbers go first, equal ones in arrival order."""

    def __init__(self, engine: Any, identity: str, home: Path, max_len: int) -> None:
        self.engine, self.identity, self.home, self.max_len = engine, identity, home, max_len
        self.queue: queue.PriorityQueue[tuple[int, int, Work]] = queue.PriorityQueue()
        self.order = itertools.count()
        self.lock = threading.Lock()
        self.closed = False
        self.counts = {"requests": 0, "sequences": 0, "tokens": 0, "busy_seconds": 0.0, "errors": 0}
        self.failed: str | None = None

    def score(self, payload: dict[str, Any]) -> dict[str, Any]:
        """An HTTP thread: check the request, queue its pieces, wait for them."""
        if payload.get("identity") != self.identity:
            raise ValueError("Request for another teacher")
        sequences = payload.get("sequences")
        priority = int(payload.get("priority", 0))
        if priority not in (0, 1) or not isinstance(sequences, list) or not sequences or any(
                not isinstance(s, list) or not 1 < len(s) < self.max_len or not all(type(t) is int for t in s)
                for s in sequences):
            raise ValueError("Expected 'sequences': non-empty token-id lists shorter than the teacher's context")
        chunks = [sequences] if priority == 0 else [sequences[i:i + LOW_PRIORITY_CHUNK]
                                                    for i in range(0, len(sequences), LOW_PRIORITY_CHUNK)]
        works = [Work(chunk) for chunk in chunks]
        started = time.monotonic()
        with self.lock:
            if self.closed:
                raise ServiceClosed("The teacher is not serving")
            for work in works:
                self.queue.put((priority, next(self.order), work))
        out: list[list[float | None]] = []
        busy = 0.0
        for work in works:
            work.done.wait()
            if work.error is not None:
                raise work.error
            # NaN/inf cannot travel as JSON; the caller treats None as missing (its dialogue is not trained)
            out.extend([v if v is None or math.isfinite(v) else None for v in row] for row in work.rows or [])
            busy += work.seconds
        waited = time.monotonic() - started - busy
        tokens = sum(map(len, sequences))
        used = gpu_used_mib()  # after the call: the workers' caching allocators keep what the prompt log-probs took
        record = {"time": time.time(), "caller": str(payload.get("caller", ""))[:80], "priority": priority,
                  "sequences": len(sequences), "tokens": tokens, "wait_seconds": waited, "score_seconds": busy,
                  "incomplete": sum(1 for row in out if any(v is None for v in row[1:])), "gpu_used_mib": used}
        with self.lock:  # one writer at a time: two HTTP threads appending at once can tear a line
            self.counts["requests"] += 1
            self.counts["sequences"] += len(sequences)
            self.counts["tokens"] += tokens
            self.counts["busy_seconds"] += busy
            append_record(self.home / "requests.jsonl", record)
        return {"logprobs": out, "score_seconds": busy, "wait_seconds": waited}

    def run(self, keep_serving: Callable[[], bool], poll: float) -> str:
        """The engine's thread: one queued piece at a time until keep_serving() turns false or the engine fails; then
        everything still queued, and anything new, is answered 503."""
        stopped_by = "stop"
        while keep_serving():
            try:
                _, _, work = self.queue.get(timeout=poll)
            except queue.Empty:
                continue
            started = time.monotonic()
            try:
                work.rows = self.engine.score(work.sequences)
            except ValueError as exc:  # this request (vLLM refused it): answered 400, the engine goes on
                work.error = exc
            except Exception as exc:  # noqa: BLE001 - the engine is suspect now: answer, end, a fresh service follows
                self.failed = f"{type(exc).__name__}: {str(exc)[:300]}"
                work.error = EngineFailure(self.failed)
            if work.error is not None:
                with self.lock:
                    self.counts["errors"] += 1
            work.seconds = time.monotonic() - started
            work.done.set()
            if self.failed:
                stopped_by = "engine_error"
                break
        with self.lock:
            self.closed = True
        while True:
            try:
                _, _, work = self.queue.get_nowait()
            except queue.Empty:
                break
            work.error = ServiceClosed("The teacher stopped serving")
            work.done.set()
        return stopped_by


def _handler(service: Service) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args: Any) -> None:  # requests.jsonl is the log
            return

        def reply(self, code: int, body: dict[str, Any]) -> None:
            data = json.dumps(body, allow_nan=False).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self) -> None:  # noqa: N802
            if self.path != "/health":
                return self.reply(404, {"error": "not found"})
            return self.reply(200, {"ok": service.failed is None, "identity": service.identity, **service.counts})

        def do_POST(self) -> None:  # noqa: N802
            if self.path != "/score":
                return self.reply(404, {"error": "not found"})
            try:
                payload = json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0"))))
            except (ValueError, OSError) as exc:
                return self.reply(400, {"error": f"bad request: {exc}"})
            try:
                body = service.score(payload)
            except (ValueError, TypeError) as exc:
                return self.reply(400, {"error": str(exc)[:500]})
            except Exception as exc:  # noqa: BLE001 - the engine failed or the service is ending: the caller retries
                return self.reply(503, {"error": f"{type(exc).__name__}: {str(exc)[:300]}"})
            return self.reply(200, body)
    return Handler


def serve(engine: Any, *, identity: str, home: Path, max_len: int, keep_serving: Callable[[], bool],
          poll: float = 5.0) -> dict[str, Any]:
    """Serve on 127.0.0.1 until keep_serving() turns false or the engine fails; serving.json carries the port. Call it
    on the thread that built the engine: that thread runs every scoring call."""
    service = Service(engine, identity, home, max_len)
    server = ThreadingHTTPServer(("127.0.0.1", 0), _handler(service))
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    started = time.time()
    atomic_json(home / "serving.json", {"port": server.server_address[1], "pid": os.getpid(), "identity": identity,
                                        "started": started})
    try:
        stopped_by = service.run(keep_serving, poll)
    finally:
        with contextlib.suppress(OSError):
            (home / "serving.json").unlink()
        server.shutdown()
        server.server_close()
    return {"stopped_by": stopped_by, "error": service.failed, "serving_seconds": time.time() - started,
            **service.counts}


def run_service(run: Path, *, home: Path, until: str | None, engine_factory: Callable[..., Any] = TeacherEngine,
                tokenizer: Any = None, poll: float = 5.0) -> dict[str, Any]:
    """Load the teacher of `run` (the main run; the smoke run pins the same one), pass the gate, serve until `home/stop`
    appears or `until` passes. gate.json and served.json record what happened; a failed gate never serves. A restart
    after an engine error keeps a gate this run's teacher passed: a second sampling could only turn a passed check
    into a stop for every run."""
    from ..models import load_tokenizer
    from ..v6.rollouts import deadline_from
    from ..v7.engine import stop_ids
    from .freeze import load, model_ready

    frozen, pool = load(run, "pool")
    s = frozen["settings"]
    home.mkdir(parents=True, exist_ok=True)
    if (home / "served.json").exists():
        return read_json(home / "served.json")
    real = engine_factory is TeacherEngine
    spec = model_ready(frozen, "teacher") if real else frozen["models"]["teacher"]["spec"]
    identity = identity_of(frozen)
    deadline = deadline_from(until)
    started = time.monotonic()
    engine = engine_factory(spec, s)
    load_seconds = time.monotonic() - started
    previous = _read(home / "gate.json")
    if previous and previous.get("passed") and previous.get("identity") == identity \
            and previous.get("run") == frozen["identity"]:
        check = previous
        append_record(home / "gate-attempts.jsonl", {"kept": "passed before", "engine_load_seconds": load_seconds,
                                                     "gpu": os.environ.get("CUDA_VISIBLE_DEVICES"),
                                                     "gpu_used_mib": gpu_used_mib(), "time": time.time()})
    else:
        tokenizer = tokenizer if tokenizer is not None else load_tokenizer(spec)
        prompts = gate_prompts(pool, Path(frozen["raw_root"]), tokenizer, s, s["teacher_gate_samples"])
        check = {**gate(engine, prompts, max_tokens=s["max_tokens"], stop_ids=stop_ids(spec, tokenizer),
                        max_gap=s["teacher_gate_max_gap"]),
                 "identity": identity, "run": frozen["identity"], "engine_load_seconds": load_seconds,
                 "gpu": os.environ.get("CUDA_VISIBLE_DEVICES"), "gpu_used_mib": gpu_used_mib(), "time": time.time()}
        append_record(home / "gate-attempts.jsonl", check)
        atomic_json(home / "gate.json", check)
        if not check["passed"]:
            return {"served": False, "gate": check}
    keep = lambda: not (home / "stop").exists() and (deadline is None or time.monotonic() < deadline)
    record = {"served": True, **serve(engine, identity=identity, home=home, max_len=s["teacher_score_context"],
                                      keep_serving=keep, poll=poll), "gate": check}
    if record["stopped_by"] != "engine_error":  # an engine error restarts the service (the scheduler retries it)
        record["stopped_by"] = "stop" if (home / "stop").exists() else "deadline"
        atomic_json(home / "served.json", record)
    return record


# ------------------------------------------------------------------------------------------------ the client
class TeacherClient:
    """What an OPD run calls: score(sequences) through the service, waiting for it (it may be loading, restarting after
    a failure or busy with other runs) up to `wait` seconds per call, and never past `until` (a time.monotonic() value:
    the run's own deadline). A lower-priority run is held back whenever a main request is queued, so it has no `wait`
    of its own: it waits up to its deadline."""

    def __init__(self, home: Path, identity: str, *, wait: float | None, priority: int = 0, caller: str = "",
                 poll: float = 10.0, until: float | None = None) -> None:
        if wait is None and until is None:
            raise ValueError("A teacher client needs a wait or a deadline")
        self.home, self.identity, self.wait, self.priority, self.caller, self.poll = home, identity, wait, priority, \
            caller, poll
        self.until = until
        self.last: dict[str, float] = {}

    def gone(self) -> bool:
        """No service will answer again: it stopped for good, or it failed for the last time."""
        return (self.home / "served.json").exists() or (self.home / "over").exists()

    def score(self, sequences: list[list[int]]) -> list[list[float | None]]:
        deadline = min(t for t in (None if self.wait is None else time.monotonic() + self.wait, self.until)
                       if t is not None)
        last_error = "the teacher is not serving yet"
        body = json.dumps({"identity": self.identity, "sequences": sequences, "priority": self.priority,
                           "caller": self.caller}).encode()
        while True:
            if (self.home / "over").exists():
                raise TeacherUnavailable("The teacher service failed for the last time")
            serving = _read(self.home / "serving.json")
            if serving and serving.get("identity") == self.identity:
                request = urllib.request.Request(f"http://127.0.0.1:{serving['port']}/score", data=body,
                                                 headers={"Content-Type": "application/json"})
                try:
                    with urllib.request.urlopen(request, timeout=max(30.0, deadline - time.monotonic())) as response:
                        answer = json.loads(response.read())
                    rows = answer["logprobs"]
                    if len(rows) != len(sequences) or any(len(r) != len(s) for r, s in zip(rows, sequences)):
                        raise TeacherMismatch("The service returned rows that do not match the sequences sent")
                    self.last = {"score_seconds": answer.get("score_seconds", 0.0),
                                 "wait_seconds": answer.get("wait_seconds", 0.0)}
                    return rows
                except urllib.error.HTTPError as exc:
                    if exc.code == 400:
                        raise ValueError(f"The teacher refused the request: {exc.read()[:300]!r}") from exc
                    last_error = f"HTTP {exc.code}"
                except (urllib.error.URLError, OSError, ValueError) as exc:
                    if isinstance(exc, TeacherMismatch):
                        raise
                    last_error = f"{type(exc).__name__}: {str(exc)[:200]}"
            elif (self.home / "served.json").exists():
                raise TeacherUnavailable("The teacher has stopped serving")
            if time.monotonic() > deadline:
                raise TeacherUnavailable(f"No teacher answer in time ({last_error})")
            time.sleep(self.poll)


def _read(path: Path) -> dict[str, Any] | None:
    try:
        return read_json(path)
    except (OSError, ValueError):
        return None
