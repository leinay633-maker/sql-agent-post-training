"""Deployment-study engineering tests: the passk fixture run's registration and imports, a fake OpenAI-compatible model
server on loopback (scripted actions, vLLM's metrics and prefix-cache endpoints), the real analysis service in-process,
and test_v09's scripted greedy generator; no GPU, vLLM or real weights. Nothing here is a model result."""
from __future__ import annotations

import contextlib
import copy
import datetime as dt
import functools
import json
import os
import shutil
import socket
import stat
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_passk import fixture_run  # noqa: E402,F401  (the registration, the 4B pin and v0.11's imports)
from test_v09 import GOLD, ChatTokenizer, factory as scripted_factory  # noqa: E402

from exec_feedback.agent import dialogue  # noqa: E402
from exec_feedback.pipeline import HTTPPolicy  # noqa: E402
from exec_feedback.storage import read_json  # noqa: E402
from exec_feedback.deploy import CONDITIONS, MODEL, SERVED  # noqa: E402
from exec_feedback.deploy import analysis, bench as bm, freeze as fm  # noqa: E402
from exec_feedback.deploy.config import DEFAULT, validate_settings  # noqa: E402
from exec_feedback.deploy.evaluate import engine_for, evaluate  # noqa: E402
from exec_feedback.v10.merge import record_of  # noqa: E402
from exec_feedback.v10.weights import dir_identity  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
WRONG = "SELECT max(v) FROM items"


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def smoke_settings() -> dict:
    settings = json.loads((REPO / "configs" / "deploy-smoke.json").read_text(encoding="utf-8"))
    settings.update(vllm_port=free_port(), service_port=free_port())
    return settings


# ------------------------------------------------------------------------------------------------ settings
def test_registered_settings_and_configs():
    validate_settings(copy.deepcopy(DEFAULT))
    assert json.loads((REPO / "configs" / "h20-deploy.json").read_text(encoding="utf-8")) == DEFAULT
    smoke = json.loads((REPO / "configs" / "deploy-smoke.json").read_text(encoding="utf-8"))
    validate_settings(smoke)
    assert smoke["phase"] == "smoke" and smoke["bench_tasks"] + smoke["warmup_tasks"] <= 16
    for bad in ({"roles": ["validation"]}, {"concurrency": [4, 1]}, {"concurrency": [1, 512]}, {"max_steps": 6},
                {"vllm_port": 8090}, {"request_timeout": 600}, {"serve_gpu_memory": 1.2}):
        with pytest.raises(ValueError):
            validate_settings({**copy.deepcopy(DEFAULT), **bad})


def test_a_quantization_reaches_vllm_and_is_off_by_default(monkeypatch):
    from exec_feedback.v8 import engine

    seen = []
    monkeypatch.setitem(sys.modules, "vllm", SimpleNamespace(LLM=lambda **kw: seen.append(kw) or "llm"))
    monkeypatch.setattr("exec_feedback.models.load_tokenizer", lambda spec: "tokenizer")
    spec, settings = {"local_path": "/models/merged"}, {"max_context": 8192, "lora_r": 16, "teacher_gpus": 4}
    engine.make_engine(spec, settings, "student", mem=0.85, seed=1, lora=False, max_num_seqs=256)
    engine.make_engine(spec, settings, "student", mem=0.85, seed=1, lora=False, max_num_seqs=256, quantization="fp8")
    assert "quantization" not in seen[0] and seen[1]["quantization"] == "fp8"


def test_stop_ids_are_every_eos_of_the_generation_config(tmp_path):
    (tmp_path / "generation_config.json").write_text(json.dumps({"eos_token_id": [151645, 151643]}), encoding="utf-8")
    assert fm.stop_ids({"local_path": str(tmp_path)}) == [151643, 151645]


# ------------------------------------------------------------------------------------------------ fake model server
class FakeVLLM:
    """OpenAI-compatible chat completions on loopback: a query first, then a finish with GOLD; usage counts the prompt
    with test_v09's character tokenizer; /metrics, /v1/models and /reset_prefix_cache as vLLM's server has them."""

    def __init__(self, port: int, delay: float = 0.0, fail: bool = False) -> None:
        self.tok, self.bodies, self.resets, self.delay, self.lock = ChatTokenizer(), [], 0, delay, threading.Lock()
        self.counters = {"prompt_tokens": 0, "generation_tokens": 0, "prefix_cache_queries": 0, "prefix_cache_hits": 0}
        self.running, self.fail = 0, fail  # requests in flight (vLLM's running gauge); every completion a 500
        fake = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def reply(self, code: int, payload, text: bool = False) -> None:
                raw = payload.encode() if text else json.dumps(payload).encode()
                self.send_response(code)
                self.send_header("Content-Type", "text/plain" if text else "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            def do_GET(self):
                if self.path == "/v1/models":
                    return self.reply(200, {"object": "list", "data": [{"id": SERVED}]})
                if self.path == "/metrics":
                    with fake.lock:
                        lines = [f'vllm:{k}_total{{engine="0",model_name="{SERVED}"}} {v}.0'
                                 for k, v in fake.counters.items()]
                        lines += [f'vllm:num_requests_running{{engine="0"}} {fake.running}.0']
                    lines += ['vllm:kv_cache_usage_perc{engine="0"} 0.25', 'vllm:num_requests_waiting{engine="0"} 0.0',
                              'vllm:e2e_request_latency_seconds_bucket{le="1.0"} 3.0']
                    return self.reply(200, "\n".join(lines) + "\n", text=True)
                self.reply(404, {})

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])) or b"{}")
                if self.path == "/reset_prefix_cache":
                    fake.resets += 1
                    return self.reply(200, {})
                if self.path != "/v1/chat/completions":
                    return self.reply(404, {})
                if fake.fail:
                    return self.reply(500, {"error": "engine dead"})
                with fake.lock:
                    fake.running += 1
                time.sleep(fake.delay)
                messages = body["messages"]
                prompt = len(fake.tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=True))
                first = sum(m["role"] == "assistant" for m in messages) == 0
                action = {"action": "query", "sql": "SELECT v FROM items"} if first else {"action": "finish", "sql": GOLD}
                with fake.lock:
                    fake.running -= 1
                    fake.bodies.append(body)
                    fake.counters["prompt_tokens"] += prompt
                    fake.counters["generation_tokens"] += 7
                    fake.counters["prefix_cache_queries"] += prompt
                    fake.counters["prefix_cache_hits"] += prompt // 2
                self.reply(200, {"id": "x", "model": body["model"], "choices": [
                    {"message": {"role": "assistant", "content": json.dumps(action)}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": prompt, "completion_tokens": 7, "total_tokens": prompt + 7,
                              "prompt_tokens_details": None}})

        self.server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def __enter__(self) -> "FakeVLLM":
        self.thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self.server.shutdown()
        self.server.server_close()


def test_http_policy_sends_extra_fields_and_reports_each_calls_usage():
    port = free_port()
    with FakeVLLM(port) as fake:
        policy = HTTPPolicy(f"http://127.0.0.1:{port}/v1", SERVED, extra_body={"stop_token_ids": [2, 3]})
        text, usage = policy.complete([{"role": "user", "content": "Question 1?"}])
        assert json.loads(text)["action"] == "query" and usage["completion_tokens"] == 7
        assert "prompt_tokens_details" not in usage  # only integer counters
        assert fake.bodies[0]["stop_token_ids"] == [2, 3] and fake.bodies[0]["temperature"] == 0.0
        with ThreadPoolExecutor(8) as pool:
            results = list(pool.map(policy.complete, [[{"role": "user", "content": "x" * i}] for i in range(16)]))
        assert policy.metadata["calls"] == 17
        assert policy.metadata["usage"]["prompt_tokens"] == usage["prompt_tokens"] + sum(u["prompt_tokens"]
                                                                                          for _, u in results)
    with pytest.raises(ValueError, match="fixed request fields"):
        HTTPPolicy("http://127.0.0.1:1/v1", SERVED, extra_body={"temperature": 1.0})


def test_dialogue_takes_usage_from_each_call_when_the_policy_reports_it(fixture_run):
    _, _, ctx = fixture_run
    database = Path(ctx["raw"]) / "shop_t.sqlite"

    class Shared:
        """Another dialogue's calls land in the shared metadata between this dialogue's calls."""
        metadata = {"usage": {"prompt_tokens": 0}}

        def __init__(self):
            self.actions = iter([{"action": "query", "sql": "SELECT v FROM items"}, {"action": "finish", "sql": GOLD}])

        def complete(self, messages):
            self.metadata["usage"]["prompt_tokens"] += 1000
            return json.dumps(next(self.actions)), {"prompt_tokens": 5, "completion_tokens": 2}

    record = dialogue(database, "Question 41?", "", Shared(), review_success=True, fenced_json=True)
    assert record["final_sql"] == GOLD and record["usage"] == {"prompt_tokens": 10, "completion_tokens": 4}
    fenced = "```json\n" + json.dumps({"action": "finish", "sql": GOLD}) + "\n```"
    plain = type("P", (), {"kind": "model", "metadata": {}, "__call__": lambda self, m: fenced})()
    assert dialogue(database, "Question 41?", "", plain, fenced_json=True)["final_sql"] == GOLD
    assert dialogue(database, "Question 41?", "", plain, max_steps=2)["final_sql"] is None  # refused by default


# ------------------------------------------------------------------------------------------------ service
def service_app(tmp_path: Path, concurrent: int, policy, **switches):
    from exec_feedback.cli import make_fixture
    from exec_feedback.configuration import DEFAULT as CONFIG
    from exec_feedback.server import create_app

    root = tmp_path / "db"
    task = make_fixture(root)
    config = copy.deepcopy(CONFIG)
    config["service"].update(database_root=str(root), results_root=str(tmp_path / "results"),
                             max_concurrent=concurrent, **switches)
    config["model"]["backend"] = "http"
    return create_app(config, policy=policy), task


def test_service_runs_up_to_max_concurrent_analyses_and_reports_each_requests_calls(tmp_path):
    from fastapi.testclient import TestClient

    started, release = threading.Semaphore(0), threading.Event()

    class Blocking:
        kind, metadata = "model", {"provider": "fixture", "model": "unit-test-only", "usage": {}}

        def complete(self, messages):
            started.release()
            release.wait(10)
            return json.dumps({"action": "finish", "sql": "SELECT 1"}), {"prompt_tokens": 3, "completion_tokens": 1}

    app, _ = service_app(tmp_path, 2, Blocking())
    with TestClient(app, base_url="http://127.0.0.1") as client, ThreadPoolExecutor(3) as pool:
        ask = lambda: client.post("/api/analyze", json={"database": "shop", "question": "How many?"})
        first, second = pool.submit(ask), pool.submit(ask)
        assert started.acquire(timeout=10) and started.acquire(timeout=10)  # two analyses running at once
        third = ask()
        assert third.status_code == 429 and "All 2 request slots are busy" in third.json()["detail"]
        release.set()
        for response in (first.result(10), second.result(10)):
            assert response.status_code == 200, response.text
            assert response.json()["calls"] == [{"prompt_tokens": 3, "completion_tokens": 1}]
    with pytest.raises(ValueError, match="HTTP model backend"):
        from exec_feedback.configuration import DEFAULT as CONFIG
        from exec_feedback.server import create_app
        config = copy.deepcopy(CONFIG)
        config["service"].update(database_root=str(tmp_path / "db"), max_concurrent=2)
        config["model"]["backend"] = "transformers"
        create_app(config, policy=Blocking())


def test_service_protocol_switches_default_to_the_old_behaviour(tmp_path):
    from fastapi.testclient import TestClient

    def run(**switches):
        answers = iter(["```json\n" + json.dumps({"action": "finish", "sql": "SELECT 1"}) + "\n```"] * 4)
        policy = type("P", (), {"kind": "model", "metadata": {"provider": "fixture", "model": "unit-test-only"},
                                "__call__": lambda self, m: next(answers)})()
        app, _ = service_app(tmp_path / str(len(switches)), 1, policy, **switches)
        with TestClient(app, base_url="http://127.0.0.1") as client:
            return client.post("/api/analyze", json={"database": "shop", "question": "How many?"}).json()
    old = run()
    assert old["final_sql"] is None and old["status"] == "budget_exhausted"  # fenced JSON refused, as before
    new = run(review_success=True, fenced_json=True)
    assert new["final_sql"] == "SELECT 1" and len(new["steps"]) == 2  # accepted, then the one result review


# ------------------------------------------------------------------------------------------------ freeze, merge
@pytest.fixture
def deploy_run(fixture_run, tmp_path, monkeypatch):
    _, _, ctx = fixture_run
    settings = smoke_settings()
    run = tmp_path / "deploy"
    frozen = fm.freeze(ctx["registration"], ctx["raw"], run, settings, **ctx["kw"], v11_run=ctx["ids"]["v11_run"])

    def fake_merge(base, adapter, out, *, name, inputs):
        out.mkdir(parents=True)
        for file in ("config.json", "model.safetensors"):
            (out / file).write_text(f"{name}:{file}", encoding="utf-8")
        (out / "generation_config.json").write_text(json.dumps({"eos_token_id": [2]}), encoding="utf-8")
        record = {"name": name, "inputs": {**inputs, "peft": "test"}, "fp32_merge_max_logit_delta": 1e-6,
                  "bf16_rounding_max_logit_delta": 0.01, "keys": 2, "checksum": [0.0, 0.0],
                  "identity": dir_identity(out)}
        record_of(out).write_text(json.dumps(record), encoding="utf-8")
        return record
    monkeypatch.setattr(fm, "merge_adapter", fake_merge)
    student = Path(frozen["models"]["student"]["spec"]["local_path"])
    if not (student / "generation_config.json").exists():
        monkeypatch.setattr(bm, "stop_ids", lambda spec: [2])
    return run, frozen, ctx


def test_freeze_keeps_the_roles_the_bench_tasks_and_rlb_seed44(deploy_run, tmp_path):
    run, frozen, ctx = deploy_run
    assert frozen["counts"] == {"test": 12, "validation": 2} and len(frozen["bench"]["tasks"]) == 8
    assert set(frozen["imports"]["v11"]["adapters"]) == {MODEL}
    assert fm.adapter(run, frozen)[2] == frozen["imports"]["v11"]["k"]
    assert fm.bench_tasks(fm.load(run, "test")[1], frozen["settings"]["salt"], 8) == \
        [t for i in frozen["bench"]["tasks"] for t in fm.load(run, "test")[1] if t.id == i]
    (run / "imports" / "v11" / "adapters" / MODEL / "adapter" / "adapter_config.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="adapter changed"):
        fm.adapter(run, frozen)
    with pytest.raises(ValueError, match="registered v11 run"):
        fm.freeze(ctx["registration"], ctx["raw"], tmp_path / "other", smoke_settings(), **ctx["kw"])


def test_merge_is_built_once_verified_and_rebuilt_when_changed(deploy_run):
    run, frozen, _ = deploy_run
    with pytest.raises(ValueError, match="not been merged"):
        fm.merged_model(run, frozen)
    record = fm.merge(run)
    assert fm.merge(run) == record and fm.merged_model(run, frozen)[1] == record
    folder = fm.merged_dir(frozen)
    assert folder.name == f"{MODEL}-{frozen['imports']['v11']['k']:04d}"
    (folder / "model.safetensors").write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="changed on disk"):
        fm.merged_model(run, frozen)
    rebuilt = fm.merge(run)  # the changed directory is kept aside, never deleted
    assert fm.merged_model(run, frozen)[1] == rebuilt and any(".stale-" in p.name for p in folder.parent.iterdir())


def test_each_deployment_loads_what_it_says(deploy_run):
    run, frozen, _ = deploy_run
    fm.merge(run)
    lora, merged, fp8 = (engine_for(run, frozen, c, ready=False) for c in CONDITIONS)
    assert lora[1] is not None and lora[2] is None and lora[0]["local_path"] == frozen["models"]["student"]["spec"]["local_path"]
    assert merged[1] is None and merged[2] is None and merged[0]["local_path"] == str(fm.merged_dir(frozen))
    assert fp8[0] == merged[0] and fp8[2] == "fp8" and fp8[3]["merged"] == merged[3]["merged"]
    commands = {c: bm.vllm_command("vpy", run, frozen, c) for c in CONDITIONS}
    assert f"{SERVED}={lora[1]}" in commands["lora"] and "--enable-lora" in commands["lora"]
    assert commands["merged"][commands["merged"].index("--served-model-name") + 1] == SERVED
    assert commands["fp8"][-2:] == ["--quantization", "fp8"] and "--quantization" not in commands["merged"]
    for cmd in commands.values():
        assert cmd[cmd.index("--generation-config") + 1] == "vllm" and "--enable-prefix-caching" in cmd
        assert cmd[cmd.index("--max-num-batched-tokens") + 1] == "16384"  # the offline engine's budget per step


# ------------------------------------------------------------------------------------------------ evaluate, bench
def evaluate_all(run: Path, frozen: dict) -> None:
    fm.merge(run)
    for role in frozen["settings"]["roles"]:
        for condition in CONDITIONS:
            assert evaluate(run, condition=condition, role=role, stop=None, factory=scripted_factory)["complete"]


@contextlib.contextmanager
def fake_servers(run, frozen, condition, *, vllm_python, main_python, logs, host, port, results, concurrent, key="",
                 fail=False):
    import uvicorn
    from exec_feedback.server import create_app

    config = bm.service_config(run, frozen, host=host, port=port, results=results, concurrent=concurrent)
    with FakeVLLM(frozen["settings"]["vllm_port"], delay=0.01, fail=fail):
        server = uvicorn.Server(uvicorn.Config(create_app(config), host=host, port=port, log_level="warning"))
        thread = threading.Thread(target=server.run, daemon=True)
        thread.start()
        deadline = time.time() + 20
        while not server.started and time.time() < deadline:
            time.sleep(0.05)
        try:
            yield f"http://127.0.0.1:{frozen['settings']['vllm_port']}", f"http://127.0.0.1:{port}"
        finally:
            server.should_exit = True
            thread.join(20)


def test_load_test_and_report_end_to_end(deploy_run):
    run, frozen, _ = deploy_run
    evaluate_all(run, frozen)
    summary = read_json(run / "eval" / "test" / "fp8" / "summary.json")
    assert summary["quantization"] == "fp8" and summary["merged"] == fm.merged_model(run, frozen)[1]["identity"]
    result = bm.bench(run, condition="merged", launch=fake_servers)
    assert result["complete"] and result["levels"] == ["1", "4"] and result["warmup"]["failures"] == 0
    assert result["attempt"] == 1 and result["failure"] is None and result["stopped"] is None
    for c in (1, 4):
        data = read_json(run / "bench" / "merged" / "attempt-1" / f"level-{c}.json")
        assert [r["task_id"] for r in data["rows"]] == frozen["bench"]["tasks"] and data["stopped"] == "done"
        assert all(r["code"] == 200 and r["calls"] and r["final_sql"] == GOLD and r["run_id"] for r in data["rows"]), \
            data["rows"][:2]
        assert data["prefix_cache_reset"] == {"status": 200, "drained": True}
        assert data["metrics_after"]["vllm:prompt_tokens"] > data["metrics_before"].get("vllm:prompt_tokens", 0)
        assert "vllm:e2e_request_latency_seconds_bucket" not in data["metrics_after"] and data["peaks"]["samples"] >= 1
        journal = [json.loads(line) for line in (run / "bench" / "merged" / "attempt-1" / f"level-{c}.jsonl")
                   .read_text(encoding="utf-8").splitlines()]
        assert sorted(r["task_id"] for r in journal) == sorted(frozen["bench"]["tasks"])  # each row as it completed
    assert bm.bench(run, condition="merged", launch=None) == result  # finished: not run again
    report = analysis.report(run, tokenizer=ChatTokenizer())
    assert set(report["accuracy"]["test"]) == set(CONDITIONS) and set(report["deltas"]["test"]) == {
        "merged-lora", "fp8-merged", "fp8-lora"}
    assert report["deltas"]["test"]["merged-lora"]["delta"] == 0.0  # the scripted generator ignores the deployment
    level = report["bench"]["merged"]["levels"]["4"]
    assert level["served"] == 8 and level["failures"] == 0 and level["calls_per_task"] >= 2
    assert level["prefix_cache_hit_rate"] == pytest.approx(0.5, 0.05) and report["bench"]["merged"]["attempt"] == 1
    golds = {t.id: t.gold_sql for t in fm.load(run, "test")[1]}
    assert level["correct"] == sum(golds[i] == GOLD for i in frozen["bench"]["tasks"])
    align = report["alignment"]["merged"]
    assert align["service_low_vs_high"]["same_final_sql"] == 8 and align["service_low_vs_high"]["both_without_sql"] == 0
    assert align["first_prompt_tokens"] == {"tasks": 8, "identical": 8, "tokenizer": True, "differences": {}}
    assert align["evaluation_vs_service_low"]["tasks"] == 8
    text = (run / "report" / "report.md").read_text(encoding="utf-8")
    assert "压测" in text and "对齐" in text and "合并" in text
    assert "部署研究自动记录" in analysis.progress_entry(run, "M:report", "bundle")


def test_infrastructure_failures_fail_the_deployment_and_a_rerun_keeps_the_first_attempt(deploy_run):
    run, frozen, _ = deploy_run
    fm.merge(run)
    broken = bm.bench(run, condition="merged", launch=functools.partial(fake_servers, fail=True))
    assert not broken["complete"] and "warm-up" in broken["failure"] and broken["levels"] == []
    assert not (run / "bench" / "merged" / "result.json").exists()
    first = run / "bench" / "merged" / "attempt-1"
    warm = [json.loads(line) for line in (first / "warmup.jsonl").read_text(encoding="utf-8").splitlines()]
    # the service answered 200 with its model's error: an infrastructure failure, with its reason kept
    assert warm and all(r["code"] == 200 and r["status"] == "provider_error" and r["reason"] for r in warm)
    saved = {p.name: p.read_bytes() for p in first.iterdir() if p.is_file()}
    ok = bm.bench(run, condition="merged", launch=fake_servers)
    assert ok["complete"] and ok["attempt"] == 2 and (run / "bench" / "merged" / "result.json").exists()
    assert {p.name: p.read_bytes() for p in first.iterdir() if p.is_file()} == saved  # never overwritten
    assert analysis.attempt_of(run / "bench" / "merged").name == "attempt-2"


def test_an_attempt_that_cannot_be_trusted_fails_and_says_why(deploy_run, monkeypatch):
    import http.client

    run, frozen, _ = deploy_run
    fm.merge(run)
    folder = run / "bench" / "merged"
    # a response cut off mid-read is a failed request, never a lost one
    def cut_off(*a, **k):
        raise http.client.IncompleteRead(b"")
    with monkeypatch.context() as m:
        m.setattr(bm, "post", cut_off)
        row = bm.request("http://127.0.0.1:1", SimpleNamespace(id="t", db_id="d", question="q", evidence=""), 1.0)
    assert row["code"] is None and "IncompleteRead" in row["error"] and bm.failed(row)
    # the prefix cache not reset (vLLM answers 200 even when it could not): no level is measured on a warm cache
    with monkeypatch.context() as m:
        m.setattr(bm, "reset_prefix_cache", lambda vllm, wait=120.0: {"status": 200, "drained": False})
        out = bm.bench(run, condition="merged", launch=fake_servers)
    assert not out["complete"] and "prefix cache" in out["failure"] and out["levels"] == []
    # a level (not the 2-task warm-up) that lost a task's row
    real = bm.level

    def lossy(service, tasks, *a, **k):
        rows, wall, why = real(service, tasks, *a, **k)
        return (rows[:-1] if len(tasks) > 2 else rows), wall, why
    with monkeypatch.context() as m:
        m.setattr(bm, "level", lossy)
        out = bm.bench(run, condition="merged", launch=fake_servers)
    assert not out["complete"] and "answered 7 of 8" in out["failure"]

    # servers that never came up: the attempt still ends with a record, and the report shows it
    @contextlib.contextmanager
    def busy_ports(*a, **k):
        raise RuntimeError("Ports [8000] are in use: a server from an earlier attempt may still run")
        yield
    with pytest.raises(RuntimeError):
        bm.bench(run, condition="merged", launch=busy_ports)
    record = read_json(folder / "attempt-3" / "attempt.json")
    assert record["status"] == "failed" and "in use" in record["failure"]
    assert not (folder / "result.json").exists()
    report = analysis.report(run, tokenizer=ChatTokenizer())
    shown = report["bench"]["merged"]
    assert shown["attempt"] == 2 and not shown["complete"]  # the latest attempt that measured something
    assert [a["status"] for a in shown["attempts"]] == ["failed", "failed", "failed"]
    assert "service_low_vs_high" not in report["alignment"].get("merged", {})  # no complete level to compare
    text = (run / "report" / "report.md").read_text(encoding="utf-8")
    assert "in use" in text and "answered 7 of 8" in text


def test_the_report_shows_an_evaluation_cut_short_and_a_single_level(deploy_run):
    run, frozen, _ = deploy_run
    fm.merge(run)
    evaluate(run, condition="lora", role="test", stop=None, factory=scripted_factory)
    (run / "eval" / "test" / "lora" / "summary.json").unlink()  # as if killed after its last row
    report = analysis.report(run, tokenizer=ChatTokenizer())
    assert report["accuracy"]["test"]["lora"] == {"incomplete": True, "episodes": 12, "tasks": 12}
    assert "未完成（12/12 题）" in (run / "report" / "report.md").read_text(encoding="utf-8")


def test_a_level_stops_at_the_deadline_or_past_its_failure_allowance(monkeypatch, tmp_path):
    asked = []

    def answer(service, task, timeout):
        asked.append(task)
        return {"task_id": task, "code": 200, "status": "provider_error" if task % 2 else "submitted", "seconds": 0.0}
    monkeypatch.setattr(bm, "request", answer)
    rows, _, why = bm.level("svc", list(range(20)), 1, 1.0, journal=tmp_path / "level.jsonl", allowed=2)
    assert why == "failures" and [r["task_id"] for r in rows] == list(range(6)) and len(asked) == 6
    assert len((tmp_path / "level.jsonl").read_text(encoding="utf-8").splitlines()) == 6
    rows, _, why = bm.level("svc", list(range(5)), 2, 1.0, end=time.time() - 1)
    assert rows == [] and why == "deadline"
    rows, _, why = bm.level("svc", list(range(5)), 3, 1.0)
    assert why == "done" and [r["task_id"] for r in rows] == list(range(5))
    assert bm.allowance(256) == 5 and bm.allowance(8) == 0
    assert bm.failed({"code": 200, "status": "provider_error"}) and bm.failed({"code": 429, "status": None})
    assert not bm.failed({"code": 200, "status": "budget_exhausted"})  # the model ran out of steps: a model outcome


def test_alignment_compares_final_sql_verbatim_and_leaves_failures_out():
    left = {"a": {"final_sql": "SELECT 'A'", "correct": True}, "b": {"final_sql": "SELECT  'a b'", "correct": False},
            "c": {"final_sql": None, "correct": False}, "d": {"final_sql": "SELECT 1", "correct": True},
            "e": {"final_sql": "SELECT 2", "correct": True}}
    right = {"a": {"final_sql": "SELECT 'a'", "correct": True, "code": 200},
             "b": {"final_sql": "SELECT 'a b'", "correct": False, "code": 200},
             "c": {"final_sql": None, "correct": False, "code": 200},
             "d": {"final_sql": "SELECT 1", "correct": True, "code": 200},
             "e": {"final_sql": None, "correct": False, "code": 200, "status": "provider_error"}}
    assert analysis.agreement(left, right, list("abcde")) == {
        "tasks": 4, "left_out": 1, "same_final_sql": 1, "both_without_sql": 1, "same_correctness": 4,
        "left_correct": 2, "right_correct": 2}


def test_the_demo_key_file_is_created_owner_only_and_replaces_what_was_there(tmp_path):
    key = tmp_path / "demo.key"
    key.write_text("old", encoding="utf-8")
    bm.write_secret(key, "s" * 32)
    assert key.read_text(encoding="utf-8") == "s" * 32 + "\n"
    if os.name == "posix":
        assert stat.S_IMODE(key.stat().st_mode) == 0o600
        target = tmp_path / "elsewhere"
        target.write_text("keep", encoding="utf-8")
        key.unlink()
        key.symlink_to(target)
        bm.write_secret(key, "t" * 32)
        assert target.read_text(encoding="utf-8") == "keep" and not key.is_symlink()


def test_servers_refuse_ports_another_server_still_holds(deploy_run):
    run, frozen, _ = deploy_run
    with socket.socket() as busy:
        busy.bind(("127.0.0.1", frozen["settings"]["vllm_port"]))
        busy.listen()
        with pytest.raises(RuntimeError, match="in use"):
            with bm.servers(run, frozen, "lora", vllm_python="none", main_python="none", logs=run / "logs",
                            host="127.0.0.1", port=frozen["settings"]["service_port"], results=run / "svc",
                            concurrent=1):
                pytest.fail("servers started on a busy port")


@pytest.mark.skipif(not hasattr(os, "killpg") or not Path("/proc/self/stat").exists(),
                    reason="process groups and /proc are Linux's (where the servers run)")
def test_servers_left_by_a_killed_load_test_are_stopped(tmp_path):
    ledger = tmp_path / bm.SERVERS
    ledger.mkdir()
    gone = subprocess.Popen([sys.executable, "-c", "pass"])
    gone.wait()  # an owner that is gone
    zombie = subprocess.Popen([sys.executable, "-c", "pass"])  # an owner that exited and was never reaped
    deadline = time.time() + 10
    while (bm.proc_state(zombie.pid) or ("Z",))[0] != "Z" and time.time() < deadline:
        time.sleep(0.05)
    sleepers = [subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"], start_new_session=True)
                for _ in range(2)]
    for owner, sleeper in ((gone.pid, sleepers[0]), (zombie.pid, sleepers[1])):
        (ledger / f"{owner}.json").write_text(json.dumps({"owner": owner, "groups": [sleeper.pid]}), encoding="utf-8")
    me = {"owner": os.getpid(), "owner_start": bm.proc_state(os.getpid())[1], "groups": []}
    (ledger / f"{os.getpid()}.json").write_text(json.dumps(me), encoding="utf-8")
    (ledger / "reused.json").write_text(json.dumps({**me, "owner_start": -1}), encoding="utf-8")  # a reused pid
    started = time.monotonic()
    notes = bm.reap_servers(tmp_path)
    assert time.monotonic() - started < 15  # killed servers left as this test's zombies are not waited for
    assert all(s.wait(30) is not None for s in sleepers) and len(notes) == 2
    assert sorted(p.name for p in ledger.iterdir()) == [f"{os.getpid()}.json"]  # a live owner's are its own
    zombie.wait()


# ------------------------------------------------------------------------------------------------ driver
def test_the_driver_runs_the_smoke_first_and_skips_what_a_failure_makes_pointless(deploy_run, tmp_path):
    from exec_feedback.deploy.orchestrate import Driver
    from exec_feedback.storage import fingerprint

    smoke, frozen, _ = deploy_run
    main = tmp_path / "main"
    shutil.copytree(smoke, main)
    manifest = json.loads((main / "manifest.json").read_text(encoding="utf-8"))
    manifest.pop("identity")
    manifest["smoke_limit"] = 0
    manifest["identity"] = fingerprint(manifest)
    (main / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    ran, exports = [], []
    driver = Driver(run=main, smoke_run=smoke, main_py="py", vllm_py="vpy", stop=time.time() + 9 * 3600,
                    status=tmp_path / "STATUS.txt", log_dir=tmp_path / "logs", milestone=exports.append)
    fails = {"S:eval-fp8-test", "M:merge"}
    driver.execute = lambda step: ran.append(step.name) or step.name not in fails
    driver.loop()
    status = (tmp_path / "STATUS.txt").read_text(encoding="utf-8")
    assert ran[0] == "S:merge" and ran.index("S:report") < ran.index("M:merge")
    assert "M:eval-fp8-test" not in ran and "not starting M:eval-fp8-test: the smoke run failed evaluate:fp8" in status
    assert "M:eval-merged-test" not in ran and "M:bench-merged" not in ran and "M:eval-lora-test" in ran
    assert "M:bench-lora" in ran and "merge failed" in status
    assert exports[:2] == ["S:report", "M:eval"] and exports[-1] == "M:report"
    late = Driver(run=main, smoke_run=smoke, main_py="py", vllm_py="vpy", stop=time.time() + 600,
                  status=tmp_path / "LATE.txt", log_dir=tmp_path / "logs")
    late.execute = lambda step: pytest.fail("nothing fits")
    late.loop()
    assert "not starting S:merge: needs" in (tmp_path / "LATE.txt").read_text(encoding="utf-8")


def test_each_try_has_its_own_limit_and_its_log_is_kept(deploy_run, tmp_path):
    from exec_feedback.deploy.orchestrate import MIN_LIMIT, WIND_DOWN, Driver, Step

    smoke, _, _ = deploy_run
    seen = []
    driver = Driver(run=smoke, smoke_run=smoke, main_py="py", vllm_py="vpy", stop=time.time() + 9 * 3600,
                    status=tmp_path / "STATUS.txt", log_dir=tmp_path / "logs",
                    wrap=lambda cmd, seconds: seen.append(seconds) or cmd)
    step = Step("S:probe", "probe", lambda until: seen.append(until) or [sys.executable, "-c", "print('ran')"],
                lambda: False, 600.0, retries=1)
    assert not driver.execute(step)  # never done: both tries ran
    log = (tmp_path / "logs" / "S_probe.log").read_text(encoding="utf-8")
    assert log.count("===== ") == 2 and log.count("ran") == 2  # appended, not overwritten
    until, limit = seen[0], seen[1]
    assert limit == MIN_LIMIT  # twice the estimate is less than the floor
    assert dt.datetime.fromisoformat(until).timestamp() == pytest.approx(time.time() + MIN_LIMIT - WIND_DOWN, abs=120)


# ------------------------------------------------------------------------------------------------ scripts
def test_the_scripts_parse():
    bash = shutil.which("bash")
    if bash is None:
        pytest.skip("no bash")
    for script in ("deploy_run.sh", "deploy_export.sh", "deploy_demo.sh"):
        done = subprocess.run([bash, "-n", str(REPO / "scripts" / script)], capture_output=True, text=True)
        assert done.returncode == 0, done.stderr
    subprocess.run([sys.executable, "-m", "py_compile", str(REPO / "scripts" / "deploy_bundle.py")], check=True)
