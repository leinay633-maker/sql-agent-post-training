"""Single-user analysis service and compatible model endpoint; local by default."""

import contextlib
import hmac
import ipaddress
import json
import os
import re
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from .agent import dialogue, preview
from .core import Limits, digest
from .datasets import _database_index
from .models import create_policy
from .storage import atomic_json, fingerprint, read_json, CallBudget, BudgetExceeded


class _Recorded:
    """One request's view of a shared policy with per-call usage: the calls of this request, in order."""
    def __init__(self, policy: Any, calls: list[dict[str, int]]) -> None:
        self.policy, self.calls, self.metadata = policy, calls, policy.metadata
    def complete(self, messages: list[dict[str, str]]) -> tuple[str, dict[str, int]]:
        text, usage = self.policy.complete(messages)
        self.calls.append(usage)
        return text, usage


def create_app(config: dict[str, Any], *, policy: Any = None) -> Any:
    try:
        from fastapi import FastAPI, HTTPException, Request
        from fastapi.responses import HTMLResponse, JSONResponse
        from starlette.concurrency import run_in_threadpool
    except ImportError as exc:
        raise RuntimeError("Install the server extra: python -m pip install -e '.[server]'") from exc
    settings = config["service"]
    # Requests running at once; more than one only in front of an HTTP model server (the in-process model is not
    # thread-safe). The two protocol switches default to the v0.2 service behaviour.
    concurrent = settings.get("max_concurrent", 1)
    if type(concurrent) is not int or not 1 <= concurrent <= 256:
        raise ValueError("service.max_concurrent must be 1..256")
    if concurrent > 1 and config["model"]["backend"] != "http":
        raise ValueError("Concurrent analyses need the HTTP model backend")
    review_success, fenced_json = settings.get("review_success", False), settings.get("fenced_json", False)
    if type(review_success) is not bool or type(fenced_json) is not bool:
        raise ValueError("service.review_success and service.fenced_json are true or false")
    key = os.environ.get("EFA_SERVER_KEY", "")
    try:
        local = ipaddress.ip_address(settings["host"]).is_loopback
    except ValueError:
        local = settings["host"] == "localhost"
    if not local and len(key) < 24:
        raise ValueError("Non-loopback serving requires EFA_SERVER_KEY with at least 24 characters; use a TLS proxy or SSH tunnel")
    database_root = Path(settings["database_root"]).resolve(strict=True)
    catalog = _database_index(database_root)
    snapshots = {name: digest(path) for name,path in catalog.items()}
    output = Path(settings["results_root"])
    gate = threading.BoundedSemaphore(concurrent)
    holder = {"policy": policy}
    @contextlib.asynccontextmanager
    async def lifespan(app: Any):
        if concurrent > 40:  # each analysis runs in anyio's worker threads, 40 by default
            import anyio.to_thread
            anyio.to_thread.current_default_thread_limiter().total_tokens = concurrent
        if holder["policy"] is None:
            budget = CallBudget(output / "service-budget.json", config["resources"]["max_model_calls"], config["resources"]["max_active_seconds"])
            holder["policy"] = create_policy(config["model"], Path(settings["adapter"]) if settings["adapter"] else None, budget)
        try:
            yield
        finally:
            if hasattr(holder["policy"], "close"):
                holder["policy"].close()
    app = FastAPI(title="Execution Feedback Agent", version="0.2.0", lifespan=lifespan,
                  docs_url=None, redoc_url=None, openapi_url=None)

    @app.middleware("http")
    async def guard(request: Request, call_next: Any):
        if request.method not in {"GET", "POST"}:
            return JSONResponse({"error": "method not allowed"}, status_code=405)
        host = request.headers.get("host", "").split(":")[0].lower()
        if not key and host not in {"127.0.0.1", "localhost", "["}:
            return JSONResponse({"error": "invalid host"}, status_code=400)
        if request.url.path not in {"/", "/healthz"} and key:
            supplied = request.headers.get("authorization", "")
            if not hmac.compare_digest(supplied.encode(), ("Bearer " + key).encode()):
                return JSONResponse({"error": "unauthorized"}, status_code=401)
        if request.method == "POST":
            length = request.headers.get("content-length")
            if length is None:
                return JSONResponse({"error": "Content-Length required"}, status_code=411)
            try:
                if not 1 <= int(length) <= 262144:
                    raise ValueError
            except ValueError:
                return JSONResponse({"error": "request too large"}, status_code=413)
            if request.headers.get("content-type", "").split(";")[0] != "application/json":
                return JSONResponse({"error": "JSON required"}, status_code=415)
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Cache-Control"] = "no-store"
        return response

    @app.exception_handler(BudgetExceeded)
    async def exhausted(request: Request, error: Any):
        return JSONResponse({"error": "Configured service model-call/time budget exhausted"}, status_code=429)

    @app.exception_handler(json.JSONDecodeError)
    async def invalid_json(request: Request, error: Any):
        return JSONResponse({"error": "Malformed JSON"}, status_code=422)

    @app.get("/", response_class=HTMLResponse)
    async def index():
        return (Path(__file__).parent / "web" / "index.html").read_text(encoding="utf-8")

    @app.get("/healthz")
    async def health():
        return {"status": "ok"}

    @app.get("/readyz")
    async def ready():
        if holder["policy"] is None:
            raise HTTPException(503, "model not loaded")
        return {"status": "ready", "model": holder["policy"].metadata, "databases": len(catalog)}

    @app.get("/api/databases")
    async def databases():
        return {"databases": sorted(catalog)}

    @app.get("/v1/models")
    async def models():
        return {"object": "list", "data": [{"id": config["model"]["id"], "object": "model",
                 "owned_by": "local", "identity": holder["policy"].metadata.get("identity", {})}]}

    async def exclusive(function: Any, *args: Any):
        if holder["policy"] is None:
            raise HTTPException(503, "model not ready")
        if not gate.acquire(blocking=False):
            raise HTTPException(429, ("One request is already running" if concurrent == 1 else
                                      f"All {concurrent} request slots are busy")
                                + "; this server intentionally has no unbounded queue")
        try:
            return await run_in_threadpool(function, *args)
        finally:
            gate.release()

    @app.post("/api/analyze")
    async def analyze(request: Request):
        body = await request.json()
        if not isinstance(body, dict) or set(body) - {"database", "question", "evidence"}:
            raise HTTPException(422, "Expected database/question/evidence")
        name, question, evidence = body.get("database"), body.get("question"), body.get("evidence", "")
        if not isinstance(name, str) or name not in catalog or not isinstance(question, str) or not 1 <= len(question) <= 8000 or not isinstance(evidence, str) or len(evidence) > 8000:
            raise HTTPException(422, "Invalid database/question/evidence")
        def run():
            path = catalog[name]
            if digest(path) != snapshots[name]:
                raise HTTPException(409, "Database changed since server startup")
            model = holder["policy"]
            if hasattr(model, "set_task"): model.set_task(fingerprint(body))
            calls: list[dict[str, int]] = []
            result = dialogue(path, question, evidence, _Recorded(model, calls) if hasattr(model, "complete") else model,
                              max_steps=settings["max_steps"], limits=Limits(**config["limits"]),
                              review_success=review_success, fenced_json=fenced_json)
            if digest(path) != snapshots[name]:
                raise HTTPException(409, "Database changed during analysis")
            if "result" in result:
                result["result"] = preview(result["result"], max_rows=100)
            if calls:
                result["calls"] = calls  # this request's own usage per model call
            result.pop("correct", None)  # Live requests have no reference answer.
            result.update(run_id=uuid.uuid4().hex, database=name, question=question, model=model.metadata["model"],
                          correctness="not_scored_no_reference", created_unix=time.time())
            atomic_json(output / (result["run_id"] + ".json"), result)
            return result
        return await exclusive(run)

    @app.get("/api/results/{run_id}")
    async def result(run_id: str):
        if not re.fullmatch(r"[0-9a-f]{32}", run_id):
            raise HTTPException(404, "not found")
        path = output / f"{run_id}.json"
        if not path.is_file(): raise HTTPException(404, "not found")
        return read_json(path)

    @app.post("/v1/chat/completions")
    async def chat(request: Request):
        body = await request.json()
        if not isinstance(body, dict) or body.get("model") != config["model"]["id"] or body.get("stream", False):
            raise HTTPException(422, "Exact served model required; streaming is not supported")
        messages = body.get("messages")
        if not isinstance(messages, list) or not 1 <= len(messages) <= 64:
            raise HTTPException(422, "Invalid messages")
        for message in messages:
            if not isinstance(message, dict) or set(message) != {"role", "content"} or message["role"] not in {"system", "user", "assistant"} or not isinstance(message["content"], str):
                raise HTTPException(422, "Only text chat messages are supported")
        if body.get("temperature", config["model"]["temperature"]) != config["model"]["temperature"]:
            raise HTTPException(422, "Temperature must match the server's fixed experiment configuration")
        cap = body.get("max_tokens", config["model"]["max_tokens"])
        if type(cap) is not int or not 1 <= cap <= config["model"]["max_tokens"]:
            raise HTTPException(422, "Output token limit exceeds server cap")
        def generate():
            model = holder["policy"]
            if model.metadata.get("provider") == "openai-compatible":
                raise HTTPException(422, "The HTTP-backed analysis service does not expose a recursive model proxy")
            before = dict(model.metadata.get("usage", {}))
            if hasattr(model, "set_task"):
                model.set_task(fingerprint(messages))
            original = model.spec["max_tokens"] if hasattr(model, "spec") else None
            try:
                if original is not None: model.spec["max_tokens"] = cap
                text = model(messages)
            except Exception as exc:
                raise HTTPException(502, f"Model request failed: {type(exc).__name__}") from exc
            finally:
                if original is not None: model.spec["max_tokens"] = original
            usage = {k:v-before.get(k,0) for k,v in model.metadata.get("usage", {}).items()}
            return {"id": "chatcmpl-"+uuid.uuid4().hex, "object": "chat.completion", "created": int(time.time()),
                    "model": config["model"]["id"], "choices": [{"index":0, "message":{"role":"assistant", "content":text}, "finish_reason":model.metadata.get("last_finish_reason")}], "usage":usage}
        return await exclusive(generate)
    return app


def serve(config: dict[str, Any]) -> None:
    import uvicorn
    app = create_app(config)
    uvicorn.run(app, host=config["service"]["host"], port=config["service"]["port"], workers=1, log_level="info")
