import copy
import json

import pytest

from exec_feedback.cli import make_fixture
from exec_feedback.configuration import DEFAULT
from exec_feedback.pipeline import FixturePolicy
from exec_feedback.server import create_app


@pytest.fixture
def app_setup(tmp_path):
    root=tmp_path/"db"
    task=make_fixture(root)
    config=copy.deepcopy(DEFAULT)
    config["service"].update(database_root=str(root),results_root=str(tmp_path/"results"))
    policy=FixturePolicy([{"action":"query","sql":"SELECT not_a_column FROM orders"}, {"action":"finish","sql":task.gold_sql}])
    policy.metadata={"provider":"fixture","model":"unit-test-only"}
    return config,policy


def test_live_analysis_has_real_sql_but_no_fake_correctness(app_setup):
    from fastapi.testclient import TestClient
    config,policy=app_setup
    with TestClient(create_app(config,policy=policy),base_url="http://127.0.0.1") as client:
        assert client.get("/healthz").status_code==200
        assert client.get("/readyz").json()["status"]=="ready"
        assert client.get("/api/databases").json()["databases"]==["shop"]
        response=client.post("/api/analyze",json={"database":"shop","question":"Paid revenue?"})
        assert response.status_code==200,response.text
        result=response.json()
        assert result["result"]["rows"]==[[40.0]]
        assert "correct" not in result
        assert result["correctness"]=="not_scored_no_reference"
        assert client.get("/api/results/"+result["run_id"]).status_code==200
        assert client.get("/api/results/not-a-uuid").status_code==404
        assert client.get("/").status_code==200


def test_auth_and_bad_inputs(app_setup,monkeypatch):
    from fastapi.testclient import TestClient
    config,policy=app_setup
    monkeypatch.setenv("EFA_SERVER_KEY","a"*32)
    with TestClient(create_app(config,policy=policy),base_url="http://127.0.0.1") as client:
        assert client.get("/readyz").status_code==401
        client.headers["Authorization"]="Bearer "+"a"*32
        assert client.get("/readyz").status_code==200
        assert client.post("/api/analyze",json={"database":[],"question":"x"}).status_code==422
        assert client.post("/api/analyze",json={"database":"../shop","question":"x"}).status_code==422
        assert client.post("/api/analyze",content="x",headers={"Content-Type":"text/plain"}).status_code==415
        assert client.post("/v1/chat/completions",json={"model":"wrong","messages":[]}).status_code==422


def test_public_host_requires_key(app_setup,monkeypatch):
    config,policy=app_setup
    monkeypatch.delenv("EFA_SERVER_KEY",raising=False)
    config["service"]["host"]="0.0.0.0"
    with pytest.raises(ValueError,match="EFA_SERVER_KEY"): create_app(config,policy=policy)


def test_html_uses_text_nodes_not_untrusted_innerhtml(app_setup):
    from pathlib import Path
    from exec_feedback import server
    page=(Path(server.__file__).parent/"web/index.html").read_text(encoding="utf-8")
    assert "innerHTML" not in page
    assert "textContent" in page


def test_malformed_json_returns_client_error(app_setup):
    from fastapi.testclient import TestClient
    config,policy=app_setup
    with TestClient(create_app(config,policy=policy),base_url='http://127.0.0.1') as client:
        assert client.post('/api/analyze',content='{bad',headers={'Content-Type':'application/json'}).status_code==422


def test_busy_requests_are_rejected_not_queued(app_setup):
    import threading
    from concurrent.futures import ThreadPoolExecutor
    from fastapi.testclient import TestClient
    config,_=app_setup
    started,released=threading.Event(),threading.Event()
    class Blocking:
        kind='fixture'
        metadata={'provider':'unit-test','model':'test','usage':{}}
        def __call__(self,messages):
            started.set(); released.wait(10)
            return '{"action":"finish","sql":"SELECT 1"}'
    with TestClient(create_app(config,policy=Blocking()),base_url='http://127.0.0.1') as client, ThreadPoolExecutor(1) as pool:
        first=pool.submit(client.post,'/api/analyze',json={'database':'shop','question':'x'})
        assert started.wait(5)
        try:
            assert client.post('/api/analyze',json={'database':'shop','question':'y'}).status_code==429
        finally: released.set()
        assert first.result(10).status_code==200
