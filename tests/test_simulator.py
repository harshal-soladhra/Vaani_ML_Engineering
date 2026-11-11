import pytest
from fastapi.testclient import TestClient
from src.simulator.server import app, config, SimulatorConfig

client = TestClient(app)

def test_simulator_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "healthy"
    assert "in_flight" in data

def test_simulator_configure_and_reset():
    resp = client.post("/configure", json={"rpm_limit": 5, "tpm_limit": 2000, "daily_request_limit": 50})
    assert resp.status_code == 200
    assert resp.json()["config"]["rpm_limit"] == 5

    reset_resp = client.post("/reset")
    assert reset_resp.status_code == 200
    assert reset_resp.json()["status"] == "reset"

def test_simulator_daily_exhaustion():
    client.post("/configure", json={"daily_budget_exhausted": True})
    resp = client.post("/openai/v1/chat/completions", json={
        "messages": [{"role": "user", "content": "test question"}],
        "stream": False
    })
    assert resp.status_code == 429
    data = resp.json()
    assert data["error"]["type"] == "daily_quota_exceeded"
    assert resp.headers.get("x-ratelimit-remaining-requests") == "0"
    client.post("/reset")

def test_simulator_rpm_limit():
    client.post("/configure", json={"rpm_limit": 2})
    # Request 1
    r1 = client.post("/openai/v1/chat/completions", json={"messages": [{"role": "user", "content": "hi"}], "stream": False})
    # Request 2
    r2 = client.post("/openai/v1/chat/completions", json={"messages": [{"role": "user", "content": "hi"}], "stream": False})
    # Request 3 should hit RPM 429
    r3 = client.post("/openai/v1/chat/completions", json={"messages": [{"role": "user", "content": "hi"}], "stream": False})
    assert r3.status_code == 429
    assert r3.json()["error"]["type"] == "requests"
    client.post("/reset")
