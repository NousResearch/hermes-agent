"""Behavioral RPC contracts with real imports and disposable profile stores."""
import json
import threading
from pathlib import Path

import pytest

from hermes_state import SessionDB
from hermes_state_usage_events import UsageEvent
from tui_gateway import server
from tui_gateway.transport import bind_transport, reset_transport

METHODS = ("usage.codex_quota", "usage.codex_timeline", "usage.active_work")


def call(method, params=None, transport=None):
    token = bind_transport(transport if transport is not None else server._stdio_transport)
    try:
        return server.handle_request({"id": 1, "method": method, "params": params or {}})
    finally:
        reset_transport(token)


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "_hermes_home", str(tmp_path))
    monkeypatch.setattr(server, "_sessions", {})
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    return tmp_path


def test_timeline_rpc_is_registered_exact_bounded_and_profile_local(home, monkeypatch):
    import hashlib
    import hermes_state_usage_events as events
    from hermes_constants import hermes_home_key

    now = 1_800_000_000_000_000
    monkeypatch.setattr(events.time, "time_ns", lambda: now * 1000)
    profile = hashlib.sha256(hermes_home_key(home).encode()).hexdigest()
    with_db = SessionDB(home / "state.db")
    try:
        for identity, scope, stamp, tokens in [
            ("included", profile, now - 1, 13),
            ("left-edge", profile, now - events.WINDOW_US, 7),
            ("excluded", profile, now, 99),
            ("foreign", "another-profile", now - 1, 500),
        ]:
            assert with_db.record_usage_event(UsageEvent(identity, "openai-codex", "model",
                scope, stamp, input_tokens=tokens, output_tokens=0), now_us=now)
    finally:
        with_db.close()
    response = call("usage.codex_timeline")
    assert "result" in response, response
    data = response["result"]
    assert data["scope"]["profile"] == profile
    assert data["scope"]["device_local"] is True
    assert data["scope"]["device"] == "backend"
    assert data["as_of_us"] == now
    assert len(data["bins"]) == 24
    assert data["bins"][0]["start_us"] == now - events.WINDOW_US
    assert data["bins"][-1]["end_us"] == now
    assert sum(b["usage"]["processed_tokens"] for b in data["bins"]) == data["total"]["processed_tokens"] == 20
    assert data["coverage"]["status"] == "partial"
    assert data["freshness"]["status"] == "fresh"
    wire = json.dumps(data)
    assert str(home) not in wire and "model" not in wire
    assert len(wire) < 32000


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("params", [
    {"profile": "other"}, {"db_path": "fixture-secret"}, {"api_key": "fixture-secret"},
    {"as_of_us": 1}, {"bins": 1000000}, {"scope": "remote"}, {"limit": -1},
])
def test_rpc_rejects_all_renderer_selectors(home, method, params):
    response = call(method, params)
    assert response["error"]["code"] == -32602
    assert "fixture-secret" not in json.dumps(response)


@pytest.mark.parametrize("method", METHODS)
def test_rpc_rejects_untrusted_transport_before_reading(home, method):
    response = call(method, transport=object())
    assert response["error"]["code"] == 4403
    assert not (home / "state.db").exists()


def test_active_work_counts_only_launch_profile_and_never_contents(home, monkeypatch):
    def session(profile=None, running=False, queued=0):
        return {"profile_home": profile, "running": running,
                "history_lock": threading.Lock(), "history": ["fixture-secret"],
                "queued_prompt": {"text": "fixture-secret"} if queued else None,
                "queued_prompts": [{"text": "fixture-secret"}] * max(0, queued - 1)}
    monkeypatch.setattr(server, "_sessions", {
        "one": session(running=True, queued=2),
        "two": session(running=True),
        "foreign": session(str(home / "other"), True, 7),
    })
    result = call("usage.active_work")["result"]
    categories = result["categories"]
    assert categories["agent_turns"]["count"] == 2
    assert categories["queued_work"]["count"] == 2
    assert result["load"]["level"] == "heavy"
    assert result["coverage"]["status"] == "partial"
    assert "fixture-secret" not in json.dumps(result)
    assert categories["cron_executions"]["count"] is None
    assert categories["cron_executions"]["status"] == "unknown"


def test_slow_telemetry_dispatch_uses_bound_transport(home, monkeypatch):
    import asyncio
    from tui_gateway.ws import WSTransport
    loop = asyncio.new_event_loop()
    received = []
    done = threading.Event()
    transport = WSTransport(object(), loop, authenticated=True, local_telemetry=True)
    transport.write = lambda frame: (received.append(frame), done.set(), True)[-1]
    try:
        assert server.dispatch({"id": 1, "method": "usage.codex_timeline", "params": {}}, transport) is None
        assert done.wait(5)
        data = received[0]["result"]
        assert data["total"] is None
        assert len(data["bins"]) == 24 and all(b["usage"] is None for b in data["bins"])
        assert not (home / "state.db").exists()
    finally:
        loop.close()


@pytest.mark.parametrize("rid", ["x" * 33000, {"secret": "fixture-secret"}, 10**100], ids=["long", "object", "huge-int"])
@pytest.mark.parametrize("params", [{}, [], "fixture-secret"])
def test_response_id_is_bounded_and_invalid_ids_are_not_reflected(home, rid, params):
    binding = bind_transport(server._stdio_transport)
    try:
        response = server.handle_request({"id": rid, "method": "usage.active_work", "params": params})
    finally:
        reset_transport(binding)
    assert response["error"]["code"] == -32600
    assert response["id"] is None
    assert len(json.dumps(response)) < 256


def test_quota_rpc_uses_only_profile_store_without_runtime_recovery(home, monkeypatch):
    import httpx
    from agent import account_usage
    seen = []
    client = httpx.Client
    def respond(request):
        seen.append(request.headers["authorization"])
        return httpx.Response(200, json={"rate_limit": {"primary_window": {"used_percent": 0}},
            "email": "fixture-secret", "plan_type": "fixture-secret"})
    monkeypatch.setattr(account_usage.httpx, "Client", lambda **kw: client(transport=httpx.MockTransport(respond), **kw))
    def forbidden(**kw):
        raise AssertionError("runtime recovery must not run")
    monkeypatch.setattr(account_usage, "resolve_codex_runtime_credentials", forbidden)
    # No profile credential must not fall back to external CLI / another pool account.
    assert call("usage.codex_quota")["result"]["quota"]["error"] == "auth"
    for name in ("first", "second"):
        profile_home = home / name
        profile_home.mkdir()
        (profile_home / "auth.json").write_text(json.dumps({"providers": {"openai-codex": {
            "tokens": {"access_token": name + "-fixture-secret", "refresh_token": "fixture-refresh"}}}}))
        monkeypatch.setattr(server, "_hermes_home", str(profile_home))
        data = call("usage.codex_quota")["result"]
        assert data["quota"]["windows"][0]["used_percent"] == 0
        assert data["quota"]["windows"][1]["used_percent"] is None
        assert data["account_identity"] == "unknown"
        assert "fixture-secret" not in json.dumps(data)
        assert not data["freshness"]["cached"]
    assert seen == ["Bearer first-fixture-secret", "Bearer second-fixture-secret"]


def test_quota_rpc_discards_result_when_stored_credential_changes(home, monkeypatch):
    import httpx
    from agent import account_usage
    path = home / "auth.json"
    def store(value):
        path.write_text(json.dumps({"providers": {"openai-codex": {
            "tokens": {"access_token": value, "refresh_token": "fixture-refresh"}}}}))
    store("first-fixture-secret")
    client = httpx.Client
    def respond(request):
        store("second-fixture-secret")
        return httpx.Response(200, json={"rate_limit": {"primary_window": {"used_percent": 99}}})
    monkeypatch.setattr(account_usage.httpx, "Client", lambda **kw: client(transport=httpx.MockTransport(respond), **kw))
    data = call("usage.codex_quota")["result"]
    assert data["quota"]["status"] == "error"
    assert data["freshness"]["status"] == "unknown"
    assert "windows" not in data["quota"]
