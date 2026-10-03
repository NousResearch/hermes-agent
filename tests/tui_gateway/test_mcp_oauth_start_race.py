"""Concurrent start_flow calls must not defeat the capacity cap or the duplicate guard."""

import asyncio
import threading
import time

import pytest

from tui_gateway import mcp_oauth_sessions as sessions


def _fake_worker(*_args, flow, on_done=None):
    """Publish an authorization URL and stay pending so the flow counts as active."""
    asyncio.run(flow.publish_authorization_url("https://idp.example/authorize?state=test"))


def _slow_receiver(*_a, **_k):
    """Stand in for a real loopback bind; the delay widens the check-to-register window."""
    time.sleep(0.05)
    return None


def _run_concurrent(monkeypatch, server_names):
    monkeypatch.setattr(sessions, "_sessions", {})
    monkeypatch.setattr(sessions, "run_worker", _fake_worker)
    monkeypatch.setattr(sessions, "choose_callback_receiver", _slow_receiver)
    home = "/tmp/oauth-race-home"
    monkeypatch.setenv("HERMES_HOME", home)
    barrier = threading.Barrier(len(server_names))
    results, errors = [], []
    lock = threading.Lock()

    def attempt(server_name):
        barrier.wait(10)
        try:
            out = sessions.start_flow(home, server_name, {"url": "https://mcp.example"})
            with lock:
                results.append(out)
        except Exception as exc:  # noqa: BLE001 - collect the guard failures
            with lock:
                errors.append(exc)

    threads = [threading.Thread(target=attempt, args=(name,)) for name in server_names]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(15)
    assert all(not thread.is_alive() for thread in threads), "a start_flow call hung"
    return results, errors


def test_concurrent_duplicate_starts_register_exactly_one_flow(monkeypatch):
    results, errors = _run_concurrent(monkeypatch, ["reports"] * 8)
    assert len(results) == 1
    assert len(errors) == 7
    assert all("already in progress" in str(exc) for exc in errors)
    assert len(sessions._sessions) == 1


def test_concurrent_starts_never_exceed_the_pending_cap(monkeypatch):
    over = sessions._MAX_PENDING + 4
    names = [f"srv-{i}" for i in range(over)]
    results, errors = _run_concurrent(monkeypatch, names)
    assert len(results) == sessions._MAX_PENDING
    assert len(errors) == 4
    assert all("Too many MCP OAuth flows" in str(exc) for exc in errors)
    assert len(sessions._sessions) == sessions._MAX_PENDING


def test_a_receiver_bind_failure_releases_the_reserved_slot(monkeypatch):
    monkeypatch.setattr(sessions, "_sessions", {})
    monkeypatch.setattr(sessions, "run_worker", _fake_worker)
    home = "/tmp/oauth-race-home"
    monkeypatch.setenv("HERMES_HOME", home)

    def broken(*_a, **_k):
        raise OSError("no free loopback port")

    monkeypatch.setattr(sessions, "choose_callback_receiver", broken)
    with pytest.raises(OSError, match="no free loopback port"):
        sessions.start_flow(home, "reports", {"url": "https://mcp.example"})
    assert sessions._sessions == {}

    monkeypatch.setattr(sessions, "choose_callback_receiver", lambda *_a, **_k: None)
    out = sessions.start_flow(home, "reports", {"url": "https://mcp.example"})
    assert out["session_id"] in sessions._sessions


def _rpc_oauth_start_e2e(tmp_path, monkeypatch, calls):
    """Drive the real ``mcp.servers.oauth.start`` handler concurrently. ``calls`` is a list of
    ``params`` dicts; only the worker and the loopback bind are stubbed."""
    import tui_gateway.server as srv

    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(sessions, "_sessions", {})
    monkeypatch.setattr(sessions, "run_worker", _fake_worker)
    monkeypatch.setattr(sessions, "choose_callback_receiver", _slow_receiver)

    add = srv._methods["mcp.servers.add"]
    seeded = set()
    for params in calls:
        key = (params.get("profile"), params["name"])
        if key in seeded:
            continue
        seeded.add(key)
        if key[0]:
            (home / "profiles" / key[0]).mkdir(parents=True, exist_ok=True)
        resp = add(0, {**params, "config": {"url": "https://mcp.example"}})
        assert "error" not in resp, resp

    start = srv._methods["mcp.servers.oauth.start"]
    barrier = threading.Barrier(len(calls))
    results, errors = [], []
    lock = threading.Lock()

    def call(rid, params):
        barrier.wait(10)
        out = start(rid, dict(params))
        with lock:
            (results if "result" in out else errors).append(out)

    threads = [
        threading.Thread(target=call, args=(i, params)) for i, params in enumerate(calls, 1)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(15)
    assert all(not thread.is_alive() for thread in threads), "an oauth.start RPC hung"
    return results, errors


def test_e2e_concurrent_rpc_starts_hold_the_duplicate_guard(tmp_path, monkeypatch):
    """8 barrier-synchronized ``mcp.servers.oauth.start`` calls for one server: exactly one
    registers, the rest see the guard as a 5024, which is what a real RPC client observes."""
    results, errors = _rpc_oauth_start_e2e(
        tmp_path, monkeypatch, [{"name": "srv"}] * 8)
    assert len(results) == 1
    ok = results[0]["result"]
    assert ok["ok"] is True
    assert ok["auth_url"].startswith("https://idp.example")
    assert ok["session_id"] in sessions._sessions
    assert len(errors) == 7
    assert all(e["error"]["code"] == 5024 for e in errors)
    assert all("already in progress" in e["error"]["message"] for e in errors)
    assert len(sessions._sessions) == 1


def test_e2e_same_server_name_in_a_different_profile_is_not_a_duplicate(
    tmp_path, monkeypatch
):
    """The duplicate guard is (home, server)-scoped: another profile may OAuth a same-named
    server concurrently."""
    results, errors = _rpc_oauth_start_e2e(
        tmp_path,
        monkeypatch,
        [{"name": "srv", "profile": "p1"}, {"name": "srv", "profile": "p2"}],
    )
    assert errors == []
    assert len(results) == 2
    homes = {r["hermes_home"] for r in sessions._sessions.values()}
    assert len(homes) == 2
