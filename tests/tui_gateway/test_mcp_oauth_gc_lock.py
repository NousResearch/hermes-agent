"""start_flow's expired-session GC must not run blocking listener shutdowns under _sessions_lock."""

import asyncio
import threading
import time

from tools.mcp_dashboard_oauth import DashboardOAuthFlow
from tui_gateway import mcp_oauth_sessions as sessions


def _fake_worker(*_args, flow, on_done=None):
    """Publish an authorization URL and return; the flow stays pending."""
    asyncio.run(flow.publish_authorization_url("https://idp.example/authorize?state=test"))


def test_expired_session_gc_does_not_block_the_sessions_lock(monkeypatch, tmp_path):
    """An expired listener's blocking ``shutdown()`` must not run under ``_sessions_lock``;
    every session operation (lookup/poll/cancel) serializes behind it."""
    monkeypatch.setattr(sessions, "_sessions", {})
    monkeypatch.setattr(sessions, "run_worker", _fake_worker)
    monkeypatch.setattr(sessions, "choose_callback_receiver", lambda *_a, **_k: None)
    home = str(tmp_path / "home")
    monkeypatch.setenv("HERMES_HOME", home)

    release = threading.Event()
    shutdown_entered = threading.Event()

    class BlockingHttpd:
        def shutdown(self):
            shutdown_entered.set()
            release.wait(10)

        def server_close(self):
            pass

    stale = DashboardOAuthFlow(
        flow_id="stale-sid", server_name="stale", profile=None, hermes_home=home,
        redirect_uri="", reconnect_live=False)
    sessions._sessions["stale-sid"] = {
        "session_id": "stale-sid", "server_name": "stale", "hermes_home": home,
        "flow": stale, "httpd": BlockingHttpd(),
        "created_at": time.time() - sessions._SESSION_TTL_SECONDS - 60,
    }

    errors = []

    def starter():
        try:
            sessions.start_flow(home, "fresh", {"url": "https://mcp.example"})
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    starter_thread = threading.Thread(target=starter, daemon=True)
    starter_thread.start()
    assert shutdown_entered.wait(5), "GC never reached the stale listener's shutdown"

    try:
        # poll_flow acquires _sessions_lock via _lookup; it must return while the stale
        # listener's shutdown is still blocked, not after it.
        started = time.time()
        out = sessions.poll_flow("nonexistent", "fresh")
        elapsed = time.time() - started
    finally:
        release.set()
    starter_thread.join(10)
    assert not starter_thread.is_alive()
    assert not errors
    assert out["status"] == "error"  # "not found" payload; only the timing matters
    assert elapsed < 4, f"poll_flow blocked {elapsed:.2f}s behind a GC listener shutdown"
    assert "stale-sid" not in sessions._sessions


def test_e2e_gc_shutdown_does_not_stall_concurrent_poll_rpc(tmp_path, monkeypatch):
    """Real ``mcp.servers.oauth.start`` sweeping a wedged listener must not stall a real
    ``mcp.servers.oauth.poll`` handled on another thread."""
    import tui_gateway.server as srv

    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(sessions, "_sessions", {})
    monkeypatch.setattr(sessions, "run_worker", _fake_worker)
    monkeypatch.setattr(sessions, "choose_callback_receiver", lambda *_a, **_k: None)

    add = srv._methods["mcp.servers.add"]
    for name in ("live", "fresh"):
        resp = add(0, {"name": name, "config": {"url": "https://mcp.example"}})
        assert "error" not in resp, resp

    start = srv._methods["mcp.servers.oauth.start"]
    live = start(1, {"name": "live"})
    assert "result" in live, live
    live_sid = live["result"]["session_id"]

    release = threading.Event()
    shutdown_entered = threading.Event()

    class BlockingHttpd:
        def shutdown(self):
            shutdown_entered.set()
            release.wait(10)

        def server_close(self):
            pass

    stale = DashboardOAuthFlow(
        flow_id="stale-sid", server_name="stale", profile=None, hermes_home=str(home),
        redirect_uri="", reconnect_live=False)
    sessions._sessions["stale-sid"] = {
        "session_id": "stale-sid", "server_name": "stale", "hermes_home": str(home),
        "flow": stale, "httpd": BlockingHttpd(),
        "created_at": time.time() - sessions._SESSION_TTL_SECONDS - 60,
    }

    outcomes = {}

    def starter():
        outcomes["start"] = start(2, {"name": "fresh"})

    starter_thread = threading.Thread(target=starter, daemon=True)
    starter_thread.start()
    assert shutdown_entered.wait(5), "GC never reached the stale listener's shutdown"

    try:
        poll = srv._methods["mcp.servers.oauth.poll"]
        started = time.time()
        out = poll(3, {"name": "live", "session_id": live_sid})
        elapsed = time.time() - started
    finally:
        release.set()
    starter_thread.join(10)
    assert not starter_thread.is_alive()

    assert out["result"]["ok"] is True
    assert out["result"]["status"] == "pending"
    assert elapsed < 4, f"oauth.poll stalled {elapsed:.2f}s behind a GC listener shutdown"
    assert "result" in outcomes["start"], outcomes["start"]
    assert "stale-sid" not in sessions._sessions
