"""Every admission source participates in the same process retirement fence."""

from concurrent.futures import ThreadPoolExecutor
import queue
import threading

import pytest


@pytest.fixture
def runtime(monkeypatch):
    from tui_gateway import server
    from hermes_cli import backend_retirement

    fence = backend_retirement.RetirementFence()
    monkeypatch.setattr(backend_retirement, "retirement", fence)
    return server, fence


class Transport:
    def __init__(self):
        self.frames = queue.Queue()

    def write(self, frame):
        self.frames.put(frame)
        return True


def test_queued_and_running_rpc_hold_admission_until_the_response(runtime, monkeypatch):
    from hermes_cli.web_server_idle_proof import idle_proof

    server, fence = runtime
    release_pool, entered, release_handler = (threading.Event() for _ in range(3))
    transport = Transport()

    def handler(rid, params):
        entered.set()
        assert release_handler.wait(10)
        return server._ok(rid, {"done": True})

    monkeypatch.setitem(server._methods, "test.retirement", handler)
    monkeypatch.setattr(server, "_LONG_HANDLERS", {"test.retirement"})
    with ThreadPoolExecutor(max_workers=1) as pool:
        monkeypatch.setattr(server, "_pool", pool)
        blocked = pool.submit(release_pool.wait, 10)
        try:
            assert server.dispatch({"id": "test", "method": "test.retirement"}, transport) is None
            assert fence.prepare() == {"ok": False, "idle": False}
            assert idle_proof()["idle"] is False
            release_pool.set()
            blocked.result(10)
            assert entered.wait(10)
            assert fence.prepare() == {"ok": False, "idle": False}
            release_handler.set()
            assert transport.frames.get(timeout=10)["result"] == {"done": True}
        finally:
            release_pool.set()
            release_handler.set()
    assert idle_proof()["idle"] is True
    token = fence.prepare()["token"]
    assert server.dispatch({"id": "late", "method": "test.retirement"}, transport)["error"]["code"] == 5035
    assert fence.cancel(token) == {"ok": True}


def test_prompt_claim_and_automatic_continuations_cannot_cross_prepare(runtime, monkeypatch):
    server, fence = runtime
    session = {"history_lock": threading.RLock(), "running": False, "history": []}
    dispatched = []
    monkeypatch.setattr(server, "_run_prompt_submit", lambda *a, **kw: dispatched.append(a))
    token = fence.prepare()["token"]
    err, _ = server._lock_in_submit_turn("r", "s", session, "hello", {}, False, None, None, None)
    assert err and err["error"]["code"] == 5035
    assert session["running"] is False
    session["queued_prompt"] = {"text": "next", "transport": None}
    assert server._drain_queued_prompt("r", "s", session) is False
    assert session["queued_prompt"]["text"] == "next"
    assert server._notif_claim_turn(session) is False
    # Use a separate idle session so the goal branch is reached without the queued prompt.
    goal_session = {"history_lock": threading.RLock(), "running": False}
    server._run_post_turn_followups("r", "s", goal_session, {}, "keep going")
    assert goal_session["running"] is False
    assert not dispatched
    from types import SimpleNamespace
    from tools import bot_live_delivery

    monkeypatch.setattr(bot_live_delivery, "has_mailbox", lambda home: True)
    monkeypatch.setattr(bot_live_delivery, "find_canonical_live_owner", lambda home: {
        "lease_id": "l", "live_session_id": "s", "session_id": "stored"})
    monkeypatch.setattr(bot_live_delivery, "claim_pending_delivery", lambda *a: pytest.fail("retiring backend claimed a delivery"))
    mailbox_session = {"history_lock": threading.RLock(), "agent": object(), "session_key": "stored",
                       "active_session_lease": SimpleNamespace(lease_id="l", released=False)}
    assert server._poll_bot_live_delivery_once("s", mailbox_session) is False
    assert fence.cancel(token) == {"ok": True}
    assert server._notif_claim_turn(session) is True
    assert session["running"] is True
