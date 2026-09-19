"""(#111873) Desktop composer scheduled messages — firing path and RPC contract.

The messenger gesture: a draft the user defers arrives later as an ORDINARY user turn in the same
conversation. Firing rides the per-session notification poller that already drives /loop and
/heartbeat, claims the idle session first, and claims the item (pending -> fired, persisted) BEFORE
dispatch so a repeat scan cannot re-send.
"""

from __future__ import annotations

import importlib
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture()
def hermes_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    from hermes_cli import goals

    goals._DB_CACHE.clear()
    yield home
    goals._DB_CACHE.clear()


@pytest.fixture()
def server(hermes_home):
    with patch.dict("sys.modules", {"hermes_cli.env_loader": MagicMock(), "hermes_cli.banner": MagicMock()}):
        mod = importlib.import_module("tui_gateway.server")
        yield mod
        mod._sessions.clear()


@pytest.fixture()
def session(server):
    sid, key = "sid-sched-test", "tui-sched-session-1"
    s = {"session_key": key, "history": [], "history_lock": threading.Lock(), "history_version": 0,
         "running": False, "attached_images": [], "cols": 120, "agent": MagicMock()}
    server._sessions[sid] = s
    return sid, key, s


def _schedule_via_rpc(server, sid, text, *, due_delta=60.0, extra=None):
    params = {"session_id": sid, "text": text, "due_at": time.time() + due_delta}
    params.update(extra or {})
    return server._methods["schedule.message.create"](1, params)


def _result(envelope):
    return (envelope or {}).get("result") or {}


def _fire(server, sid, session):
    with patch.object(server, "_emit"):
        return server._maybe_fire_tui_scheduled_message(sid, session)


def _capturing_submit(sink):
    def submit(rid, sid, session, text, **kwargs):
        sink.append((text, kwargs))
        return True

    return submit


# ── create: writes, never sends ───────────────────────────────────────────────

def test_create_only_writes_pending(server, session):
    sid, _key, s = session
    submit = MagicMock(return_value=True)
    with patch.object(server, "_run_prompt_submit", submit), patch.object(server, "_emit"):
        response = _schedule_via_rpc(server, sid, "run the migration at 3am")

    result = _result(response)
    assert result["message"]["text"] == "run the migration at 3am"
    assert [item["text"] for item in result["messages"]] == ["run the migration at 3am"]
    assert result["message"]["status"] == "pending"
    submit.assert_not_called()
    assert s["running"] is False


def test_wire_payload_carries_no_provider_model_or_delivery_choices(server, session):
    """The issue's acceptance in one assertion: the default flow exposes nothing to configure."""
    sid, _key, _s = session
    result = _result(_schedule_via_rpc(server, sid, "hello"))

    assert set(result["message"]) == {"id", "text", "display_text", "due_at", "created_at", "status", "overdue"}
    forbidden = ("provider", "model", "delivery", "target", "cron", "recurring", "schedule_expr")
    assert not [key for key in result["message"] if any(word in key for word in forbidden)]


def test_list_returns_only_this_sessions_pending(server, session):
    sid, _key, _s = session
    _schedule_via_rpc(server, sid, "mine")

    result = _result(server._methods["schedule.message.list"](1, {"session_id": sid}))
    assert [item["text"] for item in result["messages"]] == ["mine"]


# ── firing ────────────────────────────────────────────────────────────────────

def test_due_message_fires_as_a_plain_user_turn(server, session):
    sid, key, s = session
    _schedule_via_rpc(server, sid, "say hello", due_delta=-5.0)
    dispatched: list = []

    with patch.object(server, "_run_prompt_submit", _capturing_submit(dispatched)), patch.object(server, "_emit"):
        server._maybe_fire_tui_scheduled_message(sid, s)

    assert [text for text, _kw in dispatched] == ["say hello"]
    text, kwargs = dispatched[0]
    # No display_kind / metadata: it must render as if the user had typed it.
    assert kwargs == {}
    assert s["running"] is True                    # claimed for the delivered turn
    assert _result(server._methods["schedule.message.list"](1, {"session_id": sid}))["messages"] == []


def test_not_due_yet_does_not_fire(server, session):
    sid, _key, s = session
    _schedule_via_rpc(server, sid, "later", due_delta=600.0)
    submit = MagicMock(return_value=True)

    with patch.object(server, "_run_prompt_submit", submit), patch.object(server, "_emit"):
        server._maybe_fire_tui_scheduled_message(sid, s)

    submit.assert_not_called()
    assert s["running"] is False


def test_repeat_scan_fires_at_most_once(server, session):
    sid, _key, s = session
    _schedule_via_rpc(server, sid, "only once", due_delta=-5.0)
    dispatched: list = []

    with patch.object(server, "_run_prompt_submit", _capturing_submit(dispatched)), patch.object(server, "_emit"):
        server._maybe_fire_tui_scheduled_message(sid, s)
        for _ in range(3):                                  # a coarse poll scans repeatedly
            s["running"] = False
            server._maybe_fire_tui_scheduled_message(sid, s)

    assert [text for text, _kw in dispatched] == ["only once"]


def test_busy_session_defers_instead_of_opening_a_concurrent_turn(server, session):
    sid, _key, s = session
    _schedule_via_rpc(server, sid, "queued behind the live turn", due_delta=-5.0)
    s["running"] = True                                     # a turn is in flight
    dispatched: list = []

    with patch.object(server, "_run_prompt_submit", _capturing_submit(dispatched)), patch.object(server, "_emit"):
        server._maybe_fire_tui_scheduled_message(sid, s)
        assert dispatched == []
        # still pending AND still due: the poller must not have consumed it
        pending = _result(server._methods["schedule.message.list"](1, {"session_id": sid}))["messages"]
        assert [item["text"] for item in pending] == ["queued behind the live turn"]

        s["running"] = False                                # the live turn ends
        server._maybe_fire_tui_scheduled_message(sid, s)

    assert [text for text, _kw in dispatched] == ["queued behind the live turn"]
    assert s["running"] is True


def test_a_dispatch_that_never_starts_a_turn_stays_pending(server, session):
    sid, _key, s = session
    _schedule_via_rpc(server, sid, "retry me", due_delta=-5.0)

    def refuse(*_args, **_kwargs):
        with s["history_lock"]:
            s["running"] = False                            # _admit_prompt_turn releases itself
        return False

    with patch.object(server, "_run_prompt_submit", refuse), patch.object(server, "_emit"):
        server._maybe_fire_tui_scheduled_message(sid, s)

    assert s["running"] is False
    pending = _result(server._methods["schedule.message.list"](1, {"session_id": sid}))["messages"]
    assert [item["text"] for item in pending] == ["retry me"], "a refused dispatch must not consume it"


def test_cancelled_message_never_fires(server, session):
    sid, _key, s = session
    message_id = _result(_schedule_via_rpc(server, sid, "never mind", due_delta=-5.0))["message"]["id"]

    cancelled = _result(server._methods["schedule.message.cancel"](
        1, {"session_id": sid, "message_id": message_id}))
    assert cancelled["cancelled"] is True and cancelled["messages"] == []

    submit = MagicMock(return_value=True)
    with patch.object(server, "_run_prompt_submit", submit), patch.object(server, "_emit"):
        server._maybe_fire_tui_scheduled_message(sid, s)
    submit.assert_not_called()


def test_survives_a_backend_restart(server, session):
    """Created, then the store is read by a "new process" (fresh DB handle + fresh session dict)."""
    sid, key, _s = session
    _schedule_via_rpc(server, sid, "after the restart", due_delta=-5.0)

    from hermes_cli import goals

    goals._DB_CACHE.clear()                                 # new process: new connection, same state.db
    revived = {"session_key": key, "history": [], "history_lock": threading.Lock(), "running": False,
               "agent": MagicMock()}
    dispatched: list = []
    with patch.object(server, "_run_prompt_submit", _capturing_submit(dispatched)), patch.object(server, "_emit"):
        server._maybe_fire_tui_scheduled_message(sid, revived)

    assert [text for text, _kw in dispatched] == ["after the restart"]


def test_session_without_a_durable_id_is_refused(server):
    sid, s = "sid-no-key", {"session_key": "", "history": [], "history_lock": threading.Lock(), "running": False}
    server._sessions[sid] = s

    envelope = _schedule_via_rpc(server, sid, "homeless")
    assert "error" in envelope and envelope["error"]["code"] == 4125


# ── integration: the real poll loop ───────────────────────────────────────────

def test_notification_poller_fires_a_due_scheduled_message(server, session):
    """The per-session poller itself is the driver (same loop that fires /loop and /heartbeat)."""
    sid, _key, s = session
    _schedule_via_rpc(server, sid, "poller delivers me", due_delta=-5.0)
    dispatched: list[str] = []
    stop = threading.Event()

    def submit(rid, sid_, session_, text, **kwargs):
        dispatched.append(text)
        return True

    with patch.object(server, "_run_prompt_submit", submit), patch.object(server, "_emit"):
        thread = threading.Thread(target=server._notification_poller_loop, args=(stop, sid, s), daemon=True)
        thread.start()
        deadline = time.monotonic() + 8
        while not dispatched and time.monotonic() < deadline:
            time.sleep(0.1)
        stop.set()
        thread.join(timeout=5)

    assert dispatched == ["poller delivers me"]
