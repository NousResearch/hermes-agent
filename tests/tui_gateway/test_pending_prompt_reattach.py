"""A blocking prompt must survive the client reconnecting.

``_block`` emits its request exactly once and then parks the turn thread on a
``threading.Event``. If the websocket drops before the user answers, that single
frame goes to the detached drop-transport and is lost: nothing buffers it, and
the reattach payload never carried the request either.

Nothing reaps the session afterwards, and that is deliberate -- both the WS
orphan reaper and the idle reaper spare a session that is running or parked on a
prompt. So the turn thread waits forever (a clarify configured with
``clarify_timeout <= 0`` passes ``timeout=None``), holding its agent, its slash
worker and its active-session lease until the process exits.

Contract pinned here: the live payload handed to a reattaching client carries
the outstanding request, keyed by the same ``request_id`` the waiter registered,
so answering it resolves the original waiter instead of hitting the generic
"no pending request" error.
"""

from __future__ import annotations

import contextlib
import threading
import time
import types

import pytest

from tui_gateway import server


def _session(**extra):
    return {
        "agent": types.SimpleNamespace(session_id="session-key"),
        "session_key": "session-key",
        "history": [],
        "history_lock": threading.Lock(),
        "history_version": 0,
        "running": True,
        "attached_images": [],
        "image_counter": 0,
        "cols": 80,
        "slash_worker": None,
        "show_reasoning": False,
        "tool_progress_mode": "all",
        "inflight_turn": None,
        "created_at": time.time(),
        **extra,
    }


@pytest.fixture()
def emits(monkeypatch):
    captured: list = []
    monkeypatch.setattr(
        server,
        "_emit",
        lambda event, sid, payload=None: captured.append((event, sid, payload)),
    )
    return captured


@pytest.fixture(autouse=True)
def isolated_prompt_registries(monkeypatch):
    """Keep the module-global blocking-prompt state per-test."""
    monkeypatch.setattr(server, "_pending", {})
    monkeypatch.setattr(server, "_pending_prompt_payloads", {})
    monkeypatch.setattr(server, "_answers", {})


@pytest.fixture(autouse=True)
def no_db(monkeypatch):
    monkeypatch.setattr(server, "_get_db", lambda: None)


@contextlib.contextmanager
def _outstanding_prompt(sid, event="clarify.request", payload=None, timeout=None):
    """Park a real ``_block`` waiter on *sid* for the body of the with-block."""
    result: dict = {}

    def _run():
        result["answer"] = server._block(event, sid, dict(payload or {}), timeout=timeout)

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    deadline = time.time() + 5.0
    while time.time() < deadline and not server._session_pending_kind(sid):
        time.sleep(0.005)
    assert server._session_pending_kind(sid), "_block never registered its request"
    try:
        yield result
    finally:
        # Join first: a test that answered the prompt must not have its answer
        # clobbered by the release-with-empty-string path.
        thread.join(timeout=2.0)
        if thread.is_alive():
            server._clear_pending(sid)
            thread.join(timeout=5.0)
        assert not thread.is_alive()


def test_live_payload_carries_the_outstanding_prompt(emits):
    """What session.resume / session.activate hand a reconnecting client."""
    session = _session()
    question = "Which environment should I deploy to?"
    choices = ["staging", "production"]

    with _outstanding_prompt(
        "sid", payload={"question": question, "choices": choices}
    ):
        payload = server._live_session_payload("sid", session)

        pending = payload.get("pending")
        assert pending is not None, "reattach payload dropped the outstanding prompt"
        assert pending["kind"] == "clarify"
        assert pending["event"] == "clarify.request"
        assert pending["payload"]["question"] == question
        assert pending["payload"]["choices"] == choices


def test_pending_request_id_matches_the_live_waiter(emits):
    """The replayed id must be the waiter's, so a late answer still resolves it."""
    session = _session()

    with _outstanding_prompt("sid", payload={"question": "q", "choices": []}) as result:
        payload = server._live_session_payload("sid", session)
        request_id = payload["pending"]["request_id"]

        assert request_id in server._pending
        assert server._pending[request_id][0] == "sid"

        respond = server._methods["clarify.respond"]
        reply = respond("rpc-1", {"request_id": request_id, "answer": "staging"})
        assert "error" not in reply

    assert result["answer"] == "staging"


def test_no_pending_key_when_nothing_is_blocked(emits):
    """Absent, not None -- clients branch on the key the way they do for inflight."""
    payload = server._live_session_payload("sid", _session(running=False))
    assert "pending" not in payload


def test_pending_does_not_leak_across_sessions(emits):
    """A prompt raised by another session must not surface in this payload."""
    session = _session()

    with _outstanding_prompt("other-sid", payload={"question": "q", "choices": []}):
        payload = server._live_session_payload("sid", session)
        assert "pending" not in payload


def test_reattach_replays_without_re_emitting_the_request(emits):
    """Replay rides the resume payload; it must not fire a second event frame."""
    session = _session()

    with _outstanding_prompt("sid", payload={"question": "q", "choices": []}):
        emitted_before = [e for e in emits if e[0] == "clarify.request"]
        assert len(emitted_before) == 1

        server._live_session_payload("sid", session)
        server._live_session_payload("sid", session)

        emitted_after = [e for e in emits if e[0] == "clarify.request"]
        assert len(emitted_after) == 1, "reattach duplicated the prompt"


@pytest.mark.parametrize(
    ("event", "kind"),
    [
        ("clarify.request", "clarify"),
        ("sudo.request", "sudo"),
        ("secret.request", "secret"),
        ("terminal.read.request", "terminal.read"),
    ],
)
def test_every_blocking_bridge_is_replayable(emits, event, kind):
    """All four _block bridges share one lifecycle, so all four must replay."""
    session = _session()

    with _outstanding_prompt("sid", event=event, payload={}):
        payload = server._live_session_payload("sid", session)
        assert payload["pending"]["kind"] == kind
        assert payload["pending"]["event"] == event
