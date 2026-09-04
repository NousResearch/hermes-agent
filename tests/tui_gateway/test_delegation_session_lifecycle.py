"""Fail-closed ownership + session-scoped delegation lifecycle (#55578).

Covers the two hardening rules layered on top of the origin-routing salvage:

1. ``_session_owns_notification_event`` — positive-proof ownership. An
   async-delegation completion may only be injected into a session that
   PROVABLY commissioned it (origin UI id, or session-key/lineage match).
   Orphans are never adopted by a foreign chat.

2. ``interrupt_for_session`` — a session's in-flight async delegations end
   with the session. ``_finalize_session`` interrupts delegations owned by
   the closing session (by origin UI id always; by durable key only when the
   TUI owns the lifecycle).
"""

import threading
from unittest.mock import MagicMock, patch

import pytest

import tools.async_delegation as ad
from tui_gateway.server import (
    _finalize_session,
    _session_owns_notification_event,
)


@pytest.fixture(autouse=True)
def _reset_async_delegation():
    ad._reset_for_tests()
    yield
    ad._reset_for_tests()


class TestSessionOwnsNotificationEvent:
    def _session(self, key="sess_key_1"):
        return {"session_key": key, "_finalized": False}

    def test_origin_ui_match_owns(self):
        evt = {"type": "async_delegation", "origin_ui_session_id": "tab1", "session_key": "other"}
        assert _session_owns_notification_event("tab1", self._session(), evt) is True

    def test_session_key_match_owns(self):
        evt = {"type": "async_delegation", "origin_ui_session_id": "", "session_key": "sess_key_1"}
        assert _session_owns_notification_event("tabX", self._session("sess_key_1"), evt) is True

    def test_orphan_is_not_owned(self):
        """No origin match, no key match, owner gone → NOT ours (fail closed)."""
        evt = {"type": "async_delegation", "origin_ui_session_id": "dead_tab", "session_key": "gone_key"}
        assert _session_owns_notification_event("tab1", self._session(), evt) is False

    def test_empty_key_and_origin_not_owned(self):
        """A delegation event with no return address at all is never adopted."""
        evt = {"type": "async_delegation", "origin_ui_session_id": "", "session_key": ""}
        assert _session_owns_notification_event("tab1", self._session(), evt) is False

    def test_finalized_session_owns_nothing(self):
        evt = {"type": "async_delegation", "origin_ui_session_id": "tab1", "session_key": "sess_key_1"}
        sess = self._session()
        sess["_finalized"] = True
        assert _session_owns_notification_event("tab1", sess, evt) is False

    def test_compression_chain_resolution_owns(self):
        evt = {"type": "async_delegation", "origin_ui_session_id": "", "session_key": "parent_key"}
        db = MagicMock()
        db.resolve_resume_session_id.return_value = "child_key"
        with patch("tui_gateway.server._get_db", return_value=db):
            assert _session_owns_notification_event("tabX", self._session("child_key"), evt) is True


class TestInterruptForSession:
    def _seed_record(self, delegation_id, session_key="", origin_ui_session_id="", status="running"):
        fn = MagicMock()
        with ad._records_lock:
            ad._records[delegation_id] = {
                "delegation_id": delegation_id,
                "status": status,
                "session_key": session_key,
                "origin_ui_session_id": origin_ui_session_id,
                "interrupt_fn": fn,
            }
        return fn

    def test_interrupts_only_matching_session(self):
        mine = self._seed_record("d1", session_key="sess_A")
        other = self._seed_record("d2", session_key="sess_B")
        n = ad.interrupt_for_session(session_key="sess_A")
        assert n == 1
        mine.assert_called_once()
        other.assert_not_called()

    def test_matches_by_origin_ui_session_id(self):
        mine = self._seed_record("d1", origin_ui_session_id="tab1")
        other = self._seed_record("d2", origin_ui_session_id="tab2")
        n = ad.interrupt_for_session(origin_ui_session_id="tab1")
        assert n == 1
        mine.assert_called_once()
        other.assert_not_called()

    def test_no_selector_is_noop(self):
        fn = self._seed_record("d1", session_key="sess_A")
        assert ad.interrupt_for_session() == 0
        fn.assert_not_called()

    def test_completed_records_untouched(self):
        fn = self._seed_record("d1", session_key="sess_A", status="completed")
        assert ad.interrupt_for_session(session_key="sess_A") == 0
        fn.assert_not_called()


class TestFinalizeInterruptsOwnDelegations:
    def _make_session(self, session_key="sess_A", sid="tab1"):
        agent = MagicMock()
        agent.session_id = session_key
        agent._session_messages = None
        agent.model = "m"
        agent.platform = "tui"
        return {
            "agent": agent,
            "history": [{"role": "user", "content": "x"}],
            "history_lock": threading.Lock(),
            "session_key": session_key,
            "_finalized": False,
            "_sid": sid,
        }

    @patch("tui_gateway.server._get_db")
    def test_finalize_interrupts_sessions_delegations(self, mock_get_db):
        mock_db = MagicMock()
        mock_db.get_session.return_value = {"source": "tui"}
        mock_get_db.return_value = mock_db

        with patch("tools.async_delegation.interrupt_for_session") as mock_int:
            _finalize_session(self._make_session(), end_reason="tui_close")

        mock_int.assert_called_once()
        kwargs = mock_int.call_args.kwargs
        assert kwargs["session_key"] == "sess_A"
        assert kwargs["origin_ui_session_id"] == "tab1"

    @patch("tui_gateway.server._get_db")
    def test_viewer_of_gateway_session_only_interrupts_by_origin(self, mock_get_db):
        """Closing a TUI viewer tab on a live gateway session must not kill
        the gateway's own background work — key-based interrupt is skipped,
        origin-id interrupt (this tab's own dispatches) still applies."""
        mock_db = MagicMock()
        mock_db.get_session.return_value = {"source": "telegram"}
        mock_get_db.return_value = mock_db

        with patch("tools.async_delegation.interrupt_for_session") as mock_int:
            _finalize_session(
                self._make_session(session_key="agent:main:telegram:dm:123", sid="tab9"),
                end_reason="ws_orphan_reap",
            )

        kwargs = mock_int.call_args.kwargs
        assert kwargs["session_key"] == ""
        assert kwargs["origin_ui_session_id"] == "tab9"


# ---------------------------------------------------------------------------
# 3. Parked completions reach the owning session when it comes back
#
# Incident: the backend was reaped mid-delegation; on restart the recovered row was replayed
# onto the shared queue while only an UNRELATED chat was open, whose poller dropped the
# in-memory copy. The durable row stayed pending and nothing ever delivered it to the owner.
# ---------------------------------------------------------------------------

import queue as _queue
import time as _time

from tui_gateway import server as _server


def _wait_for(predicate, timeout: float = 5.0) -> bool:
    deadline = _time.monotonic() + timeout
    while _time.monotonic() < deadline:
        if predicate():
            return True
        _time.sleep(0.02)
    return False


@pytest.fixture
def _poller_harness(monkeypatch):
    """Real poller threads against the temp HERMES_HOME ledger; pollers + sessions are reaped after."""
    from tools.process_registry import process_registry

    while not process_registry.completion_queue.empty():
        process_registry.completion_queue.get_nowait()
    submits: list = []

    def _fake_submit(rid, sid, session, text, **kwargs):
        submits.append({"sid": sid, "text": text, **kwargs})
        session["running"] = False  # the turn finished

    monkeypatch.setattr(_server, "_emit", lambda event, sid, payload=None: None)
    monkeypatch.setattr(_server, "_run_prompt_submit", _fake_submit)
    monkeypatch.setattr("gateway.status._pid_exists", lambda pid: False)
    registered: list = []

    def start(sid: str, session_key: str) -> dict:
        session = {"session_key": session_key, "history_lock": threading.Lock(), "running": False,
                   "_finalized": False}
        with _server._sessions_lock:
            _server._sessions[sid] = session
        registered.append(sid)
        session["_notif_stop"] = _server._start_notification_poller(sid, session)
        return session

    yield start, submits
    pollers = [(stop, th) for stop, th in list(_server._notification_pollers) if th.is_alive()]
    for stop, _th in pollers:
        stop.set()
    deadline = _time.time() + 3.0
    for _stop, th in pollers:
        th.join(timeout=max(0.0, deadline - _time.time()))
    _server._notification_pollers[:] = [(s, th) for s, th in _server._notification_pollers if th.is_alive()]
    with _server._sessions_lock:
        for sid in registered:
            _server._sessions.pop(sid, None)
    while not process_registry.completion_queue.empty():
        process_registry.completion_queue.get_nowait()


def _park_row_from_dead_backend(delegation_id: str, session_key: str) -> None:
    """The incident's durable state: dispatched by a backend that died, recovered as ``unknown``."""
    ad._persist_dispatch({
        "delegation_id": delegation_id, "goal": "reaped mid-work", "context": None, "toolsets": None,
        "role": "leaf", "model": "m", "session_key": session_key, "origin_ui_session_id": "dead-ui-sid",
        "parent_session_id": session_key, "status": "running", "dispatched_at": _time.time() - 5.0,
        "completed_at": None, "interrupt_fn": None})
    assert ad.recover_abandoned_delegations() == 1


def test_parked_completion_is_delivered_when_the_owning_session_resumes(_poller_harness):
    from tools.process_registry import process_registry
    start, submits = _poller_harness
    _park_row_from_dead_backend("d-incident", "owner-key")

    # Process start: the recovered row is replayed onto the shared queue...
    assert ad.restore_undelivered_completions(process_registry.completion_queue) == 1
    # ...while only an unrelated chat is open: its poller must not deliver it (fail-closed drop).
    start("sid-unrelated", "unrelated-key")
    assert _wait_for(lambda: process_registry.completion_queue.empty())
    _time.sleep(0.3)
    assert submits == []
    assert ad.get_durable_delegation("d-incident")["delivery_state"] == "pending"

    # The owner comes back (session_key == the row's origin_session): its poller must deliver it.
    start("sid-owner", "owner-key")
    assert _wait_for(lambda: len(submits) == 1), "parked completion never reached the owning session"
    assert submits[0]["sid"] == "sid-owner"
    assert submits[0]["display_kind"] == "async_delegation_complete"
    assert "d-incident" in submits[0]["text"]
    assert _wait_for(lambda: ad.get_durable_delegation("d-incident")["delivery_state"] == "delivered")


def test_parked_completion_is_not_replayed_to_a_session_that_does_not_own_it(_poller_harness):
    start, submits = _poller_harness
    _park_row_from_dead_backend("d-foreign", "owner-key")

    start("sid-unrelated", "unrelated-key")
    _time.sleep(0.5)
    assert submits == []
    assert ad.get_durable_delegation("d-foreign")["delivery_state"] == "pending"


def test_replay_does_not_double_deliver_a_copy_still_queued_from_process_start(_poller_harness):
    """Process-start copy still in memory + session-resume copy: exactly one turn, session left idle."""
    from tools.process_registry import process_registry
    start, submits = _poller_harness
    _park_row_from_dead_backend("d-twice", "owner-key")
    assert ad.restore_undelivered_completions(process_registry.completion_queue) == 1

    session = start("sid-owner", "owner-key")
    assert _wait_for(lambda: process_registry.completion_queue.empty()
                     and ad.get_durable_delegation("d-twice")["delivery_state"] == "delivered")
    _time.sleep(0.5)  # give a second copy every chance to be (wrongly) injected
    assert len(submits) == 1
    assert session["running"] is False


class TestOwnershipAcrossCompressionLineage:
    def test_row_stamped_with_tip_is_owned_by_a_session_resumed_at_an_ancestor(self):
        """Resume at the root id while the delegation was dispatched under the compressed continuation."""
        evt = {"type": "async_delegation", "origin_ui_session_id": "dead-ui-sid", "session_key": "tip_key"}
        db = MagicMock()
        db.resolve_resume_session_id.side_effect = lambda k: k
        db.get_compression_lineage.side_effect = lambda k: ["root_key", "tip_key"] if k == "root_key" else [k]
        with patch("tui_gateway.server._get_db", return_value=db):
            assert _session_owns_notification_event("tabX", {"session_key": "root_key", "_finalized": False}, evt)
            assert not _session_owns_notification_event(
                "tabY", {"session_key": "other_key", "_finalized": False}, evt)
