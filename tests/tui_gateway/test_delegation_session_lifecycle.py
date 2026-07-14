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


    def test_origin_ui_blocks_session_key_fallback(self):
        """When origin UI id is stamped, only that tab owns — not session_key."""
        evt = {
            "type": "async_delegation",
            "origin_ui_session_id": "tab_origin",
            "session_key": "sess_key_1",
        }
        # Same durable key, different tab → NOT owner
        assert _session_owns_notification_event("tabX", self._session("sess_key_1"), evt) is False
        # Exact origin tab → owner
        assert _session_owns_notification_event("tab_origin", self._session("other"), evt) is True

    def test_profile_mismatch_never_owns(self):
        from tui_gateway.server import _session_owns_notification_event
        evt = {
            "type": "async_delegation",
            "origin_ui_session_id": "tab1",
            "session_key": "sess_key_1",
            "origin_profile": "highbeam",
            "origin_hermes_home": "/tmp/profiles/highbeam",
        }
        sess = self._session()
        sess["profile"] = "default"
        sess["profile_home"] = "/tmp/profiles/default"
        assert _session_owns_notification_event("tab1", sess, evt) is False

    def test_profile_match_with_origin_owns(self):
        evt = {
            "type": "async_delegation",
            "origin_ui_session_id": "tab1",
            "session_key": "sess_key_1",
            "origin_profile": "highbeam",
            "origin_hermes_home": "/tmp/profiles/highbeam",
        }
        sess = self._session()
        sess["profile"] = "highbeam"
        sess["profile_home"] = "/tmp/profiles/highbeam"
        assert _session_owns_notification_event("tab1", sess, evt) is True


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
