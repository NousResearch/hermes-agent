"""Tests for the TUI-side kanban notification poller (issue #59890).

``kanban_create`` auto-subscribes TUI/desktop sessions with
``platform="tui"`` / ``chat_id=HERMES_SESSION_KEY``, but no component ever
read those rows back: the gateway notifier skips them (no "tui" messaging
adapter) and the TUI notification poller only watched process completions.
``last_event_id`` stayed 0 forever and no notification was ever delivered.

These tests cover the delivery half that now lives in tui_gateway/server.py:
``_collect_kanban_notifications`` (cursor claim + formatting + archive-only
unsubscribe) and ``_format_kanban_event_text``.
"""

from contextlib import contextmanager
import sqlite3
from threading import Event, Thread
import time
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from hermes_cli import kanban_db as kb
from hermes_state import SessionDB
import tui_gateway.server as tui_server
from tui_gateway.server import (
    _collect_kanban_notifications,
    _format_kanban_event_text,
)

SESSION_KEY = "tui-session-key-1"


def _session(key: str = SESSION_KEY) -> dict:
    return {"session_key": key}


def _create_subscribed_task(*, chat_id: str = SESSION_KEY, platform: str = "tui"):
    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="notify tui", assignee="worker")
        kb.add_notify_sub(conn, task_id=tid, platform=platform, chat_id=chat_id)
        return tid
    finally:
        conn.close()


def _complete(tid: str, summary: str = "all done") -> None:
    conn = kb.connect()
    try:
        kb.complete_task(conn, tid, summary=summary)
    finally:
        conn.close()


def _sub_rows(tid: str) -> list:
    conn = kb.connect()
    try:
        return kb.list_notify_subs(conn, task_id=tid)
    finally:
        conn.close()


@pytest.fixture(autouse=True)
def _live_session_db(monkeypatch, tmp_path):
    state_db = SessionDB(tmp_path / "default-state.db")
    state_db.create_session(SESSION_KEY, source="desktop")
    monkeypatch.setattr(tui_server, "_get_db", lambda: state_db)


class TestCollectKanbanNotifications:
    def test_bounded_claim_rejects_delegated_child_before_cursor_mutation(
        self, monkeypatch
    ):
        tid = _create_subscribed_task()
        _complete(tid, summary="delegated child must not claim")
        pre_cursor = _sub_rows(tid)[0]["last_event_id"]

        conn = kb.connect()
        try:
            monkeypatch.setenv("HERMES_DELEGATED_CHILD_CONTEXT", "1")
            with pytest.raises(PermissionError, match="delegate_task child contexts"):
                kb.claim_unseen_events_for_sub(
                    conn,
                    task_id=tid,
                    platform="tui",
                    chat_id=SESSION_KEY,
                    lock_wait_timeout_ms=75,
                )
            assert _sub_rows(tid)[0]["last_event_id"] == pre_cursor

            monkeypatch.delenv("HERMES_DELEGATED_CHILD_CONTEXT")
            old_cursor, new_cursor, events = kb.claim_unseen_events_for_sub(
                conn,
                task_id=tid,
                platform="tui",
                chat_id=SESSION_KEY,
                lock_wait_timeout_ms=75,
            )
        finally:
            conn.close()

        assert old_cursor == pre_cursor
        assert new_cursor > old_cursor
        assert len(events) == 1
        assert events[0].kind == "completed"
        assert _sub_rows(tid)[0]["last_event_id"] == new_cursor

    def test_board_lock_expiry_releases_state_and_retries_claim_once(
        self, monkeypatch, tmp_path
    ):
        """A contended board must not pin the state ownership transaction."""
        state_path = tmp_path / "state.db"
        state_db = SessionDB(state_path)
        state_db.create_session("parent", source="desktop")
        monkeypatch.setattr(tui_server, "_get_db", lambda: state_db)

        tid = _create_subscribed_task(chat_id="parent")
        _complete(tid, summary="retry after board contention")
        pre_cursor = _sub_rows(tid)[0]["last_event_id"]

        # Keep the general connection wait small enough for a fast RED while
        # requiring the claim-specific path to fit within one bounded wait,
        # rather than multiplying it by write_txn's boundary retries.
        claim_budget_ms = 75
        monkeypatch.setenv("HERMES_KANBAN_BUSY_TIMEOUT_MS", str(claim_budget_ms))
        monkeypatch.setattr(
            tui_server,
            "_KANBAN_NOTIFY_CLAIM_BUSY_TIMEOUT_MS",
            claim_budget_ms,
            raising=False,
        )

        board_lock = kb.connect()
        board_lock.execute("BEGIN IMMEDIATE")
        claim_entered = Event()
        original_claim = kb.claim_unseen_events_for_sub

        def record_claim(*args, **kwargs):
            claim_entered.set()
            return original_claim(*args, **kwargs)

        monkeypatch.setattr(kb, "claim_unseen_events_for_sub", record_claim)
        outcome = {}

        def collect():
            started = time.monotonic()
            try:
                outcome["value"] = _collect_kanban_notifications(_session("parent"))
            except BaseException as exc:  # surfaced on the asserting thread
                outcome["error"] = exc
            finally:
                outcome["elapsed"] = time.monotonic() - started

        claim_thread = Thread(target=collect, name="bounded-notify-claim")
        claim_thread.start()
        assert claim_entered.wait(5), "collector never reached the locked board claim"

        state_write_done = Event()
        state_outcome = {}

        def write_state():
            started = time.monotonic()
            state_writer = sqlite3.connect(state_path, timeout=2, isolation_level=None)
            try:
                state_writer.execute("BEGIN IMMEDIATE")
                state_writer.execute("ROLLBACK")
            except BaseException as exc:  # surfaced on the asserting thread
                state_outcome["error"] = exc
            finally:
                state_writer.close()
                state_outcome["elapsed"] = time.monotonic() - started
                state_write_done.set()

        state_thread = Thread(target=write_state, name="state-writer-after-claim-expiry")
        state_thread.start()
        claim_thread.join(2)
        assert not claim_thread.is_alive(), "board claim exceeded its bounded wait"
        assert "error" not in outcome, outcome
        assert outcome["value"] == []
        assert outcome["elapsed"] < 0.5
        assert state_write_done.wait(1), "state.db stayed reserved after claim expiry"
        state_thread.join(1)
        assert "error" not in state_outcome, state_outcome
        assert state_outcome["elapsed"] < 0.5

        assert _sub_rows(tid)[0]["last_event_id"] == pre_cursor
        board_lock.execute("ROLLBACK")
        board_lock.close()

        retry = _collect_kanban_notifications(_session("parent"))
        assert len(retry) == 1
        assert "retry after board contention" in retry[0]
        assert _collect_kanban_notifications(_session("parent")) == []
        assert _sub_rows(tid)[0]["last_event_id"] > pre_cursor

    def test_parent_subscription_claims_only_at_live_multi_hop_compression_tip(
        self, monkeypatch, tmp_path
    ):
        """A stale parent poller must not consume its live tip's event.

        This is a current-main regression derived independently after inspecting
        the lineage contract in PR #69035; no test bytes are copied from that PR.
        """
        state_db = SessionDB(tmp_path / "state.db")
        state_db.create_session("parent", source="desktop")
        state_db.end_session("parent", "compression")
        state_db.create_session(
            "middle", source="desktop", parent_session_id="parent"
        )
        state_db.end_session("middle", "compression")
        state_db.create_session("tip", source="desktop", parent_session_id="middle")
        state_db.append_message("tip", role="assistant", content="live continuation")
        assert state_db.resolve_resume_session_id("parent") == "tip"
        monkeypatch.setattr(tui_server, "_get_db", lambda: state_db)

        tid = _create_subscribed_task(chat_id="parent")
        conn = kb.connect()
        try:
            kb.block_task(conn, tid, reason="wake the live tip")
        finally:
            conn.close()
        pre_cursor = _sub_rows(tid)[0]["last_event_id"]

        # Neither the stale parent runtime nor an unrelated lineage may claim.
        assert _collect_kanban_notifications(_session("parent")) == []
        assert _collect_kanban_notifications(_session("unrelated")) == []
        assert _sub_rows(tid)[0]["last_event_id"] == pre_cursor

        first = _collect_kanban_notifications(_session("tip"))
        assert len(first) == 1
        assert tid in first[0]
        assert "wake the live tip" in first[0]
        assert _collect_kanban_notifications(_session("tip")) == []
        assert _sub_rows(tid)[0]["last_event_id"] > pre_cursor

    def test_new_session_is_a_hard_subscription_boundary(self, monkeypatch, tmp_path):
        state_db = SessionDB(tmp_path / "state.db")
        state_db.create_session("before-new", source="desktop")
        state_db.end_session("before-new", "reset")
        state_db.create_session(
            "after-new",
            source="desktop",
            parent_session_id="before-new",
            model_config={"_reset_from": "before-new"},
        )
        state_db.append_message("after-new", role="user", content="fresh conversation")
        assert state_db.resolve_resume_session_id("before-new") == "before-new"
        monkeypatch.setattr(tui_server, "_get_db", lambda: state_db)

        tid = _create_subscribed_task(chat_id="before-new")
        _complete(tid, summary="belongs to the old conversation")
        pre_cursor = _sub_rows(tid)[0]["last_event_id"]

        assert _collect_kanban_notifications(_session("after-new")) == []
        assert _sub_rows(tid)[0]["last_event_id"] == pre_cursor

    def test_real_ambiguous_compression_children_fail_closed(
        self, monkeypatch, tmp_path
    ):
        state_db = SessionDB(tmp_path / "state.db")
        state_db.create_session("parent", source="desktop")
        state_db.end_session("parent", "compression")
        state_db.create_session("older", source="desktop", parent_session_id="parent")
        state_db.create_session("newer", source="desktop", parent_session_id="parent")
        state_db.append_message("newer", role="assistant", content="chosen by resume")
        state_db._conn.execute(
            "UPDATE sessions SET started_at = CASE id "
            "WHEN 'older' THEN 1 WHEN 'newer' THEN 2 ELSE started_at END"
        )
        state_db._conn.commit()
        assert state_db.resolve_resume_session_id("parent") == "newer"
        monkeypatch.setattr(tui_server, "_get_db", lambda: state_db)

        tid = _create_subscribed_task(chat_id="parent")
        _complete(tid, summary="ambiguous ownership must remain retryable")
        pre_cursor = _sub_rows(tid)[0]["last_event_id"]

        assert _collect_kanban_notifications(_session("newer")) == []
        assert _sub_rows(tid)[0]["last_event_id"] == pre_cursor

    def test_real_lineage_query_corruption_fails_closed(
        self, monkeypatch, tmp_path
    ):
        state_db = SessionDB(tmp_path / "state.db")
        state_db.create_session("parent", source="desktop")
        state_db.end_session("parent", "compression")
        state_db.create_session("child", source="desktop", parent_session_id="parent")
        state_db._conn.execute(
            "UPDATE sessions SET model_config = '{' WHERE id = 'child'"
        )
        state_db._conn.commit()
        # The general resume helper intentionally degrades to the input id when
        # a lineage query fails. Notification ownership must be stricter.
        assert state_db.resolve_resume_session_id("parent") == "parent"
        monkeypatch.setattr(tui_server, "_get_db", lambda: state_db)

        tid = _create_subscribed_task(chat_id="parent")
        _complete(tid, summary="corrupt lineage must remain retryable")
        pre_cursor = _sub_rows(tid)[0]["last_event_id"]

        assert _collect_kanban_notifications(_session("parent")) == []
        assert _sub_rows(tid)[0]["last_event_id"] == pre_cursor

    def test_cursor_claim_serializes_real_concurrent_compression_publication(
        self, monkeypatch, tmp_path
    ):
        """The state ownership transaction must outlive the board cursor CAS."""
        state_path = tmp_path / "state.db"
        state_db = SessionDB(state_path)
        compression_db = SessionDB(state_path)
        state_db.create_session("parent", source="desktop")
        monkeypatch.setattr(tui_server, "_get_db", lambda: state_db)

        tid = _create_subscribed_task(chat_id="parent")
        _complete(tid, summary="deliver before compression")
        pre_cursor = _sub_rows(tid)[0]["last_event_id"]

        original_claim_txn = kb._notification_claim_txn
        pause = {
            "armed": False,
            "entered": Event(),
            "release": Event(),
        }

        @contextmanager
        def pause_inside_board_claim(conn, *args, **kwargs):
            with original_claim_txn(conn, *args, **kwargs):
                if pause["armed"]:
                    pause["armed"] = False
                    pause["entered"].set()
                    assert pause["release"].wait(5), "board claim was never released"
                yield

        monkeypatch.setattr(kb, "_notification_claim_txn", pause_inside_board_claim)

        def start_thread(name, target):
            outcome = {}

            def run():
                try:
                    outcome["value"] = target()
                except BaseException as exc:  # surfaced on the asserting thread
                    outcome["error"] = exc

            thread = Thread(target=run, name=name)
            thread.start()
            return thread, outcome

        def publish(db, child_id, published):
            db.publish_compression_child(
                parent_session_id="parent",
                child_session_id=child_id,
                source="desktop",
                messages=[{"role": "assistant", "content": "live continuation"}],
                require_compression_lease=False,
            )
            published.set()

        # Production path: pause after the real board BEGIN IMMEDIATE, while
        # claim_if_notification_owner still owns the state BEGIN IMMEDIATE.
        pause["armed"] = True
        claim_thread, claim_outcome = start_thread(
            "notify-claim", lambda: _collect_kanban_notifications(_session("parent"))
        )
        assert pause["entered"].wait(5), "collector never entered the board claim"

        # Instrument the real SessionDB lock-retry boundary. This hook runs only
        # after BEGIN IMMEDIATE has returned SQLITE_BUSY/locked, so reaching it
        # proves actual state-lock contention rather than mere thread startup.
        compression_contended = Event()
        original_retry_sleep = compression_db._sleep_before_write_retry

        def record_compression_contention(*args, **kwargs):
            compression_contended.set()
            return original_retry_sleep(*args, **kwargs)

        monkeypatch.setattr(
            compression_db, "_sleep_before_write_retry", record_compression_contention
        )
        compression_published = Event()
        compression_thread, compression_outcome = start_thread(
            "compression-publication",
            lambda: publish(compression_db, "tip", compression_published),
        )
        assert compression_contended.wait(5), (
            "compression publisher never contended on state BEGIN IMMEDIATE"
        )
        assert not compression_published.is_set()
        pause["release"].set()
        claim_thread.join(5)
        compression_thread.join(5)
        assert not claim_thread.is_alive()
        assert not compression_thread.is_alive()
        assert "error" not in claim_outcome, claim_outcome
        assert "error" not in compression_outcome, compression_outcome

        delivered = claim_outcome["value"]
        assert len(delivered) == 1
        assert "deliver before compression" in delivered[0]
        claimed_cursor = _sub_rows(tid)[0]["last_event_id"]
        assert claimed_cursor > pre_cursor
        assert state_db.resolve_notification_owner_session_id("parent") == "tip"
        assert _collect_kanban_notifications(_session("parent")) == []

        # The successor owns only post-claim events and consumes each once.
        conn = kb.connect()
        try:
            with kb.write_txn(conn):
                conn.execute(
                    "UPDATE tasks SET status = 'ready' WHERE id = ?", (tid,)
                )
                kb._append_event(conn, tid, "status", {"status": "ready"})
        finally:
            conn.close()
        assert _sub_rows(tid)[0]["last_event_id"] == claimed_cursor
        successor = _collect_kanban_notifications(_session("tip"))
        assert len(successor) == 1
        assert "ready" in successor[0]
        assert _collect_kanban_notifications(_session("tip")) == []
        conn = kb.connect()
        try:
            kb.remove_notify_sub(
                conn,
                task_id=tid,
                platform="tui",
                chat_id="parent",
                thread_id="",
            )
        finally:
            conn.close()

        # Counterfactual: the old resolve-then-claim shape releases the state
        # transaction before entering the exact same board CAS. With a fresh
        # real state connection and deterministic barrier, compression now
        # publishes while the board claim is paused and the stale parent takes
        # the event. The production assertion above would therefore fail.
        broken_path = tmp_path / "broken-state.db"
        broken_state = SessionDB(broken_path)
        broken_compression = SessionDB(broken_path)
        broken_state.create_session("parent", source="desktop")
        monkeypatch.setattr(tui_server, "_get_db", lambda: broken_state)

        broken_tid = _create_subscribed_task(chat_id="parent")
        _complete(broken_tid, summary="counterfactual stale claim")

        def released_before_claim(owner_session_id, live_session_id, claim):
            resolved = broken_state.resolve_notification_owner_session_id(
                owner_session_id
            )
            return claim() if resolved == live_session_id else None

        monkeypatch.setattr(
            broken_state, "claim_if_notification_owner", released_before_claim
        )
        pause.update({"armed": True, "entered": Event(), "release": Event()})
        broken_claim_thread, broken_claim_outcome = start_thread(
            "broken-notify-claim",
            lambda: _collect_kanban_notifications(_session("parent")),
        )
        assert pause["entered"].wait(5)
        broken_published = Event()
        broken_compression_thread, broken_compression_outcome = start_thread(
            "broken-compression-publication",
            lambda: publish(
                broken_compression,
                "broken-tip",
                broken_published,
            ),
        )
        assert broken_published.wait(5), "counterfactual did not expose the race"
        pause["release"].set()
        broken_claim_thread.join(5)
        broken_compression_thread.join(5)
        assert not broken_claim_thread.is_alive()
        assert not broken_compression_thread.is_alive()
        assert "error" not in broken_claim_outcome, broken_claim_outcome
        assert "error" not in broken_compression_outcome, broken_compression_outcome
        assert len(broken_claim_outcome["value"]) == 1
        assert "counterfactual stale claim" in broken_claim_outcome["value"][0]
        assert (
            broken_state.resolve_notification_owner_session_id("parent")
            == "broken-tip"
        )

    def test_zero_sub_board_is_never_opened_writable(self):
        conn = kb.connect()
        conn.close()
        kb.create_board("second-board")

        with patch.object(kb, "connect", wraps=kb.connect) as spy_connect:
            texts = _collect_kanban_notifications(_session())

        assert texts == []
        spy_connect.assert_not_called()

    def test_done_reopen_notifies_once_per_event_until_archive(self):
        tid = _create_subscribed_task()
        _complete(tid, summary="shipped the fix")

        first = _collect_kanban_notifications(_session())

        assert len(first) == 1
        assert tid in first[0]
        assert "done" in first[0]
        assert "shipped the fix" in first[0]
        rows = _sub_rows(tid)
        assert len(rows) == 1, "done must retain the originating session"
        first_cursor = rows[0]["last_event_id"]

        # The retained subscription must not replay the completed event.
        assert _collect_kanban_notifications(_session()) == []

        conn = kb.connect()
        try:
            with kb.write_txn(conn):
                conn.execute(
                    "UPDATE tasks SET status = 'ready' WHERE id = ?", (tid,)
                )
                kb._append_event(conn, tid, "status", {"status": "ready"})
            assert kb.complete_task(conn, tid, summary="review corrections")
        finally:
            conn.close()

        reopened = _collect_kanban_notifications(_session())

        assert len(reopened) == 2
        assert "ready" in reopened[0]
        assert "review corrections" in reopened[1]
        rows = _sub_rows(tid)
        assert len(rows) == 1
        assert rows[0]["chat_id"] == SESSION_KEY
        assert rows[0]["last_event_id"] > first_cursor
        assert _collect_kanban_notifications(_session()) == []

        conn = kb.connect()
        try:
            assert kb.archive_task(conn, tid)
        finally:
            conn.close()

        # Archive is notification-terminal and removes the retained route.
        assert _collect_kanban_notifications(_session()) == []
        assert _sub_rows(tid) == []

    def test_matching_tui_sub_delivers_and_advances_cursor(self):
        tid = _create_subscribed_task()
        pre_cursor = _sub_rows(tid)[0]["last_event_id"]
        conn = kb.connect()
        try:
            kb.block_task(conn, tid, reason="waiting on review")
        finally:
            conn.close()

        with patch.object(kb, "connect", wraps=kb.connect) as spy_connect:
            first = _collect_kanban_notifications(_session())
            second = _collect_kanban_notifications(_session())

        assert len(first) == 1
        assert "blocked" in first[0]
        assert "waiting on review" in first[0]
        assert second == []
        assert spy_connect.called
        # Blocked is not a final status -> subscription stays alive so a
        # respawned task's next terminal event still reaches the user.
        rows = _sub_rows(tid)
        assert len(rows) == 1
        assert rows[0]["last_event_id"] > pre_cursor

    def test_non_tui_subscription_does_not_open_board_writable(self):
        tid = _create_subscribed_task(platform="telegram", chat_id="chat-1")
        # New subs start caught up at creation time (issue #29905); record the
        # pre-completion cursors so we can assert they were never claimed.
        pre_cursor = _sub_rows(tid)[0]["last_event_id"]
        _complete(tid)

        with patch.object(kb, "connect", wraps=kb.connect) as spy_connect:
            texts = _collect_kanban_notifications(_session())

        assert texts == []
        spy_connect.assert_not_called()
        rows = _sub_rows(tid)
        assert len(rows) == 1
        assert rows[0]["last_event_id"] == pre_cursor

    def test_other_tui_session_does_not_open_board_writable(self):
        tid = _create_subscribed_task(chat_id="some-other-session")
        pre_cursor = _sub_rows(tid)[0]["last_event_id"]
        _complete(tid)

        with patch.object(kb, "connect", wraps=kb.connect) as spy_connect:
            texts = _collect_kanban_notifications(_session())

        assert texts == []
        spy_connect.assert_not_called()
        rows = _sub_rows(tid)
        assert len(rows) == 1
        assert rows[0]["last_event_id"] == pre_cursor

    def test_probe_error_falls_back_to_writable_delivery(self, monkeypatch):
        tid = _create_subscribed_task()
        _complete(tid, summary="fallback delivery")

        def fail_probe(*args, **kwargs):
            raise OSError("probe unavailable")

        monkeypatch.setattr(kb, "list_notify_subs_readonly", fail_probe)
        with patch.object(kb, "connect", wraps=kb.connect) as spy_connect:
            texts = _collect_kanban_notifications(_session())

        assert len(texts) == 1
        assert tid in texts[0]
        spy_connect.assert_called_once()

    def test_no_session_key_is_a_noop(self):
        tid = _create_subscribed_task()
        _complete(tid)

        assert _collect_kanban_notifications({"session_key": ""}) == []
        assert _collect_kanban_notifications({"session_key": None}) == []
        assert len(_sub_rows(tid)) == 1

    def test_profile_scoped_session_reads_the_shared_board(self, tmp_path):
        """The kanban board is shared across profiles BY DESIGN (see the
        hermes_cli/kanban_db.py module docstring): ``kanban_home()`` anchors on
        ``get_default_hermes_root()``, which resolves the process env and
        ignores context-local profile overrides. A Desktop session bound to a
        non-launch profile (``session["profile_home"]``) must therefore still
        have its subscription claimed from the one shared board — the poller
        needs no per-profile home binding.
        """
        from hermes_constants import (
            reset_hermes_home_override,
            set_hermes_home_override,
        )

        tid = _create_subscribed_task()
        _complete(tid, summary="cross-profile delivery")

        other_profile_home = tmp_path / "profiles" / "reviewer"
        other_profile_home.mkdir(parents=True)
        session = {
            "session_key": SESSION_KEY,
            "profile_home": str(other_profile_home),
        }
        # Simulate the strictest case: a context-local profile override is
        # active while the poller collects (as a profile-bound RPC would set).
        token = set_hermes_home_override(str(other_profile_home))
        try:
            texts = _collect_kanban_notifications(session)
        finally:
            reset_hermes_home_override(token)

        assert len(texts) == 1
        assert tid in texts[0]
        assert "cross-profile delivery" in texts[0]
        # Completion is reversible, so the shared-board subscription remains
        # owned by this exact Desktop session until the task is archived.
        rows = _sub_rows(tid)
        assert len(rows) == 1
        assert rows[0]["chat_id"] == SESSION_KEY


class TestFormatKanbanEventText:
    SUB = {"task_id": "t_abc123"}
    TASK = SimpleNamespace(title="build the thing", assignee="worker", result=None)

    def test_silent_kinds_return_none(self):
        for kind in ("archived", "unblocked"):
            ev = SimpleNamespace(kind=kind, payload={})
            assert _format_kanban_event_text(self.SUB, self.TASK, ev, "main") is None

    def test_blocked_includes_reason(self):
        ev = SimpleNamespace(kind="blocked", payload={"reason": "needs creds"})
        text = _format_kanban_event_text(self.SUB, self.TASK, ev, "main")
        assert "t_abc123" in text
        assert "blocked" in text
        assert "needs creds" in text
        assert "[main]" in text
        assert "@worker" in text

    def test_completed_prefers_payload_summary(self):
        ev = SimpleNamespace(kind="completed", payload={"summary": "first line\nsecond"})
        text = _format_kanban_event_text(self.SUB, self.TASK, ev, "")
        assert "done" in text
        assert "first line" in text
        assert "second" not in text

    def test_timed_out_with_bad_payload_does_not_raise(self):
        ev = SimpleNamespace(kind="timed_out", payload={"limit_seconds": "not-a-number"})
        text = _format_kanban_event_text(self.SUB, self.TASK, ev, "")
        assert "timed out" in text


class TestNotificationPollerLoopKanbanWiring:
    """Drive a real TUI subscription through ``_notification_poller_loop``.

    Covers the wiring above ``_collect_kanban_notifications``: status.update
    emission, agent-turn dispatch when the session is idle, and the
    busy-session pending buffer that flushes once the session goes idle.
    """

    def _start_poller(self, session: dict, monkeypatch):
        import threading
        import tui_gateway.server as server

        emits: list = []
        submits: list = []
        monkeypatch.setattr(server, "_KANBAN_POLL_SECONDS", 0.01)
        monkeypatch.setattr(
            server, "_emit", lambda event, sid, payload=None: emits.append((event, payload))
        )
        monkeypatch.setattr(
            server,
            "_run_prompt_submit",
            lambda rid, sid, sess, text: submits.append(text),
        )
        stop = threading.Event()
        thread = threading.Thread(
            target=server._notification_poller_loop,
            args=(stop, "sid-poller-test", session),
            daemon=True,
        )
        thread.start()
        return stop, thread, emits, submits

    @staticmethod
    def _wait_for(predicate, timeout: float = 5.0) -> bool:
        import time as _time

        deadline = _time.monotonic() + timeout
        while _time.monotonic() < deadline:
            if predicate():
                return True
            _time.sleep(0.02)
        return False

    def _poller_session(self, *, running: bool = False) -> dict:
        import threading

        return {
            "session_key": SESSION_KEY,
            "history_lock": threading.Lock(),
            "running": running,
        }

    def test_idle_session_gets_status_update_and_agent_turn(self, monkeypatch):
        tid = _create_subscribed_task()
        _complete(tid, summary="poller e2e done")
        session = self._poller_session(running=False)

        stop, thread, emits, submits = self._start_poller(session, monkeypatch)
        try:
            assert self._wait_for(lambda: submits), "agent turn was never dispatched"
        finally:
            stop.set()
            thread.join(timeout=5)

        status_texts = [p["text"] for e, p in emits if e == "status.update" and p]
        assert any(tid in t for t in status_texts), status_texts
        assert any(e == "message.start" for e, _ in emits)
        assert any(tid in text for text in submits), submits
        assert session["running"] is True  # poller claimed the turn
        assert not session.get("_kanban_pending")

    def test_busy_session_buffers_then_flushes_when_idle(self, monkeypatch):
        tid = _create_subscribed_task()
        _complete(tid, summary="buffered while busy")
        session = self._poller_session(running=True)

        stop, thread, emits, submits = self._start_poller(session, monkeypatch)
        try:
            # Busy: the status line appears and the event is buffered, but no
            # agent turn is dispatched while another turn is running.
            assert self._wait_for(
                lambda: any(e == "status.update" for e, _ in emits)
                and session.get("_kanban_pending")
            )
            assert not submits

            with session["history_lock"]:
                session["running"] = False

            assert self._wait_for(lambda: submits), "pending batch never flushed"
        finally:
            stop.set()
            thread.join(timeout=5)

        assert any(tid in text for text in submits), submits
        assert session["_kanban_pending"] == []
        assert session["running"] is True
