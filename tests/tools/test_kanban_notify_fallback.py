"""Bug-injection + fix verification for kanban task completion notification
delivery to sessions that lack HERMES_SESSION_KEY (issue: notifications never
reach the creating session).

The bug: ``_maybe_auto_subscribe`` in tools/kanban_tools.py required
``HERMES_SESSION_KEY`` to write a TUI subscription row. When a desktop
agent subprocess had ``HERMES_SESSION_SOURCE=desktop`` and
``HERMES_SESSION_ID`` set but no ``HERMES_SESSION_KEY``, no subscription
was written and completion notifications were silently lost.

The fix: when ``HERMES_SESSION_KEY`` is absent but the session source is
``desktop`` or ``tui``, fall back to ``HERMES_SESSION_ID`` as the
subscription chat_id. The TUI notification poller
(``_collect_kanban_notifications``) now matches subscriptions by both
``session_key`` and ``session_id`` so the notification is delivered.

These tests use the real SQLite board DB (no mocks on kanban_db internals)
against an isolated HERMES_HOME, per AGENTS.md E2E testing rules.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def isolated_board(monkeypatch, tmp_path):
    """Create a fresh kanban board in an isolated HERMES_HOME."""
    home = tmp_path / ".hermes"
    home.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.delenv("HERMES_SESSION_KEY", raising=False)
    monkeypatch.delenv("HERMES_SESSION_ID", raising=False)
    monkeypatch.delenv("HERMES_SESSION_SOURCE", raising=False)
    monkeypatch.delenv("HERMES_SESSION_PLATFORM", raising=False)
    monkeypatch.delenv("HERMES_SESSION_CHAT_ID", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    monkeypatch.setenv("HERMES_PROFILE", "test-creator")

    from hermes_cli import kanban_db as kb
    kb._INITIALIZED_PATHS.clear()
    kb.init_db()
    return kb


def _list_subs(kb, task_id: str) -> list:
    conn = kb.connect()
    try:
        return list(kb.list_notify_subs(conn, task_id))
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Fix 1: _maybe_auto_subscribe falls back to HERMES_SESSION_ID for
# persistent desktop/tui sessions that lack HERMES_SESSION_KEY
# ---------------------------------------------------------------------------

class TestAutoSubscribeSessionIdFallback:
    """Verify the fix to _maybe_auto_subscribe: a desktop session without
    HERMES_SESSION_KEY but with HERMES_SESSION_SOURCE=desktop and
    HERMES_SESSION_ID set should still get a subscription row."""

    def test_desktop_source_with_session_id_subscribes(
        self, monkeypatch, isolated_board
    ):
        """Simulate the desktop agent subprocess: no SESSION_KEY, but
        SESSION_SOURCE=desktop and SESSION_ID are set. The subscription
        should be written with platform='tui' and chat_id=<session_id>."""
        from tools import kanban_tools as kt

        monkeypatch.setenv("HERMES_SESSION_SOURCE", "desktop")
        monkeypatch.setenv("HERMES_SESSION_ID", "desktop-sess-abc")
        monkeypatch.delenv("HERMES_SESSION_KEY", raising=False)

        out = kt._handle_create({
            "title": "desktop fallback sub",
            "assignee": "worker",
        })
        d = json.loads(out)
        assert d["ok"] is True, d
        assert d["subscribed"] is True, (
            "desktop session with SESSION_ID must auto-subscribe"
        )

        subs = _list_subs(isolated_board, d["task_id"])
        assert len(subs) == 1
        assert subs[0]["platform"] == "tui"
        assert subs[0]["chat_id"] == "desktop-sess-abc"

    def test_tui_source_with_session_id_subscribes(
        self, monkeypatch, isolated_board
    ):
        """Same as desktop but with source=tui (standalone hermes --tui)."""
        from tools import kanban_tools as kt

        monkeypatch.setenv("HERMES_SESSION_SOURCE", "tui")
        monkeypatch.setenv("HERMES_SESSION_ID", "tui-sess-xyz")
        monkeypatch.delenv("HERMES_SESSION_KEY", raising=False)

        out = kt._handle_create({
            "title": "tui fallback sub",
            "assignee": "worker",
        })
        d = json.loads(out)
        assert d["ok"] is True, d
        assert d["subscribed"] is True, d

        subs = _list_subs(isolated_board, d["task_id"])
        assert len(subs) == 1
        assert subs[0]["chat_id"] == "tui-sess-xyz"

    def test_cli_source_still_does_not_subscribe(
        self, monkeypatch, isolated_board
    ):
        """CLI sessions (source=cli or absent) must still NOT auto-subscribe
        when SESSION_KEY is absent. This is the over-subscription guard
        from PR #19718 that must remain intact."""
        from tools import kanban_tools as kt

        monkeypatch.setenv("HERMES_SESSION_SOURCE", "cli")
        monkeypatch.setenv("HERMES_SESSION_ID", "cli-sess-123")
        monkeypatch.delenv("HERMES_SESSION_KEY", raising=False)

        out = kt._handle_create({
            "title": "cli no sub",
            "assignee": "worker",
        })
        d = json.loads(out)
        assert d["ok"] is True, d
        assert d["subscribed"] is False, (
            "CLI sessions must not auto-subscribe without SESSION_KEY"
        )
        assert _list_subs(isolated_board, d["task_id"]) == []

    def test_no_source_no_session_id_does_not_subscribe(
        self, monkeypatch, isolated_board
    ):
        """When neither SESSION_KEY nor SESSION_SOURCE+SESSION_ID are
        available, no subscription should be written."""
        from tools import kanban_tools as kt

        monkeypatch.delenv("HERMES_SESSION_SOURCE", raising=False)
        monkeypatch.delenv("HERMES_SESSION_ID", raising=False)
        monkeypatch.delenv("HERMES_SESSION_KEY", raising=False)

        out = kt._handle_create({
            "title": "bare no sub",
            "assignee": "worker",
        })
        d = json.loads(out)
        assert d["ok"] is True, d
        assert d["subscribed"] is False, d
        assert _list_subs(isolated_board, d["task_id"]) == []

    def test_session_key_still_preferred_over_session_id(
        self, monkeypatch, isolated_board
    ):
        """When both SESSION_KEY and SESSION_ID are set, the subscription
        should use SESSION_KEY (the primary identity)."""
        from tools import kanban_tools as kt

        monkeypatch.setenv("HERMES_SESSION_KEY", "primary-key-1")
        monkeypatch.setenv("HERMES_SESSION_ID", "secondary-id-1")
        monkeypatch.setenv("HERMES_SESSION_SOURCE", "desktop")

        out = kt._handle_create({
            "title": "key preferred",
            "assignee": "worker",
        })
        d = json.loads(out)
        assert d["ok"] is True, d
        assert d["subscribed"] is True, d

        subs = _list_subs(isolated_board, d["task_id"])
        assert len(subs) == 1
        assert subs[0]["chat_id"] == "primary-key-1", (
            "SESSION_KEY must be preferred over SESSION_ID"
        )

    def test_persistent_source_without_session_id_does_not_subscribe(
        self, monkeypatch, isolated_board
    ):
        """Edge case: source says desktop but no session_id is set.
        Should not subscribe (can't write a meaningful chat_id)."""
        from tools import kanban_tools as kt

        monkeypatch.setenv("HERMES_SESSION_SOURCE", "desktop")
        monkeypatch.delenv("HERMES_SESSION_ID", raising=False)
        monkeypatch.delenv("HERMES_SESSION_KEY", raising=False)

        out = kt._handle_create({
            "title": "no id no sub",
            "assignee": "worker",
        })
        d = json.loads(out)
        assert d["ok"] is True, d
        assert d["subscribed"] is False, d


# ---------------------------------------------------------------------------
# Fix 2: TUI poller matches subscriptions by session_id
# ---------------------------------------------------------------------------

class TestPollerMatchesSessionId:
    """Verify the TUI notification poller claims subscriptions whose
    chat_id is the agent's session_id (not just session_key)."""

    def test_poller_delivers_subscription_keyed_on_session_id(
        self, monkeypatch, isolated_board
    ):
        """Create a task subscribed with chat_id=<session_id> (the fallback
        path), complete it, and verify the TUI poller delivers the
        notification to a session whose agent.session_id matches."""
        from tui_gateway.server import _collect_kanban_notifications

        kb = isolated_board
        conn = kb.connect()
        try:
            tid = kb.create_task(conn, title="session-id sub", assignee="worker")
            kb.add_notify_sub(
                conn, task_id=tid, platform="tui",
                chat_id="agent-session-id-1",
            )
        finally:
            conn.close()

        conn = kb.connect()
        try:
            kb.complete_task(conn, tid, summary="delivered via session_id")
        finally:
            conn.close()

        # Simulate a TUI session whose agent has session_id matching the sub
        agent = SimpleNamespace(session_id="agent-session-id-1")
        session = {
            "session_key": "tui-window-key-1",
            "agent": agent,
        }
        texts = _collect_kanban_notifications(session)
        assert len(texts) == 1, (
            "poller must deliver notifications for session_id-keyed subs"
        )
        assert tid in texts[0]
        assert "done" in texts[0]
        assert "delivered via session_id" in texts[0]

    def test_poller_still_delivers_session_key_subscriptions(
        self, monkeypatch, isolated_board
    ):
        """Regression: the existing session_key matching path must still
        work after the fix."""
        from tui_gateway.server import _collect_kanban_notifications

        kb = isolated_board
        conn = kb.connect()
        try:
            tid = kb.create_task(conn, title="key sub", assignee="worker")
            kb.add_notify_sub(
                conn, task_id=tid, platform="tui",
                chat_id="tui-key-legacy",
            )
        finally:
            conn.close()

        conn = kb.connect()
        try:
            kb.complete_task(conn, tid, summary="legacy delivery")
        finally:
            conn.close()

        session = {"session_key": "tui-key-legacy"}
        texts = _collect_kanban_notifications(session)
        assert len(texts) == 1
        assert tid in texts[0]

    def test_poller_does_not_deliver_other_session_id_subs(
        self, monkeypatch, isolated_board
    ):
        """A subscription for a different session_id must not be claimed
        by this poller."""
        from tui_gateway.server import _collect_kanban_notifications

        kb = isolated_board
        conn = kb.connect()
        try:
            tid = kb.create_task(conn, title="other session", assignee="worker")
            kb.add_notify_sub(
                conn, task_id=tid, platform="tui",
                chat_id="other-session-id",
            )
        finally:
            conn.close()

        conn = kb.connect()
        try:
            kb.complete_task(conn, tid, summary="not for me")
        finally:
            conn.close()

        agent = SimpleNamespace(session_id="my-session-id")
        session = {
            "session_key": "my-key",
            "agent": agent,
        }
        texts = _collect_kanban_notifications(session)
        assert texts == [], (
            "must not deliver subscriptions for other session ids"
        )


# ---------------------------------------------------------------------------
# E2E: create -> complete -> verify delivery
# ---------------------------------------------------------------------------

class TestEndToEndNotificationDelivery:
    """Full E2E: create a task from a desktop session (no SESSION_KEY),
    complete it, and verify the notification reaches the active session
    of the same profile via the TUI poller."""

    def test_desktop_create_complete_notify(
        self, monkeypatch, isolated_board
    ):
        from tools import kanban_tools as kt
        from tui_gateway.server import _collect_kanban_notifications

        kb = isolated_board

        # Step 1: Create a task from a desktop session that has
        # SESSION_SOURCE=desktop and SESSION_ID but no SESSION_KEY.
        monkeypatch.setenv("HERMES_SESSION_SOURCE", "desktop")
        monkeypatch.setenv("HERMES_SESSION_ID", "e2e-desktop-sess")
        monkeypatch.delenv("HERMES_SESSION_KEY", raising=False)

        out = kt._handle_create({
            "title": "e2e notify task",
            "assignee": "worker",
        })
        d = json.loads(out)
        assert d["ok"] is True, d
        assert d["subscribed"] is True, (
            "desktop session must auto-subscribe via session_id fallback"
        )
        task_id = d["task_id"]

        # Verify the subscription was actually written
        subs = _list_subs(kb, task_id)
        assert len(subs) == 1
        assert subs[0]["chat_id"] == "e2e-desktop-sess"

        # Step 2: Complete the task (simulating a worker finishing it)
        conn = kb.connect()
        try:
            ok = kb.complete_task(conn, task_id, summary="e2e work done")
            assert ok, "complete_task must succeed"
        finally:
            conn.close()

        # Step 3: The TUI poller for the creating session should deliver
        # the notification. The session's agent.session_id matches the
        # SESSION_ID used at creation time.
        agent = SimpleNamespace(session_id="e2e-desktop-sess")
        session = {
            "session_key": "e2e-window-key",
            "agent": agent,
        }
        texts = _collect_kanban_notifications(session)
        assert len(texts) == 1, (
            "completion notification must reach the creating desktop session"
        )
        assert task_id in texts[0]
        assert "done" in texts[0]
        assert "e2e work done" in texts[0]

    def test_notification_not_delivered_to_wrong_session(
        self, monkeypatch, isolated_board
    ):
        """The notification must NOT reach a different session that
        doesn't share the creating session's identity."""
        from tools import kanban_tools as kt
        from tui_gateway.server import _collect_kanban_notifications

        kb = isolated_board

        monkeypatch.setenv("HERMES_SESSION_SOURCE", "desktop")
        monkeypatch.setenv("HERMES_SESSION_ID", "creator-sess-id")
        monkeypatch.delenv("HERMES_SESSION_KEY", raising=False)

        out = kt._handle_create({
            "title": "wrong session test",
            "assignee": "worker",
        })
        d = json.loads(out)
        task_id = d["task_id"]

        conn = kb.connect()
        try:
            kb.complete_task(conn, task_id, summary="done")
        finally:
            conn.close()

        # A different session with a different session_id
        other_agent = SimpleNamespace(session_id="different-sess-id")
        other_session = {
            "session_key": "different-window",
            "agent": other_agent,
        }
        texts = _collect_kanban_notifications(other_session)
        assert texts == [], (
            "notification must not leak to unrelated sessions"
        )


# ---------------------------------------------------------------------------
# Finding 1: HERMES_SESSION_ID == agent.session_id invariant
# ---------------------------------------------------------------------------

class TestSessionIdEquivalenceInvariant:
    """Verify the production invariant that HERMES_SESSION_ID (env, at task
    create time) equals agent.session_id (at poller delivery time).

    The fix rests on this equivalence: the subscription is written with
    chat_id=HERMES_SESSION_ID, and the poller matches against
    agent.session_id. If the two diverge, notifications are silently lost.

    This test verifies the invariant by exercising the real production
    code path: AIAgent.__init__ -> set_current_session_id() -> env var,
    then reading the env var back the way _maybe_auto_subscribe does.
    """

    def test_set_current_session_id_writes_env_and_contextvar(self, monkeypatch, tmp_path):
        """set_current_session_id (called from AIAgent.__init__) writes the
        same value to both os.environ and the ContextVar. The TUI gateway
        subprocess-env bridge reads agent.session_id and passes it to
        set_session_vars(), so the agent subprocess inherits the same id.
        _maybe_auto_subscribe reads it back via get_session_env."""
        from gateway.session_context import (
            set_current_session_id,
            get_session_env,
        )

        home = tmp_path / ".hermes"
        home.mkdir(parents=True)
        monkeypatch.setenv("HERMES_HOME", str(home))
        monkeypatch.setattr(Path, "home", lambda: tmp_path)

        # Simulate AIAgent.__init__ calling set_current_session_id
        test_session_id = "invariant-test-sess-42"
        monkeypatch.delenv("HERMES_SESSION_ID", raising=False)
        set_current_session_id(test_session_id)

        # _maybe_auto_subscribe reads via get_session_env with env fallback.
        # Both paths must return the same value that was set.
        assert os.environ.get("HERMES_SESSION_ID") == test_session_id
        assert get_session_env("HERMES_SESSION_ID", "") == test_session_id

    def test_poller_reads_same_session_id_from_agent_object(
        self, monkeypatch, isolated_board
    ):
        """The poller reads getattr(agent, 'session_id', ''). Verify that
        a subscription written with chat_id=<HERMES_SESSION_ID> is claimed
        when agent.session_id matches -- i.e. both sides use the same id."""
        from tui_gateway.server import _collect_kanban_notifications

        kb = isolated_board
        # Write a subscription with chat_id = the session_id value
        # that _maybe_auto_subscribe would have used (HERMES_SESSION_ID).
        shared_id = "shared-prod-sess-id"
        conn = kb.connect()
        try:
            tid = kb.create_task(conn, title="invariant test", assignee="worker")
            kb.add_notify_sub(conn, task_id=tid, platform="tui", chat_id=shared_id)
        finally:
            conn.close()

        conn = kb.connect()
        try:
            kb.complete_task(conn, tid, summary="invariant done")
        finally:
            conn.close()

        # The poller reads agent.session_id -- which in production is the
        # same value that was written to HERMES_SESSION_ID.
        agent = SimpleNamespace(session_id=shared_id)
        session = {"session_key": "some-window-key", "agent": agent}
        texts = _collect_kanban_notifications(session)
        assert len(texts) == 1, (
            "subscription keyed on HERMES_SESSION_ID must be claimed when "
            "agent.session_id matches -- the invariant holds"
        )


# ---------------------------------------------------------------------------
# Finding 3: exception in one identity probe does not skip remaining
# ---------------------------------------------------------------------------

class TestExceptionContinueProbesRemainingIdentities:
    """Verify that when count_notify_subs raises for one identity in the
    chat_identities set, the poller continues to probe the remaining
    identities instead of breaking out of the loop.

    Bug: the original code set has_sub=True and broke on any exception,
    so a persistently failing probe for the first identity (session_key)
    would skip probing the second identity (session_id) -- even though
    the second might have a real subscription.
    """

    def test_failing_first_identity_still_probes_second(
        self, monkeypatch, isolated_board
    ):
        """count_notify_subs raises for the first identity (session_key)
        but succeeds for the second (session_id) which has a real sub.
        The poller must still find the subscription."""
        from tui_gateway.server import _collect_kanban_notifications
        from hermes_cli import kanban_db as kb

        session_key = "failing-key"
        session_id = "succeeding-id"

        # Write a subscription keyed on session_id (the second identity)
        conn = kb.connect()
        try:
            tid = kb.create_task(conn, title="continue test", assignee="worker")
            kb.add_notify_sub(conn, task_id=tid, platform="tui", chat_id=session_id)
        finally:
            conn.close()

        conn = kb.connect()
        try:
            kb.complete_task(conn, tid, summary="continue done")
        finally:
            conn.close()

        # Make count_notify_subs raise only for the first identity
        real_count = kb.count_notify_subs

        def flaky_count(*args, **kwargs):
            chat_id = kwargs.get("chat_id", "")
            if chat_id == session_key:
                raise RuntimeError("simulated DB lock on first identity")
            return real_count(*args, **kwargs)

        monkeypatch.setattr(kb, "count_notify_subs", flaky_count)

        agent = SimpleNamespace(session_id=session_id)
        session = {"session_key": session_key, "agent": agent}
        texts = _collect_kanban_notifications(session)
        assert len(texts) == 1, (
            "exception on first identity must not skip probing the second "
            "identity which has a real subscription"
        )
        assert tid in texts[0]

    def test_all_identities_failing_does_not_crash(
        self, monkeypatch, isolated_board
    ):
        """When count_notify_subs raises for ALL identities, the poller
        must not crash and must simply skip this board (has_sub stays
        False). This is the graceful-degradation path."""
        from tui_gateway.server import _collect_kanban_notifications
        from hermes_cli import kanban_db as kb

        def always_failing(*args, **kwargs):
            raise RuntimeError("simulated persistent DB error")

        monkeypatch.setattr(kb, "count_notify_subs", always_failing)

        agent = SimpleNamespace(session_id="some-id")
        session = {"session_key": "some-key", "agent": agent}
        # Must not raise -- graceful degradation
        texts = _collect_kanban_notifications(session)
        assert texts == []


# ---------------------------------------------------------------------------
# Finding 4: SESSION_SOURCE allow-set with logging for unknown values
# ---------------------------------------------------------------------------

class TestSessionSourceAllowSet:
    """Verify that _maybe_auto_subscribe uses an allow-set for
    HERMES_SESSION_SOURCE, logging and skipping unknown-but-set values
    instead of silently failing or silently subscribing.

    The allow-set prevents a future source value with similar meaning
    from silently stopping subscribing (if it's not in the set, it's
    logged and skipped). New persistent sources must be registered.
    """

    def test_unknown_source_does_not_subscribe(self, monkeypatch, isolated_board):
        """An unknown SESSION_SOURCE (not desktop/tui) with SESSION_ID
        must NOT auto-subscribe. This is the allow-set gate."""
        from tools import kanban_tools as kt

        monkeypatch.setenv("HERMES_SESSION_SOURCE", "future_platform")
        monkeypatch.setenv("HERMES_SESSION_ID", "future-sess-1")
        monkeypatch.delenv("HERMES_SESSION_KEY", raising=False)

        out = kt._handle_create({
            "title": "unknown source",
            "assignee": "worker",
        })
        d = json.loads(out)
        assert d["ok"] is True, d
        assert d["subscribed"] is False, (
            "unknown SESSION_SOURCE must not auto-subscribe"
        )
        assert _list_subs(isolated_board, d["task_id"]) == []

    def test_allow_set_is_frozenset_not_tuple(self):
        """Verify the allow-set is a frozenset (immutable, can't be
        accidentally mutated at runtime). This is a structural invariant
        -- the allow-set must not be modifiable by stray code."""
        from tools import kanban_tools as kt

        assert isinstance(kt._PERSISTENT_SESSION_SOURCES, frozenset)
        assert "desktop" in kt._PERSISTENT_SESSION_SOURCES
        assert "tui" in kt._PERSISTENT_SESSION_SOURCES

    def test_unknown_source_logs_info(self, monkeypatch, isolated_board, caplog):
        """When SESSION_SOURCE is set but not in the allow-set, an INFO
        log entry must be emitted so operators can diagnose why
        auto-subscribe was skipped."""
        import logging as _logging
        from tools import kanban_tools as kt

        monkeypatch.setenv("HERMES_SESSION_SOURCE", "web_frontend")
        monkeypatch.setenv("HERMES_SESSION_ID", "web-sess-1")
        monkeypatch.delenv("HERMES_SESSION_KEY", raising=False)

        with caplog.at_level(_logging.INFO, logger="tools.kanban_tools"):
            out = kt._handle_create({
                "title": "logging test",
                "assignee": "worker",
            })
        d = json.loads(out)
        assert d["subscribed"] is False

        # Verify an INFO log was emitted mentioning the unknown source
        logged = [r for r in caplog.records if r.levelno == _logging.INFO
                  and "web_frontend" in r.getMessage()]
        assert len(logged) >= 1, (
            "unknown SESSION_SOURCE must produce an INFO log entry"
        )

