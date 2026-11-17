"""An explicitly cleared session context is not another chat's session key.

``clear_session_vars`` / ``set_session_vars(session_key="")`` means "no session in THIS
context" (an off-turn RPC, a sessionless desktop request). ``get_session_env`` already answers
``""`` for that and falls back to ``os.environ`` only when the var was never bound; the extra
``or os.environ.get("HERMES_SESSION_KEY")`` in ``_resolve_notify_target`` then borrowed the
process-global key of whichever session the process built last, so a card filed from such a
context was subscribed to that chat and its finish landed in a chat that did not ask for it.
"""

from __future__ import annotations

import contextvars

import pytest


@pytest.fixture
def board(monkeypatch, tmp_path):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_PROFILE", "test-profile")
    # The process-global value a borrowed fallback would reach.
    monkeypatch.setenv("HERMES_SESSION_KEY", "process-global-key")
    for name in ("HERMES_SESSION_PLATFORM", "HERMES_SESSION_CHAT_ID", "HERMES_SESSION_THREAD_ID",
                 "HERMES_SESSION_USER_ID", "HERMES_SESSION_ID"):
        monkeypatch.delenv(name, raising=False)
    from hermes_cli import kanban_db as kb

    kb._INITIALIZED_PATHS.clear()
    kb.init_db()
    return home


def _in_cleared_context(fn):
    """Run *fn* where the session vars are bound and then cleared, as after an off-turn RPC."""
    from gateway.session_context import clear_session_vars, set_session_vars

    def run():
        clear_session_vars(set_session_vars(session_key="live-turn-key"))
        return fn()

    return contextvars.Context().run(run)


def test_cleared_context_has_no_notify_target(board):
    from tools import kanban_tools as kt

    assert _in_cleared_context(kt._resolve_notify_target) is None


def test_card_filed_from_a_cleared_context_subscribes_nobody(board):
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc
    from hermes_cli import kanban_db_notify as kbn
    from tools import kanban_tools as kt

    conn = kbc.connect()
    try:
        tid = kb.create_task(conn, title="filed off-turn", assignee="default")
        assert _in_cleared_context(lambda: kt._maybe_auto_subscribe(conn, tid)) is False
        assert kbn.list_notify_subs(conn, tid) == []
    finally:
        conn.close()


def test_a_context_that_never_bound_a_session_still_inherits_the_process_key(board):
    """Control: a subprocess spawned by a TUI turn binds no session vars and inherits the key in
    its environment; that is the inheritance the fallback exists for, and it still works."""
    from tools import kanban_tools as kt

    target = contextvars.Context().run(kt._resolve_notify_target)
    assert target is not None
    assert (target["platform"], target["chat_id"]) == ("tui", "process-global-key")
