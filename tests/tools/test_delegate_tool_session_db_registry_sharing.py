"""Regression coverage for the WAL-generation swarm investigation (2026-09-13).

A prior incident (docs/incidents/deleted-wal-generation-error/) found that the CLI's
long-lived session opened a bare, unregistered ``SessionDB()`` in
``HermesCLI._init_session_store``. When that session later delegated work to a
subagent, ``delegate_tool.py``'s child-session lookup independently ``acquire()``-d
the same physical file through ``hermes_state_registry`` -- minting a second,
genuinely independent writer connection on one inode, since the parent's bare handle
was invisible to the registry's dedup bookkeeping.

Commit 876e444e4e (2026-09-12, landed for an unrelated startup-freeze fix) changed
``_init_session_store`` to route through ``hermes_state_registry.acquire()`` instead
of a bare ``SessionDB()``, which closes this gap as a side effect. This test proves
the fix end-to-end for the ACTUAL CLI code path (not just the registry's own
acquire() contract, which was already correct before that commit) by calling
``HermesCLI._init_session_store`` itself, then handing that instance to
``delegate_tool._open_child_session_db`` -- the exact two-hop shape the incident
described.

Verified RED on the pre-fix commit (876e444e4e~1): both assertions below fail
there because ``cli._session_db`` is a bare ``SessionDB()`` object with a
connection the registry has never seen, so the child's ``acquire()`` mints a
second, different connection.
"""

from types import SimpleNamespace

from cli import HermesCLI
from tools.delegate_tool import _open_child_session_db


def test_delegate_tool_child_shares_one_connection_with_the_cli_session(tmp_path, monkeypatch):
    """End-to-end: HermesCLI._init_session_store's real handle, handed to a real
    delegate_tool child lookup, must resolve to the SAME physical connection --
    not a second independent writer on the same state.db file."""
    import hermes_state_registry

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    cli = SimpleNamespace()
    try:
        HermesCLI._init_session_store(cli)
        assert cli._session_db is not None and not cli._session_db_unavailable

        child_db = _open_child_session_db(cli)

        assert child_db is not None
        assert child_db._conn is cli._session_db._conn, (
            "delegate_tool child opened a SECOND independent connection to the "
            "same state.db as the CLI's own session handle -- this is the exact "
            "double-writer shape from the 2026-09-13 DeletedWalGenerationError "
            "incident swarm (docs/incidents/deleted-wal-generation-error/)"
        )
    finally:
        hermes_state_registry.close_all()
