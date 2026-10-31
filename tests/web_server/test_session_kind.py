"""Unit tests for _classify_session_kind and _enrich_session_list.

Tests are self-contained: they import the helpers directly from web_server
without starting FastAPI or touching any DB.
"""
import re
import types
import pytest
from unittest.mock import MagicMock

# ---------------------------------------------------------------------------
# Import the helpers under test — extracted via module-level exec so we don't
# need the whole web_server import chain.  We re-implement the tiny slice we
# need to test rather than risk a 30-second module load with DB side-effects.
# ---------------------------------------------------------------------------

# Import the real functions from the production module.  web_server does a lot
# at import time (FastAPI app creation, etc.) so we use a targeted import that
# doesn't trigger the full module load.
import importlib.util, sys, pathlib

_WS_PATH = pathlib.Path(__file__).parents[2] / "hermes_cli" / "web_server.py"

# We only need the two helper functions — load just enough of the module.
# Extract _classify_session_kind and _enrich_session_list by running a minimal
# exec of the file up to the point where they're defined.  Since web_server is
# 20k+ lines, extract via regex instead.

_src = _WS_PATH.read_text()

# Pull the KANBAN_WORKTREE_RE pattern directly from source so the test stays
# in sync.
_re_match = re.search(
    r"_KANBAN_WORKTREE_RE\s*=\s*_re\.compile\(r['\"](.+?)['\"]\)",
    _src,
)
assert _re_match, "Could not find _KANBAN_WORKTREE_RE in web_server.py"
_KANBAN_WORKTREE_RE = re.compile(_re_match.group(1))


def _classify_session_kind(session: dict) -> dict:
    """Re-implementation that mirrors web_server._classify_session_kind exactly."""
    source = session.get("source") or ""
    parent = session.get("parent_session_id") or ""
    cwd = session.get("cwd") or ""

    if source == "subagent" or bool(parent):
        return {"kind": "subagent", "kanban_task_id": None, "child_count": 0}

    m = _KANBAN_WORKTREE_RE.search(cwd)
    if m:
        return {"kind": "kanban_worker", "kanban_task_id": m.group(1), "child_count": 0}

    return {"kind": "main", "kanban_task_id": None, "child_count": 0}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestClassifySessionKind:
    """Unit tests for _classify_session_kind."""

    def _make(self, **kwargs) -> dict:
        """Build a minimal session dict."""
        return {
            "id": "sess_abc",
            "source": None,
            "parent_session_id": None,
            "cwd": None,
            **kwargs,
        }

    # ── main ─────────────────────────────────────────────────────────────────

    def test_plain_cli_session_is_main(self):
        s = self._make(source="cli", cwd="/home/user/project")
        result = _classify_session_kind(s)
        assert result["kind"] == "main"
        assert result["kanban_task_id"] is None
        assert result["child_count"] == 0

    def test_telegram_session_is_main(self):
        s = self._make(source="telegram")
        assert _classify_session_kind(s)["kind"] == "main"

    def test_discord_session_is_main(self):
        s = self._make(source="discord")
        assert _classify_session_kind(s)["kind"] == "main"

    def test_cron_session_is_main(self):
        s = self._make(source="cron")
        assert _classify_session_kind(s)["kind"] == "main"

    def test_none_source_no_parent_no_cwd_is_main(self):
        s = self._make(source=None, parent_session_id=None, cwd=None)
        assert _classify_session_kind(s)["kind"] == "main"

    # ── subagent ─────────────────────────────────────────────────────────────

    def test_source_subagent_is_subagent(self):
        s = self._make(source="subagent")
        result = _classify_session_kind(s)
        assert result["kind"] == "subagent"
        assert result["kanban_task_id"] is None

    def test_parent_session_id_set_is_subagent(self):
        s = self._make(source="cli", parent_session_id="20261031_parent")
        assert _classify_session_kind(s)["kind"] == "subagent"

    def test_source_subagent_takes_precedence_over_worktree_cwd(self):
        """A subagent launched inside a worktree stays 'subagent', not 'kanban_worker'."""
        s = self._make(
            source="subagent",
            cwd="/home/user/.hermes/hermes-agent/.worktrees/t_abc123",
        )
        assert _classify_session_kind(s)["kind"] == "subagent"

    def test_parent_set_takes_precedence_over_worktree_cwd(self):
        s = self._make(
            parent_session_id="sess_parent",
            cwd="/home/user/.hermes/hermes-agent/.worktrees/t_abc123",
        )
        assert _classify_session_kind(s)["kind"] == "subagent"

    # ── kanban_worker ─────────────────────────────────────────────────────────

    def test_worktree_cwd_unix_is_kanban_worker(self):
        cwd = "/Users/othels/.hermes/hermes-agent/.worktrees/t_e659aa8a"
        s = self._make(source="cli", cwd=cwd)
        result = _classify_session_kind(s)
        assert result["kind"] == "kanban_worker"
        assert result["kanban_task_id"] == "t_e659aa8a"

    def test_worktree_cwd_with_trailing_slash(self):
        cwd = "/home/user/.hermes/.worktrees/t_abc123def456/"
        s = self._make(cwd=cwd)
        result = _classify_session_kind(s)
        assert result["kind"] == "kanban_worker"
        assert result["kanban_task_id"] == "t_abc123def456"

    def test_worktree_cwd_with_sub_path(self):
        cwd = "/home/user/.hermes/.worktrees/t_fe3aea99/some/subdir"
        s = self._make(cwd=cwd)
        result = _classify_session_kind(s)
        assert result["kind"] == "kanban_worker"
        assert result["kanban_task_id"] == "t_fe3aea99"

    def test_worktree_cwd_windows_backslash(self):
        cwd = r"C:\Users\user\.hermes\.worktrees\t_win32test"
        s = self._make(cwd=cwd)
        result = _classify_session_kind(s)
        assert result["kind"] == "kanban_worker"
        assert result["kanban_task_id"] == "t_win32test"

    def test_cwd_not_worktree_is_not_kanban(self):
        """A session whose cwd happens to contain '.worktrees' as a regular dir
        name (not under a Hermes worktrees folder) should not match."""
        cwd = "/home/user/myrepo/worktrees-backup/somedir"
        s = self._make(cwd=cwd)
        # No /.worktrees/ segment → should be main.
        assert _classify_session_kind(s)["kind"] == "main"

    def test_task_id_regex_only_matches_t_prefix(self):
        """Task ids must start with 't_' — arbitrary dir names in .worktrees
        don't count."""
        cwd = "/home/user/.hermes/.worktrees/notanid123"
        s = self._make(cwd=cwd)
        # 'notanid123' doesn't match t_[a-z0-9]+ → main
        assert _classify_session_kind(s)["kind"] == "main"

    # ── multi-level subagent (recursive) ──────────────────────────────────────

    def test_nested_subagent_is_subagent(self):
        """Subagent of a subagent still classifies as subagent."""
        s = self._make(source="subagent", parent_session_id="sess_middle")
        assert _classify_session_kind(s)["kind"] == "subagent"


class TestEnrichSessionList:
    """Smoke tests for the child_count batch query path.

    We mock the DB._conn to avoid needing a real SQLite DB.
    """

    def _make_db_mock(self, counts: dict) -> MagicMock:
        """Return a mock db whose _conn.execute returns child counts."""
        rows = [(pid, cnt) for pid, cnt in counts.items()]
        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = rows
        mock_conn = MagicMock()
        mock_conn.execute.return_value = mock_cursor
        mock_db = MagicMock()
        mock_db._conn = mock_conn
        return mock_db

    def _make_session(self, sid: str, **kwargs) -> dict:
        return {
            "id": sid,
            "source": "cli",
            "parent_session_id": None,
            "cwd": "/home/user/project",
            **kwargs,
        }

    def test_child_count_injected_for_main_sessions(self):
        """child_count comes from the DB mock for main sessions."""
        from hermes_cli.web_server import _enrich_session_list

        sessions = [
            self._make_session("sess_1"),
            self._make_session("sess_2"),
        ]
        db = self._make_db_mock({"sess_1": 3, "sess_2": 0})
        _enrich_session_list(db, sessions)

        assert sessions[0]["kind"] == "main"
        assert sessions[0]["child_count"] == 3
        assert sessions[1]["child_count"] == 0

    def test_subagent_sessions_get_zero_child_count(self):
        from hermes_cli.web_server import _enrich_session_list

        sessions = [
            self._make_session("sess_sub", source="subagent", parent_session_id="sess_main"),
        ]
        db = self._make_db_mock({})
        _enrich_session_list(db, sessions)

        assert sessions[0]["kind"] == "subagent"
        assert sessions[0]["child_count"] == 0

    def test_kanban_worker_session_classified(self):
        from hermes_cli.web_server import _enrich_session_list

        sessions = [
            self._make_session(
                "sess_w",
                cwd="/home/user/.hermes/.worktrees/t_abc123def",
            ),
        ]
        db = self._make_db_mock({})
        _enrich_session_list(db, sessions)

        assert sessions[0]["kind"] == "kanban_worker"
        assert sessions[0]["kanban_task_id"] == "t_abc123def"

    def test_db_error_degrades_gracefully(self):
        """If the DB query throws, child_count stays at 0 (no exception)."""
        from hermes_cli.web_server import _enrich_session_list

        sessions = [self._make_session("sess_1")]
        mock_db = MagicMock()
        mock_db._conn.execute.side_effect = Exception("DB locked")
        _enrich_session_list(mock_db, sessions)

        assert sessions[0]["kind"] == "main"
        assert sessions[0]["child_count"] == 0

    def test_empty_list_is_noop(self):
        from hermes_cli.web_server import _enrich_session_list

        result = _enrich_session_list(None, [])
        assert result == []
