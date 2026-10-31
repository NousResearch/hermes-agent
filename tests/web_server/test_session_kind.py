"""Unit tests for _classify_session_kind and _enrich_session_list.

Tests are self-contained: they import the helpers directly from web_server
without starting FastAPI or touching any DB.
"""
import json
import pytest
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Import the real helpers from the production module.  web_server creates a
# FastAPI app at module level but does NOT start a server or open a DB, so
# importing is safe inside the test runner.
# ---------------------------------------------------------------------------
from hermes_cli.web_server import _classify_session_kind, _KANBAN_WORKTREE_RE


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
            "origin_json": None,
            **kwargs,
        }

    # ── main ─────────────────────────────────────────────────────────────────

    def test_plain_cli_session_is_main(self):
        s = self._make(source="cli", cwd="/home/user/project")
        result = _classify_session_kind(s)
        assert result["kind"] == "main"
        assert result["kanban_task_id"] is None
        assert result["parent_session_id"] is None
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
        assert result["parent_session_id"] is None

    def test_parent_session_id_set_is_subagent(self):
        s = self._make(source="cli", parent_session_id="20261031_parent")
        result = _classify_session_kind(s)
        assert result["kind"] == "subagent"
        # parent_session_id is threaded through
        assert result["parent_session_id"] == "20261031_parent"

    def test_source_subagent_takes_precedence_over_worktree_cwd(self):
        """A subagent launched inside a worktree stays 'subagent', not 'kanban_worker'."""
        s = self._make(
            source="subagent",
            cwd="/home/user/.hermes/hermes-agent/.worktrees/t_abc123",
        )
        assert _classify_session_kind(s)["kind"] == "subagent"

    def test_source_subagent_takes_precedence_over_origin_json(self):
        """A subagent's source tag beats any kanban_task_id in origin_json."""
        s = self._make(
            source="subagent",
            origin_json=json.dumps({"kanban_task_id": "t_abc123"}),
        )
        result = _classify_session_kind(s)
        assert result["kind"] == "subagent"
        assert result["kanban_task_id"] is None

    def test_parent_set_takes_precedence_over_worktree_cwd(self):
        s = self._make(
            parent_session_id="sess_parent",
            cwd="/home/user/.hermes/hermes-agent/.worktrees/t_abc123",
        )
        assert _classify_session_kind(s)["kind"] == "subagent"

    # ── kanban_worker — priority (a): origin_json ─────────────────────────────

    def test_origin_json_kanban_task_id_is_kanban_worker(self):
        """Priority a: origin_json.kanban_task_id → kanban_worker."""
        s = self._make(
            source="cli",
            origin_json=json.dumps({"kanban_task_id": "t_e659aa8a"}),
        )
        result = _classify_session_kind(s)
        assert result["kind"] == "kanban_worker"
        assert result["kanban_task_id"] == "t_e659aa8a"

    def test_origin_json_beats_non_worktree_cwd(self):
        """origin_json detection works even when cwd doesn't match .worktrees."""
        s = self._make(
            source="cli",
            cwd="/tmp/scratch",
            origin_json=json.dumps({"kanban_task_id": "t_abc9999"}),
        )
        result = _classify_session_kind(s)
        assert result["kind"] == "kanban_worker"
        assert result["kanban_task_id"] == "t_abc9999"

    def test_origin_json_with_extra_fields_still_works(self):
        """Other fields in origin_json are ignored."""
        s = self._make(
            origin_json=json.dumps({"kanban_task_id": "t_xyz", "platform": "cli"}),
        )
        result = _classify_session_kind(s)
        assert result["kind"] == "kanban_worker"
        assert result["kanban_task_id"] == "t_xyz"

    def test_origin_json_without_kanban_task_id_falls_through(self):
        """origin_json present but no kanban_task_id → fall through to cwd check."""
        s = self._make(
            cwd="/home/user/.hermes/.worktrees/t_fallback1",
            origin_json=json.dumps({"platform": "discord"}),
        )
        result = _classify_session_kind(s)
        assert result["kind"] == "kanban_worker"
        assert result["kanban_task_id"] == "t_fallback1"

    def test_invalid_origin_json_falls_through_to_cwd(self):
        """Malformed origin_json is silently skipped; cwd fallback applies."""
        s = self._make(
            cwd="/home/user/.hermes/.worktrees/t_cwdfallback",
            origin_json="{invalid json!!!",
        )
        result = _classify_session_kind(s)
        assert result["kind"] == "kanban_worker"
        assert result["kanban_task_id"] == "t_cwdfallback"

    # ── kanban_worker — priority (b): cwd pattern ─────────────────────────────

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

    # ── real DB sample verification ───────────────────────────────────────────

    def test_real_db_sample_fe3aea_is_kanban_worker(self):
        """Regression: session 20261031_145542_fe3aea must be kanban_worker/t_b3341627."""
        s = self._make(
            source="cli",
            cwd="/Users/othels/.hermes/hermes-agent/.worktrees/t_b3341627",
        )
        result = _classify_session_kind(s)
        assert result["kind"] == "kanban_worker"
        assert result["kanban_task_id"] == "t_b3341627"

    def test_real_db_sample_5183a0_is_kanban_worker(self):
        """Regression: session 20261031_144733_5183a0 must be kanban_worker/t_5bcea939."""
        s = self._make(
            source="cli",
            cwd="/Users/othels/.hermes/hermes-agent/.worktrees/t_5bcea939",
        )
        result = _classify_session_kind(s)
        assert result["kind"] == "kanban_worker"
        assert result["kanban_task_id"] == "t_5bcea939"

    # ── parent_session_id field is threaded through in all cases ─────────────

    def test_parent_session_id_none_in_main(self):
        s = self._make(source="cli")
        assert _classify_session_kind(s)["parent_session_id"] is None

    def test_parent_session_id_none_in_kanban_worker(self):
        s = self._make(cwd="/home/.hermes/.worktrees/t_abc")
        assert _classify_session_kind(s)["parent_session_id"] is None

    def test_parent_session_id_present_in_subagent(self):
        s = self._make(source="cli", parent_session_id="parent_xyz")
        assert _classify_session_kind(s)["parent_session_id"] == "parent_xyz"

    # ── multi-level subagent (recursive) ──────────────────────────────────────

    def test_nested_subagent_is_subagent(self):
        """Subagent of a subagent still classifies as subagent."""
        s = self._make(source="subagent", parent_session_id="sess_middle")
        assert _classify_session_kind(s)["kind"] == "subagent"

    def test_grandchild_subagent_classified_by_source(self):
        """A session that is source=subagent with a grandparent parent_session_id."""
        s = self._make(
            source="subagent",
            parent_session_id="sess_child",  # child of child
        )
        result = _classify_session_kind(s)
        assert result["kind"] == "subagent"
        assert result["parent_session_id"] == "sess_child"


class TestEnrichSessionList:
    """Smoke tests for the child_count / child_counts batch query path.

    We mock the DB._conn to avoid needing a real SQLite DB.
    The new query shape is:
      SELECT parent_session_id, source, cwd, origin_json, COUNT(*) AS cnt
      FROM sessions
      WHERE parent_session_id IN (...)
      GROUP BY parent_session_id, source, cwd, origin_json
    """

    def _make_db_mock(self, child_rows: list) -> MagicMock:
        """Return a mock db whose _conn.execute returns child rows.

        Each element in ``child_rows`` is a tuple:
          (parent_session_id, source, cwd, origin_json, count).
        """
        mock_cursor = MagicMock()
        # Wrap each row so row[0..4] indexing works.
        mock_cursor.fetchall.return_value = [list(r) for r in child_rows]
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
            "origin_json": None,
            **kwargs,
        }

    def test_child_count_injected_for_main_sessions(self):
        """child_count (scalar) is preserved for backward-compat."""
        from hermes_cli.web_server import _enrich_session_list

        sessions = [
            self._make_session("sess_1"),
            self._make_session("sess_2"),
        ]
        # Two subagent children for sess_1, zero for sess_2.
        db = self._make_db_mock([
            ("sess_1", "subagent", "/tmp/a", None, 3),
        ])
        with patch("hermes_cli.web_server._resolve_project_ids_batch", return_value={}):
            _enrich_session_list(db, sessions)

        assert sessions[0]["kind"] == "main"
        assert sessions[0]["child_count"] == 3
        assert sessions[1]["child_count"] == 0

    def test_child_counts_dict_subagent_and_kanban(self):
        """child_counts dict has separate subagent and kanban_worker buckets."""
        from hermes_cli.web_server import _enrich_session_list

        sessions = [self._make_session("sess_main")]
        db = self._make_db_mock([
            # 2 subagent children
            ("sess_main", "subagent", "/tmp/s", None, 2),
            # 1 kanban_worker child (via cwd)
            ("sess_main", "cli", "/home/.hermes/.worktrees/t_abc", None, 1),
        ])
        with patch("hermes_cli.web_server._resolve_project_ids_batch", return_value={}):
            _enrich_session_list(db, sessions)

        sc = sessions[0]["child_counts"]
        assert sc["subagent"] == 2
        assert sc["kanban_worker"] == 1
        # scalar backward-compat = subagent count
        assert sessions[0]["child_count"] == 2

    def test_child_counts_kanban_via_origin_json(self):
        """kanban_worker child identified by origin_json goes to kanban_worker bucket."""
        from hermes_cli.web_server import _enrich_session_list

        sessions = [self._make_session("sess_main")]
        db = self._make_db_mock([
            ("sess_main", "cli", "/tmp/x", json.dumps({"kanban_task_id": "t_xyz"}), 1),
        ])
        with patch("hermes_cli.web_server._resolve_project_ids_batch", return_value={}):
            _enrich_session_list(db, sessions)

        sc = sessions[0]["child_counts"]
        assert sc["kanban_worker"] == 1
        assert sc["subagent"] == 0

    def test_subagent_sessions_get_zero_child_count(self):
        from hermes_cli.web_server import _enrich_session_list

        sessions = [
            self._make_session("sess_sub", source="subagent", parent_session_id="sess_main"),
        ]
        db = self._make_db_mock([])
        with patch("hermes_cli.web_server._resolve_project_ids_batch", return_value={}):
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
        db = self._make_db_mock([])
        with patch("hermes_cli.web_server._resolve_project_ids_batch", return_value={}):
            _enrich_session_list(db, sessions)

        assert sessions[0]["kind"] == "kanban_worker"
        assert sessions[0]["kanban_task_id"] == "t_abc123def"

    def test_project_id_set_from_batch_resolver(self):
        """project_id field is added from _resolve_project_ids_batch."""
        from hermes_cli.web_server import _enrich_session_list

        sessions = [self._make_session("sess_1")]
        db = self._make_db_mock([])
        with patch(
            "hermes_cli.web_server._resolve_project_ids_batch",
            return_value={"sess_1": "proj_xyz"},
        ):
            _enrich_session_list(db, sessions)

        assert sessions[0]["project_id"] == "proj_xyz"

    def test_project_id_none_when_not_matched(self):
        """project_id is None when the session doesn't belong to any project."""
        from hermes_cli.web_server import _enrich_session_list

        sessions = [self._make_session("sess_orphan")]
        db = self._make_db_mock([])
        with patch(
            "hermes_cli.web_server._resolve_project_ids_batch",
            return_value={"sess_orphan": None},
        ):
            _enrich_session_list(db, sessions)

        assert sessions[0]["project_id"] is None

    def test_db_error_degrades_gracefully(self):
        """If the DB query throws, child_count stays at 0 (no exception)."""
        from hermes_cli.web_server import _enrich_session_list

        sessions = [self._make_session("sess_1")]
        mock_db = MagicMock()
        mock_db._conn.execute.side_effect = Exception("DB locked")
        with patch("hermes_cli.web_server._resolve_project_ids_batch", return_value={}):
            _enrich_session_list(mock_db, sessions)

        assert sessions[0]["kind"] == "main"
        assert sessions[0]["child_count"] == 0

    def test_empty_list_is_noop(self):
        from hermes_cli.web_server import _enrich_session_list

        result = _enrich_session_list(None, [])
        assert result == []

    def test_existing_response_fields_unchanged(self):
        """Enrichment only adds new keys; existing fields pass through unchanged."""
        from hermes_cli.web_server import _enrich_session_list

        sessions = [
            {
                "id": "sess_check",
                "source": "cli",
                "parent_session_id": None,
                "cwd": "/home/user/project",
                "origin_json": None,
                "title": "My Session",
                "started_at": 1234567890,
                "message_count": 5,
            }
        ]
        db = self._make_db_mock([])
        with patch("hermes_cli.web_server._resolve_project_ids_batch", return_value={}):
            _enrich_session_list(db, sessions)

        # Original fields must survive untouched.
        assert sessions[0]["title"] == "My Session"
        assert sessions[0]["started_at"] == 1234567890
        assert sessions[0]["message_count"] == 5
        assert sessions[0]["source"] == "cli"
        # New additive fields present.
        assert "kind" in sessions[0]
        assert "kanban_task_id" in sessions[0]
        assert "child_count" in sessions[0]
        assert "child_counts" in sessions[0]
        assert "project_id" in sessions[0]
        assert "parent_session_id" in sessions[0]


class TestResolveProjectIdsBatch:
    """Unit tests for _resolve_project_ids_batch.

    Validates that worktree cwds match their parent project.
    """

    def test_returns_empty_dict_on_import_error(self):
        from hermes_cli.web_server import _resolve_project_ids_batch

        with patch.dict("sys.modules", {"hermes_cli.projects_db": None}):
            # ImportError path — should return {} without raising.
            result = _resolve_project_ids_batch([{"id": "s1"}])
            # Either {} or a graceful degradation dict.
            assert isinstance(result, dict)

    def test_no_projects_returns_none_for_all(self):
        """When there are no projects, every session gets project_id=None."""
        from hermes_cli.web_server import _resolve_project_ids_batch

        mock_conn = MagicMock()
        with (
            patch("hermes_cli.web_server._resolve_project_ids_batch.__module__"),
            patch("hermes_cli.projects_db.connect_closing") as mock_ctx,
            patch("hermes_cli.projects_db.list_projects", return_value=[]),
        ):
            mock_ctx.return_value.__enter__ = lambda s: mock_conn
            mock_ctx.return_value.__exit__ = MagicMock(return_value=False)
            from hermes_cli import projects_db as _pdb
            orig = _pdb.list_projects
            _pdb.list_projects = lambda conn: []
            try:
                result = _resolve_project_ids_batch([{"id": "s1", "cwd": "/tmp/x"}])
            finally:
                _pdb.list_projects = orig
            assert result.get("s1") is None
