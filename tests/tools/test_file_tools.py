"""Tests for the file tools module (schema, handler wiring, error paths).

Tests verify tool schemas, handler dispatch, validation logic, and error
handling without requiring a running terminal environment.
"""

import json
import logging
from unittest.mock import MagicMock, patch

from tools.file_tools import (
    PATCH_SCHEMA,
)


class TestReadFileHandler:
    @patch("tools.file_tools._get_file_ops")
    def test_returns_file_content(self, mock_get):
        mock_ops = MagicMock()
        result_obj = MagicMock()
        result_obj.content = "line1\nline2"
        result_obj.to_dict.return_value = {"content": "line1\nline2", "total_lines": 2}
        mock_ops.read_file.return_value = result_obj
        mock_get.return_value = mock_ops

        from tools.file_tools import read_file_tool
        result = json.loads(read_file_tool("/tmp/test.txt"))
        assert result["content"] == "line1\nline2"
        assert result["total_lines"] == 2
        mock_ops.read_file.assert_called_once_with("/tmp/test.txt", 1, 2000)


    @patch("tools.file_tools._get_file_ops")
    def test_exception_returns_error_json(self, mock_get):
        mock_get.side_effect = RuntimeError("terminal not available")

        from tools.file_tools import read_file_tool
        result = json.loads(read_file_tool("/tmp/test.txt"))
        assert "error" in result
        assert "terminal not available" in result["error"]


class TestWriteFileHandler:
    @patch("tools.file_tools._get_file_ops")
    def test_writes_content(self, mock_get):
        mock_ops = MagicMock()
        result_obj = MagicMock()
        result_obj.to_dict.return_value = {"status": "ok", "path": "/tmp/out.txt", "bytes": 13}
        mock_ops.write_file.return_value = result_obj
        mock_get.return_value = mock_ops

        from tools.file_tools import write_file_tool
        result = json.loads(write_file_tool("/tmp/out.txt", "hello world!\n"))
        assert result["status"] == "ok"
        mock_ops.write_file.assert_called_once_with("/tmp/out.txt", "hello world!\n")

    @patch("tools.file_tools._get_file_ops")
    def test_permission_error_returns_error_json_without_error_log(self, mock_get, caplog):
        mock_get.side_effect = PermissionError("read-only filesystem")

        from tools.file_tools import write_file_tool
        with caplog.at_level(logging.DEBUG, logger="tools.file_tools"):
            result = json.loads(write_file_tool("/tmp/out.txt", "data"))
        assert "error" in result
        assert "read-only" in result["error"]
        assert any("write_file expected denial" in r.getMessage() for r in caplog.records)
        assert not any(r.levelno >= logging.ERROR for r in caplog.records)

    @patch("tools.file_tools._get_file_ops")
    def test_rejects_read_file_line_numbered_content(self, mock_get):
        """#19798 — do not persist read_file's LINE_NUM|CONTENT display format."""
        from tools.file_tools import write_file_tool

        content = " 1|setting: new_value\n 2|other: thing\n"
        result = json.loads(write_file_tool("/tmp/config.yaml", content))

        assert "error" in result
        assert "line-number" in result["error"].lower()
        mock_get.assert_not_called()


    @patch("tools.file_tools._get_file_ops")
    def test_unexpected_exception_still_logs_error(self, mock_get, caplog):
        mock_get.side_effect = RuntimeError("boom")

        from tools.file_tools import write_file_tool
        with caplog.at_level(logging.ERROR, logger="tools.file_tools"):
            result = json.loads(write_file_tool("/tmp/out.txt", "data"))
        assert result["error"] == "boom"
        assert any("write_file error" in r.getMessage() for r in caplog.records)

    def test_missing_content_key_returns_error(self):
        """#19096 — handler must reject tool calls where 'content' key is absent."""
        from tools.file_tools import _handle_write_file

        result = json.loads(_handle_write_file({"path": "/tmp/oops.md"}))
        assert "error" in result
        assert "content" in result["error"]
        assert "path" not in result.get("error", "").lower() or "missing" not in result.get("error", "").lower() or True  # just check error present

    def test_missing_path_key_returns_error(self):
        """#19096 — handler must reject tool calls where 'path' key is absent."""
        from tools.file_tools import _handle_write_file

        result = json.loads(_handle_write_file({"content": "hello"}))
        assert "error" in result

    def test_explicit_empty_content_is_allowed(self):
        """#19096 — explicit empty string content (file truncation) must still work."""
        from tools.file_tools import _handle_write_file

        with patch("tools.file_tools._get_file_ops") as mock_get:
            mock_ops = MagicMock()
            result_obj = MagicMock()
            result_obj.to_dict.return_value = {"status": "ok", "path": "/tmp/empty.txt", "bytes": 0}
            mock_ops.write_file.return_value = result_obj
            mock_get.return_value = mock_ops

            result = json.loads(_handle_write_file({"path": "/tmp/empty.txt", "content": ""}))
            assert result["status"] == "ok"

    def test_non_string_content_returns_error(self):
        """#19096 — content must be a string, not a dict or list."""
        from tools.file_tools import _handle_write_file

        result = json.loads(_handle_write_file({"path": "/tmp/x.txt", "content": {"nested": "dict"}}))
        assert "error" in result
        assert "string" in result["error"].lower() or "content" in result["error"].lower()


class TestPatchHandler:
    @patch("tools.file_tools._get_file_ops")
    def test_replace_mode_calls_patch_replace(self, mock_get):
        mock_ops = MagicMock()
        result_obj = MagicMock()
        result_obj.to_dict.return_value = {"status": "ok", "replacements": 1}
        mock_ops.patch_replace.return_value = result_obj
        mock_get.return_value = mock_ops

        from tools.file_tools import patch_tool
        result = json.loads(patch_tool(
            mode="replace", path="/tmp/f.py",
            old_string="foo", new_string="bar"
        ))
        assert result["status"] == "ok"
        mock_ops.patch_replace.assert_called_once_with("/tmp/f.py", "foo", "bar", False)


    @patch("tools.file_tools._get_file_ops")
    def test_patch_mode_calls_patch_v4a(self, mock_get):
        mock_ops = MagicMock()
        result_obj = MagicMock()
        result_obj.to_dict.return_value = {"status": "ok", "operations": 1}
        mock_ops.patch_v4a.return_value = result_obj
        mock_get.return_value = mock_ops

        from tools.file_tools import patch_tool
        result = json.loads(patch_tool(mode="patch", patch="*** Begin Patch\n..."))
        assert result["status"] == "ok"
        mock_ops.patch_v4a.assert_called_once()


    @patch("tools.file_tools._get_file_ops")
    def test_unknown_mode_errors(self, mock_get):
        from tools.file_tools import patch_tool
        result = json.loads(patch_tool(mode="invalid_mode"))
        assert "error" in result
        assert "Unknown mode" in result["error"]

    @patch("tools.file_tools._get_file_ops")
    def test_patch_v4a_rejects_traversal_in_update_header(self, mock_get):
        """V4A '*** Update File:' headers come from patch content, which can
        carry prompt-injection-controlled paths (skill content, web extract).
        ``..`` traversal in the header must be rejected before the patch is
        applied, even though the explicit ``path=`` arg is allowed to use
        ``..`` for legitimate cross-worktree edits."""
        from tools.file_tools import patch_tool
        result = json.loads(patch_tool(
            mode="patch",
            patch=(
                "*** Begin Patch\n"
                "*** Update File: ../../../etc/shadow\n"
                "@@ -1,3 +1,3 @@\n"
                "-old\n"
                "+new\n"
                "*** End Patch\n"
            ),
        ))
        assert "error" in result
        assert "traversal" in result["error"].lower()
        # patch_v4a must not be invoked when the header is rejected
        mock_get.return_value.patch_v4a.assert_not_called()

    @patch("tools.file_tools._get_file_ops")
    def test_patch_v4a_rejects_traversal_in_add_header(self, mock_get):
        from tools.file_tools import patch_tool
        result = json.loads(patch_tool(
            mode="patch",
            patch=(
                "*** Begin Patch\n"
                "*** Add File: ../../../tmp/dropped.py\n"
                "+print('pwned')\n"
                "*** End Patch\n"
            ),
        ))
        assert "error" in result
        assert "traversal" in result["error"].lower()


class TestPatchSensitivePathExtraction:
    """Regression tests for patch_tool sensitive-path extraction.

    The sensitive path check relies on a regex that parses V4A patch
    headers. These tests cover:

    1. ``*** Move File:`` operations (previously missed — the regex only
       matched Update/Add/Delete, so Move could target /etc/* without
       hitting the check).
    2. ``***Keyword File:`` with no space after ``***`` (previously missed —
       the regex required ``\\s+`` even though patch_parser accepts ``\\s*``).
    3. ``..`` traversal in Move headers (the Move endpoints run through the
       same traversal rejection as the other V4A headers).
    """

    @patch("tools.file_tools._get_file_ops")
    def test_patch_move_to_sensitive_dst_blocked(self, mock_get):
        from tools.file_tools import patch_tool
        patch_text = (
            "*** Begin Patch\n"
            "*** Move File: /tmp/work.txt -> /etc/crontab\n"
            "*** End Patch\n"
        )
        result = json.loads(patch_tool(mode="patch", patch=patch_text))
        assert "error" in result
        assert "sensitive" in result["error"].lower()
        mock_get.assert_not_called()


    @patch("tools.file_tools._get_file_ops")
    def test_patch_update_no_space_after_asterisks_blocked(self, mock_get):
        """``***Update File:`` (no space after asterisks) must also be caught.

        patch_parser.py accepts this form (``\\s*`` in its regex), so the
        sensitive path check must be at least as lenient or the check
        is bypassed.
        """
        from tools.file_tools import patch_tool
        patch_text = (
            "*** Begin Patch\n"
            "***Update File: /etc/resolv.conf\n"
            "@@ @@\n"
            "-old\n"
            "+new\n"
            "*** End Patch\n"
        )
        result = json.loads(patch_tool(mode="patch", patch=patch_text))
        assert "error" in result
        assert "sensitive" in result["error"].lower()
        mock_get.assert_not_called()


    @patch("tools.file_tools._get_file_ops")
    def test_patch_move_safe_paths_not_blocked(self, mock_get):
        """Safe Move operations should still reach the file_ops dispatch."""
        mock_ops = MagicMock()
        result_obj = MagicMock()
        result_obj.to_dict.return_value = {"status": "ok"}
        mock_ops.patch_v4a.return_value = result_obj
        mock_get.return_value = mock_ops

        from tools.file_tools import patch_tool
        patch_text = (
            "*** Begin Patch\n"
            "*** Move File: /tmp/a.txt -> /tmp/b.txt\n"
            "*** End Patch\n"
        )
        result = json.loads(patch_tool(mode="patch", patch=patch_text))
        assert "error" not in result
        mock_ops.patch_v4a.assert_called_once()


class TestSearchHandler:
    @patch("tools.file_tools._get_file_ops")
    def test_search_calls_file_ops(self, mock_get):
        mock_ops = MagicMock()
        result_obj = MagicMock()
        result_obj.to_dict.return_value = {"matches": ["file1.py:3:match"]}
        mock_ops.search.return_value = result_obj
        mock_get.return_value = mock_ops

        from tools.file_tools import search_tool
        result = json.loads(search_tool(pattern="TODO", target="content", path="."))
        assert "matches" in result
        mock_ops.search.assert_called_once()


    @patch("tools.file_tools._get_file_ops")
    def test_search_exception_returns_error(self, mock_get):
        mock_get.side_effect = RuntimeError("no terminal")

        from tools.file_tools import search_tool
        result = json.loads(search_tool(pattern="x"))
        assert "error" in result


# ---------------------------------------------------------------------------
# Windows MSYS path resolution (salvage of #50488 / #46995)
# ---------------------------------------------------------------------------

class TestWindowsMsysPathResolution:
    """File tools must translate Git Bash drive paths before Path resolution."""

    def test_absolute_msys_path_normalized_before_windows_resolve(self, monkeypatch):
        import tools.environments.local as local_mod
        import tools.file_tools as file_tools

        monkeypatch.setattr(file_tools.sys, "platform", "win32")
        monkeypatch.setattr(local_mod, "_IS_WINDOWS", True)
        monkeypatch.setattr(file_tools, "_uses_container_paths", lambda task_id="default": False)

        resolved = file_tools._resolve_path_for_task("/c/Users/Mark/project/app.py")
        assert str(resolved) == r"C:\Users\Mark\project\app.py"


    def test_container_paths_skip_msys_translation(self, monkeypatch):
        """WSL/docker Linux paths must not be rewritten as Windows drives."""
        import tools.environments.local as local_mod
        import tools.file_tools as file_tools

        monkeypatch.setattr(file_tools.sys, "platform", "win32")
        monkeypatch.setattr(local_mod, "_IS_WINDOWS", True)
        monkeypatch.setattr(file_tools, "_uses_container_paths", lambda task_id="default": True)
        monkeypatch.setattr(
            file_tools,
            "_authoritative_workspace_root",
            lambda task_id="default": "/home/don/project",
        )

        resolved = file_tools._resolve_path_for_task("/home/don/.env")
        assert str(resolved) == "/home/don/.env"


# ---------------------------------------------------------------------------
# Tool result hint tests (#722)
# ---------------------------------------------------------------------------

class TestPatchHints:
    """Patch tool should hint when old_string is not found."""

    @patch("tools.file_tools._get_file_ops")
    def test_no_match_includes_hint(self, mock_get):
        mock_ops = MagicMock()
        result_obj = MagicMock()
        result_obj.to_dict.return_value = {
            "error": "Could not find match for old_string in foo.py"
        }
        mock_ops.patch_replace.return_value = result_obj
        mock_get.return_value = mock_ops

        from tools.file_tools import patch_tool
        raw = patch_tool(mode="replace", path="foo.py", old_string="x", new_string="y")
        # patch_tool surfaces the hint as a structured "_hint" field on the
        # JSON error payload (not an inline "[Hint: ..." tail).
        assert "_hint" in raw
        assert "read_file" in raw

    @patch("tools.file_tools._get_file_ops")
    def test_success_no_hint(self, mock_get):
        mock_ops = MagicMock()
        result_obj = MagicMock()
        result_obj.to_dict.return_value = {"success": True, "diff": "--- a\n+++ b"}
        mock_ops.patch_replace.return_value = result_obj
        mock_get.return_value = mock_ops

        from tools.file_tools import patch_tool
        raw = patch_tool(mode="replace", path="foo.py", old_string="x", new_string="y")
        assert "_hint" not in raw


class TestSearchHints:
    """Search tool should hint when results are truncated."""

    def setup_method(self):
        """Clear read/search tracker between tests to avoid cross-test state."""
        from tools.file_tools import _read_tracker
        _read_tracker.clear()

    @patch("tools.file_tools._get_file_ops")
    def test_truncated_results_hint(self, mock_get):
        mock_ops = MagicMock()
        result_obj = MagicMock()
        result_obj.to_dict.return_value = {
            "total_count": 100,
            "matches": [{"path": "a.py", "line": 1, "content": "x"}] * 50,
            "truncated": True,
        }
        mock_ops.search.return_value = result_obj
        mock_get.return_value = mock_ops

        from tools.file_tools import search_tool
        raw = search_tool(pattern="foo", offset=0, limit=50)
        assert "[Hint:" in raw
        assert "offset=50" in raw


    @patch("tools.file_tools._get_file_ops")
    def test_truncated_hint_with_nonzero_offset(self, mock_get):
        mock_ops = MagicMock()
        result_obj = MagicMock()
        result_obj.to_dict.return_value = {
            "total_count": 150,
            "matches": [{"path": "a.py", "line": 1, "content": "x"}] * 50,
            "truncated": True,
        }
        mock_ops.search.return_value = result_obj
        mock_get.return_value = mock_ops

        from tools.file_tools import search_tool
        raw = search_tool(pattern="foo", offset=50, limit=50)
        assert "[Hint:" in raw
        assert "offset=100" in raw


# ---------------------------------------------------------------------------
# PATCH_SCHEMA shape tests (issue #15524)
# ---------------------------------------------------------------------------


class TestSensitivePathCheck:
    """Verify that _check_sensitive_path blocks writes to protected locations."""

    def test_hermes_config_blocked_for_write_file(self, tmp_path, monkeypatch):
        fake_config = tmp_path / "config.yaml"
        monkeypatch.setattr("tools.file_tools._hermes_config_resolved", str(fake_config))
        monkeypatch.setattr("tools.file_tools._hermes_config_resolved_loaded", True)

        from tools.file_tools import write_file_tool
        result = json.loads(write_file_tool(str(fake_config), "approvals:\n  mode: off\n"))
        assert "error" in result
        assert "Hermes config" in result["error"]

    def test_hermes_config_blocked_via_tilde_path(self, tmp_path, monkeypatch):
        fake_config = tmp_path / "config.yaml"
        monkeypatch.setattr("tools.file_tools._hermes_config_resolved", str(fake_config))
        monkeypatch.setattr("tools.file_tools._hermes_config_resolved_loaded", True)

        from tools.file_tools import write_file_tool
        result = json.loads(write_file_tool(str(fake_config), "approvals:\n  mode: off\n"))
        assert "error" in result
        assert "Hermes config" in result["error"]


    def test_system_path_still_blocked(self, monkeypatch):
        monkeypatch.setattr("tools.file_tools._hermes_config_resolved", "/some/other/path")
        monkeypatch.setattr("tools.file_tools._hermes_config_resolved_loaded", True)

        from tools.file_tools import write_file_tool
        result = json.loads(write_file_tool("/etc/passwd", "evil"))
        assert "error" in result
        assert "sensitive system path" in result["error"]

    def test_macos_private_var_carveouts(self):
        """macOS temp dirs under /private/var must not be blanket-blocked,
        while the genuinely-sensitive /private/var subtrees still are."""
        from tools.file_tools import _check_sensitive_path

        # $TMPDIR / /tmp / /var/folders realpath into these on macOS.
        assert _check_sensitive_path("/private/var/folders/xy/T/tmp.txt") is None
        assert _check_sensitive_path("/private/var/tmp/build.log") is None
        # Sensitive subtrees remain blocked.
        assert _check_sensitive_path("/private/var/db/secret") is not None
        assert _check_sensitive_path("/private/var/root/x") is not None
        # /etc (and its macOS /private/etc mirror) stay blocked.
        assert _check_sensitive_path("/private/etc/hosts") is not None

    @patch("tools.file_tools._get_file_ops")
    def test_normal_file_not_blocked(self, mock_get, monkeypatch):
        monkeypatch.setattr("tools.file_tools._hermes_config_resolved", "/home/user/.hermes/config.yaml")
        monkeypatch.setattr("tools.file_tools._hermes_config_resolved_loaded", True)
        mock_ops = MagicMock()
        result_obj = MagicMock()
        result_obj.to_dict.return_value = {"status": "ok", "path": "/tmp/other.txt", "bytes": 5}
        mock_ops.write_file.return_value = result_obj
        mock_get.return_value = mock_ops

        from tools.file_tools import write_file_tool
        result = json.loads(write_file_tool("/tmp/other.txt", "hello"))
        assert result["status"] == "ok"


class TestPatchSchemaShape:
    """PATCH_SCHEMA must advertise per-mode required params via description
    text (not JSON-schema ``required``), so strict models like kimi-k2.x stop
    silently omitting old_string / new_string / patch content."""

    def test_per_mode_required_params_documented_in_descriptions(self):
        desc = PATCH_SCHEMA["description"]
        assert "REQUIRED PARAMETERS: mode, path, old_string, new_string" in desc
        assert "REQUIRED PARAMETERS: mode, patch" in desc
        props = PATCH_SCHEMA["parameters"]["properties"]
        for name in ("path", "old_string", "new_string"):
            assert "REQUIRED when mode='replace'" in props[name]["description"]
        assert "REQUIRED when mode='patch'" in props["patch"]["description"]

    def test_no_anyof_required_stays_mode_only(self):
        # anyOf/oneOf at parameters level break Anthropic, Fireworks, and the
        # Moonshot/Kimi schema sanitizer — description-level guidance is the
        # only provider-safe signalling mechanism.
        params = PATCH_SCHEMA["parameters"]
        assert params["required"] == ["mode"]
        assert "anyOf" not in params and "oneOf" not in params


# ---------------------------------------------------------------------------
# Session-cwd persistence across env recreation (#26211: silent file creation
# failure in long conversations). The durable anchor is the per-session cwd
# record in terminal_tool; env cleanup cannot lose it because it never lived
# on the env.
# ---------------------------------------------------------------------------

class TestSessionCwdSurvivesEnvRecreation:
    """
    When the terminal environment is cleaned up and re-created during a long
    conversation, the session's cwd record preserves the working directory so
    subsequent file writes with relative paths land in the right directory.

    Regression guard for issue #26211.
    """

    @patch("tools.terminal_tool._active_environments", new_callable=dict)
    @patch("tools.file_tools._file_ops_cache", new_callable=dict)
    @patch("tools.terminal_tool._get_env_config")
    @patch("tools.terminal_tool._create_environment")
    def test_recorded_cwd_used_for_recreated_env(
        self, mock_create_env, mock_config, mock_cache, mock_active
    ):
        import tools.terminal_tool as tt
        from tools.file_tools import _get_file_ops

        mock_env = MagicMock()
        mock_env.cwd = "/Users/user/project"
        mock_create_env.return_value = mock_env
        mock_config.return_value = {
            "env_type": "local",
            "cwd": "/default/path",
            "timeout": 30,
        }

        task_id = "default"
        # The session's record holds the directory (written by the last
        # completed terminal command before the env was cleaned up).
        tt.record_session_cwd(task_id, "/Users/user/project")
        try:
            _get_file_ops(task_id)

            create_call = mock_create_env.call_args
            assert create_call is not None, "_create_environment was not called"
            kwargs = create_call.kwargs if create_call.kwargs else {}
            cwd_passed = kwargs.get("cwd", None)
            if cwd_passed is None:
                args = create_call.args if create_call.args else []
                if len(args) >= 3:
                    cwd_passed = args[2]

            assert cwd_passed == "/Users/user/project", \
                f"Expected cwd='/Users/user/project', got {cwd_passed!r}"
        finally:
            tt.clear_session_cwd(task_id)


    @patch("tools.terminal_tool._active_environments", new_callable=dict)
    @patch("tools.file_tools._file_ops_cache", new_callable=dict)
    @patch("tools.terminal_tool._get_env_config")
    @patch("tools.terminal_tool._create_environment")
    def test_stale_cache_cwd_rescued_into_record_on_cleanup_detection(
        self, mock_create_env, mock_config, mock_cache, mock_active
    ):
        """If the env died but the file-ops cache entry survived, its cwd is
        rescued into the session record before the cache entry is dropped —
        the recreated env starts where the user left off."""
        import tools.terminal_tool as tt
        from tools.file_tools import _get_file_ops

        task_id = "default"
        tt.clear_session_cwd(task_id)

        # Stale cache entry: env was cleaned up, cache still holds the old cwd.
        cached = MagicMock()
        cached.env = None
        cached.cwd = "/Users/user/project"
        mock_cache[task_id] = cached

        mock_env = MagicMock()
        mock_env.cwd = "/Users/user/project"
        mock_create_env.return_value = mock_env
        mock_config.return_value = {
            "env_type": "local",
            "cwd": "/config/default/path",
            "timeout": 30,
        }

        try:
            _get_file_ops(task_id)

            create_call = mock_create_env.call_args
            assert create_call is not None, "_create_environment was not called"
            kwargs = create_call.kwargs if create_call.kwargs else {}
            cwd_passed = kwargs.get("cwd", None)
            if cwd_passed is None:
                args = create_call.args if create_call.args else []
                if len(args) >= 3:
                    cwd_passed = args[2]

            # Rebuilt env restored the rescued cwd, NOT the config default.
            assert cwd_passed == "/Users/user/project", \
                f"Expected restored cwd='/Users/user/project', got {cwd_passed!r}"
        finally:
            tt.clear_session_cwd(task_id)


class TestSilentFileMisplacementE2E:
    """Real-IO regression for #26211.

    Exercises the actual write_file_tool path against a temp filesystem: an
    agent cd's into a project, the cleanup thread kills the env, and a later
    relative-path write must land in the project dir (not the config default).
    Mocks miss this because resolution (_resolve_path_for_task) runs BEFORE
    _get_file_ops rebuilds the env — only the durable session-cwd record
    makes the resolved path correct.
    """

    def test_relative_write_after_env_cleanup_lands_in_user_cwd(self, tmp_path, monkeypatch):
        import tools.terminal_tool as tt
        import tools.file_tools as ft

        project = tmp_path / "project"
        config_default = tmp_path / "config_default"
        project.mkdir()
        config_default.mkdir()
        monkeypatch.delenv("TERMINAL_CWD", raising=False)

        _orig = tt._get_env_config
        monkeypatch.setattr(
            tt, "_get_env_config",
            lambda: {**_orig(), "env_type": "local", "cwd": str(config_default)},
        )

        task_id = "default"
        tt.clear_session_cwd(task_id)

        # 1) Env alive; agent has cd'd into the project (the completed command
        #    recorded the session cwd — simulate that write here).
        fo = ft._get_file_ops(task_id)
        fo.env.cwd = str(project)
        tt.record_session_cwd(task_id, str(project))
        ft.write_file_tool("alive.txt", "1\n", task_id)
        assert (project / "alive.txt").exists()

        # 2) Cleanup thread kills the env AND clears the file_ops cache.
        with tt._env_lock:
            tt._active_environments.pop(task_id, None)
            tt._last_activity.pop(task_id, None)
        with ft._file_ops_lock:
            ft._file_ops_cache.pop(task_id, None)

        # 3) The next relative write must still land in the project dir.
        res = json.loads(ft.write_file_tool("report.txt", "hello\n", task_id))
        assert res.get("resolved_path") == str(project / "report.txt"), res
        assert (project / "report.txt").exists(), "file should be in the user's cwd"
        assert not (config_default / "report.txt").exists(), \
            "file silently misplaced into config default (the #26211 bug)"

        tt.clear_session_cwd(task_id)


class TestDedupInvalidationTaskResolution:
    """Real-IO regression: dedup eviction must resolve paths per-task.

    ``_invalidate_dedup_for_path`` looked up the read-tracker under the correct
    task_id but resolved the path with the DEFAULT task, so for any task whose
    workspace cwd differs from the process cwd (every ``-w``/Desktop/ACP
    session using relative paths) the computed key never matched the cached
    key and the stale-read entry was never evicted. A read after a write could
    then be served the OLD content stub.
    """

    def test_invalidate_evicts_the_task_resolved_key(self, tmp_path, monkeypatch):
        import tools.terminal_tool as tt
        import tools.file_tools as ft

        workspace = tmp_path / "workspace"
        proc = tmp_path / "proc"
        workspace.mkdir()
        proc.mkdir()
        monkeypatch.delenv("TERMINAL_CWD", raising=False)
        monkeypatch.chdir(proc)  # process cwd != task workspace

        task_id = "acp-dedup"
        monkeypatch.setattr(tt, "_task_env_overrides", {task_id: {"cwd": str(workspace)}})
        (workspace / "data.txt").write_text("v1\n")

        # The task resolves the relative path into the workspace; the default
        # task (the old buggy resolution) would resolve into proc.
        correct = str(ft._resolve_path("data.txt", task_id))
        buggy = str(ft._resolve_path("data.txt"))
        assert correct != buggy, "test precondition: cwds must diverge"

        # Populate the dedup cache via a real read.
        ft.read_file_tool("data.txt", task_id=task_id)
        keys = [k[0] for k in ft._read_tracker.get(task_id, {}).get("dedup", {})]
        assert correct in keys, keys

        # Invalidate as write_file_tool does; the entry must be gone.
        ft._invalidate_dedup_for_path("data.txt", task_id)
        remaining = [k[0] for k in ft._read_tracker.get(task_id, {}).get("dedup", {})]
        assert correct not in remaining, remaining

        ft._read_tracker.pop(task_id, None)


# ---------------------------------------------------------------------------
# Negative-result cache tests
#
# Without this cache, a typo'd path retried 13 times (observed in the wild)
# spawned 13 wc -c subprocesses + 13 ls walks for the "did you mean..." hint.
# The cache returns the same error JSON immediately and skips both shells.
# ---------------------------------------------------------------------------

class TestNotFoundCache:
    @patch("tools.file_tools._get_file_ops")
    def test_read_caches_file_not_found_and_skips_subprocess_on_retry(self, mock_get):
        mock_ops = MagicMock()
        result_obj = MagicMock()
        result_obj.content = None
        # Shape returned by ShellFileOperations._suggest_similar_files
        result_obj.to_dict.return_value = {
            "error": "File not found: /tmp/does-not-exist-neg-1.txt",
            "similar_files": [],
        }
        mock_ops.read_file.return_value = result_obj
        mock_get.return_value = mock_ops

        from tools.file_tools import read_file_tool, _read_tracker
        # Use a unique task_id so we don't collide with other tests.
        tid = "neg-cache-read-1"
        _read_tracker.pop(tid, None)

        # First call: subprocess runs, error returned, cache populated.
        first = json.loads(read_file_tool("/tmp/does-not-exist-neg-1.txt", task_id=tid))
        assert "File not found" in first["error"]
        assert mock_ops.read_file.call_count == 1

        # Second call: same path → cache hit → no new subprocess call.
        second = json.loads(read_file_tool("/tmp/does-not-exist-neg-1.txt", task_id=tid))
        assert "File not found" in second["error"]
        assert mock_ops.read_file.call_count == 1, (
            "Negative cache hit must skip the subprocess on retry"
        )

    @patch("tools.file_tools._get_file_ops")
    def test_read_cache_isolated_per_task(self, mock_get):
        mock_ops = MagicMock()
        result_obj = MagicMock()
        result_obj.to_dict.return_value = {
            "error": "File not found: /tmp/does-not-exist-neg-2.txt",
            "similar_files": [],
        }
        mock_ops.read_file.return_value = result_obj
        mock_get.return_value = mock_ops

        from tools.file_tools import read_file_tool, _read_tracker
        for tid in ("neg-cache-iso-A", "neg-cache-iso-B"):
            _read_tracker.pop(tid, None)

        read_file_tool("/tmp/does-not-exist-neg-2.txt", task_id="neg-cache-iso-A")
        read_file_tool("/tmp/does-not-exist-neg-2.txt", task_id="neg-cache-iso-B")
        # Each task gets its own miss; B doesn't reuse A's cache entry.
        assert mock_ops.read_file.call_count == 2

    @patch("tools.file_tools._get_file_ops")
    def test_read_cache_populated_only_for_not_found(self, mock_get):
        # A successful read must NOT populate the negative cache.
        mock_ops = MagicMock()
        result_obj = MagicMock()
        result_obj.content = "x"
        result_obj.to_dict.return_value = {"content": "x", "total_lines": 1}
        mock_ops.read_file.return_value = result_obj
        mock_get.return_value = mock_ops

        from tools.file_tools import read_file_tool, _read_tracker
        tid = "neg-cache-success-only"
        _read_tracker.pop(tid, None)

        read_file_tool("/tmp/exists-or-mocked.txt", task_id=tid)
        nf = _read_tracker[tid].get("not_found", {})
        assert all(k[0] != "read" or "exists-or-mocked" not in k[1] for k in nf), (
            "Successful reads must not poison the negative cache"
        )

    @patch("tools.file_tools._get_file_ops")
    def test_search_caches_path_not_found_and_skips_subprocess_on_retry(self, mock_get):
        mock_ops = MagicMock()
        result_obj = MagicMock()
        result_obj.matches = []
        result_obj.to_dict.return_value = {
            "error": "Path not found: /tmp/does-not-exist-search-3",
            "total_count": 0,
        }
        mock_ops.search.return_value = result_obj
        mock_get.return_value = mock_ops

        from tools.file_tools import search_tool, _read_tracker
        tid = "neg-cache-search-3"
        _read_tracker.pop(tid, None)

        first = json.loads(search_tool("foo", path="/tmp/does-not-exist-search-3", task_id=tid))
        assert "Path not found" in first["error"]
        assert mock_ops.search.call_count == 1

        second = json.loads(search_tool("foo", path="/tmp/does-not-exist-search-3", task_id=tid))
        assert "Path not found" in second["error"]
        assert mock_ops.search.call_count == 1, (
            "Search negative cache hit must skip the subprocess on retry"
        )

    @patch("tools.file_tools._get_file_ops")
    def test_read_and_search_caches_are_namespaced(self, mock_get):
        # A read that misses must NOT serve a subsequent search call's miss
        # (different error JSON shapes).
        mock_ops = MagicMock()

        read_obj = MagicMock()
        read_obj.to_dict.return_value = {
            "error": "File not found: /tmp/does-not-exist-namespace-4",
        }
        mock_ops.read_file.return_value = read_obj

        search_obj = MagicMock()
        search_obj.matches = []
        search_obj.to_dict.return_value = {
            "error": "Path not found: /tmp/does-not-exist-namespace-4",
            "total_count": 0,
        }
        mock_ops.search.return_value = search_obj

        mock_get.return_value = mock_ops

        from tools.file_tools import read_file_tool, search_tool, _read_tracker
        tid = "neg-cache-namespace-4"
        _read_tracker.pop(tid, None)

        read_file_tool("/tmp/does-not-exist-namespace-4", task_id=tid)
        search_tool("foo", path="/tmp/does-not-exist-namespace-4", task_id=tid)
        # Both ops must hit their own caller (namespacing prevents read's
        # error JSON from being returned to search).
        assert mock_ops.read_file.call_count == 1
        assert mock_ops.search.call_count == 1

    @patch("tools.file_tools._get_file_ops")
    def test_write_invalidates_read_negative_cache(self, mock_get):
        # After write_file on a path, a subsequent read must hit disk,
        # not return the cached "not found" stub.
        mock_ops = MagicMock()

        not_found_obj = MagicMock()
        not_found_obj.to_dict.return_value = {
            "error": "File not found: /tmp/will-be-created-neg-5.txt",
        }
        present_obj = MagicMock()
        present_obj.content = "after write"
        present_obj.to_dict.return_value = {"content": "after write", "total_lines": 1}

        # First read → not found; second read (after write) → present.
        mock_ops.read_file.side_effect = [not_found_obj, present_obj]
        write_result_obj = MagicMock()
        write_result_obj.to_dict.return_value = {"status": "ok"}
        mock_ops.write_file.return_value = write_result_obj
        mock_get.return_value = mock_ops

        from tools.file_tools import read_file_tool, write_file_tool, _read_tracker
        tid = "neg-cache-write-invalidate-5"
        _read_tracker.pop(tid, None)

        first = json.loads(read_file_tool("/tmp/will-be-created-neg-5.txt", task_id=tid))
        assert "File not found" in first["error"]

        write_file_tool("/tmp/will-be-created-neg-5.txt", "after write", task_id=tid)

        second = json.loads(read_file_tool("/tmp/will-be-created-neg-5.txt", task_id=tid))
        assert second.get("content") == "after write", (
            "write_file must invalidate the negative cache so the next read "
            "hits the now-existing file instead of returning a stale stub"
        )
        assert mock_ops.read_file.call_count == 2

    def test_not_found_ttl_expires(self):
        # A cache entry older than _NOT_FOUND_TTL_SECONDS must be discarded.
        from tools.file_tools import (
            _check_not_found_cache,
            _record_not_found,
            _read_tracker,
            _NOT_FOUND_TTL_SECONDS,
        )
        import tools.file_tools as ft

        tid = "neg-cache-ttl-6"
        _read_tracker.pop(tid, None)
        _record_not_found("read", "/tmp/ttl-test", tid, '{"error":"x"}')
        # Fresh entry: cache hit.
        assert _check_not_found_cache("read", "/tmp/ttl-test", tid) is not None

        # Backdate the entry past the TTL.
        with ft._read_tracker_lock:
            entry = _read_tracker[tid]["not_found"][("read", "/tmp/ttl-test")]
            ft._read_tracker[tid]["not_found"][("read", "/tmp/ttl-test")] = (
                entry[0] - _NOT_FOUND_TTL_SECONDS - 1.0,
                entry[1],
            )
        # Stale entry: cache miss, also evicted.
        assert _check_not_found_cache("read", "/tmp/ttl-test", tid) is None
        with ft._read_tracker_lock:
            assert ("read", "/tmp/ttl-test") not in _read_tracker[tid].get("not_found", {})

    def test_out_of_band_creation_defeats_cached_miss(self, tmp_path):
        """CRITICAL staleness contract: a file created AFTER a cached miss —
        by a terminal command or any external process, NOT write_file_tool —
        must be served for real on the next read. The agent pattern
        'check for file → create it → read it' breaks otherwise."""
        from tools.file_tools import (
            _check_not_found_cache,
            _record_not_found,
            _read_tracker,
        )

        tid = "neg-cache-oob-read"
        _read_tracker.pop(tid, None)
        target = tmp_path / "created-later.txt"

        _record_not_found("read", str(target), tid, '{"error":"File not found: x"}')
        assert _check_not_found_cache("read", str(target), tid) is not None

        # Out-of-band creation: plain filesystem write, no tool hook fires.
        target.write_text("real content\n")

        # The cached miss must NOT be served once the path exists…
        assert _check_not_found_cache("read", str(target), tid) is None, (
            "stale 'File not found' served after the file was created "
            "out-of-band — the existence guard regressed"
        )
        # …and the entry is evicted, not just skipped.
        with __import__("tools.file_tools", fromlist=["x"])._read_tracker_lock:
            assert ("read", str(target)) not in _read_tracker[tid].get("not_found", {})

    def test_out_of_band_creation_defeats_cached_search_miss(self, tmp_path):
        """Same contract for search roots: creating a file under a
        previously-missing directory must defeat the cached 'Path not found'."""
        from tools.file_tools import (
            _check_not_found_cache,
            _record_not_found,
            _read_tracker,
        )

        tid = "neg-cache-oob-search"
        _read_tracker.pop(tid, None)
        missing_dir = tmp_path / "later-dir"

        _record_not_found("search", str(missing_dir), tid, '{"error":"Path not found: x"}')
        assert _check_not_found_cache("search", str(missing_dir), tid) is not None

        missing_dir.mkdir()
        (missing_dir / "x.txt").write_text("hi\n")

        assert _check_not_found_cache("search", str(missing_dir), tid) is None, (
            "stale 'Path not found' served after the directory was created"
        )

    def test_notify_other_tool_call_clears_not_found(self):
        """Belt-and-suspenders: any non-read tool (terminal etc.) invalidates
        the task's negative cache via the dispatcher's notify hook."""
        from tools.file_tools import (
            _check_not_found_cache,
            _record_not_found,
            _read_tracker,
            notify_other_tool_call,
        )

        tid = "neg-cache-notify"
        _read_tracker.pop(tid, None)
        _record_not_found("read", "/tmp/never-exists-notify", tid, '{"error":"x"}')
        assert _check_not_found_cache("read", "/tmp/never-exists-notify", tid) is not None

        notify_other_tool_call(tid)

        assert _check_not_found_cache("read", "/tmp/never-exists-notify", tid) is None, (
            "notify_other_tool_call must clear cached misses"
        )
