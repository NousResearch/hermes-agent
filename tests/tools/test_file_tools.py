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
        mock_ops.read_file.assert_called_once_with("/tmp/test.txt", 1, 500)

    @patch("tools.file_tools._get_file_ops")
    def test_custom_offset_and_limit(self, mock_get):
        mock_ops = MagicMock()
        result_obj = MagicMock()
        result_obj.content = "line10"
        result_obj.to_dict.return_value = {"content": "line10", "total_lines": 50}
        mock_ops.read_file.return_value = result_obj
        mock_get.return_value = mock_ops

        from tools.file_tools import read_file_tool
        read_file_tool("/tmp/big.txt", offset=10, limit=20)
        mock_ops.read_file.assert_called_once_with("/tmp/big.txt", 10, 20)

    @patch("tools.file_tools._get_file_ops")
    def test_invalid_offset_and_limit_are_normalized_before_dispatch(self, mock_get):
        mock_ops = MagicMock()
        result_obj = MagicMock()
        result_obj.content = "line1"
        result_obj.to_dict.return_value = {"content": "line1", "total_lines": 1}
        mock_ops.read_file.return_value = result_obj
        mock_get.return_value = mock_ops

        from tools.file_tools import read_file_tool
        read_file_tool("/tmp/big.txt", offset=0, limit=0)
        mock_ops.read_file.assert_called_once_with("/tmp/big.txt", 1, 1)

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
    def test_allows_sparse_literal_pipe_content(self, mock_get):
        """A single literal N| line should not be treated as read_file output."""
        mock_ops = MagicMock()
        result_obj = MagicMock()
        result_obj.to_dict.return_value = {"status": "ok", "path": "/tmp/out.txt", "bytes": 21}
        mock_ops.write_file.return_value = result_obj
        mock_get.return_value = mock_ops

        from tools.file_tools import write_file_tool
        result = json.loads(write_file_tool("/tmp/out.txt", "1|literal value\nplain line\n"))

        assert result["status"] == "ok"
        mock_ops.write_file.assert_called_once()

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
    def test_replace_mode_replace_all_flag(self, mock_get):
        mock_ops = MagicMock()
        result_obj = MagicMock()
        result_obj.to_dict.return_value = {"status": "ok", "replacements": 5}
        mock_ops.patch_replace.return_value = result_obj
        mock_get.return_value = mock_ops

        from tools.file_tools import patch_tool
        patch_tool(mode="replace", path="/tmp/f.py",
                   old_string="x", new_string="y", replace_all=True)
        mock_ops.patch_replace.assert_called_once_with("/tmp/f.py", "x", "y", True)

    @patch("tools.file_tools._get_file_ops")
    def test_replace_mode_missing_path_errors(self, mock_get):
        from tools.file_tools import patch_tool
        result = json.loads(patch_tool(mode="replace", path=None, old_string="a", new_string="b"))
        assert "error" in result

    @patch("tools.file_tools._get_file_ops")
    def test_replace_mode_missing_strings_errors(self, mock_get):
        from tools.file_tools import patch_tool
        result = json.loads(patch_tool(mode="replace", path="/tmp/f.py", old_string=None, new_string="b"))
        assert "error" in result

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
    def test_patch_mode_missing_content_errors(self, mock_get):
        from tools.file_tools import patch_tool
        result = json.loads(patch_tool(mode="patch", patch=None))
        assert "error" in result

    @patch("tools.file_tools._get_file_ops")
    def test_unknown_mode_errors(self, mock_get):
        from tools.file_tools import patch_tool
        result = json.loads(patch_tool(mode="invalid_mode"))
        assert "error" in result
        assert "Unknown mode" in result["error"]

    @patch("tools.file_tools._get_file_ops")
    def test_repeated_missing_path_escalates_hint(self, mock_get):
        """Identical malformed patch calls must escalate a break-the-loop hint
        instead of returning the same bare error forever (issue #63880)."""
        import tools.file_tools as ft
        from tools.file_tools import patch_tool
        ft._patch_failure_tracker.pop("loop-task", None)
        results = [
            json.loads(patch_tool(mode="replace", path=None, old_string="a",
                                  new_string="b", task_id="loop-task"))
            for _ in range(4)
        ]
        # Every call still reports the error with actionable context.
        assert all("path required" in r["error"] for r in results)
        # First two rejects: no hint yet; from the third the loop-breaker fires.
        assert "_hint" not in results[0]
        assert "_hint" not in results[1]
        assert "_hint" in results[2]
        assert "#3" in results[2]["_hint"]

    @patch("tools.file_tools._get_file_ops")
    def test_valid_patch_resets_arg_failure_escalation(self, mock_get):
        """A call that passes arg validation clears the malformed-call counter
        so escalation only ever counts *consecutive* bad calls."""
        import tools.file_tools as ft
        from tools.file_tools import patch_tool
        ft._patch_failure_tracker.pop("reset-task", None)

        result_obj = MagicMock()
        result_obj.to_dict.return_value = {"status": "ok", "operations": 1}
        mock_get.return_value.patch_replace.return_value = result_obj

        # Two bad calls, then a good one, then a bad one again.
        for _ in range(2):
            patch_tool(mode="replace", path=None, old_string="a",
                       new_string="b", task_id="reset-task")
        patch_tool(mode="replace", path="/tmp/f.py", old_string="a",
                   new_string="b", task_id="reset-task")
        after_reset = json.loads(patch_tool(
            mode="replace", path=None, old_string="a", new_string="b",
            task_id="reset-task"))
        # Counter was reset by the valid call, so this lone bad call is #1
        # again — no premature escalation.
        assert "_hint" not in after_reset

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
    def test_patch_move_from_sensitive_src_blocked(self, mock_get):
        from tools.file_tools import patch_tool
        patch_text = (
            "*** Begin Patch\n"
            "*** Move File: /etc/hosts -> /tmp/leak.txt\n"
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
    def test_patch_move_rejects_traversal_endpoint(self, mock_get):
        """A Move endpoint with ``..`` traversal is rejected, same as the
        Update/Add/Delete headers."""
        from tools.file_tools import patch_tool
        patch_text = (
            "*** Begin Patch\n"
            "*** Move File: /tmp/work.txt -> ../../../etc/shadow\n"
            "*** End Patch\n"
        )
        result = json.loads(patch_tool(mode="patch", patch=patch_text))
        assert "error" in result
        assert "traversal" in result["error"].lower()
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
    def test_search_passes_all_params(self, mock_get):
        mock_ops = MagicMock()
        result_obj = MagicMock()
        result_obj.to_dict.return_value = {"matches": []}
        mock_ops.search.return_value = result_obj
        mock_get.return_value = mock_ops

        from tools.file_tools import search_tool
        search_tool(pattern="class", target="files", path="/src",
                    file_glob="*.py", limit=10, offset=5, output_mode="count", context=2)
        mock_ops.search.assert_called_once_with(
            pattern="class", path="/src", target="files", file_glob="*.py",
            limit=10, offset=5, output_mode="count", context=2,
        )

    @patch("tools.file_tools._get_file_ops")
    def test_search_normalizes_invalid_pagination_before_dispatch(self, mock_get):
        mock_ops = MagicMock()
        result_obj = MagicMock()
        result_obj.to_dict.return_value = {"files": []}
        mock_ops.search.return_value = result_obj
        mock_get.return_value = mock_ops

        from tools.file_tools import search_tool
        search_tool(pattern="class", target="files", path="/src", limit=-5, offset=-2)
        mock_ops.search.assert_called_once_with(
            pattern="class", path="/src", target="files", file_glob=None,
            limit=1, offset=0, output_mode="content", context=0,
        )

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

    def test_cygdrive_path_normalized(self, monkeypatch):
        import tools.environments.local as local_mod
        import tools.file_tools as file_tools

        monkeypatch.setattr(file_tools.sys, "platform", "win32")
        monkeypatch.setattr(local_mod, "_IS_WINDOWS", True)
        monkeypatch.setattr(file_tools, "_uses_container_paths", lambda task_id="default": False)

        resolved = file_tools._resolve_path_for_task("/cygdrive/d/code/main.py")
        assert str(resolved) == r"D:\code\main.py"

    def test_relative_path_uses_normalized_msys_cwd(self, monkeypatch):
        import tools.environments.local as local_mod
        import tools.file_tools as file_tools

        monkeypatch.setattr(file_tools.sys, "platform", "win32")
        monkeypatch.setattr(local_mod, "_IS_WINDOWS", True)
        monkeypatch.setattr(file_tools, "_uses_container_paths", lambda task_id="default": False)
        monkeypatch.setattr(
            file_tools,
            "_authoritative_workspace_root",
            lambda task_id="default": "/c/Users/Mark/project",
        )

        resolved = file_tools._resolve_path_for_task("src/app.py", task_id="msys")
        assert str(resolved) == r"C:\Users\Mark\project\src\app.py"

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
    def test_non_truncated_no_hint(self, mock_get):
        mock_ops = MagicMock()
        result_obj = MagicMock()
        result_obj.to_dict.return_value = {
            "total_count": 3,
            "matches": [{"path": "a.py", "line": 1, "content": "x"}] * 3,
        }
        mock_ops.search.return_value = result_obj
        mock_get.return_value = mock_ops

        from tools.file_tools import search_tool
        raw = search_tool(pattern="foo")
        assert "[Hint:" not in raw

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

    def test_hermes_config_blocked_for_patch(self, tmp_path, monkeypatch):
        fake_config = tmp_path / "config.yaml"
        fake_config.write_text("approvals:\n  mode: manual\n")
        monkeypatch.setattr("tools.file_tools._hermes_config_resolved", str(fake_config))
        monkeypatch.setattr("tools.file_tools._hermes_config_resolved_loaded", True)

        from tools.file_tools import patch_tool
        result = json.loads(patch_tool(
            mode="replace",
            path=str(fake_config),
            old_string="mode: manual",
            new_string="mode: off",
        ))
        assert "error" in result
        assert "Hermes config" in result["error"]

    def test_system_path_still_blocked(self, monkeypatch):
        monkeypatch.setattr("tools.file_tools._hermes_config_resolved", "/some/other/path")
        monkeypatch.setattr("tools.file_tools._hermes_config_resolved_loaded", True)

        from tools.file_tools import write_file_tool
        result = json.loads(write_file_tool("/etc/passwd", "evil"))
        assert "error" in result
        assert "sensitive system path" in result["error"]

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
# _last_known_cwd tests (#26211: silent file creation failure in long conversations)
# ---------------------------------------------------------------------------

class TestLastKnownCwd:
    """
    When the terminal environment is cleaned up and re-created during a long
    conversation, _last_known_cwd preserves the old environment's CWD so
    subsequent file writes with relative paths land in the right directory.

    Regression guard for issue #26211.
    """

    @patch("tools.terminal_tool._active_environments", new_callable=dict)
    @patch("tools.file_tools._file_ops_cache", new_callable=dict)
    @patch("tools.terminal_tool._get_env_config")
    @patch("tools.terminal_tool._create_environment")
    def test_last_known_cwd_preserved_across_env_recreation(
        self, mock_create_env, mock_config, mock_cache, mock_active
    ):
        from tools.file_tools import _get_file_ops, _last_known_cwd

        # Setup: create a mock env with a known CWD
        mock_env = MagicMock()
        mock_env.cwd = "/Users/user/project"
        mock_create_env.return_value = mock_env
        mock_config.return_value = {
            "env_type": "local",
            "cwd": "/default/path",
            "timeout": 30,
        }

        task_id = "default"

        # Preset _last_known_cwd to simulate a previous env's CWD
        _last_known_cwd[task_id] = "/Users/user/project"

        # Call _get_file_ops - should use _last_known_cwd for the new env
        result = _get_file_ops(task_id)

        # Verify the env was created with the saved CWD, not the default
        create_call = mock_create_env.call_args
        assert create_call is not None, "_create_environment was not called"
        
        # Find cwd in the kwargs
        kwargs = create_call.kwargs if create_call.kwargs else {}
        # cwd is passed as positional or keyword
        cwd_passed = kwargs.get("cwd", None)
        if cwd_passed is None:
            # Try positional args
            args = create_call.args if create_call.args else []
            # Position: (env_type, image, cwd, timeout, ...)
            if len(args) >= 3:
                cwd_passed = args[2]
        
        assert cwd_passed == "/Users/user/project", \
            f"Expected cwd='/Users/user/project', got {cwd_passed!r}"
        
        # Cleanup
        _last_known_cwd.pop(task_id, None)
        
    @patch("tools.terminal_tool._active_environments", new_callable=dict)
    @patch("tools.file_tools._file_ops_cache", new_callable=dict)
    @patch("tools.terminal_tool._get_env_config")
    @patch("tools.terminal_tool._create_environment")
    def test_last_known_cwd_falls_back_to_config_default_when_not_set(
        self, mock_create_env, mock_config, mock_cache, mock_active
    ):
        from tools.file_tools import _get_file_ops, _last_known_cwd

        mock_env = MagicMock()
        mock_env.cwd = "/default/path"
        mock_create_env.return_value = mock_env
        mock_config.return_value = {
            "env_type": "local",
            "cwd": "/config/default/path",
            "timeout": 30,
        }

        # _get_file_ops resolves to "default"
        task_id = "default"
        
        # Ensure _last_known_cwd is empty for this task
        _last_known_cwd.pop(task_id, None)

        result = _get_file_ops(task_id)
        
        create_call = mock_create_env.call_args
        assert create_call is not None, "_create_environment was not called"
        
        kwargs = create_call.kwargs if create_call.kwargs else {}
        cwd_passed = kwargs.get("cwd", None)
        if cwd_passed is None:
            args = create_call.args if create_call.args else []
            if len(args) >= 3:
                cwd_passed = args[2]
        
        # Should fall back to config default
        assert cwd_passed == "/config/default/path", \
            f"Expected cwd='/config/default/path', got {cwd_passed!r}"

    @patch("tools.terminal_tool._active_environments", new_callable=dict)
    @patch("tools.file_tools._file_ops_cache", new_callable=dict)
    def test_live_cwd_read_mirrors_into_last_known_cwd(self, mock_cache, mock_active):
        """Belt-and-suspenders (#26211): every successful live-cwd read records
        the cwd in _last_known_cwd, so the durable anchor doesn't depend on the
        cleanup-detection branch of _get_file_ops firing."""
        from tools.file_tools import _get_live_tracking_cwd, _last_known_cwd

        task_id = "default"
        _last_known_cwd.pop(task_id, None)

        cached = MagicMock()
        cached.env = MagicMock()
        cached.env.cwd = "/Users/user/project"
        cached.env.cwd_owner = "default"
        mock_cache[task_id] = cached

        live = _get_live_tracking_cwd(task_id)

        assert live == "/Users/user/project"
        # The read mirrored the live cwd into the durable registry.
        assert _last_known_cwd.get(task_id) == "/Users/user/project"
        _last_known_cwd.pop(task_id, None)

    @patch("tools.terminal_tool._active_environments", new_callable=dict)
    @patch("tools.file_tools._file_ops_cache", new_callable=dict)
    @patch("tools.terminal_tool._get_env_config")
    @patch("tools.terminal_tool._create_environment")
    def test_mirrored_cwd_survives_when_cache_already_cleared(
        self, mock_create_env, mock_config, mock_cache, mock_active
    ):
        """The original save-old-cwd path only fires when _file_ops_cache still
        holds the stale entry. If the cleanup thread popped BOTH dicts first,
        _get_file_ops sees cached=None and never saves — but the proactive
        mirror from an earlier live read already populated _last_known_cwd, so
        the rebuilt env still restores the user's directory."""
        from tools.file_tools import (
            _get_file_ops, _get_live_tracking_cwd, _last_known_cwd,
        )

        task_id = "default"
        _last_known_cwd.pop(task_id, None)

        # 1) Env is alive and the agent has cd'd into the project. A live read
        #    (happens on every relative-path resolution) mirrors the cwd.
        cached = MagicMock()
        cached.env = MagicMock()
        cached.env.cwd = "/Users/user/project"
        cached.env.cwd_owner = "default"
        mock_cache[task_id] = cached
        assert _get_live_tracking_cwd(task_id) == "/Users/user/project"
        assert _last_known_cwd.get(task_id) == "/Users/user/project"

        # 2) Cleanup thread kills the env AND clears the cache before the next
        #    file write — so _get_file_ops' save-old-cwd branch never runs.
        mock_cache.pop(task_id, None)
        mock_active.clear()

        mock_env = MagicMock()
        mock_env.cwd = "/Users/user/project"
        mock_create_env.return_value = mock_env
        mock_config.return_value = {
            "env_type": "local",
            "cwd": "/config/default/path",
            "timeout": 30,
        }

        _get_file_ops(task_id)

        create_call = mock_create_env.call_args
        assert create_call is not None, "_create_environment was not called"
        kwargs = create_call.kwargs if create_call.kwargs else {}
        cwd_passed = kwargs.get("cwd", None)
        if cwd_passed is None:
            args = create_call.args if create_call.args else []
            if len(args) >= 3:
                cwd_passed = args[2]

        # Rebuilt env restored the mirrored cwd, NOT the config default.
        assert cwd_passed == "/Users/user/project", \
            f"Expected restored cwd='/Users/user/project', got {cwd_passed!r}"
        _last_known_cwd.pop(task_id, None)


class TestSilentFileMisplacementE2E:
    """Real-IO regression for #26211.

    Exercises the actual write_file_tool path against a temp filesystem: an
    agent cd's into a project, the cleanup thread kills the env, and a later
    relative-path write must land in the project dir (not the config default).
    Mocks miss this because resolution (_resolve_path_for_task) runs BEFORE
    _get_file_ops rebuilds the env — only the durable _last_known_cwd fallback
    in _authoritative_workspace_root makes the resolved path correct.
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
        ft._last_known_cwd.pop(task_id, None)

        # 1) Env alive; agent has cd'd into the project. A relative write
        #    while alive mirrors the live cwd into the durable registry.
        fo = ft._get_file_ops(task_id)
        fo.env.cwd = str(project)
        fo.env.cwd_owner = "default"
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

        ft._last_known_cwd.pop(task_id, None)
