"""Tests for the file tools module (schema, handler wiring, error paths).

Tests verify tool schemas, handler dispatch, validation logic, and error
handling without requiring a running terminal environment.
"""

import json
import logging
from unittest.mock import MagicMock, patch

from tools.file_tools import (
    READ_FILE_SCHEMA,
    WRITE_FILE_SCHEMA,
    PATCH_SCHEMA,
    SEARCH_FILES_SCHEMA,
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


class TestPathLevelLoopDetection:
    """Tests for the path-level read loop guard (issue #14991).

    The guard counts *consecutive reads that surface no new lines* of the
    same file — overlapping or re-read regions — rather than the raw read
    count.  Legitimate forward pagination through a large file (advancing,
    non-overlapping windows) always surfaces new lines and is therefore
    never warned or blocked.  Thresholds in tools.file_tools: warn after 4
    consecutive no-progress reads, hard-block after 6.
    """

    def _make_mock_ops(self, content="line1\nline2", total_lines=2):
        mock_ops = MagicMock()
        result_obj = MagicMock()
        result_obj.content = content
        result_obj.to_dict.return_value = {
            "content": content, "total_lines": total_lines,
        }
        mock_ops.read_file.return_value = result_obj
        return mock_ops

    @patch("tools.file_tools.os.path.getmtime", return_value=1000.0)
    @patch("tools.file_tools._get_file_ops")
    def test_forward_pagination_never_blocks(self, mock_get, mock_mtime):
        """Advancing, non-overlapping windows always surface new lines, so
        sequential pagination through a large file is never warned/blocked."""
        from tools.file_tools import read_file_tool, _read_tracker
        mock_get.return_value = self._make_mock_ops()
        _read_tracker.clear()

        # 12 non-overlapping 500-line windows — far past the old count-based
        # block at 8.  Each advances into brand-new content.
        for i in range(12):
            result = json.loads(
                read_file_tool("/tmp/big.txt", offset=i * 500 + 1, limit=500)
            )
            assert "_warning" not in result, f"unexpected warning at read {i+1}"
            assert "BLOCKED" not in result.get("error", ""), \
                f"unexpected block at read {i+1}"

    @patch("tools.file_tools.os.path.getmtime", return_value=1000.0)
    @patch("tools.file_tools._get_file_ops")
    def test_overlapping_rereads_trigger_warning(self, mock_get, mock_mtime):
        """Re-reading already-covered regions (no new lines) warns after the
        4th consecutive no-progress read."""
        from tools.file_tools import read_file_tool, _read_tracker
        mock_get.return_value = self._make_mock_ops()
        _read_tracker.clear()

        # Establish coverage of lines [1, 501).
        read_file_tool("/tmp/t.txt", offset=1, limit=500)

        # Sub-region re-reads: distinct (offset,limit) keys (so no dedup
        # short-circuit) but every window is already covered -> no new lines.
        for lim in (400, 300, 200):           # no-progress reads 1, 2, 3
            result = json.loads(read_file_tool("/tmp/t.txt", offset=1, limit=lim))
            assert "_warning" not in result

        # 4th consecutive no-progress read -> warning.
        result = json.loads(read_file_tool("/tmp/t.txt", offset=1, limit=100))
        assert "_warning" in result
        assert "BLOCKED" not in result.get("error", "")

    @patch("tools.file_tools.os.path.getmtime", return_value=1000.0)
    @patch("tools.file_tools._get_file_ops")
    def test_overlapping_rereads_trigger_block(self, mock_get, mock_mtime):
        """6 consecutive no-progress reads escalate to a hard block."""
        from tools.file_tools import read_file_tool, _read_tracker
        mock_get.return_value = self._make_mock_ops()
        _read_tracker.clear()

        read_file_tool("/tmp/t.txt", offset=1, limit=500)   # coverage [1, 501)
        result = None
        for lim in (450, 400, 350, 300, 250, 200):          # 6 no-progress reads
            result = json.loads(read_file_tool("/tmp/t.txt", offset=1, limit=lim))

        assert "error" in result
        assert "BLOCKED" in result["error"]

    @patch("tools.file_tools.os.path.getmtime", return_value=1000.0)
    @patch("tools.file_tools._get_file_ops")
    def test_different_file_resets_no_progress_counter(self, mock_get, mock_mtime):
        """Reading a different file resets the no-progress counter to 0."""
        from tools.file_tools import read_file_tool, _read_tracker
        mock_get.return_value = self._make_mock_ops()
        _read_tracker.clear()

        # Build up no-progress reads on file A (counter -> 3).
        read_file_tool("/tmp/fileA.txt", offset=1, limit=500)
        for lim in (400, 300, 200):
            read_file_tool("/tmp/fileA.txt", offset=1, limit=lim)
        assert _read_tracker["default"]["path_consecutive"] == 3

        # Reading file B switches the tracked path and zeroes the counter.
        read_file_tool("/tmp/fileB.txt", offset=1, limit=500)
        assert _read_tracker["default"]["path_consecutive"] == 0

        # Re-reading file A (a fresh key, so no dedup) starts from 0, not 3.
        read_file_tool("/tmp/fileA.txt", offset=1, limit=450)
        assert _read_tracker["default"]["path_consecutive"] == 0

    @patch("tools.file_tools.os.path.getmtime", return_value=1000.0)
    @patch("tools.file_tools._get_file_ops")
    def test_notify_other_tool_call_resets_no_progress_counter(self, mock_get, mock_mtime):
        """A non-read tool call resets the no-progress counter, so a
        following overlapping read does not warn."""
        from tools.file_tools import read_file_tool, notify_other_tool_call, _read_tracker
        mock_get.return_value = self._make_mock_ops()
        _read_tracker.clear()

        # Drive the no-progress counter up to 3 on the same file.
        read_file_tool("/tmp/test.txt", offset=1, limit=500)
        for lim in (400, 300, 200):
            read_file_tool("/tmp/test.txt", offset=1, limit=lim)
        assert _read_tracker["default"]["path_consecutive"] == 3

        # An intervening non-read tool call breaks the loop.
        notify_other_tool_call("default")
        assert _read_tracker["default"]["path_consecutive"] == 0

        # A following overlapping read (distinct key) starts fresh: no warning,
        # even though it covers already-seen lines.  Regression guard: if the
        # reset were dropped, this would be the 4th no-progress read and warn.
        result = json.loads(read_file_tool("/tmp/test.txt", offset=1, limit=350))
        assert "_warning" not in result
        assert "BLOCKED" not in result.get("error", "")

    @patch("tools.file_tools.os.path.getmtime", return_value=1000.0)
    @patch("tools.file_tools._get_file_ops")
    def test_reset_file_dedup_resets_no_progress_counter(self, mock_get, mock_mtime):
        """reset_file_dedup clears the path-level no-progress counter."""
        from tools.file_tools import read_file_tool, reset_file_dedup, _read_tracker
        mock_get.return_value = self._make_mock_ops()
        _read_tracker.clear()

        read_file_tool("/tmp/test.txt", offset=1, limit=500)
        for lim in (400, 300, 200):
            read_file_tool("/tmp/test.txt", offset=1, limit=lim)
        assert _read_tracker["default"]["path_consecutive"] == 3

        # Context compression resets dedup + path-level loop state.
        reset_file_dedup("default")
        assert _read_tracker["default"]["path_consecutive"] == 0

        # Fresh key (350 was never read above) so this is a real read; it
        # covers already-seen lines but starts from a reset counter -> no warn.
        result = json.loads(read_file_tool("/tmp/test.txt", offset=1, limit=350))
        assert "_warning" not in result
        assert "BLOCKED" not in result.get("error", "")
