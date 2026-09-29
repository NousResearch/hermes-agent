"""Tests for CRLF line-ending preservation in write_file and patch.

Without this, the agent silently normalizes Windows-line-ending files
to LF whenever it edits them — and patch produces a mixed-ending file
when only a substituted region changes (the rest of the file keeps its
CRLF endings while the replacement is LF-only).

See issue #507 (Roo Code deep-dive, item 2c).
"""

import json

import pytest


@pytest.fixture
def hermes_home(monkeypatch, tmp_path):
    """Isolate HERMES_HOME so the tests don't pollute the real config.

    Also clears module-level caches (file_ops, active_environments,
    file-staleness state) after the test so subsequent tests in the
    same pytest process aren't affected by our shell-out side effects
    (real file_ops and terminal environments get created under
    task_id='default' via _resolve_container_task_id).
    """
    home = tmp_path / "hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    yield home
    # Cleanup: drop the cached file_ops and active environment so the
    # next test sees a fresh state.  Without this, _get_live_tracking_cwd
    # returns the stale cwd from this test's ops and breaks tests like
    # test_resolve_path that rely on TERMINAL_CWD env var.
    try:
        from tools.file_tools import clear_file_ops_cache
        from tools.file_tools_read_tracking import _read_tracker_lock, _read_tracker
        clear_file_ops_cache()
        with _read_tracker_lock:
            _read_tracker.clear()
    except Exception:
        pass
    try:
        from tools.terminal_tool import _active_environments, _env_lock
        with _env_lock:
            _active_environments.clear()
    except Exception:
        pass


def _crlf_count(b: bytes) -> int:
    return b.count(b"\r\n")


def _bare_lf_count(b: bytes) -> int:
    return b.count(b"\n") - b.count(b"\r\n")


class TestPatchCRLFPreservation:
    def test_patch_on_crlf_file_stays_pure_crlf(self, hermes_home, tmp_path):
        """LLM sends LF old/new; file has CRLF.  Result must be all CRLF,
        no mixed endings."""
        from tools.file_tools import _handle_patch

        target = tmp_path / "config.ini"
        target.write_bytes(b"[a]\r\nkey=1\r\n\r\n[b]\r\nkey=2\r\n")

        result = _handle_patch(
            {
                "mode": "replace",
                "path": str(target),
                "old_string": "key=1",
                "new_string": "key=99",
            },
            task_id="crlf_patch_1",
        )
        d = json.loads(result)
        assert not d.get("error"), d

        raw = target.read_bytes()
        assert _bare_lf_count(raw) == 0, (
            f"Mixed line endings after patch: {raw!r}"
        )
        # Same number of line breaks as before; just the value swapped.
        assert _crlf_count(raw) == 5
        assert b"key=99\r\n" in raw


    def test_patch_multiline_replacement_on_crlf(self, hermes_home, tmp_path):
        """Multi-line new_string with bare LFs should be CRLF-converted
        before write."""
        from tools.file_tools import _handle_patch

        target = tmp_path / "f.py"
        target.write_bytes(b"def foo():\r\n    return 1\r\n")

        result = _handle_patch(
            {
                "mode": "replace",
                "path": str(target),
                "old_string": "def foo():\n    return 1",
                "new_string": "def foo():\n    x = 1\n    return x",
            },
            task_id="crlf_patch_3",
        )
        d = json.loads(result)
        assert not d.get("error"), d

        raw = target.read_bytes()
        assert _bare_lf_count(raw) == 0, (
            f"Mixed endings after multi-line patch: {raw!r}"
        )
        assert raw == b"def foo():\r\n    x = 1\r\n    return x\r\n"


# An edit writes the whole file back, so the ending of a line it did not touch is the
# file's own data: a CRLF inside a fixture, a lone CR in a captured progress bar, the
# LF lines of a file that mixes both.
_UNTOUCHED_ENDINGS = {
    "lf_file_one_crlf_line": b"HEADER\nfixture = 'a\r\nb'\nkeep\nx = 1\n",
    "lf_file_crlf_past_the_sample": b"HEADER\n# " + b"y" * 5000 + b"\nfixture = 'a\r\nb'\nx = 1\n",
    "lf_file_lone_cr": b"HEADER\nbar 10%\rbar 100%\nx = 1\n",
    "crlf_file_lf_line": b"HEADER\r\nkeep\nalso\r\nx = 1\r\n",
    "crlf_file_lone_cr": b"HEADER\r\nbar 10%\rbar 100%\r\nx = 1\r\n",
}


def _edit(mode, target, old, new, task_id):
    from tools.file_tools import _handle_patch

    if mode == "replace":
        args = {"mode": "replace", "path": str(target), "old_string": old, "new_string": new}
    else:
        removed = "".join(f"-{line}\n" for line in old.split("\n"))
        added = "".join(f"+{line}\n" for line in new.split("\n"))
        args = {"mode": "patch",
                "patch": f"*** Begin Patch\n*** Update File: {target}\n@@\n{removed}{added}*** End Patch"}
    result = json.loads(_handle_patch(args, task_id=task_id))
    assert not result.get("error"), result


class TestEditKeepsUntouchedLineEndings:
    @pytest.mark.parametrize("mode", ["replace", "v4a"])
    @pytest.mark.parametrize("case", sorted(_UNTOUCHED_ENDINGS))
    def test_one_line_edit_changes_only_that_line(self, hermes_home, tmp_path, mode, case):
        original = _UNTOUCHED_ENDINGS[case]
        target = tmp_path / "f.txt"
        target.write_bytes(original)

        _edit(mode, target, "x = 1", "x = 2", f"endings_{mode}_{case}")

        assert target.read_bytes() == original.replace(b"x = 1", b"x = 2")

    @pytest.mark.parametrize("mode", ["replace", "v4a", "v4a_distant_hunks"])
    def test_new_lines_take_the_ending_of_the_lines_they_replace(self, hermes_home, tmp_path, mode, monkeypatch):
        target = tmp_path / "f.txt"
        if mode != "v4a_distant_hunks":
            target.write_bytes(b"keep\nx = 1\r\ny = 2\r\nlast")
            _edit(mode, target, "x = 1\ny = 2", "x = 9\nnew\ny = 8", f"endings_multi_{mode}")
            assert target.read_bytes() == b"keep\nx = 9\r\nnew\r\ny = 8\r\nlast"
            return

        # Past the match limit, the lines between two hunks keep their own endings.
        from tools import file_operations_common
        from tools.file_tools import _handle_patch

        monkeypatch.setattr(file_operations_common, "_LINE_MATCH_LIMIT", 2)
        target.write_bytes(b"top = 1\na\r\nb\r\nc\r\nend = 1\r\n")
        patch = (f"*** Begin Patch\n*** Update File: {target}\n@@\n-top = 1\n+top = 2\n a\n"
                 "@@\n c\n-end = 1\n+end = 2\n*** End Patch")
        result = json.loads(_handle_patch({"mode": "patch", "patch": patch}, task_id="endings_span"))
        assert not result.get("error"), result
        assert target.read_bytes() == b"top = 2\na\r\nb\r\nc\r\nend = 2\r\n"


class TestWriteFileCRLFPreservation:
    def test_overwrite_crlf_file_with_lf_content_preserves_crlf(
        self, hermes_home, tmp_path
    ):
        """The agent typically sends bare-LF content; if the file existed
        with CRLF, the write should convert to CRLF rather than silently
        flipping the endings."""
        from tools.file_tools import _handle_write_file, read_file_tool

        target = tmp_path / "config.bat"
        target.write_bytes(b"@echo off\r\nset X=1\r\n")
        # write_file refuses to overwrite an existing file the task never read.
        assert "error" not in json.loads(read_file_tool(str(target), task_id="crlf_write_1"))

        result = _handle_write_file(
            {
                "path": str(target),
                "content": "@echo off\nset X=99\nset Y=42\n",
            },
            task_id="crlf_write_1",
        )
        d = json.loads(result)
        assert "error" not in d, d

        raw = target.read_bytes()
        assert _bare_lf_count(raw) == 0, (
            f"CRLF file got normalized to LF: {raw!r}"
        )
        assert _crlf_count(raw) == 3


    def test_overwrite_lf_file_stays_lf(self, hermes_home, tmp_path):
        """Pre-existing LF file should not get spurious CRLFs."""
        from tools.file_tools import _handle_write_file, read_file_tool

        target = tmp_path / "lf.txt"
        target.write_bytes(b"line1\nline2\n")
        assert "error" not in json.loads(read_file_tool(str(target), task_id="crlf_write_3"))

        result = _handle_write_file(
            {"path": str(target), "content": "X\nY\nZ\n"},
            task_id="crlf_write_3",
        )
        d = json.loads(result)
        assert "error" not in d, d

        raw = target.read_bytes()
        assert _crlf_count(raw) == 0
        assert raw == b"X\nY\nZ\n"


class TestLineEndingHelpers:
    """Direct unit tests for the pure helpers — easier to debug than the
    integration tests above."""


    def test_normalize_to_crlf_idempotent(self):
        from tools.file_operations import _normalize_line_endings

        once = _normalize_line_endings("a\nb\n", "\r\n")
        twice = _normalize_line_endings(once, "\r\n")
        assert once == twice == "a\r\nb\r\n"
