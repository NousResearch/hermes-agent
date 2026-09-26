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

    def test_patch_whole_line_block_with_trailing_newline_on_crlf(self, hermes_home, tmp_path):
        """Regression for #124166: models copy whole lines, so old_string ends
        with a newline. On a CRLF file the raw text can never contain that
        bare-LF block, and the line-based strategies split the trailing
        newline into a phantom empty line — the edit was refused with a
        "Did you mean" hint quoting the very lines that were sent."""
        from tools.file_tools import _handle_patch

        target = tmp_path / "a.py"
        target.write_bytes(b"def load():\r\n    cfg = read()\r\n    validate(cfg)\r\n    return cfg\r\n")

        result = _handle_patch(
            {
                "mode": "replace",
                "path": str(target),
                "old_string": "    cfg = read()\n    validate(cfg)\n",
                "new_string": "    cfg = read(strict=True)\n    validate(cfg)\n",
            },
            task_id="crlf_patch_4",
        )
        d = json.loads(result)
        assert not d.get("error"), d

        raw = target.read_bytes()
        assert raw == b"def load():\r\n    cfg = read(strict=True)\r\n    validate(cfg)\r\n    return cfg\r\n"

    def test_patch_replace_all_on_crlf_reaches_every_copy(self, hermes_home, tmp_path):
        """Regression for #124166: the phantom-empty-line window only matched
        where a blank line followed the block, so replace_all silently changed
        some copies and still reported success."""
        from tools.file_tools import _handle_patch

        target = tmp_path / "b.ini"
        target.write_bytes(b"debug = true\r\nport = 1\r\ndebug = true\r\n\r\nhost = x\r\ndebug = true\r\nend = 1\r\n")

        result = _handle_patch(
            {
                "mode": "replace",
                "path": str(target),
                "replace_all": True,
                "old_string": "debug = true\n",
                "new_string": "debug = false\n",
            },
            task_id="crlf_patch_5",
        )
        d = json.loads(result)
        assert not d.get("error"), d

        raw = target.read_bytes()
        assert raw.count(b"debug = false") == 3
        assert raw.count(b"debug = true") == 0
        assert _bare_lf_count(raw) == 0

    def test_patch_ambiguous_whole_line_block_on_crlf_is_refused(self, hermes_home, tmp_path):
        """Regression for #124166: an old_string appearing twice is refused on
        an LF file ("Found 2 matches"); on a CRLF file the phantom-empty-line
        window matched only one copy and edited it silently."""
        from tools.file_tools import _handle_patch

        target = tmp_path / "c.py"
        target.write_bytes(b"retries = 3\r\nconnect()\r\n\r\nretries = 3\r\n\r\nrun()\r\n")

        result = _handle_patch(
            {
                "mode": "replace",
                "path": str(target),
                "old_string": "retries = 3\n",
                "new_string": "retries = 5\n",
            },
            task_id="crlf_patch_6",
        )
        d = json.loads(result)
        assert "Found 2 matches" in d.get("error", ""), d

        raw = target.read_bytes()
        assert raw == b"retries = 3\r\nconnect()\r\n\r\nretries = 3\r\n\r\nrun()\r\n"


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
