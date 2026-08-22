"""Coalesced read_file execs: probe + page, not 4–5 sequential shells.

Live LocalEnvironment cases run wherever bash is available (Linux CI and
native Windows with Git Bash). Parser tests pin the wire format so a
dropped ``wc -l`` / ``tail -c 1`` cannot hide behind an updated fixture.
"""

from __future__ import annotations

import os
import stat

import pytest

from tools.environments.local import LocalEnvironment
from tools.file_operations import (
    IMAGE_EXTENSIONS,
    ShellFileOperations,
    _decode_base64_sample,
    _new_read_sentinel,
    _parse_page_header,
    _parse_size_header,
    _split_sentinel_payload,
)


@pytest.fixture
def ops():
    return ShellFileOperations(LocalEnvironment())


class TestWireFormatParsers:
    def test_split_keeps_body_after_first_sentinel(self):
        token = _new_read_sentinel()
        body = f"hello\n{token}\nstill body"
        header, payload = _split_sentinel_payload(f"12\n{token}\n{body}", token)
        assert header.strip() == "12"
        assert payload == body

    def test_parse_size_header_last_int_wins_junk_prefix(self):
        assert _parse_size_header("noise\n42") == 42
        assert _parse_size_header("5") == 5
        assert _parse_size_header("") == 0

    def test_parse_page_header_independent_statuses(self):
        fields = _parse_page_header("sed_rc=0\nwc_rc=1\ntotal=\ntail_ran=0\ntail_nl=1\n")
        assert fields["sed_rc"] == "0"
        assert fields["wc_rc"] == "1"
        assert fields["tail_ran"] == "0"

    def test_decode_base64_sample_roundtrip(self):
        import base64
        payload = b"abc\x00def"
        assert _decode_base64_sample(base64.b64encode(payload).decode()) == payload
        assert _decode_base64_sample("") == b""
        assert _decode_base64_sample("!!!!") is None


class TestPageScriptShape:
    def test_page_script_still_runs_wc_and_conditional_tail(self, ops):
        script = ops._page_text_file_cmd(
            "/tmp/notes.txt",
            offset=1,
            end_line=2000,
            line_clamp_bytes=8001,
            sentinel="__HERMES_READ_deadbeefdeadbeef__",
        )
        assert "sed -n '1,2000p'" in script
        assert "wc -l <" in script
        assert "tail -c 1" in script
        assert "tail_ran=1" in script
        assert "set -o pipefail" in script

    def test_probe_script_does_not_head_not_regular_paths(self, ops):
        script = ops._probe_regular_file_cmd(
            "/tmp/notes.txt",
            "__HERMES_READ_deadbeefdeadbeef__",
        )
        assert "[ -f " in script
        assert "head -c 1000" in script
        # head is inside the regular-file arm, after [ -f ]
        regular_arm, _, rest = script.partition("elif [ -e ")
        assert "head -c 1000" in regular_arm
        assert "head -c" not in rest
        assert "size_rc" in regular_arm
        assert "sample_rc" in regular_arm
        assert "exit 0" in regular_arm


class TestProbeStatusIsolation:
    def test_failed_sample_is_not_file_not_found(self):
        from unittest.mock import MagicMock

        from tools.file_operations import ShellFileOperations

        env = MagicMock()
        env.cwd = "/tmp"
        calls = []

        def execute(command, **kwargs):
            calls.append(command)
            sentinel = None
            import re
            match = re.search(r"__HERMES_READ_[0-9a-f]+__", command)
            if match:
                sentinel = match.group(0)
            if command.startswith("if [ -f ") and sentinel:
                return {
                    "output": f"size=12\nsize_rc=0\nsample_rc=1\n{sentinel}\n",
                    "returncode": 0,
                }
            if command.startswith("head -c") and "base64" in command:
                import base64 as b64
                return {
                    "output": b64.b64encode(b"hello\n").decode(),
                    "returncode": 0,
                }
            if "sed -n" in command and sentinel:
                return {
                    "output": (
                        f"sed_rc=0\nwc_rc=0\ntotal=1\ntail_ran=1\ntail_nl=1\n"
                        f"{sentinel}\nhello\n"
                    ),
                    "returncode": 0,
                }
            return {"output": "", "returncode": 0}

        env.execute.side_effect = execute
        result = ShellFileOperations(env).read_file("/tmp/notes.txt")
        assert result.error is None
        assert "File not found" not in (result.error or "")
        assert any(c.startswith("head -c") and "base64" in c for c in calls)

    def test_failed_size_is_read_error_not_empty_file(self):
        from unittest.mock import MagicMock
        import re

        from tools.file_operations import ShellFileOperations

        env = MagicMock()
        env.cwd = "/tmp"

        def execute(command, **kwargs):
            match = re.search(r"__HERMES_READ_[0-9a-f]+__", command)
            sentinel = match.group(0) if match else "x"
            if command.startswith("if [ -f "):
                return {
                    "output": f"size=0\nsize_rc=1\nsample_rc=1\n{sentinel}\n",
                    "returncode": 0,
                }
            return {"output": "", "returncode": 0}

        env.execute.side_effect = execute
        result = ShellFileOperations(env).read_file("/tmp/secret.txt")
        assert result.error
        assert "Failed to read file" in result.error
        assert result.hint != "File is empty (0 bytes)."


class TestLiveCoalescedRead:
    def test_text_path_uses_two_execs(self, tmp_path, ops, monkeypatch):
        target = tmp_path / "notes.txt"
        target.write_text("alpha\nbeta\ngamma\n", encoding="utf-8")
        calls = []
        original = ops._exec

        def counted(command, *args, **kwargs):
            calls.append(command)
            return original(command, *args, **kwargs)

        monkeypatch.setattr(ops, "_exec", counted)
        result = ops.read_file(str(target))
        assert result.error is None
        assert "1|alpha" in result.content
        assert "3|gamma" in result.content
        assert len(calls) == 2
        assert "sed -n" in calls[1]
        assert "wc -l <" in calls[1]

    def test_no_trailing_newline_stripped(self, tmp_path, ops):
        target = tmp_path / "nonl.txt"
        target.write_text("a\nb", encoding="utf-8")
        result = ops.read_file(str(target))
        assert result.error is None
        assert result.content == "1|a\n2|b"

    def test_truncated_page_still_reports_total(self, tmp_path, ops, monkeypatch):
        target = tmp_path / "long.txt"
        target.write_text("".join(f"line {i}\n" for i in range(1, 30)), encoding="utf-8")
        calls = []
        original = ops._exec

        def counted(command, *args, **kwargs):
            calls.append(command)
            return original(command, *args, **kwargs)

        monkeypatch.setattr(ops, "_exec", counted)
        result = ops.read_file(str(target), offset=1, limit=10)
        assert result.error is None
        assert result.truncated is True
        assert result.total_lines == 29
        assert "Use offset=11" in (result.hint or "")
        # Skip condition lives in the page script; tail probe must be present
        # as a command but gated on total <= end_line (here 10).
        assert "tail -c 1" in calls[1]
        assert "-le 10" in calls[1]

    def test_image_never_runs_page_script(self, tmp_path, ops, monkeypatch):
        ext = next(iter(IMAGE_EXTENSIONS))
        target = tmp_path / f"pic{ext}"
        target.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 32)
        calls = []
        original = ops._exec

        def counted(command, *args, **kwargs):
            calls.append(command)
            return original(command, *args, **kwargs)

        monkeypatch.setattr(ops, "_exec", counted)
        result = ops.read_file(str(target))
        assert result.is_image is True
        assert len(calls) == 1
        assert "sed -n" not in calls[0]
        assert "wc -l <" not in calls[0]

    def test_directory_errors_without_paging(self, tmp_path, ops, monkeypatch):
        calls = []
        original = ops._exec

        def counted(command, *args, **kwargs):
            calls.append(command)
            return original(command, *args, **kwargs)

        monkeypatch.setattr(ops, "_exec", counted)
        result = ops.read_file(str(tmp_path))
        assert result.error
        assert "not a regular file" in result.error
        assert len(calls) == 1
        assert "sed -n" not in calls[0]

    @pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="no mkfifo on this host")
    def test_fifo_errors_without_blocking(self, tmp_path, ops):
        fifo = tmp_path / "pipe.fifo"
        os.mkfifo(fifo)
        # Drop write bit so a mistaken open cannot hang the test waiting
        # for a writer. The size probe must not open it at all.
        os.chmod(fifo, stat.S_IRUSR)
        result = ops.read_file(str(fifo))
        assert result.error
        assert "not a regular file" in result.error
