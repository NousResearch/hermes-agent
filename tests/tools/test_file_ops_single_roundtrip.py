"""``read_file`` / ``write_file`` cost one shell round-trip, not four.

Real ``LocalEnvironment`` against ``tmp_path`` (no mocks), with a spy on
``env.execute`` counting round-trips. The cases below are exactly the ones
that used to need their own probe (existence, size, binary sample, page,
line count, trailing newline), so each proves the compound reply carries
that answer.
"""

import os
import sys
import threading
import unicodedata
from unittest.mock import patch

import pytest

from tools.environments.local import LocalEnvironment
from tools.file_operations import ExecuteResult, ShellFileOperations

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="POSIX shell probes")

READ_PROBE_MARK = "__HERMES_RF_"


@pytest.fixture
def shell(tmp_path, monkeypatch):
    """(ops, calls): file ops over a real local shell, every execute recorded."""
    # Pin the shell path even where a native fast path exists.
    monkeypatch.setenv("HERMES_NATIVE_FILE_READ", "0")
    env = LocalEnvironment(cwd=str(tmp_path))
    calls = []
    real_execute = env.execute

    def spy(command, *args, **kwargs):
        calls.append(command)
        return real_execute(command, *args, **kwargs)

    env.execute = spy
    return ShellFileOperations(env, cwd=str(tmp_path)), calls


def _write(tmp_path, name, data: bytes):
    p = tmp_path / name
    p.write_bytes(data)
    return str(p)


class TestReadFileOneRoundTrip:
    def test_text_read_is_one_round_trip(self, shell, tmp_path):
        ops, calls = shell
        p = _write(tmp_path, "a.txt", b"one\ntwo\nthree\n")
        r = ops.read_file(p)
        assert len(calls) == 1 and READ_PROBE_MARK in calls[0]
        assert r.error is None
        # ``_add_line_numbers`` numbers the empty tail after the final
        # newline: long-standing behaviour, preserved byte for byte.
        assert r.content == "1|one\n2|two\n3|three\n4|"
        assert (r.total_lines, r.file_size, r.truncated) == (3, 14, False)

    def test_no_trailing_newline_needs_no_extra_probe(self, shell, tmp_path):
        ops, calls = shell
        p = _write(tmp_path, "b.txt", b"a\nb")
        r = ops.read_file(p)
        assert len(calls) == 1
        # ``cut`` newline-terminates the last line; the artifact is stripped
        # from the same reply that used to need a fifth ``tail -c 1`` call.
        assert r.content == "1|a\n2|b"
        assert r.total_lines == 1  # wc -l semantics, unchanged

    def test_pagination_window_and_hint(self, shell, tmp_path):
        ops, calls = shell
        p = _write(tmp_path, "c.txt", b"".join(b"l%d\n" % i for i in range(1, 11)))
        r = ops.read_file(p, offset=3, limit=2)
        assert len(calls) == 1
        assert r.content == "3|l3\n4|l4\n5|"
        assert r.truncated is True and r.total_lines == 10
        assert "offset=5" in r.hint

    def test_offset_past_eof_note(self, shell, tmp_path):
        ops, calls = shell
        p = _write(tmp_path, "c.txt", b"".join(b"l%d\n" % i for i in range(1, 6)))
        r = ops.read_file(p, offset=50)
        assert len(calls) == 1
        assert r.content == "" and r.error is None
        assert "beyond the end" in r.hint and "5" in r.hint

    def test_empty_file(self, shell, tmp_path):
        ops, calls = shell
        r = ops.read_file(_write(tmp_path, "e.txt", b""))
        assert len(calls) == 1
        assert r.error is None and r.content == "" and r.total_lines == 0
        assert "empty" in r.hint

    def test_bom_stripped_on_first_page(self, shell, tmp_path):
        ops, calls = shell
        r = ops.read_file(_write(tmp_path, "f.txt", "﻿hello\n".encode("utf-8")))
        assert len(calls) == 1
        assert r.content == "1|hello\n2|"

    def test_crlf_bytes_survive(self, shell, tmp_path):
        ops, calls = shell
        r = ops.read_file(_write(tmp_path, "g.txt", b"x\r\ny\r\n"))
        assert r.content == "1|x\r\n2|y\r\n3|"

    def test_long_line_clamped_and_marked(self, shell, tmp_path):
        ops, calls = shell
        r = ops.read_file(_write(tmp_path, "L.txt", b"a" * 9000 + b"\nshort\n"))
        assert len(calls) == 1
        first, second, tail = r.content.split("\n")
        assert first.endswith("... [truncated]") and len(first) < 9000
        assert second == "2|short" and tail == "3|"

    def test_relative_path_resolves_against_env_cwd(self, shell, tmp_path):
        ops, calls = shell
        _write(tmp_path, "rel.txt", b"here\n")
        r = ops.read_file("rel.txt")
        assert r.error is None and r.content == "1|here\n2|"

    def test_sentinel_lookalike_in_content_reads_intact(self, shell, tmp_path):
        ops, calls = shell
        lookalike = "__HERMES_RF_" + "ab" * 16 + "__"
        p = _write(tmp_path, "s.txt", f"x\n{lookalike}\ny\n".encode("utf-8"))
        r = ops.read_file(p)
        assert r.error is None and r.total_lines == 3
        assert r.content == f"1|x\n2|{lookalike}\n3|y\n4|"


class TestReadFileNonTextPaths:
    def test_missing_file_probes_once_then_suggests(self, shell, tmp_path):
        ops, calls = shell
        _write(tmp_path, "notes.txt", b"x\n")
        r = ops.read_file(str(tmp_path / "note.txt"))
        assert READ_PROBE_MARK in calls[0]
        assert r.error and "File not found" in r.error
        assert any(s.endswith("notes.txt") for s in r.similar_files)

    def test_unicode_variant_retry_still_works(self, shell, tmp_path):
        ops, calls = shell
        nfc = unicodedata.normalize("NFC", "café.txt")
        nfd = unicodedata.normalize("NFD", "café.txt")
        assert nfc != nfd
        _write(tmp_path, nfc, b"accent\n")
        r = ops.read_file(str(tmp_path / nfd))
        assert r.error is None and r.content == "1|accent\n2|"
        assert "unicode-equivalent" in r.hint

    def test_directory_is_not_regular(self, shell, tmp_path):
        ops, calls = shell
        r = ops.read_file(str(tmp_path))
        assert len(calls) == 1
        assert r.error and "not a regular file" in r.error

    def test_binary_sample_detected_in_same_reply(self, shell, tmp_path):
        ops, calls = shell
        p = _write(tmp_path, "blob", b"\x00\x01\x02" + b"\x00" * 50)
        r = ops.read_file(p)
        assert READ_PROBE_MARK in calls[0]
        assert r.is_binary is True and r.error
        # Only the UTF-16 rescue may add round-trips, never a second sample.
        assert not any("head -c 1000" in c for c in calls[1:])

    def test_image_extension_stops_at_size_probe(self, shell, tmp_path):
        ops, calls = shell
        r = ops.read_file(_write(tmp_path, "p.png", b"\x89PNG\r\n"))
        assert len(calls) == 1 and READ_PROBE_MARK not in calls[0]
        assert r.is_image is True and r.file_size == 6

    @pytest.mark.linux_only
    def test_fifo_returns_not_regular_without_blocking(self, shell, tmp_path):
        if not hasattr(os, "mkfifo"):
            pytest.skip("no mkfifo")
        ops, calls = shell
        fifo = tmp_path / "pipe"
        os.mkfifo(fifo)
        box = {}

        def run():
            box["r"] = ops.read_file(str(fifo))

        t = threading.Thread(target=run, daemon=True)
        t.start()
        t.join(20)
        assert not t.is_alive(), "read_file blocked on a writer-less FIFO"
        assert "not a regular file" in box["r"].error
        assert len(calls) == 1


class TestCompoundFallback:
    def test_unparseable_reply_falls_back_to_sequential_probes(self, shell, tmp_path):
        ops, calls = shell
        p = _write(tmp_path, "a.txt", b"one\ntwo\n")
        real_exec = ops._exec

        def garbled(command, *args, **kwargs):
            if READ_PROBE_MARK in command:
                return ExecuteResult(stdout="[Command timed out after 1s]\n", exit_code=124)
            return real_exec(command, *args, **kwargs)

        with patch.object(ops, "_exec", side_effect=garbled):
            r = ops.read_file(p)
        assert r.error is None and r.content == "1|one\n2|two\n3|"
        assert r.total_lines == 2
