"""``read_file`` / ``write_file`` cost one shell round-trip, not four.

Real ``LocalEnvironment`` against ``tmp_path`` (no mocks), with a spy on
``env.execute`` counting round-trips. The cases below are exactly the ones
that used to need their own probe (existence, size, binary sample, page,
line count, trailing newline), so each proves the compound reply carries
that answer.
"""

import logging
import os
import sys
import threading
from unittest.mock import patch

import pytest

from tools.environments.local import LocalEnvironment
from tools.file_operations import ExecuteResult, ShellFileOperations

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="POSIX shell probes")

READ_PROBE_MARK = "__HERMES_RF_"


@pytest.fixture(scope="module")
def _local_env(tmp_path_factory):
    """One real LocalEnvironment per module; constructing one costs ~0.8 s."""
    return LocalEnvironment(cwd=str(tmp_path_factory.mktemp("file-ops")))


@pytest.fixture
def _ops(_local_env, tmp_path):
    """(ops, calls): file ops over the real local shell, every execute recorded."""
    env = _local_env
    env.cwd = str(tmp_path)
    calls = []
    real_execute = type(env).execute.__get__(env, type(env))

    def spy(command, *args, **kwargs):
        calls.append(command)
        return real_execute(command, *args, **kwargs)

    env.execute = spy
    try:
        yield ShellFileOperations(env, cwd=str(tmp_path)), calls
    finally:
        env.__dict__.pop("execute", None)


@pytest.fixture
def shell(_ops, monkeypatch):
    """Pin the shell path even where a native fast path exists."""
    monkeypatch.setenv("HERMES_NATIVE_FILE_READ", "0")
    return _ops


@pytest.fixture
def native(_ops, monkeypatch):
    """Same wiring with the native fast path on."""
    monkeypatch.delenv("HERMES_NATIVE_FILE_READ", raising=False)
    return _ops


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
        # A curly apostrophe vs the ASCII one: visually identical in a
        # terminal, and — unlike NFC/NFD — never aliased by the filesystem
        # (APFS resolves NFD lookups to NFC files directly, which would skip
        # the retry path this test exists to exercise).
        on_disk = "it\u2019s.txt"
        typed = "it's.txt"
        assert on_disk != typed
        _write(tmp_path, on_disk, b"accent\n")
        r = ops.read_file(str(tmp_path / typed))
        assert r.error is None and r.content == "1|accent\n2|"
        assert r.hint is not None and "unicode-equivalent" in r.hint

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


class TestWriteFileRoundTrips:
    """write_file: one probe, one atomic write, one hash check (three calls)."""

    @staticmethod
    def _execs(calls):
        return [c for c in calls]

    def test_new_text_file_is_three_round_trips(self, shell, tmp_path):
        ops, calls = shell
        p = str(tmp_path / "new.txt")
        r = ops.write_file(p, "line one\nline two\n")
        assert r.error is None and r.verified is True
        assert len(calls) == 3
        assert "__HERMES_WF_" in calls[0]          # probe
        assert "mv -f" in calls[1]                  # atomic write
        assert calls[2].startswith("sha256sum ")   # verify
        assert (tmp_path / "new.txt").read_bytes() == b"line one\nline two\n"

    def test_crlf_file_keeps_crlf_from_the_probe(self, shell, tmp_path):
        ops, calls = shell
        p = tmp_path / "crlf.txt"
        p.write_bytes(b"a\r\nb\r\n")
        r = ops.write_file(str(p), "x\ny\n")
        assert r.error is None and len(calls) == 3
        assert p.read_bytes() == b"x\r\ny\r\n"

    def test_bom_is_read_from_disk_and_preserved(self, shell, tmp_path):
        ops, calls = shell
        p = tmp_path / "bom.txt"
        p.write_bytes("﻿old\n".encode("utf-8"))
        r = ops.write_file(str(p), "new\n")
        assert r.error is None and len(calls) == 3
        assert p.read_bytes() == "﻿new\n".encode("utf-8")

    def test_pre_content_read_rides_the_same_probe(self, shell, tmp_path):
        """A lintable extension wants the old text (lint delta); it comes
        back in the probe reply instead of a separate ``cat``."""
        ops, calls = shell
        p = tmp_path / "code.py"
        p.write_bytes(b"x = 1\r\ny = 2\r\n")
        r = ops.write_file(str(p), "x = 1\ny = 3\n")
        assert r.error is None
        probes = [c for c in calls if "__HERMES_WF_" in c]
        assert len(probes) == 1 and "cat " in probes[0]
        assert not any(c.startswith("cat ") for c in calls)
        assert p.read_bytes() == b"x = 1\r\ny = 3\r\n"

    def test_missing_file_probe_does_not_block_the_write(self, shell, tmp_path):
        ops, calls = shell
        p = tmp_path / "deep" / "er" / "new.md"
        r = ops.write_file(str(p), "hi\n")
        assert r.error is None and r.dirs_created is True
        assert len(calls) == 3
        assert p.read_bytes() == b"hi\n"

    def test_unparseable_probe_reply_falls_back_to_separate_probes(self, shell, tmp_path):
        ops, calls = shell
        p = tmp_path / "crlf.txt"
        p.write_bytes(b"a\r\nb\r\n")
        real_exec = ops._exec

        def garbled(command, *args, **kwargs):
            if "__HERMES_WF_" in command:
                return ExecuteResult(stdout="[Command timed out after 1s]\n", exit_code=124)
            return real_exec(command, *args, **kwargs)

        with patch.object(ops, "_exec", side_effect=garbled):
            r = ops.write_file(str(p), "x\ny\n")
        assert r.error is None
        assert p.read_bytes() == b"x\r\ny\r\n"


class TestNativeRead:
    def test_native_read_makes_no_shell_call(self, native, tmp_path):
        ops, calls = native
        r = ops.read_file(_write(tmp_path, "a.txt", b"one\ntwo\n"))
        assert calls == []
        assert r.error is None and r.content == "1|one\n2|two\n3|"
        assert (r.total_lines, r.file_size) == (2, 8)

    def test_kill_switch_routes_to_the_shell(self, native, tmp_path, monkeypatch):
        ops, calls = native
        p = _write(tmp_path, "a.txt", b"one\n")
        monkeypatch.setenv("HERMES_NATIVE_FILE_READ", "0")
        ops.read_file(p)
        assert len(calls) == 1 and READ_PROBE_MARK in calls[0]

    def test_non_local_environment_keeps_the_shell_path(self):
        from unittest.mock import MagicMock

        env = MagicMock()
        env.cwd = "/tmp"
        assert ShellFileOperations(env)._native_read_enabled() is False

    def test_tilde_still_expands_through_the_shell(self, native):
        ops, calls = native
        r = ops.read_file("~/hermes-no-such-file-7f3a.txt")
        assert calls[0] == "echo $HOME"
        assert r.error and "File not found" in r.error

    def test_injection_lookalike_path_is_never_expanded(self, native, tmp_path):
        ops, calls = native
        marker = tmp_path / "pwned"
        r = ops.read_file(f"~; echo PWNED > {marker}")
        assert r.error and not marker.exists()
        # The text reaches the shell only single-quoted, inside the missing-
        # file recovery's directory listing; the tilde probe is a fixed
        # ``echo $HOME`` that never embeds the path. Nothing else runs.
        for c in calls:
            assert c == "echo $HOME" or c.startswith("ls -1 '~; echo PWNED"), c

    @pytest.mark.linux_only
    def test_fifo_refused_without_a_shell_and_without_blocking(self, native, tmp_path):
        if not hasattr(os, "mkfifo"):
            pytest.skip("no mkfifo")
        ops, calls = native
        fifo = tmp_path / "pipe"
        os.mkfifo(fifo)
        box = {}

        def run():
            box["r"] = ops.read_file(str(fifo))

        t = threading.Thread(target=run, daemon=True)
        t.start()
        t.join(20)
        assert not t.is_alive(), "native read_file blocked on a writer-less FIFO"
        assert "not a regular file" in box["r"].error
        assert calls == []


# The native reader scans 1 MiB chunks and clamps each page line to
# ``4 * get_max_line_length() + 1`` bytes (8001 by default), exactly as
# ``sed | cut -b1-N`` does. These shapes put a newline, a line, the clamp
# point, a CRLF pair and EOF precisely on those chunk boundaries.
_CHUNK = 1 << 20
_CLAMP = 8001

PARITY_CASES = [
    ("plain", b"one\ntwo\nthree\n", {}),
    ("no_trailing_newline", b"a\nb", {}),
    ("blank_tail", b"a\n\n", {}),
    ("crlf", b"x\r\ny\r\n", {}),
    ("lone_cr", b"a\rb\n", {}),
    ("bom", "﻿hello\n".encode("utf-8"), {}),
    ("empty", b"", {}),
    ("single_no_newline", b"solo", {}),
    ("only_newline", b"\n", {}),
    ("unicode", "héllo wörld\n汉字\n".encode("utf-8"), {}),
    ("long_line", b"a" * 9000 + b"\nshort\n", {}),
    ("multibyte_long_line", ("汉" * 4000 + "\nx\n").encode("utf-8"), {}),
    ("multi_chunk_line", b"b" * 3_000_000 + b"\nz\n", {}),
    ("newline_last_byte_of_chunk", b"a" * (_CHUNK - 1) + b"\nsecond\n", {}),
    ("newline_first_byte_of_next_chunk", b"a" * _CHUNK + b"\nsecond\n", {}),
    ("line_spans_three_boundaries", b"p\n" + b"b" * (3 * _CHUNK + 5) + b"\nz\n", {}),
    (
        "clamp_fills_on_boundary",
        b"x" * (_CHUNK - _CLAMP - 1) + b"\n" + b"c" * 20000 + b"\ntail\n",
        {"offset": 2, "limit": 3},
    ),
    (
        "clamp_fills_after_boundary",
        b"x" * (_CHUNK - _CLAMP) + b"\n" + b"c" * 20000 + b"\ntail\n",
        {"offset": 2, "limit": 3},
    ),
    ("eof_midline_on_boundary", b"l1\n" + b"d" * (2 * _CHUNK - 3), {}),
    ("blank_lines_on_boundary", b"e" * (_CHUNK - 2) + b"\n\n\n" + b"f\n", {}),
    ("crlf_split_on_boundary", b"r" * (_CHUNK - 1) + b"\r\n" + b"s\r\n", {}),
    (
        "page_starts_in_second_chunk",
        b"q\n" * (_CHUNK // 2 + 3) + b"target1\ntarget2\n",
        {"offset": _CHUNK // 2 + 3, "limit": 4},
    ),
    ("window", b"".join(b"l%d\n" % i for i in range(1, 11)), {"offset": 3, "limit": 2}),
    ("window_reaches_eof", b"".join(b"l%d\n" % i for i in range(1, 11)), {"offset": 9, "limit": 5}),
    ("past_eof", b"".join(b"l%d\n" % i for i in range(1, 6)), {"offset": 50}),
    ("nul_binary", b"\x00\x01\x02" * 20, {}),
    ("latin1_tail", b"caf\xe9\n", {}),
    ("sentinel_lookalike", b"x\n__HERMES_RF_" + b"ab" * 16 + b"__\ny\n", {}),
]


class TestNativeReadParity:
    """The native path must be indistinguishable from the shell path."""

    @pytest.mark.parametrize("name,data,kwargs", PARITY_CASES, ids=[c[0] for c in PARITY_CASES])
    def test_shell_and_native_agree(self, native, tmp_path, monkeypatch, name, data, kwargs):
        ops, calls = native
        p = _write(tmp_path, f"{name}.txt", data)
        monkeypatch.setenv("HERMES_NATIVE_FILE_READ", "0")
        via_shell = ops.read_file(p, **kwargs).to_dict()
        assert calls and READ_PROBE_MARK in calls[0]
        calls.clear()
        monkeypatch.delenv("HERMES_NATIVE_FILE_READ")
        via_native = ops.read_file(p, **kwargs).to_dict()
        assert via_native == via_shell
        # The native path touches the shell only for the UTF-16 rescue of binaries.
        assert not any(READ_PROBE_MARK in c for c in calls)

    def test_special_paths_agree(self, native, tmp_path, monkeypatch):
        ops, calls = native
        _write(tmp_path, "real.txt", b"target\n")
        os.symlink(tmp_path / "real.txt", tmp_path / "link.txt")
        os.symlink(tmp_path / "gone", tmp_path / "dangling.txt")
        (tmp_path / "sub").mkdir()
        _write(tmp_path, "pic.png", b"\x89PNG\r\n")
        _write(tmp_path, "notes.txt", b"n\n")
        for p in (
            str(tmp_path / "link.txt"),
            str(tmp_path / "dangling.txt"),
            str(tmp_path / "sub"),
            str(tmp_path / "pic.png"),
            str(tmp_path / "note.txt"),   # missing → similar-file suggestions
            "real.txt",                   # relative to env.cwd
        ):
            monkeypatch.setenv("HERMES_NATIVE_FILE_READ", "0")
            via_shell = ops.read_file(p).to_dict()
            monkeypatch.delenv("HERMES_NATIVE_FILE_READ")
            calls.clear()
            via_native = ops.read_file(p).to_dict()
            assert via_native == via_shell, p
            assert not any(READ_PROBE_MARK in c for c in calls), p


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

    def test_fallback_is_logged_at_debug(self, shell, tmp_path, caplog):
        """A backend that keeps falling back shows up in debug logs."""
        ops, calls = shell
        p = _write(tmp_path, "a.txt", b"one\n")
        real_exec = ops._exec

        def garbled(command, *args, **kwargs):
            if READ_PROBE_MARK in command:
                return ExecuteResult(stdout="garbage\n", exit_code=0)
            return real_exec(command, *args, **kwargs)

        with caplog.at_level(logging.DEBUG, logger="tools.file_operations"), \
             patch.object(ops, "_exec", side_effect=garbled):
            r = ops.read_file(p)
        assert r.error is None and r.content == "1|one\n2|"
        assert any(
            "falling back to sequential probes" in rec.getMessage()
            and str(p) in rec.getMessage()
            for rec in caplog.records
        )
