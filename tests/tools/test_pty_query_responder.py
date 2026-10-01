"""Tests for tools.pty_query_responder (port of openai/codex#41436).

Covers the byte-level scanner (exact queries, chunk splits, DEC private-mode
queries, passthrough of unhandled sequences) and a live PTY E2E where a
subprocess blocks on a cursor-position report until answered.
"""

import os
import sys
import time

import pytest

from tools.pty_query_responder import PtyQueryResponder


def test_plain_output_passes_through():
    r = PtyQueryResponder()
    out, resp = r.process(b"hello world\n")
    assert out == b"hello world\n"
    assert resp == b""


def test_device_status_report_answered_and_stripped():
    r = PtyQueryResponder()
    out, resp = r.process(b"before\x1b[5nafter")
    assert out == b"beforeafter"
    assert resp == b"\x1b[0n"


def test_window_size_query_reports_spawn_dimensions():
    r = PtyQueryResponder(rows=30, cols=120)
    out, resp = r.process(b"\x1b[18t")
    assert out == b""
    assert resp == b"\x1b[8;30;120t"


def test_cursor_position_report():
    r = PtyQueryResponder()
    out, resp = r.process(b"\x1b[6n")
    assert out == b""
    assert resp == b"\x1b[1;1R"


def test_dec_private_mode_query_reported_unrecognized():
    r = PtyQueryResponder()
    out, resp = r.process(b"\x1b[?1049$p")
    assert out == b""
    assert resp == b"\x1b[?1049;0$y"


def test_combined_stream_matches_codex_fixture():
    # Mirrors the driver-backed test in openai/codex#41436: queries split
    # across chunks, mixed with a color escape and plain text.
    r = PtyQueryResponder()
    out1, resp1 = r.process(b"before\x1b[")
    out2, resp2 = r.process(b"5n\x1b[18t\x1b[6n\x1b[?1049$p\x1b[31mafter")
    assert resp1 + resp2 == b"\x1b[0n\x1b[8;24;80t\x1b[1;1R\x1b[?1049;0$y"
    assert out1 + out2 + r.flush() == b"before\x1b[31mafter"


def test_query_split_across_many_chunks():
    r = PtyQueryResponder()
    total_out = b""
    total_resp = b""
    for b in (b"\x1b", b"[", b"6", b"n"):
        out, resp = r.process(b)
        total_out += out
        total_resp += resp
    assert total_out == b""
    assert total_resp == b"\x1b[1;1R"


def test_unhandled_csi_sequence_passes_through():
    r = PtyQueryResponder()
    out, resp = r.process(b"\x1b[31mred\x1b[0m")
    assert out == b"\x1b[31mred\x1b[0m"
    assert resp == b""


def test_non_csi_escape_passes_through():
    r = PtyQueryResponder()
    out, resp = r.process(b"\x1bMreverse")
    assert out == b"\x1bMreverse"
    assert resp == b""


def test_fresh_esc_aborts_partial_sequence():
    r = PtyQueryResponder()
    out, resp = r.process(b"\x1b[6\x1b[5n")
    # The aborted partial "\x1b[6" is flushed through; the complete
    # device-status query is answered and stripped.
    assert out == b"\x1b[6"
    assert resp == b"\x1b[0n"


def test_oversized_mode_query_passes_through():
    r = PtyQueryResponder()
    seq = b"\x1b[?12345678901$p"  # 11 digits > MAX_MODE_DIGITS
    out, resp = r.process(seq)
    assert resp == b""
    assert out + r.flush() == seq


def test_flush_returns_incomplete_tail():
    r = PtyQueryResponder()
    out, resp = r.process(b"text\x1b[1")
    assert out == b"text"
    assert resp == b""
    assert r.flush() == b"\x1b[1"
    # flush is destructive
    assert r.flush() == b""


def test_dec_mode_non_digit_passes_through():
    r = PtyQueryResponder()
    seq = b"\x1b[?10a9$p"
    out, resp = r.process(seq)
    assert resp == b""
    assert out + r.flush() == seq


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX ptyprocess only")
def test_live_pty_subprocess_unblocked_by_cursor_report(tmp_path):
    """E2E: a PTY subprocess blocking on ESC[6n exits once answered.

    Mirrors direct_terminal_queries_are_answered from openai/codex#41436.
    """
    ptyprocess = pytest.importorskip("ptyprocess")
    from tools.pty_query_responder import PtyQueryResponder

    script = (
        "stty -echo -icanon; printf 'alpha\\033[6n'; "
        "dd bs=1 count=6 2>/dev/null; printf '\\nok'"
    )
    proc = ptyprocess.PtyProcess.spawn(
        ["/bin/sh", "-c", script], dimensions=(24, 80)
    )
    responder = PtyQueryResponder()
    output = b""
    deadline = time.time() + 10
    try:
        while proc.isalive() and time.time() < deadline:
            try:
                chunk = proc.read(4096)
            except EOFError:
                break
            if not chunk:
                continue
            out, replies = responder.process(chunk)
            output += out
            if replies:
                proc.write(replies)
        # Drain anything left after exit.
        while True:
            try:
                chunk = proc.read(4096)
            except EOFError:
                break
            if not chunk:
                break
            out, replies = responder.process(chunk)
            output += out
    finally:
        if proc.isalive():
            proc.terminate(force=True)
            pytest.fail(f"subprocess still blocked; output={output!r}")
    output += responder.flush()
    assert b"alpha" in output
    assert b"ok" in output
    # The query itself must have been stripped from the captured output,
    # and the echoed reply is what dd consumed (not visible w/ -echo).
    assert b"\x1b[6n" not in output
