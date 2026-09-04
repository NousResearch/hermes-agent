"""The readiness poll must drain the started app's stdout while it polls.

Regression tests for a start command that logs more than the OS pipe buffer
holds before it binds its port. Nothing read that pipe until after teardown,
so the child blocked in ``write()``, never bound, and a healthy app was
reported as a readiness failure for the whole ``ready_timeout`` window.

These drive the real ``_run_start_phase`` against real subprocesses — the bug
lives in the interaction between an OS pipe and the poll loop, so a mocked
Popen cannot reproduce it.
"""

from __future__ import annotations

import socket
import sys
import textwrap

import pytest

from agent.verify.recipes import Recipe
from agent.verify.runner import _run_start_phase

# Comfortably larger than the 64 KiB pipe buffer on Linux (and the smaller
# buffers elsewhere), so the child is guaranteed to block without a drain.
_CHATTY_LINES = 4000
_LINE_PADDING = 40


def _free_port() -> int:
    sock = socket.socket()
    try:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])
    finally:
        sock.close()


def _write_app(tmp_path, *, noisy: bool):
    """A server that logs ``_CHATTY_LINES`` lines *before* it binds its port."""
    app = tmp_path / "app.py"
    preamble = (
        textwrap.dedent(
            f"""
            for i in range({_CHATTY_LINES}):
                print("[boot] line %d %s" % (i, "x" * {_LINE_PADDING}), flush=True)
            """
        )
        if noisy
        else ""
    )
    app.write_text(
        textwrap.dedent(
            """
            import sys, http.server, socketserver
            {preamble}
            port = int(sys.argv[1])
            socketserver.TCPServer.allow_reuse_address = True
            with socketserver.TCPServer(("127.0.0.1", port),
                                        http.server.SimpleHTTPRequestHandler) as srv:
                srv.serve_forever()
            """
        ).format(preamble=textwrap.indent(preamble, "")),
        encoding="utf-8",
    )
    return app


def _recipe_for(app, port: int) -> Recipe:
    return Recipe(
        name="chatty",
        kind="test",
        start=f"{sys.executable} {app} {port}",
        port=port,
        readiness_path="/",
    )


@pytest.mark.timeout(120)
def test_chatty_start_command_still_becomes_ready(tmp_path):
    """The bug: >64 KiB of startup logs must not stop the app binding its port."""
    port = _free_port()
    app = _write_app(tmp_path, noisy=True)

    result = _run_start_phase(_recipe_for(app, port), tmp_path, ready_timeout=25.0)

    assert result.ready is True, (
        "a healthy app that logs before binding was reported unready — "
        f"error={result.error!r}"
    )
    assert result.status_code is not None


@pytest.mark.timeout(120)
def test_quiet_start_command_unaffected(tmp_path):
    """Guard: the drain must not disturb the ordinary quiet-server path."""
    port = _free_port()
    app = _write_app(tmp_path, noisy=False)

    result = _run_start_phase(_recipe_for(app, port), tmp_path, ready_timeout=25.0)

    assert result.ready is True
    assert result.status_code is not None


@pytest.mark.timeout(120)
def test_startup_output_is_captured_not_lost_to_teardown(tmp_path):
    """The tail now comes from the drain, so early startup logs survive."""
    port = _free_port()
    app = _write_app(tmp_path, noisy=True)

    result = _run_start_phase(_recipe_for(app, port), tmp_path, ready_timeout=25.0)

    assert "[boot] line" in result.output_tail


def test_reader_retains_only_the_tail():
    """A server polled for 60s must not grow the capture buffer without bound."""
    from agent.verify.runner import _BackgroundOutputReader


    class _Stream:
        def __init__(self, lines):
            self._lines = iter(lines)

        def __iter__(self):
            return self._lines

    reader = _BackgroundOutputReader(_Stream([f"line {i}\n" for i in range(5000)]), limit=100)
    reader.start()
    captured = reader.result()

    assert len(captured) <= 100
    assert captured.endswith("line 4999\n")


def test_reader_survives_a_stream_that_raises():
    """Diagnostics must never fail the run: a closed pipe is not an error."""
    from agent.verify.runner import _BackgroundOutputReader


    class _Broken:
        def __iter__(self):
            return self

        def __next__(self):
            raise ValueError("I/O operation on closed file")

    reader = _BackgroundOutputReader(_Broken())
    reader.start()

    assert reader.result() == ""
