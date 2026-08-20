"""Regression: concurrent execute_code tool dispatch must not poison sys.stdout.

``code_execution_tool``'s sandbox RPC handler runs on a per-connection socket
thread, so several ``execute_code`` calls can be in flight simultaneously (e.g.
a parallel ``delegate_task`` batch, where every subagent uses execute_code).

The old implementation silenced internal tool handlers with a raw
process-global assignment::

    _real_stdout, _real_stderr = sys.stdout, sys.stderr
    devnull = open(os.devnull, "w")
    try:
        sys.stdout = devnull
        sys.stderr = devnull
        result = handle_function_call(...)
    finally:
        sys.stdout, sys.stderr = _real_stdout, _real_stderr
        devnull.close()

``sys.stdout`` is process-global. With two threads interleaving, thread B
captures thread A's devnull as its ``_real_stdout``, A closes that handle, and
B then "restores" the CLOSED handle. Every subsequent bare ``print`` anywhere
in the process raises::

    ValueError: I/O operation on closed file.

Observed 2026-07-28 on a live gateway: 328 occurrences beginning the instant a
3-way parallel delegation launched; all three subagents died. The tracebacks
landed on an innocent ``print`` in agent/conversation_loop.py, nowhere near the
actual culprit.
"""

from __future__ import annotations

import io
import json
import os
import socket
import sys
import threading
import time
from unittest.mock import patch

import pytest


def _drain(fn):
    """Bind a StringIO as the real stdout, run fn, return what reached it."""
    real_out = io.StringIO()
    orig = sys.stdout
    sys.stdout = real_out
    try:
        fn()
    finally:
        sys.stdout = orig
    return real_out.getvalue()


def test_concurrent_raw_assignment_poisons_stdout():
    """Characterize the exact race the old code had.

    Two threads each save/replace/restore sys.stdout by direct assignment. The
    interleaving leaves sys.stdout bound to a closed handle.
    """

    def body():
        # Events force the exact poisoning interleaving:
        #   A swaps -> B captures A's devnull & swaps -> A restores & CLOSES
        #   its devnull -> B "restores" the now-closed handle.
        a_swapped = threading.Event()
        b_swapped = threading.Event()
        a_restored = threading.Event()

        def worker_a():
            real_out, real_err = sys.stdout, sys.stderr
            devnull = open(os.devnull, "w", encoding="utf-8")
            try:
                sys.stdout = devnull
                sys.stderr = devnull
                a_swapped.set()
                assert b_swapped.wait(timeout=5)
            finally:
                sys.stdout, sys.stderr = real_out, real_err
                devnull.close()
                a_restored.set()

        def worker_b():
            assert a_swapped.wait(timeout=5)
            real_out, real_err = sys.stdout, sys.stderr  # <- A's devnull
            devnull = open(os.devnull, "w", encoding="utf-8")
            try:
                sys.stdout = devnull
                sys.stderr = devnull
                b_swapped.set()
                assert a_restored.wait(timeout=5)
            finally:
                # Installs the handle A already closed.
                sys.stdout, sys.stderr = real_out, real_err
                devnull.close()

        a = threading.Thread(target=worker_a)
        b = threading.Thread(target=worker_b)
        a.start()
        b.start()
        a.join()
        b.join()

        with pytest.raises(ValueError, match="closed file"):
            sys.stdout.write("this must fail")

    _drain(body)


def test_thread_scoped_silence_survives_concurrency():
    """The replacement primitive is safe under the same interleaving."""
    from agent.thread_scoped_output import thread_scoped_silence

    def body():
        barrier = threading.Barrier(2)

        def worker(hold: float):
            with thread_scoped_silence():
                print("internal tool chatter")
                barrier.wait(timeout=5)
                time.sleep(hold)

        a = threading.Thread(target=worker, args=(0.05,))
        b = threading.Thread(target=worker, args=(0.30,))
        a.start()
        b.start()
        a.join()
        b.join()

        print("survivor")

    captured = _drain(body)
    assert "survivor" in captured
    assert "internal tool chatter" not in captured


class _OneShotListener:
    """Minimal object exposing the .accept()/.settimeout() the RPC loop uses."""

    def __init__(self, conn):
        self._conn = conn
        self._served = False

    def settimeout(self, _t):
        pass

    def accept(self):
        if self._served:
            raise socket.timeout()
        self._served = True
        return self._conn, ("peer", 0)


def _blocking_handler(in_handler, unrelated_done):
    """A handle_function_call stub that prints chatter and blocks mid-dispatch.

    The prints stand in for internal tool-handler status output; the block lets
    the test overlap an unrelated thread's write with an in-flight dispatch.
    """

    def handler(*_args, **_kwargs):
        print("handler chatter")
        print("more handler chatter", file=sys.stderr)
        in_handler.set()
        assert unrelated_done.wait(timeout=5)
        return '{"output": "handled"}'

    return handler


def test_rpc_server_loop_dispatch_is_thread_scoped():
    """Behavioral: while _rpc_server_loop dispatches a tool call, an unrelated
    thread's stdout writes still reach the real stream, the handler's own
    chatter is silenced, and stdout stays usable afterwards.

    Fails on the pre-fix implementation, which rebound sys.stdout
    process-globally around dispatch.
    """
    from tools.code_execution_tool import _rpc_server_loop

    def body():
        in_handler = threading.Event()
        unrelated_done = threading.Event()

        srv, cli = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
        stop_event = threading.Event()

        def unrelated_writer():
            assert in_handler.wait(timeout=5)
            print("unrelated write during dispatch")
            unrelated_done.set()

        def run_server():
            _rpc_server_loop(
                _OneShotListener(srv),
                "test-task",
                [],
                [0],
                max_tool_calls=10,
                allowed_tools=frozenset({"terminal"}),
                stop_event=stop_event,
                rpc_token="tok",
            )

        writer = threading.Thread(target=unrelated_writer)
        server = threading.Thread(target=run_server, daemon=True)
        try:
            with patch(
                "model_tools.handle_function_call",
                side_effect=_blocking_handler(in_handler, unrelated_done),
            ):
                writer.start()
                server.start()
                cli.sendall((json.dumps({
                    "tool": "terminal",
                    "args": {"command": "echo hi"},
                    "token": "tok",
                }) + "\n").encode())
                cli.settimeout(5)
                resp = json.loads(cli.recv(65536).split(b"\n", 1)[0].decode())
                assert "handled" in json.dumps(resp)
        finally:
            stop_event.set()
            cli.close()
            srv.close()
            writer.join(timeout=5)
            server.join(timeout=5)

        # stdout must not be poisoned after dispatch completes.
        print("survivor")

    captured = _drain(body)
    assert "unrelated write during dispatch" in captured
    assert "survivor" in captured
    assert "handler chatter" not in captured


def test_rpc_poll_loop_dispatch_is_thread_scoped():
    """Behavioral: same contract for the remote-sandbox dispatch site.

    _rpc_poll_loop runs the identical silent-dispatch operation against a
    filesystem-polling env; a stub env feeds it one request and the test
    asserts an unrelated thread can still write while the handler runs.
    """
    from tools.code_execution_tool import _rpc_poll_loop

    request = {
        "tool": "terminal",
        "args": {"command": "echo hi"},
        "seq": 1,
        "token": "tok",
    }

    class _StubEnv:
        """Serves exactly one pending request file, then reports none."""

        def __init__(self):
            self._served = False

        def execute(self, command, cwd="/", timeout=10):
            if command.startswith("ls "):
                if self._served:
                    return {"output": ""}
                return {"output": "/rpc/req_000001"}
            if command.startswith("cat "):
                self._served = True
                return {"output": json.dumps(request)}
            return {"output": ""}

    def body():
        in_handler = threading.Event()
        unrelated_done = threading.Event()
        stop_event = threading.Event()

        def unrelated_writer():
            assert in_handler.wait(timeout=5)
            print("unrelated write during dispatch")
            unrelated_done.set()

        tool_call_counter = [0]

        def run_poll_loop():
            _rpc_poll_loop(
                _StubEnv(),
                "/rpc",
                "test-task",
                [],
                tool_call_counter,
                max_tool_calls=10,
                allowed_tools=frozenset({"terminal"}),
                stop_event=stop_event,
                rpc_token="tok",
            )

        writer = threading.Thread(target=unrelated_writer)
        poller = threading.Thread(target=run_poll_loop, daemon=True)
        try:
            with patch(
                "model_tools.handle_function_call",
                side_effect=_blocking_handler(in_handler, unrelated_done),
            ):
                writer.start()
                poller.start()
                assert unrelated_done.wait(timeout=10)
                # Wait for the dispatch to be recorded, then stop the loop.
                deadline = time.monotonic() + 5
                while tool_call_counter[0] < 1 and time.monotonic() < deadline:
                    time.sleep(0.01)
                assert tool_call_counter[0] == 1
        finally:
            stop_event.set()
            writer.join(timeout=5)
            poller.join(timeout=5)

        print("survivor")

    captured = _drain(body)
    assert "unrelated write during dispatch" in captured
    assert "survivor" in captured
    assert "handler chatter" not in captured
