"""A session kernel's RPC stub must answer each call with that call's reply.

Replay of #138373: one reply failed to parse, and every later ``web_extract`` in
that kernel returned the PREVIOUS call's page — a silent wrong answer that looks
successful (``error: None``, plausible content).
"""
import json
import os
import shutil
import socket
import tempfile
import threading
import time

import pytest

from tools.code_execution_tool import generate_hermes_tools_module

pytestmark = pytest.mark.platforms("posix")

TOOL = "web_extract"


class _FakeKernel:
    """UDS RPC listener answering each accepted connection from a script.

    One entry per accepted connection: ``{"chunks": [bytes, ...], "delay": float,
    "close": bool}``. The delay between chunks is what makes the framing
    deterministic — without it both chunks land in one ``recv`` and the stub
    parses them together. ``close=False`` leaves the connection open, so bytes
    the client did not read stay queued for its next call (the desync).
    """

    def __init__(self, replies):
        # AF_UNIX paths are capped at ~104 bytes: pytest's tmp_path is too long
        # on macOS, so bind under a short /tmp dir of our own.
        self._dir = tempfile.mkdtemp(prefix="rpc-", dir="/tmp")
        self.path = os.path.join(self._dir, "s.sock")
        self._replies = list(replies)
        self._held = []
        self.requests = []
        self._listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._listener.bind(self.path)
        self._listener.listen(4)
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()

    def _serve(self):
        while self._replies:
            try:
                conn, _ = self._listener.accept()
            except OSError:
                return
            spec = self._replies.pop(0)
            try:
                buf = b""
                while b"\n" not in buf:
                    chunk = conn.recv(65536)
                    if not chunk:
                        break
                    buf += chunk
                if buf:
                    self.requests.append(json.loads(buf.split(b"\n", 1)[0].decode()))
                for index, chunk in enumerate(spec["chunks"]):
                    conn.sendall(chunk)
                    if index + 1 < len(spec["chunks"]):
                        # The split between the two chunks is what the test is about.
                        time.sleep(spec.get("delay", 0.2))  # health: allow HX013 -- the chunk split is the behaviour under test
            except OSError:
                # The stub dropped the connection — that is the fix working.
                continue
            if spec.get("close", True):
                conn.close()
            else:
                self._held.append(conn)

    def close(self):
        for conn in self._held:
            try:
                conn.close()
            except OSError:
                pass
        self._held.clear()
        try:
            self._listener.close()
        except OSError:
            pass
        shutil.rmtree(self._dir, ignore_errors=True)


def _stub(socket_path, monkeypatch, *, persistent=False):
    monkeypatch.setenv("HERMES_RPC_SOCKET", socket_path)
    if persistent:
        monkeypatch.setenv("HERMES_RPC_PERSISTENT", "1")
    else:
        monkeypatch.delenv("HERMES_RPC_PERSISTENT", raising=False)
    namespace = {}
    exec(generate_hermes_tools_module([TOOL], transport="uds"), namespace)
    return namespace


def test_blank_line_is_not_consumed_as_the_reply(monkeypatch):
    """The blank line that shifted the stream in #138373 is not a reply body."""
    answer = json.dumps({"url": "https://example.test/first"})
    server = _FakeKernel([{"chunks": [b"\n", answer.encode() + b"\n"]}])
    try:
        stub = _stub(server.path, monkeypatch)
        assert stub[TOOL](urls=["https://example.test/first"])["url"] == "https://example.test/first"
    finally:
        server.close()


def test_unparseable_reply_is_not_handed_to_the_next_call(monkeypatch):
    """An unreadable reply must not be replayed as the answer of the next call."""
    stale = json.dumps({"url": "https://example.test/stale"})
    fresh = json.dumps({"url": "https://example.test/fresh"})
    server = _FakeKernel([
        {"chunks": [b"not json\n", stale.encode() + b"\n"], "close": False},
        {"chunks": [fresh.encode() + b"\n"]},
    ])
    try:
        stub = _stub(server.path, monkeypatch)
        with pytest.raises(json.JSONDecodeError):
            stub[TOOL](urls=["https://example.test/first"])
        assert stub[TOOL](urls=["https://example.test/fresh"])["url"] == "https://example.test/fresh"
    finally:
        server.close()


def test_persistent_kernel_retries_on_a_fresh_socket(monkeypatch):
    """A session kernel reconnects and still gets its own reply."""
    answer = json.dumps({"url": "https://example.test/answer"})
    server = _FakeKernel([
        {"chunks": [b"not json\n"]},
        {"chunks": [answer.encode() + b"\n"]},
    ])
    try:
        stub = _stub(server.path, monkeypatch, persistent=True)
        assert stub[TOOL](urls=["https://example.test/answer"])["url"] == "https://example.test/answer"
        assert [request["tool"] for request in server.requests] == [TOOL, TOOL]
    finally:
        server.close()
