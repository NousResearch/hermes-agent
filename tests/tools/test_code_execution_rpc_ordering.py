"""execute_code UDS RPC: a reply must always belong to the request that is waiting for it.

The wire protocol is one JSON line per request, answered in order on one long-lived socket.
Before request ids, a call interrupted after its request was sent (an alarm-based timeout,
KeyboardInterrupt, a cancelled thread) left its reply on the socket, and every later call read
the previous call's reply: web_search/web_extract results one request behind, silently.
"""

import json
import queue
import socket
import threading
from unittest.mock import patch

import pytest

from tools.code_execution_rpc import _rpc_server_loop
from tools.code_execution_tool import generate_hermes_tools_module

TOKEN = "test-token"


class _Interrupted(BaseException):
    """Stands in for any non-Exception interrupt that unwinds out of a blocking recv()."""


def _echo_dispatch(function_name, function_args, task_id=None, user_task=None):
    return json.dumps({"output": function_args.get("command", "")})


class _Listener:
    """The accept()/settimeout() surface _rpc_server_loop uses, handing out queued socketpair ends."""

    def __init__(self):
        self.pending = queue.Queue()

    def settimeout(self, _seconds):
        pass

    def accept(self):
        try:
            return self.pending.get(timeout=0.05), ("peer", 0)
        except queue.Empty:
            raise TimeoutError from None

    def connect(self):
        client, server = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
        client.settimeout(5)
        self.pending.put(server)
        return client


@pytest.fixture
def rpc_server():
    """The real _rpc_server_loop, re-accepting after a disconnect like a session kernel does."""
    listener = _Listener()
    stop = threading.Event()

    def _serve():
        while not stop.is_set():
            _rpc_server_loop(listener, "t", [], [0], 1000, frozenset({"terminal"}), stop, TOKEN)

    with patch("model_tools.handle_function_call", side_effect=_echo_dispatch):
        thread = threading.Thread(target=_serve, daemon=True)
        thread.start()
        try:
            yield listener
        finally:
            stop.set()
            thread.join(timeout=5)


def _stub_namespace(listener, monkeypatch):
    """The generated hermes_tools stub, its socket connect pointed at *listener* (reconnects like the real one)."""
    monkeypatch.setenv("HERMES_RPC_TOKEN", TOKEN)
    monkeypatch.setenv("HERMES_RPC_PERSISTENT", "1")
    namespace = {}
    exec(generate_hermes_tools_module(["terminal"]), namespace)

    def _connect():
        if namespace["_sock"] is None:
            namespace["_sock"] = listener.connect()
        return namespace["_sock"]

    namespace["_connect"] = _connect
    return namespace


def test_interrupted_call_does_not_hand_its_reply_to_the_next_call(rpc_server, monkeypatch):
    ns = _stub_namespace(rpc_server, monkeypatch)
    connect = ns["_connect"]
    interrupt_next_recv = [True]

    class _InterruptAfterSend:
        def __init__(self, conn):
            self._conn = conn

        def sendall(self, data):
            return self._conn.sendall(data)

        def recv(self, size):
            if interrupt_next_recv[0]:
                interrupt_next_recv[0] = False
                raise _Interrupted()
            return self._conn.recv(size)

    ns["_connect"] = lambda: _InterruptAfterSend(connect())

    with pytest.raises(_Interrupted):
        ns["terminal"]("REQUEST-A")
    reply = ns["terminal"]("REQUEST-B")

    assert reply["output"] == "REQUEST-B"


def test_reply_carrying_another_requests_id_is_rejected(monkeypatch):
    """A bridge that answers out of step must fail loudly, never return the wrong result."""
    listener = _Listener()

    def _answer_with_a_foreign_id():
        for _attempt in range(2):  # the stub retries once on a fresh connection
            conn, _ = _accept_blocking(listener)
            conn.recv(65536)
            stale = json.dumps({"rid": "some-earlier-request", "result": json.dumps({"output": "STALE"})})
            conn.sendall((stale + "\n").encode())

    thread = threading.Thread(target=_answer_with_a_foreign_id, daemon=True)
    thread.start()
    ns = _stub_namespace(listener, monkeypatch)

    with pytest.raises(RuntimeError, match="id mismatch"):
        ns["terminal"]("REQUEST-B")
    thread.join(timeout=5)


def _accept_blocking(listener):
    while True:
        try:
            return listener.accept()
        except TimeoutError:
            continue


def test_server_echoes_the_request_id_and_still_serves_clients_without_one(rpc_server):
    def _ask(request):
        client = rpc_server.connect()
        try:
            client.sendall((json.dumps(request) + "\n").encode())
            buf = b""
            while not buf.endswith(b"\n"):
                buf += client.recv(65536)
            return json.loads(buf.decode())
        finally:
            client.close()

    call = {"tool": "terminal", "args": {"command": "hi"}, "token": TOKEN}
    with_id = _ask({**call, "rid": "abc-1"})
    without_id = _ask(call)

    assert with_id["rid"] == "abc-1"
    assert json.loads(with_id["result"]) == {"output": "hi"}
    assert without_id == {"output": "hi"}


def test_ordinary_calls_still_return_their_own_results(rpc_server, monkeypatch):
    ns = _stub_namespace(rpc_server, monkeypatch)

    replies = [ns["terminal"](f"call-{i}")["output"] for i in range(3)]

    assert replies == ["call-0", "call-1", "call-2"]
