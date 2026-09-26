"""A generated local RPC client must not replay a timed-out tool call."""

import json
import socket
import socketserver
import threading
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from tools.code_execution_tool import generate_hermes_tools_module


def _client(monkeypatch, persistent=True):
    monkeypatch.setenv("HERMES_RPC_PERSISTENT", "1" if persistent else "0")
    namespace = {}
    exec(generate_hermes_tools_module(["terminal"]), namespace)
    return namespace


@pytest.mark.parametrize("partial", [False, True], ids=["no-response", "partial-response"])
def test_timeout_does_not_replay_and_next_call_uses_fresh_connection(monkeypatch, partial):
    release = threading.Event()
    requests = []

    class Handler(socketserver.StreamRequestHandler):
        def handle(self):
            request = json.loads(self.rfile.readline())
            requests.append(request)
            if request["args"]["command"] == "slow-effect":
                if partial:
                    self.wfile.write(b'{"output":')
                    self.wfile.flush()
                release.wait(10)
                return
            self.wfile.write(b'{"output":"fresh-result"}\n')
            self.wfile.flush()

    with socketserver.ThreadingTCPServer(("127.0.0.1", 0), Handler) as server:
        thread = threading.Thread(target=server.serve_forever)
        thread.start()
        client = _client(monkeypatch)
        monkeypatch.setenv("HERMES_RPC_SOCKET", f"tcp://127.0.0.1:{server.server_address[1]}")
        original_connect = client["_connect"]

        def connect():
            connection = original_connect()
            connection.settimeout(2)
            return connection

        client["_connect"] = connect
        try:
            with pytest.raises(socket.timeout):
                client["terminal"]("slow-effect")
            assert [r["args"]["command"] for r in requests] == ["slow-effect"]
            assert client["_sock"] is None
            release.set()
            assert client["terminal"]("next-call") == {"output": "fresh-result"}
            assert [r["args"]["command"] for r in requests] == ["slow-effect", "next-call"]
        finally:
            release.set()
            if client["_sock"] is not None:
                client["_sock"].close()
            server.shutdown()
            thread.join(10)


@pytest.mark.parametrize("stage", ["connect", "sendall", "recv"])
@pytest.mark.parametrize("persistent", [False, True])
def test_timeout_closes_socket_without_retry(monkeypatch, stage, persistent):
    client = _client(monkeypatch, persistent)
    monkeypatch.setenv("HERMES_RPC_SOCKET", "tcp://127.0.0.1:12345")
    connection = Mock()
    getattr(connection, stage).side_effect = socket.timeout("test timeout")
    factory = Mock(return_value=connection)
    client["socket"] = SimpleNamespace(
        socket=factory, AF_INET=socket.AF_INET, SOCK_STREAM=socket.SOCK_STREAM,
        timeout=socket.timeout,
    )
    with pytest.raises(socket.timeout, match="test timeout"):
        client["terminal"]("effect")
    assert factory.call_count == 1
    connection.close.assert_called_once()
    assert client["_sock"] is None


@pytest.mark.parametrize("failure", [BrokenPipeError("idle disconnect"), b""])
def test_persistent_idle_disconnect_still_reconnects_once(monkeypatch, failure):
    client = _client(monkeypatch)
    stale = Mock()
    if isinstance(failure, Exception):
        stale.sendall.side_effect = failure
    else:
        stale.recv.return_value = failure
    fresh = Mock()
    fresh.recv.return_value = b'{"output":"ok"}\n'
    connections = iter([stale, fresh])

    def connect():
        client["_sock"] = next(connections)
        return client["_sock"]

    client["_connect"] = connect
    assert client["terminal"]("normal-call") == {"output": "ok"}
    stale.close.assert_called_once()
    fresh.sendall.assert_called_once()
