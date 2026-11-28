"""Cloud-sandbox egress (proxy.cloud_relay): the relay carries proxy connections over the exec
stream, and every half-configured or unrelayable state fails closed."""

import socket
import subprocess
import sys
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from tools.environments import cloud_egress, egress_relay
from tests.waiting import wait_until
from tools.environments.egress_relay import AGENT_SOURCE, LineSplitter, RelayHub


def _free_port_pair() -> int:
    """A port whose +1 neighbour is free too (the agent binds both, like the proxy listeners)."""
    while True:
        with socket.socket() as a, socket.socket() as b:
            a.bind(("127.0.0.1", 0))
            port = a.getsockname()[1]
            try:
                b.bind(("127.0.0.1", port + 1))
            except OSError:
                continue
            return port


def test_relay_carries_a_connection_over_the_exec_stream(tmp_path, monkeypatch):
    """The in-sandbox agent and the host hub, joined only by the agent's stdin/stdout, round-trip a
    TCP connection to the host-side target — tolerating PTY noise before the first frame and
    keeping bytes that arrive while the host side is still dialling."""
    dialling, release = threading.Event(), threading.Event()
    real_dial = egress_relay.socket.create_connection

    def _held_dial(*args, **kwargs):
        dialling.set()
        assert release.wait(10)
        return real_dial(*args, **kwargs)
    monkeypatch.setattr(egress_relay.socket, "create_connection", _held_dial)

    upstream = socket.socket()
    upstream.bind(("127.0.0.1", 0))
    upstream.listen()

    def _echo():
        conn, _ = upstream.accept()
        conn.sendall(b"ECHO:" + conn.recv(100))
        conn.close()
    threading.Thread(target=_echo, daemon=True).start()

    agent = tmp_path / "agent.py"
    agent.write_text(AGENT_SOURCE, encoding="utf-8")
    port = _free_port_pair()
    proc = subprocess.Popen([sys.executable, "-u", str(agent), str(port)],
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    def _send(line: str) -> None:
        proc.stdin.write(line.encode("ascii"))
        proc.stdin.flush()
    hub = RelayHub(_send, (("127.0.0.1", upstream.getsockname()[1]),) * 2)
    try:
        splitter = LineSplitter(hub.feed_line)
        splitter.feed("\x1b[?2004hbash# stty raw -echo; exec python3 ...\r\n\x1b[?2004l\r")  # PTY echo + reset

        def _pump():
            while chunk := proc.stdout.read1(65536):
                splitter.feed(chunk)
        threading.Thread(target=_pump, daemon=True).start()
        hub.wait_ready("test")

        with real_dial(("127.0.0.1", port), timeout=10) as client:
            assert dialling.wait(10)
            client.sendall(b"hello")  # reaches the hub while its dial is held open
            wait_until(lambda: any(not q.empty() for q in hub._inboxes.values()), what="data frame queued mid-dial")
            release.set()
            reply = client.recv(100)
        assert reply == b"ECHO:hello"

        # A frame the stream cannot take ends the relay, so the next command starts a fresh one
        # instead of serving connections that silently lost bytes.
        def _broken(_line: str) -> None:
            raise BrokenPipeError
        broken = RelayHub(_broken, hub._targets)
        broken.feed_line("O 7 MA==")
        broken.feed_line("C 7 ")  # the client hangs up: the hub must send a close frame back
        wait_until(broken.failed.is_set, what="relay marked failed after a lost frame")
        wait_until(lambda: not broken._inboxes, what="connections ended")
    finally:
        hub.close()
        proc.kill()
        proc.wait()
        upstream.close()


def test_cloud_relay_fails_closed_and_stays_off_by_default():
    stopped = SimpleNamespace(configured=True, pid=None, listening=False, tunnel_port=9090,
                              ca_cert_path=Path("/host/ca.crt"))
    on = {"proxy": {"enabled": True, "cloud_relay": True}}
    with patch("hermes_cli.config.load_config", return_value=on), \
         patch("agent.proxy_sources.iron_proxy.get_status", return_value=stopped), \
         pytest.raises(RuntimeError, match="not running.*Modal: proxy.cloud_relay is on"):
        cloud_egress.cloud_egress_for_env("Modal")

    # Managed Modal has no stream to carry the relay: it refuses rather than run with no proxy.
    from tools.environments.managed_modal import ManagedModalEnvironment
    with patch("hermes_cli.config.load_config", return_value=on), \
         pytest.raises(RuntimeError, match="Managed Modal cannot carry the egress relay"):
        ManagedModalEnvironment(image="python:3.11")

    with patch("hermes_cli.config.load_config", return_value={"proxy": {"enabled": True}}):
        assert cloud_egress.cloud_egress_for_env("Daytona") is None  # opt-in only
        cloud_egress.refuse_unrelayable("Managed Modal", "n/a")  # no-op when off
