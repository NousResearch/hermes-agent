"""Sandbox port forwarding must propagate a remote close to the waiting client."""

import os
import socket
import subprocess
import sys
from threading import Event, Thread

import pytest

from tools.environments import streams


@pytest.mark.platforms("posix")
def test_remote_eof_reaches_client_while_upload_waits(monkeypatch):
    upload_waiting = Event()
    release_read, release_write = os.pipe()
    relay = subprocess.Popen(
        [sys.executable, "-c", "import os, sys; sys.stdin.buffer.read(1); "
         "os.read(int(sys.argv[1]), 1); print('reply', flush=True)", str(release_read)],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        pass_fds=(release_read,),
    )
    os.close(release_read)
    monkeypatch.setattr(streams, "open_stream", lambda *args, **kwargs: relay)
    client, accepted = socket.socketpair()
    client.settimeout(5)

    class Connection:
        reads = 0

        def recv(self, size):
            self.reads += 1
            if self.reads == 2:
                upload_waiting.set()
            return accepted.recv(size)

        def __getattr__(self, name):
            return getattr(accepted, name)

    forward = object.__new__(streams.PortForward)
    forward.env, forward.remote_host, forward.remote_port, forward.user = object(), "127.0.0.1", 1234, None
    serving = Thread(target=forward._serve, args=(Connection(),), daemon=True)
    serving.start()
    try:
        client.sendall(b"x")
        assert upload_waiting.wait(5)
        os.write(release_write, b"x")
        assert client.recv(1024) == b"reply\n"
        assert client.recv(1024) == b""
    finally:
        client.close()
        serving.join(5)
        accepted.close()
        os.close(release_write)
        relay.kill()
        relay.wait(timeout=5)
        for stream in (relay.stdin, relay.stdout, relay.stderr):
            stream.close()
