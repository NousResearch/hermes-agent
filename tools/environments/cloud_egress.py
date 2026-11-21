"""Egress posture for cloud sandboxes (Modal direct, Daytona) behind ``proxy.cloud_relay``.

Same contract as the SSH tunnel: the sandbox holds opaque proxy tokens under the real provider
env names and reaches iron-proxy on ``127.0.0.1:<tunnel_port>``; real keys never leave the host.
The route there is the relay in ``egress_relay`` over the provider's own authenticated exec
stream — Modal ``Sandbox.exec`` stdin/stdout, a raw-mode Daytona PTY — never a public endpoint.

Managed Modal runs commands through the tool gateway's request/poll HTTP API, which has no
long-lived stream to carry the relay, so it refuses to start while ``cloud_relay`` is on.
"""

from __future__ import annotations

import contextlib
import logging
import os
import shlex
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from agent.memory_provider import spawn_context_thread
from tools.environments.egress_common import HOME_PREFIX, egress_env_overrides, export_lines, ready_egress
from tools.environments.egress_relay import (
    AGENT_SOURCE, READY_TIMEOUT_S, REMOTE_AGENT_NAME, LineSplitter, RelayHub, agent_command)
from tools.environments.ssh_egress import REMOTE_CA_NAME, REMOTE_ENV_NAME

logger = logging.getLogger("tools.environments.egress")

_OPT_OUT = "set proxy.cloud_relay: false to opt out"
_PTY_START_RETRY_S = 3.0


@dataclass(frozen=True)
class CloudEgress:
    tunnel_port: int
    targets: tuple[tuple[str, int], tuple[str, int]]  # iron-proxy CONNECT / plain-HTTP on the host
    ca_host_path: Path
    env_overrides: dict[str, str]

    def install(self, upload: Callable[[str, str], None], run: Callable[[str], None], remote_home: str) -> None:
        """Upload the CA, the token env file and the relay agent beside (not inside) ``~/.hermes``,
        then make them owner-only (uploads land with the sandbox's umask)."""
        remote = [f"{remote_home}/{name}" for name in (REMOTE_CA_NAME, REMOTE_ENV_NAME, REMOTE_AGENT_NAME)]
        upload(str(self.ca_host_path), remote[0])
        for path, text in zip(remote[1:], (export_lines(self.env_overrides), AGENT_SOURCE)):
            with tempfile.NamedTemporaryFile("w", encoding="utf-8", prefix="hermes-egress-", delete=False) as fh:
                fh.write(text)
            try:
                upload(fh.name, path)
            finally:
                os.unlink(fh.name)
        run("chmod 600 " + " ".join(shlex.quote(p) for p in remote))

    def hub(self, send_line: Callable[[str], None]) -> RelayHub:
        return RelayHub(send_line, self.targets)


def cloud_relay_requested() -> bool:
    try:
        from hermes_cli.config import load_config
    except ImportError as exc:  # same contract as ready_egress(): no config layer, no proxy
        logger.debug("Egress proxy plumbing unavailable: %s", exc)
        return False
    proxy_cfg = load_config().get("proxy") or {}
    return bool(proxy_cfg.get("enabled") and proxy_cfg.get("cloud_relay"))


def cloud_egress_for_env(backend: str) -> CloudEgress | None:
    """The posture for one cloud sandbox, or ``None`` when ``proxy.cloud_relay`` is off. Opting in
    is explicit, so a half-configured proxy fails closed (RuntimeError) instead of degrading."""
    if not cloud_relay_requested():
        return None
    from agent.proxy_sources import iron_proxy as ip

    def _degraded(msg: str) -> None:
        raise RuntimeError(f"{msg}  ({backend}: proxy.cloud_relay is on, so the sandbox refuses to start "
                           f"unprotected; {_OPT_OUT}.)")

    ready = ready_egress(_degraded)
    if ready is None:
        return None
    status, mappings = ready
    assert status.ca_cert_path is not None  # ready_egress() refused otherwise
    bind_host, _ = ip._probe_target()
    port = status.tunnel_port
    return CloudEgress(
        tunnel_port=port, targets=((bind_host, port), (bind_host, port + 1)), ca_host_path=status.ca_cert_path,
        env_overrides=egress_env_overrides("127.0.0.1", port, f"{HOME_PREFIX}{REMOTE_CA_NAME}", mappings))


def refuse_unrelayable(backend: str, why: str) -> None:
    """Fail closed for a backend that has no stream to carry the relay."""
    if cloud_relay_requested():
        raise RuntimeError(f"{backend} cannot carry the egress relay: {why}  proxy.cloud_relay is on, so it "
                           f"refuses to start rather than run without the proxy; {_OPT_OUT}.")


class Relay:
    """A running relay: the agent process inside the sandbox plus its host-side hub."""

    def __init__(self, hub: RelayHub, reader: threading.Thread, stop: Callable[[], None]):
        self.hub, self._reader, self._stop = hub, reader, stop

    def alive(self) -> bool:
        return self._reader.is_alive() and not self.hub.failed.is_set()

    def close(self) -> None:
        self.hub.close()
        with contextlib.suppress(Exception):  # health: allow BLE001 -- the stream may already be gone
            self._stop()


def start_modal_relay(egress: CloudEgress, sandbox, worker, remote_home: str) -> Relay:
    """Run the agent with ``Sandbox.exec`` and pump frames over its stdin/stdout on the worker loop."""
    proc = worker.run_coroutine(sandbox.exec.aio(*agent_command(remote_home, egress.tunnel_port)), timeout=60)

    async def _write(line: str) -> None:
        proc.stdin.write(line.encode("ascii"))
        await proc.stdin.drain.aio()

    hub = egress.hub(lambda line: worker.run_coroutine(_write(line), timeout=30))
    splitter = LineSplitter(hub.feed_line)

    async def _read() -> None:
        async for chunk in proc.stdout:
            splitter.feed(chunk)

    def _reader() -> None:
        try:
            worker.run_coroutine(_read(), timeout=None)
        except Exception:  # health: allow BLE001 -- SDK stream errors are untyped; the next command restarts the relay
            logger.debug("Modal egress relay ended", exc_info=True)
        finally:
            hub.close()

    async def _eof() -> None:  # the agent exits on stdin EOF, freeing its ports for a replacement
        proc.stdin.write_eof()
        await proc.stdin.drain.aio()

    reader = spawn_context_thread(_reader, name="hermes-egress-relay-modal")
    reader.start()
    hub.wait_ready("Modal")
    return Relay(hub, reader, stop=lambda: worker.run_coroutine(_eof(), timeout=10))


def start_daytona_relay(egress: CloudEgress, sandbox, remote_home: str) -> Relay:
    """Run the agent in a raw, no-echo Daytona PTY so frame lines cross the WebSocket unchanged."""
    session_id = f"hermes-egress-{uuid.uuid4().hex[:12]}"
    handle = sandbox.process.create_pty_session(id=session_id, cwd=remote_home)
    hub = egress.hub(handle.send_input)
    splitter = LineSplitter(hub.feed_line)

    def _reader() -> None:
        try:
            for chunk in handle:
                splitter.feed(chunk)
        except Exception:  # health: allow BLE001 -- SDK PTY errors are untyped; the next command restarts the relay
            logger.debug("Daytona egress relay ended", exc_info=True)
        finally:
            hub.close()

    reader = spawn_context_thread(_reader, name="hermes-egress-relay-daytona")
    reader.start()
    # Raw mode: no echo, no CR translation, no canonical line-length cap on a frame.
    start = "stty raw -echo; exec " + " ".join(agent_command(remote_home, egress.tunnel_port)) + "\n"
    # A shell that is still attaching can drop early input, so re-send until the agent answers; a
    # repeat that reaches a running agent is a non-frame line it ignores.
    deadline = time.monotonic() + READY_TIMEOUT_S
    while not hub.ready.is_set() and time.monotonic() < deadline:
        handle.send_input(start)
        hub.ready.wait(_PTY_START_RETRY_S)
    try:
        hub.wait_ready("Daytona")
    except RuntimeError:
        handle.kill()
        raise
    return Relay(hub, reader, stop=handle.kill)
