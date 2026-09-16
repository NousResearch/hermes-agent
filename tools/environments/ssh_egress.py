"""Egress-proxy plumbing for the SSH backend (opt-in via ``proxy.ssh_tunnel``).

Inspired by Perplexity Computer's SPACE runtime, whose network gateway injects credentials from
outside the sandbox for every backend it runs on. Hermes had that posture for Docker only; the SSH
backend shipped real ``terminal.env_passthrough`` keys to the remote host. With this module the
remote holds opaque proxy tokens and reaches iron-proxy on the Hermes host through an SSH reverse
forward (``-R``), so real keys never leave this machine.

Transport: the CA cert and an ``export``-lines env file are uploaded once per environment
(``~/.hermes-egress-ca.crt``, ``~/.hermes-egress.env`` mode 0600 — outside the synced ``~/.hermes``
tree) and every command sources the env file before running. Values therefore never appear in the
remote ``bash -c`` argv and the remote sshd needs no ``AcceptEnv`` for them.
"""

from __future__ import annotations

import logging
import shlex
from dataclasses import dataclass
from pathlib import Path

from tools.environments.egress_common import (
    _NODE_OPTIONS_SENTINEL, critical_egress_env_names, egress_env_overrides, ready_egress)

logger = logging.getLogger("tools.environments.ssh")

REMOTE_CA_NAME = ".hermes-egress-ca.crt"
REMOTE_ENV_NAME = ".hermes-egress.env"
_HOME_PREFIX = "$HOME/"


@dataclass(frozen=True)
class SSHEgress:
    tunnel_port: int
    bind_host: str  # where iron-proxy listens on the Hermes host
    ca_host_path: Path
    env_overrides: dict[str, str]

    def remote_forward_args(self) -> list[str]:
        """``-R`` flags binding the proxy's CONNECT and plain-HTTP listeners on the remote loopback.
        Passed on every invocation: a mux client re-requests the forward through the live master,
        which self-heals the tunnel after a master expires (a duplicate bind is a harmless warning)."""
        return [arg for port in (self.tunnel_port, self.tunnel_port + 1)
                for arg in ("-R", f"127.0.0.1:{port}:{self.bind_host}:{port}")]

    def env_file_text(self) -> str:
        lines = []
        for name, value in sorted(self.env_overrides.items()):
            if name == _NODE_OPTIONS_SENTINEL:
                # Append to whatever NODE_OPTIONS the remote login shell set, never clobber it.
                lines.append(f'export NODE_OPTIONS="${{NODE_OPTIONS:+$NODE_OPTIONS }}{value}"')
            elif value.startswith(_HOME_PREFIX):
                # CA path is resolved by the remote shell: the posture is built before the remote
                # home is known (the forward must ride the very first master connection).
                lines.append(f'export {name}="$HOME/{shlex.quote(value[len(_HOME_PREFIX):])}"')
            else:
                lines.append(f"export {name}={shlex.quote(value)}")
        return "\n".join(lines) + "\n"

    def critical_names(self) -> set[str]:
        return critical_egress_env_names(self.env_overrides)


def ssh_egress_for_env() -> SSHEgress | None:
    """The egress posture for one SSH environment, or ``None`` when ``proxy.ssh_tunnel`` is off.
    Opting in is explicit, so a half-configured proxy always fails closed (RuntimeError)."""
    try:
        from hermes_cli.config import load_config
        from agent.proxy_sources import iron_proxy as ip
    except ImportError as exc:
        logger.debug("Egress proxy plumbing unavailable: %s", exc)
        return None
    proxy_cfg = load_config().get("proxy") or {}
    if not (proxy_cfg.get("enabled") and proxy_cfg.get("ssh_tunnel")):
        return None

    def _degraded(msg: str) -> None:
        raise RuntimeError(f"{msg}  (proxy.ssh_tunnel is enabled, so the SSH backend refuses to start unprotected; "
                           "set proxy.ssh_tunnel: false to opt out.)")

    ready = ready_egress(_degraded)
    if ready is None:  # proxy.enabled flipped off between the check above and now
        return None
    status, mappings = ready
    assert status.ca_cert_path is not None  # ready_egress() refused otherwise
    bind_host, _ = ip._probe_target()
    return SSHEgress(
        tunnel_port=status.tunnel_port, bind_host=bind_host, ca_host_path=status.ca_cert_path,
        env_overrides=egress_env_overrides(
            "127.0.0.1", status.tunnel_port, f"{_HOME_PREFIX}{REMOTE_CA_NAME}", mappings))


def check_passthrough_collisions(passthrough_names, egress: SSHEgress) -> None:
    """``terminal.env_passthrough`` (or a skill's required vars) naming a proxied provider key would
    SendEnv the REAL value to the remote — exactly what the tunnel exists to prevent."""
    collisions = sorted(n for n in passthrough_names if n in egress.critical_names())
    if collisions:
        raise RuntimeError(
            f"env passthrough would send real egress-protected variables {collisions} to the SSH host.  "
            "Remove them from terminal.env_passthrough / the skill's required_environment_variables, "
            "or set proxy.ssh_tunnel: false to opt out of egress isolation.")


def source_prefix(remote_home: str) -> str:
    """Shell prefix that loads the uploaded env file; a missing file is loud, not a silent bypass."""
    env_file = shlex.quote(f"{remote_home}/{REMOTE_ENV_NAME}")
    return (f". {env_file} || {{ echo 'hermes: egress env file missing on remote' >&2; exit 97; }}\n")
