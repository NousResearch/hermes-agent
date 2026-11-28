"""Per-profile ``--internal`` Docker network that makes iron-proxy the sandbox's only exit.

Under ``proxy.docker_internal_network`` the Docker sandbox joins a network created with
``--internal`` (no default route, no NAT) and IPv6 off. The host's address on that network
(its gateway) is the one place the sandbox can reach, and iron-proxy's tunnel listener binds
exactly there. A process that ignores ``HTTPS_PROXY`` (``curl --noproxy '*'``, a raw socket)
then has no route out, instead of getting the default bridge's unrestricted egress.

Every step fails closed: a network that cannot be created, is not ``Internal``, has IPv6 on,
or whose gateway is not where iron-proxy listens raises ``RuntimeError`` so the sandbox
never starts on a weaker network than the operator asked for.
"""

from __future__ import annotations

import hashlib
import ipaddress
import json
import re
import subprocess
from dataclasses import dataclass

from hermes_constants import get_hermes_home
from hermes_platform.host import facts
from tools.terminal_tool_config import home_profile_name

_NETWORK_LABEL_KEY = "hermes-egress-network"
_NAME_UNSAFE_RE = re.compile(r"[^a-z0-9_.-]")


@dataclass(frozen=True)
class InternalNetwork:
    name: str
    gateway: str  # host-side IPv4 on the network; iron-proxy binds here


def internal_network_name() -> str:
    """``hermes-egress-<profile>-<digest of HERMES_HOME>``: one network per profile, and two
    installs whose profiles share a name never share a network."""
    profile = _NAME_UNSAFE_RE.sub("_", home_profile_name().lower())[:32] or "default"
    digest = hashlib.sha256(str(get_hermes_home()).encode("utf-8")).hexdigest()[:8]
    return f"hermes-egress-{profile}-{digest}"


def _docker(argv: list[str], *, timeout: float) -> subprocess.CompletedProcess:
    # Plain subprocess, not remote_common.run_capture: `hermes egress` imports this module while
    # building the CLI parser, and remote_common's import chain pulls in the secret backends.
    return subprocess.run(argv, check=False, capture_output=True, text=True, encoding="utf-8", errors="replace",
                          timeout=timeout, stdin=subprocess.DEVNULL)


def _inspect(docker_exe: str, name: str) -> dict | None:
    try:
        result = _docker([docker_exe, "network", "inspect", name], timeout=15)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    try:
        data = json.loads(result.stdout)
    except ValueError:
        return None
    return data[0] if isinstance(data, list) and data and isinstance(data[0], dict) else None


def _create(docker_exe: str, name: str) -> str:
    """``docker network create --internal --ipv6=false``; returns stderr on failure, else ''."""
    try:
        result = _docker(
            [docker_exe, "network", "create", "--internal", "--ipv6=false",
             "--label", f"{_NETWORK_LABEL_KEY}=1", name], timeout=30)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return str(exc)
    return "" if result.returncode == 0 else (result.stderr.strip() or f"exit {result.returncode}")


def _gateway_ipv4(info: dict) -> str | None:
    for cfg in ((info.get("IPAM") or {}).get("Config") or []):
        try:
            addr = ipaddress.ip_address(str((cfg or {}).get("Gateway") or ""))
        except ValueError:
            continue
        if addr.version == 4 and addr.is_private and not addr.is_loopback:
            return str(addr)
    return None


def _verified(name: str, info: dict) -> InternalNetwork:
    if info.get("Internal") is not True:
        raise RuntimeError(
            f"Docker network {name!r} exists but is not internal (it has a route to the internet), "
            f"so sandboxes on it could bypass iron-proxy. Remove it (`docker network rm {name}`) "
            "and Hermes will recreate it with --internal.")
    if info.get("EnableIPv6"):
        raise RuntimeError(
            f"Docker network {name!r} has IPv6 enabled; iron-proxy only binds its IPv4 gateway. "
            f"Remove it (`docker network rm {name}`) and Hermes will recreate it with IPv6 off.")
    gateway = _gateway_ipv4(info)
    if gateway is None:
        raise RuntimeError(f"Docker network {name!r} reports no private IPv4 gateway for iron-proxy to bind.")
    return InternalNetwork(name=name, gateway=gateway)


def ensure_internal_network(docker_exe: str) -> InternalNetwork:
    """Create the profile's internal network if missing, then verify it (fail closed)."""
    if facts.os_family() != "linux":
        raise RuntimeError(
            "proxy.docker_internal_network needs Docker Engine on Linux: iron-proxy must bind the "
            "internal network's gateway, which Docker Desktop keeps inside its VM. Set it to false.")
    name = internal_network_name()
    info = _inspect(docker_exe, name)
    if info is None:
        error = _create(docker_exe, name)
        # A concurrent create ("already exists") is fine; the re-inspect decides.
        info = _inspect(docker_exe, name)
        if info is None:
            raise RuntimeError(f"Could not create the internal Docker network {name!r}: {error or 'inspect failed'}")
    return _verified(name, info)
