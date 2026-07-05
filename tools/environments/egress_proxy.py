"""Domain-allowlisted egress proxy for the Docker sandbox ("allowlist" mode).

When ``terminal.container_network: "allowlist"`` is set, sandbox containers are
attached to a per-allowlist ``--internal`` Docker network (no route to the
internet) and pointed at a shared forward proxy via ``HTTP(S)_PROXY``. The proxy
(:mod:`tools.environments.egress_proxy_server`, run inside a ``python:3.13-slim``
container that is dual-homed onto the bridge network) only forwards to allowlisted
domains. Net effect:

- HTTP(S) to an allowlisted domain  -> allowed (via proxy)
- HTTP(S) to any other domain       -> 403 from the proxy
- raw sockets / non-proxied egress  -> no route out (internal network)

The proxy is keyed by a hash of the (sorted) allowlist, so different allowlists
get isolated networks/proxies and identical ones are reused.
"""

import hashlib
import logging
import os
import subprocess
import threading
from dataclasses import dataclass, field

from tools.environments.docker import find_docker

logger = logging.getLogger(__name__)

# python-slim is pulled from the same registry as the sandbox images; override
# with HERMES_EGRESS_PROXY_IMAGE if you mirror images internally.
PROXY_IMAGE = os.environ.get("HERMES_EGRESS_PROXY_IMAGE", "python:3.13-slim")
PROXY_PORT = 8888
PROXY_ALIAS = "egress-proxy"  # stable DNS name on the internal network
_SERVER_SCRIPT = os.path.join(os.path.dirname(__file__), "egress_proxy_server.py")
# Label stamped on every egress network/proxy so prune_egress_proxies() (and
# users, via `docker ps --filter label=...`) can enumerate what we created.
_EGRESS_LABEL = "hermes.egress"

# Serialize provisioning so concurrent sandbox starts don't race on the same
# network/proxy (T-1.1: isolate shared mutable state behind an explicit gate).
_lock = threading.Lock()


@dataclass
class EgressProxy:
    """Result of provisioning: the internal network name and proxy env vars."""

    network: str
    proxy_env: dict = field(default_factory=dict)


def _allowlist_hash(allowlist: list[str]) -> str:
    canonical = ",".join(sorted(h.strip().lower() for h in allowlist if h.strip()))
    return hashlib.sha256(canonical.encode()).hexdigest()[:10]


def _names(allowlist: list[str]) -> tuple[str, str]:
    h = _allowlist_hash(allowlist)
    return f"hermes-egress-{h}", f"hermes-egress-proxy-{h}"


def _docker(*args: str, timeout: int = 120) -> subprocess.CompletedProcess:
    exe = find_docker() or "docker"
    return subprocess.run(
        [exe, *args], capture_output=True, text=True, timeout=timeout, check=False
    )


def _outbound_network() -> str:
    """Name of the runtime's default external network (proxy's second home)."""
    exe = find_docker() or "docker"
    return "podman" if os.path.basename(exe).startswith("podman") else "bridge"


def _proxy_env() -> dict:
    url = f"http://{PROXY_ALIAS}:{PROXY_PORT}"
    no_proxy = "localhost,127.0.0.1,::1"
    return {
        "HTTP_PROXY": url, "HTTPS_PROXY": url,
        "http_proxy": url, "https_proxy": url,
        "NO_PROXY": no_proxy, "no_proxy": no_proxy,
    }


def _network_exists(network: str) -> bool:
    return _docker("network", "inspect", network).returncode == 0


def _ensure_network(network: str) -> None:
    if _network_exists(network):
        return
    # --internal: containers on this network have no external connectivity.
    res = _docker(
        "network", "create", "--internal", "--label", f"{_EGRESS_LABEL}=1", network
    )
    if res.returncode != 0 and not _network_exists(network):
        # Lost a race or a real failure — re-check, then surface.
        raise RuntimeError(f"failed to create egress network {network}: {res.stderr.strip()}")


def _proxy_running(proxy: str) -> bool:
    res = _docker("inspect", "-f", "{{.State.Running}}", proxy)
    return res.returncode == 0 and res.stdout.strip() == "true"


def _proxy_exists(proxy: str) -> bool:
    return _docker("inspect", "-f", "{{.State.Status}}", proxy).returncode == 0


def _start_proxy(network: str, proxy: str, allowlist: list[str]) -> None:
    """Create (or start) the dual-homed forward-proxy container."""
    if _proxy_running(proxy):
        return
    if _proxy_exists(proxy):
        # Exists but stopped — restart it (config is baked in at create time).
        _docker("start", proxy)
        if _proxy_running(proxy):
            return
        # Stale/broken container; remove and recreate cleanly.
        _docker("rm", "-f", proxy)

    run = _docker(
        "run", "-d",
        "--name", proxy,
        "--label", f"{_EGRESS_LABEL}=1",
        "--restart", "unless-stopped",
        "--network", network,
        "--network-alias", PROXY_ALIAS,
        "--cap-drop", "ALL",
        "--security-opt", "no-new-privileges",
        "--pids-limit", "256",
        "-e", f"EGRESS_ALLOWLIST={','.join(allowlist)}",
        "-e", f"PROXY_PORT={PROXY_PORT}",
        "-v", f"{_SERVER_SCRIPT}:/egress_proxy_server.py:ro",
        PROXY_IMAGE,
        "python", "/egress_proxy_server.py",
    )
    if run.returncode != 0:
        # Concurrent create is fine if the other one is now running.
        if _proxy_running(proxy):
            return
        raise RuntimeError(f"failed to start egress proxy {proxy}: {run.stderr.strip()}")

    # Dual-home the proxy onto the runtime's default network so IT can reach the
    # internet (the sandbox stays internal-only and must go through the proxy).
    outbound = _outbound_network()
    conn = _docker("network", "connect", outbound, proxy)
    if conn.returncode != 0 and "already exists" not in conn.stderr.lower():
        # Without outbound connectivity every proxied request would 502 forever.
        # Remove the half-configured proxy and raise so the caller fails closed.
        _docker("rm", "-f", proxy)
        raise RuntimeError(
            f"failed to connect egress proxy to {outbound!r} network: {conn.stderr.strip()}"
        )


def ensure_allowlisted_network(allowlist: list[str]) -> EgressProxy:
    """Provision (idempotently) the internal network + filtered proxy.

    Returns the network name to attach the sandbox to and the proxy env vars to
    inject. Raises on unrecoverable provisioning failure — callers should treat a
    raise as "cut egress" (fail-closed), never as "fall back to open network".
    """
    allowlist = list(allowlist or [])
    network, proxy = _names(allowlist)
    with _lock:
        _ensure_network(network)
        _start_proxy(network, proxy, allowlist)
    logger.info(
        "Egress allowlist active: network=%s proxy=%s allowlist=%s",
        network, proxy, allowlist or "DENY-ALL",
    )
    return EgressProxy(network=network, proxy_env=_proxy_env())


def prune_egress_proxies() -> int:
    """Best-effort removal of egress proxies/networks with no sandbox attached.

    Provisioning is lazy and shared (one proxy per distinct allowlist), so
    nothing tears these down per-sandbox. Called from
    ``cleanup_all_environments``; a proxy whose network still has a non-proxy
    container attached (e.g. a sandbox from another hermes process, or one
    whose async ``cleanup()`` hasn't finished detaching yet — such proxies are
    caught on the next prune) is left alone. Returns the number of networks
    removed; failures are logged only.
    """
    removed = 0
    with _lock:
        nets = _docker("network", "ls", "--filter", f"label={_EGRESS_LABEL}",
                       "--format", "{{.Name}}")
        if nets.returncode != 0:
            logger.debug("Egress prune: could not list networks: %s", nets.stderr.strip())
            return 0
        for network in nets.stdout.split():
            insp = _docker("network", "inspect", "-f",
                           "{{range .Containers}}{{.Name}} {{end}}", network)
            if insp.returncode != 0:
                continue
            attached = insp.stdout.split()
            proxy = network.replace("hermes-egress-", "hermes-egress-proxy-", 1)
            if any(name != proxy for name in attached):
                continue  # still in use by a sandbox
            _docker("rm", "-f", proxy)
            res = _docker("network", "rm", network)
            if res.returncode == 0:
                removed += 1
            else:
                logger.debug("Egress prune: could not remove %s: %s",
                             network, res.stderr.strip())
    if removed:
        logger.info("Pruned %d unused egress proxy network(s)", removed)
    return removed
