"""Auto host-path mapping for Docker backend bind mounts (#50798, #72389).

When the Hermes gateway itself runs inside a Docker container and uses the Docker
terminal backend, the bind-mount sources generated for sandbox containers are
GATEWAY-internal paths (e.g. ``/opt/data/cache/web``). The Docker daemon interprets
bind-mount sources as HOST paths, so those mounts point at empty or non-existent
host directories — sandboxes get empty dirs where content was expected.

This module introspects the gateway's OWN container through ``docker inspect`` (the
Docker socket/CLI is already required to spawn sandbox containers, so this adds no
new security surface) and builds a ``{container_path: host_path}`` mapping. Callers
(:func:`apply_host_mapping`) translate auto-generated bind-mount sources to the host
view; every failure path degrades to the pre-fix behavior with WARNING-level logs.

Resolution: auto-detection (this module) or an empty mapping — identical to the
pre-feature behavior. A manual ``terminal.docker_host_path_mapping`` config override
is layered on top of this in Phase 3.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Probe paths; module-level so tests can point them at temp fixtures.
_DOCKERENV_PATHS = ("/.dockerenv", "/run/.containerenv")
_ETC_HOSTNAME_PATH = "/etc/hostname"
_PROC_1_CGROUP_PATH = "/proc/1/cgroup"
_SELF_CGROUP_PATH = "/proc/self/cgroup"
_SELF_MOUNTINFO_PATH = "/proc/self/mountinfo"

# docker-<64hex>.scope (cgroup v2 systemd driver), cri-containerd-<64hex>,
# plain containerd parent paths; also matches cgroup v1 line contents.
_CONTAINER_ID_IN_CGROUP = re.compile(r"(?:docker|cri-containerd)-([0-9a-f]{64})")
_CONTAINER_ID_IN_MOUNTINFO = re.compile(r"/docker/containers/([0-9a-f]{64})/")
# Docker's default container hostname is the 12-char short id; longer hex is also valid.
_HEX_CONTAINER_ID = re.compile(r"^[0-9a-f]{12,64}$")
# Names docker/podman accept on the CLI ([a-zA-Z0-9][a-zA-Z0-9_.-]* prefix); used only
# as a last-resort inspect candidate when the hostname is not hex-shaped.
_CONTAINER_NAME = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_.-]*$")

_INSPECT_TIMEOUT_S = 10.0


def _read_text(path: str) -> str:
    try:
        with open(path, encoding="utf-8-sig", errors="replace") as fh:
            return fh.read()
    except OSError:
        return ""


def is_running_in_docker() -> bool:
    """True when THIS process is inside a Docker (or podman) container.

    ``/.dockerenv`` is written by Docker itself, ``/run/.containerenv`` by podman; the
    cgroup probe covers runtimes that create neither. Bare metal, VMs and LXC return
    False on purpose — this feature targets Docker-in-Docker gateways only, and their
    bind-mount sources are already host-correct.
    """
    if any(os.path.exists(path) for path in _DOCKERENV_PATHS):
        return True
    return bool(re.search(r"docker|containerd", _read_text(_PROC_1_CGROUP_PATH), re.IGNORECASE))


def _container_id_candidates() -> list[str]:
    """Ordered ``docker inspect`` candidates: hex hostname (Docker's default short id),
    cgroup-embedded full ids, mountinfo container-state ids, then a custom hostname."""
    candidates: list[str] = []

    def add(candidate: str) -> None:
        if candidate and candidate not in candidates:
            candidates.append(candidate)

    hostname = _read_text(_ETC_HOSTNAME_PATH).strip()
    if hostname and _HEX_CONTAINER_ID.match(hostname):
        add(hostname)
    for match in _CONTAINER_ID_IN_CGROUP.finditer(_read_text(_SELF_CGROUP_PATH)):
        add(match.group(1))
    for match in _CONTAINER_ID_IN_MOUNTINFO.finditer(_read_text(_SELF_MOUNTINFO_PATH)):
        add(match.group(1))
    if hostname and _CONTAINER_NAME.match(hostname):
        add(hostname)  # user-set --hostname/--name: still a valid inspect argument
    return candidates


def get_own_container_id() -> Optional[str]:
    """Best container-id candidate for this process, or None when detection fails."""
    candidates = _container_id_candidates()
    return candidates[0] if candidates else None


def _find_docker_cli() -> Optional[str]:
    """Docker CLI locator, lazy-imported so this module stays import-light and
    cycle-free (docker.py imports it); stdlib which() is the degradation fallback."""
    try:
        from tools.environments.docker import find_docker
        return find_docker()
    except Exception:  # pragma: no cover - a broken import env must not sink detection
        logger.debug("docker_host_mapping: find_docker import failed", exc_info=True)
        return shutil.which("docker") or shutil.which("podman")


def _inspect_mounts(docker_exe: str, container_id: str) -> Optional[list[dict[str, Any]]]:
    """Mounts of *container_id* via ``docker inspect -f {{json .Mounts}}`` (full-JSON
    inspect fallback for CLIs without template support); None = this candidate failed."""
    for argv in ([docker_exe, "inspect", "--format", "{{json .Mounts}}", container_id],
                 [docker_exe, "inspect", container_id]):
        try:
            result = subprocess.run(argv, capture_output=True, text=True, encoding="utf-8",
                                    errors="replace", check=False,
                                    timeout=_INSPECT_TIMEOUT_S)
        except (OSError, subprocess.TimeoutExpired) as exc:
            logger.warning("docker_host_mapping: '%s inspect' failed: %s", docker_exe, exc)
            return None
        if result.returncode != 0:
            continue
        try:
            data = json.loads(result.stdout.strip() or "null")
        except json.JSONDecodeError:
            continue
        if "{{json .Mounts}}" not in argv:
            # Full inspect: array of container objects — read the first entry's Mounts.
            data = (data[0].get("Mounts") if isinstance(data, list) and data
                    and isinstance(data[0], dict) else None)
        if isinstance(data, list):
            return [m for m in data if isinstance(m, dict)]
    return None


def get_own_mounts() -> list[dict[str, Any]]:
    """Own container's Mounts list; empty list + WARNING on any failure (graceful:
    the caller then keeps today's unmapped bind-mount sources)."""
    docker_exe = _find_docker_cli()
    if docker_exe is None:
        logger.warning(
            "docker_host_mapping: no docker/podman CLI reachable from the gateway; cannot "
            "introspect its own bind mounts — sandbox bind mounts keep gateway-internal "
            "source paths (see user-guide/docker.md, 'Troubleshooting').")
        return []
    for candidate in _container_id_candidates():
        mounts = _inspect_mounts(docker_exe, candidate)
        if mounts is not None:
            return mounts
    logger.warning(
        "docker_host_mapping: docker inspect could not resolve this container's id "
        "(hostname/cgroup/mountinfo candidates all failed); sandbox bind mounts keep "
        "gateway-internal source paths.")
    return []


# Process-scoped memo: the gateway's own mounts never change while it runs, and the
# sandbox builder calls this on every container creation.
_auto_mapping_cache: Optional[dict[str, str]] = None


def _is_mapable_bind(mount: dict[str, Any]) -> tuple[str, str]:
    """``(container_path, host_path)`` for a user-visible bind mount; ``("", "")`` otherwise.

    Named volumes and tmpfs are excluded: volume sources are daemon-managed
    (``/var/lib/docker/volumes/...``), tmpfs sources are empty — neither is a host
    directory the gateway can read. A bind of ``/`` is too broad to substitute.
    """
    if str(mount.get("Type", "")).lower() != "bind":
        return "", ""
    source = str(mount.get("Source") or "").rstrip("/")
    destination = str(mount.get("Destination") or "").rstrip("/")
    if not source.startswith("/") or not destination.startswith("/"):
        return "", ""
    if destination == "/":
        return "", ""
    return destination, source


def build_auto_host_mapping() -> dict[str, str]:
    """``{container_path: host_path}`` from the gateway's own bind mounts, ordered
    longest-prefix-first; empty when detection fails (never raises)."""
    global _auto_mapping_cache
    if _auto_mapping_cache is not None:
        return dict(_auto_mapping_cache)
    mapping: dict[str, str] = {}
    for mount in get_own_mounts():
        container_path, host_path = _is_mapable_bind(mount)
        if container_path:
            mapping[container_path] = host_path
    _auto_mapping_cache = dict(sorted(mapping.items(), key=lambda kv: len(kv[0]), reverse=True))
    if _auto_mapping_cache:
        logger.debug("docker_host_mapping: auto host mapping %s", _auto_mapping_cache)
    return dict(_auto_mapping_cache)


def apply_host_mapping(path: str, mapping: dict[str, str]) -> str:
    """Translate *path* from the gateway/container view to the host view via the longest
    matching container prefix; *path* unchanged when no prefix matches (no-op fallback)."""
    if not mapping or not path:
        return path
    best_prefix = ""
    best_host = ""
    for container_prefix, host_prefix in mapping.items():
        normalized = str(container_prefix).rstrip("/")
        if not normalized:
            continue
        if path == normalized or path.startswith(normalized + "/"):
            if len(normalized) > len(best_prefix):
                best_prefix, best_host = normalized, str(host_prefix).rstrip("/")
    if not best_prefix:
        return path
    return best_host + path[len(best_prefix):]


_fallback_warned = False


def resolve_host_mapping() -> dict[str, str]:
    """The mapping to apply to auto-generated bind-mount sources: auto-detection
    (Docker gateway only), else an empty mapping — identical to pre-feature behavior.
    The manual ``terminal.docker_host_path_mapping`` override is added in Phase 3."""
    if not is_running_in_docker():
        return {}
    auto = build_auto_host_mapping()
    global _fallback_warned
    if not auto and not _fallback_warned:
        _fallback_warned = True
        logger.warning(
            "docker_host_mapping: gateway looks containerized but no host mapping could be "
            "built (docker inspect unavailable?); sandbox bind mounts fall back to "
            "gateway-internal paths and may appear empty inside sandboxes.")
    return dict(auto)


def reset_host_mapping_caches() -> None:
    """Clear all process caches (auto mapping, warn-once flag); tests call this around
    detection-fixture changes."""
    global _auto_mapping_cache, _fallback_warned
    _auto_mapping_cache = None
    _fallback_warned = False
