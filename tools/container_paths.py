"""Translate paths across the Docker terminal's bind mounts, in both directions.

With the Docker terminal backend the agent loop runs on the host while the
terminal runs inside the sandbox, so the two sides name the same file by
different paths.

Container -> host (``to_host_dir``): the terminal's configured cwd is a path
such as ``/workspace``, which exists only as the bind-mount target; host-side
code that scans it must use the mount source. Read-only mounts are excluded in
this direction, so a path resolves only through a mount the sandbox can write.

Host -> container (``to_container_path``): host-side code that tells the
model a path, such as the skill loader announcing a skill's directory, must
give the path the sandboxed terminal can open. Read-only mounts are included
here, because reading bundled files through one is exactly the use case.

Backend and mounts are read the way the terminal tool reads them: through
``tools.terminal_scope.terminal_env``, so a multiplexed process translates
with the active profile's policy rather than the launch process's. When no
scope is bound and the terminal tool has not yet mirrored config.yaml into the
environment, ``terminal.backend`` and ``terminal.docker_volumes`` are read
from config.yaml directly. Nothing here writes to ``os.environ``.

Return values when there is nothing to translate (a non-Docker backend, no
matching mount, an unreadable config): ``to_host_dir`` and
``to_container_path`` return None, and ``container_mount_map`` and
``host_mount_map`` return an empty list. Callers keep the path they already
had, for example ``to_container_path(p) or p``. Under a refusal scope the
scoped read raises ``TerminalPolicyUnavailable``, as it does for the terminal
tool itself.

``gateway/platforms/base.py`` carries a private equivalent for media
delivery (``_translate_docker_container_media_path``); the two could share
this module in a follow-up.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Scope-aware reads of the terminal policy.
# ---------------------------------------------------------------------------


def _scoped_terminal_env(name: str) -> str:
    """Read a ``TERMINAL_*`` value through the per-turn terminal scope.

    Only an ImportError falls back to ``os.environ``, matching
    ``agent.runtime_cwd.scope_terminal_cwd``: an active refusal scope must
    raise rather than resolve the launch profile's value.
    """
    try:
        from tools.terminal_scope import terminal_env
    except ImportError:
        return os.environ.get(name, "")
    return terminal_env(name, "")


def _scope_bound() -> bool:
    try:
        from tools.terminal_scope import get_terminal_scope
    except ImportError:
        return False
    return get_terminal_scope() is not None


def _config_terminal_value(config_key: str) -> Optional[str]:
    """``terminal.<config_key>`` from config.yaml, rendered as its env string.

    Returns None when the key is unset or the config cannot be read.
    """
    try:
        from hermes_cli.config import _terminal_env_value, load_config_readonly

        terminal_cfg = (load_config_readonly() or {}).get("terminal") or {}
        value = terminal_cfg.get(config_key)
        if value is None:
            return None
        return _terminal_env_value(value)
    except Exception:
        logger.debug("could not read terminal.%s from config", config_key, exc_info=True)
        return None


def _terminal_setting(env_name: str, config_key: str) -> str:
    """The effective terminal setting: scoped env first, then config.yaml.

    A bound scope is a complete policy, so config.yaml is consulted only when
    no scope is bound and the environment does not carry the value yet (the
    terminal tool mirrors config into it lazily, on first use).
    """
    value = (_scoped_terminal_env(env_name) or "").strip()
    if value or _scope_bound():
        return value
    return (_config_terminal_value(config_key) or "").strip()


def _docker_backend_active() -> bool:
    """True when the terminal tool will run commands in a Docker sandbox.

    A local or SSH backend needs no translation: the agent's shell shares the
    host filesystem, so the host path it is handed is already correct.
    """
    return _terminal_setting("TERMINAL_ENV", "backend").lower() == "docker"


def _volume_specs() -> list[tuple[str, str, bool]]:
    """Parse the Docker volumes into ``[(host, container, read_only)]``.

    Returns an empty list on a non-Docker backend, or when the value is unset
    or not a JSON list. Specs that are not absolute bind mounts (named
    volumes, relative paths) are skipped.
    """
    if not _docker_backend_active():
        return []
    raw = _terminal_setting("TERMINAL_DOCKER_VOLUMES", "docker_volumes")
    if not raw:
        return []
    try:
        specs = json.loads(raw)
    except (ValueError, TypeError):
        logger.debug("TERMINAL_DOCKER_VOLUMES is not valid JSON; no translation")
        return []
    if not isinstance(specs, list):
        return []

    triples: list[tuple[str, str, bool]] = []
    for spec in specs:
        if not isinstance(spec, str):
            continue
        parts = spec.split(":")
        if len(parts) < 2:
            continue
        host_raw, container_raw = parts[0], parts[1]
        mode = parts[2] if len(parts) > 2 else ""
        read_only = "ro" in {m.strip() for m in mode.split(",")}
        if not container_raw.startswith("/"):
            continue
        if not host_raw.startswith(("/", "~")):
            continue
        host = str(Path(host_raw).expanduser()).rstrip("/")
        container = container_raw.rstrip("/")
        if host and container:
            triples.append((host, container, read_only))
    return triples


# ---------------------------------------------------------------------------
# Container -> host. Used when host-side code opens a path the sandbox names.
# ---------------------------------------------------------------------------


def container_mount_map() -> list[tuple[str, str]]:
    """Return ``[(container_prefix, host_prefix)]`` for writable mounts.

    Longest container prefix first. Empty on a non-Docker backend or when no
    usable mount is configured.
    """
    pairs = [(container, host) for host, container, ro in _volume_specs() if not ro]
    pairs.sort(key=lambda p: len(p[0]), reverse=True)
    return pairs


def to_host_dir(declared: str) -> Optional[str]:
    """Translate a container directory path to its host equivalent.

    Context-file discovery uses it to find the host directory behind the
    terminal's configured cwd. Returns None on a non-Docker backend, when no
    writable mount matches, or when the mapped location is not a directory.
    """
    if not declared or not declared.startswith("/"):
        return None
    for container, host in container_mount_map():
        if declared == container or declared.startswith(container + "/"):
            mapped = Path(host + declared[len(container):])
            try:
                if mapped.is_dir():
                    return str(mapped.resolve())
            except OSError:
                logger.debug("could not stat %s", mapped, exc_info=True)
                return None
            return None
    return None


# ---------------------------------------------------------------------------
# Host -> container. Used when host-side code names a path FOR the agent.
# ---------------------------------------------------------------------------


def host_mount_map() -> list[tuple[str, str]]:
    """Return ``[(host_prefix, container_prefix)]``, longest host first.

    Unlike :func:`container_mount_map` this KEEPS read-only mounts, and adds
    the skill directories that the Docker backend bind-mounts read-only under
    ``/root/.hermes`` without listing them in ``docker_volumes`` (see
    ``tools.credential_files.get_skills_directory_mount``). Empty on a
    non-Docker backend.
    """
    if not _docker_backend_active():
        return []

    # One host directory can be mounted more than once, for example writable
    # at ``/workspace`` and read-only at a second path. Ties on host prefix
    # resolve to the writable alias, so a caller naming a file it intends to
    # write is not told a read-only path. A self-map (host mounted at its own
    # path) needs no rewrite and would mask the real container location.
    triples = [t for t in _volume_specs() if t[0] != t[1]]

    try:
        from tools.credential_files import _skill_dir_roots

        for host_dir, container_root in _skill_dir_roots("/root/.hermes"):
            host = str(host_dir).rstrip("/")
            if host:
                triples.append((host, container_root.rstrip("/"), True))
    except Exception:
        logger.debug("could not resolve skill directory mounts; omitted", exc_info=True)

    triples.sort(key=lambda x: (-len(x[0]), x[2]))
    return [(host, container) for host, container, _ in triples]


def to_container_path(host_path: str) -> Optional[str]:
    """Translate a host path to the path the agent's sandbox can open.

    Returns None on a non-Docker backend or when no mount matches, so callers
    keep the host path as-is. Purely textual: the host file may not exist yet
    and the container may not be running.
    """
    if not host_path or not str(host_path).startswith("/"):
        return None
    declared = str(Path(host_path)).rstrip("/") or "/"
    for host, container in host_mount_map():
        if declared == host:
            return container
        if declared.startswith(host + "/"):
            return container + declared[len(host):]
    return None
