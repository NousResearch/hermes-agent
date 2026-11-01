"""Docker-container hygiene for stdio MCP servers spawned as ``docker run ...`` (#138312).

The terminal-tool sandbox reaps its ``hermes-*`` containers (``tools/environments/docker.py``),
but a ``mcp_servers`` entry whose ``command`` is ``docker`` gets no such care: the stdio child is
the ``docker run`` CLI client, and when this process dies ungracefully (SIGKILL, crash, gateway
restart) the client's death does not stop the container — ``--rm`` only fires on the client's own
graceful path. Restart loops accumulate one orphaned container per docker-based server per restart.

Fix: label every spawned container with its server name and owning process
(``pid:create_time``, the identity ``hermes_cli.process_identity`` already uses for its ledger,
so PID reuse cannot resurrect a dead owner), and reap same-server containers whose owner is
provably dead before spawning a new one — running ones included, since that is the orphan's usual
shape. Containers without an owner label (older Hermes, user-managed) are never touched.
"""

import logging
import os
import re

logger = logging.getLogger("tools.mcp_tool")

# Docker label values must match [a-zA-Z0-9_.-] and stay <=63 chars to round-trip
# through `docker ps --filter label=key=value` (same constraint as tools/environments/docker.py).
_LABEL_VALUE_OK_RE = re.compile(r"[^A-Za-z0-9_.-]")

_SERVER_LABEL = "hermes-mcp-server"
_OWNER_LABEL = "hermes-mcp-owner"
_PROFILE_LABEL = "hermes-profile"


def _sanitize_label_value(value: str) -> str:
    """Lossy coercion into a Docker label-safe form; empty/invalid input becomes ``"unknown"``."""
    if not isinstance(value, str) or not value:
        return "unknown"
    return _LABEL_VALUE_OK_RE.sub("_", value)[:63] or "unknown"


def is_docker_run_command(command: str, args: list) -> bool:
    """True for ``docker run ...`` argv (absolute path and ``docker.exe`` included)."""
    basename = os.path.basename(command).lower().removesuffix(".exe")
    return basename == "docker" and bool(args) and args[0] == "run"


def _owner_label_value() -> str:
    """``pid:create_time`` for this process; ``create_time`` omitted when psutil can't say."""
    from hermes_cli.process_identity import _process_create_time

    create = _process_create_time()
    if create is None:
        return str(os.getpid())
    return f"{os.getpid()}:{create:.3f}"


def _has_server_label(args: list) -> bool:
    """True when the user already manages the ``hermes-mcp-server`` label themselves."""
    tokens = iter(args)
    for token in tokens:
        if token == "--label":
            if next(tokens, "").startswith(f"{_SERVER_LABEL}="):
                return True
        elif token.startswith(f"--label={_SERVER_LABEL}="):
            return True
    return False


def inject_docker_run_labels(
    server_name: str, command: str, args: list
) -> tuple[str, list]:
    """Add the tracking labels to a ``docker run`` argv, right after ``run``; anything else
    (other commands, other subcommands, an explicit ``hermes-mcp-server`` label) passes through
    untouched. Labels sit before the image, which is where ``docker run`` accepts them."""
    if not is_docker_run_command(command, args) or _has_server_label(args[1:]):
        return command, args
    from hermes_cli.profiles import get_active_profile_name

    try:
        profile = get_active_profile_name() or "default"
    except Exception:
        profile = "default"
    labels = [
        "--label",
        f"{_SERVER_LABEL}={_sanitize_label_value(server_name)}",
        "--label",
        f"{_OWNER_LABEL}={_owner_label_value()}",
        "--label",
        f"{_PROFILE_LABEL}={_sanitize_label_value(profile)}",
    ]
    return command, [args[0], *labels, *args[1:]]


def _owner_alive(owner: str) -> bool | None:
    """Whether the ``pid[:create_time]`` owner label names a live process; ``None`` = can't tell
    (malformed label, psutil unavailable) — callers treat that as alive (fail-closed: no reap)."""
    pid_s, _, create_s = str(owner).partition(":")
    try:
        pid = int(pid_s)
        create = float(create_s) if create_s else None
    except ValueError:
        return None
    if pid <= 0:
        return None
    from hermes_cli.process_identity import _pid_alive_matches

    return _pid_alive_matches(pid, create)


def reap_orphaned_docker_mcp_containers(
    server_name: str, docker_exe: str, *, timeout: float = 15.0
) -> int:
    """Remove this server's labeled containers whose owning Hermes process is provably dead.

    Runs before every stdio (re)connect of a docker-based server. A container with a live owner
    (this or a sibling process — any profile) is never touched; neither is one without an owner
    label, which Hermes did not spawn. ``rm -f`` is safe here precisely because the owner check
    already proved no live client holds the container. Best-effort (``_docker_query``): Docker
    missing, listing failures and per-container errors log at debug and stop nothing else.
    Returns the count removed.
    """
    from tools.environments.docker import _docker_query

    def _query(argv: list[str], *, fail: str):
        return _docker_query(argv, timeout=timeout, fail=fail)

    server_filter = "label=" + _SERVER_LABEL + "=" + _sanitize_label_value(server_name)
    listing = _query(
        [docker_exe, "ps", "-a", "--filter", server_filter, "--format", "{{.ID}}"],
        fail="MCP docker reaper: docker ps failed: %s",
    )
    if listing is None or listing.returncode != 0:
        return 0

    owner_template = f'{{{{index .Config.Labels "{_OWNER_LABEL}"}}}}'
    removed = 0
    for cid in (line.strip() for line in listing.stdout.splitlines() if line.strip()):
        owner = _query(
            [docker_exe, "inspect", "--format", owner_template, cid],
            fail=f"MCP docker reaper: docker inspect {cid[:12]} failed: %s",
        )
        if owner is None or owner.returncode != 0:
            continue
        owner_value = owner.stdout.strip()
        if not owner_value or owner_value == "<no value>":
            continue  # not spawned by Hermes — never touch user-managed containers
        if _owner_alive(owner_value) is not False:
            continue  # live or unverifiable owner: fail closed
        result = _query(
            [docker_exe, "rm", "-f", cid],
            fail=f"MCP docker reaper: docker rm {cid[:12]} failed: %s",
        )
        if result is not None and result.returncode == 0:
            removed += 1
            logger.info(
                "MCP server '%s': reaped orphaned docker container %s (owner %s no longer running)",
                server_name,
                cid[:12],
                owner_value,
            )
    return removed
