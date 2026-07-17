"""Docker execution environment for sandboxed command execution.

Security hardened (cap-drop ALL, no-new-privileges, PID limits),
configurable resource limits (CPU, memory, disk), and optional filesystem
persistence via bind mounts.
"""

import hashlib
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Optional

from tools.environments.base import BaseEnvironment, _popen_bash
from tools.environments.local import (
    _HERMES_PROVIDER_ENV_BLOCKLIST,
    _is_hermes_internal_secret,
)

logger = logging.getLogger(__name__)


# Common Docker Desktop install paths checked when 'docker' is not in PATH.
# macOS Intel: /usr/local/bin, macOS Apple Silicon (Homebrew): /opt/homebrew/bin,
# Docker Desktop app bundle: /Applications/Docker.app/Contents/Resources/bin
_DOCKER_SEARCH_PATHS = [
    "/usr/local/bin/docker",
    "/opt/homebrew/bin/docker",
    "/Applications/Docker.app/Contents/Resources/bin/docker",
]

_docker_executable: Optional[str] = None  # resolved once, cached
_ENV_VAR_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _normalize_forward_env_names(forward_env: list[str] | None) -> list[str]:
    """Return a deduplicated list of valid environment variable names."""
    normalized: list[str] = []
    seen: set[str] = set()

    for item in forward_env or []:
        if not isinstance(item, str):
            logger.warning("Ignoring non-string docker_forward_env entry: %r", item)
            continue

        key = item.strip()
        if not key:
            continue
        if not _ENV_VAR_NAME_RE.match(key):
            logger.warning("Ignoring invalid docker_forward_env entry: %r", item)
            continue
        if key in seen:
            continue

        seen.add(key)
        normalized.append(key)

    return normalized


def _normalize_env_dict(env: dict | None) -> dict[str, str]:
    """Validate and normalize a docker_env dict to {str: str}.

    Filters out entries with invalid variable names or non-string values.
    """
    if not env:
        return {}
    if not isinstance(env, dict):
        logger.warning("docker_env is not a dict: %r", env)
        return {}

    normalized: dict[str, str] = {}
    for key, value in env.items():
        if not isinstance(key, str) or not _ENV_VAR_NAME_RE.match(key.strip()):
            logger.warning("Ignoring invalid docker_env key: %r", key)
            continue
        key = key.strip()
        if not isinstance(value, str):
            # Coerce simple scalar types (int, bool, float) to string;
            # reject complex types.
            if isinstance(value, (int, float, bool)):
                value = str(value)
            else:
                logger.warning("Ignoring non-string docker_env value for %r: %r", key, value)
                continue
        normalized[key] = value

    return normalized


def _load_hermes_env_vars() -> dict[str, str]:
    """Load ~/.hermes/.env values without failing Docker command execution."""
    try:
        from hermes_cli.config import load_env

        return load_env() or {}
    except Exception:
        return {}


def resolve_docker_init_env(
    *,
    env: dict | None = None,
    forward_env: list[str] | None = None,
) -> dict[str, str]:
    """Resolve the exact environment captured by Docker's shell snapshot.

    This is the authority for both execution and durable-approval binding.
    Explicit ``docker_env`` values form the base. Configured forwards and
    session-scoped skill passthroughs then resolve from the live process
    environment, falling back to ``~/.hermes/.env`` just as Docker execution
    historically did.
    """

    exec_env = _normalize_env_dict(env)
    explicit_forward_keys = set(_normalize_forward_env_names(forward_env))
    passthrough_keys: set[str] = set()
    try:
        from tools.env_passthrough import get_all_passthrough

        passthrough_keys = set(get_all_passthrough())
    except Exception:
        pass

    # Explicit docker_forward_env entries are an intentional opt-in and win
    # over the generic Hermes secret blocklist. Implicit skill/config
    # passthroughs remain subject to the credential and internal-secret gates.
    implicit_forward = {
        key for key in passthrough_keys if not _is_hermes_internal_secret(key)
    }
    forward_keys = explicit_forward_keys | (
        implicit_forward - _HERMES_PROVIDER_ENV_BLOCKLIST
    )
    hermes_env = _load_hermes_env_vars() if forward_keys else {}
    for key in sorted(forward_keys):
        value = os.getenv(key)
        if not value:
            value = hermes_env.get(key)
        if value:
            exec_env[key] = value
    return exec_env


def docker_init_env_digest(resolved_env: dict[str, str]) -> str:
    """Return a deterministic binding for the resolved init environment."""

    encoded = json.dumps(
        resolved_env,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


_RUNTIME_FINGERPRINT_LABEL = "hermes-runtime-fingerprint"


def docker_runtime_fingerprint(
    *,
    image: str,
    cwd: str,
    cpu: float,
    memory: int,
    disk: int,
    persistent_filesystem: bool,
    task_id: str,
    volumes: list | None,
    network: bool,
    host_cwd: str | None,
    auto_mount_cwd: bool,
    run_as_host_user: bool,
    extra_args: list | None,
    persist_across_processes: bool,
    init_env_digest: str,
    host_user_spec: str | None = None,
    persistent_mounts: list[str] | None = None,
) -> str:
    """Return a hash-only identity for an effective Docker runtime.

    Cross-process reuse is safe only when every input that can change the
    container's authority or execution target is identical.  Raw environment
    values and host paths are hashed into this identity; they are never placed
    directly in Docker labels.
    """

    normalized_volumes = [
        value.strip()
        for value in (volumes or [])
        if isinstance(value, str) and value.strip() and ":" in value
    ]
    normalized_extra_args = [
        value for value in (extra_args or []) if isinstance(value, str)
    ]
    host_cwd_abs = (
        os.path.abspath(os.path.expanduser(host_cwd)) if host_cwd else ""
    )
    host_cwd_mounted = bool(
        auto_mount_cwd and host_cwd_abs and os.path.isdir(host_cwd_abs)
    )
    payload = {
        "auto_mount_cwd": bool(auto_mount_cwd),
        "cpu": float(cpu or 0),
        "cwd": str(cwd or ""),
        "disk": int(disk or 0),
        "extra_args": normalized_extra_args,
        "host_cwd": host_cwd_abs,
        "host_cwd_mounted": host_cwd_mounted,
        "host_user_spec": str(host_user_spec or ""),
        "image": str(image or ""),
        "init_env_digest": str(init_env_digest or ""),
        "memory": int(memory or 0),
        "network": bool(network),
        "persist_across_processes": bool(persist_across_processes),
        "persistent_mounts": [
            value.strip()
            for value in (persistent_mounts or [])
            if isinstance(value, str) and value.strip() and ":" in value
        ],
        "persistent_filesystem": bool(persistent_filesystem),
        "run_as_host_user": bool(run_as_host_user),
        "task_id": str(task_id or "default"),
        "version": 1,
        "volumes": normalized_volumes,
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    # Docker label values are kept below 63 characters for portability.
    return "v1-" + hashlib.sha256(encoded).hexdigest()[:48]


def resolve_docker_persistent_mount_specs(
    *,
    persistent_filesystem: bool,
    task_id: str,
    bind_host_cwd: bool,
    workspace_explicitly_mounted: bool,
    sandbox_root: str | None = None,
) -> list[str]:
    """Resolve Hermes-managed persistent binds without creating directories.

    The host paths are runtime identity inputs.  Resolving them before an
    approval decision lets callers freeze ``TERMINAL_SANDBOX_DIR`` alongside
    the rest of the Docker authority surface without creating storage as a
    side effect.
    """

    if not persistent_filesystem:
        return []

    if sandbox_root is None:
        custom = os.getenv("TERMINAL_SANDBOX_DIR")
        if custom:
            sandbox_root = custom
        else:
            from hermes_constants import get_hermes_home

            sandbox_root = str(get_hermes_home() / "sandboxes")

    task_root = Path(sandbox_root).expanduser().absolute() / "docker" / task_id
    specs = [f"{task_root / 'home'}:/root"]
    if not bind_host_cwd and not workspace_explicitly_mounted:
        specs.append(f"{task_root / 'workspace'}:/workspace")
    return specs


def _normalize_docker_extra_args(
    extra_args: list | None,
    *,
    network: bool,
) -> list[str]:
    """Validate arbitrary ``docker run`` arguments and the lockdown contract."""

    if extra_args is not None and not isinstance(extra_args, list):
        logger.warning("docker_extra_args config is not a list: %r", extra_args)
        return []

    validated: list[str] = []
    for arg in extra_args or []:
        if not isinstance(arg, str):
            logger.warning("Ignoring non-string docker_extra_args entry: %r", arg)
            continue
        validated.append(arg)

    if network:
        return validated

    # Extra args are appended after Hermes' ``--network=none`` flag. Docker
    # accepts the last network option, so a contradictory user-supplied flag
    # would silently re-enable egress on a freshly-created container.
    index = 0
    while index < len(validated):
        arg = validated[index]
        value: str | None = None
        if arg in {"--network", "--net"}:
            if index + 1 >= len(validated):
                raise ValueError(
                    "docker_network=false requires any docker_extra_args "
                    "network override to be exactly 'none'"
                )
            value = validated[index + 1]
            index += 1
        elif arg.startswith("--network=") or arg.startswith("--net="):
            value = arg.split("=", 1)[1]

        if value is not None and value.strip().lower() != "none":
            raise ValueError(
                "docker_network=false conflicts with docker_extra_args "
                f"network override {value!r}; remove it or use 'none'"
            )
        index += 1

    return validated


def resolve_docker_implicit_mount_specs() -> list[str]:
    """Resolve existing skill/credential/cache bind mounts for this runtime."""

    specs: list[str] = []
    try:
        from tools.credential_files import (
            get_cache_directory_mounts,
            get_credential_file_mounts,
            get_skills_directory_mount,
        )

        for mount_entry in get_credential_file_mounts():
            src = Path(mount_entry["host_path"])
            if src.is_file():
                specs.append(
                    f"{mount_entry['host_path']}:{mount_entry['container_path']}:ro"
                )
            elif src.is_dir():
                logger.warning(
                    "Docker: skipping credential mount because source is a "
                    "directory: %s",
                    src,
                )
            else:
                logger.warning(
                    "Docker: skipping credential mount — source not found: %s",
                    src,
                )
        for mount_entry in get_skills_directory_mount():
            src = Path(mount_entry["host_path"])
            if src.is_dir():
                specs.append(
                    f"{mount_entry['host_path']}:{mount_entry['container_path']}:ro"
                )
            else:
                logger.warning(
                    "Docker: skipping skills mount — source is not a directory: %s",
                    src,
                )
        for mount_entry in get_cache_directory_mounts():
            src = Path(mount_entry["host_path"])
            if src.is_dir():
                specs.append(
                    f"{mount_entry['host_path']}:{mount_entry['container_path']}:ro"
                )
            else:
                logger.warning(
                    "Docker: skipping cache mount — source is not a directory: %s",
                    src,
                )
    except Exception as exc:
        logger.debug("Docker: could not resolve implicit mounts: %s", exc)
    return specs


# Docker label values must match [a-zA-Z0-9_.-] and stay ≤63 chars to round-trip
# safely through `docker ps --filter label=key=value`. Profile and task names
# can technically contain other characters; sanitize defensively.
_LABEL_VALUE_OK_RE = re.compile(r"[^A-Za-z0-9_.-]")


def _sanitize_label_value(value: str) -> str:
    """Coerce *value* into a Docker label-safe form (alnum + ``_.-``, ≤63 chars).

    Empty or all-invalid inputs collapse to ``"unknown"`` so the resulting
    label is always queryable. Used at container-create time; never round-trip
    a sanitized value back into application logic.
    """
    if not isinstance(value, str) or not value:
        return "unknown"
    cleaned = _LABEL_VALUE_OK_RE.sub("_", value)
    cleaned = cleaned[:63] or "unknown"
    return cleaned


def _get_active_profile_name() -> str:
    """Return the active Hermes profile name, or ``"default"`` on any error.

    Resolved at container-create time so a single container is permanently
    tagged with the profile that created it. Profile switches inside the
    same process don't retroactively relabel running containers.
    """
    try:
        from hermes_cli.profiles import get_active_profile_name

        return get_active_profile_name() or "default"
    except Exception:
        return "default"


def reap_orphan_containers(
    *,
    max_age_seconds: int = 600,
    profile_filter: str | None = None,
    docker_exe: str | None = None,
) -> int:
    """Remove stale hermes-tagged containers left behind by prior processes.

    Targets containers that match all of:

    * ``label=hermes-agent=1`` (created by this codebase)
    * ``status=exited`` (running containers are NEVER reaped — they may
      belong to a sibling Hermes process whose reuse path will pick them
      up; killing them would crash the sibling mid-command)
    * (optional) ``label=hermes-profile=<profile_filter>`` (sweep only the
      caller's profile by default; a hermes process in profile A must not
      tear down profile B's containers)
    * ``State.FinishedAt`` older than *max_age_seconds* ago (so a sibling
      process that just exited and is about to be replaced doesn't get
      its container yanked out from under it)

    Returns the number of containers removed. Best-effort: any failure
    (docker daemon unreachable, slow inspect, parse error) is logged at
    debug level and the function returns whatever it managed before the
    failure. Safe to call repeatedly; idempotent.

    Issue #20561 — this is the safety net for SIGKILL / OOM / crashed
    terminal exits that bypass the ``atexit`` cleanup hook. Without it,
    even with the cleanup-fix in the prior commit, a hard-killed Hermes
    process leaves its container behind permanently because there's no
    subsequent Hermes process scheduled to reuse that exact (task, profile)
    pair.
    """
    docker = docker_exe or find_docker() or "docker"
    filters = ["--filter", "label=hermes-agent=1", "--filter", "status=exited"]
    if profile_filter:
        filters.extend(["--filter", f"label=hermes-profile={_sanitize_label_value(profile_filter)}"])

    try:
        listing = subprocess.run(
            [docker, "ps", "-a", *filters, "--format", "{{.ID}}"],
            capture_output=True, text=True, timeout=15, check=False,
            stdin=subprocess.DEVNULL,
        )
    except (subprocess.TimeoutExpired, OSError) as e:
        logger.debug("orphan reaper docker ps failed: %s", e)
        return 0
    if listing.returncode != 0:
        logger.debug(
            "orphan reaper docker ps returned %d: %s",
            listing.returncode, listing.stderr.strip(),
        )
        return 0

    candidate_ids = [ln.strip() for ln in listing.stdout.splitlines() if ln.strip()]
    if not candidate_ids:
        return 0

    # Inspect each candidate to get FinishedAt; reap only those exited
    # long enough ago.  Doing this per-container (rather than bulk inspect)
    # keeps the failure blast radius to one container at a time.
    import datetime
    now = datetime.datetime.now(datetime.timezone.utc)
    removed = 0
    for cid in candidate_ids:
        finished_at = _container_finished_at(docker, cid)
        if finished_at is None:
            # Couldn't determine age — be conservative and leave it alone.
            continue
        age = (now - finished_at).total_seconds()
        if age < max_age_seconds:
            continue
        try:
            result = subprocess.run(
                [docker, "rm", "-f", cid],
                capture_output=True, text=True, timeout=30,
                stdin=subprocess.DEVNULL,
            )
            if result.returncode == 0:
                removed += 1
                logger.info(
                    "Reaped orphan container %s (exited %d seconds ago)",
                    cid[:12], int(age),
                )
            else:
                logger.debug(
                    "docker rm -f %s failed: %s",
                    cid[:12], result.stderr.strip(),
                )
        except (subprocess.TimeoutExpired, OSError) as e:
            logger.debug("orphan reaper docker rm %s failed: %s", cid[:12], e)
    return removed


def _container_finished_at(docker_exe: str, container_id: str):
    """Parse ``docker inspect`` FinishedAt for *container_id*.

    Returns a timezone-aware datetime, or ``None`` if the field is missing,
    unparseable, or the zero-value ``0001-01-01T00:00:00Z`` Docker emits
    for never-finished containers. ``None`` means "don't reap" — the caller
    leaves the container alone.
    """
    try:
        result = subprocess.run(
            [docker_exe, "inspect", "--format", "{{.State.FinishedAt}}", container_id],
            capture_output=True, text=True, timeout=10, check=False,
            stdin=subprocess.DEVNULL,
        )
    except (subprocess.TimeoutExpired, OSError) as e:
        logger.debug("orphan reaper docker inspect %s failed: %s", container_id[:12], e)
        return None
    if result.returncode != 0:
        return None
    raw = result.stdout.strip()
    if not raw or raw.startswith("0001-01-01"):
        return None
    # Docker emits RFC3339 with nanoseconds (e.g. "2026-05-28T13:45:00.123456789Z").
    # Python's fromisoformat handles microseconds but not nanoseconds; trim.
    import re as _re
    raw = _re.sub(r"(\.\d{6})\d+", r"\1", raw)
    raw = raw.replace("Z", "+00:00")
    try:
        import datetime
        return datetime.datetime.fromisoformat(raw)
    except ValueError as e:
        logger.debug("could not parse FinishedAt %r for %s: %s", raw, container_id[:12], e)
        return None


def find_docker() -> Optional[str]:
    """Locate the docker (or podman) CLI binary.

    Resolution order:
    1. ``HERMES_DOCKER_BINARY`` env var — explicit override (e.g. ``/usr/bin/podman``)
    2. ``docker`` on PATH via ``shutil.which``
    3. ``podman`` on PATH via ``shutil.which``
    4. Well-known macOS Docker Desktop install locations

    Returns the absolute path, or ``None`` if neither runtime can be found.
    """
    global _docker_executable
    if _docker_executable is not None:
        return _docker_executable

    # 1. Explicit override via env var (e.g. for Podman on immutable distros)
    override = os.getenv("HERMES_DOCKER_BINARY")
    if override and os.path.isfile(override) and os.access(override, os.X_OK):
        _docker_executable = override
        logger.info("Using HERMES_DOCKER_BINARY override: %s", override)
        return override

    # 2. docker on PATH
    found = shutil.which("docker")
    if found:
        _docker_executable = found
        return found

    # 3. podman on PATH (drop-in compatible for our use case)
    found = shutil.which("podman")
    if found:
        _docker_executable = found
        logger.info("Using podman as container runtime: %s", found)
        return found

    # 4. Well-known macOS Docker Desktop locations
    for path in _DOCKER_SEARCH_PATHS:
        if os.path.isfile(path) and os.access(path, os.X_OK):
            _docker_executable = path
            logger.info("Found docker at non-PATH location: %s", path)
            return path

    return None


# Security flags applied to every container.
# The container itself is the security boundary (isolated from host).
# We drop all capabilities then add back the minimum needed:
#   DAC_OVERRIDE - root can write to bind-mounted dirs owned by host user
#   CHOWN/FOWNER - package managers (pip, npm, apt) need to set file ownership
#   SETUID/SETGID - the image's init drops from root to the 'hermes'
#       user (via `s6-setuidgid` in the bundled image, or whatever
#       privilege-drop helper a user image uses), which requires these
#       caps. Combined with `no-new-privileges`, the dropped process
#       still cannot escalate back to root, so the security posture is
#       preserved. Omitted entirely when the container starts as a
#       non-root user via --user, since no privilege drop is needed
#       in that mode.
# Block privilege escalation.
# /tmp is size-limited and nosuid but allows exec (needed by pip/npm builds).
#
# Note: ``--pids-limit`` is *not* in this list — it lives in ``resource_args``
# and is gated on ``_cgroup_limits_available(image)`` because it requires the
# ``pids`` cgroup controller to be delegated, which is not the case on hosts
# such as unprivileged LXCs. ``--cpus``/``--memory`` are gated for the same
# reason.
_BASE_SECURITY_ARGS = [
    "--cap-drop", "ALL",
    "--cap-add", "DAC_OVERRIDE",
    "--cap-add", "CHOWN",
    "--cap-add", "FOWNER",
    "--security-opt", "no-new-privileges",
    "--tmpfs", "/tmp:rw,nosuid,size=512m",
    "--tmpfs", "/var/tmp:rw,noexec,nosuid,size=256m",
]

# Default per-container PID limit. Applied as ``--pids-limit`` only when the
# cgroup ``pids`` controller is available (see ``_cgroup_limits_available``).
_DEFAULT_PIDS_LIMIT = "256"

# /run is split out from _BASE_SECURITY_ARGS because s6-overlay images need it
# mounted ``exec``: s6 stage0 later runs ``exec /run/s6/basedir/bin/init``, which
# fails with "Permission denied" (exit 126) on a ``noexec`` mount. For all other
# images we keep the hardened ``noexec`` default.
_RUN_TMPFS_NOEXEC = "--tmpfs", "/run:rw,noexec,nosuid,size=64m"
_RUN_TMPFS_EXEC = "--tmpfs", "/run:rw,exec,nosuid,size=64m"

# Extra caps needed when the container starts as root and an init/entrypoint
# must drop privileges (via `s6-setuidgid`, `gosu`, `su`, or similar).
# Skipped when --user is passed because the container already starts
# unprivileged and never needs to switch.
_PRIVDROP_CAP_ARGS = [
    "--cap-add", "SETUID",
    "--cap-add", "SETGID",
]


def _build_security_args(run_as_host_user: bool, run_exec: bool = False) -> list[str]:
    """Return the security/cap/tmpfs args tailored to the privilege mode.

    ``run_exec`` mounts ``/run`` with ``exec`` instead of the hardened
    ``noexec`` default. This is required for s6-overlay images whose ``/init``
    entrypoint execs ``/run/s6/basedir/bin/init`` during startup; see
    ``_image_uses_init_entrypoint``.
    """
    run_tmpfs = list(_RUN_TMPFS_EXEC if run_exec else _RUN_TMPFS_NOEXEC)
    args = list(_BASE_SECURITY_ARGS) + run_tmpfs
    if run_as_host_user:
        return args
    return args + list(_PRIVDROP_CAP_ARGS)


def _image_uses_init_entrypoint(docker_exe: str, image: str) -> bool:
    """Return True if ``image``'s entrypoint is the s6-overlay ``/init``.

    Such images (e.g. anything built on ``s6-overlay``, including
    ``hermes-agent:latest``) already provide their own PID-1 init and execute
    ``/run/s6/basedir/bin/init`` during stage0 startup. They are incompatible
    with Docker's ``--init`` (two competing PID-1 inits) and with a ``noexec``
    ``/run`` mount. Detection is best-effort: on any inspection failure we
    return False and keep the hardened defaults.
    """
    try:
        result = subprocess.run(
            [docker_exe, "image", "inspect", image,
             "--format", "{{json .Config.Entrypoint}}"],
            capture_output=True,
            text=True,
            timeout=15,
            stdin=subprocess.DEVNULL,
        )
    except (subprocess.SubprocessError, OSError) as e:
        logger.debug("Docker: could not inspect entrypoint for %s: %s", image, e)
        return False
    if result.returncode != 0:
        # Image may not be pulled yet; the run will pull it. Defaults are safe
        # for non-s6 images, so don't block on this.
        logger.debug(
            "Docker: image inspect for %s returned %d (stderr=%s)",
            image, result.returncode, result.stderr.strip(),
        )
        return False
    raw = (result.stdout or "").strip()
    if not raw or raw == "null":
        return False
    try:
        entrypoint = json.loads(raw)
    except (ValueError, TypeError):
        return False
    if isinstance(entrypoint, str):
        entrypoint = [entrypoint]
    if not isinstance(entrypoint, list) or not entrypoint:
        return False
    first = str(entrypoint[0]).strip()
    return first in ("/init", "/package/admin/s6-overlay/command/init")


def _resolve_host_user_spec() -> Optional[str]:
    """Return ``<uid>:<gid>`` for the current host user, or ``None`` on platforms
    where this is not meaningful (e.g. Windows without posix ids).

    We intentionally read ``os.getuid()``/``os.getgid()`` directly rather than
    going through ``getpass``/``pwd`` so this stays cheap and never raises on
    nameless UIDs (nss lookups can fail inside sandboxed launchers).
    """
    get_uid = getattr(os, "getuid", None)
    get_gid = getattr(os, "getgid", None)
    if get_uid is None or get_gid is None:
        return None
    try:
        return f"{get_uid()}:{get_gid()}"
    except Exception:  # pragma: no cover - defensive
        return None


_storage_opt_ok: Optional[bool] = None  # cached result across instances
_cgroup_limits_ok: Optional[bool] = None  # cached result across instances


def _cgroup_limits_available(image: str) -> bool:
    """Probe whether cgroup resource limits work in this environment.

    Tests ``--cpus``, ``--memory`` and ``--pids-limit`` together by spawning
    a throwaway container from *image* (the same sandbox image we are about
    to use for real, so no extra pull and no dependency on a public
    registry). The container runs ``sleep 0`` — sleep is guaranteed to be
    present because the sandbox itself uses ``sleep 2h`` as its long-lived
    entrypoint.

    On hosts where the corresponding cgroup controllers are not delegated
    to this process (typical inside unprivileged LXCs and some rootless
    setups) these flags cause every container start to fail with ``OCI
    runtime error`` / exit 126. The probe runs once per process and the
    result — which is host-wide, not image-specific — is cached.
    """
    global _cgroup_limits_ok
    if _cgroup_limits_ok is not None:
        return _cgroup_limits_ok

    docker_exe = find_docker()
    if not docker_exe or not image:
        _cgroup_limits_ok = False
        return False

    try:
        result = subprocess.run(
            [docker_exe, "run", "--rm",
             "--cpus", "0.5", "--memory", "64m", "--pids-limit", "32",
             image, "sleep", "0"],
            capture_output=True,
            text=True,
            timeout=60,
            stdin=subprocess.DEVNULL,
        )
        _cgroup_limits_ok = result.returncode == 0
        if not _cgroup_limits_ok:
            logger.warning(
                "Cgroup resource limits (--cpus/--memory/--pids-limit) not "
                "available in this environment. Containers will run without "
                "CPU, memory or PID limits. To enable, delegate the cpu, "
                "memory and pids cgroup controllers to this container. "
                "Probe stderr: %s",
                (result.stderr or "").strip()[:500],
            )
    except Exception as e:
        _cgroup_limits_ok = False
        logger.warning("Cgroup limit probe failed; disabling resource limits: %s", e)

    return _cgroup_limits_ok


def _ensure_docker_available() -> None:
    """Best-effort check that the docker CLI is available before use.

    Reuses ``find_docker()`` so this preflight stays consistent with the rest of
    the Docker backend, including known non-PATH Docker Desktop locations.
    """
    docker_exe = find_docker()
    if not docker_exe:
        logger.error(
            "Docker backend selected but no docker executable was found in PATH "
            "or known install locations. Install Docker Desktop and ensure the "
            "CLI is available."
        )
        raise RuntimeError(
            "Docker executable not found in PATH or known install locations. "
            "Install Docker and ensure the 'docker' command is available."
        )

    try:
        result = subprocess.run(
            [docker_exe, "version"],
            capture_output=True,
            text=True,
            timeout=5,
            stdin=subprocess.DEVNULL,
        )
    except FileNotFoundError:
        logger.error(
            "Docker backend selected but the resolved docker executable '%s' could "
            "not be executed.",
            docker_exe,
            exc_info=True,
        )
        raise RuntimeError(
            "Docker executable could not be executed. Check your Docker installation."
        )
    except subprocess.TimeoutExpired:
        logger.error(
            "Docker backend selected but '%s version' timed out. "
            "The Docker daemon may not be running.",
            docker_exe,
            exc_info=True,
        )
        raise RuntimeError(
            "Docker daemon is not responding. Ensure Docker is running and try again."
        )
    except Exception:
        logger.error(
            "Unexpected error while checking Docker availability.",
            exc_info=True,
        )
        raise
    else:
        if result.returncode != 0:
            logger.error(
                "Docker backend selected but '%s version' failed "
                "(exit code %d, stderr=%s)",
                docker_exe,
                result.returncode,
                result.stderr.strip(),
            )
            raise RuntimeError(
                "Docker command is available but 'docker version' failed. "
                "Check your Docker installation."
            )


class DockerEnvironment(BaseEnvironment):
    """Hardened Docker container execution with resource limits and persistence.

    Security: all capabilities dropped, no privilege escalation, PID limits,
    size-limited tmpfs for scratch dirs. The container itself is the security
    boundary — the filesystem inside is writable so agents can install packages
    (pip, npm, apt) as needed. Writable workspace via tmpfs or bind mounts.

    Persistence: when enabled, bind mounts preserve /workspace and /root
    across container restarts.
    """

    def __init__(
        self,
        image: str,
        cwd: str = "/root",
        timeout: int = 60,
        cpu: float = 0,
        memory: int = 0,
        disk: int = 0,
        persistent_filesystem: bool = False,
        task_id: str = "default",
        volumes: list = None,
        forward_env: list[str] | None = None,
        env: dict | None = None,
        network: bool = True,
        host_cwd: str = None,
        auto_mount_cwd: bool = False,
        run_as_host_user: bool = False,
        extra_args: list = None,
        persist_across_processes: bool = True,
        resolved_init_env: dict[str, str] | None = None,
        resolved_implicit_mounts: list[str] | None = None,
    ):
        if cwd == "~":
            cwd = "/root"
        super().__init__(cwd=cwd, timeout=timeout)
        # Immutable creation cwd used when host-side session metadata must be
        # translated back into the shared container namespace.
        self._runtime_cwd = cwd
        self._persistent = persistent_filesystem
        self._persist_across_processes = persist_across_processes
        self._task_id = task_id
        self._forward_env = _normalize_forward_env_names(forward_env)
        self._env = _normalize_env_dict(env)
        # Freeze the exact values that init_session will capture. Durable
        # Kanban approval binds a hash of this same snapshot, so a forwarded or
        # skill-passthrough value cannot change between approval and execution.
        self._resolved_init_env = (
            _normalize_env_dict(resolved_init_env)
            if resolved_init_env is not None
            else resolve_docker_init_env(
                env=self._env,
                forward_env=self._forward_env,
            )
        )
        self._container_id: Optional[str] = None
        self._labels: dict[str, str] = {}
        self._image: str = ""
        self._container_name: str = ""
        self._image_uses_s6_init: bool = False
        self._all_run_args: list[str] = []
        logger.info(
            "DockerEnvironment configured with %d explicit volumes",
            len(volumes) if isinstance(volumes, list) else 0,
        )
        # Ensure volumes is a list (config.yaml could be malformed)
        if volumes is not None and not isinstance(volumes, list):
            logger.warning(f"docker_volumes config is not a list: {volumes!r}")
            volumes = []

        # Validate before building any Docker run arguments. Extra args are
        # appended last, so a contradictory --network flag could otherwise
        # override Hermes' docker_network=false lockdown on fresh creation.
        validated_extra = _normalize_docker_extra_args(
            extra_args,
            network=network,
        )

        # Fail fast if Docker is not available.
        _ensure_docker_available()

        # Build resource limit args (gated by cgroup availability probe so
        # they degrade gracefully on hosts without controller delegation,
        # e.g. unprivileged LXCs). The probe runs once per process and is
        # cached host-wide.
        resource_args = []
        if cpu > 0 and _cgroup_limits_available(image):
            resource_args.extend(["--cpus", str(cpu)])
        if memory > 0 and _cgroup_limits_available(image):
            resource_args.extend(["--memory", f"{memory}m"])
        if _cgroup_limits_available(image):
            resource_args.extend(["--pids-limit", _DEFAULT_PIDS_LIMIT])
        if disk > 0 and sys.platform != "darwin":
            if self._storage_opt_supported():
                resource_args.extend(["--storage-opt", f"size={disk}m"])
            else:
                logger.warning(
                    "Docker storage driver does not support per-container disk limits "
                    "(requires overlay2 on XFS with pquota). Container will run without disk quota."
                )
        if not network:
            resource_args.append("--network=none")

        # Persistent workspace via bind mounts from a configurable host directory
        # (TERMINAL_SANDBOX_DIR, default ~/.hermes/sandboxes/). Non-persistent
        # mode uses tmpfs (ephemeral, fast, gone on cleanup).
        from tools.environments.base import get_sandbox_dir

        # User-configured volume mounts (from config.yaml docker_volumes)
        volume_args = []
        workspace_explicitly_mounted = False
        for vol in (volumes or []):
            if not isinstance(vol, str):
                logger.warning(f"Docker volume entry is not a string: {vol!r}")
                continue
            vol = vol.strip()
            if not vol:
                continue
            if ":" in vol:
                volume_args.extend(["-v", vol])
                if ":/workspace" in vol:
                    workspace_explicitly_mounted = True
            else:
                logger.warning(f"Docker volume '{vol}' missing colon, skipping")

        host_cwd_abs = os.path.abspath(os.path.expanduser(host_cwd)) if host_cwd else ""
        bind_host_cwd = (
            auto_mount_cwd
            and bool(host_cwd_abs)
            and os.path.isdir(host_cwd_abs)
            and not workspace_explicitly_mounted
        )
        if auto_mount_cwd and host_cwd and not os.path.isdir(host_cwd_abs):
            logger.debug(f"Skipping docker cwd mount: host_cwd is not a valid directory: {host_cwd}")

        self._workspace_dir: Optional[str] = None
        self._home_dir: Optional[str] = None
        writable_args = []
        persistent_mount_specs: list[str] = []
        if self._persistent:
            sandbox = get_sandbox_dir() / "docker" / task_id
            self._home_dir = str(sandbox / "home")
            os.makedirs(self._home_dir, exist_ok=True)
            persistent_mount_specs.append(f"{self._home_dir}:/root")
            writable_args.extend([
                "-v", persistent_mount_specs[-1],
            ])
            if not bind_host_cwd and not workspace_explicitly_mounted:
                self._workspace_dir = str(sandbox / "workspace")
                os.makedirs(self._workspace_dir, exist_ok=True)
                persistent_mount_specs.append(
                    f"{self._workspace_dir}:/workspace"
                )
                writable_args.extend([
                    "-v", persistent_mount_specs[-1],
                ])
        else:
            if not bind_host_cwd and not workspace_explicitly_mounted:
                writable_args.extend([
                    "--tmpfs", "/workspace:rw,exec,size=10g",
                ])
            writable_args.extend([
                "--tmpfs", "/home:rw,exec,size=1g",
                "--tmpfs", "/root:rw,exec,size=1g",
            ])

        if bind_host_cwd:
            logger.info(f"Mounting configured host cwd to /workspace: {host_cwd_abs}")
            volume_args = ["-v", f"{host_cwd_abs}:/workspace", *volume_args]
        elif workspace_explicitly_mounted:
            logger.debug("Skipping docker cwd mount: /workspace already mounted by user config")

        # Freeze implicit credential/skill/cache mounts alongside the resolved
        # init environment.  They are part of both the runtime fingerprint and
        # the approval authority boundary, so creation must not re-resolve a
        # different set after consent is granted.
        implicit_mount_specs = (
            [
                value.strip()
                for value in resolved_implicit_mounts
                if isinstance(value, str) and value.strip() and ":" in value
            ]
            if resolved_implicit_mounts is not None
            else resolve_docker_implicit_mount_specs()
        )
        for mount_spec in implicit_mount_specs:
            volume_args.extend(["-v", mount_spec])
        if implicit_mount_specs:
            logger.info(
                "Docker: mounting %d resolved credential/skill/cache paths",
                len(implicit_mount_specs),
            )

        # Explicit environment variables (docker_env config) — set at container
        # creation so they're available to all processes (including entrypoint).
        env_args = []
        for key in sorted(self._env):
            env_args.extend(["-e", f"{key}={self._env[key]}"])

        # Optional: run the container as the host user so files written into
        # bind-mounted dirs (/workspace, /root, docker_volumes entries) are
        # owned by that user on the host instead of by root. Skip cleanly on
        # platforms without POSIX uid/gid (e.g. native Windows Docker).
        user_args: list[str] = []
        user_spec: str | None = None
        if run_as_host_user:
            user_spec = _resolve_host_user_spec()
            if user_spec is not None:
                user_args = ["--user", user_spec]
                logger.info("Docker: running container as host user %s", user_spec)
            else:
                logger.warning(
                    "docker_run_as_host_user is enabled but this platform does "
                    "not expose POSIX uid/gid; container will start as its "
                    "image default user."
                )
                # Fall back to the full cap set — without --user, an image's
                # init may still need s6-setuidgid/gosu/su to drop privileges.

        # Resolve the docker executable once so it works even when
        # /usr/local/bin is not in PATH (common on macOS gateway/service).
        self._docker_exe = find_docker() or "docker"

        # s6-overlay images (e.g. hermes-agent:latest) already use /init as PID 1
        # and exec /run/s6/basedir/bin/init during startup. For those images we
        # must (a) skip Docker's --init (two competing PID-1 inits) and (b) mount
        # /run with exec instead of noexec, or s6 stage0 dies with exit 126
        # "Permission denied". Detected once here; defaults are kept on any
        # inspection failure. See issue #34628.
        image_uses_s6_init = _image_uses_init_entrypoint(self._docker_exe, image)
        if image_uses_s6_init:
            logger.info(
                "Docker: image %s uses /init (s6-overlay) as entrypoint — "
                "skipping --init and mounting /run with exec.",
                image,
            )
        security_args = _build_security_args(
            run_as_host_user and bool(user_args),
            run_exec=image_uses_s6_init,
        )

        logger.info("Docker configured %d volume arguments", len(volume_args) // 2)

        self._runtime_fingerprint = docker_runtime_fingerprint(
            image=image,
            cwd=cwd,
            cpu=cpu,
            memory=memory,
            disk=disk,
            persistent_filesystem=self._persistent,
            task_id=task_id,
            volumes=[
                value.strip()
                for value in (volumes or [])
                if isinstance(value, str) and value.strip() and ":" in value
            ] + implicit_mount_specs,
            network=network,
            host_cwd=host_cwd_abs,
            auto_mount_cwd=bind_host_cwd,
            run_as_host_user=run_as_host_user and bool(user_args),
            extra_args=validated_extra,
            persist_across_processes=persist_across_processes,
            init_env_digest=docker_init_env_digest(self._resolved_init_env),
            host_user_spec=user_spec,
            persistent_mounts=persistent_mount_specs,
        )
        # Any explicit/implicit mount or arbitrary docker run argument can
        # expand authority beyond the nominal sandbox.  Be conservative when
        # deciding whether dangerous-command approval may be skipped.
        self._has_host_access = bool(volume_args or validated_extra)

        all_run_args = (
            security_args
            + user_args
            + writable_args
            + resource_args
            + volume_args
            + env_args
            + validated_extra
        )
        logger.info(
            "Docker runtime prepared (%d args, fingerprint=%s)",
            len(all_run_args),
            self._runtime_fingerprint,
        )

        # Start the container directly via `docker run -d`.
        container_name = f"hermes-{uuid.uuid4().hex[:8]}"
        # Labels make hermes-created containers identifiable to:
        #   * the orphan reaper (`hermes-agent=1` for the global sweep filter)
        #   * future cross-process reuse (`hermes-task-id`, `hermes-profile`)
        #   * operators running `docker ps --filter label=hermes-agent=1`
        # Values are limited to the safe character set defined by
        # _sanitize_label_value(); the active Hermes profile is captured at
        # container-start time and never changes for the container's lifetime.
        profile_name = _sanitize_label_value(_get_active_profile_name())
        task_label = _sanitize_label_value(task_id)
        label_args = [
            "--label", "hermes-agent=1",
            "--label", f"hermes-task-id={task_label}",
            "--label", f"hermes-profile={profile_name}",
            "--label", f"{_RUNTIME_FINGERPRINT_LABEL}={self._runtime_fingerprint}",
        ]
        # Save args for container recreation on "No such container" recovery.
        self._image = image
        self._container_name = container_name
        self._image_uses_s6_init = image_uses_s6_init
        self._all_run_args = all_run_args

        self._labels = {
            "hermes-agent": "1",
            "hermes-task-id": task_label,
            "hermes-profile": profile_name,
            _RUNTIME_FINGERPRINT_LABEL: self._runtime_fingerprint,
        }

        # Cross-process container reuse (issue #20561 — docs claim "ONE long-lived
        # container shared across sessions").  If a prior Hermes process
        # already started a container for this (task_id, profile) and it
        # still exists, attach to it instead of starting a fresh one.  This
        # restores the documented contract; opt out via
        # ``terminal.docker_persist_across_processes: false``.
        #
        # The runtime fingerprint label binds image, mounts, resources,
        # network policy, user mapping, extra args, persistence, and the
        # resolved init environment. Legacy or differently configured
        # containers are never eligible for reuse.
        reused = False
        if persist_across_processes:
            existing = self._find_reusable_container(
                task_label,
                profile_name,
                self._runtime_fingerprint,
            )
            if existing is not None:
                container_id, state = existing
                # Network-mode guard: reuse must not silently defeat an
                # egress lockdown.  A container created before the operator
                # set ``docker_network: false`` keeps its original bridge
                # NetworkMode, so label-only reuse would hand the agent a
                # networked container despite the config.  On mismatch we
                # detach from the stale container and start fresh. It may be
                # serving another Hermes process, so mismatch handling must
                # never stop or remove it.
                # Only the lockdown direction is guarded: a ``none``-mode
                # container under a default-network config is left alone so
                # operators using ``docker_extra_args: ["--network=none"]``
                # don't get their container churned on every startup.
                mode_mismatch = False
                actual_mode = None
                if not network:
                    actual_mode = self._container_network_mode(container_id)
                    mode_mismatch = actual_mode != "none"
                if mode_mismatch:
                    logger.warning(
                        "Existing container %s has NetworkMode=%s but "
                        "docker_network=false requests an air-gapped "
                        "container — leaving it untouched and starting fresh "
                        "(task=%s, profile=%s).",
                        container_id[:12], actual_mode or "unknown",
                        task_label, profile_name,
                    )
                    existing = None
            if existing is not None:
                container_id, state = existing
                self._container_id = container_id
                if state != "running":
                    try:
                        subprocess.run(
                            [self._docker_exe, "start", container_id],
                            capture_output=True,
                            text=True,
                            timeout=30,
                            check=True,
                            stdin=subprocess.DEVNULL,
                        )
                    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
                        logger.warning(
                            "Failed to start existing container %s (state=%s): "
                            "%s — falling back to a fresh container.",
                            container_id[:12], state, e,
                        )
                        self._container_id = None
                if self._container_id:
                    logger.info(
                        "Reusing container %s (task=%s, profile=%s, prior state=%s)",
                        container_id[:12], task_label, profile_name, state,
                    )
                    reused = True

        if not reused:
            # tini/catatonit as PID 1 reaps zombie children — but s6-overlay
            # images already provide their own /init PID 1, so adding --init
            # there creates two competing inits and breaks startup (#34628).
            init_args = [] if image_uses_s6_init else ["--init"]
            run_cmd = [
                self._docker_exe, "run", "-d",
                *init_args,
                "--name", container_name,
                *label_args,
                "-w", cwd,
                *all_run_args,
                image,
                "sleep", "infinity",  # no fixed lifetime — idle reaper handles cleanup
            ]
            logger.debug(
                "Starting Docker container image=%s cwd=%s fingerprint=%s",
                image,
                cwd,
                self._runtime_fingerprint,
            )
            try:
                result = subprocess.run(
                    run_cmd,
                    capture_output=True,
                    text=True,
                    timeout=120,  # image pull may take a while
                    check=True,
                    stdin=subprocess.DEVNULL,
                )
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
                # Docker may create the container object before `docker run`
                # fails to start it (e.g. exit code 125 when the daemon isn't
                # ready, or a timeout mid-pull). That orphan is left in
                # "Created" state — which the exited-only orphan reaper
                # (reap_orphan_containers, status=exited) never catches, so it
                # leaks permanently. Remove it by its known name before
                # re-raising. See #7439.
                logger.warning(
                    "docker run failed for %s, cleaning up orphaned container: %s",
                    container_name, e,
                )
                subprocess.run(
                    [self._docker_exe, "rm", "-f", container_name],
                    capture_output=True, timeout=10,
                    stdin=subprocess.DEVNULL,
                )
                raise
            self._container_id = result.stdout.strip()
            logger.info(f"Started container {container_name} ({self._container_id[:12]})")

        # Build the init-time env forwarding args (used only by init_session
        # to inject host env vars into the snapshot; subsequent commands get
        # them from the snapshot file).
        self._init_env_args = self._build_init_env_args()

        # Initialize session snapshot inside the container
        self.init_session()

    def _build_init_env_args(self) -> list[str]:
        """Build -e KEY=VALUE args for injecting host env vars into init_session.

        These are used once during init_session() so that export -p captures
        them into the snapshot.  Subsequent execute() calls don't need -e flags.
        """
        # Production instances freeze this snapshot in __init__. The fallback
        # keeps lightweight __new__-constructed test doubles/back-compat callers
        # working while still going through the same resolver.
        if hasattr(self, "_resolved_init_env"):
            exec_env = dict(self._resolved_init_env)
        else:
            exec_env = resolve_docker_init_env(
                env=self._env,
                forward_env=self._forward_env,
            )

        args = []
        for key in sorted(exec_env):
            args.extend(["-e", f"{key}={exec_env[key]}"])
        return args

    @property
    def init_env_digest(self) -> str:
        """Hash of the exact environment snapshot used by init_session."""

        if hasattr(self, "_resolved_init_env"):
            resolved = self._resolved_init_env
        else:
            resolved = resolve_docker_init_env(
                env=self._env,
                forward_env=self._forward_env,
            )
        return docker_init_env_digest(resolved)

    @property
    def runtime_fingerprint(self) -> str:
        """Hash-only identity required for safe in- and cross-process reuse."""

        return self._runtime_fingerprint

    @property
    def has_host_access(self) -> bool:
        """Whether user configuration bind-mounted a host path."""

        return bool(self._has_host_access)

    def _run_bash(self, cmd_string: str, *, login: bool = False,
                  timeout: int = 120,
                  stdin_data: str | None = None) -> subprocess.Popen:
        """Spawn a bash process inside the Docker container."""
        assert self._container_id, "Container not started"
        cmd = [self._docker_exe, "exec"]
        if stdin_data is not None:
            cmd.append("-i")

        # Only inject -e env args during init_session (login=True).
        # Subsequent commands get env vars from the snapshot.
        if login:
            cmd.extend(self._init_env_args)

        cmd.extend([self._container_id])

        if login:
            cmd.extend(["bash", "-l", "-c", cmd_string])
        else:
            cmd.extend(["bash", "-c", cmd_string])

        return _popen_bash(cmd, stdin_data)

    # ------------------------------------------------------------------
    # "No such container" recovery (issue #36266)
    # ------------------------------------------------------------------

    _NO_CONTAINER_PATTERNS = (
        "No such container",
        "is not running",
        "no such container",
    )

    def _is_container_gone(self, output: str) -> bool:
        """Return True if the output indicates the container no longer exists."""
        return any(p in output for p in self._NO_CONTAINER_PATTERNS)

    def _recreate_container(self) -> bool:
        """Recreate the container after it was removed out-of-band.

        Tries label-based reuse first; if no existing container is found,
        starts a fresh one with the same image and run-args.  Returns True
        on success, False if recreation fails (caller should surface the
        original error).
        """
        old_id = (self._container_id or "")[:12]
        logger.warning(
            "Container %s appears to be gone — attempting recovery", old_id,
        )
        self._container_id = None

        # 1. Try label-based reuse (another process may have recreated it).
        task_label = self._labels.get("hermes-task-id", "")
        profile_label = self._labels.get("hermes-profile", "")
        runtime_fingerprint = self._labels.get(_RUNTIME_FINGERPRINT_LABEL, "")
        existing = self._find_reusable_container(
            task_label,
            profile_label,
            runtime_fingerprint,
        )
        if existing is not None:
            cid, state = existing
            if state == "running":
                self._container_id = cid
                logger.info("Recovery: reusing running container %s", cid[:12])
            else:
                try:
                    subprocess.run(
                        [self._docker_exe, "start", cid],
                        capture_output=True, text=True, timeout=30, check=True,
                        stdin=subprocess.DEVNULL,
                    )
                    self._container_id = cid
                    logger.info("Recovery: restarted container %s", cid[:12])
                except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
                    logger.warning("Recovery: failed to start container %s: %s", cid[:12], e)

        # 2. No reusable container — create a fresh one.
        if not self._container_id:
            if not self._image:
                logger.error("Recovery: no saved image name, cannot recreate container")
                return False
            try:
                import uuid as _uuid
                new_name = f"hermes-{_uuid.uuid4().hex[:8]}"
                init_args = [] if self._image_uses_s6_init else ["--init"]
                label_args = []
                for k, v in self._labels.items():
                    label_args.extend(["--label", f"{k}={v}"])
                run_cmd = [
                    self._docker_exe, "run", "-d",
                    *init_args,
                    "--name", new_name,
                    *label_args,
                    "-w", self.cwd,
                    *self._all_run_args,
                    self._image,
                    "sleep", "infinity",
                ]
                result = subprocess.run(
                    run_cmd, capture_output=True, text=True, timeout=120, check=True,
                    stdin=subprocess.DEVNULL,
                )
                self._container_id = result.stdout.strip()
                self._container_name = new_name
                logger.info(
                    "Recovery: created fresh container %s (%s)",
                    new_name, self._container_id[:12],
                )
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError) as e:
                logger.error("Recovery: failed to create new container: %s", e)
                return False

        # 3. Re-initialize session snapshot in the (re)created container.
        try:
            self._snapshot_ready = False
            self.init_session()
        except Exception as e:
            logger.error("Recovery: init_session failed in new container: %s", e)
            return False

        logger.info("Recovery successful — new container %s", (self._container_id or "")[:12])
        return True

    def execute(self, command: str, cwd: str = "", **kwargs) -> dict:
        """Execute a command, auto-recovering from dead containers.

        If the container was removed out-of-band (idle reaper, docker prune,
        OOM kill, daemon restart), detect the error and recreate the container
        transparently before retrying once.
        """
        result = super().execute(command, cwd, **kwargs)
        if (
            result.get("returncode", 0) != 0
            and self._is_container_gone(result.get("output", ""))
            and self._persist_across_processes
        ):
            if self._recreate_container():
                result = super().execute(command, cwd, **kwargs)
        return result

    @staticmethod
    def _storage_opt_supported() -> bool:
        """Check if Docker's storage driver supports --storage-opt size=.
        
        Only overlay2 on XFS with pquota supports per-container disk quotas.
        Ubuntu (and most distros) default to ext4, where this flag errors out.
        """
        global _storage_opt_ok
        if _storage_opt_ok is not None:
            return _storage_opt_ok
        try:
            docker = find_docker() or "docker"
            result = subprocess.run(
                [docker, "info", "--format", "{{.Driver}}"],
                capture_output=True, text=True, timeout=10,
                stdin=subprocess.DEVNULL,
            )
            driver = result.stdout.strip().lower()
            if driver != "overlay2":
                _storage_opt_ok = False
                return False
            # overlay2 only supports storage-opt on XFS with pquota.
            # Probe by attempting a dry-ish run — the fastest reliable check.
            probe = subprocess.run(
                [docker, "create", "--storage-opt", "size=1m", "hello-world"],
                capture_output=True, text=True, timeout=15,
                stdin=subprocess.DEVNULL,
            )
            if probe.returncode == 0:
                # Clean up the created container
                container_id = probe.stdout.strip()
                if container_id:
                    subprocess.run([docker, "rm", container_id],
                                   capture_output=True, timeout=5,
                                   stdin=subprocess.DEVNULL)
                _storage_opt_ok = True
            else:
                _storage_opt_ok = False
        except Exception:
            _storage_opt_ok = False
        logger.debug("Docker --storage-opt support: %s", _storage_opt_ok)
        return _storage_opt_ok

    def _container_network_mode(self, container_id: str) -> Optional[str]:
        """Return the container's ``HostConfig.NetworkMode`` (e.g. ``bridge``,
        ``none``, ``host``), or ``None`` when inspection fails.

        Used by the reuse path to make sure a persisted container's network
        mode still matches the operator's ``docker_network`` setting; callers
        treat ``None`` (unknown) as a mismatch when lockdown was requested,
        so a failed inspect fails closed rather than open.
        """
        try:
            result = subprocess.run(
                [
                    self._docker_exe, "inspect",
                    "--format", "{{.HostConfig.NetworkMode}}",
                    container_id,
                ],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
                stdin=subprocess.DEVNULL,
            )
        except (subprocess.TimeoutExpired, OSError) as e:
            logger.debug("docker inspect NetworkMode failed: %s", e)
            return None
        if result.returncode != 0:
            logger.debug(
                "docker inspect NetworkMode returned %d: %s",
                result.returncode, result.stderr.strip(),
            )
            return None
        mode = result.stdout.strip()
        return mode or None

    def _find_reusable_container(
        self,
        task_label: str,
        profile_label: str,
        runtime_fingerprint: str = "",
    ) -> Optional[tuple[str, str]]:
        """Look for an exact-runtime container for this (task, profile).

        Returns ``(container_id, state)`` on hit, ``None`` on miss / on any
        failure (including ``docker ps`` itself failing). State is one of the
        values Docker reports via ``{{.State}}`` — e.g. ``running``, ``exited``,
        ``created``, ``paused``, ``restarting``, ``dead``. The caller decides
        whether the state warrants ``docker start`` before reuse.

        Restricted to the Docker-stored label set this class creates. Legacy
        containers without a runtime fingerprint and containers created under
        different mounts/images/environment are deliberately not reusable.
        """
        if not runtime_fingerprint:
            return None
        try:
            result = subprocess.run(
                [
                    self._docker_exe, "ps", "-a",
                    "--filter", "label=hermes-agent=1",
                    "--filter", f"label=hermes-task-id={task_label}",
                    "--filter", f"label=hermes-profile={profile_label}",
                    "--filter", (
                        f"label={_RUNTIME_FINGERPRINT_LABEL}="
                        f"{runtime_fingerprint}"
                    ),
                    "--format", "{{.ID}}\t{{.State}}",
                ],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
                stdin=subprocess.DEVNULL,
            )
        except (subprocess.TimeoutExpired, OSError) as e:
            logger.debug("docker ps probe failed: %s — will start a fresh container", e)
            return None
        if result.returncode != 0:
            logger.debug(
                "docker ps probe returned %d: %s — will start a fresh container",
                result.returncode, result.stderr.strip(),
            )
            return None
        lines = [ln.strip() for ln in result.stdout.splitlines() if ln.strip()]
        if not lines:
            return None
        # Multiple matches are unusual (one (task, profile) should produce one
        # container) but can happen if a previous Hermes process crashed
        # mid-cleanup. Prefer a running one if present; otherwise pick the
        # first listed. Stale duplicates get reaped by the orphan-reaper in a
        # follow-up commit; we don't try to be heroic about them here.
        running = None
        first = None
        for ln in lines:
            parts = ln.split("\t", 1)
            if len(parts) != 2:
                continue
            cid, state = parts[0], parts[1].lower()
            if first is None:
                first = (cid, state)
            if state == "running" and running is None:
                running = (cid, state)
        return running or first

    def cleanup(self, *, force_remove: bool = False):
        """Tear down the container according to persist mode and *force_remove*.

        Persist-mode (``persist_across_processes=True``, the default) leaves the
        container **running** untouched. The docs promise "ONE long-lived
        container shared across sessions" and stopping it on every Hermes exit
        breaks that promise:

        * Background processes inside the container (``npm run dev``, watchers,
          long-running pytest) get killed every time the user runs ``/quit``.
        * Every reuse requires ``docker start`` + waiting for the container to
          come back up, adding 1–2s to the first tool call of the new session.
        * The user-visible difference between "ONE long-lived container" and
          "a new container that happens to share state" is exactly this:
          processes survive in the former, die in the latter.

        Resource reclamation for the persist-mode case lives in the
        ``reap_orphan_containers()`` path (see issue #20561 commit 3): if no
        Hermes process touches a labeled container for ``2 × lifetime_seconds``
        it gets ``docker rm -f``'d at the next Hermes startup. That covers the
        SIGKILL / OOM / abandoned-laptop cases without us needing to stop the
        container on every graceful exit.

        Opt-out mode (``persist_across_processes=False``) still does
        ``docker stop`` + ``docker rm -f`` on every cleanup, matching the
        pre-PR behavior for users who explicitly want per-process isolation.

        ``force_remove=True`` overrides persist mode and always tears the
        container down (``docker stop`` + ``docker rm -f``). This is the
        explicit-teardown path for ``/reset``, ``cleanup_vm(task_id)``-driven
        resets, or any caller that wants a guaranteed fresh container on next
        ``DockerEnvironment(task_id=...)``. No current caller passes
        ``force_remove=True``; the parameter is here so the explicit-teardown
        semantics can be wired up later without changing this method's
        signature.

        Cleanup runs on a daemon thread with bounded ``subprocess.run`` calls
        (not the racy ``Popen(... &)`` pattern from before PR #33645). The
        atexit hook in ``tools/terminal_tool.py`` waits up to 15s for the
        thread to finish before the interpreter exits, so ``docker stop`` /
        ``docker rm`` actually completes when we do trigger it.
        """
        container_id = self._container_id
        if not container_id:
            # Still drop the bind-mount dirs if any were allocated and we're
            # NOT in persist mode (persist mode preserves them).
            if not self._persistent:
                for d in (self._workspace_dir, self._home_dir):
                    if d:
                        shutil.rmtree(d, ignore_errors=True)
            return

        # Decide what to actually do. Three cases:
        #
        #   force_remove=True             → stop + rm (explicit teardown)
        #   persist_across_processes=True → no-op (leave container running)
        #   persist_across_processes=False → stop + rm (per-process isolation)
        #
        # The persist-mode no-op is the issue-#20561 contract: the container
        # outlives Hermes processes, processes inside it stay alive, and
        # reuse on next startup is instant.
        if force_remove:
            should_stop = True
            should_remove = True
        elif self._persist_across_processes:
            # No-op for the container. Drop the in-process handle so a fresh
            # __init__ will re-probe via labels (and find the running
            # container) instead of trying to reuse a stale Python reference.
            self._container_id = None
            return
        else:
            should_stop = True
            should_remove = True

        # Capture state needed by the worker before we null out the attrs —
        # the worker thread can outlive ``self``.
        docker_exe = self._docker_exe
        log_id = container_id[:12]

        def _do_cleanup() -> None:
            if should_stop:
                try:
                    subprocess.run(
                        [docker_exe, "stop", "-t", "10", container_id],
                        capture_output=True, timeout=30,
                        stdin=subprocess.DEVNULL,
                    )
                except (subprocess.TimeoutExpired, OSError) as e:
                    logger.warning("docker stop %s timed out / failed: %s", log_id, e)
            if should_remove:
                try:
                    subprocess.run(
                        [docker_exe, "rm", "-f", container_id],
                        capture_output=True, timeout=30,
                        stdin=subprocess.DEVNULL,
                    )
                except (subprocess.TimeoutExpired, OSError) as e:
                    logger.warning("docker rm -f %s failed: %s", log_id, e)

        # Daemon thread: doesn't block interpreter exit (atexit returns
        # promptly), but unlike the old ``Popen(... &)`` shell trick the
        # Python-level join semantics let the thread actually run to
        # completion if the interpreter is still alive. atexit registers
        # ``_atexit_cleanup`` in terminal_tool.py which waits up to ~60s for
        # outstanding cleanups, so most exits complete the work cleanly.
        import threading
        t = threading.Thread(target=_do_cleanup, daemon=True, name=f"hermes-cleanup-{log_id}")
        t.start()
        self._cleanup_thread = t
        self._container_id = None

        # Bind-mount dir teardown only runs when we actually removed the
        # container (the dirs are the container's filesystem state; keeping
        # them around with no container would orphan the data on disk).
        if should_remove and not self._persistent:
            for d in (self._workspace_dir, self._home_dir):
                if d:
                    shutil.rmtree(d, ignore_errors=True)

    def wait_for_cleanup(self, timeout: float = 30.0) -> bool:
        """Block up to *timeout* seconds for the cleanup worker thread.

        Returns ``True`` if the thread finished (or no thread was started),
        ``False`` on timeout. The atexit hook in terminal_tool.py calls this
        on every active environment so docker stop/rm actually completes
        before the Python process exits — without this, ``hermes /quit``
        races the interpreter shutdown and leaves stopped containers behind.
        """
        thread = getattr(self, "_cleanup_thread", None)
        if thread is None or not thread.is_alive():
            return True
        thread.join(timeout=timeout)
        return not thread.is_alive()
