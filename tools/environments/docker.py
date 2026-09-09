"""Docker execution environment for sandboxed command execution.

Security hardened (cap-drop ALL, no-new-privileges, PID limits), configurable
resource limits (CPU, memory, disk), and optional filesystem persistence via
bind mounts.
"""

import datetime
import hashlib
import json
import logging
import os
import re
import shlex
import shutil
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Optional

from tools.environments.base import BaseEnvironment, EnvironmentConnectionError, _SHELL_ENV_NAME_RE
from tools.environments.base_output import _popen_bash
from tools.environments.docker_egress import (
    _EGRESS_LABEL_KEY, _critical_egress_env_names, _egress_enforce_on_docker, _egress_proxy_args_for_docker,
    _egress_reuse_fingerprint, check_docker_env_collisions, check_extra_args_collisions,
    check_forward_env_collisions, merge_egress_env,
)
from tools.environments.path_utils import sanitize_task_id_for_path
from tools.environments.remote_common import (
    bash_argv, client_env_with, load_hermes_env_vars, prepend_unset, resolve_passthrough_env, run_capture)

logger = logging.getLogger(__name__)

# Docker Desktop install paths checked when 'docker' is not in PATH
# (macOS Intel / Apple Silicon Homebrew / app bundle).
_DOCKER_SEARCH_PATHS = [
    "/usr/local/bin/docker", "/opt/homebrew/bin/docker", "/Applications/Docker.app/Contents/Resources/bin/docker",
]

_docker_executable: Optional[str] = None  # resolved once, cached
_ENV_VAR_NAME_RE = _SHELL_ENV_NAME_RE


def _normalize_forward_env_names(forward_env: list[str] | None) -> list[str]:
    """Return a deduplicated list of valid environment variable names."""
    normalized: list[str] = []
    for item in forward_env or []:
        if not isinstance(item, str):
            logger.warning("Ignoring non-string docker_forward_env entry: %r", item)
            continue
        key = item.strip()
        if not key:
            continue
        if not _ENV_VAR_NAME_RE.match(key):
            logger.warning("Ignoring invalid docker_forward_env entry: %r", item)
        elif key not in normalized:
            normalized.append(key)
    return normalized


def _normalize_env_dict(env: dict | None) -> dict[str, str]:
    """Validate a docker_env dict to {str: str}; scalars are coerced, other values dropped."""
    if not env:
        return {}
    if not isinstance(env, dict):
        logger.warning("docker_env is not a dict: %s", type(env).__name__)
        return {}
    normalized: dict[str, str] = {}
    for key, value in env.items():
        if not isinstance(key, str) or not _ENV_VAR_NAME_RE.match(key.strip()):
            logger.warning("Ignoring invalid docker_env key: %r", key)
            continue
        if not isinstance(value, (str, int, float, bool)):
            logger.warning("Ignoring non-string docker_env value for %r: %s", key.strip(), type(value).__name__)
            continue
        normalized[key.strip()] = value if isinstance(value, str) else str(value)
    return normalized


# Module-level binding: tests patch ``docker._load_hermes_env_vars`` to fake the .env file.
_load_hermes_env_vars = load_hermes_env_vars


# Docker label values must match [a-zA-Z0-9_.-] and stay <=63 chars to round-trip
# through `docker ps --filter label=key=value`.
_LABEL_VALUE_OK_RE = re.compile(r"[^A-Za-z0-9_.-]")


def _sanitize_label_value(value: str) -> str:
    """Lossy coercion into a Docker label-safe form; empty/invalid input becomes ``"unknown"``."""
    if not isinstance(value, str) or not value:
        return "unknown"
    return _LABEL_VALUE_OK_RE.sub("_", value)[:63] or "unknown"


# task_id -> host directory name; shared with every backend persisting per-task
# state on the host (Singularity overlays) so the mapping is fixed in one place.
_sandbox_dir_name = sanitize_task_id_for_path


def _get_active_profile_name() -> str:
    """Active Hermes profile name, or ``"default"`` on any error. Resolved at container-create
    time so a container stays tagged with its creator even if the process switches profiles."""
    try:
        from hermes_cli.profiles import get_active_profile_name
        return get_active_profile_name() or "default"
    except Exception:
        return "default"


def _container_identity(shared_key: str = "") -> str:
    """Profile label used for container reuse and orphan reaping. Profiles are isolated by default; an
    explicit shared key lets trusted profiles share one Docker identity. Label sanitization is lossy
    and reuse is label-keyed, so a digest of the raw key disambiguates colliding keys. Plain profile
    names keep their historical un-suffixed labels."""
    if not shared_key:
        return _sanitize_label_value(_get_active_profile_name())
    digest = hashlib.sha256(shared_key.encode("utf-8")).hexdigest()[:12]
    return f"{_sanitize_label_value(shared_key)[:50]}-{digest}"


def reap_orphan_containers(
    *, max_age_seconds: int = 600, profile_filter: str | None = None, docker_exe: str | None = None,
) -> int:
    """Remove stale hermes-tagged containers left behind by prior processes (SIGKILL/OOM
    exits that bypass atexit). Only ``status=exited`` containers (running ones may belong
    to a sibling process), only the caller's profile, and only if ``FinishedAt`` is older
    than *max_age_seconds* (a just-exited sibling may be about to reuse its container).
    Best-effort and idempotent: failures log at debug and the count removed so far is returned.
    """
    docker = docker_exe or find_docker() or "docker"
    filters = ["--filter", "label=hermes-agent=1", "--filter", "status=exited"]
    if profile_filter:
        filters.extend(["--filter", f"label=hermes-profile={_sanitize_label_value(profile_filter)}"])

    listing = _docker_query(
        [docker, "ps", "-a", *filters, "--format", "{{.ID}}"], timeout=15,
        fail="orphan reaper docker ps failed: %s", nonzero="orphan reaper docker ps returned %d: %s")
    if listing is None:
        return 0

    # Per-container inspect keeps the failure blast radius to one container.
    now = datetime.datetime.now(datetime.timezone.utc)
    removed = 0
    for cid in (ln.strip() for ln in listing.stdout.splitlines() if ln.strip()):
        finished_at = _container_finished_at(docker, cid)
        if finished_at is None:  # unknown age — be conservative
            continue
        age = (now - finished_at).total_seconds()
        if age < max_age_seconds:
            continue
        result = _docker_query(
            [docker, "rm", "-f", cid], timeout=30, fail="orphan reaper docker rm %s failed: %s", fail_args=(cid[:12],))
        if result is None:
            continue
        if result.returncode == 0:
            removed += 1
            logger.info("Reaped orphan container %s (exited %d seconds ago)", cid[:12], int(age))
        else:
            logger.debug("docker rm -f %s failed: %s", cid[:12], result.stderr.strip())
    return removed


def _container_finished_at(docker_exe: str, container_id: str):
    """``docker inspect`` FinishedAt as an aware datetime; ``None`` (= don't reap) when
    missing/unparseable or Docker's never-finished zero value ``0001-01-01T00:00:00Z``."""
    result = _docker_query(
        [docker_exe, "inspect", "--format", "{{.State.FinishedAt}}", container_id], timeout=10,
        fail="orphan reaper docker inspect %s failed: %s", fail_args=(container_id[:12],))
    if result is None or result.returncode != 0:
        return None
    raw = result.stdout.strip()
    if not raw or raw.startswith("0001-01-01"):
        return None
    # Docker emits RFC3339 with nanoseconds; fromisoformat only takes microseconds.
    raw = re.sub(r"(\.\d{6})\d+", r"\1", raw).replace("Z", "+00:00")
    try:
        return datetime.datetime.fromisoformat(raw)
    except ValueError as e:
        logger.debug("could not parse FinishedAt %r for %s: %s", raw, container_id[:12], e)
        return None


def _is_executable(path: str) -> bool:
    return os.path.isfile(path) and os.access(path, os.X_OK)


def _docker_query(
    argv: list[str], *, timeout: float, fail: str, fail_args: tuple = (), nonzero: str | None = None,
    exc_types: tuple = (subprocess.TimeoutExpired, OSError),
):
    """Best-effort ``run_capture`` for probes: logs *fail* (``*fail_args, exc``) at debug and
    returns ``None`` when the CLI raises; with *nonzero* set, a nonzero exit also logs
    (``*fail_args, returncode, stderr``) and returns ``None``."""
    try:
        result = run_capture(argv, timeout=timeout)
    except exc_types as e:
        logger.debug(fail, *fail_args, e)
        return None
    if nonzero is not None and result.returncode != 0:
        logger.debug(nonzero, *fail_args, result.returncode, result.stderr.strip())
        return None
    return result


def find_docker() -> Optional[str]:
    """Locate the docker/podman CLI (cached): ``HERMES_DOCKER_BINARY`` override, ``docker``
    on PATH, ``podman`` on PATH, then macOS Docker Desktop locations; ``None`` if absent."""
    global _docker_executable
    if _docker_executable is not None:
        return _docker_executable

    override = os.getenv("HERMES_DOCKER_BINARY")
    if override and _is_executable(override):
        logger.info("Using HERMES_DOCKER_BINARY override: %s", override)
        found = override
    elif found := shutil.which("docker"):
        pass
    elif found := shutil.which("podman"):
        logger.info("Using podman as container runtime: %s", found)
    elif found := next((p for p in _DOCKER_SEARCH_PATHS if _is_executable(p)), None):
        logger.info("Found docker at non-PATH location: %s", found)
    if found:
        _docker_executable = found
    return found


# Security flags applied to every container. The container is the security
# boundary; all caps are dropped and the minimum added back:
#   DAC_OVERRIDE  - root can write to bind-mounted dirs owned by the host user
#   CHOWN/FOWNER  - package managers need to set file ownership
# /tmp is size-limited and nosuid but allows exec (pip/npm builds need it).
# ``--pids-limit``/``--cpus``/``--memory`` live in resource_args instead, gated
# on ``_cgroup_limits_available`` (unprivileged LXCs lack the controllers).
_BASE_SECURITY_ARGS = [
    "--cap-drop", "ALL",
    "--cap-add", "DAC_OVERRIDE",
    "--cap-add", "CHOWN",
    "--cap-add", "FOWNER",
    "--tmpfs", "/tmp:rw,nosuid,size=512m",
    "--tmpfs", "/var/tmp:rw,noexec,nosuid,size=256m"]

_DEFAULT_PIDS_LIMIT = "256"  # applied only when the pids cgroup controller is available

# Docker's 64 MB /dev/shm default crashes Chromium/Playwright tabs and PyTorch
# DataLoader workers. tmpfs is lazily allocated so a 1g ceiling costs nothing
# until used (and still counts against --memory). Empty/"0" config omits the flag.
# Docker's built-in default is a tiny 64 MB, which silently breaks shared-memory-hungry workloads inside the
# sandbox: Chromium / Playwright renderers crash tabs, and PyTorch DataLoader workers die with "bus error" /
# "insufficient shared memory" once they exceed it. tmpfs is lazily allocated, so a 1g ceiling costs nothing
# until actually used (and usage still counts against the container's --memory cgroup limit). Configurable
# via ``terminal.docker_shm_size`` in config.yaml; an empty value (or "0") omits the flag and falls back to
# Docker's 64 MB default. Ported from nanocoai/nanoclaw#2748.
_DEFAULT_SHM_SIZE = "1g"


def _extra_args_set_shm_size(extra_args: list) -> bool:
    """True when docker_extra_args already set ``--shm-size`` (then our default is skipped)."""
    return any(
        isinstance(a, str) and (a == "--shm-size" or a.startswith("--shm-size="))
        for a in (extra_args or []))


_NETWORK_FLAGS = ("--network", "--net")


def _extra_args_network_mode(extra_args: list) -> Optional[str]:
    """Network mode requested by ``docker_extra_args`` (``--network none`` / ``--network=none`` /
    ``--net``), or None when the operator set no network flag. Docker rejects a repeated
    ``--network`` outright (exit 125), so the implicit ``docker_network: false`` flag and an
    operator-supplied one must be reconciled before the command line is built."""
    args = [a for a in (extra_args or []) if isinstance(a, str)]
    for i, arg in enumerate(args):
        for flag in _NETWORK_FLAGS:
            if arg == flag:
                return args[i + 1] if i + 1 < len(args) else ""
            if arg.startswith(f"{flag}="):
                return arg.split("=", 1)[1]
    return None


# /run is separate from _BASE_SECURITY_ARGS: s6-overlay images exec
# /run/s6/basedir/bin/init at stage0 and die (exit 126) on a noexec mount.
_RUN_TMPFS_NOEXEC = "--tmpfs", "/run:rw,noexec,nosuid,size=64m"
_RUN_TMPFS_EXEC = "--tmpfs", "/run:rw,exec,nosuid,size=64m"

# SETUID/SETGID let a root-started init drop privileges (s6-setuidgid, gosu,
# su). Combined with no-new-privileges the dropped process cannot escalate
# back. Skipped when --user is passed: the container already starts unprivileged.
_PRIVDROP_CAP_ARGS = ["--cap-add", "SETUID", "--cap-add", "SETGID"]

# Every ``cleanup()`` worker, independent of terminal_tool's active-env registry (see
# ``wait_for_all_teardowns``).
_TEARDOWN_THREADS: set[threading.Thread] = set()
_TEARDOWN_LOCK = threading.Lock()

_S6_INIT_ENTRYPOINTS = ("/init", "/package/admin/s6-overlay/command/init")


_NO_NEW_PRIVILEGES_ARGS = ["--security-opt", "no-new-privileges"]


def _build_security_args(run_as_host_user: bool, run_exec: bool = False, snap_compat: bool = False) -> list[str]:
    """Security/cap/tmpfs args for the privilege mode; ``run_exec`` mounts /run exec for s6 images.
    ``snap_compat`` drops no-new-privileges: snap-packaged Docker's AppArmor profile turns it into
    "exec: operation not permitted" for every process in the container (#9730, LP#1908448)."""
    args = list(_BASE_SECURITY_ARGS) + ([] if snap_compat else list(_NO_NEW_PRIVILEGES_ARGS))
    args += list(_RUN_TMPFS_EXEC if run_exec else _RUN_TMPFS_NOEXEC)
    return args if run_as_host_user else args + list(_PRIVDROP_CAP_ARGS)


def _image_uses_init_entrypoint(docker_exe: str, image: str) -> bool:
    """True if ``image``'s entrypoint is the s6-overlay ``/init`` (own PID 1: incompatible
    with ``--init`` and a noexec /run). Any inspection failure, including an image not yet
    pulled, returns False and keeps the hardened defaults."""
    result = _docker_query(
        [docker_exe, "image", "inspect", image, "--format", "{{json .Config.Entrypoint}}"], timeout=15,
        fail="Docker: could not inspect entrypoint for %s: %s", fail_args=(image,),
        nonzero="Docker: image inspect for %s returned %d (stderr=%s)",
        exc_types=(subprocess.SubprocessError, OSError))
    raw = (result.stdout or "").strip() if result is not None else ""
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
    return str(entrypoint[0]).strip() in _S6_INIT_ENTRYPOINTS


def _resolve_host_user_spec() -> Optional[str]:
    """``<uid>:<gid>`` of the host user, or ``None`` without POSIX ids. Uses os.getuid/getgid
    directly (not pwd/getpass) so nameless UIDs inside sandboxed launchers never raise."""
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
    """Probe once per process whether ``--cpus``/``--memory``/``--pids-limit`` work here, via a
    throwaway ``sleep 0`` container from *image* (no extra pull). Without delegated cgroup
    controllers (unprivileged LXCs, rootless) these flags fail every start with exit 126;
    the result is host-wide, so it is cached."""
    global _cgroup_limits_ok
    if _cgroup_limits_ok is not None:
        return _cgroup_limits_ok

    docker_exe = find_docker()
    if not docker_exe or not image:
        _cgroup_limits_ok = False
        return False

    try:
        result = run_capture(
            [docker_exe, "run", "--rm", "--cpus", "0.5", "--memory", "64m", "--pids-limit", "32",
             image, "sleep", "0"],
            timeout=60)
        _cgroup_limits_ok = result.returncode == 0
        if not _cgroup_limits_ok:
            logger.warning(
                "Cgroup resource limits (--cpus/--memory/--pids-limit) not "
                "available in this environment. Containers will run without "
                "CPU, memory or PID limits. To enable, delegate the cpu, "
                "memory and pids cgroup controllers to this container. Probe stderr: %s",
                (result.stderr or "").strip()[:500])
    except Exception as e:
        _cgroup_limits_ok = False
        logger.warning("Cgroup limit probe failed; disabling resource limits: %s", e)
    return _cgroup_limits_ok


def _docker_unavailable(log_msg: str, *log_args, error: str, hint: str, exc_info: bool = False):
    logger.error(log_msg, *log_args, exc_info=exc_info)
    return EnvironmentConnectionError(error, retry_hint=hint)


def _ensure_docker_available() -> None:
    """Fail fast with an ``EnvironmentConnectionError`` when the docker CLI/daemon is unusable."""
    docker_exe = find_docker()
    if not docker_exe:
        raise _docker_unavailable(
            "Docker backend selected but no docker executable was found in PATH "
            "or known install locations. Install Docker Desktop and ensure the CLI is available.",
            error="Docker executable not found in PATH or known install locations. "
                  "Install Docker and ensure the 'docker' command is available.",
            hint="Install Docker (or fix PATH) and retry, or switch terminal.backend to 'local'.")
    try:
        result = run_capture([docker_exe, "version"], timeout=5)
    except FileNotFoundError:
        raise _docker_unavailable(
            "Docker backend selected but the resolved docker executable '%s' could not be executed.",
            docker_exe, exc_info=True,
            error="Docker executable could not be executed. Check your Docker installation.",
            hint="Repair the Docker installation and retry.")
    except subprocess.TimeoutExpired:
        raise _docker_unavailable(
            "Docker backend selected but '%s version' timed out. The Docker daemon may not be running.",
            docker_exe, exc_info=True,
            error="Docker daemon is not responding. Ensure Docker is running and try again.",
            hint="Start the Docker daemon (e.g. `systemctl start docker` or "
                 "launch Docker Desktop), then retry the same command.")
    except Exception:
        logger.error("Unexpected error while checking Docker availability.", exc_info=True)
        raise
    if result.returncode != 0:
        raise _docker_unavailable(
            "Docker backend selected but '%s version' failed (exit code %d, stderr=%s)",
            docker_exe, result.returncode, result.stderr.strip(),
            error="Docker command is available but 'docker version' failed. Check your Docker installation.",
            hint="The Docker daemon may be down or the current user lacks "
                 "permission (docker group). Fix and retry.")


def _name_only_env_args(names) -> list[str]:
    """``-e KEY`` flags (no values): the docker CLI resolves values from its own env,
    so secrets live in owner-readable /proc/*/environ instead of world-readable cmdline."""
    return [arg for key in sorted(names) for arg in ("-e", key)]


# Mount kinds declared by skills/credential_files: (getter, expects_file, log noun).
_RO_MOUNT_SOURCES = (
    ("get_credential_file_mounts", True, "credential"),
    ("get_skills_directory_mount", False, "skills dir"),
    ("get_cache_directory_mounts", False, "cache dir"))


def _readonly_skill_mount_args() -> list[str]:
    """``-v host:container:ro`` args for credential files, skill dirs and cache dirs. Read-only so the
    container can authenticate/read but never modify host state. Missing or wrong-kind sources are
    skipped with a warning (Docker-in-Docker auto-creates a missing file source as a directory,
    which would exit 125)."""
    args: list[str] = []
    try:
        import tools.credential_files as cf
        for getter, expects_file, noun in _RO_MOUNT_SOURCES:
            for entry in getattr(cf, getter)():
                src = Path(entry["host_path"])
                if expects_file:
                    problem = ("source is a directory (likely Docker-in-Docker auto-creation)" if src.is_dir()
                               else None if src.is_file() else "source not found")
                else:
                    problem = None if src.is_dir() else "source is not a directory"
                if problem:
                    logger.warning("Docker: skipping %s mount — %s: %s", noun.split()[0], problem, src)
                    continue
                args.extend(["-v", f"{entry['host_path']}:{entry['container_path']}:ro"])
                logger.info("Docker: mounting %s %s -> %s", noun, entry["host_path"], entry["container_path"])
    except Exception as e:
        logger.debug("Docker: could not load credential file mounts: %s", e)
    return args


def _host_user_args(run_as_host_user: bool) -> list[str]:
    """``--user uid:gid`` so bind-mount writes are owned by the host user, not root. Without
    POSIX ids fall back to the full cap set — the image's init may still need to drop privileges."""
    if not run_as_host_user:
        return []
    user_spec = _resolve_host_user_spec()
    if user_spec is not None:
        logger.info("Docker: running container as host user %s", user_spec)
        return ["--user", user_spec]
    logger.warning(
        "docker_run_as_host_user is enabled but this platform does "
        "not expose POSIX uid/gid; container will start as its image default user.")
    return []


class DockerEnvironment(BaseEnvironment):
    """Hardened Docker container execution (caps dropped, no-new-privileges, PID limits,
    size-limited tmpfs). The container is the security boundary — its filesystem stays
    writable so agents can install packages. Persistence bind-mounts /workspace and /root."""

    _profile_scoped_passthrough = True

    def _additional_profile_scoped_passthrough_names(self) -> tuple[str, ...]:
        """Keep explicit docker_forward_env values out of shared snapshots."""
        return tuple(self._forward_env)

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
        host_cwd: Optional[str] = None,
        auto_mount_cwd: bool = False,
        run_as_host_user: bool = False,
        extra_args: list = None,
        persist_across_processes: bool = True,
        shm_size: str = _DEFAULT_SHM_SIZE,
        shared_container_key: str = "",
        snap_compat: bool = False):
        if cwd == "~":
            cwd = "/root"
        super().__init__(cwd=cwd, timeout=timeout)
        self._persistent = persistent_filesystem
        self._persist_across_processes = persist_across_processes
        # Set by terminal_tool._create_environment for session-scoped containers
        # (docker + container_persistent: false): removed at session close/idle timeout.
        self._session_scoped = False
        self._task_id = task_id
        self._forward_env = _normalize_forward_env_names(forward_env)
        self._env = _normalize_env_dict(env)
        self._init_unset_passthrough_names: tuple[str, ...] = ()
        self._container_id: Optional[str] = None
        self._init_env_values: dict[str, str] = {}
        self._workspace_dir: Optional[str] = None
        self._home_dir: Optional[str] = None
        logger.info("DockerEnvironment volumes: %s", volumes)
        if volumes is not None and not isinstance(volumes, list):
            logger.warning("docker_volumes config is not a list: %r", volumes)
            volumes = []

        _ensure_docker_available()

        resource_args = self._resource_args(image, cpu, memory, disk, network, shm_size, extra_args)
        volume_args, writable_args = self._mount_args(volumes, host_cwd, auto_mount_cwd, task_id)
        volume_args.extend(_readonly_skill_mount_args())
        egress_label, egress_volume_args, egress_host_args, env_args, validated_extra = (
            self._egress_and_env_args(extra_args))
        volume_args.extend(egress_volume_args)
        user_args = _host_user_args(run_as_host_user)

        # Resolved once so it works when /usr/local/bin is not in PATH (macOS services).
        self._docker_exe = find_docker() or "docker"

        # s6-overlay images (e.g. hermes-agent:latest) already use /init as PID 1 and exec
        # /run/s6/basedir/bin/init during startup. For those images we must (a) skip Docker's --init (two
        # competing PID-1 inits) and (b) mount /run with exec instead of noexec, or s6 stage0 dies with exit
        # 126 "Permission denied". Detected once here; defaults are kept on any inspection failure. See
        # issue #34628.
        image_uses_s6_init = _image_uses_init_entrypoint(self._docker_exe, image)
        if image_uses_s6_init:
            logger.info(
                "Docker: image %s uses /init (s6-overlay) as entrypoint — "
                "skipping --init and mounting /run with exec.",
                image)
        security_args = _build_security_args(
            run_as_host_user and bool(user_args), run_exec=image_uses_s6_init, snap_compat=snap_compat)
        self._snap_compat = snap_compat
        if snap_compat:
            logger.warning(
                "docker_snap_compat: running without --init and no-new-privileges (snap Docker under AppArmor)")

        logger.info("Docker volume_args: %s", volume_args)
        # docker_extra_args go last so they can override defaults.
        all_run_args = (
            security_args + user_args + writable_args + resource_args
            + egress_host_args + volume_args + env_args + validated_extra)
        logger.info("Docker run_args: %s", all_run_args)

        # Labels identify hermes containers to the orphan reaper (hermes-agent=1),
        # cross-process reuse (task-id/profile) and operators. The reuse identity
        # is captured at start and never changes for the container's lifetime.
        # Egress posture gets its own label: env/CA mounts are immutable after
        # creation, so reusing a pre-egress container would bypass the firewall.
        profile_name = _container_identity(shared_container_key)
        task_label = _sanitize_label_value(task_id)
        self._labels = {
            "hermes-agent": "1",
            "hermes-task-id": task_label,
            "hermes-profile": profile_name,
            _EGRESS_LABEL_KEY: egress_label}
        # Saved for container recreation on "No such container" recovery.
        self._image = image
        self._image_uses_s6_init = image_uses_s6_init
        self._all_run_args = all_run_args

        reused = persist_across_processes and self._attach_existing_container(
            task_label, profile_name, egress_label, network)
        if not reused:
            self._container_id = self._docker_run(cwd)

        # Init-time env forwarding args seed the snapshot.
        self._init_env_args = self._build_init_env_args()
        self.init_session()

    # --- __init__ helpers ---
    def _egress_and_env_args(self, extra_args) -> tuple[str, list[str], list[str], list[str], list[str]]:
        """Egress credential-injection proxy plumbing (CA mount + HTTPS_PROXY/CA-bundle env so
        outbound traffic routes through the host-side proxy and the sandbox receives proxy tokens
        instead of real API keys), merged with docker_env into name-only ``-e`` args, plus the
        validated docker_extra_args. Returns ``(egress_label, volume_args, host_args, env_args,
        validated_extra)``; sets ``self._run_env_values`` (injected into the docker-client
        subprocess env at run time and reused verbatim by container-recreation recovery)."""
        egress_volume_args, egress_env_overrides, egress_host_args = _egress_proxy_args_for_docker()
        egress_label = _egress_reuse_fingerprint(egress_volume_args, egress_env_overrides, egress_host_args)
        enforce_egress = _egress_enforce_on_docker() if egress_env_overrides else True
        critical_egress_names = _critical_egress_env_names(egress_env_overrides)
        if egress_env_overrides:
            check_forward_env_collisions(self._forward_env, critical_egress_names, enforce_egress)
            check_docker_env_collisions(self._env, egress_env_overrides, enforce_egress)

        merged_env = merge_egress_env(self._env, egress_env_overrides, enforce_egress)
        self._run_env_values = dict(merged_env)

        validated_extra = []
        for arg in (extra_args or []):
            if not isinstance(arg, str):
                logger.warning("Ignoring non-string docker_extra_args entry: %r", arg)
                continue
            validated_extra.append(arg)
        if egress_env_overrides:
            check_extra_args_collisions(validated_extra, critical_egress_names, enforce_egress)
        return egress_label, egress_volume_args, egress_host_args, _name_only_env_args(merged_env), validated_extra

    def _resource_args(self, image, cpu, memory, disk, network, shm_size, extra_args) -> list[str]:
        """cgroup-gated CPU/memory/pids limits, shm size, disk quota and network mode."""
        args: list[str] = []
        if _cgroup_limits_available(image):
            if cpu > 0:
                args.extend(["--cpus", str(cpu)])
            if memory > 0:
                args.extend(["--memory", f"{memory}m"])
            args.extend(["--pids-limit", _DEFAULT_PIDS_LIMIT])
        # --shm-size is a tmpfs option, not cgroup-gated. Skipped when the user
        # sets it in docker_extra_args or opts out with empty/"0".
        shm = str(shm_size or "").strip()
        if shm and shm != "0" and not _extra_args_set_shm_size(extra_args):
            args.extend(["--shm-size", shm])
        if disk > 0 and sys.platform != "darwin":
            if self._storage_opt_supported():
                args.extend(["--storage-opt", f"size={disk}m"])
            else:
                logger.warning(
                    "Docker storage driver does not support per-container disk limits "
                    "(requires overlay2 on XFS with pquota). Container will run without disk quota.")
        if not network:
            extra_network = _extra_args_network_mode(extra_args)
            if extra_network is None:
                args.append("--network=none")
            elif extra_network != "none":
                # Contradictory intent: honouring the extra arg would hand the agent a networked
                # container despite the configured lockdown, and the reuse guard (NetworkMode ==
                # "none") would churn it every startup. Fail closed and name both keys.
                raise RuntimeError(
                    f"terminal.docker_network is false (air-gapped) but terminal.docker_extra_args "
                    f"requests --network={extra_network!r}. Remove the --network entry from "
                    f"docker_extra_args, or set docker_network: true to opt into that network.")
            # extra_network == "none": same intent stated twice; the extra arg carries it once.
        return args

    def _mount_args(self, volumes, host_cwd, auto_mount_cwd, task_id) -> tuple[list[str], list[str]]:
        """``(volume_args, writable_args)`` for user volumes, host cwd and /workspace,/root.
        Persistent mode bind-mounts from TERMINAL_SANDBOX_DIR (default ~/.hermes/sandboxes/)."""
        volume_args: list[str] = []
        for vol in (volumes or []):
            if not isinstance(vol, str):
                logger.warning("Docker volume entry is not a string: %r", vol)
                continue
            vol = vol.strip()
            if not vol:
                continue
            if ":" not in vol:
                logger.warning("Docker volume '%s' missing colon, skipping", vol)
                continue
            volume_args.extend(["-v", vol])
        workspace_explicitly_mounted = any(":/workspace" in v for v in volume_args)

        host_cwd_abs = os.path.abspath(os.path.expanduser(host_cwd)) if host_cwd else ""
        bind_host_cwd = (
            auto_mount_cwd and bool(host_cwd_abs) and os.path.isdir(host_cwd_abs)
            and not workspace_explicitly_mounted)
        if auto_mount_cwd and host_cwd and not os.path.isdir(host_cwd_abs):
            logger.debug("Skipping docker cwd mount: host_cwd is not a valid directory: %s", host_cwd)
        mount_workspace = not bind_host_cwd and not workspace_explicitly_mounted

        writable_args: list[str] = []
        if self._persistent:
            from tools.environments.base import get_sandbox_dir
            # _sandbox_dir_name(): a raw session-key task_id carries colons,
            # which `-v` reads as extra spec fields (exit 125).
            sandbox = get_sandbox_dir() / "docker" / _sandbox_dir_name(task_id)
            self._home_dir = str(sandbox / "home")
            os.makedirs(self._home_dir, exist_ok=True)
            writable_args += ["-v", f"{self._home_dir}:/root"]
            if mount_workspace:
                self._workspace_dir = str(sandbox / "workspace")
                os.makedirs(self._workspace_dir, exist_ok=True)
                writable_args += ["-v", f"{self._workspace_dir}:/workspace"]
        else:
            writable_args += ["--tmpfs", "/workspace:rw,exec,size=10g"] if mount_workspace else []
            writable_args += ["--tmpfs", "/home:rw,exec,size=1g", "--tmpfs", "/root:rw,exec,size=1g"]

        if bind_host_cwd:
            logger.info("Mounting configured host cwd to /workspace: %s", host_cwd_abs)
            volume_args = ["-v", f"{host_cwd_abs}:/workspace", *volume_args]
        elif workspace_explicitly_mounted:
            logger.debug("Skipping docker cwd mount: /workspace already mounted by user config")
        return volume_args, writable_args

    def _attach_existing_container(self, task_label, profile_name, egress_label, network: bool) -> bool:
        """Attach to a prior process's labeled container ("ONE long-lived container shared
        across sessions"; opt out via ``docker_persist_across_processes: false``).
        Network guard is lockdown-only: a bridge container under ``docker_network: false``
        is removed and recreated, but a ``none`` container under default config is kept so
        ``--network=none`` in extra args doesn't churn containers every startup."""
        existing = self._find_reusable_container(task_label, profile_name, egress_label)
        if existing is None:
            return False
        container_id, state = existing
        if not network:
            actual_mode = self._container_network_mode(container_id)
            if actual_mode != "none":
                logger.warning(
                    "Existing container %s has NetworkMode=%s but "
                    "docker_network=false requests an air-gapped "
                    "container — removing it and starting fresh (task=%s, profile=%s).",
                    container_id[:12], actual_mode or "unknown", task_label, profile_name)
                try:
                    run_capture([self._docker_exe, "rm", "-f", container_id], timeout=30)
                except (subprocess.TimeoutExpired, OSError) as e:
                    logger.warning("Failed to remove mismatched container %s: %s", container_id[:12], e)
                return False

        if state != "running":
            err = self._start_container(container_id)
            if err is not None:
                logger.warning(
                    "Failed to start existing container %s (state=%s): "
                    "%s — falling back to a fresh container.",
                    container_id[:12], state, err)
                return False
        self._container_id = container_id
        logger.info(
            "Reusing container %s (task=%s, profile=%s, prior state=%s)",
            container_id[:12], task_label, profile_name, state)
        return True

    def _start_container(self, container_id: str) -> Exception | None:
        """``docker start`` a stopped container; returns the failure instead of raising."""
        try:
            run_capture([self._docker_exe, "start", container_id], timeout=30, check=True)
            return None
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            return e

    def _run_command(self, name: str, workdir: str) -> list[str]:
        """``docker run -d`` argv for a fresh ``sleep infinity`` container (idle reaper handles
        lifetime). s6-overlay images already provide PID 1, so ``--init`` is skipped for them."""
        label_args = [arg for k, v in self._labels.items() for arg in ("--label", f"{k}={v}")]
        return [
            # tini/catatonit as PID 1 reaps zombie children — but s6-overlay images already provide their
            # own /init PID 1, so adding --init there creates two competing inits and breaks startup
            # (#34628).
            self._docker_exe, "run", "-d",
            *([] if self._image_uses_s6_init or self._snap_compat else ["--init"]),
            "--name", name,
            *label_args,
            "-w", workdir,
            *self._all_run_args,
            self._image,
            "sleep", "infinity"]

    def _docker_run(self, cwd: str) -> str:
        """Start a fresh container and return its id. A failed ``docker run`` (exit 125, timeout
        mid-pull) can leave a "Created" orphan the exited-only reaper never catches, so it is
        removed by name before re-raising."""
        container_name = f"hermes-{uuid.uuid4().hex[:8]}"
        run_cmd = self._run_command(container_name, cwd)
        logger.debug("Starting container: %s", ' '.join(run_cmd))
        try:
            result = run_capture(
                run_cmd, timeout=120, check=True,  # image pull may take a while
                env=self._docker_client_env(self._run_env_values))
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            logger.warning("docker run failed for %s, cleaning up orphaned container: %s", container_name, e)
            subprocess.run(
                [self._docker_exe, "rm", "-f", container_name],
                capture_output=True, timeout=10, stdin=subprocess.DEVNULL)
            raise
        container_id = result.stdout.strip()
        logger.info("Started container %s (%s)", container_name, container_id[:12])
        return container_id

    # --- Env forwarding ---
    def _docker_client_env(self, values: dict[str, str]) -> dict[str, str] | None:
        """Env for the docker-client subprocess carrying forwarded values (pairs with name-only
        ``-e KEY`` flags to keep secrets out of cmdline); ``None`` = inherit when empty.

        Name-only ``-e KEY`` flags make the docker CLI read each value from its own process environment,
        keeping secrets out of the client's world-readable ``/proc/<pid>/cmdline`` (issue #96268). Values
        live in ``/proc/<pid>/environ`` instead, which is owner/root-only. Returns ``None`` (inherit as
        before) when there is nothing to add.
        """
        return client_env_with(values)

    def _build_init_env_args(self) -> list[str]:
        """Name-only ``-e`` args for init_session so ``export -p`` captures docker_env
        plus the current profile's forwarded values (values travel via the client env).

        The VALUES intentionally do not appear in the argv — they are passed via the docker client
        subprocess env (see _docker_client_env and issue #96268); the flags here are name-only ``-e KEY``.
        """
        passthrough_env, unset_names = self._resolve_passthrough_env()
        exec_env = {**self._env, **passthrough_env}
        for name in unset_names:
            exec_env.pop(name, None)
        self._init_unset_passthrough_names = tuple(sorted(unset_names))
        self._init_env_values = dict(exec_env)
        return _name_only_env_args(exec_env)

    def _resolve_passthrough_env(self) -> tuple[dict[str, str], set[str]]:
        """See ``remote_common.resolve_passthrough_env``; explicit docker_forward_env bypasses the blocklist."""
        return resolve_passthrough_env(self._forward_env, hermes_env_loader=_load_hermes_env_vars)

    def _build_runtime_env_args_with_unsets(self) -> tuple[list[str], tuple[str, ...], dict[str, str]]:
        """Runtime name-only forwarding args, names absent from scope, and the values
        to inject into the docker client subprocess env.

        See #96268.
        """
        passthrough_env, unset_names = self._resolve_passthrough_env()
        return _name_only_env_args(passthrough_env), tuple(sorted(unset_names)), dict(passthrough_env)

    def _run_bash(self, cmd_string: str, *, login: bool = False,
                  timeout: int = 120,
                  stdin_data: str | None = None) -> subprocess.Popen:
        """Spawn bash inside the container. Init seeds the snapshot; profile-scoped passthrough
        values are re-injected on every command because one container can be shared by
        multiple routed profiles in a gateway process."""
        assert self._container_id, "Container not started"
        cmd = [self._docker_exe, "exec"]
        if stdin_data is not None:
            cmd.append("-i")

        # Init seeds the snapshot. Profile-scoped passthrough values are also injected on every later
        # command because this container can be shared by multiple routed profiles in one gateway process.
        # Env flags are name-only; values travel via the client subprocess env so they never hit
        # world-readable /proc/*/cmdline (#96268).
        unset_names: tuple[str, ...] = ()
        env_values: dict[str, str] = {}
        if login:
            cmd.extend(self._init_env_args)
            env_values = dict(getattr(self, "_init_env_values", {}))
            unset_names = getattr(self, "_init_unset_passthrough_names", ())
        elif self._profile_scoped_passthrough:
            runtime_args, unset_names, env_values = self._build_runtime_env_args_with_unsets()
            cmd.extend(runtime_args)
        cmd += [self._container_id, *bash_argv(prepend_unset(cmd_string, unset_names), login)]

        client_env = self._docker_client_env(env_values)
        return _popen_bash(cmd, stdin_data, env=client_env) if client_env is not None else _popen_bash(cmd, stdin_data)

    # --- "No such container" recovery ---
    _NO_CONTAINER_PATTERNS = ("No such container", "is not running", "no such container")

    def _is_container_gone(self, output: str) -> bool:
        return any(p in output for p in self._NO_CONTAINER_PATTERNS)

    def _recreate_container(self) -> bool:
        """Recreate a container removed out-of-band: label-based reuse first (another process
        may have recreated it), else a fresh one from the saved image/run-args. False when
        recovery fails so the caller surfaces the original error."""
        logger.warning("Container %s appears to be gone — attempting recovery", (self._container_id or "")[:12])
        self._container_id = None

        existing = self._find_reusable_container(
            self._labels.get("hermes-task-id", ""),
            self._labels.get("hermes-profile", ""),
            self._labels.get(_EGRESS_LABEL_KEY, "off"))
        if existing is not None:
            cid, state = existing
            if state == "running":
                self._container_id = cid
                logger.info("Recovery: reusing running container %s", cid[:12])
            elif (err := self._start_container(cid)) is None:
                self._container_id = cid
                logger.info("Recovery: restarted container %s", cid[:12])
            else:
                logger.warning("Recovery: failed to start container %s: %s", cid[:12], err)

        if not self._container_id:
            if not self._image:
                logger.error("Recovery: no saved image name, cannot recreate container")
                return False
            try:
                new_name = f"hermes-{uuid.uuid4().hex[:8]}"
                result = run_capture(
                    self._run_command(new_name, self.cwd), timeout=120, check=True,
                    env=self._docker_client_env(self._run_env_values))
                self._container_id = result.stdout.strip()
                logger.info("Recovery: created fresh container %s (%s)", new_name, self._container_id[:12])
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError) as e:
                logger.error("Recovery: failed to create new container: %s", e)
                return False

        try:
            self._snapshot_ready = False
            self.init_session()
        except Exception as e:
            logger.error("Recovery: init_session failed in new container: %s", e)
            return False

        logger.info("Recovery successful — new container %s", (self._container_id or "")[:12])
        return True

    def execute(self, command: str, cwd: str = "", **kwargs) -> dict:
        """Execute a command; if the container was removed out-of-band (idle reaper,
        docker prune, OOM, daemon restart) recreate it and retry once."""
        result = super().execute(command, cwd, **kwargs)
        if (
            result.get("returncode", 0) != 0
            and self._is_container_gone(result.get("output", ""))
            and self._persist_across_processes
            and self._recreate_container()):
            result = super().execute(command, cwd, **kwargs)
        return result

    @staticmethod
    def _storage_opt_supported() -> bool:
        """Whether ``--storage-opt size=`` works (only overlay2 on XFS with pquota; ext4 errors out)."""
        global _storage_opt_ok
        if _storage_opt_ok is not None:
            return _storage_opt_ok
        try:
            docker = find_docker() or "docker"
            result = run_capture([docker, "info", "--format", "{{.Driver}}"], timeout=10)
            if result.stdout.strip().lower() != "overlay2":
                _storage_opt_ok = False
                return False
            # Probe with a real create — the fastest reliable check.
            probe = run_capture([docker, "create", "--storage-opt", "size=1m", "hello-world"], timeout=15)
            _storage_opt_ok = probe.returncode == 0
            if _storage_opt_ok and probe.stdout.strip():
                subprocess.run([docker, "rm", probe.stdout.strip()],
                               capture_output=True, timeout=5, stdin=subprocess.DEVNULL)
        except Exception:
            _storage_opt_ok = False
        logger.debug("Docker --storage-opt support: %s", _storage_opt_ok)
        return _storage_opt_ok

    def _container_network_mode(self, container_id: str) -> Optional[str]:
        """``HostConfig.NetworkMode`` of a container, or ``None`` when inspection fails (callers
        treat ``None`` as a mismatch under lockdown, so a failed inspect fails closed)."""
        result = _docker_query(
            [self._docker_exe, "inspect", "--format", "{{.HostConfig.NetworkMode}}", container_id], timeout=10,
            fail="docker inspect NetworkMode failed: %s", nonzero="docker inspect NetworkMode returned %d: %s")
        return (result.stdout.strip() or None) if result is not None else None

    def _find_reusable_container(
        self, task_label: str, profile_label: str, egress_label: str) -> Optional[tuple[str, str]]:
        """``(container_id, state)`` of an existing container labeled for this task/profile/
        egress posture, or ``None`` on miss or any failure. The egress posture is a label
        FILTER for every posture, "off" included: a container built with egress on must not be
        reused after ``hermes egress disable`` (baked-in proxy env and CA mounts), and every
        container this class creates carries the label. The ``{{.Label "key"}}`` template
        function is Docker-only — podman ps exits 125 on it — so the probe never uses it (#99213)."""
        filters = [
            "--filter", "label=hermes-agent=1",
            "--filter", f"label=hermes-task-id={task_label}",
            "--filter", f"label=hermes-profile={profile_label}",
            "--filter", f"label={_EGRESS_LABEL_KEY}={egress_label}"]
        result = _docker_query(
            [self._docker_exe, "ps", "-a", *filters, "--format", "{{.ID}}\t{{.State}}"], timeout=10,
            fail="docker ps probe failed: %s — will start a fresh container",
            nonzero="docker ps probe returned %d: %s — will start a fresh container")
        if result is None:
            return None
        # Multiple matches can happen after a crash mid-cleanup: prefer a running
        # one, else the first listed; stale duplicates are the orphan reaper's job.
        running = first = None
        for ln in (ln for ln in result.stdout.splitlines() if ln.strip()):
            parts = ln.split("\t", 1)
            if len(parts) != 2:
                continue
            cid, state = parts[0], parts[1].strip().lower()
            if first is None:
                first = (cid, state)
            if state == "running" and running is None:
                running = (cid, state)
        return running or first

    def _remove_bind_dirs(self) -> None:
        for d in (self._workspace_dir, self._home_dir):
            if d:
                shutil.rmtree(d, ignore_errors=True)

    def cleanup(self, *, force_remove: bool = False):
        """Tear down per persist mode. Persist mode (default) leaves the container RUNNING —
        stopping it on every exit would kill background processes and add a ``docker start``
        delay per session; reclamation is ``reap_orphan_containers()`` at next startup.
        ``persist_across_processes=False`` or ``force_remove=True`` (explicit-teardown hook,
        unused so far) does ``docker stop`` + ``docker rm -f`` on a daemon thread that the
        atexit hook joins via ``wait_for_cleanup`` so the work completes before exit.

        Cleanup runs on a daemon thread with bounded ``subprocess.run`` calls (not the racy ``Popen(... &)``
        pattern from before PR #33645). The atexit hook in ``tools/terminal_tool.py`` waits up to 15s for
        the thread to finish before the interpreter exits, so ``docker stop`` / ``docker rm`` actually
        completes when we do trigger it.
        """
        container_id = self._container_id
        if not container_id:
            # Bind-mount dirs are still dropped in non-persistent mode.
            if not self._persistent:
                self._remove_bind_dirs()
            return

        if not force_remove and self._persist_across_processes:
            # Drop the in-process handle so a fresh __init__ re-probes via
            # labels instead of reusing a stale Python reference.
            self._container_id = None
            return

        # Capture what the worker needs — the thread can outlive ``self``.
        docker_exe = self._docker_exe
        log_id = container_id[:12]

        def _do_cleanup() -> None:
            for argv, fail_msg in ((["stop", "-t", "10"], "docker stop %s timed out / failed: %s"),
                                   (["rm", "-f"], "docker rm -f %s failed: %s")):
                try:
                    subprocess.run(
                        [docker_exe, *argv, container_id],
                        capture_output=True, timeout=30, stdin=subprocess.DEVNULL)
                except (subprocess.TimeoutExpired, OSError) as e:
                    logger.warning(fail_msg, log_id, e)

        t = threading.Thread(target=_do_cleanup, daemon=True, name=f"hermes-cleanup-{log_id}")
        with _TEARDOWN_LOCK:
            _TEARDOWN_THREADS.add(t)
        t.start()
        self._cleanup_thread = t
        self._container_id = None

        # Bind-mount dirs are the container's filesystem state; only drop them
        # once the container itself is removed.
        if not self._persistent:
            self._remove_bind_dirs()

    @staticmethod
    def wait_for_all_teardowns(timeout: float = 15.0) -> bool:
        """Join every in-flight teardown worker, including those of envs the idle reaper already
        popped out of the active registry (their handles are otherwise unreachable at exit, so
        ``docker rm`` died with the interpreter and left a stopped container behind — #86317).
        Re-snapshots each pass: the reaper may start a worker while the drain runs."""
        deadline = time.monotonic() + timeout
        while True:
            with _TEARDOWN_LOCK:
                _TEARDOWN_THREADS.difference_update({t for t in _TEARDOWN_THREADS if not t.is_alive()})
                pending = list(_TEARDOWN_THREADS)
            if not pending:
                return True
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False
            pending[0].join(timeout=remaining)

    def wait_for_cleanup(self, timeout: float = 30.0) -> bool:
        """Block up to *timeout* seconds for the cleanup thread (atexit hook). True if it
        finished or none was started, False on timeout."""
        thread = getattr(self, "_cleanup_thread", None)
        if thread is None or not thread.is_alive():
            return True
        thread.join(timeout=timeout)
        return not thread.is_alive()
