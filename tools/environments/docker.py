"""Docker execution environment for sandboxed command execution.

Security hardened (cap-drop ALL, no-new-privileges, PID limits),
configurable resource limits (CPU, memory, disk), and optional filesystem
persistence via bind mounts.
"""

import logging
import os
import re
import shutil
import subprocess
import sys
import uuid
from typing import Optional

from tools.environments.base import BaseEnvironment, _popen_bash
from tools.environments.local import _HERMES_PROVIDER_ENV_BLOCKLIST

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


def find_docker() -> Optional[str]:
    """Locate the docker CLI binary.

    Checks ``shutil.which`` first (respects PATH), then probes well-known
    install locations on macOS where Docker Desktop may not be in PATH
    (e.g. when running as a gateway service via launchd).

    Returns the absolute path, or ``None`` if docker cannot be found.
    """
    global _docker_executable
    if _docker_executable is not None:
        return _docker_executable

    found = shutil.which("docker")
    if found:
        _docker_executable = found
        return found

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
# Block privilege escalation and limit PIDs.
# /tmp is size-limited and nosuid but allows exec (needed by pip/npm builds).
_SECURITY_ARGS = [
    "--cap-drop", "ALL",
    "--cap-add", "DAC_OVERRIDE",
    "--cap-add", "CHOWN",
    "--cap-add", "FOWNER",
    "--security-opt", "no-new-privileges",
    "--pids-limit", "256",
    "--tmpfs", "/tmp:rw,nosuid,size=512m",
    "--tmpfs", "/var/tmp:rw,noexec,nosuid,size=256m",
    "--tmpfs", "/run:rw,noexec,nosuid,size=64m",
]


_storage_opt_ok: Optional[bool] = None  # cached result across instances


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
    ):
        if cwd == "~":
            cwd = "/root"
        super().__init__(cwd=cwd, timeout=timeout)
        self._persistent = persistent_filesystem
        self._task_id = task_id
        self._forward_env = _normalize_forward_env_names(forward_env)
        self._env = _normalize_env_dict(env)
        self._container_id: Optional[str] = None
        logger.info(f"DockerEnvironment volumes: {volumes}")
        # Ensure volumes is a list (config.yaml could be malformed)
        if volumes is not None and not isinstance(volumes, list):
            logger.warning(f"docker_volumes config is not a list: {volumes!r}")
            volumes = []

        # Fail fast if Docker is not available.
        _ensure_docker_available()

        # Build resource limit args
        resource_args = []
        if cpu > 0:
            resource_args.extend(["--cpus", str(cpu)])
        if memory > 0:
            resource_args.extend(["--memory", f"{memory}m"])
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
        if self._persistent:
            sandbox = get_sandbox_dir() / "docker" / task_id
            self._home_dir = str(sandbox / "home")
            os.makedirs(self._home_dir, exist_ok=True)
            writable_args.extend([
                "-v", f"{self._home_dir}:/root",
            ])
            if not bind_host_cwd and not workspace_explicitly_mounted:
                self._workspace_dir = str(sandbox / "workspace")
                os.makedirs(self._workspace_dir, exist_ok=True)
                writable_args.extend([
                    "-v", f"{self._workspace_dir}:/workspace",
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

        # Mount credential files (OAuth tokens, etc.) declared by skills.
        # Read-only so the container can authenticate but not modify host creds.
        try:
            from tools.credential_files import (
                get_credential_file_mounts,
                get_skills_directory_mount,
                get_cache_directory_mounts,
            )

            for mount_entry in get_credential_file_mounts():
                volume_args.extend([
                    "-v",
                    f"{mount_entry['host_path']}:{mount_entry['container_path']}:ro",
                ])
                logger.info(
                    "Docker: mounting credential %s -> %s",
                    mount_entry["host_path"],
                    mount_entry["container_path"],
                )

            # Mount skill directories (local + external) so skill
            # scripts/templates are available inside the container.
            for skills_mount in get_skills_directory_mount():
                volume_args.extend([
                    "-v",
                    f"{skills_mount['host_path']}:{skills_mount['container_path']}:ro",
                ])
                logger.info(
                    "Docker: mounting skills dir %s -> %s",
                    skills_mount["host_path"],
                    skills_mount["container_path"],
                )

            # Mount host-side cache directories (documents, images, audio,
            # screenshots) so the agent can access uploaded files and other
            # cached media from inside the container.  Read-only — the
            # container reads these but the host gateway manages writes.
            for cache_mount in get_cache_directory_mounts():
                volume_args.extend([
                    "-v",
                    f"{cache_mount['host_path']}:{cache_mount['container_path']}:ro",
                ])
                logger.info(
                    "Docker: mounting cache dir %s -> %s",
                    cache_mount["host_path"],
                    cache_mount["container_path"],
                )
        except Exception as e:
            logger.debug("Docker: could not load credential file mounts: %s", e)

        # Explicit environment variables (docker_env config) — set at container
        # creation so they're available to all processes (including entrypoint).
        env_args = []
        for key in sorted(self._env):
            env_args.extend(["-e", f"{key}={self._env[key]}"])

        logger.info(f"Docker volume_args: {volume_args}")
        all_run_args = list(_SECURITY_ARGS) + writable_args + resource_args + volume_args + env_args
        logger.info(f"Docker run_args: {all_run_args}")

        # Resolve the docker executable once so it works even when
        # /usr/local/bin is not in PATH (common on macOS gateway/service).
        self._docker_exe = find_docker() or "docker"

        # Start the container directly via `docker run -d`.
        container_name = f"hermes-{uuid.uuid4().hex[:8]}"
        run_cmd = [
            self._docker_exe, "run", "-d",
            "--init",           # tini/catatonit as PID 1 — reaps zombie children
            "--name", container_name,
            "-w", cwd,
            *all_run_args,
            image,
            "sleep", "infinity",  # no fixed lifetime — idle reaper handles cleanup
        ]
        logger.debug(f"Starting container: {' '.join(run_cmd)}")
        result = subprocess.run(
            run_cmd,
            capture_output=True,
            text=True,
            timeout=120,  # image pull may take a while
            check=True,
        )
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
        exec_env: dict[str, str] = dict(self._env)

        explicit_forward_keys = set(self._forward_env)
        passthrough_keys: set[str] = set()
        try:
            from tools.env_passthrough import get_all_passthrough
            passthrough_keys = set(get_all_passthrough())
        except Exception:
            pass
        # Explicit docker_forward_env entries are an intentional opt-in and must
        # win over the generic Hermes secret blocklist. Only implicit passthrough
        # keys are filtered.
        forward_keys = explicit_forward_keys | (passthrough_keys - _HERMES_PROVIDER_ENV_BLOCKLIST)
        hermes_env = _load_hermes_env_vars() if forward_keys else {}
        for key in sorted(forward_keys):
            value = os.getenv(key)
            if value is None:
                value = hermes_env.get(key)
            if value is not None:
                exec_env[key] = value

        args = []
        for key in sorted(exec_env):
            args.extend(["-e", f"{key}={exec_env[key]}"])
        return args

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
            )
            if probe.returncode == 0:
                # Clean up the created container
                container_id = probe.stdout.strip()
                if container_id:
                    subprocess.run([docker, "rm", container_id],
                                   capture_output=True, timeout=5)
                _storage_opt_ok = True
            else:
                _storage_opt_ok = False
        except Exception:
            _storage_opt_ok = False
        logger.debug("Docker --storage-opt support: %s", _storage_opt_ok)
        return _storage_opt_ok

    def cleanup(self):
        """Stop and remove the container. Bind-mount dirs persist if persistent=True."""
        if self._container_id:
            try:
                # Stop in background so cleanup doesn't block
                stop_cmd = (
                    f"(timeout 60 {self._docker_exe} stop {self._container_id} || "
                    f"{self._docker_exe} rm -f {self._container_id}) >/dev/null 2>&1 &"
                )
                subprocess.Popen(stop_cmd, shell=True)
            except Exception as e:
                logger.warning("Failed to stop container %s: %s", self._container_id, e)

            if not self._persistent:
                # Also schedule removal (stop only leaves it as stopped)
                try:
                    subprocess.Popen(
                        f"sleep 3 && {self._docker_exe} rm -f {self._container_id} >/dev/null 2>&1 &",
                        shell=True,
                    )
                except Exception:
                    pass
            self._container_id = None

        if not self._persistent:
            for d in (self._workspace_dir, self._home_dir):
                if d:
                    shutil.rmtree(d, ignore_errors=True)
