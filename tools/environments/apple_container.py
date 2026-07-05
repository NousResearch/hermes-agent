"""Apple container execution environment.

Runs Hermes terminal commands inside Apple's ``container`` runtime, which
executes Linux containers as lightweight VMs on macOS. The backend intentionally
uses the public CLI surface so it remains decoupled from the Swift package API.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
import uuid
from pathlib import Path
from typing import Optional

from tools.environments.base import BaseEnvironment, _popen_bash
from tools.environments.local import (
    _HERMES_PROVIDER_ENV_BLOCKLIST,
    _is_hermes_internal_secret,
)

logger = logging.getLogger(__name__)


_APPLE_CONTAINER_SEARCH_PATHS = [
    "/usr/local/bin/container",
    "/opt/homebrew/bin/container",
]

_apple_container_executable: Optional[str] = None
_ENV_VAR_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_CONTAINER_NAME_UNSAFE_RE = re.compile(r"[^A-Za-z0-9_.-]")


def _normalize_forward_env_names(forward_env: list[str] | None) -> list[str]:
    """Return a deduplicated list of valid environment variable names."""
    normalized: list[str] = []
    seen: set[str] = set()

    for item in forward_env or []:
        if not isinstance(item, str):
            logger.warning("Ignoring non-string apple_container_forward_env entry: %r", item)
            continue
        key = item.strip()
        if not key:
            continue
        if not _ENV_VAR_NAME_RE.match(key):
            logger.warning("Ignoring invalid apple_container_forward_env entry: %r", item)
            continue
        if key in seen:
            continue
        seen.add(key)
        normalized.append(key)

    return normalized


def _normalize_env_dict(env: dict | None) -> dict[str, str]:
    """Validate and normalize an ``apple_container_env`` dict."""
    if not env:
        return {}
    if not isinstance(env, dict):
        logger.warning("apple_container_env is not a dict: %r", env)
        return {}

    normalized: dict[str, str] = {}
    for key, value in env.items():
        if not isinstance(key, str) or not _ENV_VAR_NAME_RE.match(key.strip()):
            logger.warning("Ignoring invalid apple_container_env key: %r", key)
            continue
        key = key.strip()
        if not isinstance(value, str):
            if isinstance(value, (int, float, bool)):
                value = str(value)
            else:
                logger.warning(
                    "Ignoring non-string apple_container_env value for %r: %r",
                    key,
                    value,
                )
                continue
        normalized[key] = value

    return normalized


def _load_hermes_env_vars() -> dict[str, str]:
    """Load ~/.hermes/.env values without failing command execution."""
    try:
        from hermes_cli.config import load_env

        return load_env() or {}
    except Exception:
        return {}


def _get_active_profile_name() -> str:
    """Return the active Hermes profile name, or ``default`` on any error."""
    try:
        from hermes_cli.profiles import get_active_profile_name

        return get_active_profile_name() or "default"
    except Exception:
        return "default"


def _sanitize_container_name_part(value: str) -> str:
    if not isinstance(value, str) or not value:
        return "default"
    cleaned = _CONTAINER_NAME_UNSAFE_RE.sub("-", value).strip(".-")
    return cleaned or "default"


def _deterministic_container_name(task_id: str, profile_name: str) -> str:
    """Return a stable Apple container name for cross-process reuse."""
    task = _sanitize_container_name_part(task_id)[:24]
    profile = _sanitize_container_name_part(profile_name)[:18]
    suffix = uuid.uuid5(
        uuid.NAMESPACE_URL,
        f"hermes-agent/apple-container/{profile_name}/{task_id}",
    ).hex[:10]
    return f"hermes-{profile}-{task}-{suffix}"[:63]


def _resolve_host_user_spec() -> Optional[str]:
    """Return ``uid:gid`` for the current host user, if POSIX ids exist."""
    get_uid = getattr(os, "getuid", None)
    get_gid = getattr(os, "getgid", None)
    if get_uid is None or get_gid is None:
        return None
    try:
        return f"{get_uid()}:{get_gid()}"
    except Exception:  # pragma: no cover - defensive
        return None


def _bind_mount_arg(source: str, target: str, *, readonly: bool = False) -> str:
    arg = f"type=bind,source={source},target={target}"
    if readonly:
        arg += ",readonly"
    return arg


def find_apple_container(binary: str | None = None) -> Optional[str]:
    """Locate Apple's ``container`` CLI binary.

    Resolution order:
    1. Explicit constructor/config value.
    2. ``TERMINAL_APPLE_CONTAINER_BINARY`` or ``HERMES_APPLE_CONTAINER_BINARY``.
    3. ``container`` on PATH.
    4. Common installer/Homebrew paths.
    """
    global _apple_container_executable

    candidates = [
        binary,
        os.getenv("TERMINAL_APPLE_CONTAINER_BINARY"),
        os.getenv("HERMES_APPLE_CONTAINER_BINARY"),
    ]
    for candidate in candidates:
        if not candidate:
            continue
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate
        found = shutil.which(candidate)
        if found:
            return found

    if _apple_container_executable is not None:
        return _apple_container_executable

    found = shutil.which("container")
    if found:
        _apple_container_executable = found
        return found

    for path in _APPLE_CONTAINER_SEARCH_PATHS:
        if os.path.isfile(path) and os.access(path, os.X_OK):
            _apple_container_executable = path
            logger.info("Found Apple container at non-PATH location: %s", path)
            return path

    return None


def _ensure_apple_container_available(binary: str | None = None) -> str:
    """Return a usable CLI path or raise a clear configuration error."""
    container_exe = find_apple_container(binary)
    if not container_exe:
        raise RuntimeError(
            "Apple container backend selected but no 'container' executable was "
            "found. Install apple/container, run 'container system start', or "
            "set TERMINAL_APPLE_CONTAINER_BINARY."
        )

    try:
        result = subprocess.run(
            [container_exe, "system", "status", "--format", "json"],
            capture_output=True,
            text=True,
            timeout=10,
            stdin=subprocess.DEVNULL,
        )
    except FileNotFoundError:
        raise RuntimeError(
            "Apple container executable could not be executed. Check "
            "TERMINAL_APPLE_CONTAINER_BINARY."
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError(
            "Apple container system status timed out. Ensure the container "
            "system service is running."
        )

    if result.returncode != 0:
        stderr = (result.stderr or "").strip()
        raise RuntimeError(
            "Apple container system is not ready. Run 'container system start' "
            f"and retry. {stderr}".strip()
        )

    return container_exe


class AppleContainerEnvironment(BaseEnvironment):
    """Hermes execution backend backed by Apple's ``container`` CLI."""

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
        persist_across_processes: bool = False,
        binary: str | None = None,
    ):
        if cwd == "~":
            cwd = "/root"
        super().__init__(cwd=cwd, timeout=timeout)
        self._persistent = persistent_filesystem
        self._persist_across_processes = persist_across_processes
        self._task_id = task_id
        self._forward_env = _normalize_forward_env_names(forward_env)
        self._env = _normalize_env_dict(env)
        self._container_id: Optional[str] = None
        self._workspace_dir: Optional[str] = None
        self._home_dir: Optional[str] = None
        self._cleanup_thread = None

        if volumes is not None and not isinstance(volumes, list):
            logger.warning("apple_container_volumes config is not a list: %r", volumes)
            volumes = []

        self._container_exe = _ensure_apple_container_available(binary)

        resource_args: list[str] = []
        if cpu > 0:
            resource_args.extend(["--cpus", str(cpu)])
        if memory > 0:
            resource_args.extend(["--memory", f"{memory}M"])
        if disk > 0:
            logger.debug(
                "Apple container backend does not expose a per-container disk "
                "quota flag; ignoring container_disk=%s.",
                disk,
            )
        if not network:
            resource_args.extend(["--network", "none"])

        mount_args: list[str] = []
        workspace_explicitly_mounted = False
        for vol in volumes or []:
            if not isinstance(vol, str):
                logger.warning("Apple container volume entry is not a string: %r", vol)
                continue
            vol = vol.strip()
            if not vol:
                continue
            if ":" not in vol:
                logger.warning("Apple container volume %r missing colon, skipping", vol)
                continue
            mount_args.extend(["--volume", vol])
            if ":/workspace" in vol:
                workspace_explicitly_mounted = True

        host_cwd_abs = os.path.abspath(os.path.expanduser(host_cwd)) if host_cwd else ""
        bind_host_cwd = (
            auto_mount_cwd
            and bool(host_cwd_abs)
            and os.path.isdir(host_cwd_abs)
            and not workspace_explicitly_mounted
        )
        if auto_mount_cwd and host_cwd and not os.path.isdir(host_cwd_abs):
            logger.debug(
                "Skipping Apple container cwd mount: host_cwd is not a valid directory: %s",
                host_cwd,
            )

        from tools.environments.base import get_sandbox_dir

        sandbox = get_sandbox_dir() / "apple_container" / (
            task_id if self._persistent else uuid.uuid4().hex[:12]
        )
        self._home_dir = str(sandbox / "home")
        os.makedirs(self._home_dir, exist_ok=True)
        mount_args.extend([
            "--mount", _bind_mount_arg(self._home_dir, "/root"),
        ])

        if bind_host_cwd:
            logger.info("Mounting configured host cwd to /workspace: %s", host_cwd_abs)
            mount_args = [
                "--mount", _bind_mount_arg(host_cwd_abs, "/workspace"),
                *mount_args,
            ]
        elif not workspace_explicitly_mounted:
            self._workspace_dir = str(sandbox / "workspace")
            os.makedirs(self._workspace_dir, exist_ok=True)
            mount_args.extend([
                "--mount", _bind_mount_arg(self._workspace_dir, "/workspace"),
            ])
        else:
            logger.debug("Skipping Apple container cwd mount: /workspace already mounted")

        try:
            from tools.credential_files import (
                get_cache_directory_mounts,
                get_credential_file_mounts,
                get_skills_directory_mount,
            )

            for mount_entry in get_credential_file_mounts():
                src = Path(mount_entry["host_path"])
                if not src.is_file():
                    logger.warning(
                        "Apple container: skipping credential mount; source not found: %s",
                        src,
                    )
                    continue
                mount_args.extend([
                    "--mount",
                    _bind_mount_arg(
                        mount_entry["host_path"],
                        mount_entry["container_path"],
                        readonly=True,
                    ),
                ])

            for skills_mount in get_skills_directory_mount():
                src = Path(skills_mount["host_path"])
                if not src.is_dir():
                    logger.warning(
                        "Apple container: skipping skills mount; source is not a directory: %s",
                        src,
                    )
                    continue
                mount_args.extend([
                    "--mount",
                    _bind_mount_arg(
                        skills_mount["host_path"],
                        skills_mount["container_path"],
                        readonly=True,
                    ),
                ])

            for cache_mount in get_cache_directory_mounts():
                src = Path(cache_mount["host_path"])
                if not src.is_dir():
                    logger.warning(
                        "Apple container: skipping cache mount; source is not a directory: %s",
                        src,
                    )
                    continue
                mount_args.extend([
                    "--mount",
                    _bind_mount_arg(
                        cache_mount["host_path"],
                        cache_mount["container_path"],
                        readonly=True,
                    ),
                ])
        except Exception as e:
            logger.debug("Apple container: could not load credential file mounts: %s", e)

        env_args: list[str] = []
        for key in sorted(self._env):
            env_args.extend(["--env", f"{key}={self._env[key]}"])

        user_args: list[str] = []
        if run_as_host_user:
            user_spec = _resolve_host_user_spec()
            if user_spec is not None:
                user_args = ["--user", user_spec]
                logger.info("Apple container: running as host user %s", user_spec)
            else:
                logger.warning(
                    "apple_container_run_as_host_user is enabled but this platform "
                    "does not expose POSIX uid/gid; container will start as its "
                    "image default user."
                )

        extra_run_args: list[str] = []
        for arg in extra_args or []:
            if not isinstance(arg, str):
                logger.warning("Ignoring non-string apple_container_extra_args entry: %r", arg)
                continue
            extra_run_args.append(arg)

        profile_name = _get_active_profile_name()
        if persist_across_processes:
            container_name = _deterministic_container_name(task_id, profile_name)
        else:
            container_name = f"hermes-{uuid.uuid4().hex[:8]}"

        self._container_id = container_name
        run_args = [
            self._container_exe,
            "run",
            "--detach",
            "--init",
            "--name",
            container_name,
            "--label",
            "hermes-agent=1",
            "--label",
            f"hermes-task-id={_sanitize_container_name_part(task_id)}",
            "--label",
            f"hermes-profile={_sanitize_container_name_part(profile_name)}",
            "--workdir",
            cwd,
            "--cap-drop",
            "ALL",
            "--cap-add",
            "DAC_OVERRIDE",
            "--cap-add",
            "CHOWN",
            "--cap-add",
            "FOWNER",
            *user_args,
            *resource_args,
            *mount_args,
            *env_args,
            *extra_run_args,
            image,
            "sleep",
            "infinity",
        ]

        reused = self._try_reuse_container(container_name) if persist_across_processes else False
        if not reused:
            try:
                subprocess.run(
                    run_args,
                    capture_output=True,
                    text=True,
                    timeout=180,
                    check=True,
                    stdin=subprocess.DEVNULL,
                )
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
                subprocess.run(
                    [self._container_exe, "delete", "--force", container_name],
                    capture_output=True,
                    timeout=15,
                    stdin=subprocess.DEVNULL,
                )
                self._container_id = None
                raise
            logger.info("Started Apple container %s", container_name)

        self._init_env_args = self._build_init_env_args()
        self.init_session()

    def _try_reuse_container(self, container_name: str) -> bool:
        """Reuse or restart a named Apple container when available."""
        try:
            probe = subprocess.run(
                [self._container_exe, "exec", container_name, "true"],
                capture_output=True,
                text=True,
                timeout=15,
                check=False,
                stdin=subprocess.DEVNULL,
            )
        except (subprocess.TimeoutExpired, OSError):
            return False
        if probe.returncode == 0:
            logger.info("Reusing running Apple container %s", container_name)
            return True

        try:
            started = subprocess.run(
                [self._container_exe, "start", container_name],
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
                stdin=subprocess.DEVNULL,
            )
        except (subprocess.TimeoutExpired, OSError):
            return False
        if started.returncode == 0:
            logger.info("Restarted Apple container %s", container_name)
            return True
        try:
            subprocess.run(
                [self._container_exe, "delete", "--force", container_name],
                capture_output=True,
                timeout=15,
                stdin=subprocess.DEVNULL,
            )
        except (subprocess.TimeoutExpired, OSError):
            pass
        return False

    def _build_init_env_args(self) -> list[str]:
        """Build ``--env KEY=VALUE`` args for init-session env capture."""
        exec_env: dict[str, str] = dict(self._env)

        explicit_forward_keys = set(self._forward_env)
        passthrough_keys: set[str] = set()
        try:
            from tools.env_passthrough import get_all_passthrough

            passthrough_keys = set(get_all_passthrough())
        except Exception:
            pass

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

        args: list[str] = []
        for key in sorted(exec_env):
            args.extend(["--env", f"{key}={exec_env[key]}"])
        return args

    def _run_bash(
        self,
        cmd_string: str,
        *,
        login: bool = False,
        timeout: int = 120,
        stdin_data: str | None = None,
    ) -> subprocess.Popen:
        """Spawn a bash process inside the Apple container."""
        assert self._container_id, "Container not started"
        cmd = [self._container_exe, "exec"]
        if stdin_data is not None:
            cmd.append("--interactive")
        if login:
            cmd.extend(self._init_env_args)
        cmd.append(self._container_id)
        if login:
            cmd.extend(["bash", "-l", "-c", cmd_string])
        else:
            cmd.extend(["bash", "-c", cmd_string])
        return _popen_bash(cmd, stdin_data)

    def cleanup(self, *, force_remove: bool = False):
        container_id = self._container_id
        if not container_id:
            if not self._persistent:
                for directory in (self._workspace_dir, self._home_dir):
                    if directory:
                        shutil.rmtree(directory, ignore_errors=True)
            return

        if self._persist_across_processes and not force_remove:
            self._container_id = None
            return

        container_exe = self._container_exe
        log_id = container_id[:12]

        def _do_cleanup() -> None:
            try:
                subprocess.run(
                    [container_exe, "stop", "--time", "10", container_id],
                    capture_output=True,
                    timeout=30,
                    stdin=subprocess.DEVNULL,
                )
            except (subprocess.TimeoutExpired, OSError) as e:
                logger.warning("Apple container stop %s timed out / failed: %s", log_id, e)
            try:
                subprocess.run(
                    [container_exe, "delete", "--force", container_id],
                    capture_output=True,
                    timeout=30,
                    stdin=subprocess.DEVNULL,
                )
            except (subprocess.TimeoutExpired, OSError) as e:
                logger.warning("Apple container delete %s failed: %s", log_id, e)

        import threading

        thread = threading.Thread(
            target=_do_cleanup,
            daemon=True,
            name=f"hermes-apple-container-cleanup-{log_id}",
        )
        thread.start()
        self._cleanup_thread = thread
        self._container_id = None

        if not self._persistent:
            for directory in (self._workspace_dir, self._home_dir):
                if directory:
                    shutil.rmtree(directory, ignore_errors=True)

    def wait_for_cleanup(self, timeout: float = 30.0) -> bool:
        thread = getattr(self, "_cleanup_thread", None)
        if thread is None or not thread.is_alive():
            return True
        thread.join(timeout=timeout)
        return not thread.is_alive()
