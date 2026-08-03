"""Apple Container execution environment for macOS.

Uses Apple's native containerization framework (macOS 26+) which runs each
Linux container inside its own lightweight virtual machine via
Virtualization.framework. Provides VM-level isolation (separate kernel per
container) with sub-second startup on Apple Silicon.

Security model: each container gets its own Linux kernel, so the VM boundary
is the primary isolation mechanism (stronger than Docker's namespace-based
isolation). Inside the container we additionally apply --read-only root
filesystem and writable tmpfs mounts for scratch directories, matching Docker
backend conventions where the CLI supports them.

Requires: macOS 26+, Apple Silicon, and the separately installed `container` CLI.
"""

import logging
import os
import platform
import posixpath
import shutil
import subprocess
import uuid
from pathlib import Path
from typing import Optional

from tools.environments.base import BaseEnvironment, _popen_bash, get_sandbox_dir

logger = logging.getLogger(__name__)

_CONTAINER_SEARCH_PATHS = [
    "/opt/homebrew/bin/container",
    "/usr/local/bin/container",
]

_container_executable: Optional[str] = None
_system_resources: Optional[dict] = None  # cached after first query
_HOST_REQUIREMENT = "macOS 26 or later on Apple Silicon (arm64)"


def is_apple_container_supported_host() -> bool:
    """Return whether this host can run Apple's Container runtime."""
    if platform.system() != "Darwin" or platform.machine().lower() != "arm64":
        return False
    try:
        major = int(platform.mac_ver()[0].split(".", 1)[0])
    except (AttributeError, IndexError, TypeError, ValueError):
        return False
    return major >= 26


def find_container_cli() -> Optional[str]:
    """Locate the Apple ``container`` CLI binary.

    Checks PATH first, then probes Homebrew install locations.
    Returns the absolute path, or None if not found.
    """
    global _container_executable
    if _container_executable is not None:
        return _container_executable

    found = shutil.which("container")
    if found:
        _container_executable = found
        return found

    for path in _CONTAINER_SEARCH_PATHS:
        if os.path.isfile(path) and os.access(path, os.X_OK):
            _container_executable = path
            logger.info("Found container CLI at non-PATH location: %s", path)
            return path

    return None


def container_system_status(executable: str | None = None) -> tuple[bool, str]:
    """Return Apple Container system readiness without changing host state."""
    exe = executable or find_container_cli()
    if not exe:
        return False, "CLI not found"
    try:
        result = subprocess.run(
            [exe, "system", "status"],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except subprocess.TimeoutExpired:
        return False, "status check timed out"
    except OSError as exc:
        return False, f"status check failed: {exc}"
    detail = (result.stdout or result.stderr or "").strip()
    return result.returncode == 0 and "running" in detail.lower(), detail


def _ensure_container_available() -> str:
    """Verify the Apple container CLI is available and the system is running.

    Returns the path to the container executable.
    Raises RuntimeError with actionable messages on failure.
    """
    if not is_apple_container_supported_host():
        raise RuntimeError(f"Apple Container requires {_HOST_REQUIREMENT}.")

    exe = find_container_cli()
    if not exe:
        raise RuntimeError(
            "Apple Container CLI not found. Install Apple Container manually; "
            "macOS 26 or later on Apple Silicon is required."
        )

    # Check version
    try:
        result = subprocess.run(
            [exe, "--version"], capture_output=True, text=True, timeout=5
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"'container --version' failed (exit {result.returncode}). "
                "Check your Apple Containers installation."
            )
    except subprocess.TimeoutExpired:
        raise RuntimeError("'container --version' timed out.")

    # Check system status
    running, _detail = container_system_status(exe)
    if not running:
        raise RuntimeError(
            "Apple Container system is not running. "
            "Run manually: container system start"
        )

    return exe


def query_system_resources() -> dict:
    """Query the host system for available CPU and memory.

    Results are cached after the first call.
    Returns dict with 'total_cpus' and 'total_memory_mb' keys.
    """
    global _system_resources
    if _system_resources is not None:
        return _system_resources

    info = {"total_cpus": os.cpu_count() or 4, "total_memory_mb": 8192}
    try:
        result = subprocess.run(
            ["sysctl", "-n", "hw.memsize"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            info["total_memory_mb"] = int(result.stdout.strip()) // (1024 * 1024)
    except Exception:
        pass

    _system_resources = info
    return info


def suggest_resources(total_cpus: int, total_memory_mb: int) -> dict:
    """Suggest container resource allocation based on system specs.

    Reserves roughly half the CPUs and a quarter of RAM for the host
    (LM Studio / Ollama needs significant resources for model inference).
    """
    container_cpus = max(2, total_cpus // 2)
    container_memory_mb = max(4096, total_memory_mb // 4)
    return {
        "cpus": container_cpus,
        "memory_mb": container_memory_mb,
    }


# Sensitive host paths that should never be volume-mounted into a container.
_SENSITIVE_MOUNT_SOURCES = {
    "/.ssh", "/ssh", ".ssh",
    "/.gnupg", "/gnupg", ".gnupg",
    "/.aws", "/aws", ".aws",
    "/.config/gcloud", "/gcloud",
    "/.azure", "/azure",
    "/.kube", "/kube",
}


def _warn_sensitive_volumes(volumes: list[str]) -> None:
    """Log warnings for volume mounts that expose sensitive host directories."""
    for vol in volumes:
        src = vol.split(":")[0] if ":" in vol else vol
        src_lower = src.lower()
        for pattern in _SENSITIVE_MOUNT_SOURCES:
            if src_lower.endswith(pattern) or f"{pattern}/" in src_lower:
                logger.warning(
                    "Volume mount '%s' exposes a sensitive host directory. "
                    "This may leak credentials into the container.",
                    vol,
                )
                break


def _bind_mount_args(source: str, target: str, *, readonly: bool) -> list[str]:
    """Build one structured Apple Container bind-mount argument."""
    unsafe = {",", "\x00", "\r", "\n"}
    if any(character in source for character in unsafe):
        raise ValueError(
            f"Apple Container mount source contains an unsafe character: {source!r}"
        )
    source_path = str(Path(source).expanduser().resolve())
    if any(character in source_path for character in unsafe):
        raise ValueError(
            f"Apple Container resolved mount source contains an unsafe character: {source_path!r}"
        )
    if not os.path.isabs(target):
        raise ValueError(f"Apple Container mount target must be absolute: {target!r}")
    if any(character in target for character in unsafe):
        raise ValueError(
            f"Apple Container mount target contains an unsafe character: {target!r}"
        )
    spec = f"type=bind,source={source_path},target={target}"
    if readonly:
        spec += ",readonly"
    return ["--mount", spec]


def _parse_user_mount(value: str) -> tuple[str, str, bool]:
    """Parse ``HOST:TARGET[:ro]`` without accepting backend-specific syntax."""
    if not isinstance(value, str):
        raise ValueError(f"Apple Container mount must be a string: {value!r}")
    if any(character in value for character in {"\x00", "\r", "\n"}):
        raise ValueError(
            f"Apple Container mount contains an unsafe character: {value!r}"
        )
    parts = value.strip().split(":")
    if len(parts) not in {2, 3} or any(not part for part in parts):
        raise ValueError(
            f"Invalid Apple Container mount {value!r}; expected HOST:TARGET[:ro]"
        )
    source, target = parts[:2]
    if not os.path.isabs(os.path.expanduser(source)) or not os.path.isabs(target):
        raise ValueError(
            f"Apple Container mount paths must be absolute: {value!r}"
        )
    if len(parts) == 3 and parts[2] != "ro":
        raise ValueError(
            f"Invalid Apple Container mount suffix in {value!r}; only ':ro' is supported"
        )
    return source, target, len(parts) == 3


class AppleContainerEnvironment(BaseEnvironment):
    """Apple Container execution with VM-level isolation.

    Each container runs inside its own lightweight Linux VM via Apple's
    Virtualization.framework. Commands are executed via ``container exec``,
    similar to Docker's ``docker exec``, with container lifecycle managed
    by this class.

    Security: the VM boundary provides kernel-level isolation. Additionally,
    the root filesystem is mounted read-only with writable tmpfs scratch
    directories, and credential/skills files are mounted read-only.
    """

    def __init__(
        self,
        image: str = "python:3.11-slim-bookworm",
        cwd: str = "/root",
        timeout: int = 180,
        cpu: int = 0,
        memory: int = 0,
        persistent_filesystem: bool = False,
        task_id: str = "default",
        volumes: list = None,
    ):
        parsed_volumes = [_parse_user_mount(volume) for volume in (volumes or [])]
        if cwd == "~":
            cwd = "/root"
        super().__init__(cwd=cwd, timeout=timeout)

        self._exe = _ensure_container_available()
        self._base_image = image
        self._persistent = persistent_filesystem
        self._task_id = task_id
        self._container_name: Optional[str] = None
        self._workspace_dir: Optional[str] = None
        self._credential_staging_dirs: list[Path] = []

        # Resolve resource limits (cached sysctl query)
        sys_info = query_system_resources()
        suggested = suggest_resources(sys_info["total_cpus"], sys_info["total_memory_mb"])
        self._cpus = cpu if cpu > 0 else suggested["cpus"]
        self._memory_mb = memory if memory > 0 else suggested["memory_mb"]

        try:
            # Build and start the container, then initialize its session snapshot.
            self._start_container(image, parsed_volumes)
            self.init_session()
        except Exception:
            self.cleanup()
            raise

    def _start_container(
        self, image: str, volumes: list[tuple[str, str, bool]]
    ) -> None:
        """Pull image if needed and start the container."""
        container_name = f"hermes-{uuid.uuid4().hex[:8]}"

        run_cmd = [
            self._exe, "run",
            "--name", container_name,
            "--detach",
            "--cpus", f"{self._cpus:g}",
            "--memory", f"{self._memory_mb}M",
            # Apple Container accepts a mount path only (no Docker-style
            # ``:rw,size=...`` suffix). These writable tmpfs mounts sit on top
            # of the read-only root filesystem.
            "--read-only",
            "--tmpfs", "/tmp",
            "--tmpfs", "/var/tmp",
            "--tmpfs", "/run",
        ]

        # Persistent workspace via bind mount, or ephemeral tmpfs
        if self._persistent:
            sandbox = get_sandbox_dir() / "apple_container" / self._task_id
            self._workspace_dir = str(sandbox / "workspace")
            os.makedirs(self._workspace_dir, exist_ok=True)
            root_dir = str(sandbox / "root")
            os.makedirs(root_dir, exist_ok=True)
            run_cmd.extend(_bind_mount_args(self._workspace_dir, "/workspace", readonly=False))
            run_cmd.extend(_bind_mount_args(root_dir, "/root", readonly=False))
        else:
            run_cmd.extend([
                "--tmpfs", "/workspace",
                "--tmpfs", "/root",
                "--tmpfs", "/home",
            ])

        # Mount credential files, skills, and cache directories read-only
        try:
            from tools.credential_files import (
                get_credential_file_mounts,
                get_skills_directory_mount,
                get_cache_directory_mounts,
            )

            credential_mounts = get_credential_file_mounts()
            skills_mounts = get_skills_directory_mount()
            cache_mounts = get_cache_directory_mounts()
            nested_mount_targets = [
                mount["container_path"] for mount in skills_mounts + cache_mounts
            ] + [target for _source, target, _readonly in volumes]
            for source_dir, target_dir in self._stage_credential_mounts(
                credential_mounts, nested_mount_targets
            ):
                run_cmd.extend(
                    _bind_mount_args(str(source_dir), target_dir, readonly=True)
                )
                logger.debug(
                    "Apple Container: mounting staged credentials -> %s",
                    target_dir,
                )

            for skills_mount in skills_mounts:
                run_cmd.extend(_bind_mount_args(
                    skills_mount["host_path"], skills_mount["container_path"], readonly=True
                ))
                logger.debug(
                    "Apple Container: mounting skills dir %s -> %s",
                    skills_mount["host_path"],
                    skills_mount["container_path"],
                )

            for cache_mount in cache_mounts:
                run_cmd.extend(_bind_mount_args(
                    cache_mount["host_path"], cache_mount["container_path"], readonly=True
                ))
                logger.debug(
                    "Apple Container: mounting cache dir %s -> %s",
                    cache_mount["host_path"],
                    cache_mount["container_path"],
                )
        except (OSError, ValueError):
            raise
        except Exception as e:
            logger.debug("Apple Container: could not load credential file mounts: %s", e)

        # User-supplied volume mounts
        _warn_sensitive_volumes([f"{source}:{target}" for source, target, _ in volumes])
        for source, target, readonly in volumes:
            run_cmd.extend(_bind_mount_args(source, target, readonly=readonly))

        run_cmd.append(image)
        # Keep the container alive with a long sleep
        run_cmd.extend(["sleep", "infinity"])

        logger.debug("Starting Apple Container: %s", " ".join(run_cmd))
        self._container_name = container_name
        try:
            result = subprocess.run(
                run_cmd,
                capture_output=True,
                text=True,
                timeout=300,  # image pull can take a while
            )
            if result.returncode != 0:
                stderr = (result.stderr or "").strip()
                self._force_delete_candidate()
                raise RuntimeError(
                    f"Failed to start Apple Container (exit {result.returncode}): {stderr}"
                )
        except subprocess.TimeoutExpired:
            self._force_delete_candidate()
            raise RuntimeError(
                "Apple Container startup timed out. The image may be too large "
                "or the container system may not be running."
            )
        except RuntimeError:
            raise
        except Exception as exc:
            self._force_delete_candidate()
            raise RuntimeError(f"Apple Container startup failed: {exc}") from exc

        logger.info(
            "Started Apple Container '%s' (%d CPUs, %d MB RAM)",
            container_name, self._cpus, self._memory_mb,
        )

    def _stage_credential_mounts(
        self,
        mount_entries: list[dict[str, str]],
        nested_mount_targets: list[str],
    ) -> list[tuple[Path, str]]:
        """Copy credentials into directory mounts supported by Apple Container.

        Apple Container 1.2 rejects bind mounts whose source is a regular file.
        Grouping copies by target parent preserves least privilege: each mounted
        directory contains only explicitly registered credentials, never their
        host-side siblings. Placeholder directories allow narrower skills,
        cache, user, or credential mounts below a read-only staged parent.
        """
        if not mount_entries:
            return []

        unsafe = {",", "\x00", "\r", "\n"}
        grouped: dict[str, list[tuple[Path, str]]] = {}
        for entry in mount_entries:
            source = entry.get("host_path", "")
            target = entry.get("container_path", "")
            if any(character in source for character in unsafe):
                raise ValueError(
                    "Apple Container mount source contains an unsafe character: "
                    f"{source!r}"
                )
            if (
                not target.startswith("/")
                or posixpath.normpath(target) != target
                or any(character in target for character in unsafe)
            ):
                raise ValueError(
                    "Apple Container mount target is unsafe or not absolute: "
                    f"{target!r}"
                )
            source_path = Path(source).expanduser().resolve()
            if not source_path.is_file():
                raise OSError(f"Credential mount source is not a file: {source_path}")
            target_parent = posixpath.dirname(target)
            target_name = posixpath.basename(target)
            if not target_parent or target_name in {"", ".", ".."}:
                raise ValueError(f"Invalid credential mount target: {target!r}")
            grouped.setdefault(target_parent, []).append((source_path, target_name))

        staging_parent = (
            get_sandbox_dir() / "apple_container" / self._task_id / "credential-mounts"
        )
        staging_root = staging_parent / uuid.uuid4().hex
        staging_root.mkdir(parents=True, mode=0o700)
        self._credential_staging_dirs.append(staging_root)

        result: list[tuple[Path, str]] = []
        try:
            mount_targets = [*nested_mount_targets, *grouped]
            for target_parent, files in sorted(grouped.items()):
                group_dir = staging_root / str(len(result))
                group_dir.mkdir(mode=0o700)
                for source_path, target_name in files:
                    staged_file = group_dir / target_name
                    shutil.copyfile(source_path, staged_file)
                    staged_file.chmod(0o400)

                prefix = target_parent.rstrip("/") + "/"
                for mount_target in mount_targets:
                    normalized_target = posixpath.normpath(mount_target)
                    if not normalized_target.startswith(prefix):
                        continue
                    relative_target = normalized_target[len(prefix):]
                    placeholder = group_dir.joinpath(*relative_target.split("/"))
                    if placeholder.exists() and not placeholder.is_dir():
                        raise ValueError(
                            "Credential path conflicts with nested mount target: "
                            f"{mount_target!r}"
                        )
                    placeholder.mkdir(parents=True, exist_ok=True)

                result.append((group_dir, target_parent))
        except Exception:
            self._cleanup_credential_staging()
            raise
        return result

    def _cleanup_credential_staging(self) -> None:
        for staging_dir in self._credential_staging_dirs:
            shutil.rmtree(staging_dir, ignore_errors=True)
        self._credential_staging_dirs.clear()

    def _run_bash(
        self,
        cmd_string: str,
        *,
        login: bool = False,
        timeout: int = 120,
        stdin_data: str | None = None,
    ) -> subprocess.Popen:
        """Spawn a bash process inside the Apple Container."""
        assert self._container_name, "Container not started"

        cmd = [self._exe, "exec"]
        if stdin_data is not None:
            cmd.append("--interactive")
        cmd.append(self._container_name)

        if login:
            cmd.extend(["bash", "-l", "-c", cmd_string])
        else:
            cmd.extend(["bash", "-c", cmd_string])

        return _popen_bash(cmd, stdin_data)

    def cleanup(self):
        """Stop and remove the container, waiting for graceful shutdown."""
        if not self._container_name:
            self._cleanup_credential_staging()
            return

        name = self._container_name
        force_delete = False

        try:
            # Graceful stop with a timeout — waits for the process to exit
            stop_result = subprocess.run(
                [self._exe, "stop", name],
                capture_output=True, text=True, timeout=30,
            )
            if self._result_says_not_found(stop_result):
                self._container_name = None
                self._cleanup_credential_staging()
                return
            if stop_result.returncode != 0:
                force_delete = True
                logger.warning(
                    "Failed to stop Apple Container '%s' (exit %d); force killing",
                    name,
                    stop_result.returncode,
                )
        except subprocess.TimeoutExpired:
            force_delete = True
            logger.warning("Timed out stopping Apple Container '%s', force killing", name)
        except Exception as e:
            force_delete = True
            logger.warning("Failed to stop Apple Container '%s': %s", name, e)

        if force_delete:
            try:
                subprocess.run(
                    [self._exe, "kill", name],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
            except Exception as exc:
                logger.debug("Failed to kill Apple Container '%s': %s", name, exc)

        deleted = False
        try:
            delete_cmd = [self._exe, "delete"]
            if force_delete:
                delete_cmd.append("--force")
            delete_cmd.append(name)
            delete_result = subprocess.run(
                delete_cmd,
                capture_output=True, text=True, timeout=10,
            )
            deleted = delete_result.returncode == 0 or self._result_says_not_found(
                delete_result
            )
            if not deleted and not force_delete:
                deleted = self._force_delete_candidate()
        except Exception as e:
            logger.debug("Failed to remove Apple Container '%s': %s", name, e)
            if not force_delete:
                deleted = self._force_delete_candidate()

        if deleted:
            self._container_name = None
            logger.info("Removed Apple Container '%s'", name)

        self._cleanup_credential_staging()

        # Clean up workspace if non-persistent
        if not self._persistent and self._workspace_dir:
            import shutil as _shutil
            _shutil.rmtree(self._workspace_dir, ignore_errors=True)

    @staticmethod
    def _result_says_not_found(result: subprocess.CompletedProcess) -> bool:
        detail = f"{result.stdout or ''}\n{result.stderr or ''}".lower()
        return any(
            marker in detail
            for marker in ("not found", "no such container", "does not exist")
        )

    def _force_delete_candidate(self) -> bool:
        """Force-delete the retained candidate and clear it only when confirmed."""
        name = self._container_name
        if not name:
            return True
        try:
            result = subprocess.run(
                [self._exe, "delete", "--force", name],
                capture_output=True,
                text=True,
                timeout=10,
            )
        except Exception as exc:
            logger.debug("Failed to force-delete Apple Container '%s': %s", name, exc)
            return False
        deleted = result.returncode == 0 or self._result_says_not_found(result)
        if deleted:
            self._container_name = None
        else:
            logger.warning(
                "Failed to force-delete Apple Container '%s' (exit %d)",
                name,
                result.returncode,
            )
        return deleted
