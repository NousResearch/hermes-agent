"""Fail-closed isolated executor for generated skill tests.

Automatic lifecycle validation must not run generated code directly on the host.
On Linux, bubblewrap provides a read-only, networkless mount namespace exposing
only the skill package and Python runtime.  Unsupported platforms return no
executor instead of silently falling back to local subprocess execution.
"""

from __future__ import annotations

import os
import platform
import site
import shutil
import subprocess
import sys
import sysconfig
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from tools.skill_lifecycle_orchestrator import SkillTestRequest, TestExecutionResult

_MAX_OUTPUT_BYTES = 16_384


def _append_parent_dirs(argv: list[str], target: Path, created: set[Path]) -> None:
    """Create empty namespace parents needed by a later bind mount."""

    parents = list(target.parents)
    parents.reverse()
    for parent in parents:
        if parent == Path("/") or parent in created:
            continue
        argv.extend(("--dir", str(parent)))
        created.add(parent)


def _bounded_output(stdout: bytes, stderr: bytes) -> str:
    data = stdout + (b"\n" if stdout and stderr else b"") + stderr
    if len(data) > _MAX_OUTPUT_BYTES:
        head = _MAX_OUTPUT_BYTES // 2
        tail = _MAX_OUTPUT_BYTES - head
        omitted = len(data) - _MAX_OUTPUT_BYTES
        data = (
            data[:head]
            + f"\n... {omitted} bytes omitted ...\n".encode()
            + data[-tail:]
        )
    return data.decode("utf-8", errors="replace")


def _read_bounded_file(handle, limit: int) -> bytes:
    """Read head+tail from a subprocess output file without loading it all."""

    handle.flush()
    size = handle.seek(0, os.SEEK_END)
    if size <= limit:
        handle.seek(0)
        return handle.read()
    head_size = limit // 2
    tail_size = limit - head_size
    handle.seek(0)
    head = handle.read(head_size)
    handle.seek(-tail_size, os.SEEK_END)
    tail = handle.read(tail_size)
    omitted = size - limit
    return head + f"\n... {omitted} bytes omitted ...\n".encode() + tail


@dataclass(frozen=True)
class BubblewrapTestExecutor:
    """Callable bubblewrap-backed executor for one skill package."""

    bubblewrap: str
    prlimit: str
    systemd_run: str
    xdg_runtime_dir: str
    dbus_session_bus_address: str
    python_executable: str
    python_prefix: str
    python_base_prefix: str
    python_resolved_executable: str
    python_stdlib_paths: tuple[str, ...]
    python_site_paths: tuple[str, ...]
    runtime_library_paths: tuple[str, ...]

    @classmethod
    def discover(cls) -> Optional["BubblewrapTestExecutor"]:
        if platform.system() != "Linux":
            return None
        binary = shutil.which("bwrap") or shutil.which("bubblewrap")
        prlimit = shutil.which("prlimit")
        systemd_run = shutil.which("systemd-run")
        if not binary or not prlimit or not systemd_run:
            return None
        runtime_dir = os.environ.get("XDG_RUNTIME_DIR") or f"/run/user/{os.getuid()}"
        bus_address = os.environ.get("DBUS_SESSION_BUS_ADDRESS") or (
            f"unix:path={runtime_dir}/bus"
        )
        if not Path(runtime_dir).is_dir() or not Path(runtime_dir, "bus").exists():
            return None
        # Preserve the virtualenv entry path. Resolving its symlink would launch
        # the base interpreter without the venv's pytest/site-packages.
        executable = str(Path(sys.executable).absolute())
        executable_path = Path(executable)
        resolved_executable = executable_path.resolve(strict=True)
        stdlib_paths = {
            str(Path(path).resolve(strict=True))
            for key in ("stdlib", "platstdlib")
            if (path := sysconfig.get_path(key))
        }
        site_paths = {
            str(Path(path).resolve(strict=True))
            for path in site.getsitepackages()
            if Path(path).is_dir()
        }
        multiarch = sysconfig.get_config_var("MULTIARCH")
        runtime_library_paths = tuple(
            str(path)
            for path in (
                Path("/lib") / str(multiarch) if multiarch else None,
                Path("/lib64"),
            )
            if path is not None and path.is_dir()
        )
        if not stdlib_paths or not site_paths or not runtime_library_paths:
            return None
        return cls(
            bubblewrap=binary,
            prlimit=prlimit,
            systemd_run=systemd_run,
            xdg_runtime_dir=runtime_dir,
            dbus_session_bus_address=bus_address,
            python_executable=executable,
            python_prefix=str(Path(sys.prefix).resolve(strict=True)),
            python_base_prefix=str(Path(sys.base_prefix).resolve(strict=True)),
            python_resolved_executable=str(resolved_executable),
            python_stdlib_paths=tuple(sorted(stdlib_paths)),
            python_site_paths=tuple(sorted(site_paths)),
            runtime_library_paths=runtime_library_paths,
        )

    def _command(self, request: SkillTestRequest) -> list[str]:
        skill_dir = request.cwd.resolve(strict=True)
        executable = Path(self.python_executable).absolute()
        resolved_executable = executable.resolve(strict=True)
        requested = Path(request.argv[0]).resolve(strict=True)
        if requested != resolved_executable:
            raise ValueError("test request must use the executor's Python runtime")
        if tuple(request.argv[1:3]) != ("-m", "pytest"):
            raise ValueError("test executor only permits python -m pytest")
        if any(arg.startswith("/") or ".." in Path(arg).parts for arg in request.argv[3:]):
            raise ValueError("pytest arguments must remain package-relative")

        argv = [
            self.bubblewrap,
            "--die-with-parent",
            "--new-session",
            "--unshare-net",
            "--unshare-pid",
            "--unshare-ipc",
            "--unshare-uts",
            "--cap-drop",
            "ALL",
            "--clearenv",
        ]
        created: set[Path] = set()

        mounts: list[tuple[Path, Path]] = []
        mounted_targets: set[Path] = set()
        for candidate in (
            *(Path(path) for path in self.runtime_library_paths),
            Path(self.python_resolved_executable),
            *(Path(path) for path in self.python_stdlib_paths),
            *(Path(path) for path in self.python_site_paths),
        ):
            try:
                resolved = candidate.resolve(strict=True)
            except OSError:
                continue
            target = candidate
            if target not in mounted_targets:
                mounts.append((resolved, target))
                mounted_targets.add(target)

        for source, target in mounts:
            _append_parent_dirs(argv, target, created)
            argv.extend(("--ro-bind", str(source), str(target)))
            created.add(target)

        argv.extend(
            (
                "--proc",
                "/proc",
                "--dev",
                "/dev",
                "--size",
                "67108864",
                "--tmpfs",
                "/tmp",
                "--dir",
                "/tmp/home",
                "--ro-bind",
                str(skill_dir),
                "/skill",
                "--chdir",
                "/skill",
                "--setenv",
                "HOME",
                "/tmp/home",
                "--setenv",
                "TMPDIR",
                "/tmp",
                "--setenv",
                "PATH",
                "",
                "--setenv",
                "PYTHONDONTWRITEBYTECODE",
                "1",
                "--setenv",
                "PYTEST_DISABLE_PLUGIN_AUTOLOAD",
                "1",
                "--setenv",
                "PYTHONHOME",
                self.python_base_prefix,
                "--setenv",
                "PYTHONPATH",
                os.pathsep.join(self.python_site_paths),
                self.python_resolved_executable,
                *request.argv[1:],
            )
        )
        return argv

    def __call__(self, request: SkillTestRequest) -> TestExecutionResult:
        cpu = max(1, min(request.timeout, 300))
        command = [
            self.systemd_run,
            "--user",
            "--quiet",
            "--wait",
            "--pipe",
            "--collect",
            "--property=TasksMax=16",
            "--property=MemoryMax=2147483648",
            "--property=MemorySwapMax=0",
            f"--property=RuntimeMaxSec={cpu}s",
            "--",
            self.prlimit,
            f"--cpu={cpu}:{cpu + 1}",
            "--as=268435456:268435456",
            "--fsize=1048576:1048576",
            "--nofile=256:256",
            "--",
            *self._command(request),
        ]
        systemd_env = {
            "XDG_RUNTIME_DIR": self.xdg_runtime_dir,
            "DBUS_SESSION_BUS_ADDRESS": self.dbus_session_bus_address,
        }
        with tempfile.TemporaryFile() as stdout_file, tempfile.TemporaryFile() as stderr_file:
            try:
                completed = subprocess.run(
                    command,
                    stdin=subprocess.DEVNULL,
                    stdout=stdout_file,
                    stderr=stderr_file,
                    timeout=cpu + 5,
                    check=False,
                    env=systemd_env,
                )
            except subprocess.TimeoutExpired:
                stdout = _read_bounded_file(stdout_file, _MAX_OUTPUT_BYTES // 2)
                stderr = _read_bounded_file(stderr_file, _MAX_OUTPUT_BYTES // 2)
                stderr += b"\nSkill tests timed out."
                return TestExecutionResult(
                    exit_code=124,
                    output=_bounded_output(stdout, stderr),
                    isolation="bubblewrap",
                )
            stdout = _read_bounded_file(stdout_file, _MAX_OUTPUT_BYTES // 2)
            stderr = _read_bounded_file(stderr_file, _MAX_OUTPUT_BYTES // 2)
        output = _bounded_output(stdout, stderr)
        stripped_stderr = stderr.lstrip()
        infrastructure_error = stripped_stderr.startswith(
            (b"bwrap:", b"prlimit:", b"Failed to connect to bus")
        ) or b"Failed to start transient service unit" in stderr
        if completed.returncode != 0 and infrastructure_error:
            raise RuntimeError(f"bubblewrap sandbox failed: {output.strip()}")
        return TestExecutionResult(
            exit_code=completed.returncode,
            output=output,
            isolation="bubblewrap",
        )


__all__ = ["BubblewrapTestExecutor"]
