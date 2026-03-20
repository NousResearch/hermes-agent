"""Windows sandbox backend powered by a Hermes-owned native wrapper."""

from __future__ import annotations

import json
import logging
import os
import platform
import shutil
import subprocess
from pathlib import Path
from typing import Any

from hermes_cli.config import (
    ensure_hermes_home,
    get_hermes_bin_dir,
    get_windows_sandbox_codex_home,
)

from tools.environments.base import BaseEnvironment

logger = logging.getLogger(__name__)

_IS_WINDOWS = platform.system() == "Windows"
_WRAPPER_BASENAME = "hermes-windows-sandbox-wrapper"
_SETUP_HELPER_BASENAME = "codex-windows-sandbox-setup"
_STDERR_SEPARATOR = "\n\n[stderr]\n"
_SUPPORTED_WINDOWS_SANDBOX_ARCHITECTURES = {"amd64", "x86_64"}


def get_windows_sandbox_host_architecture() -> str:
    return platform.machine().strip().lower()


def get_windows_sandbox_unsupported_host_reason() -> str | None:
    if platform.system() != "Windows":
        return "windows-sandbox backend is only supported on Windows hosts"

    architecture = get_windows_sandbox_host_architecture() or "unknown"
    if architecture not in _SUPPORTED_WINDOWS_SANDBOX_ARCHITECTURES:
        return (
            "windows-sandbox backend is only supported on x64 Windows hosts "
            f"(got {architecture})"
        )

    return None


def _wrapper_filename() -> str:
    return f"{_WRAPPER_BASENAME}.exe" if _IS_WINDOWS else _WRAPPER_BASENAME


def _setup_helper_filename() -> str:
    return f"{_SETUP_HELPER_BASENAME}.exe" if _IS_WINDOWS else _SETUP_HELPER_BASENAME



def _default_wrapper_candidates(bin_dir: str | None = None) -> list[Path]:
    filename = _wrapper_filename()
    candidates: list[Path] = []
    if bin_dir:
        candidates.append(Path(bin_dir) / filename)

    hermes_bin = get_hermes_bin_dir() / filename
    candidates.append(hermes_bin)
    return candidates


def find_wrapper_executable(bin_dir: str | None = None) -> Path | None:
    """Locate the Hermes-owned wrapper executable."""
    for candidate in _default_wrapper_candidates(bin_dir):
        if candidate.is_file():
            return candidate

    return None


def _default_setup_helper_candidates(
    bin_dir: str | None = None,
    *,
    wrapper_path: Path | None = None,
) -> list[Path]:
    filename = _setup_helper_filename()
    candidates: list[Path] = []
    if bin_dir:
        candidates.append(Path(bin_dir) / filename)

    if wrapper_path is None:
        wrapper_path = find_wrapper_executable(bin_dir)
    if wrapper_path is not None:
        candidates.append(wrapper_path.parent / filename)

    hermes_bin = get_hermes_bin_dir() / filename
    candidates.append(hermes_bin)

    unique: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        normalized = os.path.normcase(str(candidate))
        if normalized not in seen:
            seen.add(normalized)
            unique.append(candidate)
    return unique


def find_setup_helper_executable(
    bin_dir: str | None = None,
    *,
    wrapper_path: Path | None = None,
) -> Path | None:
    """Locate the upstream setup helper executable used by the wrapper."""
    for candidate in _default_setup_helper_candidates(bin_dir, wrapper_path=wrapper_path):
        if candidate.is_file():
            return candidate

    return None


def provision_windows_sandbox_binaries(
    source_bin_dir: str | None = None,
    *,
    target_bin_dir: str | None = None,
) -> dict[str, Path]:
    """Copy the wrapper and setup helper into the authoritative Hermes bin dir."""
    ensure_hermes_home()

    target_dir = Path(target_bin_dir) if target_bin_dir else get_hermes_bin_dir()
    target_dir.mkdir(parents=True, exist_ok=True)

    wrapper_source = find_wrapper_executable(source_bin_dir)
    if wrapper_source is None:
        raise RuntimeError(
            "windows-sandbox wrapper executable not found. "
            f"Install hermes-windows-sandbox-wrapper.exe under {get_hermes_bin_dir()} "
            "or set TERMINAL_WINDOWS_SANDBOX_BIN_DIR to a directory that contains it."
        )

    helper_source = find_setup_helper_executable(
        source_bin_dir,
        wrapper_path=wrapper_source,
    )
    if helper_source is None:
        raise RuntimeError(
            "codex-windows-sandbox-setup.exe not found. "
            f"Install it under {get_hermes_bin_dir()} or make it available next to the wrapper before provisioning."
        )

    wrapper_target = target_dir / _wrapper_filename()
    helper_target = target_dir / _setup_helper_filename()

    if wrapper_source.resolve() != wrapper_target.resolve():
        shutil.copy2(wrapper_source, wrapper_target)
    if helper_source.resolve() != helper_target.resolve():
        shutil.copy2(helper_source, helper_target)

    return {
        "wrapper": wrapper_target,
        "setup_helper": helper_target,
    }


def _coalesce_output(stdout: str, stderr: str, error: str | None = None) -> str:
    pieces: list[str] = []
    if stdout:
        pieces.append(stdout)
    if stderr:
        pieces.append(stderr if not pieces else f"{_STDERR_SEPARATOR}{stderr}")
    if error and error not in stdout and error not in stderr:
        pieces.append(error if not pieces else f"{_STDERR_SEPARATOR}{error}")
    return "".join(pieces).strip()


def build_windows_sandbox_context(
    *,
    cwd: str | None = None,
    mode: str = "workspace-write",
    network_enabled: bool = False,
    writable_roots: list[str] | None = None,
    codex_home: str | None = None,
) -> dict[str, Any]:
    """Build the shared request context used by wrapper status/setup calls."""
    ensure_hermes_home()
    return {
        "cwd": cwd or os.getcwd(),
        "mode": mode,
        "network_enabled": network_enabled,
        "writable_roots": writable_roots or [],
        "codex_home": codex_home or str(get_windows_sandbox_codex_home()),
    }


def _run_wrapper_subcommand(
    wrapper_path: Path,
    subcommand: str,
    request: dict[str, Any],
    *,
    timeout: int,
) -> dict[str, Any]:
    host_timeout = max(timeout + 15, timeout)

    try:
        result = subprocess.run(
            [str(wrapper_path), subcommand],
            input=json.dumps(request),
            capture_output=True,
            text=True,
            timeout=host_timeout,
            encoding="utf-8",
            errors="replace",
        )
    except subprocess.TimeoutExpired:
        return {
            "stdout": "",
            "stderr": "",
            "exit_code": 124,
            "timed_out": True,
            "error": f"Command timed out after {timeout} seconds",
            "error_type": "timeout",
            "diagnostics": {},
        }
    except Exception as exc:
        return {
            "stdout": "",
            "stderr": "",
            "exit_code": -1,
            "timed_out": False,
            "error": f"windows-sandbox wrapper launch failed: {exc}",
            "error_type": "internal_error",
            "diagnostics": {},
        }

    stdout = result.stdout.strip()
    if not stdout:
        combined = (result.stderr or "").strip()
        if result.returncode == 0:
            combined = combined or "windows-sandbox wrapper returned no JSON output"
        else:
            combined = combined or f"windows-sandbox wrapper exited with status {result.returncode}"
        return {
            "stdout": "",
            "stderr": combined,
            "exit_code": -1,
            "timed_out": False,
            "error": combined,
            "error_type": "internal_error",
            "diagnostics": {},
        }

    try:
        return json.loads(stdout)
    except json.JSONDecodeError:
        combined = stdout
        if result.stderr:
            combined = f"{combined}{_STDERR_SEPARATOR}{result.stderr.strip()}"
        return {
            "stdout": "",
            "stderr": combined,
            "exit_code": -1,
            "timed_out": False,
            "error": "windows-sandbox wrapper returned invalid JSON",
            "error_type": "internal_error",
            "diagnostics": {},
        }


def call_windows_sandbox_wrapper(
    subcommand: str,
    request: dict[str, Any],
    *,
    bin_dir: str | None = None,
    timeout: int = 30,
    wrapper_path: Path | None = None,
) -> dict[str, Any]:
    """Invoke a windows-sandbox wrapper subcommand and return parsed JSON."""
    wrapper = wrapper_path or find_wrapper_executable(bin_dir)
    if wrapper is None:
        return {
            "stdout": "",
            "stderr": "",
            "exit_code": -1,
            "timed_out": False,
            "error": (
                "windows-sandbox wrapper executable not found. "
                f"Install Hermes-owned Windows sandbox helpers under {get_hermes_bin_dir()} "
                "or set TERMINAL_WINDOWS_SANDBOX_BIN_DIR for a custom development location."
            ),
            "error_type": "invalid_config",
            "diagnostics": {},
        }

    return _run_wrapper_subcommand(wrapper, subcommand, request, timeout=timeout)


def get_windows_sandbox_status(
    *,
    bin_dir: str | None = None,
    cwd: str | None = None,
    mode: str = "workspace-write",
    network_enabled: bool = False,
    writable_roots: list[str] | None = None,
    codex_home: str | None = None,
    timeout: int = 15,
) -> dict[str, Any]:
    """Return the wrapper-reported setup status for windows-sandbox."""
    request = build_windows_sandbox_context(
        cwd=cwd,
        mode=mode,
        network_enabled=network_enabled,
        writable_roots=writable_roots,
        codex_home=codex_home,
    )
    return call_windows_sandbox_wrapper(
        "status",
        request,
        bin_dir=bin_dir,
        timeout=timeout,
    )


def run_windows_sandbox_setup(
    *,
    bin_dir: str | None = None,
    cwd: str | None = None,
    mode: str = "workspace-write",
    network_enabled: bool = False,
    writable_roots: list[str] | None = None,
    codex_home: str | None = None,
    timeout: int = 60,
) -> dict[str, Any]:
    """Run the wrapper setup path for windows-sandbox."""
    request = build_windows_sandbox_context(
        cwd=cwd,
        mode=mode,
        network_enabled=network_enabled,
        writable_roots=writable_roots,
        codex_home=codex_home,
    )
    return call_windows_sandbox_wrapper(
        "setup",
        request,
        bin_dir=bin_dir,
        timeout=timeout,
    )


class WindowsSandboxEnvironment(BaseEnvironment):
    """Hermes execution backend for Codex windows-sandbox-rs."""

    def __init__(
        self,
        cwd: str = "",
        timeout: int = 60,
        *,
        mode: str = "workspace-write",
        network_enabled: bool = False,
        writable_roots: list[str] | None = None,
        bin_dir: str = "",
        setup_mode: str = "explicit",
        codex_home: str | None = None,
        env: dict | None = None,
    ):
        super().__init__(cwd=cwd or os.getcwd(), timeout=timeout, env=env)
        self.mode = mode
        self.network_enabled = network_enabled
        self.writable_roots = writable_roots or []
        self.bin_dir = bin_dir
        self.setup_mode = setup_mode
        self.codex_home = codex_home or str(get_windows_sandbox_codex_home())
        self.wrapper_path = find_wrapper_executable(bin_dir)
        self.setup_helper_path = find_setup_helper_executable(
            bin_dir,
            wrapper_path=self.wrapper_path,
        )
        self.last_result: dict[str, Any] | None = None

        if self.setup_mode != "explicit":
            raise ValueError(
                "windows-sandbox only supports explicit setup in the first release "
                f"(got {self.setup_mode!r})"
            )
        if self.mode not in {"workspace-write", "read-only"}:
            raise ValueError(
                "windows-sandbox mode must be 'workspace-write' or 'read-only' "
                f"(got {self.mode!r})"
            )
        unsupported_reason = get_windows_sandbox_unsupported_host_reason()
        if unsupported_reason:
            raise RuntimeError(unsupported_reason)
        if self.wrapper_path is None:
            raise RuntimeError(
                "windows-sandbox wrapper executable not found. "
                f"Install Hermes-owned Windows sandbox helpers under {get_hermes_bin_dir()} "
                "or set TERMINAL_WINDOWS_SANDBOX_BIN_DIR for a custom development location."
            )
        if self.setup_helper_path is None:
            raise RuntimeError(
                "windows-sandbox setup helper executable not found. "
                f"Install Hermes-owned Windows sandbox helpers under {get_hermes_bin_dir()} "
                "or set TERMINAL_WINDOWS_SANDBOX_BIN_DIR for a custom development location."
            )

    def _call_wrapper(self, subcommand: str, request: dict[str, Any], *, timeout: int) -> dict[str, Any]:
        return _run_wrapper_subcommand(self.wrapper_path, subcommand, request, timeout=timeout)

    def execute(
        self,
        command: str,
        cwd: str = "",
        *,
        timeout: int | None = None,
        stdin_data: str | None = None,
    ) -> dict:
        effective_timeout = timeout or self.timeout
        work_dir = cwd or self.cwd or os.getcwd()
        prepared_command, sudo_stdin = self._prepare_command(command)

        if stdin_data is not None:
            return {
                "output": "windows-sandbox does not support stdin piping in the first release",
                "returncode": 1,
            }
        if sudo_stdin is not None:
            return {
                "output": "windows-sandbox does not support sudo-style stdin injection",
                "returncode": 1,
            }

        request = {
            "command": prepared_command,
            "cwd": work_dir,
            "timeout_secs": effective_timeout,
            "mode": self.mode,
            "network_enabled": self.network_enabled,
            "writable_roots": self.writable_roots,
            "codex_home": self.codex_home,
            "command_mode": "foreground",
            "stdin_data": None,
        }

        payload = self._call_wrapper("exec", request, timeout=effective_timeout)
        self.last_result = payload

        if payload.get("timed_out") or payload.get("error_type") == "timeout":
            return self._timeout_result(effective_timeout)

        output = _coalesce_output(
            str(payload.get("stdout", "") or ""),
            str(payload.get("stderr", "") or ""),
            payload.get("error"),
        )
        returncode = payload.get("exit_code", 0)
        try:
            normalized_returncode = int(returncode)
        except (TypeError, ValueError):
            normalized_returncode = 1

        return {
            "output": output,
            "returncode": normalized_returncode,
        }

    def cleanup(self):
        """No persistent sandbox session state is kept for this backend."""
        self.last_result = None
