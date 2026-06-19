"""PowerShell execution environment for Windows — spawn-per-call with session snapshot.

Mirrors LocalEnvironment but uses pwsh (PowerShell 7+) instead of Git Bash.
Activated via ``terminal.shell: powershell`` in config.yaml.
"""

import logging
import os
import platform
import subprocess
import tempfile
import time
from pathlib import Path

from tools.environments.base import BaseEnvironment, _pipe_stdin
from tools.environments.local import _make_run_env, _sanitize_subprocess_env
from hermes_cli._subprocess_compat import windows_hide_flags

_IS_WINDOWS = platform.system() == "Windows"

logger = logging.getLogger(__name__)


def _find_powershell() -> str:
    """Find PowerShell 7+ (pwsh) or fall back to Windows PowerShell 5.1."""
    import shutil

    pwsh = shutil.which("pwsh")
    if pwsh:
        return pwsh

    for candidate in (
        os.path.join(os.environ.get("ProgramFiles", r"C:\Program Files"), "PowerShell", "7", "pwsh.exe"),
        os.path.join(os.environ.get("ProgramFiles", r"C:\Program Files"), "PowerShell", "7-preview", "pwsh.exe"),
    ):
        if candidate and os.path.isfile(candidate):
            return candidate

    powershell = shutil.which("powershell")
    if powershell:
        return powershell

    for candidate in (
        os.path.join(os.environ.get("SystemRoot", r"C:\Windows"), "System32", "WindowsPowerShell", "v1.0", "powershell.exe"),
    ):
        if candidate and os.path.isfile(candidate):
            return candidate

    raise RuntimeError(
        "PowerShell not found. Install PowerShell 7+ from: "
        "https://github.com/PowerShell/PowerShell/releases"
    )


class PowerShellEnvironment(BaseEnvironment):
    """Run commands via PowerShell on Windows.

    Spawn-per-call: every execute() spawns a fresh pwsh process.
    Session snapshot preserves env vars across calls via a .ps1 file.
    CWD persists via file-based read after each command.
    """

    _stdin_mode = "pipe"
    shell_type = "powershell"

    def __init__(self, cwd: str = "", timeout: int = 60, env: dict = None):
        if cwd:
            cwd = os.path.expanduser(cwd)
        super().__init__(cwd=cwd or os.getcwd(), timeout=timeout, env=env)
        self.init_session()

    def get_temp_dir(self) -> str:
        if _IS_WINDOWS:
            try:
                from hermes_constants import get_hermes_home
                cache_dir = get_hermes_home() / "cache" / "terminal"
            except Exception:
                cache_dir = Path(tempfile.gettempdir()) / "hermes_terminal"
            cache_dir.mkdir(parents=True, exist_ok=True)
            return str(cache_dir).replace("\\", "/")
        return "/tmp"

    def _run_bash(self, cmd_string: str, *, login: bool = False,
                  timeout: int = 120,
                  stdin_data: str | None = None) -> subprocess.Popen:
        """Spawn a PowerShell process to run *cmd_string*.

        Despite the name (kept for BaseEnvironment compatibility), this
        runs pwsh, not bash.
        """
        pwsh = _find_powershell()
        args = [pwsh, "-NoProfile", "-NoLogo", "-Command", cmd_string]
        run_env = _make_run_env(self.env)

        safe_cwd = self.cwd
        if _IS_WINDOWS and not os.path.isdir(safe_cwd):
            parent = os.path.dirname(safe_cwd)
            while parent:
                if os.path.isdir(parent):
                    safe_cwd = parent
                    break
                next_parent = os.path.dirname(parent)
                if next_parent == parent:
                    break
                parent = next_parent
            else:
                safe_cwd = tempfile.gettempdir()

        _popen_kwargs = {"creationflags": windows_hide_flags()} if _IS_WINDOWS else {}

        proc = subprocess.Popen(
            args,
            text=True,
            env=run_env,
            encoding="utf-8",
            errors="replace",
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            stdin=subprocess.PIPE if stdin_data is not None else subprocess.DEVNULL,
            preexec_fn=None if _IS_WINDOWS else os.setsid,
            cwd=safe_cwd,
            **_popen_kwargs,
        )
        if not _IS_WINDOWS:
            try:
                proc._hermes_pgid = os.getpgid(proc.pid)
            except ProcessLookupError:
                pass

        if stdin_data is not None:
            _pipe_stdin(proc, stdin_data)

        return proc

    def init_session(self):
        """Capture PowerShell environment into a snapshot file."""
        snap_path = self._snapshot_path.replace("/", "\\")
        cwd_file = self._cwd_file.replace("/", "\\")
        cwd_escaped = self.cwd.replace("'", "''")
        marker = self._cwd_marker

        bootstrap = (
            f"$ErrorActionPreference = 'Continue'\n"
            f"Get-ChildItem Env: | ForEach-Object {{ "
            f"  \"$($_.Name)=$($_.Value)\" "
            f"}} | Set-Content -Path '{snap_path}' -Encoding UTF8\n"
            f"Set-Location -LiteralPath '{cwd_escaped}' -ErrorAction SilentlyContinue\n"
            f"(Get-Location).Path | Set-Content -Path '{cwd_file}' -Encoding UTF8\n"
            f"Write-Output \"`n{marker}$((Get-Location).Path){marker}`n\"\n"
        )
        try:
            proc = self._run_bash(bootstrap, login=False, timeout=self._snapshot_timeout)
            result = self._wait_for_process(proc, timeout=self._snapshot_timeout)
            self._snapshot_ready = True
            self._update_cwd(result)
            logger.info(
                "PowerShell session snapshot created (session=%s, cwd=%s)",
                self._session_id,
                self.cwd,
            )
        except Exception as exc:
            logger.warning(
                "PowerShell init_session failed (session=%s): %s — "
                "falling back to per-command execution",
                self._session_id,
                exc,
            )
            self._snapshot_ready = False

    def _wrap_command(self, command: str, cwd: str) -> str:
        """Build a PowerShell script that sources snapshot, cd's, runs command,
        re-dumps env vars, and emits CWD markers."""
        snap_path = self._snapshot_path.replace("/", "\\")
        cwd_file = self._cwd_file.replace("/", "\\")
        cwd_escaped = cwd.replace("'", "''")
        marker = self._cwd_marker
        command_escaped = command.replace("'", "''")

        parts = [
            f"$ErrorActionPreference = 'Continue'",
        ]

        if self._snapshot_ready:
            parts.append(
                f"if (Test-Path -LiteralPath '{snap_path}') {{ "
                f"  Get-Content -LiteralPath '{snap_path}' | ForEach-Object {{ "
                f"    $idx = $_.IndexOf('='); "
                f"    if ($idx -gt 0) {{ "
                f"      [Environment]::SetEnvironmentVariable("
                f"        $_.Substring(0, $idx), $_.Substring($idx + 1), 'Process') "
                f"    }} "
                f"  }} "
                f"}}"
            )

        parts.append(f"Set-Location -LiteralPath '{cwd_escaped}' -ErrorAction SilentlyContinue")

        parts.append(
            f"try {{ & {{ {command_escaped} }}; $__hermes_ec = $LASTEXITCODE }} "
            f"catch {{ Write-Error $_.Exception.Message; $__hermes_ec = 1 }}; "
            f"if ($__hermes_ec -eq $null) {{ $__hermes_ec = 0 }}"
        )

        if self._snapshot_ready:
            parts.append(
                f"Get-ChildItem Env: | ForEach-Object {{ "
                f"  \"$($_.Name)=$($_.Value)\" "
                f"}} | Set-Content -Path '{snap_path}' -Encoding UTF8 -ErrorAction SilentlyContinue"
            )

        parts.append(
            f"(Get-Location).Path | Set-Content -Path '{cwd_file}' -Encoding UTF8 -ErrorAction SilentlyContinue"
        )
        parts.append(
            f"Write-Output \"`n{marker}$((Get-Location).Path){marker}`n\""
        )
        parts.append(f"exit $__hermes_ec")

        return "\n".join(parts)

    def _update_cwd(self, result: dict):
        """Read CWD from temp file (PowerShell writes native Windows paths)."""
        try:
            with open(self._cwd_file, encoding="utf-8") as f:
                cwd_path = f.read().strip()
            if cwd_path and os.path.isdir(cwd_path):
                self.cwd = cwd_path
        except (OSError, FileNotFoundError):
            pass

        self._extract_cwd_from_output(result)

    def _kill_process(self, proc):
        """Kill the process (Windows: terminate, POSIX: process group)."""
        try:
            if _IS_WINDOWS:
                proc.terminate()
            else:
                try:
                    pgid = os.getpgid(proc.pid)
                except ProcessLookupError:
                    pgid = getattr(proc, "_hermes_pgid", None)
                    if pgid is None:
                        raise
                import signal as _signal
                try:
                    os.killpg(pgid, _signal.SIGTERM)
                except ProcessLookupError:
                    return
                time.sleep(1.0)
                try:
                    os.killpg(pgid, _signal.SIGKILL)
                except ProcessLookupError:
                    return
        except (ProcessLookupError, PermissionError, OSError):
            try:
                proc.kill()
            except Exception:
                pass

    def cleanup(self):
        """Clean up temp files."""
        for f in (self._snapshot_path, self._cwd_file):
            try:
                os.unlink(f)
            except OSError:
                pass
