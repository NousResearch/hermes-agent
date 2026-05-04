"""Android shell execution environment for the native Android app.

The Kotlin bridge prefers APK-packaged executable native launchers for the
embedded Linux suite, then falls back to Android's platform shell if that probe
fails. This backend mirrors whichever mode the bridge selected.
"""

from __future__ import annotations

import os
import signal
import shlex
import subprocess
from pathlib import Path

try:
    from tools.environments.base import BaseEnvironment, _pipe_stdin
except ImportError:  # lightweight test stubs may only expose BaseEnvironment
    from tools.environments.base import BaseEnvironment

    def _pipe_stdin(proc, data: str) -> None:
        try:
            proc.stdin.write(data)
            proc.stdin.close()
        except Exception:
            pass


class AndroidLinuxEnvironment(BaseEnvironment):
    """Run commands through Android's native system shell.

    Files persist in the app-private Hermes workspace, while command execution
    uses /system/bin/sh and Android's built-in command set. This keeps the
    native app independent of Termux and avoids Android's noexec policy for
    writable app storage.
    """

    def __init__(self, cwd: str = "", timeout: int = 60, env: dict | None = None):
        self.prefix_path = os.environ.get("HERMES_ANDROID_LINUX_PREFIX", "").strip()
        self.shell_path = (
            os.environ.get("HERMES_ANDROID_SHELL", "").strip()
            or os.environ.get("HERMES_ANDROID_LINUX_BASH", "").strip()
            or "/system/bin/sh"
        )
        self.bin_path = os.environ.get("HERMES_ANDROID_LINUX_BIN", "").strip()
        self.lib_path = os.environ.get("HERMES_ANDROID_LINUX_LIB", "").strip()
        self.native_library_dir = os.environ.get("HERMES_ANDROID_NATIVE_LIB", "").strip()
        self.home_path = os.environ.get("HERMES_ANDROID_LINUX_HOME", "").strip()
        self.tmp_path = os.environ.get("HERMES_ANDROID_LINUX_TMP", "").strip()
        self.execution_mode = os.environ.get("HERMES_ANDROID_EXECUTION_MODE", "android_system_shell").strip()

        if not self.shell_path:
            raise ValueError("Android shell environment is not configured")

        for required in (self.prefix_path, self.home_path, self.tmp_path):
            if required:
                Path(required).mkdir(parents=True, exist_ok=True)

        super().__init__(cwd=cwd or self.home_path or os.getcwd(), timeout=timeout, env=env)
        self.init_session()

    def get_temp_dir(self) -> str:
        return self.tmp_path or self.home_path or self.prefix_path or os.getcwd()

    def _build_run_env(self) -> dict[str, str]:
        run_env = dict(os.environ)
        run_env.update(self.env)

        system_path = "/system/bin:/system/xbin:/vendor/bin:/odm/bin"
        existing_path = run_env.get("PATH", "")
        path_parts = [system_path]
        if existing_path:
            path_parts.append(existing_path)
        if (
            self.bin_path
            and run_env.get("HERMES_ANDROID_ALLOW_PREFIX_BIN") == "1"
            and self.bin_path not in path_parts
        ):
            path_parts.append(self.bin_path)
        run_env["PATH"] = ":".join(path_parts)

        if self.prefix_path:
            run_env["PREFIX"] = self.prefix_path
        run_env["HOME"] = self.home_path or self.prefix_path
        run_env["TMPDIR"] = self.tmp_path or self.get_temp_dir()
        run_env["HERMES_ANDROID_SHELL"] = self.shell_path
        run_env["HERMES_ANDROID_EXECUTION_MODE"] = self.execution_mode or "android_system_shell"
        ld_parts = [
            self.native_library_dir,
            self.lib_path,
            run_env.get("LD_LIBRARY_PATH", ""),
        ]
        run_env["LD_LIBRARY_PATH"] = ":".join(item for item in ld_parts if item)
        run_env.setdefault("TERM", "xterm-256color")
        run_env.setdefault("LANG", "C.UTF-8")
        return run_env

    def init_session(self):
        """Use a lightweight native-shell session without bash snapshots."""
        self._snapshot_ready = False
        try:
            Path(self._cwd_file).parent.mkdir(parents=True, exist_ok=True)
        except OSError:
            pass

    def _wrap_command(self, command: str, cwd: str) -> str:
        escaped = command.replace("'", "'\\''")
        quoted_cwd = self._quote_cwd_for_cd(cwd)
        quoted_cwd_file = shlex.quote(self._cwd_file)
        return "\n".join(
            [
                f"cd {quoted_cwd} || exit 126",
                f"eval '{escaped}'",
                "__hermes_ec=$?",
                f"pwd -P > {quoted_cwd_file} 2>/dev/null || true",
                f"printf '\\n{self._cwd_marker}%s{self._cwd_marker}\\n' \"$(pwd -P)\"",
                "exit $__hermes_ec",
            ]
        )

    def _run_bash(
        self,
        cmd_string: str,
        *,
        login: bool = False,
        timeout: int = 120,
        stdin_data: str | None = None,
    ) -> subprocess.Popen:
        del timeout  # spawn-per-call; enforced by BaseEnvironment
        del login
        args = [self.shell_path, "-c", cmd_string]
        proc = subprocess.Popen(
            args,
            text=True,
            env=self._build_run_env(),
            encoding="utf-8",
            errors="replace",
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            stdin=subprocess.PIPE if stdin_data is not None else subprocess.DEVNULL,
            preexec_fn=os.setsid,
        )
        if stdin_data is not None:
            _pipe_stdin(proc, stdin_data)
        return proc

    def _kill_process(self, proc):
        try:
            pgid = os.getpgid(proc.pid)
            os.killpg(pgid, signal.SIGTERM)
            try:
                proc.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                os.killpg(pgid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            try:
                proc.kill()
            except Exception:
                pass

    def cleanup(self):
        for file_path in (self._snapshot_path, self._cwd_file):
            try:
                os.unlink(file_path)
            except OSError:
                pass
