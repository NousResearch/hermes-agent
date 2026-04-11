"""Android Linux execution environment backed by an app-private Termux-style prefix."""

from __future__ import annotations

import os
import signal
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
    """Run commands inside an Android app-private Linux/CLI prefix.

    The prefix is prepared by the Android app at runtime and contains a relocated
    Termux-style command suite. Commands run directly on Android through the
    extracted bash binary, with PATH/LD_LIBRARY_PATH pointed at the prefix.
    """

    def __init__(self, cwd: str = "", timeout: int = 60, env: dict | None = None):
        self.prefix_path = os.environ.get("HERMES_ANDROID_LINUX_PREFIX", "").strip()
        self.bash_path = os.environ.get("HERMES_ANDROID_LINUX_BASH", "").strip()
        self.bin_path = os.environ.get("HERMES_ANDROID_LINUX_BIN", "").strip()
        self.lib_path = os.environ.get("HERMES_ANDROID_LINUX_LIB", "").strip()
        self.home_path = os.environ.get("HERMES_ANDROID_LINUX_HOME", "").strip()
        self.tmp_path = os.environ.get("HERMES_ANDROID_LINUX_TMP", "").strip()

        if not self.prefix_path or not self.bash_path:
            raise ValueError("Android Linux environment is not configured")

        for required in (self.prefix_path, self.bin_path, self.home_path, self.tmp_path):
            if required:
                Path(required).mkdir(parents=True, exist_ok=True)

        super().__init__(cwd=cwd or self.home_path or os.getcwd(), timeout=timeout, env=env)
        self.init_session()

    def get_temp_dir(self) -> str:
        return self.tmp_path or self.home_path or self.prefix_path or os.getcwd()

    def _build_run_env(self) -> dict[str, str]:
        run_env = dict(os.environ)
        run_env.update(self.env)
        existing_ld = run_env.get("LD_LIBRARY_PATH", "")
        if self.lib_path:
            run_env["LD_LIBRARY_PATH"] = (
                f"{self.lib_path}:{existing_ld}" if existing_ld else self.lib_path
            )
        prefix_path = self.bin_path or self.prefix_path
        existing_path = run_env.get("PATH", "")
        run_env["PATH"] = f"{prefix_path}:{existing_path}" if existing_path else prefix_path
        run_env["PREFIX"] = self.prefix_path
        run_env["TERMUX_PREFIX"] = self.prefix_path
        run_env["HOME"] = self.home_path or self.prefix_path
        run_env["TMPDIR"] = self.tmp_path or self.get_temp_dir()
        run_env.setdefault("TERM", "xterm-256color")
        run_env.setdefault("LANG", "C.UTF-8")
        terminfo_dir = Path(self.prefix_path) / "share" / "terminfo"
        if terminfo_dir.is_dir():
            run_env.setdefault("TERMINFO", str(terminfo_dir))
        git_exec_path = Path(self.prefix_path) / "libexec" / "git-core"
        if git_exec_path.is_dir():
            run_env.setdefault("GIT_EXEC_PATH", str(git_exec_path))
        return run_env

    def _run_bash(
        self,
        cmd_string: str,
        *,
        login: bool = False,
        timeout: int = 120,
        stdin_data: str | None = None,
    ) -> subprocess.Popen:
        del timeout  # spawn-per-call; enforced by BaseEnvironment
        args = [self.bash_path, "-l", "-c", cmd_string] if login else [self.bash_path, "-c", cmd_string]
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
