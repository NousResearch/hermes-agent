"""Persistent shell mixin: file-based IPC protocol for long-lived bash shells.

Provides the shared logic for maintaining a persistent bash shell across
execute() calls.  Backend-specific operations (spawning the shell, reading
temp files, killing child processes) are implemented by subclasses via
abstract methods.

The IPC protocol writes each command's stdout/stderr/exit-code/cwd to temp
files, then polls the status file for completion.  A daemon thread drains
the shell's stdout to prevent pipe deadlock and detect shell death.
"""

import glob as glob_mod
import logging
import os
import shlex
import subprocess
import threading
import time
import uuid
from abc import abstractmethod

from tools.interrupt import is_interrupted

logger = logging.getLogger(__name__)


class PersistentShellMixin:
    """Mixin that adds persistent shell capability to any BaseEnvironment.

    Subclasses MUST implement:
        _spawn_shell_process() -> subprocess.Popen
        _read_temp_files(*paths) -> list[str]
        _kill_shell_children()

    Subclasses MUST also provide ``_execute_oneshot()`` for the stdin_data
    fallback path (commands with piped stdin cannot use the persistent shell).
    """

    # -- State (initialized by _init_persistent_shell) ---------------------
    _shell_proc: subprocess.Popen | None = None
    _shell_alive: bool = False
    _shell_pid: int | None = None
    _session_id: str = ""

    # -- Abstract methods (backend-specific) -------------------------------

    @abstractmethod
    def _spawn_shell_process(self) -> subprocess.Popen:
        """Spawn a long-lived bash shell and return the Popen handle.

        Must use ``stdin=PIPE, stdout=PIPE, stderr=PIPE, text=True``.
        """
        ...

    @abstractmethod
    def _read_temp_files(self, *paths: str) -> list[str]:
        """Read temp files from the execution context.

        Returns contents in the same order as *paths*.  Falls back to
        empty strings on failure.
        """
        ...

    @abstractmethod
    def _kill_shell_children(self):
        """Kill the running command's processes but keep the shell alive."""
        ...

    # -- Overridable properties --------------------------------------------

    @property
    def _temp_prefix(self) -> str:
        """Base path for temp files.  Override per backend."""
        return f"/tmp/hermes-persistent-{self._session_id}"

    # -- Shared implementation ---------------------------------------------

    def _init_persistent_shell(self):
        """Call from ``__init__`` when ``persistent=True``."""
        self._shell_lock = threading.Lock()
        self._session_id = ""
        self._shell_proc = None
        self._shell_alive = False
        self._shell_pid = None
        self._start_persistent_shell()

    def _start_persistent_shell(self):
        """Spawn the shell, create temp files, capture PID."""
        self._session_id = uuid.uuid4().hex[:12]
        p = self._temp_prefix
        self._pshell_stdout = f"{p}-stdout"
        self._pshell_stderr = f"{p}-stderr"
        self._pshell_status = f"{p}-status"
        self._pshell_cwd = f"{p}-cwd"
        self._pshell_pid_file = f"{p}-pid"

        self._shell_proc = self._spawn_shell_process()
        self._shell_alive = True

        self._drain_thread = threading.Thread(
            target=self._drain_shell_output, daemon=True,
        )
        self._drain_thread.start()

        # Initialize temp files and capture shell PID
        init_script = (
            f"touch {self._pshell_stdout} {self._pshell_stderr} "
            f"{self._pshell_status} {self._pshell_cwd} {self._pshell_pid_file}\n"
            f"echo $$ > {self._pshell_pid_file}\n"
            f"pwd > {self._pshell_cwd}\n"
        )
        self._send_to_shell(init_script)

        # Poll for PID file
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline:
            pid_str = self._read_temp_files(self._pshell_pid_file)[0].strip()
            if pid_str.isdigit():
                self._shell_pid = int(pid_str)
                break
            time.sleep(0.05)
        else:
            logger.warning("Could not read persistent shell PID")
            self._shell_pid = None

        if self._shell_pid:
            logger.info(
                "Persistent shell started (session=%s, pid=%d)",
                self._session_id, self._shell_pid,
            )

        # Update cwd from what the shell reports
        reported_cwd = self._read_temp_files(self._pshell_cwd)[0].strip()
        if reported_cwd:
            self.cwd = reported_cwd

    def _drain_shell_output(self):
        """Drain stdout to prevent pipe deadlock; detect shell death."""
        try:
            for _ in self._shell_proc.stdout:
                pass  # Real output goes to temp files
        except Exception:
            pass
        self._shell_alive = False

    def _send_to_shell(self, text: str):
        """Write text to the persistent shell's stdin."""
        if not self._shell_alive or self._shell_proc is None:
            return
        try:
            self._shell_proc.stdin.write(text)
            self._shell_proc.stdin.flush()
        except (BrokenPipeError, OSError):
            self._shell_alive = False

    def _read_persistent_output(self) -> tuple[str, int, str]:
        """Read stdout, stderr, status, cwd.  Returns (output, exit_code, cwd)."""
        stdout, stderr, status_raw, cwd = self._read_temp_files(
            self._pshell_stdout, self._pshell_stderr,
            self._pshell_status, self._pshell_cwd,
        )
        output = self._merge_output(stdout, stderr)
        # Status format: "cmd_id:exit_code" — strip the ID prefix
        status = status_raw.strip()
        if ":" in status:
            status = status.split(":", 1)[1]
        try:
            exit_code = int(status.strip())
        except ValueError:
            exit_code = 1
        return output, exit_code, cwd.strip()

    def _execute_persistent(self, command: str, cwd: str, *,
                            timeout: int | None = None,
                            stdin_data: str | None = None) -> dict:
        """Execute a command in the persistent shell."""
        if not self._shell_alive:
            logger.info("Persistent shell died, restarting...")
            self._start_persistent_shell()

        exec_command, sudo_stdin = self._prepare_command(command)
        effective_timeout = timeout or self.timeout

        # Fall back to one-shot for commands needing piped stdin
        if stdin_data or sudo_stdin:
            return self._execute_oneshot(
                command, cwd, timeout=timeout, stdin_data=stdin_data,
            )

        with self._shell_lock:
            return self._execute_persistent_locked(
                exec_command, cwd, effective_timeout,
            )

    def _execute_persistent_locked(self, command: str, cwd: str,
                                   timeout: int) -> dict:
        """Inner persistent execution — caller must hold ``_shell_lock``."""
        work_dir = cwd or self.cwd

        # Each command gets a unique ID written into the status file so the
        # poll loop can distinguish the *current* command's result from a
        # stale value left over from the previous command.  This eliminates
        # the race where a fast local file read sees the old status before
        # the shell has processed the truncation.
        cmd_id = uuid.uuid4().hex[:8]

        # Truncate temp files
        truncate = (
            f": > {self._pshell_stdout}\n"
            f": > {self._pshell_stderr}\n"
            f": > {self._pshell_status}\n"
        )
        self._send_to_shell(truncate)

        # Escape command for eval
        escaped = command.replace("'", "'\\''")

        ipc_script = (
            f"cd {shlex.quote(work_dir)}\n"
            f"eval '{escaped}' < /dev/null > {self._pshell_stdout} 2> {self._pshell_stderr}\n"
            f"__EC=$?\n"
            f"pwd > {self._pshell_cwd}\n"
            f"echo {cmd_id}:$__EC > {self._pshell_status}\n"
        )
        self._send_to_shell(ipc_script)

        # Poll the status file for current command's ID
        deadline = time.monotonic() + timeout
        poll_interval = 0.15

        while True:
            if is_interrupted():
                self._kill_shell_children()
                output, _, _ = self._read_persistent_output()
                return {
                    "output": output + "\n[Command interrupted]",
                    "returncode": 130,
                }

            if time.monotonic() > deadline:
                self._kill_shell_children()
                output, _, _ = self._read_persistent_output()
                if output:
                    return {
                        "output": output + f"\n[Command timed out after {timeout}s]",
                        "returncode": 124,
                    }
                return self._timeout_result(timeout)

            if not self._shell_alive:
                return {
                    "output": "Persistent shell died during execution",
                    "returncode": 1,
                }

            status_content = self._read_temp_files(self._pshell_status)[0].strip()
            if status_content.startswith(cmd_id + ":"):
                break

            time.sleep(poll_interval)

        output, exit_code, new_cwd = self._read_persistent_output()
        if new_cwd:
            self.cwd = new_cwd
        return {"output": output, "returncode": exit_code}

    @staticmethod
    def _merge_output(stdout: str, stderr: str) -> str:
        """Combine stdout and stderr into a single output string."""
        parts = []
        if stdout.strip():
            parts.append(stdout.rstrip("\n"))
        if stderr.strip():
            parts.append(stderr.rstrip("\n"))
        return "\n".join(parts)

    def _cleanup_persistent_shell(self):
        """Clean up persistent shell resources.  Call from ``cleanup()``."""
        if self._shell_proc is None:
            return

        if self._session_id:
            self._cleanup_temp_files()

        try:
            self._shell_proc.stdin.close()
        except Exception:
            pass
        try:
            self._shell_proc.terminate()
            self._shell_proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            self._shell_proc.kill()

        self._shell_alive = False
        self._shell_proc = None

        if hasattr(self, "_drain_thread") and self._drain_thread.is_alive():
            self._drain_thread.join(timeout=1.0)

    def _cleanup_temp_files(self):
        """Remove local temp files.  Override for remote backends (SSH, Docker)."""
        for f in glob_mod.glob(f"{self._temp_prefix}-*"):
            try:
                os.remove(f)
            except OSError:
                pass
