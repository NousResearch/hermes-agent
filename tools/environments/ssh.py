"""SSH remote execution environment with ControlMaster connection persistence."""

import logging
import shlex
import subprocess
import tempfile
import threading
import time
import uuid
from pathlib import Path

from tools.environments.base import BaseEnvironment
from tools.interrupt import is_interrupted

logger = logging.getLogger(__name__)


class SSHEnvironment(BaseEnvironment):
    """Run commands on a remote machine over SSH.

    Uses SSH ControlMaster for connection persistence so subsequent
    commands are fast. Security benefit: the agent cannot modify its
    own code since execution happens on a separate machine.

    Foreground commands are interruptible: the local ssh process is killed
    and a remote kill is attempted over the ControlMaster socket.

    When ``persistent=True``, a single long-lived bash shell is kept alive
    over SSH and state (cwd, env vars, shell variables) persists across
    ``execute()`` calls.  Output capture uses file-based IPC on the remote
    host (stdout/stderr/exit-code written to temp files, polled via fast
    ControlMaster one-shot reads).
    """

    def __init__(self, host: str, user: str, cwd: str = "~",
                 timeout: int = 60, port: int = 22, key_path: str = "",
                 persistent: bool = False):
        super().__init__(cwd=cwd, timeout=timeout)
        self.host = host
        self.user = user
        self.port = port
        self.key_path = key_path
        self.persistent = persistent

        self.control_dir = Path(tempfile.gettempdir()) / "hermes-ssh"
        self.control_dir.mkdir(parents=True, exist_ok=True)
        self.control_socket = self.control_dir / f"{user}@{host}:{port}.sock"
        self._establish_connection()

        # Persistent shell state
        self._shell_proc: subprocess.Popen | None = None
        self._shell_lock = threading.Lock()
        self._shell_alive = False
        self._session_id: str = ""
        self._remote_stdout: str = ""
        self._remote_stderr: str = ""
        self._remote_status: str = ""
        self._remote_cwd: str = ""
        self._remote_pid: str = ""
        self._remote_shell_pid: int | None = None

        if self.persistent:
            self._start_persistent_shell()

    def _build_ssh_command(self, extra_args: list = None) -> list:
        cmd = ["ssh"]
        cmd.extend(["-o", f"ControlPath={self.control_socket}"])
        cmd.extend(["-o", "ControlMaster=auto"])
        cmd.extend(["-o", "ControlPersist=300"])
        cmd.extend(["-o", "BatchMode=yes"])
        cmd.extend(["-o", "StrictHostKeyChecking=accept-new"])
        cmd.extend(["-o", "ConnectTimeout=10"])
        if self.port != 22:
            cmd.extend(["-p", str(self.port)])
        if self.key_path:
            cmd.extend(["-i", self.key_path])
        if extra_args:
            cmd.extend(extra_args)
        cmd.append(f"{self.user}@{self.host}")
        return cmd

    def _establish_connection(self):
        cmd = self._build_ssh_command()
        cmd.append("echo 'SSH connection established'")
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
            if result.returncode != 0:
                error_msg = result.stderr.strip() or result.stdout.strip()
                raise RuntimeError(f"SSH connection failed: {error_msg}")
        except subprocess.TimeoutExpired:
            raise RuntimeError(f"SSH connection to {self.user}@{self.host} timed out")

    # ------------------------------------------------------------------
    # Persistent shell management
    # ------------------------------------------------------------------

    def _start_persistent_shell(self):
        """Spawn a long-lived bash shell over SSH."""
        self._session_id = uuid.uuid4().hex[:12]
        prefix = f"/tmp/hermes-ssh-{self._session_id}"
        self._remote_stdout = f"{prefix}-stdout"
        self._remote_stderr = f"{prefix}-stderr"
        self._remote_status = f"{prefix}-status"
        self._remote_cwd = f"{prefix}-cwd"
        self._remote_pid = f"{prefix}-pid"

        cmd = self._build_ssh_command()
        cmd.append("bash -l")

        self._shell_proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        self._shell_alive = True

        # Start daemon thread to drain stdout/stderr and detect shell death
        self._drain_thread = threading.Thread(
            target=self._drain_shell_output, daemon=True
        )
        self._drain_thread.start()

        # Initialize remote temp files and capture shell PID
        init_script = (
            f"touch {self._remote_stdout} {self._remote_stderr} "
            f"{self._remote_status} {self._remote_cwd} {self._remote_pid}\n"
            f"echo $$ > {self._remote_pid}\n"
            f"pwd > {self._remote_cwd}\n"
        )
        self._send_to_shell(init_script)

        # Give shell time to initialize and write PID file
        time.sleep(0.3)

        # Read the remote shell PID
        pid_str = self._read_remote_file(self._remote_pid).strip()
        if pid_str.isdigit():
            self._remote_shell_pid = int(pid_str)
            logger.info("Persistent shell started (session=%s, pid=%d)",
                        self._session_id, self._remote_shell_pid)
        else:
            logger.warning("Could not read persistent shell PID (got %r)", pid_str)
            self._remote_shell_pid = None

        # Update cwd from what the shell reports
        remote_cwd = self._read_remote_file(self._remote_cwd).strip()
        if remote_cwd:
            self.cwd = remote_cwd

    def _drain_shell_output(self):
        """Drain the shell's stdout/stderr to prevent pipe deadlock.

        Also detects when the shell process dies.
        """
        try:
            for _ in self._shell_proc.stdout:
                pass  # Discard — real output goes to temp files
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

    def _read_remote_file(self, path: str) -> str:
        """Read a file on the remote host via a one-shot SSH command.

        Uses ControlMaster so this is very fast (~5ms on LAN).
        """
        cmd = self._build_ssh_command()
        cmd.append(f"cat {path} 2>/dev/null")
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=10
            )
            return result.stdout
        except (subprocess.TimeoutExpired, OSError):
            return ""

    def _kill_shell_children(self):
        """Kill children of the persistent shell (the running command),
        but not the shell itself."""
        if self._remote_shell_pid is None:
            return
        cmd = self._build_ssh_command()
        cmd.append(f"pkill -P {self._remote_shell_pid} 2>/dev/null; true")
        try:
            subprocess.run(cmd, capture_output=True, timeout=5)
        except (subprocess.TimeoutExpired, OSError):
            pass

    def _execute_persistent(self, command: str, cwd: str, *,
                            timeout: int | None = None,
                            stdin_data: str | None = None) -> dict:
        """Execute a command in the persistent shell."""
        # If shell is dead, restart it
        if not self._shell_alive:
            logger.info("Persistent shell died, restarting...")
            self._start_persistent_shell()

        exec_command, sudo_stdin = self._prepare_command(command)
        effective_timeout = timeout or self.timeout

        # Fall back to one-shot for commands needing piped stdin
        if stdin_data or sudo_stdin:
            return self._execute_oneshot(
                command, cwd, timeout=timeout, stdin_data=stdin_data
            )

        with self._shell_lock:
            return self._execute_persistent_locked(
                exec_command, cwd, effective_timeout
            )

    def _execute_persistent_locked(self, command: str, cwd: str,
                                   timeout: int) -> dict:
        """Inner persistent execution — caller must hold _shell_lock."""
        work_dir = cwd or self.cwd

        # Truncate temp files
        truncate = (
            f": > {self._remote_stdout}\n"
            f": > {self._remote_stderr}\n"
            f": > {self._remote_status}\n"
        )
        self._send_to_shell(truncate)

        # Escape command for eval — use single quotes with proper escaping
        escaped = command.replace("'", "'\\''")

        # Send the IPC script
        ipc_script = (
            f"cd {shlex.quote(work_dir)}\n"
            f"eval '{escaped}' < /dev/null > {self._remote_stdout} 2> {self._remote_stderr}\n"
            f"__EC=$?\n"
            f"pwd > {self._remote_cwd}\n"
            f"echo $__EC > {self._remote_status}\n"
        )
        self._send_to_shell(ipc_script)

        # Poll the status file
        deadline = time.monotonic() + timeout
        poll_interval = 0.05  # 50ms

        while True:
            if is_interrupted():
                self._kill_shell_children()
                stdout = self._read_remote_file(self._remote_stdout)
                stderr = self._read_remote_file(self._remote_stderr)
                output = self._merge_output(stdout, stderr)
                return {
                    "output": output + "\n[Command interrupted]",
                    "returncode": 130,
                }

            if time.monotonic() > deadline:
                self._kill_shell_children()
                stdout = self._read_remote_file(self._remote_stdout)
                stderr = self._read_remote_file(self._remote_stderr)
                output = self._merge_output(stdout, stderr)
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

            # Check if status file has content (command is done)
            status_content = self._read_remote_file(self._remote_status).strip()
            if status_content:
                break

            time.sleep(poll_interval)

        # Read results
        stdout = self._read_remote_file(self._remote_stdout)
        stderr = self._read_remote_file(self._remote_stderr)
        exit_code_str = status_content
        new_cwd = self._read_remote_file(self._remote_cwd).strip()

        # Parse exit code
        try:
            exit_code = int(exit_code_str)
        except ValueError:
            exit_code = 1

        # Update cwd
        if new_cwd:
            self.cwd = new_cwd

        output = self._merge_output(stdout, stderr)
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

    # ------------------------------------------------------------------
    # One-shot execution (original behavior)
    # ------------------------------------------------------------------

    def _execute_oneshot(self, command: str, cwd: str = "", *,
                         timeout: int | None = None,
                         stdin_data: str | None = None) -> dict:
        """Execute a command via a fresh one-shot SSH invocation."""
        work_dir = cwd or self.cwd
        exec_command, sudo_stdin = self._prepare_command(command)
        wrapped = f'cd {work_dir} && {exec_command}'
        effective_timeout = timeout or self.timeout

        # Merge sudo password (if any) with caller-supplied stdin_data.
        if sudo_stdin is not None and stdin_data is not None:
            effective_stdin = sudo_stdin + stdin_data
        elif sudo_stdin is not None:
            effective_stdin = sudo_stdin
        else:
            effective_stdin = stdin_data

        cmd = self._build_ssh_command()
        cmd.extend(["bash", "-c", wrapped])

        try:
            kwargs = self._build_run_kwargs(timeout, effective_stdin)
            # Remove timeout from kwargs -- we handle it in the poll loop
            kwargs.pop("timeout", None)

            _output_chunks = []

            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                stdin=subprocess.PIPE if effective_stdin else subprocess.DEVNULL,
                text=True,
            )

            if effective_stdin:
                try:
                    proc.stdin.write(effective_stdin)
                    proc.stdin.close()
                except Exception:
                    pass

            def _drain():
                try:
                    for line in proc.stdout:
                        _output_chunks.append(line)
                except Exception:
                    pass

            reader = threading.Thread(target=_drain, daemon=True)
            reader.start()
            deadline = time.monotonic() + effective_timeout

            while proc.poll() is None:
                if is_interrupted():
                    proc.terminate()
                    try:
                        proc.wait(timeout=1)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                    reader.join(timeout=2)
                    return {
                        "output": "".join(_output_chunks) + "\n[Command interrupted]",
                        "returncode": 130,
                    }
                if time.monotonic() > deadline:
                    proc.kill()
                    reader.join(timeout=2)
                    return self._timeout_result(effective_timeout)
                time.sleep(0.2)

            reader.join(timeout=5)
            return {"output": "".join(_output_chunks), "returncode": proc.returncode}

        except Exception as e:
            return {"output": f"SSH execution error: {str(e)}", "returncode": 1}

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def execute(self, command: str, cwd: str = "", *,
                timeout: int | None = None,
                stdin_data: str | None = None) -> dict:
        if self.persistent:
            return self._execute_persistent(
                command, cwd, timeout=timeout, stdin_data=stdin_data
            )
        return self._execute_oneshot(
            command, cwd, timeout=timeout, stdin_data=stdin_data
        )

    def cleanup(self):
        # Persistent shell teardown
        if self.persistent and self._shell_proc is not None:
            # Remove remote temp files
            if self._session_id:
                try:
                    cmd = self._build_ssh_command()
                    cmd.append(
                        f"rm -f /tmp/hermes-ssh-{self._session_id}-*"
                    )
                    subprocess.run(cmd, capture_output=True, timeout=5)
                except (OSError, subprocess.SubprocessError):
                    pass

            # Close the shell
            try:
                self._shell_proc.stdin.close()
            except Exception:
                pass
            try:
                self._shell_proc.terminate()
                self._shell_proc.wait(timeout=3)
            except Exception:
                try:
                    self._shell_proc.kill()
                except Exception:
                    pass
            self._shell_alive = False
            self._shell_proc = None

        # ControlMaster cleanup
        if self.control_socket.exists():
            try:
                cmd = ["ssh", "-o", f"ControlPath={self.control_socket}",
                       "-O", "exit", f"{self.user}@{self.host}"]
                subprocess.run(cmd, capture_output=True, timeout=5)
            except (OSError, subprocess.SubprocessError):
                pass
            try:
                self.control_socket.unlink()
            except OSError:
                pass
