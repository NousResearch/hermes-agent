"""Local execution environment with interrupt support and non-blocking I/O."""

import os
import signal
import subprocess
import threading
import time

from tools.environments.base import BaseEnvironment


class LocalEnvironment(BaseEnvironment):
    """Run commands directly on the host machine.

    Features:
    - Popen + polling for interrupt support (user can cancel mid-command)
    - Background stdout drain thread to prevent pipe buffer deadlocks
    - stdin_data support for piping content (bypasses ARG_MAX limits)
    - sudo -S transform via SUDO_PASSWORD env var
    """

    def __init__(self, cwd: str = "", timeout: int = 60, env: dict = None):
        super().__init__(cwd=cwd or os.getcwd(), timeout=timeout, env=env)

    def execute(
        self, command: str, cwd: str = "", *, timeout: int | None = None, stdin_data: str | None = None
    ) -> dict:
        from tools.terminal_tool import _interrupt_event

        work_dir = cwd or self.cwd or os.getcwd()
        effective_timeout = timeout or self.timeout
        exec_command = self._prepare_command(command)

        try:
            proc = subprocess.Popen(
                exec_command,
                shell=True,
                text=True,
                cwd=work_dir,
                env=os.environ | self.env,
                encoding="utf-8",
                errors="replace",
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                stdin=subprocess.PIPE if stdin_data is not None else subprocess.DEVNULL,
                preexec_fn=os.setsid,
            )

            if stdin_data is not None:

                def _write_stdin():
                    try:
                        proc.stdin.write(stdin_data)
                        proc.stdin.close()
                    except (BrokenPipeError, OSError):
                        pass

                threading.Thread(target=_write_stdin, daemon=True).start()

            _output_chunks: list[str] = []

            def _drain_stdout():
                try:
                    if proc.stdout:
                        for line in proc.stdout:
                            _output_chunks.append(line)
                except ValueError:
                    pass
                finally:
                    try:
                        proc.stdout.close()
                    except Exception:
                        pass

            reader = threading.Thread(target=_drain_stdout, daemon=True)
            reader.start()
            deadline = time.monotonic() + effective_timeout

            while proc.poll() is None:
                if _interrupt_event.is_set():
                    try:
                        pgid = os.getpgid(proc.pid)
                        os.killpg(pgid, signal.SIGTERM)
                        try:
                            proc.wait(timeout=1.0)
                        except subprocess.TimeoutExpired:
                            os.killpg(pgid, signal.SIGKILL)
                    except (ProcessLookupError, PermissionError):
                        proc.kill()
                    reader.join(timeout=2)
                    return {
                        "output": "".join(_output_chunks) + "\n[Command interrupted — user sent a new message]",
                        "returncode": 130,
                    }
                if time.monotonic() > deadline:
                    try:
                        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                    except (ProcessLookupError, PermissionError):
                        proc.kill()
                    reader.join(timeout=2)
                    return self._timeout_result(effective_timeout)
                time.sleep(0.2)

            reader.join(timeout=5)
            return {"output": "".join(_output_chunks), "returncode": proc.returncode}

        except Exception as e:
            return {"output": f"Execution error: {str(e)}", "returncode": 1}

    def cleanup(self):
        pass
