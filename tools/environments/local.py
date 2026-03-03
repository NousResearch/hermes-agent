"""Local execution environment with interrupt support and non-blocking I/O."""

import os
import platform
import shutil
import signal
import subprocess
import threading
import time

_IS_WINDOWS = platform.system() == "Windows"

from tools.environments.base import BaseEnvironment
from tools.environments.shell_utils import (
    build_local_subprocess_invocation,
    terminate_process_tree,
)

# Noise lines emitted by interactive shells when stdin is not a terminal.
# Filtered from output to keep tool results clean.
_SHELL_NOISE_SUBSTRINGS = (
    "bash: cannot set terminal process group",
    "bash: no job control in this shell",
    "no job control in this shell",
    "cannot set terminal process group",
    "tcsetattr: Inappropriate ioctl for device",
)


def _clean_shell_noise(output: str) -> str:
    """Strip shell startup warnings that leak when using -i without a TTY.

    Removes all leading lines that match known noise patterns, not just the first.
    Some environments emit multiple noise lines (e.g. Docker, non-TTY sessions).
    """
    lines = output.split("\n")
    # Strip all leading noise lines
    while lines and any(noise in lines[0] for noise in _SHELL_NOISE_SUBSTRINGS):
        lines.pop(0)
    return "\n".join(lines)


class LocalEnvironment(BaseEnvironment):
    """Run commands directly on the host machine.

    Features:
    - Popen + polling for interrupt support (user can cancel mid-command)
    - Background stdout drain thread to prevent pipe buffer deadlocks
    - stdin_data support for piping content (bypasses ARG_MAX limits)
    - sudo -S transform via SUDO_PASSWORD env var
    - Uses interactive login shell so full user env is available
    """

    def __init__(self, cwd: str = "", timeout: int = 60, env: dict = None):
        super().__init__(cwd=cwd or os.getcwd(), timeout=timeout, env=env)

    def execute(self, command: str, cwd: str = "", *,
                timeout: int | None = None,
                stdin_data: str | None = None) -> dict:
        from tools.terminal_tool import _interrupt_event

        work_dir = cwd or self.cwd or os.getcwd()
        effective_timeout = timeout or self.timeout
        exec_command = self._prepare_command(command)

        try:
            popen_args, popen_platform_kwargs, _ = build_local_subprocess_invocation(
                exec_command, work_dir
            )
            proc = subprocess.Popen(
                popen_args,
                text=True,
                env=os.environ | self.env,
                encoding="utf-8",
                errors="replace",
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                stdin=subprocess.PIPE if stdin_data is not None else subprocess.DEVNULL,
                **popen_platform_kwargs,
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
                    terminate_process_tree(proc, force=False)
                    try:
                        proc.wait(timeout=1.0)
                    except subprocess.TimeoutExpired:
                        terminate_process_tree(proc, force=True)
                    reader.join(timeout=2)
                    return {
                        "output": "".join(_output_chunks) + "\n[Command interrupted — user sent a new message]",
                        "returncode": 130,
                    }
                if time.monotonic() > deadline:
                    terminate_process_tree(proc, force=False)
                    try:
                        proc.wait(timeout=1.0)
                    except subprocess.TimeoutExpired:
                        terminate_process_tree(proc, force=True)
                    reader.join(timeout=2)
                    return self._timeout_result(effective_timeout)
                time.sleep(0.2)

            reader.join(timeout=5)
            output = _clean_shell_noise("".join(_output_chunks))
            return {"output": output, "returncode": proc.returncode}

        except Exception as e:
            return {"output": f"Execution error: {str(e)}", "returncode": 1}

    def cleanup(self):
        pass
