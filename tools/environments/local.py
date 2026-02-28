"""Local execution environment with interrupt support and non-blocking I/O."""

import os
import platform
import shutil
import signal
import subprocess
import threading
import time

from tools.environments.base import BaseEnvironment

# Detect platform once at import time.
_IS_WINDOWS = platform.system() == "Windows"

# Noise lines emitted by interactive shells when stdin is not a terminal.
# Filtered from output to keep tool results clean.
_SHELL_NOISE = frozenset({
    "bash: no job control in this shell",
    "bash: no job control in this shell\n",
    "no job control in this shell",
    "no job control in this shell\n",
})


def _clean_shell_noise(output: str) -> str:
    """Strip shell startup warnings that leak when using -i without a TTY."""
    lines = output.split("\n", 2)  # only check first two lines
    if lines and lines[0].strip() in _SHELL_NOISE:
        return "\n".join(lines[1:])
    return output


def _kill_process_tree(proc: subprocess.Popen, *, force: bool = False) -> None:
    """Terminate a process and its entire process group/tree.

    On POSIX systems this sends SIGTERM (or SIGKILL when *force* is True) to
    the entire process group so child processes spawned by the shell are also
    cleaned up.  On Windows, ``proc.kill()`` is used directly because there is
    no ``os.killpg`` equivalent in the standard library.

    Args:
        proc: The subprocess to terminate.
        force: If True, send SIGKILL (POSIX) or ``proc.kill()`` immediately
               without attempting a graceful SIGTERM first.
    """
    if _IS_WINDOWS:
        # Windows: terminate the process directly.  For more thorough child
        # cleanup a future improvement could use ``taskkill /F /T /PID``.
        try:
            proc.kill()
        except (ProcessLookupError, PermissionError, OSError):
            pass
        return

    # POSIX: send signal to the whole process group.
    sig = signal.SIGKILL if force else signal.SIGTERM
    try:
        pgid = os.getpgid(proc.pid)
        os.killpg(pgid, sig)
    except (ProcessLookupError, PermissionError, OSError):
        # Fallback: kill the process directly if we can't reach the group.
        try:
            proc.kill() if force else proc.terminate()
        except (ProcessLookupError, PermissionError, OSError):
            pass


class LocalEnvironment(BaseEnvironment):
    """Run commands directly on the host machine.

    Features:
    - Popen + polling for interrupt support (user can cancel mid-command)
    - Background stdout drain thread to prevent pipe buffer deadlocks
    - stdin_data support for piping content (bypasses ARG_MAX limits)
    - sudo -S transform via SUDO_PASSWORD env var
    - Uses interactive login shell so full user env is available
    - Cross-platform: works on Linux, macOS, and Windows
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
            if _IS_WINDOWS:
                # On Windows, use cmd.exe with /C to run the command.
                # There is no interactive login shell equivalent, so tools
                # relying on shell init scripts (nvm, pyenv, etc.) may need
                # the user to configure PATH explicitly.
                shell_cmd = ["cmd.exe", "/C", exec_command]
                popen_kwargs: dict = {}
            else:
                # Use the user's shell as an interactive login shell (-lic) so
                # that ALL rc files are sourced — including content after the
                # interactive guard in .bashrc (case $- in *i*)..esac) where
                # tools like nvm, pyenv, and cargo install their init scripts.
                # -l alone isn't enough: .profile sources .bashrc, but the guard
                # returns early because the shell isn't interactive.
                user_shell = (
                    os.environ.get("SHELL")
                    or shutil.which("bash")
                    or "/bin/bash"
                )
                shell_cmd = [user_shell, "-lic", exec_command]
                # setsid creates a new process group so we can kill all
                # child processes atomically via killpg.  POSIX-only.
                popen_kwargs = {"preexec_fn": os.setsid}

            proc = subprocess.Popen(
                shell_cmd,
                text=True,
                cwd=work_dir,
                env=os.environ | self.env,
                encoding="utf-8",
                errors="replace",
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                stdin=subprocess.PIPE if stdin_data is not None else subprocess.DEVNULL,
                **popen_kwargs,
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
                    _kill_process_tree(proc)
                    try:
                        proc.wait(timeout=1.0)
                    except subprocess.TimeoutExpired:
                        _kill_process_tree(proc, force=True)
                    reader.join(timeout=2)
                    return {
                        "output": "".join(_output_chunks) + "\n[Command interrupted — user sent a new message]",
                        "returncode": 130,
                    }
                if time.monotonic() > deadline:
                    _kill_process_tree(proc)
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
