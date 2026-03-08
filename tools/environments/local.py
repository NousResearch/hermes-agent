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

# Unique marker to isolate real command output from shell init/exit noise.
# printf (no trailing newline) keeps the boundaries clean for splitting.
_OUTPUT_FENCE = "__HERMES_FENCE_a9f7b3__"


def _find_shell() -> str:
    """Find the best shell for command execution.

    On Unix: uses $SHELL, falls back to bash.
    On Windows: uses Git Bash (bundled with Git for Windows).
    Raises RuntimeError if no suitable shell is found on Windows.
    """
    if not _IS_WINDOWS:
        return (
            os.environ.get("SHELL")
            or shutil.which("bash")
            or ("/usr/bin/bash" if os.path.isfile("/usr/bin/bash") else None)
            or ("/bin/bash" if os.path.isfile("/bin/bash") else None)
            or "/bin/sh"
        )

    # Windows: look for Git Bash (installed with Git for Windows).
    # Allow override via env var (same pattern as Claude Code).
    custom = os.environ.get("HERMES_GIT_BASH_PATH")
    if custom and os.path.isfile(custom):
        return custom

    # shutil.which finds bash.exe if Git\bin is on PATH
    found = shutil.which("bash")
    if found:
        return found

    # Check common Git for Windows install locations
    for candidate in (
        os.path.join(os.environ.get("ProgramFiles", r"C:\Program Files"), "Git", "bin", "bash.exe"),
        os.path.join(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)"), "Git", "bin", "bash.exe"),
        os.path.join(os.environ.get("LOCALAPPDATA", ""), "Programs", "Git", "bin", "bash.exe"),
    ):
        if candidate and os.path.isfile(candidate):
            return candidate

    raise RuntimeError(
        "Git Bash not found. Hermes Agent requires Git for Windows on Windows.\n"
        "Install it from: https://git-scm.com/download/win\n"
        "Or set HERMES_GIT_BASH_PATH to your bash.exe location."
    )

# Noise lines emitted by interactive shells when stdin is not a terminal.
# Used as a fallback when output fence markers are missing.
_SHELL_NOISE_SUBSTRINGS = (
    # bash
    "bash: cannot set terminal process group",
    "bash: no job control in this shell",
    "no job control in this shell",
    "cannot set terminal process group",
    "tcsetattr: Inappropriate ioctl for device",
    # zsh / oh-my-zsh / macOS terminal session
    "Restored session:",
    "Saving session...",
    "Last login:",
    "command not found:",
    "Oh My Zsh",
    "compinit:",
)


def _clean_shell_noise(output: str) -> str:
    """Strip shell startup/exit warnings that leak when using -i without a TTY.

    Removes lines matching known noise patterns from both the beginning
    and end of the output.  Lines in the middle are left untouched.
    """

    def _is_noise(line: str) -> bool:
        return any(noise in line for noise in _SHELL_NOISE_SUBSTRINGS)

    lines = output.split("\n")

    # Strip leading noise
    while lines and _is_noise(lines[0]):
        lines.pop(0)

    # Strip trailing noise (walk backwards, skip empty lines from split)
    end = len(lines) - 1
    while end >= 0 and (not lines[end] or _is_noise(lines[end])):
        end -= 1

    if end < 0:
        return ""

    cleaned = lines[: end + 1]
    result = "\n".join(cleaned)

    # Preserve trailing newline if original had one
    if output.endswith("\n") and result and not result.endswith("\n"):
        result += "\n"
    return result


def _extract_fenced_output(raw: str) -> str:
    """Extract real command output from between fence markers.

    The execute() method wraps each command with printf(FENCE) markers.
    This function finds the first and last fence and returns only the
    content between them, which is the actual command output free of
    any shell init/exit noise.

    Falls back to pattern-based _clean_shell_noise if fences are missing.
    """
    first = raw.find(_OUTPUT_FENCE)
    if first == -1:
        return _clean_shell_noise(raw)

    start = first + len(_OUTPUT_FENCE)
    last = raw.rfind(_OUTPUT_FENCE)

    if last <= first:
        # Only start fence found (e.g. user command called `exit`)
        return _clean_shell_noise(raw[start:])

    return raw[start:last]


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
            # Use the user's shell as an interactive login shell (-lic) so
            # that ALL rc files are sourced — including content after the
            # interactive guard in .bashrc (case $- in *i*)..esac) where
            # tools like nvm, pyenv, and cargo install their init scripts.
            # -l alone isn't enough: .profile sources .bashrc, but the guard
            # returns early because the shell isn't interactive.
            user_shell = _find_shell()
            # Wrap with output fences so we can later extract the real
            # command output and discard shell init/exit noise.
            fenced_cmd = (
                f"printf '{_OUTPUT_FENCE}';"
                f" {exec_command};"
                f" __hermes_rc=$?;"
                f" printf '{_OUTPUT_FENCE}';"
                f" exit $__hermes_rc"
            )
            # Ensure PATH always includes standard dirs — systemd services
            # and some terminal multiplexers inherit a minimal PATH.
            _SANE_PATH = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
            run_env = dict(os.environ | self.env)
            existing_path = run_env.get("PATH", "")
            if "/usr/bin" not in existing_path.split(":"):
                run_env["PATH"] = f"{existing_path}:{_SANE_PATH}" if existing_path else _SANE_PATH

            proc = subprocess.Popen(
                [user_shell, "-lic", fenced_cmd],
                text=True,
                cwd=work_dir,
                env=run_env,
                encoding="utf-8",
                errors="replace",
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                stdin=subprocess.PIPE if stdin_data is not None else subprocess.DEVNULL,
                preexec_fn=None if _IS_WINDOWS else os.setsid,
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
                    try:
                        if _IS_WINDOWS:
                            proc.terminate()
                        else:
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
                        if _IS_WINDOWS:
                            proc.terminate()
                        else:
                            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                    except (ProcessLookupError, PermissionError):
                        proc.kill()
                    reader.join(timeout=2)
                    return self._timeout_result(effective_timeout)
                time.sleep(0.2)

            reader.join(timeout=5)
            output = _extract_fenced_output("".join(_output_chunks))
            return {"output": output, "returncode": proc.returncode}

        except Exception as e:
            return {"output": f"Execution error: {str(e)}", "returncode": 1}

    def cleanup(self):
        pass
