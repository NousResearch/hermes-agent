import os
import signal
import subprocess
import time
import re
import asyncio

_ANSI_ESCAPE = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')

def strip_ansi(text: str) -> str:
    return _ANSI_ESCAPE.sub("", text).strip()


def default_home() -> str:
    """User's home directory.  Override via OMARCHY_USER_HOME env var."""
    return os.environ.get("OMARCHY_USER_HOME") or os.path.expanduser("~")


async def run_command_async(cmd: list[str], cwd: str | None = None,
                            timeout: int = 300, env: dict | None = None,
                            stdin_data: bytes | None = None) -> dict:
    """Async subprocess runner using asyncio.create_subprocess_exec.

    Fully cancellable: if the calling asyncio task is cancelled while a
    subprocess is running, CancelledError propagates and the process group
    is killed with SIGTERM then SIGKILL. No orphaned children, no leaked
    workers.

    Returns a dict with keys: ok, output, exit_code, duration_s, error.
    """
    start = time.monotonic()
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)

    try:
        proc = await asyncio.wait_for(
            asyncio.create_subprocess_exec(
                *cmd,
                stdin=asyncio.subprocess.PIPE if stdin_data is not None
                      else asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                cwd=cwd or default_home(),
                env=merged_env,
                start_new_session=True,
            ),
            timeout=max(timeout, 10),
        )
    except (asyncio.TimeoutError, TimeoutError):
        return _timeout_result(time.monotonic() - start, timeout)
    except FileNotFoundError as e:
        return _error_result(time.monotonic() - start,
                             f"Command not found: {e}")
    except Exception as e:
        return _error_result(time.monotonic() - start, str(e))

    # Communicate with the process.  If the asyncio task is cancelled while
    # communicate() is waiting (e.g. the HTTP request that triggered this
    # tool call is dropped), the process group is reaped immediately.
    try:
        stdout_bytes, _ = await asyncio.wait_for(
            proc.communicate(input=stdin_data),
            timeout=timeout,
        )
        exit_code = proc.returncode
    except (asyncio.TimeoutError, TimeoutError):
        _kill_process_group(proc)
        return _timeout_result(time.monotonic() - start, timeout)
    except asyncio.CancelledError:
        _kill_process_group(proc)
        raise  # let the caller know the request was cancelled
    except Exception as e:
        _kill_process_group(proc)
        return _error_result(time.monotonic() - start, str(e))

    duration = round(time.monotonic() - start, 2)
    output = strip_ansi(stdout_bytes.decode("utf-8", errors="replace"))
    return {
        "ok": exit_code == 0,
        "output": output,
        "exit_code": exit_code,
        "duration_s": duration,
        "error": None if exit_code == 0 else f"exit code {exit_code}",
    }


def _kill_process_group(proc: asyncio.subprocess.Process) -> None:
    """Kill the entire process group. Best-effort; the child may already be
    gone or owned by root."""
    try:
        pgid = os.getpgid(proc.pid)
    except (ProcessLookupError, PermissionError):
        return
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(pgid, sig)
        except (ProcessLookupError, PermissionError):
            return
        if sig == signal.SIGTERM:
            try:
                asyncio.wait_for(proc.wait(), timeout=5)
                return
            except (asyncio.TimeoutError, ProcessLookupError):
                continue


def _timeout_result(elapsed: float, timeout: int, msg: str = "") -> dict:
    return {
        "ok": False,
        "output": f"Command timed out after {timeout}s ({msg})",
        "exit_code": -1,
        "duration_s": round(elapsed, 2),
        "error": "timeout",
    }


def _error_result(elapsed: float, msg: str) -> dict:
    return {
        "ok": False, "output": msg,
        "exit_code": -1, "duration_s": round(elapsed, 2), "error": msg,
    }


def run_command(cmd: list[str], cwd: str | None = None,
                timeout: int = 300, env: dict | None = None) -> dict:
    """Synchronous subprocess runner (used by system_run internally
    via asyncio.to_thread).  Kept for callers that do not have an event
    loop."""
    start = time.monotonic()
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    try:
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            cwd=cwd or default_home(),
            env=merged_env,
            start_new_session=True,
        )
        try:
            stdout_bytes, _ = proc.communicate(timeout=timeout)
            exit_code = proc.returncode
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
            return _timeout_result(time.monotonic() - start, timeout)
        duration = time.monotonic() - start
        output = strip_ansi(stdout_bytes.decode("utf-8", errors="replace"))
        return {
            "ok": exit_code == 0,
            "output": output,
            "exit_code": exit_code,
            "duration_s": round(duration, 2),
            "error": None if exit_code == 0 else f"exit code {exit_code}",
        }
    except FileNotFoundError as e:
        return _error_result(time.monotonic() - start,
                             f"Command not found: {e}")
    except Exception as e:
        return _error_result(time.monotonic() - start, str(e))
