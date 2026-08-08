"""Shared subprocess runner for user-configured "command provider" shell templates.

TTS (``tools/tts_tool.py``) and STT (``tools/transcription_tools.py``) both let a
user point a provider at an arbitrary shell command. Both need the same three
behaviors, which live here so the two call sites cannot drift apart:

- process-tree termination, so a stalled ``curl | ffmpeg`` pipeline does not
  leave orphaned children behind;
- a **progress-based idle timeout**: ``timeout`` is reset whenever the command
  emits output, so a slow-but-alive provider survives while a silently stalled
  one is killed (#50081);
- a child environment scrubbed of Hermes secrets (salvage of #56332) with an
  opt-out ``env_passthrough`` allowlist for the provider's own API keys.
"""

from __future__ import annotations

import logging
import os
import queue
import subprocess
import threading
import time
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

_READ_CHUNK_SIZE = 65536


def terminate_process_tree(proc: subprocess.Popen) -> None:
    """Best-effort termination of a shell process and all of its children."""
    if proc.poll() is not None:
        return

    if os.name == "nt":
        try:
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=5,
                stdin=subprocess.DEVNULL,
            )
        except Exception:
            proc.kill()
        return

    try:
        import psutil  # type: ignore
    except ImportError:
        # psutil is optional — fall back to single-process terminate/kill
        proc.terminate()
        try:
            proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            proc.kill()
        return

    try:
        parent = psutil.Process(proc.pid)
        for child in parent.children(recursive=True):
            try:
                child.terminate()
            except psutil.NoSuchProcess:
                pass
        parent.terminate()
    except psutil.NoSuchProcess:
        return
    except Exception:
        proc.terminate()

    try:
        proc.wait(timeout=2)
        return
    except subprocess.TimeoutExpired:
        pass

    try:
        parent = psutil.Process(proc.pid)
        for child in parent.children(recursive=True):
            try:
                child.kill()
            except psutil.NoSuchProcess:
                pass
        parent.kill()
    except psutil.NoSuchProcess:
        return
    except Exception:
        proc.kill()


def provider_env_passthrough(config: Dict[str, Any]) -> list:
    """Return the provider's ``env_passthrough`` allowlist (opt-out of scrub).

    Command providers legitimately reference their own API keys in the shell
    template (curl one-liners). The child env is scrubbed of Hermes secrets by
    default; ``env_passthrough: [MY_API_KEY, ...]`` copies the named variables
    back from the parent environment so a trusted template keeps working.
    """
    raw = config.get("env_passthrough")
    if not isinstance(raw, (list, tuple)):
        return []
    return [str(item).strip() for item in raw if str(item).strip()]


def run_command_provider(
    command: str,
    timeout: float,
    env_passthrough: Optional[list] = None,
) -> subprocess.CompletedProcess:
    """Run a command-provider shell command with process-tree idle cleanup.

    ``timeout`` is an IDLE timeout, reset whenever the command emits output on
    stdout/stderr. Raises :class:`subprocess.TimeoutExpired` (carrying the
    output captured so far) when the command goes silent for longer than
    ``timeout``, and :class:`subprocess.CalledProcessError` on a non-zero exit.
    """
    from agent.delegation_context import delegated_child_subprocess_env
    from tools.environments.local import hermes_subprocess_env

    scrubbed = hermes_subprocess_env(inherit_credentials=False)
    for key in env_passthrough or []:
        value = os.environ.get(key)
        if value is not None:
            scrubbed[key] = value
    popen_kwargs: Dict[str, Any] = {
        "shell": True,
        "stdout": subprocess.PIPE,
        "stderr": subprocess.PIPE,
        "text": True,
        # Lossy UTF-8 decode — locale-mismatched bytes from the provider
        # command must not raise in the reader threads on non-UTF-8
        # Windows (#45099).
        "encoding": "utf-8",
        "errors": "replace",
        "env": delegated_child_subprocess_env(scrubbed),
    }
    if os.name == "nt":
        popen_kwargs["creationflags"] = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    else:
        popen_kwargs["start_new_session"] = True

    proc = subprocess.Popen(command, **popen_kwargs, stdin=subprocess.DEVNULL)
    output_queue: "queue.Queue[tuple[str, Optional[str]]]" = queue.Queue()
    chunks: Dict[str, list] = {"stdout": [], "stderr": []}
    open_streams = {"stdout", "stderr"}

    def read_stream(name: str, stream: Any) -> None:
        encoding = getattr(stream, "encoding", None) or "utf-8"
        read1 = getattr(getattr(stream, "buffer", None), "read1", None)
        try:
            while True:
                if read1 is None:
                    chunk = stream.read(_READ_CHUNK_SIZE)
                else:
                    data = read1(_READ_CHUNK_SIZE)
                    chunk = data.decode(encoding, errors="replace")
                if not chunk:
                    break
                output_queue.put((name, chunk))
        finally:
            output_queue.put((name, None))

    readers = [
        threading.Thread(
            target=read_stream,
            args=("stdout", proc.stdout),
            daemon=True,
        ),
        threading.Thread(
            target=read_stream,
            args=("stderr", proc.stderr),
            daemon=True,
        ),
    ]
    for reader in readers:
        reader.start()

    deadline = time.monotonic() + timeout
    timed_out = False
    while open_streams:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            timed_out = True
            break
        try:
            name, chunk = output_queue.get(timeout=min(0.05, remaining))
        except queue.Empty:
            continue
        if chunk is None:
            open_streams.discard(name)
            continue
        chunks[name].append(chunk)
        deadline = time.monotonic() + timeout

    if not timed_out:
        try:
            proc.wait(timeout=max(0.0, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            timed_out = True

    if timed_out:
        terminate_process_tree(proc)
        for reader in readers:
            reader.join(timeout=0.5)
        while True:
            try:
                name, chunk = output_queue.get_nowait()
            except queue.Empty:
                break
            if chunk:
                chunks[name].append(chunk)
        stdout = "".join(chunks["stdout"])
        stderr = "".join(chunks["stderr"])
        try:
            raise subprocess.TimeoutExpired(command, timeout)
        except subprocess.TimeoutExpired as exc:
            raise subprocess.TimeoutExpired(
                command,
                timeout,
                output=stdout,
                stderr=stderr,
            ) from exc

    stdout = "".join(chunks["stdout"])
    stderr = "".join(chunks["stderr"])

    if proc.returncode:
        raise subprocess.CalledProcessError(
            proc.returncode,
            command,
            output=stdout,
            stderr=stderr,
        )
    return subprocess.CompletedProcess(command, proc.returncode, stdout, stderr)
