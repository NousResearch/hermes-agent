"""
Process Registry -- In-memory registry for managed background processes.

Tracks processes spawned via terminal(background=true), providing:
  - Output buffering (rolling 200KB window)
  - Status polling and log retrieval
  - Blocking wait with interrupt support
  - Process killing
  - Crash recovery via JSON checkpoint file
  - Session-scoped tracking for gateway reset protection

Background processes execute THROUGH the environment interface -- nothing
runs on the host machine unless TERMINAL_ENV=local. For Docker, Singularity,
Modal, Daytona, and SSH backends, the command runs inside the sandbox.

Usage:
    from tools.process_registry import process_registry

    # Spawn a background process (called from terminal_tool)
    session = process_registry.spawn(env, "pytest -v", task_id="task_123")

    # Poll for status
    result = process_registry.poll(session.id)

    # Block until done
    result = process_registry.wait(session.id, timeout=300)

    # Kill it
    process_registry.kill(session.id)
"""

import json
import logging
import os
import platform
import shlex
import signal
import subprocess
import threading
import time
import uuid

_IS_WINDOWS = platform.system() == "Windows"
from tools.environments.local import _find_shell, _sanitize_subprocess_env
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from hermes_cli.config import get_hermes_home

logger = logging.getLogger(__name__)


# Checkpoint file for crash recovery (gateway only)
CHECKPOINT_PATH = get_hermes_home() / "processes.json"

# Limits
MAX_OUTPUT_CHARS = 200_000      # 200KB rolling output buffer
FINISHED_TTL_SECONDS = 1800     # Keep finished processes for 30 minutes
MAX_PROCESSES = 64              # Max concurrent tracked processes (LRU pruning)

# Watch pattern rate limiting
WATCH_MAX_PER_WINDOW = 8        # Max notifications delivered per window
WATCH_WINDOW_SECONDS = 10       # Rolling window length
WATCH_OVERLOAD_KILL_SECONDS = 45  # Sustained overload duration before disabling watch


def format_uptime_short(seconds: int) -> str:
    s = max(0, int(seconds))
    if s < 60:
        return f"{s}s"
    mins, secs = divmod(s, 60)
    if mins < 60:
        return f"{mins}m {secs}s"
    hours, mins = divmod(mins, 60)
    return f"{hours}h {mins}m"


@dataclass
class ProcessSession:
    """A tracked background process with output buffering."""
    id: str                                     # Unique session ID ("proc_xxxxxxxxxxxx")
    command: str                                 # Original command string
    task_id: str = ""                           # Task/sandbox isolation key
    session_key: str = ""                       # Gateway session key (for reset protection)
    pid: Optional[int] = None                   # OS process ID
    process: Optional[subprocess.Popen] = None  # Popen handle (local only)
    env_ref: Any = None                         # Reference to the environment object
    cwd: Optional[str] = None                   # Working directory
    started_at: float = 0.0                     # time.time() of spawn
    exited: bool = False                        # Whether the process has finished
    exit_code: Optional[int] = None             # Exit code (None if still running)
    output_buffer: str = ""                     # Rolling output (last MAX_OUTPUT_CHARS)
    max_output_chars: int = MAX_OUTPUT_CHARS
    detached: bool = False                      # True if recovered from crash (no pipe)
    pid_scope: str = "host"                     # "host" for local/PTY PIDs, "sandbox" for env-local PIDs
    # Watcher/notification metadata (persisted for crash recovery)
    watcher_platform: str = ""
    watcher_chat_id: str = ""
    watcher_user_id: str = ""
    watcher_user_name: str = ""
    watcher_thread_id: str = ""
    watcher_interval: int = 0                   # 0 = no watcher configured
    notify_on_complete: bool = False             # Queue agent notification on exit
    # Watch patterns — trigger agent notification when output matches any pattern
    watch_patterns: List[str] = field(default_factory=list)
    _watch_hits: int = field(default=0, repr=False)          # total matches delivered
    _watch_suppressed: int = field(default=0, repr=False)    # matches dropped by rate limit
    _watch_overload_since: float = field(default=0.0, repr=False)  # when sustained overload began
    _watch_disabled: bool = field(default=False, repr=False) # permanently killed by overload
    _watch_window_hits: int = field(default=0, repr=False)   # hits in current rate window
    _watch_window_start: float = field(default=0.0, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock)
    _reader_thread: Optional[threading.Thread] = field(default=None, repr=False)
    _pty: Any = field(default=None, repr=False)  # ptyprocess handle (when use_pty=True)


class ProcessRegistry:
    """
    In-memory registry of running and finished background processes.

    Thread-safe. Accessed from:
      - Executor threads (terminal_tool, process tool handlers)
      - Gateway asyncio loop (watcher tasks, session reset checks)
      - Cleanup thread (sandbox reaping coordination)
    """

    _SHELL_NOISE_SUBSTRINGS = (
        "bash: cannot set terminal process group",
        "bash: no job control in this shell",
        "no job control in this shell",
        "cannot set terminal process group",
        "tcsetattr: Inappropriate ioctl for device",
    )

    def __init__(self):
        self._running: Dict[str, ProcessSession] = {}
        self._finished: Dict[str, ProcessSession] = {}
        self._lock = threading.Lock()

        # Side-channel for check_interval watchers (gateway reads after agent run)
        self.pending_watchers: List[Dict[str, Any]] = []

        # Notification queue — unified queue for all background process events.
        # Completion notifications (notify_on_complete) and watch pattern matches
        # both land here, distinguished by "type" field.  CLI process_loop and
        # gateway drain this after each agent turn to auto-trigger new turns.
        import queue as _queue_mod
        self.completion_queue: _queue_mod.Queue = _queue_mod.Queue()

        # Track sessions whose completion was already consumed by the agent
        # via wait/poll/log.  Drain loops skip notifications for these.
        self._completion_consumed: set = set()

    @staticmethod
    def _clean_shell_noise(text: str) -> str:
        """Strip shell startup warnings from the beginning of output."""
        lines = text.split("\n")
        while lines and any(noise in lines[0] for noise in ProcessRegistry._SHELL_NOISE_SUBSTRINGS):
            lines.pop(0)
        return "\n".join(lines)

    def _check_watch_patterns(self, session: ProcessSession, new_text: str) -> None:
        """Scan new output for watch patterns and queue notifications.

        Called from reader threads with new_text being the freshly-read chunk.
        Rate-limited: max WATCH_MAX_PER_WINDOW notifications per WATCH_WINDOW_SECONDS.
        If sustained overload exceeds WATCH_OVERLOAD_KILL_SECONDS, watching is
        disabled permanently for this process.
        """
        if not session.watch_patterns or session._watch_disabled:
            return

        # Scan new text line-by-line for pattern matches
        matched_lines = []
        matched_pattern = None
        for line in new_text.splitlines():
            for pat in session.watch_patterns:
                if pat in line:
                    matched_lines.append(line.rstrip())
                    if matched_pattern is None:
                        matched_pattern = pat
                    break  # one match per line is enough

        if not matched_lines:
            return

        now = time.time()
        with session._lock:
            # Reset window if it's expired
            if now - session._watch_window_start >= WATCH_WINDOW_SECONDS:
                session._watch_window_hits = 0
                session._watch_window_start = now

            # Check rate limit
            if session._watch_window_hits >= WATCH_MAX_PER_WINDOW:
                session._watch_suppressed += len(matched_lines)

                # Track sustained overload for kill switch
                if session._watch_overload_since == 0.0:
                    session._watch_overload_since = now
                elif now - session._watch_overload_since > WATCH_OVERLOAD_KILL_SECONDS:
                    session._watch_disabled = True
                    self.completion_queue.put({
                        "session_id": session.id,
                        "session_key": session.session_key,
                        "command": session.command,
                        "type": "watch_disabled",
                        "suppressed": session._watch_suppressed,
                        "platform": session.watcher_platform,
                        "chat_id": session.watcher_chat_id,
                        "user_id": session.watcher_user_id,
                        "user_name": session.watcher_user_name,
                        "thread_id": session.watcher_thread_id,
                        "message": (
                            f"Watch patterns disabled for process {session.id} — "
                            f"too many matches ({session._watch_suppressed} suppressed). "
                            f"Use process(action='poll') to check output manually."
                        ),
                    })
                return

            # Under the rate limit — deliver notification
            session._watch_window_hits += 1
            session._watch_hits += 1
            # Clear overload tracker since we got a delivery through
            session._watch_overload_since = 0.0

            # Include suppressed count if any events were dropped
            suppressed = session._watch_suppressed
            session._watch_suppressed = 0

        # Trim matched output to a reasonable size
        output = "\n".join(matched_lines[:20])
        if len(output) > 2000:
            output = output[:2000] + "\n...(truncated)"

        self.completion_queue.put({
            "session_id": session.id,
            "session_key": session.session_key,
            "command": session.command,
            "type": "watch_match",
            "pattern": matched_pattern,
            "output": output,
            "suppressed": suppressed,
            "platform": session.watcher_platform,
            "chat_id": session.watcher_chat_id,
            "user_id": session.watcher_user_id,
            "user_name": session.watcher_user_name,
            "thread_id": session.watcher_thread_id,
        })

    @staticmethod
    def _is_host_pid_alive(pid: Optional[int]) -> bool:
        """Best-effort liveness check for host-visible PIDs."""
        if not pid:
            return False
        try:
            os.kill(pid, 0)
            return True
        except (ProcessLookupError, PermissionError):
            return False

    def _refresh_detached_session(self, session: Optional[ProcessSession]) -> Optional[ProcessSession]:
        """Update recovered host-PID sessions when the underlying process has exited."""
        if session is None or session.exited or not session.detached or session.pid_scope != "host":
            return session

        if self._is_host_pid_alive(session.pid):
            return session

        with session._lock:
            if session.exited:
                return session
            session.exited = True
            # Recovered sessions no longer have a waitable handle, so the real
            # exit code is unavailable once the original process object is gone.
            session.exit_code = None

        self._move_to_finished(session)
        return session

    @staticmethod
    def _terminate_host_pid(pid: int) -> None:
        """Terminate a host-visible PID without requiring the original process handle."""
        if _IS_WINDOWS:
            os.kill(pid, signal.SIGTERM)
            return

        try:
            os.killpg(os.getpgid(pid), signal.SIGTERM)
        except (OSError, ProcessLookupError, PermissionError):
            os.kill(pid, signal.SIGTERM)

    # ----- Spawn -----

    @staticmethod
    def _env_temp_dir(env: Any) -> str:
        """Return the writable sandbox temp dir for env-backed background tasks."""
        get_temp_dir = getattr(env, "get_temp_dir", None)
        if callable(get_temp_dir):
            try:
                temp_dir = get_temp_dir()
                if isinstance(temp_dir, str) and temp_dir.startswith("/"):
                    return temp_dir.rstrip("/") or "/"
            except Exception as exc:
                logger.debug("Could not resolve environment temp dir: %s", exc)
        return "/tmp"

    def spawn_local(
        self,
        command: str,
        cwd: str = None,
        task_id: str = "",
        session_key: str = "",
        env_vars: dict = None,
        use_pty: bool = False,
    ) -> ProcessSession:
        """
        Spawn a background process locally.

        Only for TERMINAL_ENV=local. Other backends use spawn_via_env().

        Args:
            use_pty: If True, use a pseudo-terminal via ptyprocess for interactive
                     CLI tools (Codex, Claude Code, Python REPL). Falls back to
                     subprocess.Popen if ptyprocess is not installed.
        """
        session = ProcessSession(
            id=f"proc_{uuid.uuid4().hex[:12]}",
            command=command,
            task_id=task_id,
            session_key=session_key,
            cwd=cwd or os.getcwd(),
            started_at=time.time(),
        )

        if use_pty:
            # Try PTY mode for interactive CLI tools
            try:
                if _IS_WINDOWS:
                    from winpty import PtyProcess as _PtyProcessCls
                else:
                    from ptyprocess import PtyProcess as _PtyProcessCls
                user_shell = _find_shell()
                pty_env = _sanitize_subprocess_env(os.environ, env_vars)
                pty_env["PYTHONUNBUFFERED"] = "1"
                pty_proc = _PtyProcessCls.spawn(
                    [user_shell, "-lic", f"set +m; {command}"],
                    cwd=session.cwd,
                    env=pty_env,
                    dimensions=(30, 120),
                )
                session.pid = pty_proc.pid
                # Store the pty handle on the session for read/write
                session._pty = pty_proc

                # PTY reader thread
                reader = threading.Thread(
                    target=self._pty_reader_loop,
                    args=(session,),
                    daemon=True,
                    name=f"proc-pty-reader-{session.id}",
                )
                session._reader_thread = reader
                reader.start()

                with self._lock:
                    self._prune_if_needed()
                    self._running[session.id] = session

                self._write_checkpoint()
                return session

            except ImportError:
                logger.warning("ptyprocess not installed, falling back to pipe mode")
            except Exception as e:
                logger.warning("PTY spawn failed (%s), falling back to pipe mode", e)

        # Standard Popen path (non-PTY or PTY fallback)
        # Use the user's login shell for consistency with LocalEnvironment --
        # ensures rc files are sourced and user tools are available.
        user_shell = _find_shell()
        # Force unbuffered output for Python scripts so progress is visible
        # during background execution (libraries like tqdm/datasets buffer when
        # stdout is a pipe, hiding output from process(action="poll")).
        bg_env = _sanitize_subprocess_env(os.environ, env_vars)
        bg_env["PYTHONUNBUFFERED"] = "1"
        proc = subprocess.Popen(
            [user_shell, "-lic", f"set +m; {command}"],
            text=True,
            cwd=session.cwd,
            env=bg_env,
            encoding="utf-8",
            errors="replace",
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            stdin=subprocess.PIPE,
            preexec_fn=None if _IS_WINDOWS else os.setsid,
        )

        session.process = proc
        session.pid = proc.pid

        # Start output reader thread
        reader = threading.Thread(
            target=self._reader_loop,
            args=(session,),
            daemon=True,
            name=f"proc-reader-{session.id}",
        )
        session._reader_thread = reader
        reader.start()

        with self._lock:
            self._prune_if_needed()
            self._running[session.id] = session

        self._write_checkpoint()
        return session

    def spawn_via_env(
        self,
        env: Any,
        command: str,
        cwd: str = None,
        task_id: str = "",
        session_key: str = "",
        timeout: int = 10,
    ) -> ProcessSession:
        """
        Spawn a background process through a non-local environment backend.

        For Docker/Singularity/Modal/Daytona/SSH: runs the command inside the sandbox
        using the environment's execute() interface. We wrap the command to
        capture the in-sandbox PID and redirect output to a log file inside
        the sandbox, then poll the log via subsequent execute() calls.

        This is less capable than local spawn (no live stdout pipe, no stdin),
        but it ensures the command runs in the correct sandbox context.
        """
        session = ProcessSession(
            id=f"proc_{uuid.uuid4().hex[:12]}",
            command=command,
            task_id=task_id,
            session_key=session_key,
            cwd=cwd,
            started_at=time.time(),
            env_ref=env,
            pid_scope="sandbox",
        )

        # Run the command in the sandbox with output capture
        temp_dir = self._env_temp_dir(env)
        log_path = f"{temp_dir}/hermes_bg_{session.id}.log"
        pid_path = f"{temp_dir}/hermes_bg_{session.id}.pid"
        exit_path = f"{temp_dir}/hermes_bg_{session.id}.exit"
        quoted_command = shlex.quote(command)
        quoted_temp_dir = shlex.quote(temp_dir)
        quoted_log_path = shlex.quote(log_path)
        quoted_pid_path = shlex.quote(pid_path)
        quoted_exit_path = shlex.quote(exit_path)
        bg_command = (
            f"mkdir -p {quoted_temp_dir} && "
            f"( nohup bash -lc {quoted_command} > {quoted_log_path} 2>&1; "
            f"rc=$?; printf '%s\\n' \"$rc\" > {quoted_exit_path} ) & "
            f"echo $! > {quoted_pid_path} && cat {quoted_pid_path}"
        )

        try:
            result = env.execute(bg_command, timeout=timeout)
            output = result.get("output", "").strip()
            # Try to extract the PID from the output
            for line in output.splitlines():
                line = line.strip()
                if line.isdigit():
                    session.pid = int(line)
                    break
        except Exception as e:
            session.exited = True
            session.exit_code = -1
            session.output_buffer = f"Failed to start: {e}"

        if not session.exited:
            # Start a poller thread that periodically reads the log file
            reader = threading.Thread(
                target=self._env_poller_loop,
                args=(session, env, log_path, pid_path, exit_path),
                daemon=True,
                name=f"proc-poller-{session.id}",
            )
            session._reader_thread = reader
            reader.start()

        with self._lock:
            self._prune_if_needed()
            self._running[session.id] = session

        self._write_checkpoint()
        return session

    # ----- Reader / Poller Threads -----

    def _reader_loop(self, session: ProcessSession):
        """Background thread: read stdout from a local Popen process."""
        first_chunk = True
        try:
            while True:
                chunk = session.process.stdout.read(4096)
                if not chunk:
                    break
                if first_chunk:
                    chunk = self._clean_shell_noise(chunk)
                    first_chunk = False
                with session._lock:
                    session.output_buffer += chunk
                    if len(session.output_buffer) > session.max_output_chars:
                        session.output_buffer = session.output_buffer[-session.max_output_chars:]
                self._check_watch_patterns(session, chunk)
        except Exception as e:
            logger.debug("Process stdout reader ended: %s", e)
        finally:
            # Always reap the child to prevent zombie processes.
            try:
                session.process.wait(timeout=5)
            except Exception as e:
                logger.debug("Process wait timed out or failed: %s", e)
            session.exited = True
            session.exit_code = session.process.returncode
            self._move_to_finished(session)

    def _env_poller_loop(
        self, session: ProcessSession, env: Any, log_path: str, pid_path: str, exit_path: str
    ):
        """Background thread: poll a sandbox log file for non-local backends."""
        quoted_log_path = shlex.quote(log_path)
        quoted_pid_path = shlex.quote(pid_path)
        quoted_exit_path = shlex.quote(exit_path)
        prev_output_len = 0  # track delta for watch pattern scanning
        while not session.exited:
            time.sleep(2)  # Poll every 2 seconds
            try:
                # Read new output from the log file
                result = env.execute(f"cat {quoted_log_path} 2>/dev/null", timeout=10)
                new_output = result.get("output", "")
                if new_output:
                    # Compute delta for watch pattern scanning
                    delta = new_output[prev_output_len:] if len(new_output) > prev_output_len else ""
                    prev_output_len = len(new_output)
                    with session._lock:
                        session.output_buffer = new_output
                        if len(session.output_buffer) > session.max_output_chars:
                            session.output_buffer = session.output_buffer[-session.max_output_chars:]
                    if delta:
                        self._check_watch_patterns(session, delta)

                # Check if process is still running
                check = env.execute(
                    f"kill -0 \"$(cat {quoted_pid_path} 2>/dev/null)\" 2>/dev/null; echo $?",
                    timeout=5,
                )
                check_output = check.get("output", "").strip()
                if check_output and check_output.splitlines()[-1].strip() != "0":
                    # Process has exited -- get exit code captured by the wrapper shell.
                    exit_result = env.execute(
                        f"cat {quoted_exit_path} 2>/dev/null",
                        timeout=5,
                    )
                    exit_str = exit_result.get("output", "").strip()
                    try:
                        session.exit_code = int(exit_str.splitlines()[-1].strip())
                    except (ValueError, IndexError):
                        session.exit_code = -1
                    session.exited = True
                    self._move_to_finished(session)
                    return

            except Exception:
                # Environment might be gone (sandbox reaped, etc.)
                session.exited = True
                session.exit_code = -1
                self._move_to_finished(session)
                return

    def _pty_reader_loop(self, session: ProcessSession):
        """Background thread: read output from a PTY process."""
        pty = session._pty
        try:
            while pty.isalive():
                try:
                    chunk = pty.read(4096)
                    if chunk:
                        # ptyprocess returns bytes
                        text = chunk if isinstance(chunk, str) else chunk.decode("utf-8", errors="replace")
                        with session._lock:
                            session.output_buffer += text
                            if len(session.output_buffer) > session.max_output_chars:
                                session.output_buffer = session.output_buffer[-session.max_output_chars:]
                        self._check_watch_patterns(session, text)
                except EOFError:
                    break
                except Exception:
                    break
        except Exception as e:
            logger.debug("PTY stdout reader ended: %s", e)

        # Process exited
        try:
            pty.wait()
        except Exception as e:
            logger.debug("PTY wait timed out or failed: %s", e)
        session.exited = True
        session.exit_code = pty.exitstatus if hasattr(pty, 'exitstatus') else -1
        self._move_to_finished(session)

    def _move_to_finished(self, session: ProcessSession):
        """Move a session from running to finished.

        Idempotent: if the session was already moved (e.g. kill_process raced
        with the reader thread), the second call is a no-op — no duplicate
        completion notification is enqueued.
        """
        with self._lock:
            was_running = self._running.pop(session.id, None) is not None
            self._finished[session.id] = session
        self._write_checkpoint()

        # Only enqueue completion notification on the FIRST move.  Without
        # this guard, kill_process() and the reader thread can both call
        # _move_to_finished(), producing duplicate [SYSTEM: ...] messages.
        if was_running and session.notify_on_complete:
            from tools.ansi_strip import strip_ansi
            output_tail = strip_ansi(session.output_buffer[-2000:]) if session.output_buffer else ""
            self.completion_queue.put({
                "type": "completion",
                "session_id": session.id,
                "command": session.command,
                "exit_code": session.exit_code,
                "output": output_tail,
            })

    # ----- Query Methods -----

    def is_completion_consumed(self, session_id: str) -> bool:
        """Check if a completion notification was already consumed via wait/poll/log."""
        return session_id in self._completion_consumed

    def get(self, session_id: str) -> Optional[ProcessSession]:
        """Get a session by ID (running or finished)."""
        with self._lock:
            session = self._running.get(session_id) or self._finished.get(session_id)
        return self._refresh_detached_session(session)

    def poll(self, session_id: str) -> dict:
        """Check status and get new output for a background process."""
        from tools.ansi_strip import strip_ansi

        session = self.get(session_id)
        if session is None:
            return {"status": "not_found", "error": f"No process with ID {session_id}"}

        with session._lock:
            output_preview = strip_ansi(session.output_buffer[-1000:]) if session.output_buffer else ""

        result = {
            "session_id": session.id,
            "command": session.command,
            "status": "exited" if session.exited else "running",
            "pid": session.pid,
            "uptime_seconds": int(time.time() - session.started_at),
            "output_preview": output_preview,
        }
        if session.exited:
            result["exit_code"] = session.exit_code
            self._completion_consumed.add(session_id)
        if session.detached:
            result["detached"] = True
            result["note"] = "Process recovered after restart -- output history unavailable"
        return result

    def read_log(self, session_id: str, offset: int = 0, limit: int = 200) -> dict:
        """Read the full output log with optional pagination by lines."""
        from tools.ansi_strip import strip_ansi

        session = self.get(session_id)
        if session is None:
            return {"status": "not_found", "error": f"No process with ID {session_id}"}

        with session._lock:
            full_output = strip_ansi(session.output_buffer)

        lines = full_output.splitlines()
        total_lines = len(lines)

        # Default: last N lines
        if offset == 0 and limit > 0:
            selected = lines[-limit:]
        else:
            selected = lines[offset:offset + limit]

        result = {
            "session_id": session.id,
            "status": "exited" if session.exited else "running",
            "output": "\n".join(selected),
            "total_lines": total_lines,
            "showing": f"{len(selected)} lines",
        }
        if session.exited:
            self._completion_consumed.add(session_id)
        return result

    def wait(self, session_id: str, timeout: int = None) -> dict:
        """
        Block until a process exits, timeout, or interrupt.

        Args:
            session_id: The process to wait for.
            timeout: Max seconds to block. Falls back to TERMINAL_TIMEOUT config.

        Returns:
            dict with status ("exited", "timeout", "interrupted", "not_found")
            and output snapshot.
        """
        from tools.ansi_strip import strip_ansi
        from tools.interrupt import is_interrupted as _is_interrupted

        try:
            default_timeout = int(os.getenv("TERMINAL_TIMEOUT", "180"))
        except (ValueError, TypeError):
            default_timeout = 180
        max_timeout = default_timeout
        requested_timeout = timeout
        timeout_note = None

        if requested_timeout and requested_timeout > max_timeout:
            effective_timeout = max_timeout
            timeout_note = (
                f"Requested wait of {requested_timeout}s was clamped "
                f"to configured limit of {max_timeout}s"
            )
        else:
            effective_timeout = requested_timeout or max_timeout

        session = self.get(session_id)
        if session is None:
            return {"status": "not_found", "error": f"No process with ID {session_id}"}

        deadline = time.monotonic() + effective_timeout

        while time.monotonic() < deadline:
            session = self._refresh_detached_session(session)
            if session.exited:
                self._completion_consumed.add(session_id)
                result = {
                    "status": "exited",
                    "exit_code": session.exit_code,
                    "output": strip_ansi(session.output_buffer[-2000:]),
                }
                if timeout_note:
                    result["timeout_note"] = timeout_note
                return result

            if _is_interrupted():
                result = {
                    "status": "interrupted",
                    "output": strip_ansi(session.output_buffer[-1000:]),
                    "note": "User sent a new message -- wait interrupted",
                }
                if timeout_note:
                    result["timeout_note"] = timeout_note
                return result

            time.sleep(1)

        result = {
            "status": "timeout",
            "output": strip_ansi(session.output_buffer[-1000:]),
        }
        if timeout_note:
            result["timeout_note"] = timeout_note
        else:
            result["timeout_note"] = f"Waited {effective_timeout}s, process still running"
        return result

    def kill_process(self, session_id: str) -> dict:
        """Kill a background process."""
        session = self.get(session_id)
        if session is None:
            return {"status": "not_found", "error": f"No process with ID {session_id}"}

        if session.exited:
            return {
                "status": "already_exited",
                "exit_code": session.exit_code,
            }

        # Kill via PTY, Popen (local), or env execute (non-local)
        try:
            if session._pty:
                # PTY process -- terminate via ptyprocess
                try:
                    session._pty.terminate(force=True)
                except Exception:
                    if session.pid:
                        os.kill(session.pid, signal.SIGTERM)
            elif session.process:
                # Local process -- kill the process group
                try:
                    if _IS_WINDOWS:
                        session.process.terminate()
                    else:
                        os.killpg(os.getpgid(session.process.pid), signal.SIGTERM)
                except (ProcessLookupError, PermissionError):
                    session.process.kill()
            elif session.env_ref and session.pid:
                # Non-local -- kill inside sandbox
                session.env_ref.execute(f"kill {session.pid} 2>/dev/null", timeout=5)
            elif session.detached and session.pid_scope == "host" and session.pid:
                if not self._is_host_pid_alive(session.pid):
                    with session._lock:
                        session.exited = True
                        session.exit_code = None
                    self._move_to_finished(session)
                    return {
                        "status": "already_exited",
                        "exit_code": session.exit_code,
                    }
                self._terminate_host_pid(session.pid)
            else:
                return {
                    "status": "error",
                    "error": (
                        "Recovered process cannot be killed after restart because "
                        "its original runtime handle is no longer available"
                    ),
                }
            session.exited = True
            session.exit_code = -15  # SIGTERM
            self._move_to_finished(session)
            self._write_checkpoint()
            return {"status": "killed", "session_id": session.id}
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def write_stdin(self, session_id: str, data: str) -> dict:
        """Send raw data to a running process's stdin (no newline appended)."""
        session = self.get(session_id)
        if session is None:
            return {"status": "not_found", "error": f"No process with ID {session_id}"}
        if session.exited:
            return {"status": "already_exited", "error": "Process has already finished"}

        # PTY mode -- write through pty handle (expects bytes)
        if hasattr(session, '_pty') and session._pty:
            try:
                pty_data = data.encode("utf-8") if isinstance(data, str) else data
                session._pty.write(pty_data)
                return {"status": "ok", "bytes_written": len(data)}
            except Exception as e:
                return {"status": "error", "error": str(e)}

        # Popen mode -- write through stdin pipe
        if not session.process or not session.process.stdin:
            return {"status": "error", "error": "Process stdin not available (non-local backend or stdin closed)"}
        try:
            session.process.stdin.write(data)
            session.process.stdin.flush()
            return {"status": "ok", "bytes_written": len(data)}
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def submit_stdin(self, session_id: str, data: str = "") -> dict:
        """Send data + newline to a running process's stdin (like pressing Enter)."""
        return self.write_stdin(session_id, data + "\n")

    def close_stdin(self, session_id: str) -> dict:
        """Close a running process's stdin / send EOF without killing the process."""
        session = self.get(session_id)
        if session is None:
            return {"status": "not_found", "error": f"No process with ID {session_id}"}
        if session.exited:
            return {"status": "already_exited", "error": "Process has already finished"}

        if hasattr(session, '_pty') and session._pty:
            try:
                session._pty.sendeof()
                return {"status": "ok", "message": "EOF sent"}
            except Exception as e:
                return {"status": "error", "error": str(e)}

        if not session.process or not session.process.stdin:
            return {"status": "error", "error": "Process stdin not available (non-local backend or stdin closed)"}
        try:
            session.process.stdin.close()
            return {"status": "ok", "message": "stdin closed"}
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def list_sessions(self, task_id: str = None) -> list:
        """List all running and recently-finished processes."""
        with self._lock:
            all_sessions = list(self._running.values()) + list(self._finished.values())

        all_sessions = [self._refresh_detached_session(s) for s in all_sessions]

        if task_id:
            all_sessions = [s for s in all_sessions if s.task_id == task_id]

        result = []
        for s in all_sessions:
            entry = {
                "session_id": s.id,
                "command": s.command[:200],
                "cwd": s.cwd,
                "pid": s.pid,
                "started_at": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(s.started_at)),
                "uptime_seconds": int(time.time() - s.started_at),
                "status": "exited" if s.exited else "running",
                "output_preview": s.output_buffer[-200:] if s.output_buffer else "",
            }
            if s.exited:
                entry["exit_code"] = s.exit_code
            if s.detached:
                entry["detached"] = True
            result.append(entry)
        return result

    # ----- Session/Task Queries (for gateway integration) -----

    def has_active_processes(self, task_id: str) -> bool:
        """Check if there are active (running) processes for a task_id."""
        with self._lock:
            sessions = list(self._running.values())

        for session in sessions:
            self._refresh_detached_session(session)

        with self._lock:
            return any(
                s.task_id == task_id and not s.exited
                for s in self._running.values()
            )

    def has_active_for_session(self, session_key: str) -> bool:
        """Check if there are active processes for a gateway session key."""
        with self._lock:
            sessions = list(self._running.values())

        for session in sessions:
            self._refresh_detached_session(session)

        with self._lock:
            return any(
                s.session_key == session_key and not s.exited
                for s in self._running.values()
            )

    def kill_all(self, task_id: str = None) -> int:
        """Kill all running processes, optionally filtered by task_id. Returns count killed."""
        with self._lock:
            targets = [
                s for s in self._running.values()
                if (task_id is None or s.task_id == task_id) and not s.exited
            ]

        killed = 0
        for session in targets:
            result = self.kill_process(session.id)
            if result.get("status") in ("killed", "already_exited"):
                killed += 1
        return killed

    # ----- Cleanup / Pruning -----

    def _prune_if_needed(self):
        """Remove oldest finished sessions if over MAX_PROCESSES. Must hold _lock."""
        # First prune expired finished sessions
        now = time.time()
        expired = [
            sid for sid, s in self._finished.items()
            if (now - s.started_at) > FINISHED_TTL_SECONDS
        ]
        for sid in expired:
            del self._finished[sid]
            self._completion_consumed.discard(sid)

        # If still over limit, remove oldest finished
        total = len(self._running) + len(self._finished)
        if total >= MAX_PROCESSES and self._finished:
            oldest_id = min(self._finished, key=lambda sid: self._finished[sid].started_at)
            del self._finished[oldest_id]
            self._completion_consumed.discard(oldest_id)

        # Drop any _completion_consumed entries whose sessions are no longer
        # tracked at all — belt-and-suspenders against module-lifetime growth
        # on process-registry lookup paths that don't reach the dict prunes.
        tracked = self._running.keys() | self._finished.keys()
        stale = self._completion_consumed - tracked
        if stale:
            self._completion_consumed -= stale

    # ----- Checkpoint (crash recovery) -----

    def _write_checkpoint(self):
        """Write running process metadata to checkpoint file atomically."""
        try:
            with self._lock:
                entries = []
                for s in self._running.values():
                    if not s.exited:
                        entries.append({
                            "session_id": s.id,
                            "command": s.command,
                            "pid": s.pid,
                            "pid_scope": s.pid_scope,
                            "cwd": s.cwd,
                            "started_at": s.started_at,
                            "task_id": s.task_id,
                            "session_key": s.session_key,
                            "watcher_platform": s.watcher_platform,
                            "watcher_chat_id": s.watcher_chat_id,
                            "watcher_user_id": s.watcher_user_id,
                            "watcher_user_name": s.watcher_user_name,
                            "watcher_thread_id": s.watcher_thread_id,
                            "watcher_interval": s.watcher_interval,
                            "notify_on_complete": s.notify_on_complete,
                            "watch_patterns": s.watch_patterns,
                        })
            
            # Atomic write to avoid corruption on crash
            from utils import atomic_json_write
            atomic_json_write(CHECKPOINT_PATH, entries)
        except Exception as e:
            logger.debug("Failed to write checkpoint file: %s", e, exc_info=True)

    def recover_from_checkpoint(self) -> int:
        """
        On gateway startup, probe PIDs from checkpoint file.

        Returns the number of processes recovered as detached.
        """
        if not CHECKPOINT_PATH.exists():
            return 0

        try:
            entries = json.loads(CHECKPOINT_PATH.read_text(encoding="utf-8"))
        except Exception:
            return 0

        recovered = 0
        for entry in entries:
            pid = entry.get("pid")
            if not pid:
                continue

            pid_scope = entry.get("pid_scope", "host")
            if pid_scope != "host":
                # Sandbox-backed processes keep only in-sandbox PIDs in the
                # checkpoint, which are not meaningful to the restarted host
                # process once the original environment handle is gone.
                logger.info(
                    "Skipping recovery for non-host process: %s (pid=%s, scope=%s)",
                    entry.get("command", "unknown")[:60],
                    pid,
                    pid_scope,
                )
                continue

            # Check if PID is still alive
            alive = self._is_host_pid_alive(pid)

            if alive:
                session = ProcessSession(
                    id=entry["session_id"],
                    command=entry.get("command", "unknown"),
                    task_id=entry.get("task_id", ""),
                    session_key=entry.get("session_key", ""),
                    pid=pid,
                    pid_scope=pid_scope,
                    cwd=entry.get("cwd"),
                    started_at=entry.get("started_at", time.time()),
                    detached=True,  # Can't read output, but can report status + kill
                    watcher_platform=entry.get("watcher_platform", ""),
                    watcher_chat_id=entry.get("watcher_chat_id", ""),
                    watcher_user_id=entry.get("watcher_user_id", ""),
                    watcher_user_name=entry.get("watcher_user_name", ""),
                    watcher_thread_id=entry.get("watcher_thread_id", ""),
                    watcher_interval=entry.get("watcher_interval", 0),
                    notify_on_complete=entry.get("notify_on_complete", False),
                    watch_patterns=entry.get("watch_patterns", []),
                )
                with self._lock:
                    self._running[session.id] = session
                recovered += 1
                logger.info("Recovered detached process: %s (pid=%d)", session.command[:60], pid)

                # Re-enqueue watcher so gateway can resume notifications
                if session.watcher_interval > 0:
                    self.pending_watchers.append({
                        "session_id": session.id,
                        "check_interval": session.watcher_interval,
                        "session_key": session.session_key,
                        "platform": session.watcher_platform,
                        "chat_id": session.watcher_chat_id,
                        "user_id": session.watcher_user_id,
                        "user_name": session.watcher_user_name,
                        "thread_id": session.watcher_thread_id,
                        "notify_on_complete": session.notify_on_complete,
                    })

        self._write_checkpoint()

        return recovered


# Module-level singleton
process_registry = ProcessRegistry()


# ---------------------------------------------------------------------------
# Registry -- the "process" tool schema + handler
# ---------------------------------------------------------------------------
from tools.registry import registry, tool_error

PROCESS_SCHEMA = {
    "name": "process",
    "description": (
        "Manage background processes started with terminal(background=true). "
        "Actions: 'list' (show all), 'poll' (check status + new output), "
        "'log' (full output with pagination), 'wait' (block until done or timeout), "
        "'kill' (terminate), 'write' (send raw stdin data without newline), "
        "'submit' (send data + Enter, for answering prompts), 'close' (close stdin/send EOF)."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["list", "poll", "log", "wait", "kill", "write", "submit", "close"],
                "description": "Action to perform on background processes"
            },
            "session_id": {
                "type": "string",
                "description": "Process session ID (from terminal background output). Required for all actions except 'list'."
            },
            "data": {
                "type": "string",
                "description": "Text to send to process stdin (for 'write' and 'submit' actions)"
            },
            "timeout": {
                "type": "integer",
                "description": "Max seconds to block for 'wait' action. Returns partial output on timeout.",
                "minimum": 1
            },
            "offset": {
                "type": "integer",
                "description": "Line offset for 'log' action (default: last 200 lines)"
            },
            "limit": {
                "type": "integer",
                "description": "Max lines to return for 'log' action",
                "minimum": 1
            }
        },
        "required": ["action"]
    }
}


def _handle_process(args, **kw):
    task_id = kw.get("task_id")
    action = args.get("action", "")
    # Coerce to string — some models send session_id as an integer
    session_id = str(args.get("session_id", "")) if args.get("session_id") is not None else ""

    if action == "list":
        return json.dumps({"processes": process_registry.list_sessions(task_id=task_id)}, ensure_ascii=False)
    elif action in ("poll", "log", "wait", "kill", "write", "submit", "close"):
        if not session_id:
            return tool_error(f"session_id is required for {action}")
        if action == "poll":
            return json.dumps(process_registry.poll(session_id), ensure_ascii=False)
        elif action == "log":
            return json.dumps(process_registry.read_log(
                session_id, offset=args.get("offset", 0), limit=args.get("limit", 200)), ensure_ascii=False)
        elif action == "wait":
            return json.dumps(process_registry.wait(session_id, timeout=args.get("timeout")), ensure_ascii=False)
        elif action == "kill":
            return json.dumps(process_registry.kill_process(session_id), ensure_ascii=False)
        elif action == "write":
            return json.dumps(process_registry.write_stdin(session_id, str(args.get("data", ""))), ensure_ascii=False)
        elif action == "submit":
            return json.dumps(process_registry.submit_stdin(session_id, str(args.get("data", ""))), ensure_ascii=False)
        elif action == "close":
            return json.dumps(process_registry.close_stdin(session_id), ensure_ascii=False)
    return tool_error(f"Unknown process action: {action}. Use: list, poll, log, wait, kill, write, submit, close")


registry.register(
    name="process",
    toolset="terminal",
    schema=PROCESS_SCHEMA,
    handler=_handle_process,
    emoji="⚙️",
)
