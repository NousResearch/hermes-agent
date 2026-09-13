"""In-memory process-session state used by the process registry."""
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
import threading
import time
import subprocess

MAX_OUTPUT_CHARS = 200_000


@dataclass
class ProcessSession:
    """A tracked background process with output buffering."""
    id: str                                     # "proc_xxxxxxxxxxxx"
    command: str
    task_id: str = ""                           # Task/sandbox isolation key (CONTAINER key,
                                                # may be collapsed by _resolve_container_task_id)
    owner_task_id: str = ""                     # RAW spawning task id ("sa-..."); ownership
                                                # checks must use this, not task_id
    session_key: str = ""                       # Gateway session key (reset protection)
    pid: Optional[int] = None
    process: Optional[subprocess.Popen] = None  # Popen handle (local only)
    env_ref: Any = None                         # Environment object (sandbox spawns)
    cwd: Optional[str] = None
    started_at: float = 0.0                     # time.time() of spawn
    host_start_time: Optional[int] = None       # kernel start ticks (/proc/<pid>/stat f22) — PID-reuse guard
    exited: bool = False
    exit_code: Optional[int] = None             # None while running
    completion_reason: str = "exited"           # exited|killed|lost|failed_start|already_exited
    termination_source: str = ""                # process.kill|kill_all|backend_lost|failed_start
    output_buffer: str = ""                     # Rolling tail (last max_output_chars)
    max_output_chars: int = MAX_OUTPUT_CHARS
    detached: bool = False                      # Recovered from checkpoint (no pipe)
    pid_scope: str = "host"                     # "host" for local/PTY PIDs, "sandbox" for env-local PIDs
    systemd_unit: str = ""                      # transient scope unit name when spawned under systemd-run
    handoff_note: str = ""                      # why a subagent handed this process to its parent (rides the notice)
    # Watcher/notification routing (persisted for crash recovery)
    # systemd_unit: str = ""                      # transient scope unit name when spawned under systemd-run
    # (#70716)
    watcher_platform: str = ""
    watcher_chat_id: str = ""
    watcher_user_id: str = ""
    watcher_user_name: str = ""
    watcher_thread_id: str = ""
    watcher_message_id: str = ""                # Triggering message id — reply anchor for topic routing
    watcher_interval: int = 0                   # 0 = no watcher configured
    # Session-db id of the spawning conversation; lets the gateway drop completions whose
    # session was closed at a user boundary (/new) instead of injecting into the NEW one.
    parent_session_id: str = ""
    notify_on_complete: bool = False            # Queue agent notification on exit
    watch_patterns: List[str] = field(default_factory=list)
    _watch_hits: int = field(default=0, repr=False)          # total matches delivered
    _watch_suppressed: int = field(default=0, repr=False)    # matches dropped by rate limit
    _watch_disabled: bool = field(default=False, repr=False) # permanently killed after strike limit
    # Rate-limit window state (see WATCH_*). A strike is a WINDOW with drops, not a drop.
    _watch_cooldown_until: float = field(default=0.0, repr=False)
    _watch_strike_candidate: bool = field(default=False, repr=False)
    _watch_consecutive_strikes: int = field(default=0, repr=False)
    _completion_event: threading.Event = field(default_factory=threading.Event, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock)
    _reader_thread: Optional[threading.Thread] = field(default=None, repr=False)
    _pty: Any = field(default=None, repr=False)  # ptyprocess handle (use_pty=True)

    def append_output(self, text: str) -> None:
        """Append to the rolling output buffer under the session lock, keeping the tail."""
        with self._lock:
            self.output_buffer += text
            if len(self.output_buffer) > self.max_output_chars:
                self.output_buffer = self.output_buffer[-self.max_output_chars:]

    def mark_exited(self, exit_code, reason: str = "exited", source: str = "") -> None:
        """Record an exit. A kill that raced the observer already recorded its own
        exit_code/reason; never overwrite it."""
        self.exited = True
        if self.completion_reason != "killed":
            self.exit_code = exit_code
            self.completion_reason = reason
            if source:
                self.termination_source = source

from tools import process_registry as _registry
