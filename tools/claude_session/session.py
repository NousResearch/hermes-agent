"""session.py — Simplified Claude Code session as a context pipeline.

Replaces the former manager.py. Key simplifications:
- No state machine object — state detected fresh each poll
- No turn/tool-call tracking — just "output since last send"
- No auto-responder — that's a separate concern
- wait_for_idle uses inline polling

Core flow:
    TaskContext → format prompt → write to tmux → wait for idle → return result
"""

import json
import logging
import os
import re
import shlex
import subprocess
import threading
import time
import uuid
from typing import Optional

from tools.claude_session.idle import (
    SessionState, clean_lines, detect_state, detect_startup_scene,
    detect_activity, is_permission_in_text,
    _STATUS_BAR_RE, _PERMISSION_RE, _DONE_TIME_RE,
    _INTERVIEW_NAV_RE, _INTERVIEW_OPTION_RE, _INTERVIEW_SECTION_RE,
)
from tools.claude_session.output_buffer import OutputBuffer
from tools.claude_session.tmux_interface import TmuxInterface
from tools.claude_session.task_context import TaskContext
from tools.claude_session.observer import SessionObserver
from tools.claude_session.status_card import StatusCard
from tools.claude_session.errors import (
    SessionError, TmuxError, TmuxNotFoundError, TmuxTimeoutError,
    SessionDisconnectedError, SessionNotActiveError, StartupFailedError,
    SessionExitedError, PermissionError, InvalidPermissionResponseError,
    WaitTimeoutError, StallDetectedError, ValidationError,
    wrap_tmux_error,
)

logger = logging.getLogger(__name__)

PASTE_SUBMIT_DELAY_SECONDS = 2.0
_PASTE_POLL_INTERVAL = 0.1
JSONL_RESUME_SIZE_LIMIT = 20 * 1024 * 1024  # 20MB — skip auto-resume for oversized JSONL


class _StateView:
    """Lightweight shim for backward compatibility with mgr._sm.current_state."""

    __slots__ = ("_session",)

    def __init__(self, session: "ClaudeSession"):
        self._session = session

    @property
    def current_state(self) -> str:
        return self._session._state

    def state_duration(self) -> float:
        return time.monotonic() - self._session._state_entered


class ClaudeSession:
    """Context pipeline: TaskContext → Claude Code → Result.

    All instance state, naturally supports parallel sessions.
    """

    _PERMISSION_PROMPT_RE = re.compile(
        r"(Allow\?|Yes.*No|permission to|wants to|proceed\?|"
        r"❯\s*\d+\.\s*(Yes|Allow))",
        re.IGNORECASE,
    )

    def __init__(self):
        self._session_id: Optional[str] = None
        self._tmux: Optional[TmuxInterface] = None
        self._buf = OutputBuffer(max_lines=1000)
        self._session_active = False
        self._permission_mode = "normal"
        self._claude_session_uuid: Optional[str] = None
        self._session_start_time: Optional[float] = None
        self._workdir: Optional[str] = None

        # Simplified state tracking (no StateMachine object)
        self._state: str = SessionState.DISCONNECTED
        self._state_entered: float = time.monotonic()

        # Output tracking (replaces Turn tracking)
        self._send_marker: int = 0

        # Threading
        self._lock = threading.RLock()  # RLock: allows reentrant locking from same thread
        self._state_event = threading.Event()
        self._initializing = False

        # Gateway session isolation
        self._gateway_session_key: str = ""
        self._session_name: Optional[str] = None
        self._model: Optional[str] = None

        # Prevent double permission response
        self._permission_responded = False
        self._permission_auto_allow_low_risk = True  # auto-allow low-risk in normal mode
        self._in_auto_approve = False  # guard against recursive _auto_approve_permission
        self._startup_error_count = 0  # Track ERROR/EXITED detections during startup

        # Initialization grace period — don't trust observer non-IDLE states during startup
        self._session_ready_time: Optional[float] = None  # None until fully initialized
        self._startup_grace_period: float = 5.0  # seconds

        # Status callback
        self._status_callback = None

        # Send receipt tracking (Layer 1: echo, Layer 2: wait_for_idle receipt)
        self._send_time: Optional[float] = None
        self._send_seq: int = 0
        self._send_needs_receipt: bool = False

        # Observer (optional side-channel)
        self._observer: Optional[SessionObserver] = None
        # Observer poll interval (default 5s — tight enough to catch short tool calls
        # like Bash/Read/Write which complete in 1-10s; 180s would miss them entirely)
        self._observer_poll_interval: float = float(os.environ.get("HERMES_CLAUDE_SESSION_OBSERVER_POLL_INTERVAL", "5"))

        # Status card (optional Telegram real-time status)
        self._status_card: Optional[StatusCard] = None

        # Output streamer (optional real-time output to chat)
        self._output_streamer = None

    # ------------------------------------------------------------------
    # Backward compatibility shim
    # ------------------------------------------------------------------

    @property
    def _sm(self) -> _StateView:
        return _StateView(self)

    def _update_state(self, new_state: str) -> None:
        changed = False
        with self._lock:
            if new_state != self._state:
                old = self._state
                self._state = new_state
                self._state_entered = time.monotonic()
                self._state_event.set()
                changed = True
        if changed:
            logger.debug("State: %s → %s", old, new_state)

    # ------------------------------------------------------------------
    # Session lifecycle
    # ------------------------------------------------------------------

    def start(
        self,
        workdir: str,
        session_name: str = "hermes-default",
        model: Optional[str] = None,
        permission_mode: str = "normal",
        on_event: str = "notify",
        completion_queue=None,
        resume_uuid: Optional[str] = None,
        force_new: bool = False,
        auto_responder: bool = False,  # Accepted for API compat, no-op in pipeline architecture
        auto_responder_config: Optional[dict] = None,  # Accepted for API compat, no-op
        status_card_config: Optional[dict] = None,
    ) -> dict:
        """Start a Claude Code session in tmux."""
        # Phase 0: fast validation (under lock, no I/O)
        with self._lock:
            if self._session_active or self._initializing:
                return {
                    "session_id": self._session_id,
                    "tmux_session": session_name,
                    "state": self._state,
                    "permission_mode": self._permission_mode,
                    "claude_session_uuid": self._claude_session_uuid,
                    "note": "Session already active",
                }

            if permission_mode not in ("normal", "skip"):
                raise ValidationError(f"Invalid permission_mode: {permission_mode}")

            self._claude_session_uuid = str(uuid.uuid5(
                uuid.NAMESPACE_DNS,
                f"hermes-session:{self._gateway_session_key}:{self._session_name}:{os.path.abspath(workdir)}"
            ))

            # Auto-detect resume: if same name+workdir had a previous session, resume it.
            auto_resume_uuid = None
            if not resume_uuid and not force_new:
                jsonl_path = self._find_session_jsonl(workdir, self._claude_session_uuid)
                if jsonl_path:
                    jsonl_size = os.path.getsize(jsonl_path)
                    if jsonl_size <= JSONL_RESUME_SIZE_LIMIT:
                        auto_resume_uuid = self._claude_session_uuid
                    else:
                        logger.info(
                            "JSONL too large (%d MB) for session %s, starting fresh",
                            jsonl_size // 1024 // 1024, self._claude_session_uuid[:8],
                        )

            actually_resuming = False
            effective_resume = resume_uuid or auto_resume_uuid
            if effective_resume:
                jsonl_path = self._find_session_jsonl(workdir, effective_resume)
                if jsonl_path:
                    actually_resuming = True
                else:
                    logger.warning("resume_uuid=%s history not found, starting new", effective_resume)

            if actually_resuming:
                self._state = SessionState.DISCONNECTED
                self._buf.clear()
                self._send_marker = 0

            self._session_id = f"cs_{uuid.uuid4().hex[:8]}"
            if self._session_name is None:
                self._session_name = session_name
            self._permission_mode = permission_mode
            self._workdir = os.path.abspath(workdir)
            self._tmux = TmuxInterface(session_name)
            self._initializing = True

        # Phase 1: tmux I/O (no lock, don't block other threads)
        try:
            needs_init = False

            if not self._tmux.session_exists():
                self._tmux.create_session(workdir=workdir)
                needs_init = True
            else:
                pane = self._tmux.capture_pane(lines=50)
                pane_lower = pane.lower()

                cwd_check = subprocess.run(
                    ["tmux", "display-message", "-t", session_name,
                     "-p", "#{pane_current_path}"],
                    capture_output=True, text=True, timeout=5,
                )
                tmux_cwd = cwd_check.stdout.strip() if cwd_check.returncode == 0 else ""

                needs_rebuild = False
                if tmux_cwd and workdir not in tmux_cwd and tmux_cwd not in workdir:
                    needs_rebuild = True
                elif "claude code" in pane_lower or "claude-" in pane_lower:
                    if workdir.lower() not in pane_lower:
                        needs_rebuild = True

                if needs_rebuild:
                    logger.warning("tmux session %s needs rebuild", session_name)
                    self._tmux.kill_session()
                    time.sleep(0.5)
                    self._tmux.create_session(workdir=workdir)
                    needs_init = True
                else:
                    pane_lines = clean_lines(pane)
                    result = detect_state(pane_lines)
                    if result.state == SessionState.IDLE:
                        logger.info("Reusing existing IDLE session %s", session_name)
                    elif result.state == SessionState.EXITED:
                        logger.info("Session %s EXITED (Claude gone), reinitializing", session_name)
                        needs_init = True
                    else:
                        # Wait for busy session to become IDLE before rebuilding
                        _BUSY_WAIT_TIMEOUT = 30  # seconds
                        _BUSY_WAIT_INTERVAL = 3
                        logger.warning("Session %s in %s state, waiting up to %ds to settle",
                                       session_name, result.state, _BUSY_WAIT_TIMEOUT)
                        _waited = 0
                        _settled = False
                        while _waited < _BUSY_WAIT_TIMEOUT:
                            time.sleep(_BUSY_WAIT_INTERVAL)
                            _waited += _BUSY_WAIT_INTERVAL
                            pane = self._tmux.capture_pane(lines=50)
                            pane_lines = clean_lines(pane)
                            check = detect_state(pane_lines)
                            if check.state == SessionState.IDLE:
                                logger.info("Session %s settled to IDLE after %ds", session_name, _waited)
                                _settled = True
                                break
                            elif check.state == SessionState.EXITED:
                                logger.info("Session %s EXITED while waiting", session_name)
                                needs_init = True
                                _settled = True
                                break
                        if not _settled:
                            logger.warning("Session %s still %s after %ds, forcing rebuild",
                                           session_name, result.state, _BUSY_WAIT_TIMEOUT)
                            self._tmux.kill_session()
                            time.sleep(0.5)
                            self._tmux.create_session(workdir=workdir)
                            needs_init = True

            if needs_init:
                current_uid = os.getuid()
                if current_uid == 0:
                    non_root_user = os.environ.get("SUDO_USER") or os.environ.get("USER", "user")
                    self._tmux.send_keys(f"su - {shlex.quote(non_root_user)}", enter=True)
                    time.sleep(1.5)
                    self._tmux.send_keys(f"cd {shlex.quote(workdir)}", enter=True)
                    time.sleep(0.5)
                    user_bin = f"/home/{non_root_user}/bin"
                    self._tmux.send_keys(f"export PATH={shlex.quote(user_bin)}:$PATH", enter=True)
                    time.sleep(0.5)

                claude_cmd = "claude"
                if actually_resuming:
                    claude_cmd += f" --resume {shlex.quote(effective_resume)}"
                else:
                    claude_cmd += f" --session-id {shlex.quote(self._claude_session_uuid)}"
                if permission_mode == "skip":
                    claude_cmd += " --permission-mode bypassPermissions"
                    claude_cmd = "CLAUDE_CODE_TRUST_WORKSPACE=1 " + claude_cmd
                if model:
                    claude_cmd += f" --model {shlex.quote(model)}"
                self._tmux.send_keys(claude_cmd, enter=True)
                time.sleep(2)

                if permission_mode == "skip":
                    time.sleep(1)
                    pane = self._tmux.capture_pane()
                    # Only match the actual permission prompt, NOT the welcome screen
                    # or status bar which also contain "bypass permissions" text.
                    # The prompt asks "Do you want to proceed?" or shows permission choices.
                    is_permission_prompt = (
                        ("permission" in pane.lower() and ("proceed" in pane.lower() or "allow" in pane.lower() or "yes" in pane.lower()))
                        or ("Do you want to proceed" in pane)
                    )
                    if is_permission_prompt:
                        self._tmux.send_special_key("Down")
                        time.sleep(0.3)
                        self._tmux.send_special_key("Enter")
                        time.sleep(1)

                STARTUP_HEALTH_TIMEOUT = 60
                if not self._wait_for_claude_startup(STARTUP_HEALTH_TIMEOUT, is_resume=actually_resuming):
                    logger.error("Claude Code failed to start in %s", session_name)
                    try:
                        self._tmux.kill_session()
                    except Exception:
                        pass
                    with self._lock:
                        self._initializing = False
                    raise StartupFailedError(
                        f"Claude Code did not start within {STARTUP_HEALTH_TIMEOUT}s.",
                        detail=f"session={session_name}",
                    )

        except SessionError:
            with self._lock:
                self._initializing = False
            if self._tmux:
                try:
                    self._tmux.kill_session()
                except Exception:
                    pass
            raise
        except Exception as e:
            with self._lock:
                self._initializing = False
            if self._tmux:
                try:
                    self._tmux.kill_session()
                except Exception:
                    pass
            raise wrap_tmux_error(e) from e

        # Phase 2: finalize (under lock)
        with self._lock:
            self._initializing = False
            self._session_active = True
            self._session_start_time = time.monotonic()
            self._session_ready_time = time.monotonic()  # Mark session as fully initialized
            self._model = model  # Persist for restart detection
            self._update_state(SessionState.IDLE)

            # Start status card first (sets self._status_callback for observer)
            if status_card_config:
                self._start_status_card(status_card_config)

            # Start observer (uses self._status_callback set above)
            self._observer = SessionObserver(
                tmux=self._tmux,
                buffer=self._buf,
                on_update=self._on_observer_update if self._status_callback else None,
                poll_interval=self._observer_poll_interval,
            )
            self._observer.start()

        # Synchronous initial poll
        try:
            if self._observer:
                self._observer.poll_now()
                # Refresh state from buffer
                pane = self._tmux.capture_pane()
                lines = clean_lines(pane)
                result = detect_state(lines)
                self._update_state(result.state)
        except Exception as e:
            logger.warning("Initial poll failed: %s", e)

        with self._lock:
            result = {
                "session_id": self._session_id,
                "tmux_session": session_name,
                "state": self._state,
                "permission_mode": self._permission_mode,
                "claude_session_uuid": self._claude_session_uuid,
            }
            if actually_resuming:
                result["resumed_from"] = effective_resume
                if auto_resume_uuid:
                    result["auto_resumed"] = True
            elif resume_uuid:
                result["fallback_note"] = f"resume_uuid={resume_uuid} history not found"
            return result

    def stop(self) -> dict:
        """Stop the session and clean up."""
        with self._lock:
            if not self._session_active:
                raise SessionNotActiveError("No active session")

            if self._observer:
                self._observer.stop()

            if self._status_card:
                self._status_card.stop()
                self._status_card = None

            if self._tmux:
                try:
                    self._tmux.kill_session()
                except Exception as e:
                    logger.warning("Failed to kill tmux: %s", e)

            sid = self._session_id
            uuid_to_return = self._claude_session_uuid
            self._session_active = False
            self._session_id = None
            self._claude_session_uuid = None
            self._session_start_time = None
            self._observer = None
            self._update_state(SessionState.DISCONNECTED)

            return {
                "stopped": True,
                "session_id": sid,
                "claude_session_uuid": uuid_to_return,
            }

    # ------------------------------------------------------------------
    # Send operations
    # ------------------------------------------------------------------

    def send(self, message_or_task) -> dict:
        """Send a message or TaskContext to Claude.

        Args:
            message_or_task: str (raw message) or TaskContext (structured)
        """
        if isinstance(message_or_task, TaskContext):
            message = message_or_task.to_prompt()
        else:
            message = str(message_or_task)

        return self._send_text(message)

    def send_text(self, text: str) -> dict:
        """Type text and submit atomically."""
        return self._send_text(text)

    def type_text(self, text: str) -> dict:
        """Type text without pressing Enter."""
        with self._lock:
            if not self._session_active:
                raise SessionNotActiveError("No active session")
            if self._state in (SessionState.EXITED, SessionState.DISCONNECTED):
                raise SessionExitedError("Claude Code has exited.")
            self._tmux.send_keys(text)
            return {"typed": True, "state": self._state}

    def submit(self) -> dict:
        """Submit typed text by pressing Enter."""
        with self._lock:
            if not self._session_active:
                raise SessionNotActiveError("No active session")
            if self._state in (SessionState.EXITED, SessionState.DISCONNECTED):
                raise SessionExitedError("Claude Code has exited.")
            self._tmux.send_special_key("Enter")
        return {"submitted": True, "state": self._state}

    def cancel_input(self) -> dict:
        """Cancel current input (Ctrl+C)."""
        logger.warning("cancel_input for session %s", self._session_id)
        with self._lock:
            if not self._session_active:
                raise SessionNotActiveError("No active session")
            self._tmux.send_special_key("C-c")
            return {"cancelled": True, "state": self._state}

    def _send_text(self, text: str) -> dict:
        """Internal: send text to tmux, handling multi-line.

        Layer 1 verification: lightweight echo check (pane content changed).
        Does NOT retry. Claude-level receipt verification happens in
        wait_for_idle's send receipt phase (Layer 2).
        """
        is_multiline = "\n" in text

        with self._lock:
            if not self._session_active:
                raise SessionNotActiveError("No active session")
            if not self._tmux:
                raise TmuxError("No tmux interface")
            if self._state in (SessionState.EXITED, SessionState.DISCONNECTED):
                raise SessionExitedError("Claude Code has exited.")

            self._send_marker = self._buf.total_count()
            self._send_seq += 1
            self._send_time = time.monotonic()
            self._send_needs_receipt = True

            if is_multiline:
                self._tmux.send_keys(text, enter=False)
            else:
                self._tmux.send_keys(text, enter=True)

        # Multi-line: wait for bracketed paste to land, then submit
        if is_multiline:
            self._wait_for_paste_then_submit()

        # Layer 1: echo check — did tmux pane content change?
        echo_ok = self._verify_echo(timeout=1.5)

        # Refresh state
        self._refresh_state()
        result = {
            "sent": True,
            "state": self._state,
            "send_seq": self._send_seq,
        }
        if not echo_ok:
            result["echo_status"] = "no_echo"
            result["echo_hint"] = "Paste may not have reached tmux pane"
        else:
            result["echo_status"] = "echo_detected"
        return result

    def _verify_echo(self, timeout: float = 1.5) -> bool:
        """Check that tmux pane content changed after send.

        Only verifies tmux-level delivery (text appeared in pane).
        Does NOT verify Claude received or processed the message.
        """
        baseline = self._send_marker
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            pane = self._tmux.capture_pane()
            lines = clean_lines(pane)
            if lines:
                self._buf.append_batch(lines)
            if self._buf.total_count() > baseline:
                return True
            time.sleep(0.15)
        return False

    def _send_receipt_check(self, timeout: float = 10.0) -> Optional[dict]:
        """Layer 2: verify Claude received the message by checking state transition.

        Called at the start of wait_for_idle. Polls for up to `timeout` seconds.
        Returns None if Claude left IDLE (message received).
        Returns a send_unconfirmed dict if Claude stayed IDLE.

        Note: Uses short-interval polling (0.5s) instead of _state_event because
        observer only sets _state_event for terminal states, not THINKING/TOOL_CALL.
        """
        initial_state = self._state
        receipt_deadline = time.monotonic() + timeout

        while time.monotonic() < receipt_deadline:
            pane = self._tmux.capture_pane()
            lines = clean_lines(pane)
            if lines:
                self._buf.append_batch(lines)
            result = detect_state(lines)
            self._update_state(result.state)

            if self._state != SessionState.IDLE:
                # Claude left IDLE → message received
                self._send_needs_receipt = False
                logger.info("Send receipt confirmed: state=%s (seq=%d)", self._state, self._send_seq)
                return None  # Continue to normal wait_for_idle

            # Short poll — THINKING doesn't trigger _state_event
            time.sleep(0.5)

        # Timeout: Claude stayed IDLE
        if initial_state == SessionState.IDLE:
            self._send_needs_receipt = False
            logger.warning("Send unconfirmed: Claude stayed IDLE for %.0fs (seq=%d)", timeout, self._send_seq)
            return {
                "status": "send_unconfirmed",
                "state": self._state,
                "send_seq": self._send_seq,
                "elapsed_since_send": round(time.monotonic() - self._send_time, 1) if self._send_time else 0,
                "hint": "Claude did not leave IDLE after send. Try: submit() to retry Enter, or cancel_input() + send() to resend.",
            }

        # Non-IDLE initial state → can't do receipt check, proceed normally
        return None

    def _wait_for_paste_then_submit(self) -> None:
        """Wait for bracketed paste to land in tmux, then send Enter."""
        baseline = self._send_marker
        deadline = time.monotonic() + PASTE_SUBMIT_DELAY_SECONDS
        settled = False

        while time.monotonic() < deadline:
            time.sleep(_PASTE_POLL_INTERVAL)
            if self._buf.total_count() > baseline:
                # Pane received new content — paste has landed
                settled = True
                break

        if settled:
            logger.info("Multi-line send: paste landed, submitting")
        else:
            logger.warning("Multi-line send: no pane change after %.1fs, submitting anyway", PASTE_SUBMIT_DELAY_SECONDS)

        self._tmux.send_special_key("Enter")

    # ------------------------------------------------------------------
    # Status and wait
    # ------------------------------------------------------------------

    def status(self) -> dict:
        """Return current session state."""
        if not self._session_active:
            return {"state": SessionState.DISCONNECTED}

        self._refresh_state()
        # Detect current activity from buffer
        buf_lines = self._buf.read()
        activity = detect_activity([l.text for l in buf_lines]) if buf_lines else {"activity": "idle", "detail": ""}

        return {
            "state": self._state,
            "state_duration_seconds": round(time.monotonic() - self._state_entered, 1),
            "output_tail": self._buf.last_n_chars(200),
            "current_activity": activity.get("activity", "idle"),
            "activity_detail": activity.get("detail", ""),
            "session_ready": self._session_ready_time is not None,
        }

    def wait_for_idle(self, timeout: int = 1800) -> dict:
        """Wait for Claude to return to IDLE state.

        Simplified from the original: inline polling, no adaptive intervals,
        no turn tracking. Just poll → check state → return when done.

        Returns dict with status: "idle" | "permission" | "error" |
        "disconnected" | "exited" | "timeout" | "send_unconfirmed"
        """
        if not self._session_active:
            raise SessionNotActiveError("No active session")

        # Timeout & patrol intervals (all in seconds)
        # PATROL_INTERVAL: how often to check for output growth when idle (default 300s = 5 min)
        PATROL_INTERVAL = int(os.environ.get("HERMES_CLAUDE_SESSION_PATROL_INTERVAL", "300"))
        # STALL_THRESHOLD: consecutive seconds with no buffer growth → consider stalled
        STALL_THRESHOLD = float(os.environ.get("HERMES_CLAUDE_SESSION_STALL_THRESHOLD", "1800"))
        # STATE_STALL_THRESHOLD: seconds in same non-terminal state with no growth → state stall
        STATE_STALL_THRESHOLD = float(os.environ.get("HERMES_CLAUDE_SESSION_STATE_STALL_THRESHOLD", "600"))
        # THINKING_TIMEOUT: max seconds in THINKING state before considering it stalled
        # Catches cases where the model API never responds (e.g. GLM-5 timeout)
        THINKING_TIMEOUT = float(os.environ.get("HERMES_CLAUDE_SESSION_THINKING_TIMEOUT", "300"))
        COMPACT_MIN_WAIT = 3600  # 1 hour minimum for compaction
        COMPACT_MAX_WAIT = 7200  # 2 hours maximum for compaction
        POLL_INTERVAL = 180  # 3 minutes - poll for state changes when thinking/calling

        deadline = time.monotonic() + timeout
        last_patrol_tokens = self._buf.total_count()
        last_growth_time = time.monotonic()
        last_state_change_time = time.monotonic()
        last_state_for_stall = self._state
        compact_detected = False
        compact_start = None

        # ===== Layer 2: Send Receipt Check =====
        # If a recent send hasn't been confirmed yet, verify Claude received it.
        # Claude typically leaves IDLE within 1-2s of receiving a message.
        # If still IDLE after 10s, the message likely wasn't delivered.
        if self._send_needs_receipt and self._send_time is not None:
            receipt_result = self._send_receipt_check()
            if receipt_result is not None:
                return receipt_result
        # ===== End Layer 2 =====

        while time.monotonic() < deadline:
            pane = self._tmux.capture_pane()
            lines = clean_lines(pane)
            if lines:
                self._buf.append_batch(lines)

            result = detect_state(lines)
            self._update_state(result.state)
            state = self._state

            # Push incremental output to streamer
            if self._output_streamer:
                try:
                    self._output_streamer.poll_increment()
                except Exception:
                    pass

            # Terminal states
            if state == SessionState.IDLE:
                # Guard against animation ghost (Forming/Unfurling etc. 3-8s).
                # Confirm IDLE with 3 successive stable checks over 2 seconds.
                idle_confirmed = self._confirm_idle_stable(checks=3, interval=0.7)
                if not idle_confirmed:
                    continue
                self._send_needs_receipt = False
                return {**self._build_idle_result(), "status": "idle"}

            if state == SessionState.PERMISSION:
                perm_context = self._build_permission_context(lines)

                if self._permission_mode == "skip":
                    logger.info("Auto-allowing permission in skip mode: %s",
                                perm_context.get("permission_request", ""))
                    self.respond_permission("allow")
                    self._state_event.clear()
                    self._state_event.wait(timeout=1)
                    continue  # 继续等待 IDLE

                # normal模式：low风险自动批准，medium/high交给Hermes决策
                risk = perm_context.get("risk_level", "medium")
                if risk == "low":
                    logger.info("Auto-allowing low-risk permission: %s %s",
                                perm_context.get("operation", ""),
                                perm_context.get("target", ""))
                    self.respond_permission("allow")
                    self._state_event.clear()
                    self._state_event.wait(timeout=1)
                    continue  # 继续等待 IDLE

                # medium/high风险：返回上下文让Hermes决策
                return {**perm_context, "status": "permission"}

            if state == SessionState.INTERVIEW:
                interview_context = self._build_interview_context(lines)
                return {**interview_context, "status": "interview"}

            if state == SessionState.ERROR:
                return {
                    "state": state,
                    "error_output": self._buf.last_n_chars(500),
                    "status": "error",
                    "error_type": "StateDetectionError",
                    "severity": "transient",
                    "retryable": True,
                }

            if state == SessionState.DISCONNECTED:
                return {"error": "Session disconnected", "state": state, "status": "disconnected"}

            if state == SessionState.EXITED:
                return {"error": "Claude Code has exited", "state": state, "status": "exited"}

            # Compact detection
            if result.is_compacting and not compact_detected:
                compact_detected = True
                compact_start = time.monotonic()
                logger.info("Compact detected, extending wait")

            now = time.monotonic()

            # Update growth tracking
            current_tokens = self._buf.total_count()
            if current_tokens > last_patrol_tokens:
                last_growth_time = now
                last_patrol_tokens = current_tokens

            # Track state changes for state-stall detection
            if state != last_state_for_stall:
                last_state_for_stall = state
                last_state_change_time = now

            # State stall detection: same non-terminal state for too long with no output growth
            # This catches cases where Claude appears stuck (e.g. THINKING for 10+ minutes with no progress)
            non_terminal = state in (SessionState.THINKING, SessionState.TOOL_CALL)
            if non_terminal and (now - last_state_change_time) > STATE_STALL_THRESHOLD:
                # Only trigger if also no output growth
                if (now - last_growth_time) > STATE_STALL_THRESHOLD:
                    logger.warning(
                        "State stall: %s for %.0fs, no output growth for %.0fs",
                        state, now - last_state_change_time, now - last_growth_time,
                    )
                    return {
                        "status": "stalled",
                        "state": state,
                        "stalled_seconds": round(now - last_state_change_time, 1),
                        "no_growth_seconds": round(now - last_growth_time, 1),
                        "progress_info": self._check_progress(deadline - timeout),
                        "hint": f"State {state} unchanged for {round((now - last_state_change_time) / 60, 1)} min with no output. Consider sending a follow-up or cancelling.",
                    }

            # THINKING timeout: model API may not respond (e.g. GLM-5 hangs)
            # Uses _state_entered (set by _update_state) which tracks when THINKING first appeared
            if state == SessionState.THINKING and self._state_entered:
                thinking_duration = now - self._state_entered
                if thinking_duration > THINKING_TIMEOUT:
                    logger.warning(
                        "Thinking timeout: THINKING for %.0fs (limit=%ds), model may be unresponsive",
                        thinking_duration, int(THINKING_TIMEOUT),
                    )
                    return {
                        "status": "stalled",
                        "state": state,
                        "stalled_seconds": round(thinking_duration, 1),
                        "stall_type": "thinking_timeout",
                        "model": getattr(self, "_model", None),
                        "progress_info": self._check_progress(deadline - timeout),
                        "hint": f"Model has been THINKING for {round(thinking_duration / 60, 1)} min with no response. The model API may be unresponsive. Consider stopping and retrying with a different model.",
                    }

            # Fast poll: just check state and wait
            remaining = deadline - now
            if remaining <= 0:
                if compact_detected and compact_start and (now - compact_start) < COMPACT_MAX_WAIT:
                    deadline = now + COMPACT_MIN_WAIT
                    continue
                break

            # Stall detection: no output growth for STALL_THRESHOLD seconds
            compact_active = (
                compact_detected and compact_start is not None
                and (now - compact_start) < COMPACT_MAX_WAIT
            )
            if not compact_active and (now - last_growth_time) > STALL_THRESHOLD:
                return StallDetectedError(
                    f"No output growth for {STALL_THRESHOLD}s",
                    detail=f"state={state}, stalled={round(now - last_growth_time, 1)}s",
                ).to_dict() | {
                    "status": "stalled",
                    "state": state,
                    "stalled_seconds": round(now - last_growth_time, 1),
                    "progress_info": self._check_progress(deadline - timeout),
                }

            # State-aware wait: fast poll when active, respect patrol interval when idle/waiting
            if state in (SessionState.THINKING, SessionState.TOOL_CALL):
                wait_time = min(POLL_INTERVAL, remaining)
            else:
                wait_time = min(PATROL_INTERVAL, remaining)
            self._state_event.clear()
            self._state_event.wait(timeout=wait_time)

        # Timeout
        elapsed = time.monotonic() - (deadline - timeout)
        err = WaitTimeoutError(
            f"wait_for_idle timed out after {timeout}s",
            detail=f"state={self._state}, elapsed={round(elapsed, 1)}s",
        )
        full_output = self._get_output_since_send()
        truncated = len(full_output) > self._OUTPUT_SINCE_SEND_MAX
        timeout_result = err.to_dict() | {
            "status": "timeout",
            "state": self._state,
            "timeout_reached": True,
            "elapsed_seconds": round(elapsed, 1),
            "hint": "Timeout is normal for long tasks. Call wait_for_idle again with a larger timeout.",
            "output_since_send": full_output[:self._OUTPUT_SINCE_SEND_MAX],
        }
        if truncated:
            timeout_result["output_since_send_truncated"] = True
        return timeout_result

    def wait_for_state(self, target_state: str, timeout: int = 60) -> dict:
        """Wait for a specific state."""
        if not self._session_active:
            raise SessionNotActiveError("No active session")

        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            self._refresh_state()
            if self._state == target_state:
                return {"state": target_state}
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            self._state_event.clear()
            self._state_event.wait(timeout=min(0.3, remaining))

        return {"state": self._state, "timeout_reached": True}

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------

    def output(self, offset: int = 0, limit: int = 50) -> dict:
        """Get output lines with pagination."""
        lines = self._buf.read(offset=offset, limit=limit)
        return {
            "lines": [{"text": l.text, "index": l.index} for l in lines],
            "total": self._buf.total_count(),
            "has_more": (offset + limit) < self._buf.total_count(),
        }

    def jsonl_output(self, last_reply: bool = False, last_n: int = 0,
                     max_length: int = 15000) -> dict:
        """Extract Claude's replies from the JSONL session file.

        Use this when tmux output is truncated (long Claude responses).
        The JSONL file contains the complete conversation history.

        Args:
            last_reply: If True, return only the last assistant text reply.
            last_n: If > 0, return the last N assistant text replies.
                    If both are 0/False, returns a summary.
            max_length: Max chars per reply (default 15000). Truncated with
                        a notice if exceeded.
        """
        if not self._claude_session_uuid:
            return {"error": "No session UUID", "replies": []}

        jsonl_path = self._find_session_jsonl(self._workdir or ".", self._claude_session_uuid)
        if not jsonl_path or not os.path.exists(jsonl_path):
            return {"error": "JSONL file not found", "path": str(jsonl_path), "replies": []}

        # Determine how many replies to keep
        if last_reply:
            keep = 1
        elif last_n > 0:
            keep = last_n
        else:
            keep = 0  # summary only

        try:
            assistant_texts = []
            total_tools = 0
            total_entries = 0

            with open(jsonl_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    total_entries += 1

                    # Fast path: skip non-assistant lines without parsing
                    if not line.startswith('{"type":"assistant"'):
                        # Still need to check — type might not be first key
                        try:
                            entry = json.loads(line)
                        except (json.JSONDecodeError, ValueError):
                            continue
                        if entry.get("type") != "assistant":
                            continue
                    else:
                        try:
                            entry = json.loads(line)
                        except (json.JSONDecodeError, ValueError):
                            continue

                    msg = entry.get("message", {})
                    for item in msg.get("content", []):
                        if not isinstance(item, dict):
                            continue
                        if item.get("type") == "tool_use":
                            total_tools += 1
                        if item.get("type") == "text" and item.get("text", "").strip():
                            text = item["text"]
                            if keep > 0:
                                # Sliding window: only keep last N
                                assistant_texts.append(text)
                                if len(assistant_texts) > keep:
                                    assistant_texts.pop(0)
                            else:
                                # Summary mode: just count
                                assistant_texts.append(text)

            if total_entries == 0:
                return {"status": "empty", "replies": []}

            # Summary mode
            if keep == 0:
                return {
                    "status": "active",
                    "total_entries": total_entries,
                    "assistant_text_count": len(assistant_texts),
                    "tool_call_count": total_tools,
                    "last_reply_length": len(assistant_texts[-1]) if assistant_texts else 0,
                }

            # Truncate oversized replies
            result_texts = []
            for text in assistant_texts:
                if len(text) > max_length:
                    text = text[:max_length] + f"\n... [truncated, {len(text)} total chars]"
                result_texts.append(text)

            return {
                "replies": result_texts,
                "count": len(result_texts),
                "total_assistant_texts": len(assistant_texts) + (keep - len(assistant_texts)) if keep > 0 else len(assistant_texts),
            }

        except Exception as e:
            return {"error": str(e), "replies": []}

    # ------------------------------------------------------------------
    # Permission handling
    # ------------------------------------------------------------------

    def respond_permission(self, response: str) -> dict:
        """Respond to a permission request."""
        if response not in ("allow", "deny"):
            raise InvalidPermissionResponseError(f"Invalid response: {response}")

        self._permission_responded = True
        max_retries = 3
        for attempt in range(max_retries):
            with self._lock:
                pane = self._tmux.capture_pane()
                lines = clean_lines(pane)
                result = detect_state(lines)

                if result.state != SessionState.PERMISSION:
                    if is_permission_in_text(pane):
                        self._update_state(SessionState.PERMISSION)
                    else:
                        if attempt < max_retries - 1:
                            continue
                        raise PermissionError(
                            "Not in PERMISSION state",
                            detail="Permission may have been auto-handled.",
                        )

                is_numbered = self._detect_numbered_selector(pane)
                if response == "allow":
                    if is_numbered:
                        self._tmux.send_special_key("Enter")
                    else:
                        self._tmux.send_keys("y", enter=True)
                else:
                    if is_numbered:
                        deny_num = self._find_deny_option_number()
                        if deny_num:
                            self._tmux.send_keys(str(deny_num), enter=True)
                        else:
                            self._tmux.send_special_key("Enter")
                    else:
                        self._tmux.send_keys("n", enter=True)

            time.sleep(0.5)  # 等待 Claude 处理权限
            self._refresh_state()
            return {"responded": True, "state": self._state}

        raise PermissionError("Not in PERMISSION state after retries")

    # ------------------------------------------------------------------
    # Interview handling
    # ------------------------------------------------------------------

    def respond_interview(self, option: str) -> dict:
        """Respond to an interview selector by selecting an option.

        Args:
            option: The option to select. Can be:
                - A number string (e.g. "1") to select that numbered option
                - A text string to type as custom input
                - "enter" to confirm the currently selected option
                - "escape" to cancel
        """
        if not self._session_active:
            raise SessionNotActiveError("No active session")

        # Refresh state before checking — observer cache may be stale
        # (interview menus with multi-line options can evade observer detection)
        self._refresh_state()

        if self._state != SessionState.INTERVIEW:
            raise SessionNotActiveError(
                f"Not in INTERVIEW state (current: {self._state})"
            )

        # Validate state and read current selection under lock, then act outside lock
        if option == "enter":
            with self._lock:
                self._tmux.send_special_key("Enter")
        elif option == "escape":
            with self._lock:
                self._tmux.send_special_key("Escape")
        elif option.isdigit() and int(option) > 0:
            target_num = int(option)
            # Navigate to the target option using arrow keys.
            # Claude Code's Ink select may not support number-key shortcuts in all versions,
            # so we use Up/Down navigation: press Up many times to reach the top, then Down.
            with self._lock:
                # Press Up enough times to guarantee we're at the first option
                # (assumes max ~20 options in a menu — generous upper bound)
                for _ in range(20):
                    self._tmux.send_special_key("Up")
                time.sleep(0.1)
                # Move down (target_num - 1) times to reach the target
                for _ in range(target_num - 1):
                    self._tmux.send_special_key("Down")
                    time.sleep(0.05)
                # Confirm selection
                time.sleep(0.1)
                self._tmux.send_special_key("Enter")
        else:
            # Type custom text
            with self._lock:
                self._tmux.send_keys(option, enter=True)

        time.sleep(0.5)
        self._refresh_state()
        return {"responded": True, "state": self._state}

    # ------------------------------------------------------------------
    # History / Events (simplified — no turn tracking)
    # ------------------------------------------------------------------

    def history(self) -> dict:
        """Return simplified session history."""
        return {
            "turns": [],
            "total_turns": 0,
            "total_tools_called": 0,
            "deprecated": True,
            "note": "Turn tracking removed in context-pipeline refactor",
        }

    def events(self, since_turn: int = 0) -> dict:
        """Return queued events (no-op in simplified version)."""
        return {"events": [], "deprecated": True}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _refresh_state(self) -> None:
        """Poll tmux and update internal state."""
        if not self._tmux or not self._session_active:
            return
        try:
            pane = self._tmux.capture_pane()
            lines = clean_lines(pane)
            if lines:
                self._buf.append_batch(lines)
            result = detect_state(lines)
            self._update_state(result.state)

            # Auto-approve in skip mode or low-risk in normal mode
            # Guard: skip if already inside _auto_approve_permission to prevent recursion
            if result.state == SessionState.PERMISSION and not self._in_auto_approve:
                if self._permission_mode == "skip":
                    self._auto_approve_permission()
                elif self._permission_auto_allow_low_risk:
                    self._try_auto_approve_low_risk(pane)
        except Exception as e:
            session_err = wrap_tmux_error(e)
            logger.warning("State refresh failed: %s [%s]", session_err, session_err.severity.value)
            if isinstance(session_err, SessionDisconnectedError):
                self._update_state(SessionState.DISCONNECTED)

    def _on_observer_update(self, info: dict) -> None:
        """Called by observer thread with activity updates."""
        # Signal wait_for_idle when observer detects a terminal state,
        # so it wakes immediately instead of sleeping through POLL_INTERVAL.
        observed_state = info.get("state")
        if observed_state in (
            SessionState.IDLE, SessionState.PERMISSION, SessionState.INTERVIEW,
            SessionState.ERROR, SessionState.DISCONNECTED, SessionState.EXITED,
        ):
            self._state_event.set()

        if self._status_callback:
            try:
                now = time.monotonic()
                status_info = {
                    "state": info.get("state", self._state),
                    "turn_id": None,
                    "elapsed_seconds": round(now - self._state_entered, 1),
                    "tool_calls": [],
                    "recent_output": self._buf.last_n_chars(200),
                    "tool_name": info.get("tool_name"),
                    "tool_target": info.get("tool_target"),
                }
                current_activity = info.get("current_activity", "idle")
                if current_activity != "idle":
                    status_info["current_activity"] = current_activity
                    status_info["activity_detail"] = info.get("activity_detail", "")
                self._status_callback(status_info)
            except Exception as e:
                logger.warning("Status callback error: %s", e)

    def _start_status_card(self, config: dict) -> None:
        """Create and start a StatusCard for Telegram real-time status.

        config keys:
            chat_id (str): Telegram chat ID
            loop: asyncio event loop from Gateway
            send_func: async callable(chat_id, content) -> SendResult
            edit_func: async callable(chat_id, message_id, content) -> SendResult
            delete_func: async callable(chat_id, message_id) -> bool
            poll_interval (float, optional): polling interval in seconds (default 3.0)
            max_card_length (int, optional): max characters in status card (default 500)
            bump_threshold (int, optional): consecutive failed edits before bumping (default 3)
        """
        chat_id = config.get("chat_id")
        loop = config.get("loop")
        send_func = config.get("send_func")
        edit_func = config.get("edit_func")
        delete_func = config.get("delete_func")

        if not chat_id or not loop or not send_func or not edit_func or not delete_func:
            logger.warning(
                "Status card disabled: missing gateway adapter config "
                "(chat_id=%s, loop=%s, send=%s)",
                chat_id or "missing",
                "set" if loop else "missing",
                "set" if send_func else "missing",
            )
            return
        try:
            self._status_card = StatusCard(
                session_uuid=self._claude_session_uuid,
                loop=loop,
                send_func=send_func,
                edit_func=edit_func,
                delete_func=delete_func,
                chat_id=chat_id,
                poll_interval=config.get("poll_interval", 3.0),
                max_card_length=config.get("max_card_length", 500),
                bump_threshold=config.get("bump_threshold", 3),
                session_name=self._session_name or "",
                session_id=self._session_id or "",
                tmux_session=self._tmux.session_name if self._tmux else "",
            )
            self._status_card.start()

            # Wire observer updates to StatusCard for real-time Telegram updates
            def _observer_to_status_card(info: dict) -> None:
                if self._status_card:
                    self._status_card.update_from_observer(info)

            self._status_callback = _observer_to_status_card

            logger.info("Status card started for session %s", self._claude_session_uuid[:8])
        except Exception as e:
            logger.warning("Status card start failed: %s", e)
            self._status_card = None

    def _auto_approve_permission(self) -> None:
        """Auto-approve permission in skip mode."""
        with self._lock:
            if self._permission_responded:
                self._permission_responded = False
                return
        self._in_auto_approve = True
        try:
            for _ in range(3):
                time.sleep(0.3)
                pane = self._tmux.capture_pane()
                if not is_permission_in_text(pane):
                    self._update_state(SessionState.THINKING)
                    self._permission_responded = False
                    return

                is_numbered = self._detect_numbered_selector(pane)
                if is_numbered:
                    self._tmux.send_special_key("Enter")
                else:
                    self._tmux.send_keys("y", enter=True)

                time.sleep(0.5)
                self._refresh_state()
                if self._state != SessionState.PERMISSION:
                    self._permission_responded = False
                    return
        finally:
            self._in_auto_approve = False

    def _try_auto_approve_low_risk(self, pane_text: str) -> None:
        """Auto-approve low-risk permissions even in normal mode.

        Called from _refresh_state when permission is detected and mode is 'normal'.
        Builds permission context, checks risk level, and auto-allows if low risk.
        Uses _lock to prevent race with wait_for_idle's permission handling.
        """
        with self._lock:
            if self._permission_responded:
                return
            # Claim immediately to prevent concurrent paths
            self._permission_responded = True

        lines = clean_lines(pane_text)
        perm_context = self._build_permission_context(lines)
        risk = perm_context.get("risk_level", "medium")
        if risk == "low":
            logger.info("Auto-allowing low-risk permission (observer path): %s %s",
                        perm_context.get("operation", ""),
                        perm_context.get("target", ""))
            self._auto_approve_permission()
        else:
            # Not low-risk, release the flag so wait_for_idle can handle it
            self._permission_responded = False

    def _confirm_idle_stable(self, checks: int = 3, interval: float = 0.7) -> bool:
        """Confirm IDLE is stable across multiple checks (defeats animation ghosts).

        Claude Code animations (Forming, Unfurling, etc.) can produce transient
        ❯ prompts that last 3-8 seconds. A single 0.5s confirmation is insufficient.
        This method polls `checks` times over `checks * interval` seconds and
        only returns True if ALL checks agree on IDLE.
        """
        for check_idx in range(checks):
            time.sleep(interval)
            pane = self._tmux.capture_pane()
            lines = clean_lines(pane)
            result = detect_state(lines)
            tail = [l[:80] for l in lines[-5:]] if lines else ["(empty)"]
            logger.debug("confirm_idle_stable check %d/%d: state=%s tail=%s",
                         check_idx + 1, checks, result.state, tail)
            if result.state != SessionState.IDLE:
                self._update_state(result.state)
                return False
        return True

    _OUTPUT_SINCE_SEND_MAX = 500

    def _build_idle_result(self) -> dict:
        full_output = self._get_output_since_send()
        truncated = len(full_output) > self._OUTPUT_SINCE_SEND_MAX
        result = {
            "state": SessionState.IDLE,
            "output_since_send": full_output[:self._OUTPUT_SINCE_SEND_MAX],
        }
        if truncated:
            result["output_since_send_truncated"] = True
        return result

    def _build_permission_result(self, lines: list = None) -> dict:
        result = {"state": SessionState.PERMISSION}
        if lines:
            for line in reversed(lines[-10:]):
                lower = line.lower()
                if "allow" in lower or "permission" in lower or "proceed?" in lower:
                    result["permission_request"] = line
                    break
        return result

    def _build_permission_context(self, lines: list = None) -> dict:
        """构建权限上下文，供Hermes智能决策使用"""
        context = {
            "state": SessionState.PERMISSION,
            "needs_hermes_decision": True,
            "permission_request": "",
            "operation": "",
            "target": "",
            "risk_level": "medium",
        }

        if not lines:
            return context

        # 从输出中提取权限请求详情
        for line in reversed(lines[-15:]):
            lower = line.lower()
            if "allow" in lower or "permission" in lower or "proceed?" in lower:
                context["permission_request"] = line.strip()
                break

        # 识别操作类型和目标
        for line in lines:
            stripped = line.strip()
            # Edit/Write/MultiEdit 操作
            m = re.match(r"●\s*(Edit|Write|MultiEdit)\s+(.+)", stripped)
            if m:
                context["operation"] = m.group(1)
                context["target"] = m.group(2).strip()
                context["risk_level"] = self._assess_risk(m.group(1), m.group(2))
                break
            # Create 操作（创建文件/目录）
            m = re.match(r"●\s*Create\s+(.+)", stripped)
            if m:
                context["operation"] = "Create"
                context["target"] = m.group(1).strip()
                context["risk_level"] = self._assess_risk("Create", m.group(1))
                break
            # Bash 操作
            m = re.match(r"●\s*Bash\s*\((.+)\)", stripped)
            if m:
                context["operation"] = "Bash"
                context["target"] = m.group(1).strip()
                context["risk_level"] = self._assess_risk("Bash", m.group(1))
                break
            # 只读操作（低风险）
            m = re.match(r"●\s*(Grep|Search|WebFetch|Read|LS|List|Glob)\s*\((.+)\)", stripped)
            if m:
                context["operation"] = m.group(1)
                context["target"] = m.group(2).strip()
                context["risk_level"] = "low"
                break
            # TodoWrite — 元数据操作，低风险
            m = re.match(r"●\s*TodoWrite\s*\((.+)\)", stripped)
            if m:
                context["operation"] = "TodoWrite"
                context["target"] = m.group(1).strip()
                context["risk_level"] = "low"
                break
            # Task — 子任务委派
            m = re.match(r"●\s*Task\s*\((.+)\)", stripped)
            if m:
                context["operation"] = "Task"
                context["target"] = m.group(1).strip()
                context["risk_level"] = self._assess_risk("Task", m.group(1))
                break
            # NotebookEdit — notebook 单元格编辑
            m = re.match(r"●\s*NotebookEdit\s*\((.+)\)", stripped)
            if m:
                context["operation"] = "NotebookEdit"
                context["target"] = m.group(1).strip()
                context["risk_level"] = self._assess_risk("NotebookEdit", m.group(1))
                break
            # MCP 工具调用 (MCP::server::tool 格式)
            m = re.match(r"●\s*MCP::(\S+?)::(\S+)\s*\((.+)\)", stripped)
            if m:
                context["operation"] = f"MCP::{m.group(1)}::{m.group(2)}"
                context["target"] = m.group(3).strip()
                context["risk_level"] = self._assess_risk("MCP", m.group(3))
                break

        # 如果没有检测到具体操作，从权限文本推断
        if not context["operation"] and context["permission_request"]:
            req_lower = context["permission_request"].lower()
            if "edit" in req_lower or "write" in req_lower:
                context["operation"] = "Edit"
            elif "delete" in req_lower or "remove" in req_lower:
                context["operation"] = "Delete"
                context["risk_level"] = "high"
            elif "run" in req_lower or "execute" in req_lower:
                context["operation"] = "Bash"
                context["risk_level"] = "high"

        return context

    def _build_interview_context(self, lines: list = None) -> dict:
        """构建 interview 上下文，供 Hermes 智能选择"""
        context = {
            "state": SessionState.INTERVIEW,
            "needs_hermes_decision": True,
            "question": "",
            "options": [],
            "sections": [],
        }

        if not lines:
            return context

        # 提取问题（排除编号选项行）
        for line in reversed(lines[-20:]):
            stripped = line.strip()
            if (stripped
                    and not stripped.startswith("❯")
                    and not stripped.startswith("←")
                    and "?" in stripped
                    and not re.match(r"\s*\d+\.\s+", stripped)):
                context["question"] = stripped
                break

        # 提取选项列表
        for line in lines[-20:]:
            m = re.match(r"\s*❯?\s*(\d+)\.\s+(.+)", line.strip())
            if m:
                num = int(m.group(1))
                text = m.group(2).strip()
                selected = line.strip().startswith("❯")
                context["options"].append({
                    "number": num,
                    "text": text,
                    "selected": selected,
                })

        # 提取 section 标记（如 ☐ 运行环境  ✔ Submit）
        for line in lines[-20:]:
            for m_sec in re.finditer(r"([☐✔])\s+([^☐✔←→]+?)\s*(?=[☐✔←→]|$)", line):
                label = m_sec.group(2).strip()
                if label:
                    context["sections"].append({
                        "checked": m_sec.group(1) == "✔",
                        "label": label,
                    })

        return context

    # 系统目录 — 始终高风险
    _SYSTEM_DIRS = ("/etc", "/usr", "/bin", "/sbin", "/lib", "/boot", "/System", "/Windows")
    # 敏感用户目录 — 始终高风险
    _SENSITIVE_DIRS = ("~/.ssh", "~/.gnupg", "~/.kube", "~/.aws")
    # 非破坏性 Bash 命令 — 低风险
    _SAFE_BASH_PREFIXES = (
        "cat ", "ls ", "head ", "tail ", "wc ", "grep ", "find ", "echo ",
        "test ", "mkdir -p ", "which ", "type ", "file ", "stat ",
        "du ", "df ", "pwd", "whoami", "hostname", "uname ",
        "python -c \"import", "python3 -c \"import",  # read-only python
        "git status", "git log", "git diff", "git branch", "git remote",
    )

    def _assess_risk(self, operation: str, target: str) -> str:
        """评估操作风险等级，考虑 workdir 上下文。"""
        target_lower = target.lower()
        op_lower = operation.lower()

        # 1. 绝对高风险模式（无论什么操作）
        high_risk_patterns = [
            r"rm\s+-rf\b", r"\bdelete\b.*\bforce\b", r"\bdrop\b",
            r"chmod\s+0?[0-7]{3}",
            r"\bsudo\b", r"\bsu\b\s",
            r"\.\./", r"\.\.\\",
        ]
        for pattern in high_risk_patterns:
            if re.search(pattern, target_lower, re.IGNORECASE):
                return "high"

        # 2. 系统目录和敏感目录 — 始终 high（路径前缀匹配，避免子字符串误报）
        norm_target = os.path.normpath(target)
        for sys_dir in self._SYSTEM_DIRS:
            if norm_target == sys_dir or norm_target.startswith(sys_dir + os.sep):
                return "high"
        for sens_dir in self._SENSITIVE_DIRS:
            if sens_dir in target or os.path.expanduser(sens_dir) in target:
                return "high"

        # 3. 操作类型 + workdir 感知
        is_in_workdir = self._is_in_workdir(target)

        # 只读操作 — 始终 low
        if op_lower in ("grep", "search", "webfetch", "read", "ls", "list", "glob", "todowrite"):
            return "low"

        # Bash — 根据命令内容判断
        if op_lower == "bash":
            return self._assess_bash_risk(target)

        # Edit/Write/Create/MultiEdit/NotebookEdit — workdir 内 low，否则 medium
        if op_lower in ("edit", "write", "multiedit", "create", "notebookedit"):
            if is_in_workdir:
                return "low"
            return "medium"

        # Task/MCP — 默认 medium
        if op_lower in ("task", "mcp"):
            if is_in_workdir:
                return "low"
            return "medium"

        # 未识别操作 — 默认 medium
        return "medium"

    def _is_in_workdir(self, target: str) -> bool:
        """检查目标路径是否在 session 的 workdir 内。"""
        if not self._workdir or not target:
            return False
        # 展开相对路径
        target_expanded = target
        if not os.path.isabs(target):
            target_expanded = os.path.join(self._workdir, target)
        target_expanded = os.path.abspath(target_expanded)
        return target_expanded.startswith(self._workdir + os.sep) or target_expanded == self._workdir

    def _assess_bash_risk(self, command: str) -> str:
        """评估 Bash 命令的风险等级。"""
        cmd = command.strip().lower()

        # 高风险命令
        high_patterns = [
            r"rm\s+-rf\b", r"\bsudo\b", r"dd\s+if=", r"mkfs\b",
            r"curl\s+.*\|\s*sh", r"wget\s+.*\|\s*sh",
            r">\s*/dev/sd", r"chmod\s+[0-7]{3,4}\s",
        ]
        for p in high_patterns:
            if re.search(p, cmd, re.IGNORECASE):
                return "high"

        # 非破坏性命令 — low
        for safe_prefix in self._SAFE_BASH_PREFIXES:
            if cmd.startswith(safe_prefix.lower()):
                return "low"

        # workdir 内操作 — medium（如 pytest, npm test 等）
        return "medium"

    def _get_output_since_send(self) -> str:
        lines = self._buf.since(self._send_marker)
        return "\n".join(l.text for l in lines)

    def _check_progress(self, start_time: float) -> dict:
        now = time.monotonic()
        current = self._buf.total_count()
        return {
            "elapsed_seconds": round(now - start_time, 1),
            "token_count": current,
            "current_state": self._state,
            "state_duration_seconds": round(now - self._state_entered, 1),
        }

    def _detect_numbered_selector(self, pane_text: Optional[str] = None) -> bool:
        try:
            if pane_text is None:
                pane_text = self._tmux.capture_pane()
            lines = clean_lines(pane_text)
            last_lines = lines[-8:] if len(lines) >= 8 else lines
            for line in last_lines:
                if re.match(r"\s*❯\s*\d+\.", line):
                    return True
        except Exception:
            pass
        return False

    def _find_deny_option_number(self) -> Optional[int]:
        try:
            pane = self._tmux.capture_pane()
            lines = clean_lines(pane)
            for line in lines[-8:]:
                m = re.match(r"\s*(?:❯\s*)?(\d+)\.\s*(No|Deny)\b", line, re.IGNORECASE)
                if m:
                    return int(m.group(1))
        except Exception:
            pass
        return None

    def _wait_for_claude_startup(self, timeout: int = 30, is_resume: bool = False) -> bool:
        """Wait for Claude Code to become usable."""
        deadline = time.monotonic() + timeout
        EMPTY_THRESHOLD = 3
        startup_attempts = 0
        self._startup_error_count = 0  # Reset for this startup attempt
        # For resume: allow non-IDLE states after settling (session may continue mid-task)
        resume_settle_deadline = time.monotonic() + 5 if is_resume else None

        while time.monotonic() < deadline:
            try:
                pane = self._tmux.capture_pane(lines=100)
                lines = clean_lines(pane)

                if not lines or len(lines) < EMPTY_THRESHOLD:
                    time.sleep(1.0)
                    continue

                scene = detect_startup_scene(lines)
                if scene and startup_attempts < 3:
                    startup_attempts += 1
                    if scene.action == "press_enter":
                        self._tmux.send_special_key("Enter")
                    elif scene.action == "press_down_enter":
                        self._tmux.send_special_key("Down")
                        time.sleep(0.3)
                        self._tmux.send_special_key("Enter")
                    time.sleep(2.0)
                    continue

                result = detect_state(lines)

                if result.state == SessionState.IDLE:
                    logger.info("Claude Code startup OK: IDLE")
                    return True

                # During startup, THINKING/TOOL_CALL/PERMISSION may appear from
                # the welcome screen spinner or animation — do NOT treat as ready.
                # Only accept these states if we've already seen an IDLE (prompt).
                # Skip and keep polling until the welcome screen clears.
                if result.state in ("THINKING", "TOOL_CALL", "PERMISSION", "INTERVIEW"):
                    # For resume: after 5s settling, accept non-IDLE states
                    # (session may have been interrupted mid-task and continues executing)
                    if resume_settle_deadline and time.monotonic() >= resume_settle_deadline:
                        logger.info("Claude Code startup OK (resume): %s", result.state)
                        return True
                    # Wait longer — welcome screen animation takes several seconds
                    time.sleep(1.0)
                    continue

                if result.state in (SessionState.ERROR, SessionState.EXITED):
                    error_count = getattr(self, "_startup_error_count", 0) + 1
                    self._startup_error_count = error_count
                    logger.warning(
                        "Claude startup detected %s (attempt %d/3) — pane tail: %s",
                        result.state, error_count,
                        " | ".join(l.strip() for l in lines[-8:] if l.strip()),
                    )
                    if error_count >= 3:
                        logger.error("Claude Code startup failed after %d ERROR/EXITED detections", error_count)
                        return False
                    # Transient errors (API timeout, network blip) may clear — wait and retry
                    time.sleep(3.0)
                    continue

            except Exception as e:
                logger.warning("Startup poll error: %s", e)

            time.sleep(1.0)

        logger.warning("Claude Code startup timed out after %ds", timeout)
        return False

    @staticmethod
    def _find_session_jsonl(workdir: str, session_uuid: str) -> Optional[str]:
        if not re.match(
            r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$',
            session_uuid,
        ):
            return None

        workdir = os.path.abspath(workdir).rstrip("/")
        claude_dir = os.path.expanduser("~/.claude/projects")
        dir_name = workdir.replace("/", "-")
        jsonl_path = os.path.join(claude_dir, dir_name, f"{session_uuid}.jsonl")
        if os.path.exists(jsonl_path):
            return jsonl_path
        return None
