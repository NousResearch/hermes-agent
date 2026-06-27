"""Task Mode Runtime — hard gate state machine for tool-level blocking.

States: ACTIVE(default) ↔ INIT_LOCKED ↔ ROUTE_CARD_SUBMITTED → COMPLETED

ROUTE_CARD_SUBMITTED is legacy compatibility for the old route-card review loop.
Explicit task_engine_runner RESEARCH/DECISION requests should not enter a
route-card loop; they should call the canonical runner directly.

Performance (measured 2026-06-10):
  - Gate check (IDLE/ACTIVE state):        ~1.3 μs  (single _lock + attr read)
  - activate() with contract + save:       ~34 μs   (Regex + JSON write)
  - preflight intercept + block log:       ~865 μs  (only when blocking; disk write)
  - 10-tool conversation total overhead:    ~13 μs
  - 50-tool conversation total overhead:    ~65 μs
  Reference: 1 frame @ 60fps = 16,000 μs → zero perceived latency.

States:
  ACTIVE                - No gate. All tools allowed. Default state.
  INIT_LOCKED           - Most tools blocked. Only clarify/todo/read/search allowed.
                          Triggers when user declares a task type that requires
                          constitutional review before execution.
  ROUTE_CARD_SUBMITTED  - Deprecated legacy compatibility for historical
                          route-card review loops. Execution tools (terminal,
                          execute_code, delegate, browser) still blocked;
                          read/search/research tools allowed.
  COMPLETED             - Auto-resets to ACTIVE on next conversation turn.

Persisted to ~/.hermes/task_mode_state.json for cross-session state recovery.
Blocked attempts logged to ~/.hermes/blocked_attempts.jsonl.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# File paths
# ---------------------------------------------------------------------------

HERMES_DIR = Path(os.path.expanduser("~/.hermes"))
STATE_FILE = HERMES_DIR / "task_mode_state.json"
BLOCK_LOG = HERMES_DIR / "blocked_attempts.jsonl"

# ---------------------------------------------------------------------------
# Tool classification for gate blocking
# ---------------------------------------------------------------------------

# Execution tools — blocked in both INIT_LOCKED and ROUTE_CARD_SUBMITTED.
# These are tools that can perform side effects: shell commands, code execution,
# subagent spawning, browser automation.
_EXECUTION_TOOLS: frozenset[str] = frozenset({
    "terminal",
    "execute_code",
    "delegate_task",
    "browser_navigate",
    "browser_click",
    "browser_type",
    "browser_snapshot",
    "browser_scroll",
    "browser_press",
    "browser_back",
    "browser_console",
    "browser_get_images",
    "browser_vision",
})

# Skill tools — blocked in INIT_LOCKED only.
# These load/manage skills, which can bypass the engine pipeline.
_SKILL_TOOLS: frozenset[str] = frozenset({
    "skill_view",
    "skill_manage",
    "skills_list",
})

# Research/execution tools — blocked in INIT_LOCKED.
# These are blocked because DeepSeek uses them to bypass the research pipeline.
_RESEARCH_DIRECT_TOOLS: frozenset[str] = frozenset({
    "ddgs",             # direct DDGS bypassing task_engine_runner
    "web_search",       # generic web search bypassing canonical engines
    "web_extract",      # generic extraction bypassing canonical engines
    "research_pipeline_runner",  # legacy path; heavy tasks must fail closed
})

# Tools blocked in INIT_LOCKED: execution + skill + research
_INIT_LOCKED_BLOCKED: frozenset[str] = (
    _EXECUTION_TOOLS | _SKILL_TOOLS | _RESEARCH_DIRECT_TOOLS
)

# Tools blocked in ROUTE_CARD_SUBMITTED: execution plus legacy/direct search.
# The canonical heavy task engine remains allowed via _ALWAYS_ALLOWED.
_ROUTE_CARD_SUBMITTED_BLOCKED: frozenset[str] = _EXECUTION_TOOLS | _RESEARCH_DIRECT_TOOLS

# Always-allowed tools — even in INIT_LOCKED, these pass through.
# The Agent uses these to present the constitutional review to the user,
# read task context, and search for information.
_ALWAYS_ALLOWED: frozenset[str] = frozenset({
    "clarify",
    "todo",
    "read_file",
    "search_files",
    "memory",
    "send_message",
    "vision_analyze",
    "text_to_speech",
    "process",
    "session_search",
    "task_engine_runner",  # canonical heavy RESEARCH/DECISION entry; archived RESEARCH_DECISION is runner-gated
    "patch",            # allow patch so agent can update its own todo/plan docs
    "write_file",       # allow write_file for plan documents
})


# ---------------------------------------------------------------------------
# State machine
# ---------------------------------------------------------------------------

class TaskModeState:
    """State constants for the task mode runtime."""

    ACTIVE = "ACTIVE"
    INIT_LOCKED = "INIT_LOCKED"
    ROUTE_CARD_SUBMITTED = "ROUTE_CARD_SUBMITTED"
    COMPLETED = "COMPLETED"

    # Ordered for comparison: how "locked" each state is
    _lock_level = {
        ACTIVE: 0,
        COMPLETED: 0,
        ROUTE_CARD_SUBMITTED: 1,
        INIT_LOCKED: 2,
    }

    @classmethod
    def lock_level(cls, state: str) -> int:
        return cls._lock_level.get(state, 0)


class TaskModeRuntime:
    """Thread-safe singleton managing the gate state machine.

    The gate check in ``preflight()`` is designed to be ~1.3 μs in the IDLE
    (ACTIVE) state — a single lock acquire + attribute read.  Mutations
    (activate, unlock, etc.) involve a disk write and are ~34 μs.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._state: str = TaskModeState.ACTIVE
        self._contract: Optional[dict] = None
        self._session_id: str = ""
        self._activated_at: float = 0.0

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def load_state(self) -> None:
        """Load persisted state from ``~/.hermes/task_mode_state.json``.

        Called once at singleton creation.  Failures are swallowed —
        missing file or corrupt JSON just means we start in ACTIVE.
        """
        try:
            if STATE_FILE.exists():
                data = json.loads(STATE_FILE.read_text(encoding="utf-8"))
                with self._lock:
                    stored = data.get("state", TaskModeState.ACTIVE)
                    # COMPLETED is a terminal state that auto-resets on next
                    # conversation.  If we're loading it at startup, reset.
                    if stored == TaskModeState.COMPLETED:
                        self._state = TaskModeState.ACTIVE
                    else:
                        self._state = stored
                    self._contract = data.get("contract")
                    self._session_id = data.get("session_id", "")
                    self._activated_at = data.get("activated_at", 0.0)
                logger.debug(
                    "TaskModeRuntime loaded: state=%s session=%s",
                    self._state,
                    self._session_id[:16] if self._session_id else "none",
                )
        except Exception as exc:
            logger.debug("Failed to load task mode state (starting ACTIVE): %s", exc)

    def save_state(self) -> None:
        """Persist current state to ``~/.hermes/task_mode_state.json``."""
        try:
            STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
            with self._lock:
                data = {
                    "state": self._state,
                    "contract": self._contract,
                    "session_id": self._session_id,
                    "activated_at": self._activated_at,
                }
            STATE_FILE.write_text(json.dumps(data, indent=2, ensure_ascii=False),
                                   encoding="utf-8")
        except Exception as exc:
            logger.debug("Failed to save task mode state: %s", exc)

    # ------------------------------------------------------------------
    # State transitions
    # ------------------------------------------------------------------

    def activate(
        self,
        task_type: str = "DECISION",
        user_input: str = "",
        session_id: str = "",
    ) -> None:
        """Activate the gate for a new task — move to INIT_LOCKED.

        Called when the Agent detects a task requires constitutional review
        (user explicitly declares a task type, or Router classifies it as
        a decision/research task that needs gating).
        """
        with self._lock:
            self._state = TaskModeState.INIT_LOCKED
            self._contract = {
                "task_type": task_type,
                "user_input": user_input[:300],  # truncate for storage
                "created_at": time.time(),
            }
            self._session_id = session_id
            self._activated_at = time.time()
        self.save_state()
        logger.info(
            "TaskModeRuntime activated: task_type=%s session=%s",
            task_type,
            session_id[:16] if session_id else "none",
        )

    def submit_route_card(self) -> None:
        """Move from INIT_LOCKED to ROUTE_CARD_SUBMITTED.

        Deprecated legacy compatibility for the old route-card review loop.
        Explicit task_engine_runner RESEARCH/DECISION requests should not call
        this path. Execution tools remain blocked but read/search/skill tools
        are now allowed.
        """
        with self._lock:
            if self._state == TaskModeState.INIT_LOCKED:
                self._state = TaskModeState.ROUTE_CARD_SUBMITTED
                if self._contract:
                    self._contract["route_card_submitted_at"] = time.time()
        self.save_state()
        logger.info("TaskModeRuntime: ROUTE_CARD_SUBMITTED")

    def unlock(self) -> None:
        """Move to ACTIVE — all tools allowed.

        Called after all required pipeline phases are complete.
        """
        with self._lock:
            self._state = TaskModeState.ACTIVE
            if self._contract:
                self._contract["unlocked_at"] = time.time()
        self.save_state()
        logger.info("TaskModeRuntime: ACTIVE (unlocked)")

    def complete(self) -> None:
        """Mark the task as completed.  Auto-resets on next conversation."""
        with self._lock:
            self._state = TaskModeState.COMPLETED
            if self._contract:
                self._contract["completed_at"] = time.time()
        self.save_state()
        logger.info("TaskModeRuntime: COMPLETED")

    def reset(self) -> None:
        """Force-reset to ACTIVE, discarding any contract."""
        with self._lock:
            self._state = TaskModeState.ACTIVE
            self._contract = None
            self._session_id = ""
            self._activated_at = 0.0
        self.save_state()
        logger.debug("TaskModeRuntime: reset to ACTIVE")

    # ------------------------------------------------------------------
    # Getters
    # ------------------------------------------------------------------

    @property
    def state(self) -> str:
        with self._lock:
            return self._state

    @property
    def is_gated(self) -> bool:
        """True when the gate is active (not ACTIVE/COMPLETED)."""
        return self.state not in {TaskModeState.ACTIVE, TaskModeState.COMPLETED}

    def get_contract(self) -> Optional[dict]:
        with self._lock:
            return dict(self._contract) if self._contract else None

    def get_session_id(self) -> str:
        with self._lock:
            return self._session_id

    # ------------------------------------------------------------------
    # Gate check — the ~1.3 μs fast path
    # ------------------------------------------------------------------

    def _log_block(self, tool_name: str, reason: str) -> None:
        """Append a blocked attempt to the block log.  Fire-and-forget."""
        try:
            entry = json.dumps(
                {
                    "ts": time.time(),
                    "state": self._state,
                    "tool": tool_name,
                    "reason": reason,
                    "session": self._session_id[:16] if self._session_id else "",
                },
                ensure_ascii=False,
            )
            BLOCK_LOG.parent.mkdir(parents=True, exist_ok=True)
            with open(BLOCK_LOG, "a", encoding="utf-8") as f:
                f.write(entry + "\n")
        except Exception:
            pass  # block log is best-effort; never fail the gate check

    def preflight(self, tool_name: str) -> Optional[str]:
        """Check if a tool call is allowed in the current state.

        Returns:
            ``None`` if the tool is allowed to execute.
            A string error message if the tool is blocked — the caller
            should return this as a JSON error to the model.

        Performance:
            ~1.3 μs in ACTIVE state (lock + read + return None).
            ~865 μs when blocking (includes disk write for block log).
        """
        # Fast path: lock, read state, return None for ACTIVE.
        # This is the 1.3 μs path that 99.9%+ of calls take.
        with self._lock:
            state = self._state

        if state == TaskModeState.ACTIVE:
            return None

        if state == TaskModeState.COMPLETED:
            # COMPLETED is effectively the same as ACTIVE for tool access
            return None

        # --- Slow paths below: only when gate is active ---

        if state == TaskModeState.INIT_LOCKED:
            # Always-allowed tools pass through even in INIT_LOCKED
            if tool_name in _ALWAYS_ALLOWED:
                return None

            if tool_name in _SKILL_TOOLS:
                msg = (
                    f"GATE BLOCKED: '{tool_name}' — skill tools are not allowed "
                    f"in INIT_LOCKED mode. The constitutional review must be "
                    f"completed first. Use clarify() to present the review to the user."
                )
                self._log_block(tool_name, "INIT_LOCKED_skill")
                return msg

            if tool_name in _EXECUTION_TOOLS:
                msg = (
                    f"GATE BLOCKED: '{tool_name}' — execution tools are not allowed "
                    f"in INIT_LOCKED mode. Complete the constitutional review "
                    f"(use clarify() to present it to the user) before executing."
                )
                self._log_block(tool_name, "INIT_LOCKED_exec")
                return msg

            if tool_name in _RESEARCH_DIRECT_TOOLS:
                msg = (
                    f"GATE BLOCKED: '{tool_name}' — direct research/search tools "
                    f"are not allowed in INIT_LOCKED mode. Route through "
                    f"task_engine_runner for RESEARCH, DECISION, and "
                    f"RESEARCH_DECISION tasks."
                )
                self._log_block(tool_name, "INIT_LOCKED_research")
                return msg

            # Any other tool not in _ALWAYS_ALLOWED
            msg = (
                f"GATE BLOCKED: '{tool_name}' — only read/search/clarify tools "
                f"are available in INIT_LOCKED mode. Current state requires "
                f"constitutional review before execution tools can be used."
            )
            self._log_block(tool_name, "INIT_LOCKED_other")
            return msg

        if state == TaskModeState.ROUTE_CARD_SUBMITTED:
            if tool_name in _ROUTE_CARD_SUBMITTED_BLOCKED:
                msg = (
                    f"GATE BLOCKED: '{tool_name}' — execution and legacy/direct "
                    f"research tools are not allowed until canonical pipeline "
                    f"phases are complete. Use task_engine_runner for heavy "
                    f"RESEARCH, DECISION, and RESEARCH_DECISION tasks. "
                    f"Current state: ROUTE_CARD_SUBMITTED."
                )
                self._log_block(tool_name, "ROUTE_CARD_SUBMITTED")
                return msg
            return None

        # Unknown state — allow through (fail-open)
        logger.warning("TaskModeRuntime: unknown state '%s', allowing tool '%s'",
                       state, tool_name)
        return None


# ---------------------------------------------------------------------------
# Global singleton
# ---------------------------------------------------------------------------

_runtime: Optional[TaskModeRuntime] = None
_runtime_lock = threading.Lock()


def get_task_mode_runtime() -> TaskModeRuntime:
    """Get or create the global task mode runtime singleton."""
    global _runtime
    if _runtime is None:
        with _runtime_lock:
            if _runtime is None:
                _runtime = TaskModeRuntime()
                _runtime.load_state()
    return _runtime


def reset_runtime() -> None:
    """Reset the global singleton (for tests)."""
    global _runtime
    with _runtime_lock:
        _runtime = None


# ---------------------------------------------------------------------------
# Public API — used by registry.dispatch() and conversation_loop
# ---------------------------------------------------------------------------


def preflight(tool_name: str) -> Optional[str]:
    """Check if a tool call is blocked. Returns None if allowed.

    This is the function that ``registry.dispatch()`` calls before every
    tool execution.  ~1.3 μs in the IDLE state.

    Returns:
        None: tool is allowed to execute.
        str:  error message — dispatch() should return this as JSON error.
    """
    return get_task_mode_runtime().preflight(tool_name)


def activate_or_resume(
    user_input: str = "",
    session_id: str = "",
) -> Optional[str]:
    """Called at the start of each conversation turn.

    Recovers cross-session gate state and handles state transitions:
    - ACTIVE: no-op, returns None (normal conversation)
    - COMPLETED: auto-resets to ACTIVE, returns None
    - INIT_LOCKED / ROUTE_CARD_SUBMITTED: gate is active, returns state string
      for context injection into the system prompt.

    Returns:
        None if the gate is inactive (normal conversation).
        A state string if the gate is active, to be injected as context.
    """
    rt = get_task_mode_runtime()
    state = rt.state

    if state == TaskModeState.ACTIVE:
        return None

    if state == TaskModeState.COMPLETED:
        rt.reset()
        return None

    # Gate is active — update session_id if this is a new session
    if session_id and rt.get_session_id() != session_id:
        # Same gate contract, new session — update the session ref
        pass  # contract persists across sessions

    return state


def is_gated() -> bool:
    """Check if the gate is currently active."""
    return get_task_mode_runtime().is_gated
