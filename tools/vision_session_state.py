"""In-memory Vision Router session state (Stage-3 limited use).

HERMES_VISION_ROUTER_STAGE3_LIMITED_USE_OPERATING_MODE_DESIGN_V0_1
(approved token redacted):

- user-only activation flag (slash command ``/vision on|off``); the model can
  never set it;
- per-turn / per-session logical-call budgets (enforced in the wrapper);
- in-flight marker (no concurrent Vision calls);
- session-scoped attachment-handle allowlist (``attachment://`` -> server path;
  the model never sees the path);
- session-scoped OCR full-result registry (private handle -> file) for the
  bounded ``vision_ocr_page`` retrieval tool.

Everything is process-local and non-persistent. Router=false (server flag)
still removes all model-visible Vision tools; this state only gates USE inside
a session where the server flag is already on.
"""
from __future__ import annotations

import threading
import time
from typing import Dict, Optional

DEFAULT_PER_TURN_MAX_CALLS = 1
DEFAULT_PER_SESSION_MAX_CALLS = 5


class _VisionSessionState:
    """Process-local singleton. Thread-safe for gateway concurrency."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.enabled = False
        self.per_turn_used = 0
        self.per_session_used = 0
        self.in_flight = False
        self.attachment_allowlist: Dict[str, str] = {}
        self.ocr_results: Dict[str, str] = {}  # private handle -> file path
        self._recent_calls: list = []  # (source_handle, task) dedupe
        self._last_turn_marker: Optional[str] = None

    # -- activation ----------------------------------------------------------
    def set_enabled(self, flag: bool) -> None:
        with self._lock:
            self.enabled = bool(flag)
            # any transition resets budgets and revokes session sources:
            # enabling starts a fresh limited-use session, disabling is the
            # immediate kill switch.
            self.attachment_allowlist.clear()
            self.ocr_results.clear()
            self.per_turn_used = 0
            self.per_session_used = 0
            self.in_flight = False
            self._recent_calls.clear()

    # -- turn boundary -------------------------------------------------------
    def begin_turn(self, marker: Optional[str] = None) -> None:
        """Call at the start of an Agent turn. ``marker`` may be a turn id;
        without one, each call resets the per-turn counter (conservative)."""
        with self._lock:
            if marker is None or marker != self._last_turn_marker:
                self.per_turn_used = 0
                self._last_turn_marker = marker

    def end_turn(self) -> None:
        with self._lock:
            self.per_turn_used = 0

    # -- budget --------------------------------------------------------------
    def consume_call(self, per_turn_max: int, per_session_max: int) -> Optional[str]:
        """Try to consume one logical Vision call slot.

        Returns None when allowed; otherwise a BUSY reason string.
        """
        with self._lock:
            if not self.enabled:
                return "SESSION_DISABLED"
            if self.in_flight:
                return "VISION_BUSY_IN_FLIGHT"
            if self.per_turn_used >= per_turn_max:
                return "TURN_BUDGET_EXHAUSTED"
            if self.per_session_used >= per_session_max:
                return "SESSION_BUDGET_EXHAUSTED"
            self.in_flight = True
            return None

    def finish_call(self) -> None:
        with self._lock:
            self.per_turn_used += 1
            self.per_session_used += 1
            self.in_flight = False

    def fail_call(self) -> None:
        with self._lock:
            self.in_flight = False

    # -- same-source dedupe --------------------------------------------------
    def needs_authorization(self, source_handle: str, task: str) -> bool:
        with self._lock:
            return (source_handle, task) in self._recent_calls

    def record_call(self, source_handle: str, task: str) -> None:
        with self._lock:
            self._recent_calls.append((source_handle, task))

    def authorize_source_task(self, source_handle: str, task: str) -> None:
        """Explicit user authorization for a repeated source+task."""
        with self._lock:
            while (source_handle, task) in self._recent_calls:
                self._recent_calls.remove((source_handle, task))

    # -- attachment handles --------------------------------------------------
    def register_attachment(self, handle: str, path: str) -> None:
        with self._lock:
            self.attachment_allowlist[handle] = path

    def resolve_attachment(self, handle: str) -> Optional[str]:
        with self._lock:
            return self.attachment_allowlist.get(handle)

    def has_attachment(self, handle: str) -> bool:
        with self._lock:
            return handle in self.attachment_allowlist

    def revoke_attachment(self, handle: str) -> None:
        with self._lock:
            self.attachment_allowlist.pop(handle, None)

    # -- OCR result registry -------------------------------------------------
    def register_ocr_result(self, handle: str, file_path: str) -> None:
        with self._lock:
            self.ocr_results[handle] = file_path

    def resolve_ocr_result(self, handle: str) -> Optional[str]:
        with self._lock:
            return self.ocr_results.get(handle)

    def revoke_ocr_result(self, handle: str) -> None:
        with self._lock:
            self.ocr_results.pop(handle, None)

    # -- snapshot for tests / reporting --------------------------------------
    def snapshot(self) -> Dict[str, object]:
        with self._lock:
            return {
                "enabled": self.enabled,
                "per_turn_used": self.per_turn_used,
                "per_session_used": self.per_session_used,
                "in_flight": self.in_flight,
                "attachment_handles": sorted(self.attachment_allowlist),
                "ocr_handles": sorted(self.ocr_results),
                "recent_calls": list(self._recent_calls),
            }


# Process-local singleton.
vision_session_state = _VisionSessionState()
