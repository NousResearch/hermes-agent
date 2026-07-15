"""Gateway-side client-tool relay primitive (blocking event-based queue).

Split-runtime support.  When a run is started via ``POST /v1/runs`` with a
``tools`` array (client/shell-supplied tools) and the API server has
``api_server.split_runtime`` enabled, those tool names are merged into the
agent's tool list but are NOT executed on the host.  Instead, when the model
calls one of them the agent thread SUSPENDS here — mirroring how ``clarify``
and host-side ``approval`` block the worker thread — the gateway emits a
``tool_call.request`` SSE event to the client shell, the shell executes the
tool locally (its own confirmation gate + native action) and returns the
result via ``POST /v1/runs/{run_id}/tool_result``, and the agent thread
resumes with that result as the tool output.  To the model this is
indistinguishable from a locally-executed tool.

This is a direct structural sibling of ``tools.clarify_gateway``:

  * store a pending client-tool call (keyed by ``call_id`` = the model's
    ``tool_call_id``),
  * block the agent thread on an ``Event`` (polled in 1s slices so the
    inactivity heartbeat keeps firing),
  * resolve the wait when the HTTP layer fires
    ``resolve_client_tool(call_id, result_json)``,
  * support a timeout so a shell that never answers does NOT hang the agent
    thread forever (which would pin the gateway's running-agent guard),
  * ``clear_session(run_id)`` on run end/stop so blocked threads unwind.

State is module-level (same shape as ``tools.approval`` / ``tools.clarify_gateway``)
so the api_server can call ``resolve_client_tool`` without a back-reference
to the agent.  The gateway ``session_key`` used here is the per-run
``run_id`` (the same key api_server uses for approval isolation).
"""

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


# =========================================================================
# Module-level state
# =========================================================================

@dataclass
class _ClientToolEntry:
    """One pending client-tool call inside a split-runtime run."""
    call_id: str
    session_key: str  # == run_id
    name: str
    arguments: Dict[str, Any]
    event: threading.Event = field(default_factory=threading.Event)
    result: Optional[str] = None

    def signature(self) -> Dict[str, object]:
        return {
            "call_id": self.call_id,
            "run_id": self.session_key,
            "name": self.name,
            "arguments": dict(self.arguments) if self.arguments else {},
        }


_lock = threading.RLock()
# call_id → _ClientToolEntry  (primary lookup for the tool_result POST)
_entries: Dict[str, _ClientToolEntry] = {}
# session_key (run_id) → list[call_id]  (FIFO; for session cleanup)
_session_index: Dict[str, List[str]] = {}


# =========================================================================
# Public API — agent-thread side
# =========================================================================

def register(
    call_id: str,
    session_key: str,
    name: str,
    arguments: Optional[Dict[str, Any]],
) -> _ClientToolEntry:
    """Register a pending client-tool call and return the entry.

    The caller (the invoke_tool interception) then fires the run's notify
    callback (→ ``tool_call.request`` SSE) and blocks on
    ``wait_for_result(call_id, timeout)``.
    """
    entry = _ClientToolEntry(
        call_id=call_id,
        session_key=session_key,
        name=name,
        arguments=dict(arguments) if arguments else {},
    )
    with _lock:
        _entries[call_id] = entry
        _session_index.setdefault(session_key, []).append(call_id)
    return entry


def wait_for_result(call_id: str, timeout: float) -> Optional[str]:
    """Block on the entry's event until resolved or timeout fires.

    Polls in 1-second slices so the agent's inactivity heartbeat keeps
    firing — without this a long ``Event.wait(timeout=300)`` blocks the
    thread with zero activity touches and the gateway's inactivity watchdog
    kills the agent while the shell is still executing the tool.

    Returns the resolved result string, or ``None`` on timeout.
    """
    with _lock:
        entry = _entries.get(call_id)
    if entry is None:
        return None

    try:
        from tools.environments.base import touch_activity_if_due
    except Exception:  # pragma: no cover - optional
        touch_activity_if_due = None

    deadline = time.monotonic() + max(timeout, 0.0)
    activity_state = {"last_touch": time.monotonic(), "start": time.monotonic()}
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        if entry.event.wait(timeout=min(1.0, remaining)):
            break
        if touch_activity_if_due is not None:
            touch_activity_if_due(activity_state, "waiting for client tool result")

    with _lock:
        # Remove from indices regardless of resolution outcome.
        _entries.pop(call_id, None)
        ids = _session_index.get(entry.session_key)
        if ids and call_id in ids:
            ids.remove(call_id)
            if not ids:
                _session_index.pop(entry.session_key, None)

    return entry.result


# =========================================================================
# Public API — gateway / HTTP side
# =========================================================================

def resolve_client_tool(call_id: str, result_json: str) -> bool:
    """Unblock the agent thread waiting on ``call_id``.

    ``result_json`` is the tool-result string handed back to the model.
    Returns True if a pending entry was found and resolved, False otherwise
    (already resolved, expired, or never existed → caller returns 409).
    """
    with _lock:
        entry = _entries.get(call_id)
        if entry is None:
            return False
    entry.result = str(result_json) if result_json is not None else ""
    entry.event.set()
    return True


def get_pending_for_session(session_key: str) -> Optional[_ClientToolEntry]:
    """Return the oldest pending client-tool entry for a run, or None."""
    with _lock:
        ids = _session_index.get(session_key) or []
        for cid in ids:
            entry = _entries.get(cid)
            if entry is not None:
                return entry
        return None


def has_pending(session_key: str) -> bool:
    """Return True when this run has at least one pending client-tool call."""
    with _lock:
        ids = _session_index.get(session_key) or []
        return any(_entries.get(cid) is not None for cid in ids)


def clear_session(session_key: str) -> int:
    """Resolve and drop every pending client-tool call for a run.

    Used by run-boundary cleanup (run completion, ``/stop``, gateway
    shutdown) so blocked agent threads don't hang past the end of the run.
    Each cancelled entry is resolved with an error tool-result so the model
    gets a coherent observation rather than a raw ``None``.  Returns the
    number of entries cancelled.
    """
    with _lock:
        ids = list(_session_index.pop(session_key, []) or [])
        entries = [_entries.pop(cid, None) for cid in ids]
    cancelled = 0
    for entry in entries:
        if entry is None:
            continue
        if entry.result is None:
            entry.result = json.dumps(
                {"error": "client tool call cancelled (run ended)"},
                ensure_ascii=False,
            )
        entry.event.set()
        cancelled += 1
    return cancelled


# =========================================================================
# Config
# =========================================================================

def get_client_tool_timeout() -> int:
    """Read the client-tool relay timeout (seconds) from config.

    Reads ``api_server.gateway_timeout`` (the same knob that bounds a
    host-side approval round-trip), defaulting to 300s.  Long enough for a
    user to see the shell's confirmation gate and act; short enough that a
    dead shell eventually unblocks the agent thread.
    """
    try:
        from hermes_cli.config import load_config
        cfg = load_config() or {}
        api_cfg = cfg.get("api_server", {}) or {}
        return int(api_cfg.get("gateway_timeout", 300))
    except Exception:
        return 300


# =========================================================================
# Per-run notify hook (gateway → SSE bridge)
# =========================================================================
# Mirrors tools.approval's _gateway_notify_cbs and clarify_gateway's
# _notify_cbs: the api_server registers a per-run callback that pushes a
# ``tool_call.request`` event onto the run's SSE queue.  The callback runs
# on the agent thread and schedules the put on the event loop.

_notify_cbs: Dict[str, Callable[[_ClientToolEntry], None]] = {}


def register_notify(session_key: str, cb: Callable[[_ClientToolEntry], None]) -> None:
    """Register a per-run notify callback used by the invoke_tool interception."""
    with _lock:
        _notify_cbs[session_key] = cb


def unregister_notify(session_key: str) -> None:
    """Drop the per-run notify callback and cancel any pending client-tool calls."""
    with _lock:
        _notify_cbs.pop(session_key, None)
    # Cancel any pending entries so blocked threads unwind when the run ends.
    clear_session(session_key)


def get_notify(session_key: str) -> Optional[Callable[[_ClientToolEntry], None]]:
    with _lock:
        return _notify_cbs.get(session_key)
