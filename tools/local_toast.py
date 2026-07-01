"""Local antenna toast — best-effort OS-native popup when an agent needs user input.

Fires an HTTP POST to the loopback antenna's ``/peer/notify`` endpoint, which
maps to ``prometheus_client.notify.notify()`` (Windows pystray toast + voice +
tray icon fallback). Called from ``tools/clarify_tool.py`` and
``tools/approval.py`` right before the agent thread blocks waiting for user
input.

**Design invariants:**

1. **Fail open.** A broken antenna, network hiccup, or firewall block MUST NOT
   prevent the underlying clarify/approval from proceeding. Every call is
   wrapped in ``try/except Exception``; the caller never sees a raise.
2. **Fire and forget.** The POST runs on a daemon thread with a short timeout.
   The agent thread continues to the blocking input() immediately after
   dispatch — the toast is a courtesy notification, not a synchronous barrier.
3. **Rate-limited.** Repeat calls from the same session within 3s coalesce
   into a single toast to avoid tray spam if an approval loop retries.
4. **Zero deps.** stdlib ``urllib.request`` only. Same dependency surface as
   the rest of ``tools/`` — no requests, no httpx.

**Endpoint contract:**

``POST http://127.0.0.1:7634/peer/notify``
``Content-Type: application/json``
``{"title": "<str>", "message": "<str>", "level": "info|success|warn|error"}``

Antenna responds ``{"ok": true}`` on success. Any non-2xx response is
logged and swallowed.
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
import urllib.error
import urllib.request
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Loopback antenna endpoint. Antenna binds 127.0.0.1:7634 by default.
# Override with HERMES_LOCAL_TOAST_URL for non-standard deployments.
_DEFAULT_URL = "http://127.0.0.1:7634/peer/notify"
_TIMEOUT_SEC = 1.5

# Coalesce window — if the same (profile, kind) fires more than once within
# this window, only the first toast dispatches. Prevents tray spam from an
# approval loop or a clarify that gets retried.
_COALESCE_WINDOW_SEC = 3.0

# Kill-switch. If set (any non-empty value), local_toast becomes a no-op.
# Useful for headless CI runs and for the antenna's OWN internal calls (which
# would otherwise loop back through this module and infinite-recurse).
_DISABLE_ENV = "HERMES_LOCAL_TOAST_DISABLE"

# ---------------------------------------------------------------------------
# State (module-level, thread-safe via lock)
# ---------------------------------------------------------------------------

_lock = threading.Lock()
_last_fired: dict = {}  # key -> monotonic timestamp


def _endpoint() -> str:
    return os.environ.get("HERMES_LOCAL_TOAST_URL", _DEFAULT_URL)


def _disabled() -> bool:
    return bool(os.environ.get(_DISABLE_ENV, "").strip())


def _should_fire(key: str) -> bool:
    """Return True if this key hasn't fired recently.

    Updates ``_last_fired`` when returning True — the caller is expected to
    immediately dispatch, so we mark the timestamp here to close the race.
    """
    now = time.monotonic()
    with _lock:
        last = _last_fired.get(key, 0.0)
        if (now - last) < _COALESCE_WINDOW_SEC:
            return False
        _last_fired[key] = now
        # Prune stale entries so the dict doesn't grow unbounded across a
        # long-lived agent session. Anything older than 10x the window is dead.
        cutoff = now - (_COALESCE_WINDOW_SEC * 10.0)
        for k, ts in list(_last_fired.items()):
            if ts < cutoff:
                _last_fired.pop(k, None)
        return True


def _resolve_persona() -> str:
    """Best-effort discovery of the current profile / persona name.

    Preference order:
      1. HERMES_ACTIVE_PROFILE env var (set by cli.py when known)
      2. hermes_cli.profiles.get_active_profile_name() (canonical)
      3. HERMES_PROFILE env var (legacy)
      4. "hermes" as generic fallback

    Never raises. Any lookup failure returns the fallback.
    """
    try:
        v = os.environ.get("HERMES_ACTIVE_PROFILE", "").strip()
        if v:
            return v
    except Exception:  # noqa: BLE001
        pass
    try:
        from hermes_cli.profiles import get_active_profile_name  # type: ignore
        v = (get_active_profile_name() or "").strip()
        if v:
            return v
    except Exception:  # noqa: BLE001
        pass
    try:
        v = os.environ.get("HERMES_PROFILE", "").strip()
        if v:
            return v
    except Exception:  # noqa: BLE001
        pass
    return "hermes"


def _post_toast(title: str, message: str, level: str) -> None:
    """Fire the actual POST. Runs on a daemon thread. Any failure is swallowed."""
    try:
        body = json.dumps(
            {"title": title, "message": message, "level": level},
            ensure_ascii=False,
        ).encode("utf-8")
        req = urllib.request.Request(
            _endpoint(),
            data=body,
            method="POST",
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=_TIMEOUT_SEC) as resp:
            # Drain body so the connection can be cleanly closed on Windows.
            resp.read()
    except (urllib.error.URLError, urllib.error.HTTPError, OSError, TimeoutError) as e:
        # Antenna not running / not on this host / firewall blocked — expected in
        # many environments. Log at debug so we don't spam the agent output.
        logger.debug("local_toast: antenna unreachable (%s): %s", type(e).__name__, e)
    except Exception as e:  # noqa: BLE001
        logger.debug("local_toast: unexpected error: %s", e)


def notify_input_needed(
    kind: str,
    detail: str = "",
    *,
    persona: Optional[str] = None,
    session_label: Optional[str] = None,
    level: str = "info",
) -> None:
    """Fire a fire-and-forget local toast saying "<persona> needs input".

    Never raises. Never blocks the caller (dispatches on a daemon thread).
    Silently no-ops when ``HERMES_LOCAL_TOAST_DISABLE`` is set.

    Args:
        kind: Short tag for the kind of block ("clarify", "approval",
            "sudo-approval"). Used both in the toast title and in the
            coalescing key.
        detail: Optional extra text (e.g. the first 80 chars of the question or
            the command awaiting approval). Truncated at 200 chars.
        persona: Override the persona-name resolution. Rarely used.
        session_label: Optional session identifier for the toast body (e.g.
            "chat", "sub-agent", "cron"). Falls back to "chat".
        level: pystray toast level. Only "info" and "warn" render distinctly on
            Windows; error is used for hard failures.
    """
    if _disabled():
        return
    try:
        persona = persona or _resolve_persona()
        session_label = session_label or "chat"
        key = f"{persona}:{kind}"
        if not _should_fire(key):
            return
        title = f"{persona}: needs input"
        body_prefix = {
            "clarify": "Question awaiting your reply",
            "approval": "Command awaiting approval",
            "sudo-approval": "Sudo command awaiting approval",
        }.get(kind, "Input needed")
        detail_trimmed = (detail or "").strip().replace("\n", " ")[:200]
        if detail_trimmed:
            message = f"{body_prefix} in {session_label}: {detail_trimmed}"
        else:
            message = f"{body_prefix} in {session_label}"
        t = threading.Thread(
            target=_post_toast,
            args=(title, message, level),
            daemon=True,
            name="hermes-local-toast",
        )
        t.start()
    except Exception as e:  # noqa: BLE001
        # Absolute fail-open: no exception ever propagates from this module.
        logger.debug("local_toast.notify_input_needed swallowed error: %s", e)
