"""Log tool calls that stay outstanding for a long time (PHASE=tool_wait_long).

Why: when a gateway session stops answering, the usual cause is a turn thread parked in a
long tool call (a foreground polling loop, a watch script, a slow refresh). Nothing in the logs
says so; finding it takes a py-spy of the gateway process.

This module makes that state grep-able. Every call dispatched through
``tools.registry.ToolRegistry.dispatch`` is registered while it runs; one daemon thread emits
one WARNING line per call at every multiple of ``TOOL_WAIT_LONG_INTERVAL_S`` (5 minutes) it is
still outstanding:

    PHASE=tool_wait_long tool=terminal elapsed=600s session=agent:main:... thread=... cmd='for i in ...'

and one ``PHASE=tool_wait_long_done`` line when a reported call finishes. Calls that finish
before the first interval cost a dict insert and delete.
"""
from __future__ import annotations

import itertools
import json
import logging
import threading
import time
from contextlib import contextmanager
from typing import Any, Dict, Iterator

logger = logging.getLogger(__name__)

# Report cadence. Module constant (not an env var) so tests can shrink it.
TOOL_WAIT_LONG_INTERVAL_S: float = 300.0
# Keep each log line bounded; a huge execute_code body is not useful here.
MAX_CMD_CHARS = 240

_DESCRIBE_KEYS = ("command", "code", "url", "query", "goal", "prompt", "path")

_lock = threading.Lock()
_outstanding: Dict[int, Dict[str, Any]] = {}
_ids = itertools.count(1)
_thread: "threading.Thread | None" = None
# Set on every registration so the reporter re-reads the cadence promptly
# instead of finishing a sleep computed from a stale interval.
_wake = threading.Event()


def describe_call(name: str, args: Any) -> str:
    """One-line, bounded description of what the call is doing."""
    text = ""
    if isinstance(args, dict):
        for key in _DESCRIBE_KEYS:
            value = args.get(key)
            if isinstance(value, str) and value.strip():
                text = value
                break
        else:
            try:
                text = json.dumps(args, default=str, ensure_ascii=False)
            except Exception:
                text = str(args)
    elif args is not None:
        text = str(args)
    text = " ".join(text.split())
    if len(text) > MAX_CMD_CHARS:
        text = text[:MAX_CMD_CHARS] + "\u2026"
    return text


def _session_key() -> str:
    try:
        from tools.approval_context import get_current_session_key

        return get_current_session_key(default="-") or "-"
    except Exception:
        return "-"


def outstanding_count() -> int:
    with _lock:
        return len(_outstanding)


def _scan_once(now: float) -> None:
    interval = float(TOOL_WAIT_LONG_INTERVAL_S)
    due = []
    with _lock:
        for rec in _outstanding.values():
            elapsed = now - rec["start"]
            if elapsed >= rec["next_report"]:
                while rec["next_report"] <= elapsed:
                    rec["next_report"] += interval
                rec["reported"] = True
                due.append((rec, elapsed))
    for rec, elapsed in due:
        logger.warning(
            "PHASE=tool_wait_long tool=%s elapsed=%ds session=%s thread=%s cmd=%r",
            rec["tool"], int(elapsed), rec["session"], rec["thread"], rec["cmd"],
        )


def _run() -> None:
    while True:
        interval = float(TOOL_WAIT_LONG_INTERVAL_S)
        _wake.wait(min(max(interval / 10.0, 0.02), 30.0))
        _wake.clear()
        try:
            _scan_once(time.monotonic())
        except Exception:  # never let the reporter die
            logger.debug("tool_wait_long scan failed", exc_info=True)


def _ensure_thread() -> None:
    global _thread
    if _thread is not None and _thread.is_alive():
        return
    with _lock:
        if _thread is not None and _thread.is_alive():
            return
        _thread = threading.Thread(target=_run, name="tool-wait-watchdog", daemon=True)
        _thread.start()


@contextmanager
def track_tool_call(name: str, args: Any) -> Iterator[None]:
    """Register a tool call as outstanding for the duration of the block.

    Never raises: bookkeeping failures must not affect the tool call itself.
    """
    cid = None
    try:
        _ensure_thread()
        cid = next(_ids)
        start = time.monotonic()
        with _lock:
            _outstanding[cid] = {
                "tool": name,
                "cmd": describe_call(name, args),
                "session": _session_key(),
                "thread": threading.current_thread().name,
                "start": start,
                "next_report": float(TOOL_WAIT_LONG_INTERVAL_S),
                "reported": False,
            }
        _wake.set()
    except Exception:
        logger.debug("tool_wait_long register failed", exc_info=True)
    try:
        yield
    finally:
        if cid is not None:
            with _lock:
                rec = _outstanding.pop(cid, None)
            if rec and rec.get("reported"):
                logger.warning(
                    "PHASE=tool_wait_long_done tool=%s elapsed=%ds session=%s cmd=%r",
                    rec["tool"], int(time.monotonic() - rec["start"]),
                    rec["session"], rec["cmd"],
                )
