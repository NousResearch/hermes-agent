"""Native ``watch`` tool — configurable background polling with conditions.

Implements the feature requested in
NousResearch/hermes-agent#56694 ("Native watch tool with configurable
intervals and conditions").

The tool polls a shell ``command`` on an ``interval`` and records observations.
An optional ``condition`` string gates which observations are surfaced/notify'd.
A ``duration`` bounds the total watch window.

Design notes
------------
* Registration uses the standard ``registry.register(...)`` self-registration
  pattern (same as ``cronjob_tools.py``), so ``discover_builtin_tools`` picks
  this module up automatically.
* The poll loop runs on a dedicated daemon thread so it outlives
  ``registry.dispatch`` / ``_run_async``'s disposable worker loop. A
  ``threading.Event`` on the session handle is the cancel signal; the
  worker is pinned as ``handle["_task"]``.
* Lifecycle: ``action="list"`` / ``action="stop"`` enumerate or cancel
  watches. ``stop_all_watches(agent)`` plus a ``weakref.finalize`` on the
  agent tears the workers down when the agent is collected.
* ``condition`` is a tiny, safe expression language — ``contains "x"``,
  ``not contains "x"``, ``equals "x"``, ``matches "regex"``, or a bare
  substring — deliberately avoiding ``eval``/``exec`` on agent input.
* Notifications go through ``agent.notify`` when present, otherwise
  ``agent._emit_status`` (the AIAgent lifecycle hook).
"""

from __future__ import annotations

import json
import re
import subprocess
import threading
import time
import uuid
import weakref
from typing import Any, Dict, List, Optional

from tools.registry import registry


# ---------------------------------------------------------------------------
# Condition evaluation (pure, unit-testable)
# ---------------------------------------------------------------------------


def _eval_condition(condition: str, output: str) -> bool:
    """Evaluate a ``condition`` string against command ``output``.

    Supported forms (case-insensitive keyword prefix):
      * ``contains "foo"`` / ``"foo"``        -> output contains foo
      * ``not contains "foo"``                -> output does NOT contain foo
      * ``equals "foo"``                      -> output (stripped) == foo
      * ``matches "regex"``                   -> re.search(regex, output)
    Returns ``True`` when ``condition`` is empty/whitespace (unconditional).
    """
    if not condition or not condition.strip():
        return True
    cond = condition.strip()

    m = re.match(r'^not\s+contains\s+["\'](.+?)["\']\s*$', cond, re.IGNORECASE)
    if m:
        return m.group(1) not in output

    m = re.match(r'^contains\s+["\'](.+?)["\']\s*$', cond, re.IGNORECASE)
    if m:
        return m.group(1) in output

    m = re.match(r'^equals\s+["\'](.+?)["\']\s*$', cond, re.IGNORECASE)
    if m:
        return output.strip() == m.group(1)

    m = re.match(r'^matches\s+["\'](.+?)["\']\s*$', cond, re.IGNORECASE)
    if m:
        try:
            return re.search(m.group(1), output) is not None
        except re.error:
            return False

    # bare substring -> contains
    inner = cond.strip('"\'')
    return inner in output


def _parse_duration(value: Any) -> int:
    """Parse a duration string/int into seconds.

    Accepts ``"24h"``, ``"30m"``, ``"45s"``, or a bare int (seconds).
    Returns 0 for unparseable input (caller treats 0 as 'single tick').
    """
    if value is None:
        return 0
    if isinstance(value, (int, float)):
        return int(value)
    s = str(value).strip().lower()
    m = re.match(r'^(\d+)\s*(h|m|s)?$', s)
    if not m:
        return 0
    n = int(m.group(1))
    unit = m.group(2)
    if unit == "h":
        return n * 3600
    if unit == "m":
        return n * 60
    return n  # 's' or bare -> seconds


def _plan_ticks(interval: int, duration: Optional[int]) -> int:
    """How many ticks to schedule.

    ``duration`` is in seconds; ``interval`` is seconds between ticks.
    Returns at least 1. A hard cap (enforced in the loop) prevents a forgotten
    duration from spinning forever.
    """
    if not duration or duration <= 0:
        return 1
    return max(1, int(duration // interval) + (1 if duration % interval else 0))


def _parse_bounded(value: Any, default: int, lo: int, hi: int) -> int:
    """Parse a duration-like tool arg and clamp it.

    Uses ``_parse_duration`` so ``"30s"`` / ``"5m"`` work. Unparseable input
    (e.g. ``"oops"``) falls back to ``default`` instead of raising.
    """
    n = _parse_duration(value) if value is not None else 0
    if n <= 0:
        n = default
    return max(lo, min(hi, n))


def _parse_bool(value: Any, default: bool = True) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    s = str(value).strip().lower()
    if s in {"1", "true", "yes", "on"}:
        return True
    if s in {"0", "false", "no", "off"}:
        return False
    return default


def _should_notify(
    *,
    condition: Optional[str],
    triggered: bool,
    prev_triggered: bool,
    tick: int,
) -> bool:
    """When to fire a user-visible notification.

    * Condition set: rising edge only (false → true).
    * Unconditional: first tick only (avoids up to 1000 notifies).
    """
    if condition and str(condition).strip():
        return bool(triggered) and not prev_triggered
    return tick == 1


def _notify_agent(agent: Any, message: str) -> None:
    fn = getattr(agent, "notify", None)
    if not callable(fn):
        fn = getattr(agent, "_emit_status", None)
    if not callable(fn):
        return
    try:
        fn(message)
    except Exception:
        pass


def _public_view(handle: Dict[str, Any], *, include_observations: bool = False) -> Dict[str, Any]:
    skip = {"_task", "_stop"}
    if not include_observations:
        skip = skip | {"observations"}
    view = {k: v for k, v in handle.items() if k not in skip}
    view["observation_count"] = len(handle.get("observations") or [])
    return view


def _cancel_session(sess: Dict[str, Any]) -> None:
    stop = sess.get("_stop")
    if stop is not None and hasattr(stop, "set"):
        stop.set()
    sess["status"] = "stopped"


def stop_all_watches(agent: Any) -> List[str]:
    """Cancel every watch on *agent*. Safe to call from teardown/finalize."""
    sessions = getattr(agent, "_watch_sessions", None) if agent is not None else None
    if not sessions:
        return []
    stopped = []
    for wid in list(sessions.keys()):
        sess = sessions.get(wid) or {}
        _cancel_session(sess)
        stopped.append(wid)
        sessions.pop(wid, None)
    return stopped


def _stop_sessions_dict(sessions: Dict[str, Any]) -> None:
    for sess in list((sessions or {}).values()):
        _cancel_session(sess)
    if sessions is not None:
        sessions.clear()


def _ensure_sessions(agent: Any) -> Dict[str, Any]:
    sessions = getattr(agent, "_watch_sessions", None)
    if sessions is None:
        sessions = {}
        agent._watch_sessions = sessions
        # Finalize must not capture *agent* (that would pin it forever).
        weakref.finalize(agent, _stop_sessions_dict, sessions)
    return sessions


# ---------------------------------------------------------------------------
# Synchronous command runner (unit-testable without an event loop)
# ---------------------------------------------------------------------------


def run_once(command: str, timeout: int = 30) -> str:
    """Run ``command`` once and return combined stdout/stderr text.

    Raises on timeout are caught and returned as a marker string so callers
    can record the outcome uniformly.
    """
    try:
        proc = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return (proc.stdout or "") + (proc.stderr or "")
    except subprocess.TimeoutExpired:
        return f"<timeout after {timeout}s>"
    except Exception as exc:  # pragma: no cover - defensive
        return f"<error: {exc}>"


def _make_observation(
    *,
    watch_id: str,
    tick: int,
    command: str,
    output: str,
    condition: Optional[str],
    triggered: bool,
) -> Dict[str, Any]:
    return {
        "watch_id": watch_id,
        "tick": tick,
        "command": command,
        "condition": condition,
        "triggered": triggered,
        "output": output,
        "ts": time.time(),
    }


# ---------------------------------------------------------------------------
# Handler
# ---------------------------------------------------------------------------


def _start_poll_thread(handle: Dict[str, Any], agent: Any) -> None:
    stop = handle["_stop"]
    command = handle["command"]
    interval = handle["interval"]
    timeout = handle["timeout"]
    condition = handle.get("condition")
    notify = handle.get("notify", True)
    max_ticks = handle["planned_ticks"]
    seconds = handle["duration_seconds"]
    watch_id = handle["watch_id"]

    def _poll() -> None:
        tick = 0
        deadline = time.time() + seconds
        prev_triggered = False
        try:
            while not stop.is_set() and time.time() < deadline and tick < max_ticks:
                tick += 1
                out = run_once(command, timeout)
                triggered = _eval_condition(condition or "", out)
                handle["observations"].append(
                    _make_observation(
                        watch_id=watch_id,
                        tick=tick,
                        command=command,
                        output=out,
                        condition=condition,
                        triggered=triggered,
                    )
                )
                if notify and _should_notify(
                    condition=condition,
                    triggered=triggered,
                    prev_triggered=prev_triggered,
                    tick=tick,
                ):
                    label = condition if condition else "unconditional"
                    _notify_agent(
                        agent,
                        f"[watch:{watch_id}] tick {tick} matched "
                        f"condition={label!r}: {out[:200]}",
                    )
                if condition and str(condition).strip() and triggered and not prev_triggered:
                    handle["status"] = "matched"
                    return
                prev_triggered = triggered
                if stop.wait(interval):
                    break
            if handle.get("status") == "running":
                handle["status"] = "stopped" if stop.is_set() else "completed"
        except Exception as exc:  # pragma: no cover - defensive
            handle["status"] = "error"
            handle["error"] = str(exc)

    worker = threading.Thread(
        target=_poll,
        name=f"hermes-watch-{watch_id}",
        daemon=True,
    )
    handle["_task"] = worker
    worker.start()


def watch(args: Dict[str, Any], **kw: Any) -> str:
    """Watch a command's output over time and surface observations.

    Sync handler on purpose: async dispatch is bridged through
    ``model_tools._run_async``, whose gateway path cancels leftover tasks
    when the handler returns. Background polling therefore lives on a
    daemon thread, not on that disposable loop.

    The agent object is injected via the ``agent=`` kwarg by
    ``handle_function_call``.
    """
    agent = kw.get("agent") or kw.get("ctx")
    action = str(args.get("action", "start")).lower()

    if action in ("list", "stop"):
        sessions = getattr(agent, "_watch_sessions", None) if agent is not None else None
        if sessions is None:
            sessions = {}
        if action == "list":
            running = [
                _public_view(s)
                for s in sessions.values()
                if s.get("status") == "running"
            ]
            return json.dumps(
                {"action": "list", "count": len(running), "watches": running},
                ensure_ascii=False,
            )
        target = str(args.get("watch_id", "all"))
        stopped = []
        for wid in list(sessions.keys()):
            if target == "all" or wid == target:
                sess = sessions.get(wid, {})
                _cancel_session(sess)
                stopped.append(wid)
                sessions.pop(wid, None)
        return json.dumps({"action": "stop", "stopped": stopped}, ensure_ascii=False)

    command = str(args.get("command", "") or "")
    if not command.strip():
        return json.dumps({"error": "command is required to start a watch"}, ensure_ascii=False)

    interval = _parse_bounded(args.get("interval", 60), default=60, lo=5, hi=3600)
    condition = args.get("condition")
    notify = _parse_bool(args.get("notify", True), default=True)
    duration = args.get("duration")
    timeout = _parse_bounded(args.get("timeout", 30), default=30, lo=1, hi=3600)

    seconds = _parse_duration(duration) if duration else interval
    if seconds <= 0:
        seconds = interval
    max_ticks = min(_plan_ticks(interval, seconds), 1000)

    watch_id = f"watch_{uuid.uuid4().hex[:12]}"
    handle = {
        "watch_id": watch_id,
        "command": command,
        "interval": interval,
        "timeout": timeout,
        "condition": condition,
        "notify": notify,
        "duration_seconds": seconds,
        "planned_ticks": max_ticks,
    }

    if agent is not None:
        sessions = _ensure_sessions(agent)
        handle["status"] = "running"
        handle["observations"] = []
        handle["_stop"] = threading.Event()
        handle["_task"] = None
        sessions[watch_id] = handle
        _start_poll_thread(handle, agent)
    else:
        out = run_once(command, timeout)
        triggered = _eval_condition(condition or "", out)
        handle["observations"] = [
            _make_observation(
                watch_id=watch_id,
                tick=1,
                command=command,
                output=out,
                condition=condition,
                triggered=triggered,
            )
        ]
        handle["status"] = "completed"
    return json.dumps(_public_view(handle, include_observations=True), ensure_ascii=False)


WATCH_SCHEMA = {
    "type": "function",
    "function": {
        "name": "watch",
        "description": (
            "Poll a shell command on an interval and surface observations "
            "when an optional condition is met. Useful for monitoring a "
            "service, watching for a file to appear, or polling an API — "
            "without blocking the conversation. Use action='list' / 'stop' "
            "to manage running watches (command not required for those)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "description": (
                        "What to do: 'start' (default, create a watch), "
                        "'list' (show active watches), or 'stop' "
                        "(cancel a watch by watch_id, or 'all')."
                    ),
                    "enum": ["start", "list", "stop"],
                    "default": "start",
                },
                "watch_id": {
                    "type": "string",
                    "description": "Watch ID to stop (for action='stop').",
                },
                "command": {
                    "type": "string",
                    "description": "Shell command to run each tick (required for start).",
                },
                "interval": {
                    "type": "string",
                    "description": (
                        "Seconds between ticks (5-3600), or a duration "
                        "string like '30s' / '5m'."
                    ),
                    "default": "60",
                },
                "condition": {
                    "type": "string",
                    "description": (
                        "Optional trigger: 'contains \"x\"', "
                        "'not contains \"x\"', 'equals \"x\"', "
                        "'matches \"regex\"', or a bare substring."
                    ),
                },
                "notify": {
                    "type": "boolean",
                    "description": "Surface a notification on match (rising edge / first tick).",
                    "default": True,
                },
                "duration": {
                    "type": "string",
                    "description": (
                        "Total window: '24h', '30m', or raw seconds. "
                        "Defaults to one interval."
                    ),
                },
                "timeout": {
                    "type": "string",
                    "description": "Per-tick command timeout (seconds or '30s').",
                    "default": "30",
                },
            },
            "required": [],
        },
    },
}


registry.register(
    name="watch",
    toolset="watch",
    schema=WATCH_SCHEMA,
    handler=watch,
    is_async=False,
)
