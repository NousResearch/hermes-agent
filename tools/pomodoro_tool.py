"""
Pomodoro tool for Hermes Agent.

A focused work timer based on the Pomodoro Technique. Helps users stay
productive by alternating work sessions with short breaks.

Integrates with notification_tool (if available) to send desktop alerts
when sessions start, pause, or complete.

Dependencies: none (stdlib only)
"""

import json
import threading
from datetime import datetime, timedelta
from typing import Optional

# ---------------------------------------------------------------------------
# In-memory session store (persists for the lifetime of the process)
# ---------------------------------------------------------------------------

_sessions: dict = {}
_timers:   dict = {}
_lock = threading.Lock()

# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

def register(registry):
    """Register pomodoro tools with the tool registry."""

    registry.register(
        name="pomodoro_start",
        description=(
            "Start a Pomodoro work session. The user works for a focused period "
            "(default 25 min), then takes a short break (default 5 min). "
            "Use this when the user wants to focus, be productive, or asks for "
            "a pomodoro/timer/focus session. Returns a session ID to track progress. "
            "Sends a desktop notification when the session ends (if notify tool available)."
        ),
        parameters={
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "description": "What the user is working on (e.g. 'writing code', 'studying').",
                },
                "work_minutes": {
                    "type": "integer",
                    "description": "Work session duration in minutes. Default: 25.",
                },
                "break_minutes": {
                    "type": "integer",
                    "description": "Break duration in minutes. Default: 5.",
                },
                "sessions": {
                    "type": "integer",
                    "description": "Number of pomodoro rounds to complete. Default: 4.",
                },
            },
            "required": ["task"],
        },
        handler=handle_pomodoro_start,
    )

    registry.register(
        name="pomodoro_status",
        description=(
            "Check the status of an active Pomodoro session. "
            "Returns current phase (work/break), time remaining, and completed rounds. "
            "Use this when the user asks 'how much time left?' or 'pomodoro status'."
        ),
        parameters={
            "type": "object",
            "properties": {
                "session_id": {
                    "type": "string",
                    "description": "Session ID returned by pomodoro_start. If omitted, returns the latest session.",
                },
            },
            "required": [],
        },
        handler=handle_pomodoro_status,
    )

    registry.register(
        name="pomodoro_stop",
        description=(
            "Stop an active Pomodoro session early. "
            "Use this when the user wants to cancel, take an unplanned break, "
            "or says 'stop pomodoro' / 'cancel timer'."
        ),
        parameters={
            "type": "object",
            "properties": {
                "session_id": {
                    "type": "string",
                    "description": "Session ID to stop. If omitted, stops the latest session.",
                },
            },
            "required": [],
        },
        handler=handle_pomodoro_stop,
    )

    registry.register(
        name="pomodoro_history",
        description=(
            "Show a summary of all Pomodoro sessions this process run. "
            "Includes tasks worked on, total focused time, and completion rate."
        ),
        parameters={
            "type": "object",
            "properties": {},
            "required": [],
        },
        handler=handle_pomodoro_history,
    )


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------

def handle_pomodoro_start(
    task: str,
    work_minutes: int = 25,
    break_minutes: int = 5,
    sessions: int = 4,
) -> str:
    work_minutes  = max(1, min(work_minutes, 120))
    break_minutes = max(1, min(break_minutes, 60))
    sessions      = max(1, min(sessions, 10))

    session_id = f"pomo_{datetime.now().strftime('%H%M%S')}"
    start_time = datetime.now()

    with _lock:
        _sessions[session_id] = {
            "id":             session_id,
            "task":           task,
            "work_minutes":   work_minutes,
            "break_minutes":  break_minutes,
            "total_sessions": sessions,
            "current_round":  1,
            "phase":          "work",
            "phase_start":    start_time.isoformat(),
            "phase_end":      (start_time + timedelta(minutes=work_minutes)).isoformat(),
            "started_at":     start_time.isoformat(),
            "completed_at":   None,
            "rounds_done":    0,
            "cancelled":      False,
        }

    _schedule_next(session_id)

    end_time = (start_time + timedelta(minutes=work_minutes)).strftime("%H:%M")

    return json.dumps({
        "success":       True,
        "session_id":    session_id,
        "message":       f"🍅 Pomodoro started! Focus on: {task}",
        "phase":         "work",
        "work_minutes":  work_minutes,
        "break_minutes": break_minutes,
        "rounds":        sessions,
        "phase_ends_at": end_time,
        "tip":           "Use pomodoro_status to check time remaining.",
    })


def handle_pomodoro_status(session_id: str = None) -> str:
    session = _get_session(session_id)
    if not session:
        return json.dumps({
            "success": False,
            "error":   "No active Pomodoro session found. Start one with pomodoro_start.",
        })

    now       = datetime.now()
    phase_end = datetime.fromisoformat(session["phase_end"])
    remaining = max(0, (phase_end - now).total_seconds())
    mins      = int(remaining // 60)
    secs      = int(remaining % 60)
    phase     = session["phase"]

    if phase == "done":
        return json.dumps({
            "success":      True,
            "session_id":   session["id"],
            "task":         session["task"],
            "phase":        "done",
            "rounds_done":  session["rounds_done"],
            "total_rounds": session["total_sessions"],
            "message":      f"✅ All {session['total_sessions']} pomodoros complete! Great work on: {session['task']}",
        })

    if session.get("cancelled"):
        return json.dumps({
            "success":    True,
            "session_id": session["id"],
            "phase":      "cancelled",
            "message":    "Session was cancelled.",
        })

    phase_label = "🍅 Work" if phase == "work" else "☕ Break"
    emoji       = "💪" if phase == "work" else "😌"

    return json.dumps({
        "success":           True,
        "session_id":        session["id"],
        "task":              session["task"],
        "phase":             phase,
        "phase_label":       phase_label,
        "current_round":     session["current_round"],
        "total_rounds":      session["total_sessions"],
        "rounds_done":       session["rounds_done"],
        "time_remaining":    f"{mins}m {secs:02d}s",
        "minutes_remaining": mins,
        "seconds_remaining": secs,
        "message":           f"{emoji} {phase_label} — {mins}m {secs:02d}s remaining (Round {session['current_round']}/{session['total_sessions']})",
    })


def handle_pomodoro_stop(session_id: str = None) -> str:
    session = _get_session(session_id)
    if not session:
        return json.dumps({
            "success": False,
            "error":   "No active Pomodoro session found.",
        })

    sid = session["id"]

    with _lock:
        if sid in _timers:
            _timers[sid].cancel()
            del _timers[sid]
        _sessions[sid]["cancelled"]    = True
        _sessions[sid]["phase"]        = "cancelled"
        _sessions[sid]["completed_at"] = datetime.now().isoformat()

    return json.dumps({
        "success":     True,
        "session_id":  sid,
        "message":     f"⏹️ Pomodoro stopped. Completed {session['rounds_done']}/{session['total_sessions']} rounds for: {session['task']}",
        "rounds_done": session["rounds_done"],
    })


def handle_pomodoro_history() -> str:
    with _lock:
        all_sessions = list(_sessions.values())

    if not all_sessions:
        return json.dumps({
            "success":  True,
            "message":  "No Pomodoro sessions yet. Start one with pomodoro_start!",
            "sessions": [],
        })

    summary             = []
    total_focus_minutes = 0

    for s in all_sessions:
        rounds = s["rounds_done"]
        focus  = rounds * s["work_minutes"]
        total_focus_minutes += focus
        status = (
            "✅ completed" if s["phase"] == "done" else
            "⏹️ cancelled" if s.get("cancelled") else
            "🔄 active"
        )
        summary.append({
            "task":          s["task"],
            "status":        status,
            "rounds_done":   rounds,
            "total_rounds":  s["total_sessions"],
            "focus_minutes": focus,
            "started_at":    s["started_at"],
        })

    return json.dumps({
        "success":             True,
        "total_sessions":      len(all_sessions),
        "total_focus_minutes": total_focus_minutes,
        "total_focus_hours":   round(total_focus_minutes / 60, 1),
        "sessions":            summary,
        "message":             f"📊 {len(all_sessions)} session(s) — {total_focus_minutes} minutes of focused work.",
    })


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_session(session_id: Optional[str]) -> Optional[dict]:
    with _lock:
        if session_id:
            return _sessions.get(session_id)
        active = [
            s for s in _sessions.values()
            if s["phase"] not in ("done", "cancelled") and not s.get("cancelled")
        ]
        if active:
            return sorted(active, key=lambda s: s["started_at"])[-1]
        if _sessions:
            return sorted(_sessions.values(), key=lambda s: s["started_at"])[-1]
        return None


def _schedule_next(session_id: str):
    with _lock:
        session = _sessions.get(session_id)
        if not session:
            return
        phase_end = datetime.fromisoformat(session["phase_end"])
        delay = max(0, (phase_end - datetime.now()).total_seconds())

    t = threading.Timer(delay, _on_phase_end, args=[session_id])
    t.daemon = True

    with _lock:
        _timers[session_id] = t

    t.start()


def _on_phase_end(session_id: str):
    with _lock:
        session = _sessions.get(session_id)
        if not session or session.get("cancelled"):
            return

        phase  = session["phase"]
        round_ = session["current_round"]
        total  = session["total_sessions"]
        now    = datetime.now()

        if phase == "work":
            session["rounds_done"] += 1

            if round_ >= total:
                session["phase"]        = "done"
                session["completed_at"] = now.isoformat()
                _notify_async(
                    title="🍅 Pomodoro Complete!",
                    message=f"All {total} rounds done! Great work on: {session['task']}",
                )
                return

            break_mins = session["break_minutes"]
            session["phase"]       = "break"
            session["phase_start"] = now.isoformat()
            session["phase_end"]   = (now + timedelta(minutes=break_mins)).isoformat()
            _notify_async(
                title="☕ Break Time!",
                message=f"Round {round_}/{total} done! Take a {break_mins}-min break.",
            )

        elif phase == "break":
            work_mins = session["work_minutes"]
            session["phase"]         = "work"
            session["current_round"] += 1
            session["phase_start"]   = now.isoformat()
            session["phase_end"]     = (now + timedelta(minutes=work_mins)).isoformat()
            _notify_async(
                title="🍅 Back to Work!",
                message=f"Round {session['current_round']}/{total} — focus on: {session['task']}",
            )

    _schedule_next(session_id)


def _notify_async(title: str, message: str, urgency: str = "normal"):
    """Fire-and-forget desktop notification via notification_tool."""
    def _send():
        try:
            from tools.notification_tool import handle_notify
            handle_notify(title=title, message=message, urgency=urgency)
        except Exception:
            pass  # Never crash the timer thread

    threading.Thread(target=_send, daemon=True).start()
