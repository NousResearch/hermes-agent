"""Project a hosted room member's live runtime events to the ``on_room_member_activity`` plugin hook.

A member turn runs on a hidden ``room_plumbing`` session with no client transport, so the tool /
approval / streaming frames the turn loop already emits bottom out at stdio and are lost. Between
``turn.started`` and ``turn.settled`` in the durable room log a client sees nothing. This module
re-routes those frames, stamped with the room coordinates the session carries in
``_hosted_room_task``, to plugins — off the token path, through the same bounded per-consumer
queues the ``on_stream_*`` observers use. Nothing is written to the room log: deltas at room-log
byte budgets would exhaust a room in minutes, and checkpoint replay must stay a pure function of
the durable events.
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from collections.abc import Mapping
from typing import Any

HOOK_NAME = "on_room_member_activity"

# Per running task: the tool it is in right now and when it last did anything, fed by this process's own
# room-turn frames (``project_room_member_activity``) or by the member transport's ``info`` answer that
# the room driver polls during the turn (``record_task_progress``); ``groups.state`` merges it into
# ``member_activity`` so a room can tell a bot busy in a tool from one gone quiet.
_PROGRESS_LIMIT = 512
_progress: OrderedDict[tuple[str, str], dict[str, Any]] = OrderedDict()
_progress_lock = threading.Lock()


def _note_progress(key: tuple[str, str], kind: str, payload: Mapping[str, Any] | None) -> None:
    with _progress_lock:
        entry = _progress.pop(key, None) or {"tool": None}
        if kind == "tool.started":
            entry["tool"] = str((payload or {}).get("name") or "") or None
        elif kind == "tool.completed":
            entry["tool"] = None
        entry["progress_at"] = time.time()
        _progress[key] = entry
        while len(_progress) > _PROGRESS_LIMIT:
            _progress.popitem(last=False)


def record_task_progress(room_id: str, task_id: str, info: Mapping[str, Any]) -> None:
    """Adopt the ``tool``/``progress_at`` a member transport's ``info`` reported for this running task.

    A gateway-owned room's member turn publishes its frames on the member authority's own event stream,
    never through ``project_room_member_activity``; the room driver already polls the member's ``info``
    during the turn, and this is how that answer reaches ``groups.state``. The member's clock stamps
    ``progress_at``: the time of its last frame, not of this poll."""
    progress_at, tool = info.get("progress_at"), info.get("tool")
    if str(info.get("task_id") or "") != task_id or type(progress_at) not in (int, float):
        return
    with _progress_lock:
        _progress.pop((room_id, task_id), None)
        _progress[(room_id, task_id)] = {
            "tool": tool[:128] if isinstance(tool, str) and tool else None, "progress_at": float(progress_at)}
        while len(_progress) > _PROGRESS_LIMIT:
            _progress.popitem(last=False)


def forget_task_progress(room_id: str, task_id: str) -> None:
    with _progress_lock:
        _progress.pop((room_id, task_id), None)


def task_progress(room_id: str, task_id: str) -> dict[str, Any]:
    """``{"tool", "progress_at"}`` for a task this process has seen frames from, else ``{}``."""
    with _progress_lock:
        entry = _progress.get((room_id, task_id))
        return {key: value for key, value in (entry or {}).items() if value is not None}

# Session event frame ``type`` -> room activity ``kind``. Frames not listed (session.info,
# status.update, message.start/complete, ...) are session chrome, not member activity.
KIND_BY_FRAME_TYPE: Mapping[str, str] = {
    "tool.start": "tool.started",
    "tool.complete": "tool.completed",
    "tool.output_risk": "tool.output_risk",
    "message.delta": "message.delta",
    "message.interim": "message.interim",
    "reasoning.delta": "reasoning.delta",
    "error": "turn.error",
}

_COORDINATE_FIELDS = ("room_id", "thread_id", "member_id", "turn_id", "task_id", "execution_generation")


def emit_room_member_activity(hosted_task: Mapping[str, Any], *, kind: str, payload: Mapping[str, Any] | None,
                              seq: int | None = None) -> bool:
    """Queue one activity event for every registered consumer; False when nobody listens."""
    from agent.plugin_stream_hooks import enqueue_plugin_stream_hook

    coordinates = {field: hosted_task.get(field) for field in _COORDINATE_FIELDS}
    return enqueue_plugin_stream_hook(HOOK_NAME, **coordinates, kind=kind, seq=seq, payload=dict(payload or {}))


# Server→client request frames (``{"id": "srq-…", "method": …}``) that are member activity.
KIND_BY_REQUEST_METHOD: Mapping[str, str] = {"approval": "request.opened"}


def project_room_member_activity(frame: Mapping[str, Any], sessions: Mapping[str, Mapping[str, Any]]) -> bool:
    """Fire the hook for an outgoing session frame (event notification or server→client request) when its
    session is running a room turn."""
    params = frame.get("params")
    if not isinstance(params, Mapping):
        return False
    if frame.get("method") == "event":
        kind = KIND_BY_FRAME_TYPE.get(str(params.get("type") or ""))
        payload = params.get("payload")
    elif "id" in frame:
        kind = KIND_BY_REQUEST_METHOD.get(str(frame.get("method") or ""))
        payload = {k: v for k, v in params.items() if k != "session_id"}
    else:
        return False
    if kind is None:
        return False
    session = sessions.get(str(params.get("session_id") or ""))
    hosted_task = session.get("_hosted_room_task") if isinstance(session, Mapping) else None
    if not isinstance(hosted_task, Mapping):
        return False
    payload = payload if isinstance(payload, Mapping) else None
    if hosted_task.get("task_id"):
        _note_progress((str(hosted_task.get("room_id")), str(hosted_task["task_id"])), kind, payload)
    return emit_room_member_activity(hosted_task, kind=kind, payload=payload, seq=params.get("seq"))
