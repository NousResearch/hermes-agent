"""Structured Desktop controls for persisted goal, loop, and heartbeat state.

``session.control.read`` is a stable, allowlisted view of one live session.
``session.control`` accepts a closed set of intent-level actions: goal and
loop actions use their existing TUI command handlers, while controls without a
TUI command use the public manager API. A successful action emits one matching
``session.control.update`` event.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time

from .method_ctx import HandlerRegistry, bind_module

_registry = HandlerRegistry()
method = _registry.method
_profile_scoped = _registry.profile_scoped

logger = logging.getLogger(__name__)


_ACTION_COMMAND_MAP: dict[str, tuple[str, str]] = {
    "goal.pause": ("goal", "pause"),
    "goal.resume": ("goal", "resume"),
    "goal.clear": ("goal", "clear"),
    "goal.unwait": ("goal", "unwait"),
    "loop.pause": ("loop", "pause"),
    "loop.resume": ("loop", "resume"),
    "loop.stop": ("loop", "stop"),
}

_MANAGER_ACTIONS = frozenset({
    "subgoal.add",
    "subgoal.remove",
    "subgoal.clear",
    "heartbeat.pause",
    "heartbeat.resume",
    "heartbeat.clear",
})

_VALID_ACTIONS = frozenset(_ACTION_COMMAND_MAP) | _MANAGER_ACTIONS

# Pause/resume are pure persisted-state writes: the Action Center may run them for a stored
# session with no live runtime (the slash-command path needs the session's CLI thread; the
# managers do not). Everything else — creates, updates, destructive actions — still requires
# the live session, where the command handlers can arbitrate against the running turn.
_DIRECT_STATE_ACTIONS = frozenset({
    "goal.pause",
    "goal.resume",
    "loop.pause",
    "loop.resume",
    "heartbeat.pause",
    "heartbeat.resume",
})


def _safe_goal_snapshot(state) -> dict | None:
    """Return only stable, frontend-safe GoalState fields."""
    if state is None or state.status == "cleared":
        return None
    snapshot = {
        "title": state.goal,
        "status": state.status,
        "turns_used": state.turns_used,
        "max_turns": state.max_turns,
        "contract": state.contract.to_dict(),
        "subgoals": list(state.subgoals),
        "gates": [
            {
                "command": gate.command,
                "timeout_seconds": gate.timeout_seconds,
                "max_retries": gate.max_retries,
                "attempts": gate.attempts,
                "last_exit_code": gate.last_exit_code,
            }
            for gate in state.gates
        ],
    }
    if state.created_at:
        snapshot["created_at"] = state.created_at
    if state.last_turn_at:
        snapshot["updated_at"] = state.last_turn_at
    if state.paused_reason:
        snapshot["paused_reason"] = state.paused_reason
    if state.last_verdict:
        snapshot["last_verdict"] = state.last_verdict
    if state.last_reason:
        snapshot["last_reason"] = state.last_reason
    if barrier := _extract_wait_barrier(state):
        snapshot["wait_barrier"] = barrier
    return snapshot


def _extract_wait_barrier(state) -> dict | None:
    """Expose absolute barriers; countdown presentation belongs to the client."""
    reason = state.waiting_reason or ""
    if state.waiting_until and time.time() < state.waiting_until:
        return {"type": "until", "until_at": state.waiting_until, "reason": reason}
    if state.waiting_on_session is not None:
        return {"type": "session", "target": state.waiting_on_session, "reason": reason}
    if state.waiting_on_pid is not None:
        return {"type": "pid", "target": state.waiting_on_pid, "reason": reason}
    return None


def _safe_loop_snapshot(state, *, deferred_by_goal: bool) -> dict | None:
    """Return allowlisted persisted LoopState fields, never its route."""
    if state is None or state.status == "cleared":
        return None
    snapshot = {
        "prompt": state.prompt,
        "status": state.status,
        "mode": state.mode,
        "interval_seconds": state.interval_seconds,
        "current_delay": state.current_delay,
        "times": state.times,
        "until": state.until,
        "max_ticks": state.max_ticks,
        "ticks_fired": state.ticks_fired,
        "created_at": state.created_at,
        "last_fired_at": state.last_fired_at,
        "next_due_at": state.next_due_at,
        "awaiting_response": state.awaiting_response,
        "deferred_by_goal": deferred_by_goal,
    }
    if state.paused_reason:
        snapshot["paused_reason"] = state.paused_reason
    if state.last_stop_reason:
        snapshot["last_stop_reason"] = state.last_stop_reason
    return snapshot


def _safe_heartbeat_snapshot(state) -> dict | None:
    """Return only the persisted HeartbeatState fields the Desktop renders."""
    if state is None or state.status == "cleared":
        return None
    return {
        "prompt": state.prompt,
        "status": state.status,
        "interval_seconds": state.interval_seconds,
        "created_at": state.created_at,
        "last_fired_at": state.last_fired_at,
        "fire_count": state.fire_count,
    }


def _snapshot_control(session_key: str) -> dict:
    """Serialize persisted session-control state once, without wall-clock churn."""
    goal_state = _load_goal_state(session_key)
    loop_state = _load_loop_state(session_key)
    heartbeat_state = _load_heartbeat_state(session_key)
    deferred_by_goal = bool(
        loop_state is not None
        and loop_state.status == "active"
        and goal_state is not None
        and goal_state.status == "active"
        and _goal_blocks_loop_tick(session_key)
    )
    goal = _safe_goal_snapshot(goal_state)
    loop = _safe_loop_snapshot(loop_state, deferred_by_goal=deferred_by_goal)
    heartbeat = _safe_heartbeat_snapshot(heartbeat_state)
    return {
        "goal": goal,
        "loop": loop,
        "heartbeat": heartbeat,
        "revision": _snapshot_revision(goal, loop, heartbeat),
        "updated_at": _snapshot_updated_at(goal_state, loop_state, heartbeat_state),
    }


def _snapshot_revision(goal, loop, heartbeat) -> str:
    """Hash canonical visible state so equal reads always have equal revisions."""
    if goal is None and loop is None and heartbeat is None:
        return ""
    canonical = json.dumps(
        {"goal": goal, "loop": loop, "heartbeat": heartbeat},
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _snapshot_updated_at(goal_state, loop_state, heartbeat_state):
    """Use only persisted timestamps; reads never manufacture a new timestamp."""
    candidates = []
    for state, fields in (
        (goal_state, ("created_at", "last_turn_at")),
        (loop_state, ("created_at", "last_fired_at")),
        (heartbeat_state, ("created_at", "last_fired_at")),
    ):
        if state is not None:
            candidates.extend(value for field in fields if (value := getattr(state, field, 0)))
    return max(candidates) if candidates else 0


def _load_goal_state(session_key):
    from hermes_cli.goals import load_goal

    return load_goal(session_key)


def _load_loop_state(session_key):
    from hermes_cli.loops import load_loop

    return load_loop(session_key)


def _load_heartbeat_state(session_key):
    from hermes_cli.heartbeat import load_heartbeat

    return load_heartbeat(session_key)


def _goal_blocks_loop_tick(session_key: str) -> bool:
    from hermes_cli.loops import goal_blocks_loop_tick

    return goal_blocks_loop_tick(session_key)


# Slash commands whose success changes the snapshot; ``command.dispatch`` (/goal, /loop built-ins) and the
# slash worker (/heartbeat, /subgoal) both publish after these so the Desktop card never waits for a turn.
_SESSION_CONTROL_SLASHES = frozenset({"goal", "heartbeat", "loop", "subgoal"})


def _publish_session_control_snapshot(sid: str, session: dict | None, *, only_if_present: bool = False) -> None:
    """Best-effort ``session.control.update`` for one live session. Also called after the post-turn hooks,
    because the goal judge and loop tick evaluation mutate persisted state AFTER ``message.complete`` — a
    client refresh keyed on that event reads the pre-judge turn count. ``only_if_present`` keeps the
    post-turn event stream of a session with no automation state byte-identical to today's."""
    if not session or not (session_key := str(session.get("session_key") or "")):
        return
    try:
        with _session_profile_runtime_scope(session):
            control = _snapshot_control(session_key)
        if only_if_present and not control["revision"]:
            return
        _emit("session.control.update", sid, {"control": control})
    except Exception:
        logger.debug("session.control.update publish failed for %s", sid, exc_info=True)


@method("session.control.read")
@_profile_scoped
def _(rid, params: dict) -> dict:
    """Return the current stable control snapshot for a live session."""
    session, err = _sess_nowait(params, rid)
    if err:
        return err
    session_key = str(session.get("session_key") or "")
    if not session_key:
        return _err(rid, 4001, "session has no stored key")
    try:
        return _ok(rid, {"control": _snapshot_control(session_key)})
    except Exception as exc:
        logger.debug("session.control.read failed: %s", exc, exc_info=True)
        return _err(rid, 5031, f"session.control.read failed: {exc}")


@method("session.control")
@_profile_scoped
def _(rid, params: dict) -> dict:
    """Run one allowlisted control action and emit its exact resulting snapshot."""
    raw_action = params.get("action")
    if not isinstance(raw_action, str) or not (action := raw_action.strip()):
        return _err(rid, 4004, "action is required")
    if action.startswith("goal.gate"):
        return _err(rid, 4004, "gate actions are not allowed through session.control")
    if action not in _VALID_ACTIONS:
        return _err(rid, 4004, f"unknown action: {action}")

    if "args" in params:
        args = params["args"]
        if not isinstance(args, dict):
            return _err(rid, 4004, "args must be an object")
    else:
        args = {}
    validated, validation_error = _validate_action_args(rid, action, args)
    if validation_error:
        return validation_error

    session, err = _sess_nowait(params, rid)
    if err:
        # No live runtime: the Action Center can still pause/resume stored automation directly
        # (persisted state; no CLI thread needed). Everything else keeps its live-session error.
        direct_key = str(params.get("session_key") or "").strip()
        if direct_key and action in _DIRECT_STATE_ACTIONS:
            return _direct_state_control(rid, params, direct_key, action)
        return err
    session_key = str(session.get("session_key") or "")
    if not session_key:
        return _err(rid, 4001, "session has no stored key")

    try:
        if action in _ACTION_COMMAND_MAP:
            name, arg = _ACTION_COMMAND_MAP[action]
            action_result = _dispatch_command(rid, session_id=params.get("session_id") or "", name=name, arg=arg)
        else:
            action_result = _execute_manager_action(session_key, action, validated)
    except (RuntimeError, ValueError, IndexError) as exc:
        return _err(rid, 4004, _manager_error_message(action, exc))

    if "error" in action_result:
        return action_result

    try:
        control = _snapshot_control(session_key)
    except Exception as exc:
        logger.debug("session.control snapshot after %s failed: %s", action, exc, exc_info=True)
        return _err(rid, 5031, f"session.control snapshot failed: {exc}")

    # command.dispatch already published the update for goal/loop actions; manager actions publish here.
    if action not in _ACTION_COMMAND_MAP:
        try:
            _emit("session.control.update", params.get("session_id") or "", {"control": control})
        except Exception as exc:
            logger.debug("session.control.update emit failed (best-effort): %s", exc, exc_info=True)
    return _ok(rid, {"control": control, "dispatch": _dispatch_envelope(action_result)})


def _validate_action_args(rid, action: str, args: dict):
    """Validate the only actions with input before any manager is constructed."""
    if action == "subgoal.add":
        text = args.get("text")
        if not isinstance(text, str) or not (text := text.strip()):
            return None, _err(rid, 4004, "subgoal text is required")
        return {"text": text}, None
    if action == "subgoal.remove":
        index = args.get("index")
        if type(index) is not int:
            return None, _err(rid, 4004, "subgoal index must be an integer")
        if index < 1:
            return None, _err(rid, 4004, "subgoal index must be >= 1")
        return {"index": index}, None
    return {}, None


def _dispatch_command(rid, *, session_id: str, name: str, arg: str) -> dict:
    """Delegate a fixed intent to the existing TUI command dispatcher."""
    handler = _methods.get("command.dispatch")
    if handler is None:
        return _err(rid, 5031, "command.dispatch unavailable")
    try:
        return handler(rid, {"session_id": session_id, "name": name, "arg": arg})
    except Exception as exc:
        logger.debug("command.dispatch %s %s failed: %s", name, arg, exc, exc_info=True)
        return _err(rid, 5031, f"dispatch failed: {exc}")


def _execute_manager_action(session_key: str, action: str, args: dict) -> dict:
    """Use manager APIs for controls that have no TUI command handler."""
    if action.startswith("subgoal."):
        return _execute_subgoal_action(session_key, action, args)
    return _execute_heartbeat_action(session_key, action)


def _execute_subgoal_action(session_key: str, action: str, args: dict) -> dict:
    from hermes_cli.goals import GoalManager

    manager = GoalManager(session_id=session_key)
    if action == "subgoal.add":
        text = manager.add_subgoal(args["text"])
        return {"result": {"type": "exec", "output": f"✓ Added subgoal {len(manager.state.subgoals)}: {text}"}}
    if action == "subgoal.remove":
        index = args["index"]
        text = manager.remove_subgoal(index)
        return {"result": {"type": "exec", "output": f"✓ Removed subgoal {index}: {text}"}}
    count = manager.clear_subgoals()
    output = f"✓ Cleared {count} subgoal{'s' if count != 1 else ''}." if count else "No subgoals to clear."
    return {"result": {"type": "exec", "output": output}}


def _execute_heartbeat_action(session_key: str, action: str) -> dict:
    from hermes_cli.heartbeat import HeartbeatManager, format_interval

    manager = HeartbeatManager(session_id=session_key)
    if action == "heartbeat.pause":
        state = manager.pause()
        output = f"⏸ Heartbeat paused: {state.prompt}" if state else "No heartbeat set."
    elif action == "heartbeat.resume":
        state = manager.resume()
        output = (
            f"▶ Heartbeat resumed (every {format_interval(state.interval_seconds)}): {state.prompt}"
            if state else "No heartbeat to resume."
        )
    elif action == "heartbeat.clear":
        output = "✓ Heartbeat cleared." if manager.clear() else "No heartbeat set."
    else:
        return _err(None, 4004, f"unknown heartbeat action: {action}")
    return {"result": {"type": "exec", "output": output}}


def _execute_direct_state_action(session_key: str, action: str) -> dict:
    """Run a persisted pause/resume through the public manager APIs (no live session).

    Mirrors the slash-command semantics so a control applied from the Action Center lands in
    the same state the typed command leaves (pause reasons, turn budget, cadence untouched).
    """
    kind, verb = action.split(".", 1)

    if kind == "goal":
        from hermes_cli.goals import GoalManager, load_goal

        state = load_goal(session_key)
        if state is None or state.status == "cleared":
            return _err(None, 4004, "No goal set for this session.")
        if verb == "pause" and state.status not in ("active", "paused"):
            return _err(None, 4004, f"Goal is {state.status}; only an active goal can be paused.")
        if verb == "resume" and state.status != "paused":
            return _err(None, 4004, f"Goal is {state.status}, not paused.")
        manager = GoalManager(session_id=session_key)
        updated = manager.pause() if verb == "pause" else manager.resume()
        if updated is None:
            return _err(None, 4004, "Goal update failed.")
        output = (
            f"⏸ Goal paused: {updated.goal}"
            if verb == "pause"
            else f"▶ Goal resumed: {updated.goal}"
        )
    elif kind == "loop":
        from hermes_cli.loops import LoopManager, load_loop

        state = load_loop(session_key)
        if state is None or state.status == "cleared":
            return _err(None, 4004, "No loop set for this session.")
        if verb == "pause" and state.status not in ("active", "paused"):
            return _err(None, 4004, f"Loop is {state.status}; only a running loop can be paused.")
        if verb == "resume" and state.status != "paused":
            return _err(None, 4004, f"Loop is {state.status}, not paused.")
        manager = LoopManager(session_id=session_key)
        updated = manager.pause() if verb == "pause" else manager.resume()
        if updated is None:
            return _err(None, 4004, f"Loop cannot be {'paused' if verb == 'pause' else 'resumed'}.")
        output = (
            f"⏸ Loop paused: {updated.prompt}"
            if verb == "pause"
            else f"▶ Loop resumed ({updated.cadence_label()}): {updated.prompt}"
        )
    elif kind == "heartbeat":
        return _execute_heartbeat_action(session_key, action)
    else:
        return _err(None, 4004, f"action requires a live session: {action}")
    return {"result": {"type": "exec", "output": output}}


def _direct_state_control(rid, params: dict, session_key: str, action: str) -> dict:
    """Pause/resume persisted automation for a stored session with no live runtime.

    Gate: the session row must exist in the addressed profile and be operator-facing (the same
    deny-list the inbox list applies), so the direct path cannot reach sessions the Action
    Center would never show. Success returns the same payload shape as the live path.
    """
    from .methods_inbox import _inbox_denied_source

    with _profile_db(params) as db:
        row = db.get_session(session_key) if db is not None else None
    if row is None or _inbox_denied_source(row):
        return _err(rid, 4001, "session not found")

    try:
        action_result = _execute_direct_state_action(session_key, action)
    except (RuntimeError, ValueError, IndexError) as exc:
        return _err(rid, 4004, _manager_error_message(action, exc))
    if "error" in action_result:
        return {**action_result, "id": rid}

    try:
        control = _snapshot_control(session_key)
    except Exception as exc:
        logger.debug("session.control snapshot after direct %s failed: %s", action, exc, exc_info=True)
        return _err(rid, 5031, f"session.control snapshot failed: {exc}")

    # A runtime for this key may exist after all (the client's session_id was stale): refresh it
    # so any open chat surface follows the change the same way the live path keeps it in sync.
    # Call the registered helper by name: a module-level import would bind the unrebound copy,
    # whose ``_inbox_home_key`` chain has no ``_hermes_home`` outside the server namespace.
    try:
        live_sid = _inbox_live_session_for_key(_profile_home(params.get("profile")), session_key)
        if live_sid:
            event_control = {key: value for key, value in control.items() if key != "loop_min_interval_seconds"}
            _emit("session.control.update", live_sid, {"control": event_control})
    except Exception as exc:
        logger.debug("session.control.update emit after direct %s failed (best-effort): %s", action, exc, exc_info=True)

    return _ok(rid, {"control": control, "dispatch": _dispatch_envelope(action_result)})
def _manager_error_message(action: str, exc: Exception) -> str:
    prefixes = {
        "subgoal.add": "/subgoal",
        "subgoal.remove": "/subgoal remove",
        "subgoal.clear": "/subgoal clear",
    }
    return f"{prefixes.get(action, action)}: {exc}"


def _dispatch_envelope(response: dict) -> dict:
    """Keep the command result's user-visible envelope without adding model-facing data."""
    result = response.get("result") or {}
    return {
        "type": result.get("type"),
        "output": result.get("output"),
        "notice": result.get("notice"),
        "message": result.get("message"),
        "display": result.get("display"),
    }


def register(server) -> None:
    """Rebind this module's handlers onto the server namespace."""
    bind_module(globals(), server, skip=("_",))
