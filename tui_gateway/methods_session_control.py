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
import threading

from .method_ctx import HandlerRegistry, bind_module

_registry = HandlerRegistry()
method = _registry.method
_profile_scoped = _registry.profile_scoped

logger = logging.getLogger(__name__)

# Serialize GUI check-and-create so competing requests never replace state.
_CREATION_LOCK = threading.RLock()


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
    "goal.create",
    "subgoal.add",
    "subgoal.remove",
    "subgoal.clear",
    "loop.create",
    "heartbeat.create",
    "heartbeat.pause",
    "heartbeat.resume",
    "heartbeat.clear",
})

_VALID_ACTIONS = frozenset(_ACTION_COMMAND_MAP) | _MANAGER_ACTIONS


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
        with _session_profile_runtime_scope(session):
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
        return err
    session_key = str(session.get("session_key") or "")
    if not session_key:
        return _err(rid, 4001, "session has no stored key")

    try:
        if action in _ACTION_COMMAND_MAP:
            name, arg = _ACTION_COMMAND_MAP[action]
            action_result = _dispatch_command(rid, session_id=params.get("session_id") or "", name=name, arg=arg)
        else:
            if action.endswith(".create"):
                with _CREATION_LOCK, _session_profile_runtime_scope(session):
                    action_result = _execute_manager_action(session_key, action, validated)
            else:
                with _session_profile_runtime_scope(session):
                    action_result = _execute_manager_action(session_key, action, validated)
    except (RuntimeError, ValueError, IndexError) as exc:
        return _err(rid, 4004, _manager_error_message(action, exc))

    if "error" in action_result:
        return {**action_result, "id": rid}

    try:
        with _session_profile_runtime_scope(session):
            control = _snapshot_control(session_key)
    except Exception as exc:
        logger.debug("session.control snapshot after %s failed: %s", action, exc, exc_info=True)
        return _err(rid, 5031, f"session.control snapshot failed: {exc}")

    if action.endswith(".create") and control.get(action.split(".", 1)[0]) is None:
        return _err(rid, 5031, "Automation creation could not be verified. Refresh the controls before trying again.")

    # command.dispatch already published the update for goal/loop actions; manager actions publish here.
    if action not in _ACTION_COMMAND_MAP:
        try:
            _emit("session.control.update", params.get("session_id") or "", {"control": control})
        except Exception as exc:
            logger.debug("session.control.update emit failed (best-effort): %s", exc, exc_info=True)
    return _ok(rid, {"control": control, "dispatch": _dispatch_envelope(action_result)})


def _validate_action_args(rid, action: str, args: dict):
    """Validate the only actions with input before any manager is constructed."""
    if action == "goal.create":
        return _validate_goal_create_args(rid, args)
    if action == "loop.create":
        return _validate_loop_create_args(rid, args)
    if action == "heartbeat.create":
        return _validate_heartbeat_create_args(rid, args)
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


_MAX_TURNS_CEILING = 1000
_INTERVAL_CEILING = 86_400 * 7
_RUN_LIMIT_CEILING = 10_000


def _validate_goal_create_args(rid, args):
    prompt = args.get("prompt")
    if not isinstance(prompt, str) or not (prompt := prompt.strip()):
        return None, _err(rid, 4004, "goal prompt is required")
    criteria = args.get("criteria")
    if criteria is not None:
        if not isinstance(criteria, list) or not all(isinstance(c, str) for c in criteria):
            return None, _err(rid, 4004, "criteria must be a list of strings")
        stripped = []
        for c in criteria:
            s = c.strip() if isinstance(c, str) else ""
            if not s:
                return None, _err(rid, 4004, "criteria items must be non-empty strings")
            stripped.append(s)
        criteria = stripped
    max_turns = args.get("max_turns")
    if max_turns is not None:
        if type(max_turns) is not int or max_turns < 1:
            return None, _err(rid, 4004, "max_turns must be a positive integer")
        if max_turns > _MAX_TURNS_CEILING:
            return None, _err(rid, 4004, f"max_turns must be <= {_MAX_TURNS_CEILING}")
    return {"prompt": prompt, "criteria": criteria or [], "max_turns": max_turns}, None


def _validate_loop_create_args(rid, args):
    prompt = args.get("prompt")
    if not isinstance(prompt, str) or not (prompt := prompt.strip()):
        return None, _err(rid, 4004, "loop prompt is required")
    interval = args.get("interval_seconds")
    if type(interval) is not int or interval < 1:
        return None, _err(rid, 4004, "interval_seconds must be a positive integer")
    if interval > _INTERVAL_CEILING:
        return None, _err(rid, 4004, f"interval_seconds must be <= {_INTERVAL_CEILING}")
    run_limit = args.get("run_limit")
    if run_limit is not None:
        if type(run_limit) is not int or run_limit < 1:
            return None, _err(rid, 4004, "run_limit must be a positive integer")
        if run_limit > _RUN_LIMIT_CEILING:
            return None, _err(rid, 4004, f"run_limit must be <= {_RUN_LIMIT_CEILING}")
    stop_condition = args.get("stop_condition")
    if stop_condition is not None and (not isinstance(stop_condition, str) or not stop_condition.strip()):
        return None, _err(rid, 4004, "stop_condition must be a non-empty string")
    return {
        "prompt": prompt,
        "interval_seconds": interval,
        "run_limit": run_limit,
        "stop_condition": (stop_condition or "").strip(),
    }, None


def _validate_heartbeat_create_args(rid, args):
    prompt = args.get("prompt")
    if not isinstance(prompt, str) or not (prompt := prompt.strip()):
        return None, _err(rid, 4004, "heartbeat prompt is required")
    interval = args.get("interval_seconds")
    if type(interval) is not int or interval < 1:
        return None, _err(rid, 4004, "interval_seconds must be a positive integer")
    if interval > _INTERVAL_CEILING:
        return None, _err(rid, 4004, f"interval_seconds must be <= {_INTERVAL_CEILING}")
    return {"prompt": prompt, "interval_seconds": interval}, None


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
    if action == "goal.create":
        return _execute_goal_create(session_key, args)
    if action.startswith("subgoal."):
        return _execute_subgoal_action(session_key, action, args)
    if action == "loop.create":
        return _execute_loop_create(session_key, args)
    if action == "heartbeat.create":
        return _execute_heartbeat_create(session_key, args)
    return _execute_heartbeat_action(session_key, action)


def _execute_goal_create(session_key: str, args: dict) -> dict:
    from hermes_cli.goals import GoalManager, load_goal

    existing = load_goal(session_key)
    if existing is not None and existing.status != "cleared":
        return _err(None, 4004, f"A {existing.status} goal already exists for this session. Clear it first with goal.clear.")
    try:
        default_turns = int((_load_cfg().get("goals") or {}).get("max_turns", 20) or 20)
    except (TypeError, ValueError, OverflowError):
        default_turns = 20
    manager = GoalManager(session_id=session_key, default_max_turns=default_turns)
    prompt = args["prompt"]
    criteria = args.get("criteria", [])
    max_turns = args.get("max_turns")
    state = manager.set(prompt, max_turns=max_turns)
    if criteria:
        for criterion in criteria:
            manager.add_subgoal(criterion)
        state = manager.state
    message = state and manager.next_continuation_prompt() or prompt
    notice = f"✓ Goal set: {prompt}"
    return {"result": {"type": "send", "output": notice, "message": message, "notice": notice, "display": prompt}}


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


def _execute_loop_create(session_key: str, args: dict) -> dict:
    from hermes_cli.loops import LoopManager, load_loop

    existing = load_loop(session_key)
    if existing is not None and existing.status != "cleared":
        return _err(None, 4004, f"A {existing.status} loop already exists for this session. Clear it first with loop.stop.")
    manager = LoopManager(session_id=session_key)
    prompt = args["prompt"]
    interval = args["interval_seconds"]
    times = args.get("run_limit") or 0
    until = args.get("stop_condition", "")
    state = manager.set(prompt, interval_seconds=interval, times=times, until=until)
    lines = [f"✓ Loop set ({state.cadence_label()}): {state.prompt}"]
    if state.times:
        lines.append(f"  Runs {state.times} time{'s' if state.times != 1 else ''}, then stops.")
    if state.until:
        lines.append(f"  Stops when: {state.until}")
    notice = "\n".join(lines)
    # LoopManager schedules the first accounted tick immediately.
    return {"result": {"type": "exec", "output": notice}}


def _execute_heartbeat_create(session_key: str, args: dict) -> dict:
    from hermes_cli.heartbeat import HeartbeatManager, format_interval, load_heartbeat

    existing = load_heartbeat(session_key)
    if existing is not None and existing.status != "cleared":
        return _err(None, 4004, f"A {existing.status} heartbeat already exists for this session. Clear it first with heartbeat.clear.")
    manager = HeartbeatManager(session_id=session_key)
    prompt = args["prompt"]
    interval = args["interval_seconds"]
    state = manager.set(prompt, interval_seconds=interval)
    notice = f"✓ Heartbeat set (every {format_interval(state.interval_seconds)}): {state.prompt}"
    # The owner poller fires after the first interval, never on creation.
    return {"result": {"type": "exec", "output": notice}}


def _manager_error_message(action: str, exc: Exception) -> str:
    prefixes = {
        "goal.create": "/goal",
        "subgoal.add": "/subgoal",
        "subgoal.remove": "/subgoal remove",
        "subgoal.clear": "/subgoal clear",
        "loop.create": "/loop",
        "heartbeat.create": "/heartbeat",
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
