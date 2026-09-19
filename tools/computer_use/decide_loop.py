"""Bounded decide → act → verify loop for computer_use (#113850 / RFC #112639).

Runs repeated ``decide`` calls against the System-One lane, executes the
suggested action, and re-captures until the goal is done, the lane abstains
repeatedly, or ``max_steps`` is reached. Approval semantics are unchanged —
each input action still passes through the normal computer_use gate.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Optional

from tools.computer_use.trajectory_cache import (
    CachedStep,
    cached_step_to_decision,
    invalidate_trajectory,
    load_trajectory,
    save_trajectory,
)

HandleComputerUse = Callable[[dict[str, Any]], Any]
VerifyFn = Callable[[str, dict[str, Any], dict[str, Any] | None], dict[str, Any] | None]


@dataclass
class LoopStep:
    step: int
    decide_backend: str | None = None
    decide_action: str | None = None
    target_element: int | None = None
    confidence: float | None = None
    fail_open: bool = False
    from_cache: bool = False
    executed: bool = False
    execute_ok: bool = False
    element_count_before: int = 0
    element_count_after: int = 0
    semantic_delta_ratio: float | None = None
    verify_status: str | None = None
    prepared_commit: bool = False
    elapsed_s: float = 0.0
    error: str | None = None


@dataclass
class LoopResult:
    ok: bool
    status: str  # done | completed | stuck | fail_open | max_steps | error
    steps: list[LoopStep] = field(default_factory=list)
    elapsed_s: float = 0.0
    goal: str = ""
    max_steps: int = 0
    cache_hit_steps: int = 0
    trajectory_cache_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _parse_json(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        return json.loads(raw)
    return {"error": f"unexpected computer_use result type: {type(raw)!r}"}


def _capture_element_count(handle: HandleComputerUse, *, app: str | None) -> int:
    cap_args: dict[str, Any] = {"action": "capture", "mode": "ax"}
    if app:
        cap_args["app"] = app
    cap = _parse_json(handle(cap_args))
    return int(cap.get("total_elements") or 0)


def _wait_for_elements(
    handle: HandleComputerUse,
    *,
    app: str | None,
    timeout_s: float = 2.5,
    poll_s: float = 0.2,
) -> int:
    """Poll AX capture until elements appear or timeout (dialog transitions)."""
    deadline = time.monotonic() + max(timeout_s, 0.0)
    count = _capture_element_count(handle, app=app)
    while count <= 0 and time.monotonic() < deadline:
        time.sleep(max(poll_s, 0.05))
        count = _capture_element_count(handle, app=app)
    return count


def _dispatch_action(handle: HandleComputerUse, action: str, args: dict[str, Any]) -> dict[str, Any]:
    cu_args = dict(args)
    cu_args["action"] = action
    return _parse_json(handle(cu_args))


def _execute_decision(
    handle: HandleComputerUse,
    decision: dict[str, Any],
    *,
    app: str | None,
    graph: Any | None = None,
    node_id: str | None = None,
    use_prepared: bool = False,
) -> tuple[bool, dict[str, Any] | None, bool]:
    """Execute a lane decision. Returns (ok, error_payload, used_prepared)."""
    action = (decision.get("action") or "").strip().lower()
    if action == "done":
        return True, None, False
    if action == "escalate":
        return False, {"error": "decision lane escalated to main planner"}, False
    if action == "wait":
        seconds = float(decision.get("seconds") or 0.5)
        time.sleep(min(max(seconds, 0.1), 5.0))
        return True, None, False
    if action in {"click", "double_click", "right_click", "middle_click"}:
        target = decision.get("target_element")
        if target is None:
            return False, {"error": f"{action} missing target_element"}, False
        args: dict[str, Any] = {"element": int(target)}
        if app:
            args["app"] = app
        if use_prepared and graph is not None and node_id:
            from tools.computer_use.prepared_action import (
                GROUNDING,
                FactRef,
                prepare,
                verifier_result_ok,
            )

            def dispatch(act: str, payload: dict[str, Any]) -> Any:
                return handle({**payload, "action": act})

            prepared = prepare(
                node_id=node_id,
                graph=graph,
                action=action,
                args=args,
                dependencies=[FactRef(GROUNDING, f"elem:{target}")],
                verifier=verifier_result_ok(),
            )
            outcome = prepared.commit(graph, dispatch)
            if not outcome.committed:
                return False, {"error": outcome.reason or "prepared commit refused"}, True
            result = _as_mapping(outcome.result)
            if not outcome.ok:
                return False, result or {"error": outcome.reason or "verifier failed"}, True
            return True, result, True
        result = _dispatch_action(handle, action, args)
        if result.get("error"):
            return False, result, False
        return True, result, False
    if action == "key":
        keys = decision.get("keys") or "Return"
        args = {"keys": keys}
        if app:
            args["app"] = app
        result = _dispatch_action(handle, action, args)
        if result.get("error"):
            return False, result, False
        return True, result, False
    return False, {"error": f"unsupported loop action: {action!r}"}, False


def _as_mapping(result: Any) -> dict[str, Any]:
    if isinstance(result, dict):
        return result
    if isinstance(result, str):
        try:
            parsed = json.loads(result)
        except ValueError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _invalidate_prepared_on_semantic_change(graph: Any | None, metrics: dict[str, Any] | None) -> None:
    if graph is None or not metrics:
        return
    ratio = metrics.get("changed_element_ratio")
    if ratio is not None and float(ratio) > 0:
        from tools.computer_use.prepared_action import UI_FACT

        graph.invalidate(UI_FACT, "semantic_state")


def _apply_verify(
    loop_step: LoopStep,
    verify_fn: VerifyFn | None,
    decision: dict[str, Any],
    exec_result: dict[str, Any] | None,
) -> None:
    if not verify_fn:
        return
    action = (decision.get("action") or "").strip().lower()
    payload = verify_fn(action, decision, exec_result)
    if payload and payload.get("verify_status"):
        loop_step.verify_status = str(payload["verify_status"])



def _attach_semantic_metrics(loop_step: LoopStep, metrics: dict[str, Any] | None) -> None:
    if not metrics:
        return
    ratio = metrics.get("changed_element_ratio")
    if ratio is not None:
        loop_step.semantic_delta_ratio = float(ratio)


def _record_cache_step(steps_for_cache: list[CachedStep], decision: dict[str, Any]) -> None:
    action = (decision.get("action") or "").strip().lower()
    if not action or action in {"escalate", "wait"}:
        return
    target = decision.get("target_element")
    steps_for_cache.append(
        CachedStep(
            action=action,
            target_element=int(target) if target is not None else None,
            keys=decision.get("keys"),
            backend=decision.get("backend"),
        )
    )


def run_decide_loop(
    goal: str,
    handle: HandleComputerUse,
    *,
    app: str | None = None,
    max_steps: int = 8,
    stuck_threshold: int = 3,
    step_pause_s: float = 0.35,
    external_done: Callable[[], bool] | None = None,
    use_trajectory_cache: bool = False,
    metrics_fn: Callable[[], dict[str, Any]] | None = None,
    verify_fn: VerifyFn | None = None,
    use_prepared_actions: bool = False,
    dependency_graph: Any | None = None,
) -> LoopResult:
    """Run decide → execute → capture until done, stuck, or max_steps."""
    goal = (goal or "").strip()
    if not goal:
        return LoopResult(ok=False, status="error", goal=goal, max_steps=max_steps)

    started = time.perf_counter()
    steps: list[LoopStep] = []
    fail_open_streak = 0
    no_change_streak = 0
    last_signature: tuple[Any, ...] | None = None
    cached_steps: list[CachedStep] | None = load_trajectory(goal, app) if use_trajectory_cache else None
    cache_idx = 0
    cache_hit_steps = 0
    steps_for_cache: list[CachedStep] = []
    trajectory_cache_path: str | None = None
    graph = dependency_graph
    if use_prepared_actions and graph is None:
        from tools.computer_use.prepared_action import DependencyGraph

        graph = DependencyGraph()

    for step_idx in range(1, max_steps + 1):
        step_started = time.perf_counter()
        loop_step = LoopStep(step=step_idx)
        steps.append(loop_step)

        if external_done and external_done():
            loop_step.elapsed_s = round(time.perf_counter() - step_started, 3)
            if use_trajectory_cache and steps_for_cache:
                trajectory_cache_path = save_trajectory(goal, app, steps_for_cache)
            return LoopResult(
                ok=True,
                status="completed",
                steps=steps,
                elapsed_s=round(time.perf_counter() - started, 3),
                goal=goal,
                max_steps=max_steps,
                cache_hit_steps=cache_hit_steps,
                trajectory_cache_path=trajectory_cache_path,
            )

        loop_step.element_count_before = _capture_element_count(handle, app=app)
        if loop_step.element_count_before == 0 and not (external_done and external_done()):
            loop_step.element_count_before = _wait_for_elements(handle, app=app)

        decision: dict[str, Any]
        if cached_steps is not None and cache_idx < len(cached_steps):
            decision = cached_step_to_decision(cached_steps[cache_idx])
            cache_idx += 1
            cache_hit_steps += 1
            loop_step.from_cache = True
            loop_step.fail_open = False
            decide_payload = {"ok": True, "fail_open": False, "decision": decision}
        else:
            decide_args: dict[str, Any] = {"action": "decide", "goal": goal}
            if app:
                decide_args["app"] = app
            decide_payload = _parse_json(handle(decide_args))
            if not decide_payload.get("ok"):
                loop_step.error = str(decide_payload.get("error") or "decide failed")
                loop_step.elapsed_s = round(time.perf_counter() - step_started, 3)
                return LoopResult(
                    ok=False,
                    status="error",
                    steps=steps,
                    elapsed_s=round(time.perf_counter() - started, 3),
                    goal=goal,
                    max_steps=max_steps,
                    cache_hit_steps=cache_hit_steps,
                )
            loop_step.fail_open = bool(decide_payload.get("fail_open"))
            decision = decide_payload.get("decision") or {}

        if loop_step.fail_open:
            fail_open_streak += 1
            loop_step.elapsed_s = round(time.perf_counter() - step_started, 3)
            if fail_open_streak >= stuck_threshold:
                if use_trajectory_cache:
                    invalidate_trajectory(goal, app)
                return LoopResult(
                    ok=False,
                    status="fail_open",
                    steps=steps,
                    elapsed_s=round(time.perf_counter() - started, 3),
                    goal=goal,
                    max_steps=max_steps,
                    cache_hit_steps=cache_hit_steps,
                )
            time.sleep(step_pause_s)
            continue

        fail_open_streak = 0
        loop_step.decide_backend = decision.get("backend")
        loop_step.decide_action = decision.get("action")
        loop_step.confidence = decision.get("confidence")
        if decision.get("target_element") is not None:
            loop_step.target_element = int(decision["target_element"])

        if decision.get("done"):
            loop_step.elapsed_s = round(time.perf_counter() - step_started, 3)
            if use_trajectory_cache and steps_for_cache:
                trajectory_cache_path = save_trajectory(goal, app, steps_for_cache)
            return LoopResult(
                ok=True,
                status="done",
                steps=steps,
                elapsed_s=round(time.perf_counter() - started, 3),
                goal=goal,
                max_steps=max_steps,
                cache_hit_steps=cache_hit_steps,
                trajectory_cache_path=trajectory_cache_path,
            )

        if loop_step.decide_action == "escalate" and loop_step.element_count_before == 0:
            if external_done and external_done():
                loop_step.elapsed_s = round(time.perf_counter() - step_started, 3)
                if use_trajectory_cache and steps_for_cache:
                    trajectory_cache_path = save_trajectory(goal, app, steps_for_cache)
                return LoopResult(
                    ok=True,
                    status="completed",
                    steps=steps,
                    elapsed_s=round(time.perf_counter() - started, 3),
                    goal=goal,
                    max_steps=max_steps,
                    cache_hit_steps=cache_hit_steps,
                    trajectory_cache_path=trajectory_cache_path,
                )
            loop_step.element_count_after = _wait_for_elements(handle, app=app)
            loop_step.elapsed_s = round(time.perf_counter() - step_started, 3)
            time.sleep(step_pause_s)
            continue

        ok, exec_payload, used_prepared = _execute_decision(
            handle,
            decision,
            app=app,
            graph=graph,
            node_id=f"loop-step-{step_idx}",
            use_prepared=use_prepared_actions,
        )
        loop_step.executed = True
        loop_step.execute_ok = ok
        loop_step.prepared_commit = used_prepared
        if not ok:
            loop_step.error = str((exec_payload or {}).get("error") or "execute failed")
            loop_step.elapsed_s = round(time.perf_counter() - step_started, 3)
            if use_trajectory_cache:
                invalidate_trajectory(goal, app)
            return LoopResult(
                ok=False,
                status="error",
                steps=steps,
                elapsed_s=round(time.perf_counter() - started, 3),
                goal=goal,
                max_steps=max_steps,
                cache_hit_steps=cache_hit_steps,
            )

        _apply_verify(loop_step, verify_fn, decision, exec_payload if isinstance(exec_payload, dict) else None)

        if use_trajectory_cache and not loop_step.from_cache:
            _record_cache_step(steps_for_cache, decision)

        time.sleep(step_pause_s)
        loop_step.element_count_after = _capture_element_count(handle, app=app)
        if (
            loop_step.element_count_after == 0
            and loop_step.decide_action in {"click", "double_click", "right_click", "middle_click", "key"}
            and not (external_done and external_done())
        ):
            loop_step.element_count_after = _wait_for_elements(handle, app=app)

        if metrics_fn:
            metrics = metrics_fn()
            _attach_semantic_metrics(loop_step, metrics)
            _invalidate_prepared_on_semantic_change(graph, metrics)

        signature = (
            loop_step.decide_action,
            loop_step.target_element,
            loop_step.element_count_before,
            loop_step.element_count_after,
        )
        if signature == last_signature:
            no_change_streak += 1
        elif loop_step.element_count_before == loop_step.element_count_after and loop_step.decide_action == "click":
            no_change_streak += 1
        else:
            no_change_streak = 0
        last_signature = signature

        loop_step.elapsed_s = round(time.perf_counter() - step_started, 3)

        if external_done and external_done():
            if use_trajectory_cache and steps_for_cache:
                trajectory_cache_path = save_trajectory(goal, app, steps_for_cache)
            return LoopResult(
                ok=True,
                status="completed",
                steps=steps,
                elapsed_s=round(time.perf_counter() - started, 3),
                goal=goal,
                max_steps=max_steps,
                cache_hit_steps=cache_hit_steps,
                trajectory_cache_path=trajectory_cache_path,
            )

        if no_change_streak >= stuck_threshold:
            if use_trajectory_cache:
                invalidate_trajectory(goal, app)
            return LoopResult(
                ok=False,
                status="stuck",
                steps=steps,
                elapsed_s=round(time.perf_counter() - started, 3),
                goal=goal,
                max_steps=max_steps,
                cache_hit_steps=cache_hit_steps,
            )

    if use_trajectory_cache:
        invalidate_trajectory(goal, app)
    return LoopResult(
        ok=False,
        status="max_steps",
        steps=steps,
        elapsed_s=round(time.perf_counter() - started, 3),
        goal=goal,
        max_steps=max_steps,
        cache_hit_steps=cache_hit_steps,
    )
