"""Bounded decide → act → verify loop for computer_use (#113850).

Runs repeated ``decide`` calls against the System-One lane, executes the
suggested action, and re-captures until the goal is done, the lane abstains
repeatedly, or ``max_steps`` is reached. Approval semantics are unchanged —
each input action still passes through the normal computer_use gate.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable

HandleComputerUse = Callable[[dict[str, Any]], Any]


@dataclass
class LoopStep:
    step: int
    decide_backend: str | None = None
    decide_action: str | None = None
    target_element: int | None = None
    confidence: float | None = None
    fail_open: bool = False
    executed: bool = False
    execute_ok: bool = False
    element_count_before: int = 0
    element_count_after: int = 0
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


def _execute_decision(
    handle: HandleComputerUse,
    decision: dict[str, Any],
    *,
    app: str | None,
) -> tuple[bool, dict[str, Any] | None]:
    action = (decision.get("action") or "").strip().lower()
    if action == "done":
        return True, None
    if action == "escalate":
        return False, {"error": "decision lane escalated to main planner"}
    if action == "wait":
        seconds = float(decision.get("seconds") or 0.5)
        time.sleep(min(max(seconds, 0.1), 5.0))
        return True, None
    if action in {"click", "double_click", "right_click", "middle_click"}:
        target = decision.get("target_element")
        if target is None:
            return False, {"error": f"{action} missing target_element"}
        args: dict[str, Any] = {"action": action, "element": target}
        if app:
            args["app"] = app
        result = _parse_json(handle(args))
        if result.get("error"):
            return False, result
        return True, result
    if action == "key":
        keys = decision.get("keys") or "Return"
        args = {"action": "key", "keys": keys}
        if app:
            args["app"] = app
        result = _parse_json(handle(args))
        if result.get("error"):
            return False, result
        return True, result
    return False, {"error": f"unsupported loop action: {action!r}"}


def run_decide_loop(
    goal: str,
    handle: HandleComputerUse,
    *,
    app: str | None = None,
    max_steps: int = 8,
    stuck_threshold: int = 3,
    step_pause_s: float = 0.35,
    external_done: Callable[[], bool] | None = None,
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

    for step_idx in range(1, max_steps + 1):
        step_started = time.perf_counter()
        loop_step = LoopStep(step=step_idx)
        steps.append(loop_step)

        if external_done and external_done():
            loop_step.elapsed_s = round(time.perf_counter() - step_started, 3)
            return LoopResult(
                ok=True,
                status="completed",
                steps=steps,
                elapsed_s=round(time.perf_counter() - started, 3),
                goal=goal,
                max_steps=max_steps,
            )

        loop_step.element_count_before = _capture_element_count(handle, app=app)
        if loop_step.element_count_before == 0 and not (external_done and external_done()):
            loop_step.element_count_before = _wait_for_elements(handle, app=app)

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
            )

        loop_step.fail_open = bool(decide_payload.get("fail_open"))
        decision = decide_payload.get("decision") or {}

        if loop_step.fail_open:
            fail_open_streak += 1
            loop_step.elapsed_s = round(time.perf_counter() - step_started, 3)
            if fail_open_streak >= stuck_threshold:
                return LoopResult(
                    ok=False,
                    status="fail_open",
                    steps=steps,
                    elapsed_s=round(time.perf_counter() - started, 3),
                    goal=goal,
                    max_steps=max_steps,
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
            return LoopResult(
                ok=True,
                status="done",
                steps=steps,
                elapsed_s=round(time.perf_counter() - started, 3),
                goal=goal,
                max_steps=max_steps,
            )

        if loop_step.decide_action == "escalate" and loop_step.element_count_before == 0:
            if external_done and external_done():
                loop_step.elapsed_s = round(time.perf_counter() - step_started, 3)
                return LoopResult(
                    ok=True,
                    status="completed",
                    steps=steps,
                    elapsed_s=round(time.perf_counter() - started, 3),
                    goal=goal,
                    max_steps=max_steps,
                )
            loop_step.element_count_after = _wait_for_elements(handle, app=app)
            loop_step.elapsed_s = round(time.perf_counter() - step_started, 3)
            time.sleep(step_pause_s)
            continue

        ok, exec_err = _execute_decision(handle, decision, app=app)
        loop_step.executed = True
        loop_step.execute_ok = ok
        if not ok:
            loop_step.error = str((exec_err or {}).get("error") or "execute failed")
            loop_step.elapsed_s = round(time.perf_counter() - step_started, 3)
            return LoopResult(
                ok=False,
                status="error",
                steps=steps,
                elapsed_s=round(time.perf_counter() - started, 3),
                goal=goal,
                max_steps=max_steps,
            )

        time.sleep(step_pause_s)
        loop_step.element_count_after = _capture_element_count(handle, app=app)
        if (
            loop_step.element_count_after == 0
            and loop_step.decide_action in {"click", "double_click", "right_click", "middle_click", "key"}
            and not (external_done and external_done())
        ):
            loop_step.element_count_after = _wait_for_elements(handle, app=app)

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
            return LoopResult(
                ok=True,
                status="completed",
                steps=steps,
                elapsed_s=round(time.perf_counter() - started, 3),
                goal=goal,
                max_steps=max_steps,
            )

        if no_change_streak >= stuck_threshold:
            return LoopResult(
                ok=False,
                status="stuck",
                steps=steps,
                elapsed_s=round(time.perf_counter() - started, 3),
                goal=goal,
                max_steps=max_steps,
            )

    return LoopResult(
        ok=False,
        status="max_steps",
        steps=steps,
        elapsed_s=round(time.perf_counter() - started, 3),
        goal=goal,
        max_steps=max_steps,
    )
