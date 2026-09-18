"""Bounded decide → act loop for computer_use (#113850).

High-confidence System-One decisions execute without a frontier-model round trip.
Uncertain / missing-Jev / needs_generation / escalate fail-open to the planner.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable

Handle = Callable[[dict[str, Any]], Any]


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
    elapsed_s: float = 0.0
    error: str | None = None


@dataclass
class LoopResult:
    ok: bool
    status: str
    goal: str
    max_steps: int
    steps: list[LoopStep] = field(default_factory=list)
    elapsed_s: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _parse(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return {"error": raw}
        return parsed if isinstance(parsed, dict) else {"error": "non-object result"}
    if hasattr(raw, "ok"):
        return {"ok": bool(raw.ok), "error": getattr(raw, "message", None)}
    return {"error": f"unexpected result type: {type(raw)!r}"}


def _execute(handle: Handle, decision: dict[str, Any], *, app: str | None, text: str | None) -> dict[str, Any]:
    action = (decision.get("action") or "").strip().lower()
    if action in {"done", "escalate"}:
        return {"ok": True}
    args: dict[str, Any] = {"action": action}
    if app:
        args["app"] = app
    if action == "wait":
        args["seconds"] = min(max(float(decision.get("seconds") or 0.3), 0.05), 5.0)
    elif action in {"click", "type"}:
        target = decision.get("target_element")
        if target is None:
            return {"ok": False, "error": f"{action} missing target_element"}
        args["element"] = int(target)
        if action == "type":
            if decision.get("needs_generation") and not (text or "").strip():
                return {"ok": False, "error": "needs_generation — planner must supply text"}
            if not (text or "").strip():
                return {"ok": False, "error": "type requires text"}
            args["action"] = "set_value"
            args["value"] = text
    elif action == "key":
        args["keys"] = decision.get("keys") or "Return"
    elif action == "scroll":
        args["direction"] = decision.get("direction") or "down"
    else:
        return {"ok": False, "error": f"unsupported loop action: {action!r}"}
    return _parse(handle(args))


def run_decide_loop(
    goal: str,
    handle: Handle,
    *,
    app: str | None = None,
    max_steps: int = 8,
    text: str | None = None,
) -> LoopResult:
    goal = (goal or "").strip()
    max_steps = max(1, min(int(max_steps), 20))
    if not goal:
        return LoopResult(ok=False, status="error", goal=goal, max_steps=max_steps)
    started = time.perf_counter()
    steps: list[LoopStep] = []
    for step_idx in range(1, max_steps + 1):
        t0 = time.perf_counter()
        step = LoopStep(step=step_idx)
        steps.append(step)
        decide_args: dict[str, Any] = {"action": "decide", "goal": goal}
        if app:
            decide_args["app"] = app
        payload = _parse(handle(decide_args))
        if not payload.get("ok"):
            step.error = str(payload.get("error") or "decide failed")
            step.elapsed_s = round(time.perf_counter() - t0, 3)
            return LoopResult(ok=False, status="error", goal=goal, max_steps=max_steps, steps=steps,
                              elapsed_s=round(time.perf_counter() - started, 3))
        if payload.get("fail_open") or not payload.get("decision"):
            step.fail_open = True
            step.elapsed_s = round(time.perf_counter() - t0, 3)
            return LoopResult(ok=False, status="fail_open", goal=goal, max_steps=max_steps, steps=steps,
                              elapsed_s=round(time.perf_counter() - started, 3))
        decision = payload["decision"]
        step.decide_backend = decision.get("backend")
        step.decide_action = decision.get("action")
        step.confidence = decision.get("confidence")
        if decision.get("target_element") is not None:
            step.target_element = int(decision["target_element"])
        if decision.get("done") or step.decide_action == "done":
            step.elapsed_s = round(time.perf_counter() - t0, 3)
            return LoopResult(ok=True, status="done", goal=goal, max_steps=max_steps, steps=steps,
                              elapsed_s=round(time.perf_counter() - started, 3))
        if step.decide_action == "escalate":
            step.elapsed_s = round(time.perf_counter() - t0, 3)
            return LoopResult(ok=False, status="fail_open", goal=goal, max_steps=max_steps, steps=steps,
                              elapsed_s=round(time.perf_counter() - started, 3))
        executed = _execute(handle, decision, app=app, text=text)
        step.executed = True
        step.execute_ok = bool(executed.get("ok"))
        if not step.execute_ok:
            step.error = str(executed.get("error") or "execute failed")
            step.elapsed_s = round(time.perf_counter() - t0, 3)
            return LoopResult(ok=False, status="error", goal=goal, max_steps=max_steps, steps=steps,
                              elapsed_s=round(time.perf_counter() - started, 3))
        verdict = (executed.get("verdict") or {}).get("decision")
        if verdict == "escalate":
            step.fail_open = True
            step.elapsed_s = round(time.perf_counter() - t0, 3)
            return LoopResult(ok=False, status="fail_open", goal=goal, max_steps=max_steps, steps=steps,
                              elapsed_s=round(time.perf_counter() - started, 3))
        step.elapsed_s = round(time.perf_counter() - t0, 3)
    return LoopResult(ok=False, status="max_steps", goal=goal, max_steps=max_steps, steps=steps,
                      elapsed_s=round(time.perf_counter() - started, 3))
