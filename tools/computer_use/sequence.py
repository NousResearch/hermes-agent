"""`sequence`: execute a short ordered action slice from one model decision (RFC #112639, reordered P0).

Deliberately boring V1: no DAG scheduler, no persistent state, no dependency tracking. A sequence is
orchestration over the existing per-action path — every step runs through the same hard blocks, approval
scopes, and backend handlers as a standalone call. It is not an authorization bypass.
"""

from __future__ import annotations

import json
import time
from typing import Any, Dict, List, NamedTuple, Optional, Tuple

# Steps that may appear in a V1 slice. Read-only actions (capture, list_apps, list_windows) do not belong:
# perception happens before the slice and once more as the final verify. No nested sequences.
_SEQUENCE_STEP_ACTIONS = frozenset({
    "click", "double_click", "right_click", "middle_click", "drag", "scroll",
    "type", "key", "set_value", "wait", "focus_app",
})
# Keys whose values are snapshot-bound (SOM indices / screenshot coordinates from the last capture).
_GROUNDING_KEYS = ("element", "coordinate", "from_element", "to_element", "from_coordinate", "to_coordinate")
_SEQUENCE_MAX_STEPS = 10  # a slice is short by definition; longer plans stay multi-turn
_SEQUENCE_BUDGET_S = 180.0  # whole-slice wall budget; abort is cleaner than a hung slice
_VERIFY_MODES = ("ax_first", "som", "vision")


class _StepVerdict(NamedTuple):
    ok: bool
    reason: str = ""
    detail: str = ""


def _err(error: str, **fields: Any) -> str:
    return json.dumps({"ok": False, "action": "sequence", "error": error, **fields})


def _is_grounded(step_args: Dict[str, Any]) -> bool:
    return any(step_args.get(k) is not None for k in _GROUNDING_KEYS)


def validate_steps(steps: Any) -> Tuple[Optional[List[Tuple[str, Dict[str, Any]]]], Optional[str]]:
    """(parsed, error_json): parsed = [(action, args)]. V1 allows at most one snapshot-grounded step;
    everything else must need no new visual grounding (focus/keyboard/text/wait)."""
    if not isinstance(steps, list) or not steps:
        return None, _err("`steps` must be a non-empty list of step objects")
    if len(steps) > _SEQUENCE_MAX_STEPS:
        return None, _err(f"`steps` is capped at {_SEQUENCE_MAX_STEPS} for V1")
    grounded, parsed = 0, []
    for i, step in enumerate(steps):
        if not isinstance(step, dict):
            return None, _err(f"step {i} must be an object with an `action`")
        sact = str(step.get("action") or "").strip().lower()
        if sact not in _SEQUENCE_STEP_ACTIONS:
            return None, _err(
                f"step {i}: action {sact!r} is not allowed in a V1 sequence",
                hint=f"allowed: {sorted(_SEQUENCE_STEP_ACTIONS)}")
        sargs = {k: v for k, v in step.items() if k != "action"}
        if "capture_after" in sargs:
            return None, _err(f"step {i}: capture_after is not allowed on steps — the slice does one final capture")
        if _is_grounded(sargs) and (grounded := grounded + 1) > 1:
            return None, _err(
                "V1 allows at most one snapshot-grounded step per sequence: after the first "
                "element/coordinate-targeted transition the UI may have changed, so later steps must be "
                "focus/keyboard/text/wait operations. Multi-grounding slices belong to the later "
                "persistent-state work.")
        parsed.append((sact, sargs))
    return parsed, None


def _assess_step(res: Any) -> _StepVerdict:
    """Whether the slice may continue after this step's raw handler result. ``_dispatch`` renders a
    handler's ActionResult as a *text* payload (``ok:false`` with no ``error`` key), so the dict branch
    must read the verdict fields, not just ``error``."""
    payload: Optional[Dict[str, Any]] = None
    if isinstance(res, str):
        try:
            res = json.loads(res)
        except (TypeError, ValueError):
            return _StepVerdict(True)
    if isinstance(res, dict):
        payload = res
    if payload is not None:
        if "error" in payload:
            return _StepVerdict(False, "step_error", str(payload.get("error"))[:200])
        if payload.get("ok") is False:
            return _StepVerdict(False, "action_failed", str(payload.get("message") or payload.get("action"))[:200])
        if payload.get("effect") == "suspected_noop":
            return _StepVerdict(False, "noop_verdict", "backend reports the input likely did not land")
        if payload.get("code") is not None:
            return _StepVerdict(False, "backend_refused", f"{payload['code']}: {str(payload.get('message') or '')[:160]}")
        return _StepVerdict(True)
    # Raw ActionResult (handlers that return it verbatim): transport ok is not enough — a no-op verdict or
    # backend refusal stops the slice.
    if not res.ok:
        return _StepVerdict(False, "action_failed", (res.message or res.action)[:200])
    if res.effect == "suspected_noop":
        return _StepVerdict(False, "noop_verdict", "backend reports the input likely did not land")
    if res.code is not None:
        return _StepVerdict(False, "backend_refused", f"{res.code}: {(res.message or '')[:160]}")
    return _StepVerdict(True)


def _final_verify(backend, args: Dict[str, Any], verify_mode: str) -> Tuple[Any, str, bool, float, float]:
    """(capture, mode_used, fell_back, capture_ms, verification_ms). ax_first tries the AX tree and falls
    back to a screenshot only when the tree is empty (V1 ambiguity heuristic, documented as such)."""
    exact = {k: (getattr(backend, "_last_target", None) or {}).get(k) for k in ("pid", "window_id")}
    target = exact if None not in exact.values() else {"app": getattr(backend, "_last_app", None)}
    t0 = time.perf_counter()
    capture_ms = verification_ms = 0.0
    if verify_mode == "ax_first":
        cap = backend.capture(mode="ax", **target)
        capture_ms = (time.perf_counter() - t0) * 1000.0
        if not cap.elements:
            t1 = time.perf_counter()
            cap = backend.capture(mode="som", **target)
            capture_ms += (time.perf_counter() - t1) * 1000.0
            verification_ms = (time.perf_counter() - t0) * 1000.0
            return cap, "som", True, capture_ms, verification_ms
        verification_ms = (time.perf_counter() - t0) * 1000.0
        return cap, "ax", False, capture_ms, verification_ms
    cap = backend.capture(mode=verify_mode, **target)
    capture_ms = verification_ms = (time.perf_counter() - t0) * 1000.0
    return cap, verify_mode, False, capture_ms, verification_ms


def run_sequence(backend, args: Dict[str, Any], session_id: Optional[str] = None) -> Any:
    """Execute the validated slice, then (on success, with capture_after) one final AX-first verify.
    Aborts at the first failure / no-op verdict / approval denial / lost target / timeout — never cascades."""
    from tools.computer_use import tool as _tool  # late: tool.py imports this module at top level

    t0 = time.perf_counter()
    specs, err = validate_steps(args.get("steps"))
    if err is not None:
        return err
    assert specs is not None
    verify_mode = str(args.get("verify_mode") or "ax_first").strip().lower()
    if verify_mode not in _VERIFY_MODES:
        return _err(f"bad verify_mode {verify_mode!r}", hint=f"use one of {list(_VERIFY_MODES)}")

    metrics: Dict[str, Any] = {
        "task_wall_ms": 0.0, "computer_use_calls": 1, "actions_executed": 0,
        "captures": 0, "image_captures": 0, "slice_length": len(specs),
        "slice_abort_step": None, "tool_ms": 0.0, "capture_ms": 0.0,
        "verification_ms": 0.0, "success": False,
        "verify_mode_requested": verify_mode, "verify_mode_used": None, "verify_fallback": False,
    }
    step_results: List[Dict[str, Any]] = []
    abort: Optional[Dict[str, Any]] = None

    for i, (sact, sargs) in enumerate(specs):
        if (time.perf_counter() - t0) > _SEQUENCE_BUDGET_S:
            abort = {"step": i, "reason": "timeout",
                     "detail": f"slice budget of {_SEQUENCE_BUDGET_S:.0f}s exceeded"}; break
        if (rej := _tool._reject_unsafe(sact, sargs)) is not None:
            abort = {"step": i, "reason": "rejected", "detail": rej}; break
        if _tool._ACTIONS[sact].destructive and (denied := _tool._request_approval(sact, sargs)) is not None:
            abort = {"step": i, "reason": "approval_denied", "detail": denied}; break
        ts = time.perf_counter()
        try:
            res = _tool._dispatch(backend, sact, dict(sargs), session_id=session_id)
        except Exception as e:  # never let a step exception cascade into later steps
            abort = {"step": i, "reason": "exception", "detail": f"{type(e).__name__}: {e}"}; break
        step_ms = (time.perf_counter() - ts) * 1000.0
        metrics["tool_ms"] += step_ms
        verdict = _assess_step(res)
        step_results.append({"step": i, "action": sact, "ok": verdict.ok,
                             "ms": round(step_ms, 1), **({"abort_reason": verdict.reason} if not verdict.ok else {})})
        if not verdict.ok:
            abort = {"step": i, "reason": verdict.reason, "detail": verdict.detail}; break
        metrics["actions_executed"] += 1

    metrics["task_wall_ms"] = (time.perf_counter() - t0) * 1000.0
    if abort is not None:
        metrics["slice_abort_step"] = abort["step"]
        return _err(f"sequence aborted at step {abort['step']}: {abort['reason']}",
                    code="sequence_aborted", abort_reason=abort["reason"],
                    abort_detail=abort["detail"][:300], steps=step_results, sequence_metrics=metrics)

    metrics["success"] = True
    seq_payload = {"ok": True, "action": "sequence", "steps": step_results, "sequence_metrics": metrics}
    if not args.get("capture_after"):
        return json.dumps(seq_payload)

    try:
        cap, mode_used, fell_back, capture_ms, verification_ms = _final_verify(backend, args, verify_mode)
    except Exception as e:
        metrics["success"] = False
        return _err(f"final verification capture failed: {e}", code="sequence_aborted",
                    abort_reason="verify_failed", steps=step_results, sequence_metrics=metrics)
    metrics.update(captures=2 if fell_back else 1, image_captures=1 if cap.png_b64 else 0,
                   capture_ms=capture_ms, verification_ms=verification_ms,
                   verify_mode_used=mode_used, verify_fallback=fell_back,
                   task_wall_ms=(time.perf_counter() - t0) * 1000.0)
    seq_head = json.dumps(seq_payload)
    resp = _tool._capture_response(cap, session_id=session_id)
    if isinstance(resp, dict) and resp.get("_multimodal"):
        resp["content"][0]["text"] = resp["text_summary"] = seq_head + "\n\n" + resp["text_summary"]
        resp["sequence_result"] = seq_payload
        return resp
    return json.dumps({**json.loads(resp), **seq_payload})
