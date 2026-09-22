"""Opt-in delegate_task pre-build interception -> sidecar_service (Sidecar-Adoption STEP 3,
part B). Reuses ``hermes_cli.kanban_sidecar_route``'s classifier registry and in-process
dispatch (``sidecar_suite.contract.execute()``) so eligibility rules live in exactly one
place; delegate_task never talks to the sidecar over HTTP either, matching the
architecture-correction comment on t_9dbf5547.

Design: annotate task dicts in place (never remove/reindex ``task_list``) so
``task_index`` stays a stable positional id from normalization all the way to the
returned result entries -- the model's per-task correlation contract in
``tools.delegate_tool_dispatch._Batch``/``_execute_and_aggregate`` depends on that. A
routed task keeps its slot in ``task_list`` (so ``_build_children``/schema/image coercion
still see a consistent index space) but carries a precomputed result entry; the batch
runner (``_Batch.run_child``) returns that entry directly instead of running a child.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from hermes_cli.kanban_sidecar_route import sidecar_routing_enabled, try_sidecar_route

_ROUTED_ENTRY_KEY = "_sidecar_routed_entry"


def _fabricate_routed_entry(task_index: int, sidecar_result: dict) -> Dict[str, Any]:
    """Result entry shaped like ``tools.delegate_tool_child_run._build_result_entry``'s
    contract (status/summary/api_calls/duration_seconds/...) so the tool's return shape to
    the model is identical whether a task ran as a real subagent or was sidecar-routed."""
    import json

    payload = sidecar_result.get("payload")
    return {
        "task_index": task_index, "status": "completed", "exit_reason": "completed", "truncated": False,
        "summary": json.dumps(payload, ensure_ascii=False),
        "error": None, "api_calls": 0, "duration_seconds": 0,
        "model": None, "tokens": {"input": 0, "output": 0}, "tool_trace": [],
        "cost_usd": 0.0, "cost_status": "sidecar_routed",
        "_sidecar_operation": sidecar_result.get("operation"), "_child_role": None, "_child_cost_usd": 0.0,
    }


class _TaskDictLike(dict):
    """Thin wrapper so ``classify_task_for_sidecar`` (which expects a Kanban ``Task``-like
    object with ``.title``/``.body``) can also read a delegate_task task dict's ``goal``/
    ``context`` fields. ``goal`` maps to ``title`` and ``context`` to ``body`` -- the two
    free-text fields any real classifier would inspect."""

    @property
    def title(self) -> str:
        return str(self.get("goal") or "")

    @property
    def body(self) -> Optional[str]:
        return self.get("context")


def annotate_sidecar_routes(task_list: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], int]:
    """Mutate a copy of each eligible task in ``task_list`` to carry a precomputed sidecar
    result entry (see ``_ROUTED_ENTRY_KEY``); ineligible/failed tasks are returned
    unchanged. Returns ``(new_task_list, routed_count)`` — same length and order as the
    input, so callers never need to renumber ``task_index``. No-op (returns the same list
    object) when the feature flag is off, so this call costs nothing when disabled."""
    if not sidecar_routing_enabled():
        return task_list, 0
    new_list: List[Dict[str, Any]] = []
    routed_count = 0
    for i, task in enumerate(task_list):
        sidecar_result = try_sidecar_route(_TaskDictLike(task)) if isinstance(task, dict) else None
        if sidecar_result is None:
            new_list.append(task)
            continue
        annotated = dict(task)
        annotated[_ROUTED_ENTRY_KEY] = _fabricate_routed_entry(i, sidecar_result)
        new_list.append(annotated)
        routed_count += 1
    return new_list, routed_count


def routed_entry(task: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """The precomputed entry stashed by ``annotate_sidecar_routes``, or None."""
    return task.get(_ROUTED_ENTRY_KEY) if isinstance(task, dict) else None
