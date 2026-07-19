#!/usr/bin/env python3
"""Dynamic workflow tool.

This is a thin coordinator over Hermes' existing async delegation primitive.
The model owns the workflow shape: it creates or extends dependent worker
steps, records completed worker outputs, and decides what follow-on work to
add. The tool owns the mechanics that are easy to get wrong: dependency
validation, readiness, state, background dispatch through
``delegate_task(background=true)``, and cancellation of dispatched async
delegations.
"""

from __future__ import annotations

import json
import logging
import re
import threading
import time
import uuid
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

from tools.registry import registry, tool_error

logger = logging.getLogger(__name__)


_WORKFLOW_ID_RE = re.compile(r"^[A-Za-z0-9_.:-]{1,96}$")
_NODE_ID_RE = re.compile(r"^[A-Za-z0-9_.:-]{1,96}$")
_RESULT_STATUSES = {"completed", "failed", "cancelled"}
_ASYNC_COMPLETED = {"completed", "success", "succeeded"}
_ASYNC_FAILED = {"error", "failed", "failure", "timeout"}
_ASYNC_CANCELLED = {"cancelled", "canceled", "interrupted"}
_MAX_NODES_PER_WORKFLOW = 256
_MAX_DISPATCH_PER_CALL = 16
_MAX_TEXT_CHARS = 16_000

_workflows_lock = threading.RLock()
_workflows: Dict[Tuple[str, str], Dict[str, Any]] = {}
_reconciled_async_delegations: set[str] = set()


def _now() -> float:
    return time.time()


def _json(data: Dict[str, Any]) -> str:
    return json.dumps(data, ensure_ascii=False, sort_keys=True)


def _new_workflow_id() -> str:
    return f"wf_{uuid.uuid4().hex[:10]}"


def _workflow_scope(parent_agent: Any = None) -> str:
    """Return the in-process visibility scope for a workflow."""
    for attr in ("session_id", "_gateway_session_key"):
        value = getattr(parent_agent, attr, None)
        if value:
            return f"{attr}:{value}"
    return "global"


def _cap_text(value: Any, *, default: str = "") -> str:
    text = str(value if value is not None else default).strip()
    if len(text) > _MAX_TEXT_CHARS:
        return text[: _MAX_TEXT_CHARS - 14] + "\n...[truncated]"
    return text


def _normalise_id(value: Any, *, field: str, pattern: re.Pattern[str]) -> Tuple[Optional[str], Optional[str]]:
    text = str(value or "").strip()
    if not text:
        return None, f"{field} is required"
    if not pattern.match(text):
        return None, f"{field} must match {pattern.pattern}"
    return text, None


def _as_string_list(value: Any, *, field: str) -> Tuple[List[str], Optional[str]]:
    if value is None:
        return [], None
    if not isinstance(value, list):
        return [], f"{field} must be an array"
    result: List[str] = []
    for item in value:
        text = str(item or "").strip()
        if not text:
            continue
        result.append(text)
    return result, None


def _normalise_role(value: Any) -> str:
    role = str(value or "leaf").strip().lower()
    if role not in {"leaf", "orchestrator"}:
        return "leaf"
    return role


def _slug_id(value: str, *, fallback: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.:-]+", "-", value.strip().lower()).strip("-")
    return (slug or fallback)[:96]


def _normalise_nodes(raw_nodes: Any) -> Tuple[List[Dict[str, Any]], List[str]]:
    issues: List[str] = []
    if raw_nodes is None:
        return [], issues
    if not isinstance(raw_nodes, list):
        return [], ["nodes must be an array"]

    nodes: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for index, raw in enumerate(raw_nodes):
        if not isinstance(raw, dict):
            issues.append(f"nodes[{index}] must be an object")
            continue

        node_id, issue = _normalise_id(raw.get("node_id"), field=f"nodes[{index}].node_id", pattern=_NODE_ID_RE)
        if issue:
            issues.append(issue)
            continue
        assert node_id is not None
        if node_id in seen:
            issues.append(f"duplicate node_id: {node_id}")
            continue
        seen.add(node_id)

        goal = _cap_text(raw.get("goal"))
        if not goal:
            issues.append(f"nodes[{index}].goal is required")
            continue

        depends_on, depends_issue = _as_string_list(raw.get("depends_on"), field=f"nodes[{index}].depends_on")
        if depends_issue:
            issues.append(depends_issue)
            continue
        if node_id in depends_on:
            issues.append(f"node {node_id} cannot depend on itself")
            continue

        toolsets, toolsets_issue = _as_string_list(raw.get("toolsets"), field=f"nodes[{index}].toolsets")
        if toolsets_issue:
            issues.append(toolsets_issue)
            continue

        phase_title = _cap_text(raw.get("phase_title") or raw.get("phase"))
        phase_id = ""
        if raw.get("phase_id"):
            phase_id, phase_issue = _normalise_id(
                raw.get("phase_id"),
                field=f"nodes[{index}].phase_id",
                pattern=_NODE_ID_RE,
            )
            if phase_issue:
                issues.append(phase_issue)
                continue
            assert phase_id is not None
        elif phase_title:
            phase_id = _slug_id(phase_title, fallback=f"phase-{index + 1}")

        nodes.append(
            {
                "node_id": node_id,
                "title": _cap_text(raw.get("title") or raw.get("task_title") or goal),
                "phase_id": phase_id or None,
                "phase_title": phase_title or None,
                "goal": goal,
                "context": _cap_text(raw.get("context")),
                "depends_on": depends_on,
                "toolsets": toolsets or None,
                "role": _normalise_role(raw.get("role")),
                "status": "pending",
                "delegation_id": None,
                "summary": None,
                "error": None,
                "created_at": _now(),
                "updated_at": _now(),
            }
        )
    return nodes, issues


def _validate_graph(nodes: Dict[str, Dict[str, Any]]) -> List[str]:
    issues: List[str] = []
    for node in nodes.values():
        for dep in node.get("depends_on") or []:
            if dep not in nodes:
                issues.append(f"node {node['node_id']} depends on unknown node {dep}")
    if issues:
        return issues

    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(node_id: str, stack: List[str]) -> None:
        if node_id in visited:
            return
        if node_id in visiting:
            issues.append("cycle detected: " + " -> ".join(stack + [node_id]))
            return
        visiting.add(node_id)
        for dep in nodes[node_id].get("depends_on") or []:
            visit(dep, stack + [node_id])
        visiting.remove(node_id)
        visited.add(node_id)

    for node_id in nodes:
        visit(node_id, [])
    return issues


def _ready_node_ids(workflow: Dict[str, Any]) -> List[str]:
    nodes = workflow["nodes"]
    ready: List[str] = []
    for node_id, node in nodes.items():
        if node.get("status") != "pending":
            continue
        if all(nodes[dep].get("status") == "completed" for dep in node.get("depends_on") or []):
            ready.append(node_id)
    return ready


def _workflow_status(workflow: Dict[str, Any]) -> str:
    if workflow.get("cancelled_at"):
        return "cancelled"
    nodes = workflow["nodes"]
    if not nodes:
        return "empty"
    statuses = {node.get("status") for node in nodes.values()}
    if "failed" in statuses:
        return "failed"
    if "dispatched" in statuses:
        return "running"
    if statuses <= {"completed"}:
        return "completed"
    if _ready_node_ids(workflow):
        return "ready"
    return "waiting"


def _node_status_from_async(status: Any) -> Optional[str]:
    text = str(status or "").strip().lower()
    if text in _ASYNC_COMPLETED:
        return "completed"
    if text in _ASYNC_FAILED:
        return "failed"
    if text in _ASYNC_CANCELLED:
        return "cancelled"
    return None


def _reconcile_async_delegations(workflow: Dict[str, Any]) -> List[str]:
    """Refresh dispatched workflow nodes from retained async delegation records."""
    dispatched = {
        node.get("delegation_id"): node
        for node in workflow["nodes"].values()
        if node.get("status") == "dispatched" and node.get("delegation_id")
    }
    if not dispatched:
        return []

    try:
        from tools.async_delegation import list_async_delegations

        records = list_async_delegations()
    except Exception as exc:  # pragma: no cover - status must stay best-effort
        logger.debug("dynamic_workflow async reconciliation failed: %s", exc)
        return []

    updated: List[str] = []
    for record in records:
        delegation_id = record.get("delegation_id")
        node = dispatched.get(delegation_id)
        if not node:
            continue
        next_status = _node_status_from_async(record.get("status"))
        if not next_status:
            continue

        node["status"] = next_status
        node["summary"] = _cap_text(record.get("summary"))
        node["error"] = _cap_text(record.get("error")) if record.get("error") else None
        node["completed_at"] = record.get("completed_at") or _now()
        for key in (
            "duration_seconds",
            "api_calls",
            "input_tokens",
            "output_tokens",
            "reasoning_tokens",
            "cost_usd",
            "exit_reason",
            "model",
        ):
            if key in record and record.get(key) is not None:
                node[key] = record.get(key)
        node["async_completion_reconciled"] = True
        node["updated_at"] = _now()
        if delegation_id:
            _reconciled_async_delegations.add(str(delegation_id))
        updated.append(node["node_id"])

    if updated:
        workflow["updated_at"] = _now()
    return updated


def is_async_completion_reconciled(
    delegation_id: Any,
    *,
    workflow_id: Any = None,
    node_id: Any = None,
) -> bool:
    """Return true when a workflow already consumed an async completion."""
    deleg_id = str(delegation_id or "").strip()
    if not deleg_id:
        return False

    with _workflows_lock:
        if deleg_id in _reconciled_async_delegations:
            return True
        for workflow in _workflows.values():
            if workflow_id and workflow.get("workflow_id") != workflow_id:
                continue
            for node in workflow["nodes"].values():
                if node_id and node.get("node_id") != node_id:
                    continue
                if (
                    node.get("delegation_id") == deleg_id
                    and node.get("async_completion_reconciled") is True
                ):
                    _reconciled_async_delegations.add(deleg_id)
                    return True
    return False


def _public_workflow(workflow: Dict[str, Any]) -> Dict[str, Any]:
    _reconcile_async_delegations(workflow)
    public = {
        "workflow_id": workflow["workflow_id"],
        "objective": workflow["objective"],
        "context": workflow.get("context") or "",
        "status": _workflow_status(workflow),
        "ready_node_ids": _ready_node_ids(workflow),
        "created_at": workflow["created_at"],
        "updated_at": workflow["updated_at"],
        "cancelled_at": workflow.get("cancelled_at"),
        "nodes": [],
    }
    for node_id in workflow["node_order"]:
        node = workflow["nodes"][node_id]
        public["nodes"].append(deepcopy(node))
    return public


def _workflow_key(workflow_id: Any, parent_agent: Any = None) -> Tuple[Optional[Tuple[str, str]], Optional[str]]:
    wf_id, issue = _normalise_id(workflow_id, field="workflow_id", pattern=_WORKFLOW_ID_RE)
    if issue:
        return None, issue
    assert wf_id is not None
    return (_workflow_scope(parent_agent), wf_id), None


def _get_workflow(workflow_id: Any, parent_agent: Any = None) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    key, issue = _workflow_key(workflow_id, parent_agent)
    if issue:
        return None, issue
    assert key is not None
    workflow = _workflows.get(key)
    if workflow is None:
        return None, f"unknown workflow_id: {key[1]}"
    return workflow, None


def _merge_nodes(workflow: Dict[str, Any], new_nodes: List[Dict[str, Any]]) -> Optional[str]:
    if len(workflow["nodes"]) + len(new_nodes) > _MAX_NODES_PER_WORKFLOW:
        return f"workflow node limit exceeded ({_MAX_NODES_PER_WORKFLOW})"

    merged = dict(workflow["nodes"])
    for node in new_nodes:
        node_id = node["node_id"]
        if node_id in merged:
            return f"duplicate node_id: {node_id}"
        merged[node_id] = node

    issues = _validate_graph(merged)
    if issues:
        return "; ".join(issues)

    for node in new_nodes:
        workflow["nodes"][node["node_id"]] = node
        workflow["node_order"].append(node["node_id"])
    workflow["updated_at"] = _now()
    return None


def _worker_context(workflow: Dict[str, Any], node: Dict[str, Any]) -> str:
    lines = [
        "[Dynamic workflow worker]",
        f"workflow_id: {workflow['workflow_id']}",
        f"node_id: {node['node_id']}",
        f"objective: {workflow['objective']}",
    ]
    if node.get("phase_title") or node.get("phase_id"):
        lines.append(f"phase: {node.get('phase_title') or node.get('phase_id')}")
    if node.get("title"):
        lines.append(f"task_title: {node['title']}")
    if workflow.get("context"):
        lines.extend(["", "Workflow context:", workflow["context"]])
    if node.get("context"):
        lines.extend(["", "Node context:", node["context"]])
    if node.get("depends_on"):
        lines.extend(["", "Completed dependency node ids:", ", ".join(node["depends_on"])])
    lines.extend(
        [
            "",
            "When you finish, include workflow_id and node_id in your summary. "
            "Hermes reconciles async completion into the workflow automatically; "
            "the parent coordinator may still record or refine results with "
            "dynamic_workflow(action='record_result') and may extend the workflow "
            "based on your output.",
        ]
    )
    return "\n".join(lines)


def _dispatch_ready(workflow: Dict[str, Any], parent_agent: Any, max_dispatch: int) -> Dict[str, Any]:
    _reconcile_async_delegations(workflow)
    if parent_agent is None:
        return {
            "dispatched": [],
            "dispatch_errors": [{"error": "dynamic_workflow dispatch requires a parent agent context"}],
        }
    if workflow.get("cancelled_at"):
        return {"dispatched": [], "dispatch_errors": [{"error": "workflow is cancelled"}]}

    try:
        max_dispatch = max(1, min(int(max_dispatch), _MAX_DISPATCH_PER_CALL))
    except (TypeError, ValueError):
        max_dispatch = _MAX_DISPATCH_PER_CALL

    dispatched: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []
    for node_id in _ready_node_ids(workflow)[:max_dispatch]:
        node = workflow["nodes"][node_id]
        from tools import delegate_tool

        raw = delegate_tool.delegate_task(
            goal=node["goal"],
            context=_worker_context(workflow, node),
            toolsets=node.get("toolsets"),
            role=node.get("role") or "leaf",
            background=True,
            parent_agent=parent_agent,
            _observability_context={
                "workflow_id": workflow["workflow_id"],
                "workflow_node_id": node_id,
                "workflow_phase_id": node.get("phase_id") or "",
                "workflow_phase_title": node.get("phase_title") or "",
                "workflow_task_title": node.get("title") or node["goal"],
                "workflow_objective": workflow["objective"],
                "task_prompt": node["goal"],
                "task_context": node.get("context") or "",
            },
        )
        try:
            parsed = json.loads(raw)
        except Exception:
            parsed = {"error": raw}

        if parsed.get("status") == "dispatched" and parsed.get("delegation_id"):
            optional_ids = {
                key: parsed[key]
                for key in ("subagent_id", "child_session_id")
                if parsed.get(key)
            }
            node["status"] = "dispatched"
            node["delegation_id"] = parsed["delegation_id"]
            node.update(optional_ids)
            node["dispatched_at"] = _now()
            node["updated_at"] = _now()
            dispatched.append({"node_id": node_id, "delegation_id": parsed["delegation_id"], **optional_ids})
        else:
            errors.append({"node_id": node_id, "error": parsed.get("error") or parsed})

    workflow["updated_at"] = _now()
    return {"dispatched": dispatched, "dispatch_errors": errors}


def _create(args: Dict[str, Any], parent_agent: Any) -> str:
    objective = _cap_text(args.get("objective"))
    if not objective:
        return tool_error("objective is required for action='create'.")
    workflow_id = str(args.get("workflow_id") or _new_workflow_id()).strip()
    wf_id, issue = _normalise_id(workflow_id, field="workflow_id", pattern=_WORKFLOW_ID_RE)
    if issue:
        return tool_error(issue)
    assert wf_id is not None

    nodes, issues = _normalise_nodes(args.get("nodes"))
    if issues:
        return tool_error("; ".join(issues))

    with _workflows_lock:
        key = (_workflow_scope(parent_agent), wf_id)
        if key in _workflows:
            return tool_error(f"workflow_id already exists: {wf_id}")
        workflow = {
            "workflow_id": wf_id,
            "scope": key[0],
            "objective": objective,
            "context": _cap_text(args.get("context")),
            "nodes": {},
            "node_order": [],
            "created_at": _now(),
            "updated_at": _now(),
            "cancelled_at": None,
        }
        issue = _merge_nodes(workflow, nodes)
        if issue:
            return tool_error(issue)
        _workflows[key] = workflow
        dispatch = (
            _dispatch_ready(workflow, parent_agent, args.get("max_dispatch") or _MAX_DISPATCH_PER_CALL)
            if args.get("dispatch_ready")
            else {"dispatched": [], "dispatch_errors": []}
        )
        return _json({"workflow": _public_workflow(workflow), **dispatch})


def _add_nodes(args: Dict[str, Any], parent_agent: Any) -> str:
    nodes, issues = _normalise_nodes(args.get("nodes"))
    if issues:
        return tool_error("; ".join(issues))
    if not nodes:
        return tool_error("nodes is required for action='add_nodes'.")

    with _workflows_lock:
        workflow, issue = _get_workflow(args.get("workflow_id"), parent_agent)
        if issue:
            return tool_error(issue)
        assert workflow is not None
        if workflow.get("cancelled_at"):
            return tool_error("cannot add nodes to a cancelled workflow")
        issue = _merge_nodes(workflow, nodes)
        if issue:
            return tool_error(issue)
        dispatch = (
            _dispatch_ready(workflow, parent_agent, args.get("max_dispatch") or _MAX_DISPATCH_PER_CALL)
            if args.get("dispatch_ready")
            else {"dispatched": [], "dispatch_errors": []}
        )
        return _json({"workflow": _public_workflow(workflow), **dispatch})


def _record_result(args: Dict[str, Any], parent_agent: Any) -> str:
    node_id, issue = _normalise_id(args.get("node_id"), field="node_id", pattern=_NODE_ID_RE)
    if issue:
        return tool_error(issue)
    assert node_id is not None

    status = str(args.get("status") or "completed").strip().lower()
    if status not in _RESULT_STATUSES:
        return tool_error(f"status must be one of: {', '.join(sorted(_RESULT_STATUSES))}")

    with _workflows_lock:
        workflow, issue = _get_workflow(args.get("workflow_id"), parent_agent)
        if issue:
            return tool_error(issue)
        assert workflow is not None
        node = workflow["nodes"].get(node_id)
        if node is None:
            return tool_error(f"unknown node_id: {node_id}")
        if status == "completed" and not all(
            workflow["nodes"][dep].get("status") == "completed"
            for dep in node.get("depends_on") or []
        ):
            return tool_error(f"node {node_id} cannot complete before its dependencies complete")
        node["status"] = status
        node["summary"] = _cap_text(args.get("summary"))
        node["error"] = _cap_text(args.get("error")) if args.get("error") else None
        if "result" in args:
            node["result"] = deepcopy(args.get("result"))
        node["completed_at"] = _now()
        node["updated_at"] = _now()
        workflow["updated_at"] = _now()
        return _json({"workflow": _public_workflow(workflow)})


def _status(args: Dict[str, Any], parent_agent: Any) -> str:
    with _workflows_lock:
        if args.get("workflow_id"):
            workflow, issue = _get_workflow(args.get("workflow_id"), parent_agent)
            if issue:
                return tool_error(issue)
            assert workflow is not None
            return _json({"workflow": _public_workflow(workflow)})
        scope = _workflow_scope(parent_agent)
        return _json(
            {
                "workflows": [
                    _public_workflow(wf)
                    for (wf_scope, _wf_id), wf in _workflows.items()
                    if wf_scope == scope
                ]
            }
        )


def _dispatch(args: Dict[str, Any], parent_agent: Any) -> str:
    with _workflows_lock:
        workflow, issue = _get_workflow(args.get("workflow_id"), parent_agent)
        if issue:
            return tool_error(issue)
        assert workflow is not None
        dispatch = _dispatch_ready(workflow, parent_agent, args.get("max_dispatch") or _MAX_DISPATCH_PER_CALL)
        return _json({"workflow": _public_workflow(workflow), **dispatch})


def _cancel(args: Dict[str, Any], parent_agent: Any) -> str:
    interrupt = bool(args.get("interrupt", True))
    interrupted: List[str] = []
    with _workflows_lock:
        workflow, issue = _get_workflow(args.get("workflow_id"), parent_agent)
        if issue:
            return tool_error(issue)
        assert workflow is not None
        workflow["cancelled_at"] = _now()
        workflow["updated_at"] = _now()
        for node in workflow["nodes"].values():
            if node.get("status") == "pending":
                node["status"] = "cancelled"
                node["updated_at"] = _now()
            elif node.get("status") == "dispatched":
                node["cancel_requested"] = True
                node["updated_at"] = _now()
                if interrupt and node.get("delegation_id"):
                    from tools.async_delegation import interrupt_delegation

                    if interrupt_delegation(node["delegation_id"], reason="dynamic_workflow cancelled"):
                        interrupted.append(node["delegation_id"])
        return _json({"workflow": _public_workflow(workflow), "interrupted_delegation_ids": interrupted})


def handle_dynamic_workflow(args: Dict[str, Any], parent_agent: Any = None) -> str:
    """Tool entrypoint."""
    if not isinstance(args, dict):
        return tool_error("dynamic_workflow arguments must be an object.")
    action = str(args.get("action") or "").strip().lower()
    if action == "create":
        return _create(args, parent_agent)
    if action == "add_nodes":
        return _add_nodes(args, parent_agent)
    if action == "record_result":
        return _record_result(args, parent_agent)
    if action == "dispatch_ready":
        return _dispatch(args, parent_agent)
    if action == "status":
        return _status(args, parent_agent)
    if action == "cancel":
        return _cancel(args, parent_agent)
    return tool_error("action must be one of: create, add_nodes, record_result, dispatch_ready, status, cancel")


def _reset_for_tests() -> None:
    with _workflows_lock:
        _workflows.clear()
        _reconciled_async_delegations.clear()


DYNAMIC_WORKFLOW_SCHEMA = {
    "name": "dynamic_workflow",
    "description": (
        "Create and run model-authored dynamic workflows using Hermes async "
        "delegation for ready worker nodes."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["create", "add_nodes", "record_result", "dispatch_ready", "status", "cancel"],
                "description": "Workflow operation to perform.",
            },
            "workflow_id": {
                "type": "string",
                "description": "Workflow id. Optional for create; required for all other actions.",
            },
            "objective": {
                "type": "string",
                "description": "Overall objective for action=create.",
            },
            "context": {
                "type": "string",
                "description": "Workflow-level context shared with worker nodes.",
            },
            "nodes": {
                "type": "array",
                "description": (
                    "Worker nodes to create or add. Add new nodes after recording "
                    "phase outputs when the next steps depend on what workers found."
                ),
                "items": {
                    "type": "object",
                    "properties": {
                        "node_id": {"type": "string", "description": "Stable node id unique within the workflow."},
                        "title": {
                            "type": "string",
                            "description": "Short reader-facing task title for workflow monitors.",
                        },
                        "phase_id": {
                            "type": "string",
                            "description": "Stable phase id used to group related workflow tasks in monitors.",
                        },
                        "phase_title": {
                            "type": "string",
                            "description": "Reader-facing phase label, e.g. Investigate, Build, Verify.",
                        },
                        "goal": {"type": "string", "description": "Self-contained worker goal."},
                        "context": {"type": "string", "description": "Node-specific context."},
                        "depends_on": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Node ids that must complete before this node is ready.",
                        },
                        "toolsets": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Toolsets to enable for this node's delegated worker.",
                        },
                        "role": {
                            "type": "string",
                            "enum": ["leaf", "orchestrator"],
                            "description": "Delegated worker role. Use leaf unless the worker must delegate.",
                        },
                    },
                    "required": ["node_id", "goal"],
                },
            },
            "node_id": {
                "type": "string",
                "description": "Node id for action=record_result.",
            },
            "status": {
                "type": "string",
                "enum": ["completed", "failed", "cancelled"],
                "description": "Result status for action=record_result.",
            },
            "summary": {
                "type": "string",
                "description": "Bounded worker result summary for action=record_result.",
            },
            "error": {
                "type": "string",
                "description": "Failure details for action=record_result when status=failed or cancelled.",
            },
            "result": {
                "type": "object",
                "description": "Optional structured result for action=record_result.",
            },
            "dispatch_ready": {
                "type": "boolean",
                "description": "When true, dispatch ready pending nodes immediately after create/add_nodes.",
            },
            "max_dispatch": {
                "type": "integer",
                "description": f"Maximum ready nodes to dispatch in this call, capped at {_MAX_DISPATCH_PER_CALL}.",
            },
            "interrupt": {
                "type": "boolean",
                "description": "For action=cancel, interrupt dispatched async delegations when possible.",
            },
        },
        "required": ["action"],
    },
}


registry.register(
    name="dynamic_workflow",
    toolset="dynamic_workflow",
    schema=DYNAMIC_WORKFLOW_SCHEMA,
    handler=lambda args, **kw: handle_dynamic_workflow(args, parent_agent=kw.get("parent_agent")),
    emoji="DW",
)
