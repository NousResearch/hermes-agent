"""Delegation adapter for guided model routing (design §4 step 4, §6 "Delegation").

Wires `tools/delegate_tool.py::_build_child_agent` to the SAME neutral primitives Kanban already
uses (`agent.managed_route_runtime`, `agent.model_selection`, `agent.model_selection_store`,
`agent.model_selection_guard`) -- no duplicate selection/persistence/guard logic here, only the
delegation-specific requirements shape and child-authority rules.

Public entry: `resolve_delegation_route(task, parent_agent, routing_cfg)` -> `None` (task is
unmanaged and the parent carries no managed authority to inherit) or a dict of AIAgent overrides
plus `receipt_id`/`routing_home` for the caller to stamp onto the constructed child via
`stamp_managed_route(child, resolution)`.

Nested-child authority (design §6, §12 "no silent escape through parent inheritance"):
  - A parent agent with its OWN wired receipt (`agent._managed_routing_receipt_id`) is itself a
    managed child -- ANY further delegation from it inherits that authority ceiling. It cannot
    clear managed status by simply omitting `routing_role` on the sub-task (that would let a
    managed run silently escape into an unmanaged one at the next hop), and it cannot request a
    DIFFERENT policy_id / role than the one it was itself authorized under (that would let it
    "widen the roster" by asking for a broader mandate than its own).
  - A task's `routing_role`/`routing_requirements`/`routing_policy_id` are structured, operator/
    caller-supplied inputs (mirroring the Kanban `--routing-role`/`--routing-requirements` CLI
    contract) -- the model calling delegate_task cannot invent admission into the roster: the
    selector call is the ONLY place a route is chosen, and it runs against the same trusted
    `policy_id` used everywhere else (default `"kanban-default"` unless the parent's own receipt
    pins a different one, which then becomes mandatory).
"""
from __future__ import annotations

import time
from typing import Any, Optional

from agent.managed_route_runtime import resolve_route
from agent.model_selection_types import RoutingBlocked

__all__ = [
    "resolve_delegation_route", "stamp_managed_route", "stamp_shadow_route",
    "DelegationRoutingBlocked",
]

DEFAULT_DELEGATION_POLICY_ID = "kanban-default"


class DelegationRoutingBlocked(RoutingBlocked):
    """Re-raised with the delegation call site's context; same reason codes as RoutingBlocked."""


def _parent_managed_context(parent_agent) -> tuple[Optional[str], Optional[dict]]:
    """``(hermes_home, decision)`` the parent's OWN receipt was authorized under, or
    ``(None, None)`` when the parent carries no managed authority to inherit. Read from the
    parent's already-persisted receipt (never re-derived/guessed) so the ceiling this function
    enforces is the ceiling the parent itself was actually given."""
    receipt_id = getattr(parent_agent, "_managed_routing_receipt_id", None)
    hermes_home = getattr(parent_agent, "_managed_routing_home", None)
    moa_authority = getattr(parent_agent, "_managed_moa_authority", None)
    if isinstance(moa_authority, tuple) and len(moa_authority) == 2:
        turn_id, resolution = moa_authority
        if turn_id == getattr(parent_agent, "_current_turn_id", None):
            receipt_id = resolution["receipt_id"]
            hermes_home = resolution["routing_home"]
    # Real values are always a non-empty str (receipt id) / str-or-Path (home) -- a test double
    # (MagicMock etc.) that auto-vivifies unset attributes returns a Mock object here, not None;
    # require the actual expected types so an unconfigured mock parent is correctly "unmanaged"
    # rather than accidentally treated as carrying inherited authority.
    if not isinstance(receipt_id, str) or not receipt_id:
        return None, None
    if not isinstance(hermes_home, (str,)) and not hasattr(hermes_home, "__fspath__"):
        raise DelegationRoutingBlocked("stale_or_revoked_decision", "managed parent is missing its origin home")
    from agent.model_selection_store import get_receipt

    decision = get_receipt(hermes_home, receipt_id)
    if decision is None:
        # The parent's own receipt vanished/was never persisted -- fail closed rather than treat
        # a managed parent as suddenly unmanaged (§12 no silent escape).
        raise DelegationRoutingBlocked(
            "stale_or_revoked_decision",
            "parent agent carries a managed routing receipt id with no persisted decision; "
            "refusing to authorize a nested delegation under a vanished parent receipt",
        )
    from agent.managed_route_runtime import enforce_worker_route
    from agent.model_selection_guard import managed_child_kwargs

    route = managed_child_kwargs(decision)
    enforce_worker_route(hermes_home, receipt_id, actual_provider=route["provider"],
                         actual_model=route["model"], actual_endpoint=route["endpoint"],
                         actual_reasoning=route["reasoning_effort"], record_outcome=False)
    return hermes_home, decision


def _task_intake(task: dict) -> tuple[Optional[str], Optional[dict], Optional[str], Optional[str]]:
    """``(routing_role, routing_requirements, routing_policy_id, routing_mode)`` as supplied on the task dict --
    genuine caller intake, never a model-invented default. Validated with the SAME validator
    Kanban's `--routing-requirements` CLI flag uses so the two adapters share one schema."""
    role = task.get("routing_role")
    role = role.strip() if isinstance(role, str) and role.strip() else None
    raw_requirements = task.get("routing_requirements")
    requirements = None
    if raw_requirements is not None:
        from hermes_cli.kanban_model_routing import validate_routing_requirements

        requirements = validate_routing_requirements(raw_requirements)  # raises ValueError on malformed intake
    policy_id = task.get("routing_policy_id")
    policy_id = policy_id.strip() if isinstance(policy_id, str) and policy_id.strip() else None
    from hermes_cli.kanban_model_routing import normalize_routing_mode

    mode = normalize_routing_mode(task.get("routing_mode"), has_role=bool(role))
    return role, requirements, policy_id, mode


def resolve_delegation_route(
    task: dict, parent_agent, task_index: int, attempt_id: str = "0",
) -> Optional[dict]:
    """Resolve a managed route for one delegation task, or ``None`` when neither the task nor an
    inherited parent ceiling calls for routing (the existing unmanaged path, byte-for-byte).

    Raises ``RoutingBlocked``/``DelegationRoutingBlocked`` on any policy/eligibility/ceiling
    failure; the caller (``_build_children``) must propagate it as a real spawn failure -- the
    existing ``ValueError`` preflight-failure contract in ``_build_child_agent`` -- and must NEVER
    fall back to constructing an unmanaged/default-route child instead (§5 no silent escape).
    """
    task_role, task_requirements, task_policy_id, task_mode = _task_intake(task)
    parent_home, parent_decision = _parent_managed_context(parent_agent)
    parent_policy_id = parent_decision["policy_id"] if parent_decision else None
    parent_role = parent_decision["requirements"]["role"] if parent_decision else None

    if parent_policy_id is None:
        # Parent carries no managed authority: this task is managed ONLY if it explicitly asked.
        if task_role is None:
            return None
        hermes_home = _current_hermes_home()
        policy_id = task_policy_id or DEFAULT_DELEGATION_POLICY_ID
        role = task_role
        mode = task_mode
    else:
        # Parent IS a managed child: authority ceiling applies regardless of what this sub-task
        # asks for. Cannot clear managed status (an omitted routing_role does not de-manage this
        # hop) and cannot widen the roster (a different policy_id or a role outside the parent's
        # own is a widened mandate, not a narrower/equal one -- refuse rather than silently
        # substitute the parent's role, which would misattribute the sub-task's actual mandate).
        hermes_home = parent_home
        policy_id = parent_policy_id
        if task_policy_id is not None and task_policy_id != parent_policy_id:
            raise DelegationRoutingBlocked(
                "unsupported_executor",
                f"nested delegation under a managed parent (policy={parent_policy_id!r}) "
                f"requested a different policy_id={task_policy_id!r}; a managed child cannot "
                "widen its authority to a different roster",
            )
        if task_role is not None and task_role != parent_role:
            raise DelegationRoutingBlocked(
                "unsupported_executor",
                f"nested delegation under a managed parent (role={parent_role!r}) requested "
                f"role={task_role!r}; a managed child cannot widen its role beyond the parent's "
                "own authorized role",
            )
        role = parent_role
        # Inherited authority always wins: a managed parent cannot demote its
        # child by requesting observational shadow mode or omitting the mode.
        mode = "enforced"

        parent_requirements = parent_decision["requirements"]
        task_requirements = dict(task_requirements or {})
        task_requirements["required_capabilities"] = sorted(set(
            task_requirements.get("required_capabilities", [])
        ) | set(parent_requirements.get("required_capabilities", [])))
        if parent_requirements["quality"] == "deep":
            task_requirements["task_class"] = parent_requirements["task_class"]
        parent_provenance = parent_requirements.get("provenance")
        if parent_provenance is None:
            raise DelegationRoutingBlocked("provenance_incomplete", "parent receipt predates nested authority requirements; start a new attempt")
        if "provenance" in task_requirements and task_requirements["provenance"] != parent_provenance:
            raise DelegationRoutingBlocked("unsupported_executor", "nested delegation cannot replace parent provenance")
        task_requirements["provenance"] = parent_provenance

    try:
        requirements = _build_requirements(
            task, task_requirements, role=role, task_index=task_index, attempt_id=attempt_id,
        )
        if parent_decision is not None:
            parent_requirements = parent_decision["requirements"]
            requirements["allowed_route_ids"] = [parent_decision["selected"]["route_id"], *parent_decision["alternates"]]
            requirements["inherited_quality"] = parent_requirements["quality"]
            classification = parent_requirements.get("classification") or {}
            requirements["inherited_risk_flags"] = sorted(
                set(parent_requirements.get("risk_flags", []))
                | set(parent_requirements.get("inherited_risk_flags", []))
                | set(classification.get("risk_flags", []))
            )
        decision_kwargs = resolve_route(
            hermes_home, policy_id, requirements, now=int(time.time()),
            policy_revision=parent_decision["policy_revision"] if parent_decision else None,
        )
        if mode == "shadow":
            from agent.model_selection_store import append_outcome, get_receipt

            shadow_decision = get_receipt(hermes_home, decision_kwargs["receipt_id"])
            append_outcome(hermes_home, decision_kwargs["receipt_id"], "routing_shadow", {
                "execution_kind": "delegation", "execution_id": requirements["execution_id"],
                "recommended_route_id": shadow_decision["selected"]["route_id"] if shadow_decision else None,
            })
            return {
                "routing_mode": "shadow", "routing_home": hermes_home,
                "receipt_id": decision_kwargs["receipt_id"],
            }
    except Exception as exc:
        if mode != "shadow":
            raise
        return {
            "routing_mode": "shadow", "routing_home": hermes_home,
            "shadow_error": exc.reason if isinstance(exc, RoutingBlocked) else "observation_failed",
        }
    if task.get("model") and task["model"] != decision_kwargs["model"]:
        raise DelegationRoutingBlocked("unsupported_executor", "model preference is outside the selected managed route")
    decision_kwargs["routing_mode"] = "enforced"
    decision_kwargs["routing_home"] = hermes_home
    return decision_kwargs


def _build_requirements(
    task: dict, intake: Optional[dict], *, role: str, task_index: int, attempt_id: str,
) -> dict:
    """The real `TaskRequirements` shape for a delegation task, mirroring
    `hermes_cli.kanban_model_routing._build_requirements` (design §3.B): genuine per-task intake
    (`task["routing_requirements"]`) or the honest, non-fabricated conservative defaults -- never a
    fixed placeholder shape shared across every task in the batch."""
    intake = intake or {}
    requirements = {
        "schema_version": 1,
        "role": role,
        "execution_kind": "delegation",
        "execution_id": task.get("_delegation_id") or f"task-{task_index}",
        "attempt_id": attempt_id,
        "slot_id": "",
        "task_class": intake.get("task_class", ""),
        "risk_flags": list(intake.get("risk_flags", [])),
        "required_capabilities": list(intake.get("required_capabilities", [])),
        "input_tokens": int(intake.get("input_tokens", 0)),
        "reserve_tokens": int(intake.get("reserve_tokens", 0)),
        "reasoning": task.get("reasoning_effort") or "medium",
    }
    provenance = intake.get("provenance")
    from hermes_cli.kanban_model_routing import _is_review_role

    if _is_review_role(role):
        if provenance is None:
            raise DelegationRoutingBlocked(
                "provenance_incomplete",
                f"delegation task {task_index}: role {role!r} is an independent-review role and "
                "requires genuine contributor-maker provenance (task['routing_requirements']"
                "['provenance']); none was supplied -- refusing to manufacture completeness",
            )
        requirements["provenance"] = provenance
    elif provenance is not None:
        requirements["provenance"] = provenance
    else:
        requirements["provenance"] = {
            "frozen_sha": "", "verified_by": "delegation-adapter", "complete": True, "contributors": [],
        }
    return requirements


def _current_hermes_home():
    from hermes_constants import get_hermes_home

    return get_hermes_home()


def build_lifecycle_child(request, parent):
    """Public host lifecycle uses the same routed construction as the tool."""
    from tools import delegate_tool as dt

    cfg = dt._load_config()
    creds = dt._resolve_delegation_credentials(cfg, parent)
    task = {"goal": request.goal, "context": request.context, "model": request.model,
            "routing_role": request.routing_role, "routing_mode": request.routing_mode,
            "routing_policy_id": request.routing_policy_id,
            "routing_requirements": request.routing_requirements}
    children, error = dt._build_children(
        [task], [None], creds, top_role=request.role,
        max_iterations=cfg.get("max_iterations", dt.DEFAULT_MAX_ITERATIONS),
        parent_agent=parent, routing_cfg=cfg, live_deleg_id=None, live_writers=[],
        allowed_toolsets=list(request.allowed_toolsets) if request.allowed_toolsets else None,
    )
    if error:
        raise DelegationRoutingBlocked("unsupported_executor", error)
    return children[0][2]


def stamp_managed_route(child, resolution: dict) -> None:
    """Wire the same two attributes `agent.managed_route_guard.enforce_managed_route_per_request`
    reads on EVERY subsequent request in the child's own turn (§12: re-checked before each request,
    not only at construction) -- the exact seam Kanban workers use, so a revoked policy or a
    fallback/rotation that silently swapped the child's actual client mid-turn is caught by the
    SAME shared guard, not a delegation-specific reimplementation."""
    child._managed_routing_receipt_id = resolution["receipt_id"]
    child._managed_routing_home = resolution["routing_home"]


def stamp_shadow_route(child, resolution: dict) -> None:
    """Expose observation provenance without granting per-request authority."""
    child._shadow_routing_receipt_id = resolution.get("receipt_id")
    child._shadow_routing_error = resolution.get("shadow_error")
