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

__all__ = ["resolve_delegation_route", "stamp_managed_route", "DelegationRoutingBlocked"]

DEFAULT_DELEGATION_POLICY_ID = "kanban-default"


class DelegationRoutingBlocked(RoutingBlocked):
    """Re-raised with the delegation call site's context; same reason codes as RoutingBlocked."""


def _parent_managed_context(parent_agent) -> tuple[Optional[str], Optional[str], Optional[str]]:
    """``(hermes_home, policy_id, role)`` the parent's OWN receipt was authorized under, or
    ``(None, None, None)`` when the parent carries no managed authority to inherit. Read from the
    parent's already-persisted receipt (never re-derived/guessed) so the ceiling this function
    enforces is the ceiling the parent itself was actually given."""
    receipt_id = getattr(parent_agent, "_managed_routing_receipt_id", None)
    hermes_home = getattr(parent_agent, "_managed_routing_home", None)
    # Real values are always a non-empty str (receipt id) / str-or-Path (home) -- a test double
    # (MagicMock etc.) that auto-vivifies unset attributes returns a Mock object here, not None;
    # require the actual expected types so an unconfigured mock parent is correctly "unmanaged"
    # rather than accidentally treated as carrying inherited authority.
    if not isinstance(receipt_id, str) or not receipt_id or hermes_home is None:
        return None, None, None
    if not isinstance(hermes_home, (str,)) and not hasattr(hermes_home, "__fspath__"):
        return None, None, None
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
    return hermes_home, decision["policy_id"], decision["requirements"]["role"]


def _task_intake(task: dict) -> tuple[Optional[str], Optional[dict], Optional[str]]:
    """``(routing_role, routing_requirements, routing_policy_id)`` as supplied on the task dict --
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
    return role, requirements, policy_id


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
    task_role, task_requirements, task_policy_id = _task_intake(task)
    parent_home, parent_policy_id, parent_role = _parent_managed_context(parent_agent)

    if parent_policy_id is None:
        # Parent carries no managed authority: this task is managed ONLY if it explicitly asked.
        if task_role is None:
            return None
        hermes_home = _current_hermes_home()
        policy_id = task_policy_id or DEFAULT_DELEGATION_POLICY_ID
        role = task_role
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

    requirements = _build_requirements(
        task, task_requirements, role=role, task_index=task_index, attempt_id=attempt_id,
    )
    decision_kwargs = resolve_route(hermes_home, policy_id, requirements, now=int(time.time()))
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


def stamp_managed_route(child, resolution: dict) -> None:
    """Wire the same two attributes `agent.managed_route_guard.enforce_managed_route_per_request`
    reads on EVERY subsequent request in the child's own turn (§12: re-checked before each request,
    not only at construction) -- the exact seam Kanban workers use, so a revoked policy or a
    fallback/rotation that silently swapped the child's actual client mid-turn is caught by the
    SAME shared guard, not a delegation-specific reimplementation."""
    child._managed_routing_receipt_id = resolution["receipt_id"]
    child._managed_routing_home = resolution["routing_home"]
