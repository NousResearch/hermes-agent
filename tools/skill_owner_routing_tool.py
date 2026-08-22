"""Owner-aware registration adapter for ``skill_manage``.

The core skill manager is intentionally kept below Hermes' 2k production-file
limit. This module owns only the cross-profile create admission seam: it runs the
existing write-approval gate in the caller's profile, then (when explicitly
enabled by Default policy) executes the complete existing ``skill_manage``
transaction under the declared owner profile. Non-create actions are delegated
unchanged.

The adapter re-registers the same ``skill_manage`` tool after
``skill_manager_tool`` in built-in discovery order, so there is no second user-
visible tool and no duplicate action implementation.
"""

from __future__ import annotations

import copy
import json
from typing import Any

from tools import skill_manager_tool as _manager
from tools.registry import registry
from tools.skill_owner_routing import (
    create_skill_with_owner_routing,
    owner_routing_policy,
    reset_owner_routing_scope,
)


def _call_base(
    *,
    action: str,
    name: str,
    content: str | None,
    category: str | None,
    file_path: str | None,
    file_content: str | None,
    old_string: str | None,
    new_string: str | None,
    replace_all: bool,
    absorbed_into: str | None,
    task_id: str | None,
    session_id: str | None,
) -> str:
    """Delegate to the existing manager without changing its semantics."""
    return _manager.skill_manage(
        action=action,
        name=name,
        content=content,
        category=category,
        file_path=file_path,
        file_content=file_content,
        old_string=old_string,
        new_string=new_string,
        replace_all=replace_all,
        absorbed_into=absorbed_into,
        task_id=task_id,
        session_id=session_id,
    )


def _call_base_after_gate(
    *,
    name: str,
    content: str,
    category: str | None,
    task_id: str | None,
    session_id: str | None,
) -> dict[str, Any]:
    """Run the full base create transaction after caller-scope admission.

    The adapter has already run ``_apply_skill_write_gate`` in the caller's
    profile. Set the manager's existing replay bypass while delegating so the
    same create is not staged a second time after HERMES_HOME switches to the
    owner. The base function still owns validation, filesystem mutation, audit
    ledger, cache invalidation, usage/provenance telemetry, and sync.
    """
    bypass = _manager._skill_gate_bypass.set(True)
    try:
        raw = _call_base(
            action="create",
            name=name,
            content=content,
            category=category,
            file_path=None,
            file_content=None,
            old_string=None,
            new_string=None,
            replace_all=False,
            absorbed_into=None,
            task_id=task_id,
            session_id=session_id,
        )
    finally:
        _manager._skill_gate_bypass.reset(bypass)

    try:
        result = json.loads(raw)
    except (TypeError, ValueError):
        return {
            "success": False,
            "error": "skill_manage returned a malformed result during owner-routed create",
        }
    if not isinstance(result, dict):
        return {
            "success": False,
            "error": "skill_manage returned a non-object result during owner-routed create",
        }
    return result


def skill_manage(
    action: str,
    name: str,
    content: str = None,
    category: str = None,
    file_path: str = None,
    file_content: str = None,
    old_string: str = None,
    new_string: str = None,
    replace_all: bool = False,
    absorbed_into: str = None,
    task_id: str = None,
    session_id: str = None,
) -> str:
    """Route only newly-created skills; preserve every other manager action."""
    if action != "create":
        return _call_base(
            action=action,
            name=name,
            content=content,
            category=category,
            file_path=file_path,
            file_content=file_content,
            old_string=old_string,
            new_string=new_string,
            replace_all=replace_all,
            absorbed_into=absorbed_into,
            task_id=task_id,
            session_id=session_id,
        )

    policy = owner_routing_policy()
    if not policy.get("enabled"):
        return _call_base(
            action=action,
            name=name,
            content=content,
            category=category,
            file_path=file_path,
            file_content=file_content,
            old_string=old_string,
            new_string=new_string,
            replace_all=replace_all,
            absorbed_into=absorbed_into,
            task_id=task_id,
            session_id=session_id,
        )

    # Keep write approval in the CALLER's scope. Switching to the owner before
    # this check could let a Default-routed write inherit a specialist's weaker
    # approval policy. Approved staged writes already have the manager bypass
    # set, so this becomes a no-op during replay.
    gate_result = _manager._apply_skill_write_gate(
        action,
        name,
        content=content,
        category=category,
        file_path=file_path,
        file_content=file_content,
        old_string=old_string,
        new_string=new_string,
        replace_all=replace_all,
        absorbed_into=absorbed_into,
    )
    if gate_result is not None:
        return gate_result

    # Preserve the base manager's exact required-content error instead of
    # turning a malformed create into an owner-metadata error.
    if not content:
        bypass = _manager._skill_gate_bypass.set(True)
        try:
            return _call_base(
                action=action,
                name=name,
                content=content,
                category=category,
                file_path=file_path,
                file_content=file_content,
                old_string=old_string,
                new_string=new_string,
                replace_all=replace_all,
                absorbed_into=absorbed_into,
                task_id=task_id,
                session_id=session_id,
            )
        finally:
            _manager._skill_gate_bypass.reset(bypass)

    result: dict[str, Any]
    owner_scope = None
    try:
        result, owner_scope = create_skill_with_owner_routing(
            name=name,
            content=content,
            category=category,
            policy=policy,
            create_fn=lambda create_name, create_content, create_category: _call_base_after_gate(
                name=create_name,
                content=create_content,
                category=create_category,
                task_id=task_id,
                session_id=session_id,
            ),
        )
        return json.dumps(result, ensure_ascii=False)
    finally:
        reset_owner_routing_scope(owner_scope)


def apply_skill_pending(payload: dict[str, Any]) -> str:
    """Replay an approved staged skill write through the owner-aware adapter."""
    token = _manager._skill_gate_bypass.set(True)
    try:
        return skill_manage(
            action=payload.get("action", ""),
            name=payload.get("name", ""),
            content=payload.get("content"),
            category=payload.get("category"),
            file_path=payload.get("file_path"),
            file_content=payload.get("file_content"),
            old_string=payload.get("old_string"),
            new_string=payload.get("new_string"),
            replace_all=payload.get("replace_all", False),
            absorbed_into=payload.get("absorbed_into"),
        )
    finally:
        _manager._skill_gate_bypass.reset(token)


SKILL_MANAGE_SCHEMA = copy.deepcopy(_manager.SKILL_MANAGE_SCHEMA)
SKILL_MANAGE_SCHEMA["description"] += (
    "\n\nProfile ownership: when `skills.owner_routing.enabled` is true in the "
    "Default profile, new skills declare `metadata.hermes.owner_profile`. "
    "Default may route creation to that registered owner; a named profile may "
    "create for itself but may not write sideways into another named profile."
)


registry.register(
    name="skill_manage",
    toolset="skills",
    schema=SKILL_MANAGE_SCHEMA,
    handler=lambda args, **kw: skill_manage(
        action=args.get("action", ""),
        name=args.get("name", ""),
        content=args.get("content"),
        category=args.get("category"),
        file_path=args.get("file_path"),
        file_content=args.get("file_content"),
        old_string=args.get("old_string"),
        new_string=args.get("new_string"),
        replace_all=args.get("replace_all", False),
        absorbed_into=args.get("absorbed_into"),
        task_id=kw.get("task_id"),
        session_id=kw.get("session_id"),
    ),
    emoji="📝",
)
