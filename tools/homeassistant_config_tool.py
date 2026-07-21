"""Native Home Assistant configuration inspection and approved mutation tools."""

from __future__ import annotations

import json
import logging
import os
import uuid
from typing import Any

from hermes_cli.config import cfg_get, load_config_readonly
from tools.approval import request_tool_approval
from tools.homeassistant_client import HomeAssistantClient
from tools.homeassistant_config import HomeAssistantResources
from tools.homeassistant_store import (
    HomeAssistantChangeStore,
    canonical_fingerprint,
    structured_diff,
)
from tools.homeassistant_tool import _run_async
from tools.registry import registry, tool_error

logger = logging.getLogger(__name__)

_SECRET_MARKERS = ("token", "secret", "password", "api_key", "access_key", "credential")


def _redact(value: Any, key: str = "") -> Any:
    if any(marker in key.lower() for marker in _SECRET_MARKERS):
        return "[REDACTED]"
    if isinstance(value, dict):
        return {item_key: _redact(item, item_key) for item_key, item in value.items()}
    if isinstance(value, list):
        return [_redact(item) for item in value]
    return value


def _json_result(value: Any) -> str:
    return json.dumps({"result": _redact(value)}, default=str)


def _config_management() -> tuple[bool, int, int]:
    config = load_config_readonly()
    enabled = bool(
        cfg_get(config, "homeassistant", "config_management", "enabled", default=False)
    )
    ttl = int(
        cfg_get(
            config,
            "homeassistant",
            "config_management",
            "proposal_ttl_seconds",
            default=900,
        )
    )
    history_limit = int(
        cfg_get(
            config,
            "homeassistant",
            "config_management",
            "history_limit",
            default=500,
        )
    )
    return enabled, max(1, ttl), max(1, history_limit)


def _check_available() -> bool:
    enabled, _, _ = _config_management()
    return enabled and bool(os.getenv("HASS_TOKEN"))


def _get_runtime():
    _, ttl, history_limit = _config_management()
    client = HomeAssistantClient(
        os.getenv("HASS_URL", "http://homeassistant.local:8123"),
        os.getenv("HASS_TOKEN", ""),
    )
    return HomeAssistantResources(client), HomeAssistantChangeStore(
        history_limit=history_limit
    ), ttl


def _handle_inspect(args: dict, **kw) -> str:
    del kw
    try:
        action = args.get("action")
        manager, store, _ = _get_runtime()
        if action == "capabilities":
            return _json_result(_run_async(lambda: manager.capabilities()))
        if action == "list":
            resource_type = args.get("resource_type", "")
            return _json_result(_run_async(lambda: manager.list(resource_type)))
        if action == "get":
            resource_type = args.get("resource_type", "")
            resource_id = args.get("resource_id", "")
            return _json_result(_run_async(lambda: manager.get(resource_type, resource_id)))
        if action == "history":
            reconciled = _run_async(lambda: _reconcile_unfinished(manager, store))
            return _json_result(
                {
                    "changes": store.list_history(args.get("limit")),
                    "reconciliation": reconciled,
                }
            )
        return tool_error("action must be capabilities, list, get, or history")
    except Exception as exc:
        logger.error("ha_inspect_config error: %s", exc)
        return tool_error(f"Home Assistant configuration inspection failed: {exc}")


def _strict_approval(message: str, operation: str) -> dict[str, Any]:
    return request_tool_approval(
        "ha_manage_config",
        message,
        rule_key=f"homeassistant-config:{operation}:{uuid.uuid4().hex}",
        allow_yolo=False,
        allow_headless=False,
        allow_permanent=False,
    )


def _resource_id_from_result(
    resource_type: str, result: Any, fallback: str
) -> str:
    if not isinstance(result, dict):
        return fallback
    keys = {
        "group": ("entry_id",),
        "area": ("area_id", "id"),
        "entity": ("entity_id", "id"),
        "device": ("device_id", "id"),
    }.get(resource_type, ("id",))
    return next((result[key] for key in keys if result.get(key)), fallback)


def _desired_matches_current(desired: Any, current: Any) -> bool:
    """Check desired fields as a subset of a richer Home Assistant response."""
    if isinstance(desired, dict) and isinstance(current, dict):
        return all(
            key in current and _desired_matches_current(value, current[key])
            for key, value in desired.items()
        )
    return desired == current


async def _reconcile_unfinished(manager, store) -> list[dict[str, str]]:
    outcomes = []
    for change in store.list_unfinished():
        try:
            current = await manager.get(change["resource_type"], change["resource_id"])
        except Exception:
            outcomes.append({"change_id": change["id"], "status": "unreachable"})
            continue
        if change["status"] in {"applying", "apply_uncertain"}:
            if change["operation"] == "create" and not change["authoritative_id"]:
                store.mark_apply_uncertain(change["id"])
                outcomes.append({"change_id": change["id"], "status": "manual_review"})
                continue
            if _desired_matches_current(change["after"], current):
                store.finalize_applied(change["id"], after=current)
                status = "applied"
            elif canonical_fingerprint(current) == canonical_fingerprint(change["before"]):
                store.mark_apply_not_applied(change["id"])
                status = "not_applied"
            else:
                store.mark_apply_uncertain(change["id"])
                status = "manual_review"
        else:
            if canonical_fingerprint(current) == canonical_fingerprint(change["before"]):
                store.record_rolled_back(change["id"])
                status = "rolled_back"
            elif canonical_fingerprint(current) == change["after_fingerprint"]:
                store.mark_rollback_failed(change["id"])
                status = "not_rolled_back"
            else:
                store.mark_rollback_uncertain(change["id"])
                status = "manual_review"
        outcomes.append({"change_id": change["id"], "status": status})
    return outcomes


def _handle_preview(args: dict, manager, store, ttl: int) -> str:
    resource_type = args.get("resource_type", "")
    resource_id = args.get("resource_id", "")
    operation = args.get("operation", "")
    definition = args.get("definition")
    if operation not in {"create", "update"}:
        return tool_error("operation must be create or update")
    if not resource_type or not resource_id or not isinstance(definition, dict):
        return tool_error("resource_type, resource_id, and object definition are required")
    before = _run_async(lambda: manager.get(resource_type, resource_id))
    if operation == "create" and before is not None:
        return tool_error("resource already exists; preview an update instead")
    if operation == "update" and before is None:
        return tool_error("resource does not exist; preview a create instead")
    proposal = store.create_proposal(
        resource_type=resource_type,
        resource_id=resource_id,
        operation=operation,
        before=before,
        desired=definition,
        ttl_seconds=ttl,
    )
    return _json_result(
        {
            "proposal_id": proposal["id"],
            "status": proposal["status"],
            "expires_at": proposal["expires_at"],
            "resource_type": resource_type,
            "resource_id": resource_id,
            "operation": operation,
            "diff": structured_diff(before, definition),
        }
    )


def _handle_apply(args: dict, manager, store) -> str:
    proposal_id = args.get("proposal_id", "")
    proposal = store.get_proposal(proposal_id)
    if proposal is None:
        return tool_error("proposal not found")
    message = (
        f"Apply Home Assistant {proposal['operation']} for "
        f"{proposal['resource_type']} {proposal['resource_id']}? "
        f"Changes: {json.dumps(_redact(structured_diff(proposal['before'], proposal['desired'])))}"
    )
    approval = _strict_approval(message, "apply")
    if not approval.get("approved"):
        return tool_error(
            str(approval.get("message") or "Home Assistant configuration change was not approved")
        )
    current = _run_async(
        lambda: manager.get(proposal["resource_type"], proposal["resource_id"])
    )
    attempt = store.claim_and_begin_apply(
        proposal_id,
        canonical_fingerprint(current),
        created_by_hermes=proposal["operation"] == "create",
        resource_id=proposal["resource_id"],
    )
    try:
        applied_result = _run_async(
            lambda: manager.apply(
                proposal["resource_type"],
                proposal["resource_id"],
                proposal["operation"],
                proposal["desired"],
            )
        )
    except Exception:
        store.mark_apply_uncertain(attempt["id"])
        raise
    actual_resource_id = _resource_id_from_result(
        proposal["resource_type"], applied_result, proposal["resource_id"]
    )
    if proposal["operation"] == "create":
        store.identify_created_resource(attempt["id"], actual_resource_id)
    after = _run_async(
        lambda: manager.get(proposal["resource_type"], actual_resource_id)
    )
    if after is None:
        after = applied_result
    change = store.finalize_applied(
        attempt["id"],
        after=after,
        resource_id=actual_resource_id,
    )
    return _json_result({"status": "applied", "change_id": change["id"], "after": after})


def _handle_rollback(args: dict, manager, store) -> str:
    change_id = args.get("change_id", "")
    change = store.get_change(change_id)
    if change is None:
        return tool_error("change not found")
    message = (
        f"Rollback Home Assistant change {change_id} for "
        f"{change['resource_type']} {change['resource_id']}?"
    )
    approval = _strict_approval(message, "rollback")
    if not approval.get("approved"):
        return tool_error(str(approval.get("message") or "Home Assistant rollback was not approved"))
    current = _run_async(
        lambda: manager.get(change["resource_type"], change["resource_id"])
    )
    store.claim_rollback(change_id, canonical_fingerprint(current))
    try:
        result = _run_async(lambda: manager.rollback(change))
    except Exception:
        store.mark_rollback_uncertain(change_id)
        raise
    rolled_back = store.record_rolled_back(change_id)
    return _json_result(
        {"status": rolled_back["status"], "change_id": change_id, "result": result}
    )


def _handle_manage(args: dict, **kw) -> str:
    del kw
    try:
        manager, store, ttl = _get_runtime()
        action = args.get("action")
        if action == "preview":
            return _handle_preview(args, manager, store, ttl)
        if action == "apply":
            return _handle_apply(args, manager, store)
        if action == "rollback":
            return _handle_rollback(args, manager, store)
        return tool_error("action must be preview, apply, or rollback")
    except Exception as exc:
        logger.error("ha_manage_config error: %s", exc)
        return tool_error(f"Home Assistant configuration change failed: {exc}")


HA_INSPECT_CONFIG_SCHEMA = {
    "name": "ha_inspect_config",
    "description": "Inspect supported Home Assistant configuration and Hermes change history.",
    "parameters": {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["capabilities", "list", "get", "history"]},
            "resource_type": {"type": "string"},
            "resource_id": {"type": "string"},
            "limit": {"type": "integer", "minimum": 1, "maximum": 500},
        },
        "required": ["action"],
    },
}

HA_MANAGE_CONFIG_SCHEMA = {
    "name": "ha_manage_config",
    "description": (
        "Preview, explicitly approve and apply, or rollback one Home Assistant configuration object. "
        "Always preview before apply."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["preview", "apply", "rollback"]},
            "resource_type": {"type": "string"},
            "resource_id": {"type": "string"},
            "operation": {"type": "string", "enum": ["create", "update"]},
            "definition": {"type": "object", "additionalProperties": True},
            "proposal_id": {"type": "string"},
            "change_id": {"type": "string"},
        },
        "required": ["action"],
    },
}

registry.register(
    name="ha_inspect_config", toolset="homeassistant", schema=HA_INSPECT_CONFIG_SCHEMA,
    handler=_handle_inspect, check_fn=_check_available, emoji="🏠",
)
registry.register(
    name="ha_manage_config", toolset="homeassistant", schema=HA_MANAGE_CONFIG_SCHEMA,
    handler=_handle_manage, check_fn=_check_available, emoji="🏠",
)
