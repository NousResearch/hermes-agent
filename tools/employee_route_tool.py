"""Tool for listing and managing dynamic employee background routes."""

from __future__ import annotations

import json
import os
from typing import Any

from gateway.config import GatewayConfig, Platform
from gateway.employee_route_store import (
    clear_employee_route_store,
    list_employee_routes,
    set_employee_route,
)
from gateway.employee_routes import get_employee_routes
from tools.group_session_helpers import require_admin_session, session_actor_label
from tools.registry import registry, tool_error, tool_result


_SUPPORTED_PLATFORMS = (Platform.QQ_NAPCAT, Platform.WEIXIN)
_ACTION_ALIASES = {
    "list": "list_routes",
    "set": "set_route",
    "update": "set_route",
    "clear": "clear_route",
    "remove": "clear_route",
    "delete": "clear_route",
}


EMPLOYEE_ROUTE_CONTROL_PROPERTIES = {
    "worker_name": {
        "type": "string",
        "description": "Worker name to inspect, update, or remove, such as 铁柱 or 阿旺.",
    },
    "aliases": {
        "type": "array",
        "items": {"type": "string"},
        "description": "Optional alternate names for explicit routing, such as 老铁 or 旺财.",
    },
    "preloaded_skills": {
        "type": "array",
        "items": {"type": "string"},
        "description": "Skills to preload when this worker is dispatched.",
    },
    "match_modes": {
        "type": "array",
        "items": {"type": "string", "enum": ["explicit", "heuristic"]},
        "description": "Routing modes. Use explicit for named assignment only, heuristic for automatic matching.",
    },
    "action_terms": {
        "type": "array",
        "items": {"type": "string"},
        "description": "Action hint terms like 打磨, 优化, 排查.",
    },
    "subject_terms": {
        "type": "array",
        "items": {"type": "string"},
        "description": "Subject hint terms like 页面, 主页, 服务器.",
    },
    "pain_terms": {
        "type": "array",
        "items": {"type": "string"},
        "description": "Pain hint terms like 粗糙, 炸了, 异常.",
    },
    "enabled": {
        "type": "boolean",
        "description": "Whether the dynamic route stays enabled. Defaults to true for set_route.",
    },
    "updated_by": {
        "type": "string",
        "description": "Optional operator label override for auditing. Defaults to the current session actor.",
    },
}


EMPLOYEE_ROUTE_TOOL_SCHEMA = {
    "name": "employee_route_control",
    "description": (
        "Inspect or manage dynamic employee background routes for QQ NapCat and Weixin. "
        "Use this when assigning a worker like 铁柱/阿旺 to a class of tasks, tuning heuristic routing hints, "
        "or removing a previously configured dynamic route. Mutating actions are admin-only."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": [
                    "list_routes",
                    "set_route",
                    "clear_route",
                    "list",
                    "set",
                    "update",
                    "clear",
                    "remove",
                    "delete",
                ],
                "description": "Employee-route operation to perform.",
            },
            "platform": {
                "type": "string",
                "enum": ["qq_napcat", "weixin"],
                "description": "Target platform. Defaults to the current QQ/Weixin session platform when available.",
            },
            "worker_name": {
                "type": "string",
                "description": "Worker name to inspect/update/remove, such as 铁柱 or 阿旺.",
            },
            "aliases": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional alternate names for explicit routing, such as 老铁 or 旺财.",
            },
            "preloaded_skills": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Skills to preload when this worker is dispatched.",
            },
            "match_modes": {
                "type": "array",
                "items": {"type": "string", "enum": ["explicit", "heuristic"]},
                "description": "Routing modes. Use explicit for named assignment only, heuristic to auto-pick matching tasks.",
            },
            "action_terms": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Action hint terms like 打磨, 优化, 排查.",
            },
            "subject_terms": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Subject hint terms like 页面, 主页, 服务器.",
            },
            "pain_terms": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Pain hint terms like 粗糙, 炸了, 异常.",
            },
            "enabled": {
                "type": "boolean",
                "description": "Whether the dynamic route stays enabled. Defaults to true for set_route.",
            },
            "updated_by": {
                "type": "string",
                "description": "Optional operator label override for auditing. Defaults to the current session actor.",
            },
        },
        "required": ["action"],
    },
}


def employee_route_tool(args, **kwargs):
    del kwargs

    normalized_args = dict(args or {})
    action = _normalize_action(normalized_args.get("action"))
    if action not in {"list_routes", "set_route", "clear_route"}:
        return tool_error(
            "Unsupported action. Use 'list_routes', 'set_route', or 'clear_route'.",
            success=False,
        )

    from tools.interrupt import is_interrupted

    if is_interrupted():
        return tool_error("Interrupted", success=False)

    try:
        platform = _resolve_platform(normalized_args.get("platform"))
    except ValueError as exc:
        return tool_error(str(exc), success=False)

    try:
        from gateway.config import load_gateway_config

        config = load_gateway_config()
    except Exception as exc:
        return tool_error(f"Failed to load gateway config: {exc}", success=False)

    if action == "list_routes":
        return tool_result(
            _build_list_payload(
                action=action,
                platform=platform,
                config=config,
            )
        )

    admin_error = require_admin_session("调整员工路由")
    if admin_error:
        return tool_error(admin_error, success=False)

    worker_name = _require_worker_name(normalized_args.get("worker_name"))
    updated_by = str(normalized_args.get("updated_by") or "").strip() or session_actor_label()

    try:
        if action == "set_route":
            route = set_employee_route(
                platform,
                worker_name=worker_name,
                aliases=_coerce_str_list(normalized_args.get("aliases")),
                preloaded_skills=_coerce_str_list(normalized_args.get("preloaded_skills")),
                match_modes=_coerce_str_list(normalized_args.get("match_modes")),
                action_terms=_coerce_str_list(normalized_args.get("action_terms")),
                subject_terms=_coerce_str_list(normalized_args.get("subject_terms")),
                pain_terms=_coerce_str_list(normalized_args.get("pain_terms")),
                enabled=_coerce_bool(normalized_args.get("enabled"), default=True),
                updated_by=updated_by,
            )
            payload = _build_list_payload(
                action=action,
                platform=platform,
                config=config,
            )
            payload["route"] = _json_safe_route(route)
            return tool_result(payload)

        cleared_route = clear_employee_route_store(
            platform,
            worker_name,
            updated_by=updated_by,
        )
        payload = _build_list_payload(
            action=action,
            platform=platform,
            config=config,
        )
        payload["cleared_route"] = _json_safe_route(cleared_route) if cleared_route else None
        return tool_result(payload)
    except ValueError as exc:
        return tool_error(str(exc), success=False)
    except Exception as exc:
        return tool_error(f"Employee route action failed: {exc}", success=False)


def _normalize_action(value: Any) -> str:
    action = str(value or "").strip().lower()
    return _ACTION_ALIASES.get(action, action)


def _resolve_platform(value: Any) -> Platform:
    text = str(value or "").strip().lower()
    if text:
        try:
            platform = Platform(text)
        except ValueError as exc:
            raise ValueError("Unsupported platform. Use 'qq_napcat' or 'weixin'.") from exc
        if platform not in _SUPPORTED_PLATFORMS:
            raise ValueError("Unsupported platform. Use 'qq_napcat' or 'weixin'.")
        return platform

    session_platform = str(os.getenv("HERMES_SESSION_PLATFORM") or "").strip().lower()
    if session_platform in {platform.value for platform in _SUPPORTED_PLATFORMS}:
        return Platform(session_platform)
    raise ValueError("No platform specified. Use platform='qq_napcat' or platform='weixin'.")


def _require_worker_name(value: Any) -> str:
    worker_name = str(value or "").strip()
    if not worker_name:
        raise ValueError("'worker_name' is required.")
    return worker_name


def _coerce_str_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        value = [value]
    elif isinstance(value, (tuple, set)):
        value = list(value)
    elif not isinstance(value, list):
        return []
    normalized: list[str] = []
    for item in value:
        text = str(item or "").strip()
        if text:
            normalized.append(text)
    return normalized


def _coerce_bool(value: Any, *, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes", "on"}:
            return True
        if lowered in {"false", "0", "no", "off"}:
            return False
        return default
    return bool(value)


def _json_safe_route(route: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(route, dict):
        return None
    return {
        "worker_name": str(route.get("worker_name") or "").strip(),
        "aliases": list(route.get("aliases") or []),
        "preloaded_skills": list(route.get("preloaded_skills") or []),
        "match_modes": list(route.get("match_modes") or []),
        "action_terms": list(route.get("action_terms") or []),
        "subject_terms": list(route.get("subject_terms") or []),
        "pain_terms": list(route.get("pain_terms") or []),
        "updated_at": route.get("updated_at"),
        "updated_by": route.get("updated_by"),
        "enabled": bool(route.get("enabled", True)),
    }


def _json_safe_effective_routes(routes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for route in routes:
        if not isinstance(route, dict):
            continue
        result.append(
            {
                "worker_name": str(route.get("worker_name") or "").strip(),
                "aliases": list(route.get("aliases") or []),
                "preloaded_skills": list(route.get("preloaded_skills") or []),
                "match_modes": list(route.get("match_modes") or []),
                "action_terms": list(route.get("action_terms") or []),
                "subject_terms": list(route.get("subject_terms") or []),
                "pain_terms": list(route.get("pain_terms") or []),
            }
        )
    return result


def _build_list_payload(
    *,
    action: str,
    platform: Platform,
    config: GatewayConfig | None,
) -> dict[str, Any]:
    dynamic_routes = list_employee_routes(platform)
    effective_routes = get_employee_routes(config, platform=platform)
    return {
        "success": True,
        "action": action,
        "platform": platform.value,
        "routes": [_json_safe_route(route) for route in dynamic_routes],
        "effective_routes": _json_safe_effective_routes(effective_routes),
    }


registry.register(
    name="employee_route_control",
    toolset="messaging",
    schema=EMPLOYEE_ROUTE_TOOL_SCHEMA,
    handler=employee_route_tool,
    emoji="🧭",
)
