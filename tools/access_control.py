"""Runtime scope enforcement for connected-account tools.

This module adds a lightweight, config-driven access layer on top of the
existing toolset enable/disable system. Tool authors can opt in by providing an
``access_fn`` when registering a tool. The function returns a dict describing
what external service/account the tool touches and whether the action is a
read or write operation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

_READ_ONLY = "read-only"
_FULL = "full"
_NONE = "none"

_READ_VERBS = (
    "get",
    "list",
    "read",
    "search",
    "find",
    "fetch",
    "query",
    "describe",
    "inspect",
    "view",
    "show",
    "preview",
)

_WRITE_VERBS = (
    "create",
    "update",
    "delete",
    "remove",
    "set",
    "send",
    "write",
    "edit",
    "patch",
    "post",
    "put",
    "call",
    "invoke",
    "run",
    "execute",
    "close",
    "open",
    "archive",
    "approve",
    "reject",
    "resume",
    "pause",
    "trigger",
)


@dataclass(frozen=True)
class AccessDecision:
    allowed: bool
    scope: str
    platform: str
    operation: str
    service: str
    account: str
    reason: str


def build_static_access_fn(service: str, operation: str, *, account: Optional[str] = None) -> Callable[..., dict[str, str]]:
    """Build a simple access context function for a fixed service/action."""

    def _access_fn(args: Optional[dict[str, Any]] = None, **_: Any) -> dict[str, str]:
        return {
            "service": service,
            "account": account or service,
            "operation": operation,
        }

    return _access_fn


def infer_mcp_operation(tool_name: str) -> str:
    """Best-effort read/write inference for MCP tool names.

    MCP servers can expose arbitrary tools, so fail closed: ambiguous verbs are
    treated as writes unless they clearly look read-only.
    """
    normalized = tool_name.lower().replace("-", "_").replace(".", "_")
    parts = [part for part in normalized.split("_") if part and part != "mcp"]

    for part in parts:
        if part in _WRITE_VERBS:
            return "write"
    for part in parts:
        if part in _READ_VERBS:
            return "read"
    return "write"


def evaluate_access(
    tool_name: str,
    access_context: Optional[dict[str, Any]],
    *,
    platform: Optional[str] = None,
    config: Optional[dict[str, Any]] = None,
) -> AccessDecision:
    """Return whether a tool call is allowed for the given platform/account."""
    platform_key = _normalize_platform(platform)
    service = str((access_context or {}).get("service") or "")
    account = str((access_context or {}).get("account") or service or tool_name)
    operation = _normalize_operation((access_context or {}).get("operation"))

    if not access_context or not service:
        return AccessDecision(
            allowed=True,
            scope=_FULL,
            platform=platform_key,
            operation=operation,
            service=service,
            account=account,
            reason="No scoped access policy applies to this tool.",
        )

    settings = _get_access_control_config(config)
    if not settings.get("enabled", True):
        return AccessDecision(
            allowed=True,
            scope=_FULL,
            platform=platform_key,
            operation=operation,
            service=service,
            account=account,
            reason="Scoped access control is disabled.",
        )

    scope = _resolve_scope(settings, platform_key, service, account, tool_name)
    allowed = _scope_allows(scope, operation)
    if allowed:
        reason = f"Allowed by scope '{scope}'."
    else:
        reason = f"Scope '{scope}' does not permit '{operation}' operations."

    return AccessDecision(
        allowed=allowed,
        scope=scope,
        platform=platform_key,
        operation=operation,
        service=service,
        account=account,
        reason=reason,
    )


def _get_access_control_config(config: Optional[dict[str, Any]]) -> dict[str, Any]:
    if config is None:
        try:
            from hermes_cli.config import load_config

            config = load_config()
        except Exception:
            config = {}

    return dict(config.get("access_control") or {})


def _normalize_platform(platform: Optional[str]) -> str:
    value = str(platform or "cli").strip().lower()
    if value in {"", "local", "none"}:
        return "cli"
    return value


def _normalize_scope(scope: Any) -> str:
    value = str(scope or _FULL).strip().lower()
    aliases = {
        "allow": _FULL,
        "all": _FULL,
        "full-access": _FULL,
        "read_only": _READ_ONLY,
        "readonly": _READ_ONLY,
        "read": _READ_ONLY,
        "deny": _NONE,
        "disabled": _NONE,
        "off": _NONE,
    }
    return aliases.get(value, value if value in {_FULL, _READ_ONLY, _NONE} else _FULL)


def _normalize_operation(operation: Any) -> str:
    value = str(operation or "write").strip().lower()
    return "read" if value == "read" else "write"


def _scope_allows(scope: str, operation: str) -> bool:
    if scope == _FULL:
        return True
    if scope == _READ_ONLY:
        return operation == "read"
    return False


def _resolve_scope(settings: dict[str, Any], platform: str, service: str, account: str, tool_name: str) -> str:
    scope = _normalize_scope(settings.get("default_scope", _FULL))
    scope = _pick_scope(settings.get("platform_profiles"), platform, scope)

    services = settings.get("services") or {}
    service_cfg = services.get(service) or {}
    scope = _pick_scope(service_cfg, None, scope)
    scope = _pick_scope(service_cfg.get("platform_scopes"), platform, scope)
    scope = _pick_scope(service_cfg.get("tool_scopes"), tool_name, scope)

    accounts = settings.get("accounts") or {}
    account_cfg = accounts.get(account) or {}
    scope = _pick_scope(account_cfg, None, scope)
    scope = _pick_scope(account_cfg.get("platform_scopes"), platform, scope)
    scope = _pick_scope(account_cfg.get("tool_scopes"), tool_name, scope)
    return scope


def _pick_scope(source: Any, key: Optional[str], fallback: str) -> str:
    if not source:
        return fallback
    if isinstance(source, str):
        return _normalize_scope(source)
    if isinstance(source, dict):
        if key is not None and key in source:
            return _normalize_scope(source.get(key))
        if key is None and "scope" in source:
            return _normalize_scope(source.get("scope"))
    return fallback
