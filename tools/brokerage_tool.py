"""Hermes tool wrappers for the local brokerage FastAPI service."""

from __future__ import annotations

import json
import os
from typing import Any

import httpx

from tools.registry import registry, tool_error, tool_result


DEFAULT_TIMEOUT_SECONDS = 30.0


BROKERAGE_TOOL_DESCRIPTION = (
    "Create and manage deterministic brokerage trade intents through the local brokerage service. "
    "Always create a pending trade intent first. Never submit a trade directly from natural language. "
    "Do not call confirm_trade_intent unless the user has explicitly confirmed the trade."
)


CREATE_TRADE_INTENT_SCHEMA = {
    "name": "create_trade_intent",
    "description": (
        BROKERAGE_TOOL_DESCRIPTION
        + " Use this to turn a requested trade into a pending intent with a confirmation code."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "account_mode": {"type": "string", "enum": ["paper", "live"]},
            "symbol": {"type": "string"},
            "side": {"type": "string", "enum": ["buy", "sell", "BUY", "SELL"]},
            "quantity": {"type": "integer", "minimum": 1},
            "order_type": {"type": "string", "enum": ["market", "limit", "MARKET", "LIMIT"]},
            "limit_price": {"type": "number", "exclusiveMinimum": 0},
            "asset_class": {"type": "string", "default": "stock"},
            "time_in_force": {"type": "string", "default": "DAY"},
            "raw_user_text": {
                "type": "string",
                "description": "Optional original user message for audit logging.",
            },
        },
        "required": ["account_mode", "symbol", "side", "quantity", "order_type"],
    },
}


CONFIRM_TRADE_INTENT_SCHEMA = {
    "name": "confirm_trade_intent",
    "description": (
        BROKERAGE_TOOL_DESCRIPTION
        + " Only call this after the user explicitly provides the confirmation text."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "intent_id": {"type": "string"},
            "confirmation_text": {"type": "string"},
        },
        "required": ["intent_id", "confirmation_text"],
    },
}


CANCEL_TRADE_INTENT_SCHEMA = {
    "name": "cancel_trade_intent",
    "description": "Cancel a pending trade intent before broker submission.",
    "parameters": {
        "type": "object",
        "properties": {
            "intent_id": {"type": "string"},
        },
        "required": ["intent_id"],
    },
}


GET_TRADE_INTENT_STATUS_SCHEMA = {
    "name": "get_trade_intent_status",
    "description": "Fetch the current stored status for a trade intent.",
    "parameters": {
        "type": "object",
        "properties": {
            "intent_id": {"type": "string"},
        },
        "required": ["intent_id"],
    },
}


def _load_brokerage_config() -> dict[str, Any]:
    """Load the brokerage section from Hermes config, returning an empty dict on failure."""
    try:
        from hermes_cli.config import load_config

        config = load_config()
        brokerage = config.get("brokerage", {}) if isinstance(config, dict) else {}
        return brokerage if isinstance(brokerage, dict) else {}
    except Exception:
        return {}


def check_brokerage_requirements() -> bool:
    config = _load_brokerage_config()
    enabled = config.get("enabled")
    # Also accept BROKERAGE_ENABLED env var as a fallback
    if not enabled:
        enabled = os.environ.get("BROKERAGE_ENABLED", "").lower() in ("1", "true", "yes")
    return bool(enabled and (str(config.get("service_url", "")).strip() or os.environ.get("BROKERAGE_SERVICE_URL", "").strip()))


def _build_headers(config: dict[str, Any]) -> dict[str, str]:
    # Token can come from config YAML (brokerage.service_token) or .env (BROKERAGE_SERVICE_TOKEN)
    token = config.get("service_token") or os.environ.get("BROKERAGE_SERVICE_TOKEN", "")
    if token and str(token).strip():
        return {"Authorization": f"Bearer {str(token).strip()}"}
    return {}


def _service_request(method: str, path: str, *, payload: dict[str, Any] | None = None) -> str:
    config = _load_brokerage_config()
    base_url = str(config.get("service_url", "")).rstrip("/") or os.environ.get("BROKERAGE_SERVICE_URL", "http://127.0.0.1:8787").rstrip("/")
    url = f"{base_url}{path}"
    headers = _build_headers(config)

    try:
        with httpx.Client(timeout=DEFAULT_TIMEOUT_SECONDS) as client:
            response = client.request(method, url, json=payload, headers=headers)
            response.raise_for_status()
            return tool_result(response.json())
    except httpx.HTTPStatusError as exc:
        detail = None
        try:
            body = exc.response.json()
            if isinstance(body, dict):
                detail = body.get("detail") or body.get("error")
        except Exception:
            detail = None
        detail = detail or exc.response.text or str(exc)
        return tool_error(f"Brokerage service error ({exc.response.status_code}): {detail}")
    except Exception as exc:
        return tool_error(f"Brokerage service request failed: {exc}")


def create_trade_intent_tool(args: dict[str, Any], **kwargs) -> str:
    payload = {
        "account_mode": args.get("account_mode"),
        "symbol": args.get("symbol"),
        "side": args.get("side"),
        "quantity": args.get("quantity"),
        "order_type": args.get("order_type"),
        "asset_class": args.get("asset_class", "stock"),
    }
    if args.get("limit_price") is not None:
        payload["limit_price"] = args.get("limit_price")
    if args.get("raw_user_text"):
        payload["raw_request_text"] = args.get("raw_user_text")
    return _service_request("POST", "/trade-intents", payload=payload)


def confirm_trade_intent_tool(args: dict[str, Any], **kwargs) -> str:
    intent_id = args.get("intent_id")
    if not intent_id:
        return tool_error("intent_id is required")
    return _service_request(
        "POST",
        f"/trade-intents/{intent_id}/confirm",
        payload={"confirmation_text": args.get("confirmation_text")},
    )


def cancel_trade_intent_tool(args: dict[str, Any], **kwargs) -> str:
    intent_id = args.get("intent_id")
    if not intent_id:
        return tool_error("intent_id is required")
    return _service_request("POST", f"/trade-intents/{intent_id}/cancel")


def get_trade_intent_status_tool(args: dict[str, Any], **kwargs) -> str:
    intent_id = args.get("intent_id")
    if not intent_id:
        return tool_error("intent_id is required")
    return _service_request("GET", f"/trade-intents/{intent_id}")


BROKERAGE_HEALTH_SCHEMA = {
    "name": "brokerage_health",
    "description": "Check if the brokerage service is running and the broker connection is healthy. Returns service status and IBKR connection state.",
    "parameters": {
        "type": "object",
        "properties": {},
    },
}


def brokerage_health_tool(args: dict[str, Any], **kwargs) -> str:
    return _service_request("GET", "/healthz")


registry.register(
    name="create_trade_intent",
    toolset="brokerage",
    schema=CREATE_TRADE_INTENT_SCHEMA,
    handler=create_trade_intent_tool,
    check_fn=check_brokerage_requirements,
    emoji="💸",
)

registry.register(
    name="confirm_trade_intent",
    toolset="brokerage",
    schema=CONFIRM_TRADE_INTENT_SCHEMA,
    handler=confirm_trade_intent_tool,
    check_fn=check_brokerage_requirements,
    emoji="✅",
)

registry.register(
    name="cancel_trade_intent",
    toolset="brokerage",
    schema=CANCEL_TRADE_INTENT_SCHEMA,
    handler=cancel_trade_intent_tool,
    check_fn=check_brokerage_requirements,
    emoji="🛑",
)

registry.register(
    name="get_trade_intent_status",
    toolset="brokerage",
    schema=GET_TRADE_INTENT_STATUS_SCHEMA,
    handler=get_trade_intent_status_tool,
    check_fn=check_brokerage_requirements,
    emoji="📈",
)

registry.register(
    name="brokerage_health",
    toolset="brokerage",
    schema=BROKERAGE_HEALTH_SCHEMA,
    handler=brokerage_health_tool,
    check_fn=check_brokerage_requirements,
    emoji="🩺",
)
