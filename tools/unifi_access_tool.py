#!/usr/bin/env python3
"""
UniFi Access Tool — Caged API tool for the Byrd-IT support agent.

This tool wraps UniFi Access API calls internally using Python urllib.
The agent calls structured tools like `unifi_access_get_doors` and gets
JSON back — it NEVER touches the terminal, never runs curl, and never
sees API keys or tokens. All credential resolution happens internally.

The agent only needs to provide the site name; the tool reads
UniFi-Site.yaml, resolves the credentials, and makes the API call.
"""

import json
import urllib.request
import urllib.error
import ssl
import yaml
import time
import logging
from typing import Dict, Any, List, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

# Path to the UniFi Site config
SITE_CONFIG_PATH = Path("/home/brandonabyrd/.hermes/secrets/UniFi-Site.yaml")

# SSL context that skips cert verification (Access uses self-signed certs)
_SSL_CTX = ssl.create_default_context()
_SSL_CTX.check_hostname = False
_SSL_CTX.verify_mode = ssl.CERT_NONE

# Cache for the site config (reload every 60s)
_config_cache = None
_config_cache_time = 0
_CONFIG_TTL = 60


def _load_site_config() -> dict:
    """Load and cache UniFi-Site.yaml."""
    global _config_cache, _config_cache_time
    now = time.time()
    if _config_cache is None or (now - _config_cache_time) > _CONFIG_TTL:
        with open(SITE_CONFIG_PATH, encoding="utf-8") as f:
            _config_cache = yaml.safe_load(f)
        _config_cache_time = now
    return _config_cache


def _resolve_access_credentials(site_name: str) -> Dict[str, str]:
    """
    Resolve Access API credentials for a given site.

    Returns dict with:
      - access_base: base URL including port (e.g., https://10.10.2.2:12445)
      - access_token: Bearer token
      - site_group: site group name

    Raises ValueError if site not found or no Access API token.
    """
    config = _load_site_config()
    sites = config.get("sites", {})
    api_keys = config.get("api_keys", {})

    # Try exact match first, then case-insensitive, then partial match
    site_data = None
    matched_name = None

    if site_name in sites:
        site_data = sites[site_name]
        matched_name = site_name
    else:
        site_lower = site_name.lower()
        for name, sdata in sites.items():
            if name.lower() == site_lower:
                site_data = sdata
                matched_name = name
                break
        if site_data is None:
            # Partial match on site name or customer
            for name, sdata in sites.items():
                customer = sdata.get("customer", "").lower()
                if site_lower in name.lower() or site_lower in customer:
                    site_data = sdata
                    matched_name = name
                    break

    if site_data is None:
        raise ValueError(f"Site '{site_name}' not found in UniFi-Site.yaml")

    # Check if the site has Access
    controllers = site_data.get("controller", {}).get("controllers", [])
    if isinstance(controllers, list) and "access" not in controllers:
        raise ValueError(f"Site '{matched_name}' does not have UniFi Access deployed")

    # Get the API key reference
    key_ref = site_data.get("api_key_ref")
    if not key_ref:
        raise ValueError(f"Site '{matched_name}' has no api_key_ref")

    # Resolve the API key
    key_data = api_keys.get(key_ref)
    if key_data is None:
        raise ValueError(f"API key '{key_ref}' not found in api_keys section")

    # Verify it's an Access API key
    api_type = key_data.get("api_type", "")
    if api_type != "local_access":
        raise ValueError(
            f"Site '{matched_name}' API key '{key_ref}' has api_type '{api_type}', "
            f"not 'local_access'. No UniFi Access API token available for this site."
        )

    # Get the token and base URL
    token = key_data.get("api_key")
    local_base_url = key_data.get("local_base_url", "")

    if not local_base_url:
        # Fall back to controller local_ip
        local_ip = site_data.get("controller", {}).get("local_ip")
        if local_ip:
            local_base_url = f"https://{local_ip}"
        else:
            raise ValueError(f"Site '{matched_name}' has no local_base_url or local_ip")

    # Construct the base URL with port
    # Remove trailing slash if present
    local_base_url = local_base_url.rstrip("/")
    access_base = f"{local_base_url}:12445"

    site_group = key_data.get("site_group", "")

    return {
        "access_base": access_base,
        "access_token": token,
        "site_name": matched_name,
        "site_group": site_group,
    }


def _make_access_request(
    access_base: str,
    access_token: str,
    method: str,
    path: str,
    body: Optional[dict] = None,
) -> Dict[str, Any]:
    """
    Make an HTTP request to the UniFi Access API.

    Returns the parsed JSON response.
    """
    url = f"{access_base}{path}"

    headers = {
        "Authorization": f"Bearer {access_token}",
        "accept": "application/json",
        "content-type": "application/json",
    }

    data = None
    if body is not None:
        data = json.dumps(body).encode("utf-8")

    req = urllib.request.Request(url, data=data, headers=headers, method=method)

    try:
        with urllib.request.urlopen(req, context=_SSL_CTX, timeout=25) as response:
            response_data = response.read().decode("utf-8")
            if response_data:
                return json.loads(response_data)
            return {"code": "SUCCESS", "data": None}
    except urllib.error.HTTPError as e:
        error_body = e.read().decode("utf-8") if e.fp else ""
        try:
            error_json = json.loads(error_body) if error_body else {}
        except json.JSONDecodeError:
            error_json = {"raw": error_body}
        return {
            "code": "error",
            "http_status": e.code,
            "error": e.reason,
            "details": error_json,
        }
    except urllib.error.URLError as e:
        return {
            "code": "error",
            "error": f"Connection failed: {e.reason}",
            "hint": "The Access controller may be offline or unreachable from this server.",
        }
    except Exception as e:
        return {
            "code": "error",
            "error": str(e),
        }


# ============================================================
# Tool Handlers
# ============================================================


def _sanitize_response(data: Any) -> Any:
    """
    Remove any credentials from the response before returning to the agent.
    The agent should never see API keys, tokens, or internal IPs.
    """
    if isinstance(data, dict):
        sanitized = {}
        for key, value in data.items():
            # Skip any key that might contain credentials
            if key.lower() in ("api_key", "token", "access_token", "password", "secret", "authorization"):
                continue
            sanitized[key] = _sanitize_response(value)
        return sanitized
    elif isinstance(data, list):
        return [_sanitize_response(item) for item in data]
    else:
        return data


def unifi_access_get_doors(site_name: str, **kwargs) -> Dict[str, Any]:
    """Get all doors at a site."""
    try:
        creds = _resolve_access_credentials(site_name)
        result = _make_access_request(
            creds["access_base"], creds["access_token"], "GET", "/api/v1/developer/doors"
        )
        return _sanitize_response(result)
    except ValueError as e:
        return {"code": "error", "error": str(e)}
    except Exception as e:
        return {"code": "error", "error": str(e)}


def unifi_access_get_door(site_name: str, door_id: str, **kwargs) -> Dict[str, Any]:
    """Get details for a specific door."""
    try:
        creds = _resolve_access_credentials(site_name)
        result = _make_access_request(
            creds["access_base"], creds["access_token"], "GET",
            f"/api/v1/developer/doors/{door_id}"
        )
        return _sanitize_response(result)
    except ValueError as e:
        return {"code": "error", "error": str(e)}
    except Exception as e:
        return {"code": "error", "error": str(e)}


def unifi_access_get_lock_rule(site_name: str, door_id: str, **kwargs) -> Dict[str, Any]:
    """Get the current locking rule for a door."""
    try:
        creds = _resolve_access_credentials(site_name)
        result = _make_access_request(
            creds["access_base"], creds["access_token"], "GET",
            f"/api/v1/developer/doors/{door_id}/lock_rule"
        )
        return _sanitize_response(result)
    except ValueError as e:
        return {"code": "error", "error": str(e)}
    except Exception as e:
        return {"code": "error", "error": str(e)}


def unifi_access_get_emergency_status(site_name: str, **kwargs) -> Dict[str, Any]:
    """Get the emergency status for all doors at a site."""
    try:
        creds = _resolve_access_credentials(site_name)
        result = _make_access_request(
            creds["access_base"], creds["access_token"], "GET",
            "/api/v1/developer/doors/settings/emergency"
        )
        return _sanitize_response(result)
    except ValueError as e:
        return {"code": "error", "error": str(e)}
    except Exception as e:
        return {"code": "error", "error": str(e)}


def unifi_access_get_devices(site_name: str, **kwargs) -> Dict[str, Any]:
    """Get all Access devices (readers, hubs, intercoms) at a site."""
    try:
        creds = _resolve_access_credentials(site_name)
        result = _make_access_request(
            creds["access_base"], creds["access_token"], "GET",
            "/api/v1/developer/devices?refresh=true"
        )
        return _sanitize_response(result)
    except ValueError as e:
        return {"code": "error", "error": str(e)}
    except Exception as e:
        return {"code": "error", "error": str(e)}


def unifi_access_get_system_logs(
    site_name: str, topic: str, since: Optional[int] = None, until: Optional[int] = None,
    page_size: int = 50, page_num: int = 0, **kwargs
) -> Dict[str, Any]:
    """
    Get system logs (access events).

    Args:
        site_name: Name of the site
        topic: Log topic (door_openings, critical, updates, device_events, admin_activity, visitor)
        since: Start time (unix timestamp, optional)
        until: End time (unix timestamp, optional)
        page_size: Number of results per page
        page_num: Page number (0-indexed)
    """
    try:
        creds = _resolve_access_credentials(site_name)
        body = {"topic": topic}
        if since is not None:
            body["since"] = since
        if until is not None:
            body["until"] = until

        result = _make_access_request(
            creds["access_base"], creds["access_token"], "POST",
            f"/api/v1/developer/system/logs?page_size={page_size}&page_num={page_num}",
            body=body,
        )
        return _sanitize_response(result)
    except ValueError as e:
        return {"code": "error", "error": str(e)}
    except Exception as e:
        return {"code": "error", "error": str(e)}


def unifi_access_hold_door_open(
    site_name: str, door_id: str, minutes: int, **kwargs
) -> Dict[str, Any]:
    """
    Hold a door/gate open for a specific number of minutes.

    After the interval expires, the door automatically returns to its normal
    locked state.
    """
    try:
        creds = _resolve_access_credentials(site_name)
        body = {"type": "custom", "interval": minutes}
        result = _make_access_request(
            creds["access_base"], creds["access_token"], "PUT",
            f"/api/v1/developer/doors/{door_id}/lock_rule",
            body=body,
        )
        return _sanitize_response(result)
    except ValueError as e:
        return {"code": "error", "error": str(e)}
    except Exception as e:
        return {"code": "error", "error": str(e)}


def unifi_access_unlock_door(site_name: str, door_id: str, **kwargs) -> Dict[str, Any]:
    """
    One-time remote unlock of a door.

    Unlocks for ~2 seconds (long enough for a gate to open or a person to
    pass through), then automatically returns to locked state.
    """
    try:
        creds = _resolve_access_credentials(site_name)
        result = _make_access_request(
            creds["access_base"], creds["access_token"], "PUT",
            f"/api/v1/developer/doors/{door_id}/unlock",
        )
        return _sanitize_response(result)
    except ValueError as e:
        return {"code": "error", "error": str(e)}
    except Exception as e:
        return {"code": "error", "error": str(e)}


def unifi_access_lock_now(site_name: str, door_id: str, **kwargs) -> Dict[str, Any]:
    """
    Immediately lock a door and cancel any active unlock schedule or temporary unlock.
    """
    try:
        creds = _resolve_access_credentials(site_name)
        body = {"type": "lock_now"}
        result = _make_access_request(
            creds["access_base"], creds["access_token"], "PUT",
            f"/api/v1/developer/doors/{door_id}/lock_rule",
            body=body,
        )
        return _sanitize_response(result)
    except ValueError as e:
        return {"code": "error", "error": str(e)}
    except Exception as e:
        return {"code": "error", "error": str(e)}


def unifi_access_reset_door(site_name: str, door_id: str, **kwargs) -> Dict[str, Any]:
    """
    Reset a door to its normal schedule.

    Cancels any temporary locking rule (custom, keep_unlock, keep_lock) and
    returns the door to its normal schedule.
    """
    try:
        creds = _resolve_access_credentials(site_name)
        body = {"type": "reset"}
        result = _make_access_request(
            creds["access_base"], creds["access_token"], "PUT",
            f"/api/v1/developer/doors/{door_id}/lock_rule",
            body=body,
        )
        return _sanitize_response(result)
    except ValueError as e:
        return {"code": "error", "error": str(e)}
    except Exception as e:
        return {"code": "error", "error": str(e)}


# ============================================================
# Tool Definitions (Schemas)
# ============================================================


# Common site_name parameter description
_SITE_DESC = (
    "Name of the UniFi site. Use the site name from UniFi-Site.yaml "
    "(e.g., 'Byrd_Ranch_Gate', 'Byrd_Ranch_Shop'). Case-insensitive, "
    "partial matches supported."
)
_DOOR_ID_DESC = (
    "The unique ID of the door/gate. Get this from unifi_access_get_doors first."
)


GET_DOORS_SCHEMA = {
    "name": "unifi_access_get_doors",
    "description": (
        "Get all doors and gates at a UniFi site with their current lock status "
        "and door position. Returns door names, lock relay status (lock/unlock), "
        "door position (open/close), and door IDs. Use this to check the status "
        "of gates and doors, and to get door IDs for other operations."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "site_name": {"type": "string", "description": _SITE_DESC},
        },
        "required": ["site_name"],
    },
}

GET_DOOR_SCHEMA = {
    "name": "unifi_access_get_door",
    "description": "Get detailed information about a specific door or gate.",
    "parameters": {
        "type": "object",
        "properties": {
            "site_name": {"type": "string", "description": _SITE_DESC},
            "door_id": {"type": "string", "description": _DOOR_ID_DESC},
        },
        "required": ["site_name", "door_id"],
    },
}

GET_LOCK_RULE_SCHEMA = {
    "name": "unifi_access_get_lock_rule",
    "description": (
        "Get the current locking rule for a door. Shows if a temporary rule "
        "(hold-open, lock override) is active and when it ends."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "site_name": {"type": "string", "description": _SITE_DESC},
            "door_id": {"type": "string", "description": _DOOR_ID_DESC},
        },
        "required": ["site_name", "door_id"],
    },
}

GET_EMERGENCY_SCHEMA = {
    "name": "unifi_access_get_emergency_status",
    "description": (
        "Get the emergency status for all doors at a site. Shows if emergency "
        "unlock (evacuation) or lockdown is active."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "site_name": {"type": "string", "description": _SITE_DESC},
        },
        "required": ["site_name"],
    },
}

GET_DEVICES_SCHEMA = {
    "name": "unifi_access_get_devices",
    "description": (
        "Get all Access devices (card readers, hubs, intercoms) at a site "
        "with their online status and type."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "site_name": {"type": "string", "description": _SITE_DESC},
        },
        "required": ["site_name"],
    },
}

GET_LOGS_SCHEMA = {
    "name": "unifi_access_get_system_logs",
    "description": (
        "Get system logs (access events) for a site. Use to check for access "
        "denied events, door open/close history, device disconnects, and critical "
        "errors. Available topics: door_openings, critical, updates, "
        "device_events, admin_activity, visitor. "
        "For immediate issues, use the last 1-2 hours; for broader patterns, "
        "use last 24 hours. Omit since/until to get recent logs."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "site_name": {"type": "string", "description": _SITE_DESC},
            "topic": {
                "type": "string",
                "enum": ["door_openings", "critical", "updates", "device_events", "admin_activity", "visitor"],
                "description": "Type of logs to fetch",
            },
            "since": {
                "type": "integer",
                "description": "Start time as unix timestamp (optional). Omit for recent logs.",
            },
            "until": {
                "type": "integer",
                "description": "End time as unix timestamp (optional). Omit for now.",
            },
            "page_size": {"type": "integer", "description": "Results per page (default 50)"},
            "page_num": {"type": "integer", "description": "Page number, 0-indexed (default 0)"},
        },
        "required": ["site_name", "topic"],
    },
}

HOLD_OPEN_SCHEMA = {
    "name": "unifi_access_hold_door_open",
    "description": (
        "Hold a door or gate open for a specific number of minutes. "
        "After the interval expires, the door automatically returns to its "
        "normal locked state. Use when a customer asks to 'hold the gate open' "
        "for a time period. ALWAYS confirm the action with the customer first."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "site_name": {"type": "string", "description": _SITE_DESC},
            "door_id": {"type": "string", "description": _DOOR_ID_DESC},
            "minutes": {
                "type": "integer",
                "description": "Number of minutes to hold the door open",
            },
        },
        "required": ["site_name", "door_id", "minutes"],
    },
}

UNLOCK_DOOR_SCHEMA = {
    "name": "unifi_access_unlock_door",
    "description": (
        "One-time remote unlock of a door or gate. Unlocks for ~2 seconds "
        "(long enough for a gate to open or person to pass through), then "
        "automatically returns to locked. Use when a customer asks to 'let "
        "someone through' or 'buzz them in'. ALWAYS confirm with the customer first."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "site_name": {"type": "string", "description": _SITE_DESC},
            "door_id": {"type": "string", "description": _DOOR_ID_DESC},
        },
        "required": ["site_name", "door_id"],
    },
}

LOCK_NOW_SCHEMA = {
    "name": "unifi_access_lock_now",
    "description": (
        "Immediately lock a door or gate and cancel any active unlock schedule "
        "or temporary unlock. Use when a customer says 'lock the gate now'. "
        "ALWAYS confirm with the customer first."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "site_name": {"type": "string", "description": _SITE_DESC},
            "door_id": {"type": "string", "description": _DOOR_ID_DESC},
        },
        "required": ["site_name", "door_id"],
    },
}

RESET_DOOR_SCHEMA = {
    "name": "unifi_access_reset_door",
    "description": (
        "Reset a door or gate to its normal schedule. Cancels any temporary "
        "locking rule (hold-open, lock override) and returns to the default "
        "schedule. Use when a customer says 'put the gate back to normal'. "
        "ALWAYS confirm with the customer first."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "site_name": {"type": "string", "description": _SITE_DESC},
            "door_id": {"type": "string", "description": _DOOR_ID_DESC},
        },
        "required": ["site_name", "door_id"],
    },
}


# ============================================================
# Registry — self-registering tools
# ============================================================

from tools.registry import registry, tool_error

# Create a custom toolset for UniFi Access tools
from toolsets import create_custom_toolset

create_custom_toolset(
    name="unifi-access",
    description="UniFi Access API tools — door/gate diagnostics and operations. No terminal required.",
    tools=[
        "unifi_access_get_doors",
        "unifi_access_get_door",
        "unifi_access_get_lock_rule",
        "unifi_access_get_emergency_status",
        "unifi_access_get_devices",
        "unifi_access_get_system_logs",
        "unifi_access_hold_door_open",
        "unifi_access_unlock_door",
        "unifi_access_lock_now",
        "unifi_access_reset_door",
    ],
)


def _check_access_requirements() -> bool:
    """Check if UniFi Access tools are available."""
    try:
        return SITE_CONFIG_PATH.exists()
    except Exception:
        return False


registry.register(
    name="unifi_access_get_doors",
    toolset="unifi-access",
    schema=GET_DOORS_SCHEMA,
    handler=lambda args, **kw: unifi_access_get_doors(site_name=args["site_name"]),
    check_fn=_check_access_requirements,
    emoji="🚪",
)

registry.register(
    name="unifi_access_get_door",
    toolset="unifi-access",
    schema=GET_DOOR_SCHEMA,
    handler=lambda args, **kw: unifi_access_get_door(
        site_name=args["site_name"], door_id=args["door_id"]
    ),
    check_fn=_check_access_requirements,
    emoji="🚪",
)

registry.register(
    name="unifi_access_get_lock_rule",
    toolset="unifi-access",
    schema=GET_LOCK_RULE_SCHEMA,
    handler=lambda args, **kw: unifi_access_get_lock_rule(
        site_name=args["site_name"], door_id=args["door_id"]
    ),
    check_fn=_check_access_requirements,
    emoji="🔐",
)

registry.register(
    name="unifi_access_get_emergency_status",
    toolset="unifi-access",
    schema=GET_EMERGENCY_SCHEMA,
    handler=lambda args, **kw: unifi_access_get_emergency_status(site_name=args["site_name"]),
    check_fn=_check_access_requirements,
    emoji="🚨",
)

registry.register(
    name="unifi_access_get_devices",
    toolset="unifi-access",
    schema=GET_DEVICES_SCHEMA,
    handler=lambda args, **kw: unifi_access_get_devices(site_name=args["site_name"]),
    check_fn=_check_access_requirements,
    emoji="📱",
)

registry.register(
    name="unifi_access_get_system_logs",
    toolset="unifi-access",
    schema=GET_LOGS_SCHEMA,
    handler=lambda args, **kw: unifi_access_get_system_logs(
        site_name=args["site_name"],
        topic=args["topic"],
        since=args.get("since"),
        until=args.get("until"),
        page_size=args.get("page_size", 50),
        page_num=args.get("page_num", 0),
    ),
    check_fn=_check_access_requirements,
    emoji="📋",
)

registry.register(
    name="unifi_access_hold_door_open",
    toolset="unifi-access",
    schema=HOLD_OPEN_SCHEMA,
    handler=lambda args, **kw: unifi_access_hold_door_open(
        site_name=args["site_name"], door_id=args["door_id"], minutes=args["minutes"]
    ),
    check_fn=_check_access_requirements,
    emoji="🔓",
)

registry.register(
    name="unifi_access_unlock_door",
    toolset="unifi-access",
    schema=UNLOCK_DOOR_SCHEMA,
    handler=lambda args, **kw: unifi_access_unlock_door(
        site_name=args["site_name"], door_id=args["door_id"]
    ),
    check_fn=_check_access_requirements,
    emoji="🔓",
)

registry.register(
    name="unifi_access_lock_now",
    toolset="unifi-access",
    schema=LOCK_NOW_SCHEMA,
    handler=lambda args, **kw: unifi_access_lock_now(
        site_name=args["site_name"], door_id=args["door_id"]
    ),
    check_fn=_check_access_requirements,
    emoji="🔒",
)

registry.register(
    name="unifi_access_reset_door",
    toolset="unifi-access",
    schema=RESET_DOOR_SCHEMA,
    handler=lambda args, **kw: unifi_access_reset_door(
        site_name=args["site_name"], door_id=args["door_id"]
    ),
    check_fn=_check_access_requirements,
    emoji="🔄",
)