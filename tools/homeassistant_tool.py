"""Home Assistant tool for controlling smart home devices via REST API.

Registers six LLM-callable tools:
- ``ha_list_entities`` -- list/filter entities by domain or area
- ``ha_get_state`` -- get detailed state of a single entity
- ``ha_list_services`` -- list available services (actions) per domain
- ``ha_call_service`` -- call a HA service (turn_on, turn_off, set_temperature, etc.)
- ``ha_get_history`` -- get state history for an entity over a time period
- ``ha_get_camera_image`` -- retrieve a snapshot image from a camera entity

Authentication uses a Long-Lived Access Token via ``HASS_TOKEN`` env var.
The HA instance URL is read from ``HASS_URL`` (default: http://homeassistant.local:8123).
"""

import asyncio
import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import quote, urlencode

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Kept for backward compatibility (e.g. test monkeypatching); prefer _get_config().
_HASS_URL: str = ""
_HASS_TOKEN: str = ""


def _get_config():
    """Return (hass_url, hass_token) from env vars at call time."""
    return (
        (_HASS_URL or os.getenv("HASS_URL", "http://homeassistant.local:8123")).rstrip("/"),
        _HASS_TOKEN or os.getenv("HASS_TOKEN", ""),
    )

# Regex for valid HA entity_id format (e.g. "light.living_room", "sensor.temperature_1")
_ENTITY_ID_RE = re.compile(r"^[a-z_][a-z0-9_]*\.[a-z0-9_]+$")

# Regex for valid HA service/domain names (e.g. "light", "turn_on", "shell_command").
# Only lowercase ASCII letters, digits, and underscores — no slashes, dots, or
# other characters that could allow path traversal in URL construction.
# The domain and service are interpolated into /api/services/{domain}/{service},
# so allowing arbitrary strings would enable SSRF via path traversal
# (e.g. domain="../../api/config") or blocked-domain bypass
# (e.g. domain="shell_command/../light").
_SERVICE_NAME_RE = re.compile(r"^[a-z][a-z0-9_]*$")

# Service domains blocked for security -- these allow arbitrary code/command
# execution on the HA host or enable SSRF attacks on the local network.
# HA provides zero service-level access control; all safety must be in our layer.
_BLOCKED_DOMAINS = frozenset({
    "shell_command",    # arbitrary shell commands as root in HA container
    "command_line",     # sensors/switches that execute shell commands
    "python_script",    # sandboxed but can escalate via hass.services.call()
    "pyscript",         # scripting integration with broader access
    "hassio",           # addon control, host shutdown/reboot, stdin to containers
    "rest_command",     # HTTP requests from HA server (SSRF vector)
})


def _get_headers(token: str = "") -> Dict[str, str]:
    """Return authorization headers for HA REST API."""
    if not token:
        _, token = _get_config()
    return {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }


# ---------------------------------------------------------------------------
# Async helpers (called from sync handlers via run_until_complete)
# ---------------------------------------------------------------------------

def _filter_and_summarize(
    states: list,
    domain: Optional[str] = None,
    area: Optional[str] = None,
) -> Dict[str, Any]:
    """Filter raw HA states by domain/area and return a compact summary."""
    if domain:
        states = [s for s in states if s.get("entity_id", "").startswith(f"{domain}.")]

    if area:
        area_lower = area.lower()
        states = [
            s for s in states
            if area_lower in (s.get("attributes", {}).get("friendly_name", "") or "").lower()
            or area_lower in (s.get("attributes", {}).get("area", "") or "").lower()
        ]

    entities = []
    for s in states:
        entities.append({
            "entity_id": s["entity_id"],
            "state": s["state"],
            "friendly_name": s.get("attributes", {}).get("friendly_name", ""),
        })

    return {"count": len(entities), "entities": entities}


async def _async_list_entities(
    domain: Optional[str] = None,
    area: Optional[str] = None,
) -> Dict[str, Any]:
    """Fetch entity states from HA and optionally filter by domain/area."""
    import aiohttp

    hass_url, hass_token = _get_config()
    url = f"{hass_url}/api/states"
    async with aiohttp.ClientSession() as session:
        async with session.get(url, headers=_get_headers(hass_token), timeout=aiohttp.ClientTimeout(total=15)) as resp:
            resp.raise_for_status()
            states = await resp.json()

    return _filter_and_summarize(states, domain, area)


async def _async_get_state(entity_id: str) -> Dict[str, Any]:
    """Fetch detailed state of a single entity."""
    import aiohttp

    hass_url, hass_token = _get_config()
    url = f"{hass_url}/api/states/{entity_id}"
    async with aiohttp.ClientSession() as session:
        async with session.get(url, headers=_get_headers(hass_token), timeout=aiohttp.ClientTimeout(total=10)) as resp:
            resp.raise_for_status()
            data = await resp.json()

    return {
        "entity_id": data["entity_id"],
        "state": data["state"],
        "attributes": data.get("attributes", {}),
        "last_changed": data.get("last_changed"),
        "last_updated": data.get("last_updated"),
    }


def _build_service_payload(
    entity_id: Optional[str] = None,
    data: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build the JSON payload for a HA service call."""
    payload: Dict[str, Any] = {}
    if data:
        payload.update(data)
    # entity_id parameter takes precedence over data["entity_id"]
    if entity_id:
        payload["entity_id"] = entity_id
    return payload


def _parse_service_response(
    domain: str,
    service: str,
    result: Any,
) -> Dict[str, Any]:
    """Parse HA service call response into a structured result."""
    affected = []
    if isinstance(result, list):
        for s in result:
            affected.append({
                "entity_id": s.get("entity_id", ""),
                "state": s.get("state", ""),
            })

    return {
        "success": True,
        "service": f"{domain}.{service}",
        "affected_entities": affected,
    }


async def _async_call_service(
    domain: str,
    service: str,
    entity_id: Optional[str] = None,
    data: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Call a Home Assistant service."""
    import aiohttp

    hass_url, hass_token = _get_config()
    url = f"{hass_url}/api/services/{domain}/{service}"
    payload = _build_service_payload(entity_id, data)

    async with aiohttp.ClientSession() as session:
        async with session.post(
            url,
            headers=_get_headers(hass_token),
            json=payload,
            timeout=aiohttp.ClientTimeout(total=15),
        ) as resp:
            resp.raise_for_status()
            result = await resp.json()

    return _parse_service_response(domain, service, result)


# ---------------------------------------------------------------------------
# Sync wrappers (handler signature: (args, **kw) -> str)
# ---------------------------------------------------------------------------

def _run_async(coro):
    """Run an async coroutine from a sync handler."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        # Already inside an event loop -- create a new thread
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(asyncio.run, coro)
            return future.result(timeout=30)
    else:
        return asyncio.run(coro)


def _handle_list_entities(args: dict, **kw) -> str:
    """Handler for ha_list_entities tool."""
    domain = args.get("domain")
    area = args.get("area")
    try:
        result = _run_async(_async_list_entities(domain=domain, area=area))
        return json.dumps({"result": result})
    except Exception as e:
        logger.error("ha_list_entities error: %s", e)
        return tool_error(f"Failed to list entities: {e}")


def _handle_get_state(args: dict, **kw) -> str:
    """Handler for ha_get_state tool."""
    entity_id = args.get("entity_id", "")
    if not entity_id:
        return tool_error("Missing required parameter: entity_id")
    if not _ENTITY_ID_RE.match(entity_id):
        return tool_error(f"Invalid entity_id format: {entity_id}")
    try:
        result = _run_async(_async_get_state(entity_id))
        return json.dumps({"result": result})
    except Exception as e:
        logger.error("ha_get_state error: %s", e)
        return tool_error(f"Failed to get state for {entity_id}: {e}")


def _handle_call_service(args: dict, **kw) -> str:
    """Handler for ha_call_service tool."""
    domain = args.get("domain", "")
    service = args.get("service", "")
    if not domain or not service:
        return tool_error("Missing required parameters: domain and service")

    # Validate domain/service format BEFORE the blocklist check — prevents
    # path traversal in /api/services/{domain}/{service} and blocklist bypass
    # via payloads like "shell_command/../light".
    if not _SERVICE_NAME_RE.match(domain):
        return tool_error(f"Invalid domain format: {domain!r}")
    if not _SERVICE_NAME_RE.match(service):
        return tool_error(f"Invalid service format: {service!r}")

    if domain in _BLOCKED_DOMAINS:
        return json.dumps({
            "error": f"Service domain '{domain}' is blocked for security. "
            f"Blocked domains: {', '.join(sorted(_BLOCKED_DOMAINS))}"
        })

    entity_id = args.get("entity_id")
    if entity_id and not _ENTITY_ID_RE.match(entity_id):
        return tool_error(f"Invalid entity_id format: {entity_id}")

    data = args.get("data")
    if isinstance(data, str):
        try:
            data = json.loads(data) if data.strip() else None
        except json.JSONDecodeError as e:
            return tool_error(f"Invalid JSON string in 'data' parameter: {e}")

    try:
        result = _run_async(_async_call_service(domain, service, entity_id, data))
        return json.dumps({"result": result})
    except Exception as e:
        logger.error("ha_call_service error: %s", e)
        return tool_error(f"Failed to call {domain}.{service}: {e}")


# ---------------------------------------------------------------------------
# History
# ---------------------------------------------------------------------------

# Maximum time window for history queries (7 days) — prevents accidental
# full-database dumps that could overwhelm the HA instance or the LLM context.
_MAX_HISTORY_HOURS = 168


def _validate_timestamp(ts: str) -> bool:
    """Check that *ts* looks like an ISO-8601 datetime (basic sanity check).

    Accepts formats like ``2024-01-15T10:30:00+00:00`` or ``2024-01-15T10:30:00Z``.
    Rejects path-traversal attempts and arbitrary strings.
    """
    return bool(re.match(
        r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?(Z|[+-]\d{2}:\d{2})?$",
        ts,
    ))


def _summarize_history(entity_id: str, history: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Distil a raw HA history list into a compact summary for LLM consumption.

    Returns total change count, first/last state with timestamps, the set of
    distinct state values with how often each appeared, and — for numeric
    sensors — min/max/average.
    """
    if not history:
        return {"entity_id": entity_id, "total_changes": 0}

    first = history[0]
    last = history[-1]

    summary: Dict[str, Any] = {
        "entity_id": entity_id,
        "total_changes": len(history),
        "first": {
            "state": first.get("state"),
            "last_changed": first.get("last_changed"),
        },
        "last": {
            "state": last.get("state"),
            "last_changed": last.get("last_changed"),
        },
    }

    # For numeric entities (sensors), add min/max/average instead of
    # distinct_states — numeric values can have very high cardinality.
    numeric_values: List[float] = []
    for s in history:
        try:
            numeric_values.append(float(s["state"]))
        except (ValueError, TypeError, KeyError):
            continue

    if numeric_values:
        summary["numeric"] = {
            "min": round(min(numeric_values), 2),
            "max": round(max(numeric_values), 2),
            "average": round(sum(numeric_values) / len(numeric_values), 2),
        }
    else:
        # Non-numeric entities (binary sensors, lights, etc.) — low cardinality,
        # so distinct_states is useful context for the agent.
        from collections import Counter
        state_counts = Counter(s.get("state") for s in history)
        summary["distinct_states"] = dict(state_counts)

    return summary


async def _async_get_history(
    entity_id: str,
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
    include_details: bool = False,
) -> Dict[str, Any]:
    """Fetch state history for an entity from the HA REST API.

    By default returns a compact summary (change count, first/last state,
    distinct values, numeric min/max/avg).  Set *include_details=True* to
    also return the full list of state changes.
    """
    import aiohttp

    hass_url, hass_token = _get_config()

    # Build URL path — timestamp in path is optional
    path = "/api/history/period"
    if start_time:
        path = f"/api/history/period/{quote(start_time, safe='')}"

    # Query parameters — always request minimal_response + significant_changes_only
    # from HA to keep the payload small; the summary provides the useful stats.
    params: Dict[str, str] = {
        "filter_entity_id": entity_id,
        "minimal_response": "",
        "significant_changes_only": "",
    }
    if end_time:
        params["end_time"] = end_time

    url = f"{hass_url}{path}?{urlencode(params)}"

    async with aiohttp.ClientSession() as session:
        async with session.get(
            url,
            headers=_get_headers(hass_token),
            timeout=aiohttp.ClientTimeout(total=30),
        ) as resp:
            resp.raise_for_status()
            data = await resp.json()

    # HA returns a list of lists — one per entity. We requested a single entity.
    history: List[Dict[str, Any]] = data[0] if data else []
    result = _summarize_history(entity_id, history)

    if include_details:
        result["states"] = history

    return result


def _handle_get_history(args: dict, **kw) -> str:
    """Handler for ha_get_history tool."""
    entity_id = args.get("entity_id", "")
    if not entity_id:
        return tool_error("Missing required parameter: entity_id")
    if not _ENTITY_ID_RE.match(entity_id):
        return tool_error(f"Invalid entity_id format: {entity_id}")

    start_time = args.get("start_time")
    end_time = args.get("end_time")

    if start_time and not _validate_timestamp(start_time):
        return tool_error(f"Invalid start_time format (expected ISO-8601): {start_time}")
    if end_time and not _validate_timestamp(end_time):
        return tool_error(f"Invalid end_time format (expected ISO-8601): {end_time}")

    include_details = args.get("include_details", False)

    try:
        result = _run_async(_async_get_history(
            entity_id,
            start_time=start_time,
            end_time=end_time,
            include_details=include_details,
        ))
        return json.dumps({"result": result})
    except Exception as e:
        logger.error("ha_get_history error: %s", e)
        return tool_error(f"Failed to get history for {entity_id}: {e}")


# ---------------------------------------------------------------------------
# Camera image
# ---------------------------------------------------------------------------

# Only camera entities are allowed — prevents using the endpoint to proxy
# requests to arbitrary entity IDs.
_CAMERA_ENTITY_RE = re.compile(r"^camera\.[a-z0-9_]+$")


_MIME_TO_EXT = {
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/gif": ".gif",
    "image/webp": ".webp",
}


def _camera_image_dir() -> Path:
    """Return (and create) the directory for camera snapshots."""
    from hermes_constants import get_hermes_home

    d = get_hermes_home() / "tmp" / "camera"
    d.mkdir(parents=True, exist_ok=True)
    return d


async def _async_get_camera_image(entity_id: str) -> Dict[str, Any]:
    """Fetch a camera snapshot image and save it to disk."""
    import aiohttp

    hass_url, hass_token = _get_config()
    url = f"{hass_url}/api/camera_proxy/{entity_id}"

    async with aiohttp.ClientSession() as session:
        async with session.get(
            url,
            headers=_get_headers(hass_token),
            timeout=aiohttp.ClientTimeout(total=30),
        ) as resp:
            resp.raise_for_status()
            image_bytes = await resp.read()
            content_type = resp.headers.get("Content-Type", "image/jpeg")

    # Determine MIME type from Content-Type header (strip charset etc.)
    mime_type = content_type.split(";")[0].strip()
    if mime_type not in _MIME_TO_EXT:
        mime_type = "image/jpeg"

    ext = _MIME_TO_EXT[mime_type]
    # Name: camera entity without "camera." prefix, e.g. "front_door.jpg"
    name = entity_id.split(".", 1)[1]
    dest = _camera_image_dir() / f"{name}{ext}"
    dest.write_bytes(image_bytes)

    return {
        "entity_id": entity_id,
        "mime_type": mime_type,
        "size_bytes": len(image_bytes),
        "image_path": str(dest),
    }


def _handle_get_camera_image(args: dict, **kw) -> str:
    """Handler for ha_get_camera_image tool."""
    entity_id = args.get("entity_id", "")
    if not entity_id:
        return tool_error("Missing required parameter: entity_id")
    if not _CAMERA_ENTITY_RE.match(entity_id):
        return tool_error(
            f"Invalid camera entity_id: {entity_id}. "
            "Must be a camera entity (e.g. 'camera.front_door')."
        )
    try:
        result = _run_async(_async_get_camera_image(entity_id))
        return json.dumps({"result": result})
    except Exception as e:
        logger.error("ha_get_camera_image error: %s", e)
        return tool_error(f"Failed to get camera image for {entity_id}: {e}")


# ---------------------------------------------------------------------------
# List services
# ---------------------------------------------------------------------------

async def _async_list_services(domain: Optional[str] = None) -> Dict[str, Any]:
    """Fetch available services from HA and optionally filter by domain."""
    import aiohttp

    hass_url, hass_token = _get_config()
    url = f"{hass_url}/api/services"
    headers = {"Authorization": f"Bearer {hass_token}", "Content-Type": "application/json"}
    async with aiohttp.ClientSession() as session:
        async with session.get(url, headers=headers, timeout=aiohttp.ClientTimeout(total=15)) as resp:
            resp.raise_for_status()
            services = await resp.json()

    if domain:
        services = [s for s in services if s.get("domain") == domain]

    # Compact the output for context efficiency
    result = []
    for svc_domain in services:
        d = svc_domain.get("domain", "")
        domain_services = {}
        for svc_name, svc_info in svc_domain.get("services", {}).items():
            svc_entry: Dict[str, Any] = {"description": svc_info.get("description", "")}
            fields = svc_info.get("fields", {})
            if fields:
                svc_entry["fields"] = {
                    k: v.get("description", "") for k, v in fields.items()
                    if isinstance(v, dict)
                }
            domain_services[svc_name] = svc_entry
        result.append({"domain": d, "services": domain_services})

    return {"count": len(result), "domains": result}


def _handle_list_services(args: dict, **kw) -> str:
    """Handler for ha_list_services tool."""
    domain = args.get("domain")
    try:
        result = _run_async(_async_list_services(domain=domain))
        return json.dumps({"result": result})
    except Exception as e:
        logger.error("ha_list_services error: %s", e)
        return tool_error(f"Failed to list services: {e}")


# ---------------------------------------------------------------------------
# Availability check
# ---------------------------------------------------------------------------

def _check_ha_available() -> bool:
    """Tool is only available when HASS_TOKEN is set."""
    return bool(os.getenv("HASS_TOKEN"))


# ---------------------------------------------------------------------------
# Tool schemas
# ---------------------------------------------------------------------------

HA_LIST_ENTITIES_SCHEMA = {
    "name": "ha_list_entities",
    "description": (
        "List Home Assistant entities. Optionally filter by domain "
        "(light, switch, climate, sensor, binary_sensor, cover, fan, etc.) "
        "or by area name (living room, kitchen, bedroom, etc.)."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "domain": {
                "type": "string",
                "description": (
                    "Entity domain to filter by (e.g. 'light', 'switch', 'climate', "
                    "'sensor', 'binary_sensor', 'cover', 'fan', 'media_player'). "
                    "Omit to list all entities."
                ),
            },
            "area": {
                "type": "string",
                "description": (
                    "Area/room name to filter by (e.g. 'living room', 'kitchen'). "
                    "Matches against entity friendly names. Omit to list all."
                ),
            },
        },
        "required": [],
    },
}

HA_GET_STATE_SCHEMA = {
    "name": "ha_get_state",
    "description": (
        "Get the detailed state of a single Home Assistant entity, including all "
        "attributes (brightness, color, temperature setpoint, sensor readings, etc.)."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "entity_id": {
                "type": "string",
                "description": (
                    "The entity ID to query (e.g. 'light.living_room', "
                    "'climate.thermostat', 'sensor.temperature')."
                ),
            },
        },
        "required": ["entity_id"],
    },
}

HA_LIST_SERVICES_SCHEMA = {
    "name": "ha_list_services",
    "description": (
        "List available Home Assistant services (actions) for device control. "
        "Shows what actions can be performed on each device type and what "
        "parameters they accept. Use this to discover how to control devices "
        "found via ha_list_entities."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "domain": {
                "type": "string",
                "description": (
                    "Filter by domain (e.g. 'light', 'climate', 'switch'). "
                    "Omit to list services for all domains."
                ),
            },
        },
        "required": [],
    },
}

HA_CALL_SERVICE_SCHEMA = {
    "name": "ha_call_service",
    "description": (
        "Call a Home Assistant service to control a device. Use ha_list_services "
        "to discover available services and their parameters for each domain."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "domain": {
                "type": "string",
                "description": (
                    "Service domain (e.g. 'light', 'switch', 'climate', "
                    "'cover', 'media_player', 'fan', 'scene', 'script')."
                ),
            },
            "service": {
                "type": "string",
                "description": (
                    "Service name (e.g. 'turn_on', 'turn_off', 'toggle', "
                    "'set_temperature', 'set_hvac_mode', 'open_cover', "
                    "'close_cover', 'set_volume_level')."
                ),
            },
            "entity_id": {
                "type": "string",
                "description": (
                    "Target entity ID (e.g. 'light.living_room'). "
                    "Some services (like scene.turn_on) may not need this."
                ),
            },
            "data": {
                "type": "string",
                "description": (
                    "Additional service data as a JSON string. Examples: "
                    '{"brightness": 255, "color_name": "blue"} for lights, '
                    '{"temperature": 22, "hvac_mode": "heat"} for climate, '
                    '{"volume_level": 0.5} for media players.'
                ),
            },
        },
        "required": ["domain", "service"],
    },
}

HA_GET_HISTORY_SCHEMA = {
    "name": "ha_get_history",
    "description": (
        "Get state history for a Home Assistant entity. By default returns a "
        "compact summary: total change count, first/last state with timestamps, "
        "distinct state values with counts, and numeric min/max/average for "
        "sensors. Set include_details=true to also get the full state list."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "entity_id": {
                "type": "string",
                "description": (
                    "The entity ID to get history for (e.g. 'sensor.temperature', "
                    "'binary_sensor.front_door', 'light.living_room')."
                ),
            },
            "start_time": {
                "type": "string",
                "description": (
                    "Start of the history period in ISO-8601 format "
                    "(e.g. '2024-01-15T10:00:00+00:00'). "
                    "Defaults to 1 day before now if omitted."
                ),
            },
            "end_time": {
                "type": "string",
                "description": (
                    "End of the history period in ISO-8601 format "
                    "(e.g. '2024-01-15T18:00:00+00:00'). "
                    "Defaults to now if omitted."
                ),
            },
            "include_details": {
                "type": "boolean",
                "description": (
                    "If true, include the full list of state changes in the "
                    "response alongside the summary. Default: false (summary only)."
                ),
            },
        },
        "required": ["entity_id"],
    },
}

HA_GET_CAMERA_IMAGE_SCHEMA = {
    "name": "ha_get_camera_image",
    "description": (
        "Get a snapshot image from a Home Assistant camera entity. "
        "Saves the image to disk and returns the file path. Use this to see "
        "what a camera is currently showing (e.g. front door camera, baby monitor, etc.)."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "entity_id": {
                "type": "string",
                "description": (
                    "The camera entity ID (e.g. 'camera.front_door', "
                    "'camera.backyard', 'camera.baby_room'). "
                    "Must start with 'camera.'."
                ),
            },
        },
        "required": ["entity_id"],
    },
}


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

from tools.registry import registry, tool_error

registry.register(
    name="ha_list_entities",
    toolset="homeassistant",
    schema=HA_LIST_ENTITIES_SCHEMA,
    handler=_handle_list_entities,
    check_fn=_check_ha_available,
    emoji="🏠",
)

registry.register(
    name="ha_get_state",
    toolset="homeassistant",
    schema=HA_GET_STATE_SCHEMA,
    handler=_handle_get_state,
    check_fn=_check_ha_available,
    emoji="🏠",
)

registry.register(
    name="ha_list_services",
    toolset="homeassistant",
    schema=HA_LIST_SERVICES_SCHEMA,
    handler=_handle_list_services,
    check_fn=_check_ha_available,
    emoji="🏠",
)

registry.register(
    name="ha_call_service",
    toolset="homeassistant",
    schema=HA_CALL_SERVICE_SCHEMA,
    handler=_handle_call_service,
    check_fn=_check_ha_available,
    emoji="🏠",
)

registry.register(
    name="ha_get_history",
    toolset="homeassistant",
    schema=HA_GET_HISTORY_SCHEMA,
    handler=_handle_get_history,
    check_fn=_check_ha_available,
    emoji="🏠",
)

registry.register(
    name="ha_get_camera_image",
    toolset="homeassistant",
    schema=HA_GET_CAMERA_IMAGE_SCHEMA,
    handler=_handle_get_camera_image,
    check_fn=_check_ha_available,
    emoji="🏠",
)
