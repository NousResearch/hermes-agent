"""Home Assistant tool for controlling smart home devices via REST API.

Registers four LLM-callable tools:
- ``ha_list_entities`` -- list/filter entities by domain or area
- ``ha_get_state`` -- get detailed state of a single entity
- ``ha_list_services`` -- list available services (actions) per domain
- ``ha_call_service`` -- call a HA service (turn_on, turn_off, set_temperature, etc.)

Authentication uses a Long-Lived Access Token via ``HASS_TOKEN`` env var.
The HA instance URL is read from ``HASS_URL`` (default: http://homeassistant.local:8123).
"""

import asyncio
import json
import logging
import os
import re
import threading
from typing import Any, Dict, Optional

from agent.async_utils import safe_schedule_threadsafe

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Kept for backward compatibility (e.g. test monkeypatching); prefer _get_config().
_HASS_URL: str = ""
_HASS_TOKEN: str = ""

# Persistent event loops for sync->async bridging. Using asyncio.run() per call
# creates and then closes a new event loop every time, which can orphan cached
# async client transports and surface as "Unclosed client session" warnings.
_tool_loop = None
_tool_loop_lock = threading.Lock()
_worker_thread_local = threading.local()
_async_context_loop = None
_async_context_loop_thread = None
_async_context_loop_lock = threading.Lock()


def _run_background_loop(loop: asyncio.AbstractEventLoop, ready: threading.Event) -> None:
    """Own a persistent event loop on a dedicated daemon thread."""
    asyncio.set_event_loop(loop)
    ready.set()
    loop.run_forever()


def _get_async_context_loop() -> asyncio.AbstractEventLoop:
    """Return a persistent background loop for calls made inside a running loop."""
    global _async_context_loop, _async_context_loop_thread
    with _async_context_loop_lock:
        if (
            _async_context_loop is None
            or _async_context_loop.is_closed()
            or _async_context_loop_thread is None
            or not _async_context_loop_thread.is_alive()
        ):
            loop = asyncio.new_event_loop()
            ready = threading.Event()
            thread = threading.Thread(
                target=_run_background_loop,
                args=(loop, ready),
                name="homeassistant-tool-async-bridge",
                daemon=True,
            )
            thread.start()
            if not ready.wait(timeout=5):
                loop.close()
                raise RuntimeError("Timed out starting Home Assistant async bridge loop")
            _async_context_loop = loop
            _async_context_loop_thread = thread
        return _async_context_loop


def _submit_to_async_context_loop(coro):
    """Run a coroutine on the persistent async-context bridge loop."""
    bridge_loop = _get_async_context_loop()
    future = safe_schedule_threadsafe(
        coro,
        bridge_loop,
        logger=logger,
        log_message="Failed to schedule Home Assistant coroutine on async bridge loop",
    )
    if future is None:
        raise RuntimeError(
            "Failed to schedule Home Assistant coroutine on async bridge loop"
        )
    try:
        return future.result(timeout=30)
    except Exception:
        future.cancel()
        raise


def _get_tool_loop():
    """Return a long-lived event loop for main-thread tool calls."""
    global _tool_loop
    with _tool_loop_lock:
        if _tool_loop is None or _tool_loop.is_closed():
            _tool_loop = asyncio.new_event_loop()
        return _tool_loop


def _get_worker_loop():
    """Return a persistent event loop for the current worker thread."""
    loop = getattr(_worker_thread_local, "loop", None)
    if loop is None or loop.is_closed():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        _worker_thread_local.loop = loop
    return loop


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
    """Run an async coroutine from a sync handler.

    When possible, reuse a persistent event loop instead of asyncio.run(),
    which creates and closes a fresh loop on every call. Reusing loops keeps
    async client cleanup bound to a live loop and avoids aiohttp unclosed
    session/connector warnings.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        return _submit_to_async_context_loop(coro)

    if threading.current_thread() is not threading.main_thread():
        worker_loop = _get_worker_loop()
        return worker_loop.run_until_complete(coro)

    tool_loop = _get_tool_loop()
    return tool_loop.run_until_complete(coro)


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
