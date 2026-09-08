"""Home Assistant tool for controlling smart home devices via REST API.

Registers ``ha_list_entities``, ``ha_get_state``, ``ha_list_services``, ``ha_call_service``.
Auth is a Long-Lived Access Token (``HASS_TOKEN``); the instance URL comes from
``HASS_URL`` (default http://homeassistant.local:8123).
"""

import asyncio
import json
import logging
import re
from typing import Any, Dict, Optional

from agent.secret_scope import get_secret
from tools.registry import registry, tool_error

logger = logging.getLogger(__name__)


def _get_config():
    """Return the active profile's Home Assistant URL and token."""
    return (
        (get_secret("HASS_URL", "http://homeassistant.local:8123") or "").rstrip("/"),
        get_secret("HASS_TOKEN", "") or "")


# Valid HA entity_id (e.g. "light.living_room", "sensor.temperature_1").
_ENTITY_ID_RE = re.compile(r"^[a-z_][a-z0-9_]*\.[a-z0-9_]+$")

# Domain/service names are interpolated into /api/services/{domain}/{service}, so only
# [a-z0-9_] is allowed: anything else enables SSRF via path traversal
# (domain="../../api/config") or blocklist bypass (domain="shell_command/../light").
_SERVICE_NAME_RE = re.compile(r"^[a-z][a-z0-9_]*$")

# Domains that allow arbitrary code/command execution on the HA host or SSRF on the
# local network. HA has zero service-level access control; all safety lives here.
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
    return {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}


async def _api_json(method: str, path: str, timeout: float, payload: Any = None) -> Any:
    """One HA REST call (GET or POST JSON) that raises on HTTP errors and returns the JSON body."""
    import aiohttp
    hass_url, hass_token = _get_config()
    kwargs: Dict[str, Any] = {"headers": _get_headers(hass_token), "timeout": aiohttp.ClientTimeout(total=timeout)}
    if method == "POST":
        kwargs["json"] = payload
    async with aiohttp.ClientSession() as session:
        async with session.request(method, f"{hass_url}{path}", **kwargs) as resp:
            resp.raise_for_status()
            return await resp.json()


# ── async helpers (called from sync handlers via _run_async) ─────────────────
def _filter_and_summarize(states: list, domain: Optional[str] = None, area: Optional[str] = None) -> Dict:
    """Filter raw HA states by domain/area (area matches friendly_name or area attr) and compact them."""
    if domain:
        states = [s for s in states if s.get("entity_id", "").startswith(f"{domain}.")]
    if area:
        area_lower = area.lower()
        states = [
            s for s in states
            if area_lower in (s.get("attributes", {}).get("friendly_name", "") or "").lower()
            or area_lower in (s.get("attributes", {}).get("area", "") or "").lower()]
    entities = [
        {
            "entity_id": s["entity_id"], "state": s["state"],
            "friendly_name": s.get("attributes", {}).get("friendly_name", "")}
        for s in states]
    return {"count": len(entities), "entities": entities}


async def _async_list_entities(domain: Optional[str] = None, area: Optional[str] = None) -> Dict[str, Any]:
    return _filter_and_summarize(await _api_json("GET", "/api/states", 15), domain, area)


async def _async_get_state(entity_id: str) -> Dict[str, Any]:
    data = await _api_json("GET", f"/api/states/{entity_id}", 10)
    return {
        "entity_id": data["entity_id"], "state": data["state"], "attributes": data.get("attributes", {}),
        "last_changed": data.get("last_changed"), "last_updated": data.get("last_updated")}


def _build_service_payload(entity_id: Optional[str] = None, data: Optional[Dict[str, Any]] = None) -> Dict:
    """JSON payload for a HA service call; ``entity_id`` overrides data["entity_id"]."""
    payload: Dict[str, Any] = dict(data or {})
    if entity_id:
        payload["entity_id"] = entity_id
    return payload


def _parse_service_response(domain: str, service: str, result: Any) -> Dict[str, Any]:
    affected = []
    if isinstance(result, list):
        affected = [{"entity_id": s.get("entity_id", ""), "state": s.get("state", "")} for s in result]
    return {"success": True, "service": f"{domain}.{service}", "affected_entities": affected}


async def _async_call_service(
    domain: str, service: str, entity_id: Optional[str] = None, data: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    result = await _api_json(
        "POST", f"/api/services/{domain}/{service}", 15, _build_service_payload(entity_id, data))
    return _parse_service_response(domain, service, result)


async def _async_list_services(domain: Optional[str] = None) -> Dict[str, Any]:
    """Available services, optionally filtered by domain, compacted for context."""
    services = await _api_json("GET", "/api/services", 15)
    if domain:
        services = [s for s in services if s.get("domain") == domain]
    result = []
    for svc_domain in services:
        domain_services = {}
        for svc_name, svc_info in svc_domain.get("services", {}).items():
            svc_entry: Dict[str, Any] = {"description": svc_info.get("description", "")}
            fields = svc_info.get("fields", {})
            if fields:
                svc_entry["fields"] = {
                    k: v.get("description", "") for k, v in fields.items() if isinstance(v, dict)}
            domain_services[svc_name] = svc_entry
        result.append({"domain": svc_domain.get("domain", ""), "services": domain_services})
    return {"count": len(result), "domains": result}


# ── sync wrappers (handler signature: (args, **kw) -> str) ───────────────────
def _run_async(coro):
    """Run a coroutine from a sync handler; hops to a thread if a loop is already running."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    if loop and loop.is_running():  # already inside a loop: asyncio.run() needs its own thread
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(asyncio.run, coro).result(timeout=30)
    return asyncio.run(coro)


def _dispatch(coro, log_name: str, fail_msg: str) -> str:
    """Run ``coro`` and wrap as ``{"result": ...}``; on error log and return tool_error."""
    try:
        return json.dumps({"result": _run_async(coro)})
    except Exception as e:
        logger.error("%s error: %s", log_name, e)
        return tool_error(f"{fail_msg}: {e}")


def _handle_get_state(args: dict, **kw) -> str:
    entity_id = args.get("entity_id", "")
    if not entity_id:
        return tool_error("Missing required parameter: entity_id")
    if not _ENTITY_ID_RE.match(entity_id):
        return tool_error(f"Invalid entity_id format: {entity_id}")
    return _dispatch(_async_get_state(entity_id), "ha_get_state", f"Failed to get state for {entity_id}")


def _handle_call_service(args: dict, **kw) -> str:
    domain = args.get("domain", "")
    service = args.get("service", "")
    if not domain or not service:
        return tool_error("Missing required parameters: domain and service")
    # Format check BEFORE the blocklist: rejects "shell_command/../light" style bypasses.
    if not _SERVICE_NAME_RE.match(domain):
        return tool_error(f"Invalid domain format: {domain!r}")
    if not _SERVICE_NAME_RE.match(service):
        return tool_error(f"Invalid service format: {service!r}")
    if domain in _BLOCKED_DOMAINS:
        return tool_error(
            f"Service domain '{domain}' is blocked for security. "
            f"Blocked domains: {', '.join(sorted(_BLOCKED_DOMAINS))}")
    entity_id = args.get("entity_id")
    if entity_id and not _ENTITY_ID_RE.match(entity_id):
        return tool_error(f"Invalid entity_id format: {entity_id}")
    data = args.get("data")
    if isinstance(data, str):  # XML tool-calling mode delivers data as a JSON string
        try:
            data = json.loads(data) if data.strip() else None
        except json.JSONDecodeError as e:
            return tool_error(f"Invalid JSON string in 'data' parameter: {e}")
    return _dispatch(
        _async_call_service(domain, service, entity_id, data),
        "ha_call_service", f"Failed to call {domain}.{service}")


def _check_ha_available() -> bool:
    """Tool is only available when HASS_TOKEN is set."""
    return bool(get_secret("HASS_TOKEN"))


# ── tool schemas ─────────────────────────────────────────────────────────────
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


for _schema, _handler in (
    (HA_LIST_ENTITIES_SCHEMA, lambda args, **kw: _dispatch(
        _async_list_entities(domain=args.get("domain"), area=args.get("area")),
        "ha_list_entities", "Failed to list entities")),
    (HA_GET_STATE_SCHEMA, _handle_get_state),
    (HA_LIST_SERVICES_SCHEMA, lambda args, **kw: _dispatch(
        _async_list_services(domain=args.get("domain")), "ha_list_services", "Failed to list services")),
    (HA_CALL_SERVICE_SCHEMA, _handle_call_service)):
    registry.register(
        name=_schema["name"], toolset="homeassistant", schema=_schema, handler=_handler,
        check_fn=_check_ha_available, emoji="🏠")
