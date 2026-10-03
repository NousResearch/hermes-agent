"""Home Assistant tool for controlling smart home devices via REST API.

Registers ``ha_list_entities``, ``ha_get_state``, ``ha_list_services``, ``ha_call_service``.
Auth is a Long-Lived Access Token (``HASS_TOKEN``); the instance URL comes from
``HASS_URL`` (default http://homeassistant.local:8123).
"""

import asyncio
import fnmatch
import json
import logging
import re
from typing import Any, Dict, Optional, Tuple

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


# ── ha_list_entities filtering ───────────────────────────────────────────────
# A domain+area query can still return dozens of entities — the issue's example
# ("sensor" in "Attic") returns 72 entities / ~9 KB when the caller wanted a few
# temperature readings. Every filter below is optional and case-insensitive, so
# the original domain/area calls keep working unchanged.


def _attr_value(state: dict, name: str) -> Any:
    """Read one attribute, tolerating states without an ``attributes`` dict."""
    attributes = state.get("attributes")
    return (attributes or {}).get(name) if isinstance(attributes, dict) else None


def _token_list(value: Any) -> list:
    """Split a filter value into lowercased, comma-separated, non-empty tokens."""
    if value is None:
        return []
    return [token.strip().lower() for token in str(value).split(",") if token.strip()]


def _matches_entity_id(entity_id: str, pattern: str) -> bool:
    """True when ``entity_id`` matches one user-supplied pattern.

    Accepts an exact id (``sensor.attic_temperature``), a partial object id
    (``sensor.attic`` or ``attic`` also match ``sensor.attic_temperature``), or
    a glob (``sensor.attic_temperature_*``).
    """
    candidate = str(entity_id or "").strip().lower()
    wanted = str(pattern or "").strip().lower()
    if not wanted:
        return True
    if not candidate:
        return False
    if any(ch in wanted for ch in "*?["):
        return fnmatch.fnmatchcase(candidate, wanted)
    if candidate == wanted:
        return True
    domain, _, object_id = candidate.partition(".")
    if "." in wanted:
        wanted_domain, _, wanted_object = wanted.partition(".")
    else:
        wanted_domain, wanted_object = "", wanted
    if wanted_domain and wanted_domain != domain:
        return False
    return object_id == wanted_object or object_id.startswith(f"{wanted_object}_")


def _area_slug(area: str) -> str:
    """Normalize an area name for slug matching: ``"Living Room"`` -> ``"living_room"``."""
    return re.sub(r"[^a-z0-9]+", "_", str(area or "").strip().lower()).strip("_")


def _matches_area(state: dict, area_lower: str) -> bool:
    """True when a state belongs to ``area_lower``.

    Matches the friendly name, the entity's ``area`` attribute, and — because
    area names are often absent from friendly names (``sensor.attic_co2`` is
    called "CO2 Level") — the area slug inside the entity_id.
    """
    friendly = str(_attr_value(state, "friendly_name") or "").lower()
    attribute_area = str(_attr_value(state, "area") or "").lower()
    if area_lower in friendly or area_lower in attribute_area:
        return True
    slug = _area_slug(area_lower)
    return bool(slug) and slug in str(state.get("entity_id") or "").lower()


def _matches_state_value(value: Any, wanted: list) -> bool:
    """True when a state string equals one token, or matches one glob token."""
    candidate = str(value or "").strip().lower()
    for token in wanted:
        if candidate == token:
            return True
        if any(ch in token for ch in "*?[") and fnmatch.fnmatchcase(candidate, token):
            return True
    return False


def _summarize_state(state: dict) -> Dict[str, Any]:
    """Compact one entity for the model.

    ``device_class``/``unit_of_measurement`` are included when present so the
    model can interpret (and use) the first result without a follow-up
    ``ha_get_state`` call; they are omitted otherwise to keep results small.
    """
    entity: Dict[str, Any] = {
        "entity_id": state.get("entity_id", ""),
        "state": state.get("state", ""),
        "friendly_name": _attr_value(state, "friendly_name") or ""}
    device_class = _attr_value(state, "device_class")
    if device_class:
        entity["device_class"] = device_class
    unit = _attr_value(state, "unit_of_measurement")
    if unit:
        entity["unit_of_measurement"] = unit
    return entity


def _filter_and_summarize(
    states: list,
    domain: Optional[str] = None,
    area: Optional[str] = None,
    device_class: Optional[str] = None,
    name: Optional[str] = None,
    entity_id: Optional[str] = None,
    state: Optional[str] = None,
    limit: Optional[int] = None,
) -> Dict:
    """Filter raw HA states by the optional filters and compact the survivors.

    ``device_class``/``state``/``entity_id`` accept comma-separated lists
    (any-of). ``limit`` caps the returned entities *after* filtering and adds
    ``total_matched``/``truncated`` so the caller knows the result was cut.
    """
    if domain:
        prefix = f"{str(domain).strip().lower()}."
        states = [s for s in states if str(s.get("entity_id") or "").lower().startswith(prefix)]
    entity_id_tokens = _token_list(entity_id)
    if entity_id_tokens:
        states = [
            s for s in states
            if any(_matches_entity_id(s.get("entity_id", ""), token) for token in entity_id_tokens)]
    device_class_tokens = _token_list(device_class)
    if device_class_tokens:
        states = [
            s for s in states
            if str(_attr_value(s, "device_class") or "").strip().lower() in device_class_tokens]
    state_tokens = _token_list(state)
    if state_tokens:
        states = [s for s in states if _matches_state_value(s.get("state"), state_tokens)]
    if name:
        needle = str(name).strip().lower()
        states = [
            s for s in states
            if needle in str(_attr_value(s, "friendly_name") or "").lower()]
    if area:
        area_lower = str(area).strip().lower()
        states = [s for s in states if _matches_area(s, area_lower)]

    matched = len(states)
    truncated = False
    if limit is not None and matched > limit:
        states = states[:limit]
        truncated = True
    entities = [_summarize_state(s) for s in states]
    result: Dict[str, Any] = {"count": len(entities), "entities": entities}
    if limit is not None:
        result["total_matched"] = matched
        result["truncated"] = truncated
    return result


def _parse_list_limit(value: Any) -> Tuple[Optional[int], Optional[str]]:
    """Validate the optional ``limit`` filter; returns ``(limit, error_message)``."""
    if value is None:
        return None, None
    if isinstance(value, bool):
        return None, f"Invalid limit: expected a positive integer, got {value!r}"
    if isinstance(value, str):
        text = value.strip()
        if not text.lstrip("+").isdigit():
            return None, f"Invalid limit: expected a positive integer, got {value!r}"
        numeric = int(text)
    elif isinstance(value, int):
        numeric = value
    elif isinstance(value, float) and float(value).is_integer():
        numeric = int(value)
    else:
        return None, f"Invalid limit: expected a positive integer, got {type(value).__name__}"
    if numeric < 1:
        return None, f"Invalid limit: must be >= 1, got {numeric}"
    return numeric, None


async def _async_list_entities(
    domain: Optional[str] = None,
    area: Optional[str] = None,
    device_class: Optional[str] = None,
    name: Optional[str] = None,
    entity_id: Optional[str] = None,
    state: Optional[str] = None,
    limit: Optional[int] = None,
) -> Dict[str, Any]:
    return _filter_and_summarize(
        await _api_json("GET", "/api/states", 15),
        domain=domain, area=area, device_class=device_class, name=name,
        entity_id=entity_id, state=state, limit=limit)


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
        import contextvars
        # Carry the caller's Context: the coroutine reads HASS_* via get_secret, which is profile-scoped.
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(contextvars.copy_context().run, asyncio.run, coro).result(timeout=30)
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


# Filters that are plain case-insensitive strings; anything else (dict/list) is
# a caller bug that would silently do nothing, so it is rejected instead.
_LIST_TEXT_FILTERS = ("domain", "area", "device_class", "name", "search", "entity_id", "state")


def _handle_list_entities(args: dict, **kw) -> str:
    """Validate the optional ha_list_entities filters, then list the matches."""
    filters: Dict[str, Any] = {}
    for key in _LIST_TEXT_FILTERS:
        value = args.get(key)
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, (str, int, float)):
            return tool_error(f"Invalid {key} filter: expected a string, got {type(value).__name__}")
        text = str(value).strip()
        if text:
            filters[key] = text
    limit, error = _parse_list_limit(args.get("limit"))
    if error:
        return tool_error(error)
    return _dispatch(
        _async_list_entities(
            domain=filters.get("domain"),
            area=filters.get("area"),
            device_class=filters.get("device_class"),
            name=filters.get("name") or filters.get("search"),
            entity_id=filters.get("entity_id"),
            state=filters.get("state"),
            limit=limit),
        "ha_list_entities", "Failed to list entities")


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
        "List Home Assistant entities, optionally filtered by domain "
        "(light, switch, climate, sensor, binary_sensor, cover, fan, etc.), area name "
        "(living room, kitchen, bedroom, etc.), device_class (temperature, humidity, motion, "
        "battery, ...), a friendly-name substring, entity_id, or state. Combine filters and "
        "set limit to keep the result small; when a limit is applied the response also "
        "reports total_matched/truncated."
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
                    "Area/room name to filter by (e.g. 'living room', 'kitchen', 'attic'). "
                    "Matches against the entity's friendly name, its area attribute, and the "
                    "area slug in its entity_id. Omit to list all."
                ),
            },
            "device_class": {
                "type": "string",
                "description": (
                    "Only entities with this device_class attribute (e.g. 'temperature', "
                    "'humidity', 'motion', 'battery'). Comma-separate for several "
                    "(e.g. 'temperature,humidity')."
                ),
            },
            "name": {
                "type": "string",
                "description": (
                    "Case-insensitive substring matched against the entity friendly name "
                    "(e.g. 'attic temperature'). Narrow it down when a domain/area query "
                    "returns too many entities."
                ),
            },
            "entity_id": {
                "type": "string",
                "description": (
                    "Entity ID to match (e.g. 'sensor.attic_temperature'). A partial id "
                    "('sensor.attic') and '*' globs ('sensor.attic_*') are accepted; "
                    "comma-separate for several."
                ),
            },
            "state": {
                "type": "string",
                "description": (
                    "Only entities currently in this state (e.g. 'on', 'off', 'open', "
                    "'unavailable'). Comma-separate for several (e.g. 'on,open')."
                ),
            },
            "limit": {
                "type": "integer",
                "description": (
                    "Maximum number of entities to return after filtering (e.g. 5). "
                    "Omit to return every match."
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
    (HA_LIST_ENTITIES_SCHEMA, _handle_list_entities),
    (HA_GET_STATE_SCHEMA, _handle_get_state),
    (HA_LIST_SERVICES_SCHEMA, lambda args, **kw: _dispatch(
        _async_list_services(domain=args.get("domain")), "ha_list_services", "Failed to list services")),
    (HA_CALL_SERVICE_SCHEMA, _handle_call_service)):
    registry.register(
        name=_schema["name"], toolset="homeassistant", schema=_schema, handler=_handler,
        check_fn=_check_ha_available, emoji="🏠")
