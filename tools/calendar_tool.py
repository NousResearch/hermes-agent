#!/usr/bin/env python3
"""Canonical agent calendar tools backed by an API-first scheduler.

This module intentionally exposes an agent-native contract instead of leaking a
specific vendor UI to the model.  The first backend is Nettu Scheduler, but the
public tool names are generic so future agents can inherit the same capability
with different adapters.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict, Optional

from tools.registry import registry

DEFAULT_BASE_URL = "http://127.0.0.1:5055/api/v1"


def _base_url() -> str:
    return os.getenv("HERMES_CALENDAR_BASE_URL", DEFAULT_BASE_URL).rstrip("/")


def _api_key() -> Optional[str]:
    return (
        os.getenv("HERMES_CALENDAR_API_KEY")
        or os.getenv("NETTU_ACCOUNT_API_KEY")
        or os.getenv("ACCOUNT_API_KEY")
    )


def _headers() -> Dict[str, str]:
    headers = {"Content-Type": "application/json"}
    key = _api_key()
    if key:
        headers["x-api-key"] = key
    return headers


def _metadata(metadata: Optional[Dict[str, Any]]) -> Dict[str, str]:
    """Return Nettu-compatible metadata while keeping agent tools generic.

    The canonical tool contract accepts JSON metadata because future calendar
    adapters may store structured values. Nettu's API currently expects string
    values in its metadata map, so stringify non-scalar values at the adapter
    boundary instead of making agents learn this backend quirk.
    """
    normalized: Dict[str, str] = {}
    for key, value in (metadata or {}).items():
        if value is None:
            continue
        if isinstance(value, str):
            normalized[str(key)] = value
        elif isinstance(value, (int, float, bool)):
            normalized[str(key)] = str(value)
        else:
            normalized[str(key)] = json.dumps(value, ensure_ascii=False, sort_keys=True)
    return normalized


def check_calendar_requirements() -> bool:
    """Expose the calendar toolset only when an API key is configured.

    A local Nettu instance may not be running at schema-generation time, so do
    not probe the network here. Runtime calls return actionable errors.
    """
    return bool(_api_key())


def _request(method: str, path: str, body: Optional[Dict[str, Any]] = None,
             query: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    url = f"{_base_url()}/{path.lstrip('/')}"
    if query:
        clean_query: Dict[str, Any] = {}
        for key, value in query.items():
            if value is None:
                continue
            if isinstance(value, (list, tuple)):
                clean_query[key] = ",".join(str(v) for v in value)
            else:
                clean_query[key] = value
        if clean_query:
            url += "?" + urllib.parse.urlencode(clean_query)

    data = None
    if body is not None:
        data = json.dumps(body).encode("utf-8")

    req = urllib.request.Request(url=url, method=method.upper(), data=data, headers=_headers())
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            raw = resp.read().decode("utf-8")
            if not raw:
                payload: Any = None
            else:
                try:
                    payload = json.loads(raw)
                except json.JSONDecodeError:
                    payload = raw
            return {"success": True, "status": resp.status, "url": url, "data": payload}
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        try:
            payload = json.loads(raw) if raw else None
        except json.JSONDecodeError:
            payload = raw
        return {"success": False, "status": exc.code, "url": url, "error": payload}
    except Exception as exc:
        return {"success": False, "url": url, "error": str(exc)}


def calendar_status() -> Dict[str, Any]:
    """Check backend reachability and API-key validity.

    Nettu does not expose a generic health route under /api/v1; /account is
    the smallest authenticated endpoint and confirms the calendar backend is
    reachable with the configured tenant key.
    """
    return _request("GET", "/account")


def calendar_create_actor(metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Create a generic schedule owner/resource in the scheduler.

    In Nettu this maps to User. In SitioUno vocabulary it may represent a human,
    room, restaurant capacity bucket, doctor, service provider, or other bookable
    resource depending on the agent profile.
    """
    return _request("POST", "/user", {"metadata": _metadata(metadata)})


def calendar_find_actor_by_metadata(key: str, value: str, limit: int = 20, skip: int = 0) -> Dict[str, Any]:
    return _request("GET", "/user/meta", query={"key": key, "value": value, "limit": limit, "skip": skip})


def calendar_create_calendar(actor_id: str, timezone: str = "UTC", week_start: str = "Mon",
                             metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    return _request(
        "POST",
        f"/user/{actor_id}/calendar",
        {"timezone": timezone, "weekStart": week_start, "metadata": _metadata(metadata)},
    )


def calendar_find_calendar_by_metadata(key: str, value: str, limit: int = 20, skip: int = 0) -> Dict[str, Any]:
    return _request("GET", "/calendar/meta", query={"key": key, "value": value, "limit": limit, "skip": skip})


def calendar_create_service(metadata: Optional[Dict[str, Any]] = None,
                            multi_person: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    body: Dict[str, Any] = {"metadata": _metadata(metadata)}
    if multi_person is not None:
        body["multiPerson"] = multi_person
    return _request("POST", "/service", body)


def calendar_find_service_by_metadata(key: str, value: str, limit: int = 20, skip: int = 0) -> Dict[str, Any]:
    return _request("GET", "/service/meta", query={"key": key, "value": value, "limit": limit, "skip": skip})


def calendar_add_actor_to_service(service_id: str, actor_id: str,
                                  availability: Optional[Dict[str, Any]] = None,
                                  buffer_before: Optional[int] = None,
                                  buffer_after: Optional[int] = None,
                                  closest_booking_time: Optional[int] = None,
                                  furthest_booking_time: Optional[int] = None) -> Dict[str, Any]:
    body = {
        "userId": actor_id,
        "availability": availability,
        "bufferBefore": buffer_before,
        "bufferAfter": buffer_after,
        "closestBookingTime": closest_booking_time,
        "furthestBookingTime": furthest_booking_time,
    }
    return _request("POST", f"/service/{service_id}/users", {k: v for k, v in body.items() if v is not None})


def calendar_add_busy_calendar(service_id: str, actor_id: str, calendar_id: str,
                               provider: str = "Nettu") -> Dict[str, Any]:
    return _request(
        "PUT",
        f"/service/{service_id}/users/{actor_id}/busy",
        {"busy": {"id": calendar_id, "provider": provider}},
    )


def calendar_find_availability(service_id: str, start_date: str, end_date: str,
                               duration_minutes: int, timezone: str = "UTC",
                               interval_minutes: int = 15,
                               host_actor_ids: Optional[list[str]] = None) -> Dict[str, Any]:
    return _request(
        "GET",
        f"/service/{service_id}/booking",
        query={
            "startDate": start_date,
            "endDate": end_date,
            "duration": duration_minutes * 60 * 1000,
            "interval": interval_minutes * 60 * 1000,
            "timezone": timezone,
            "hostUserIds": host_actor_ids,
        },
    )


def calendar_create_event(actor_id: str, calendar_id: str, start_ts: int, duration_minutes: int,
                          busy: bool = True, metadata: Optional[Dict[str, Any]] = None,
                          service_id: Optional[str] = None,
                          reminders: Optional[list[Dict[str, Any]]] = None,
                          recurrence: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    body: Dict[str, Any] = {
        "calendarId": calendar_id,
        "startTs": start_ts,
        "duration": duration_minutes * 60 * 1000,
        "busy": busy,
        "metadata": _metadata(metadata),
    }
    if service_id:
        body["serviceId"] = service_id
    if reminders is not None:
        body["reminders"] = reminders
    if recurrence is not None:
        body["recurrence"] = recurrence
    return _request("POST", f"/user/{actor_id}/events", body)


def calendar_block_time(actor_id: str, calendar_id: str, start_ts: int, duration_minutes: int,
                        reason: Optional[str] = None,
                        metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    meta = _metadata(metadata)
    meta.setdefault("kind", "block")
    if reason:
        meta.setdefault("reason", reason)
    return calendar_create_event(actor_id, calendar_id, start_ts, duration_minutes, busy=True, metadata=meta)


def calendar_list_events(calendar_id: str, start_ts: int, end_ts: int,
                         actor_scope: bool = False) -> Dict[str, Any]:
    path = f"/user/calendar/{calendar_id}/events" if actor_scope else f"/calendar/{calendar_id}/events"
    result = _request("GET", path, query={"startTs": start_ts, "endTs": end_ts})
    if not actor_scope and result.get("status") in {401, 403}:
        # The Nettu account API key can create/update user-scoped events but may
        # reject the account-level calendar endpoint in local deployments. Fall
        # back to the user-scoped endpoint so the generic agent tool keeps
        # working without exposing this backend-specific distinction.
        return _request("GET", f"/user/calendar/{calendar_id}/events", query={"startTs": start_ts, "endTs": end_ts})
    return result


def calendar_update_event(event_id: str, start_ts: Optional[int] = None,
                          duration_minutes: Optional[int] = None,
                          busy: Optional[bool] = None,
                          metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    body: Dict[str, Any] = {}
    if start_ts is not None:
        body["startTs"] = start_ts
    if duration_minutes is not None:
        body["duration"] = duration_minutes * 60 * 1000
    if busy is not None:
        body["busy"] = busy
    if metadata is not None:
        body["metadata"] = _metadata(metadata)
    return _request("PUT", f"/user/events/{event_id}", body)


def calendar_cancel_event(event_id: str) -> Dict[str, Any]:
    return _request("DELETE", f"/user/events/{event_id}")


def calendar_raw_request(method: str, path: str, body: Optional[Dict[str, Any]] = None,
                         query: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Escape hatch for advanced scheduler endpoints while the canonical layer matures."""
    allowed = {"GET", "POST", "PUT", "DELETE"}
    method = method.upper()
    if method not in allowed:
        return {"success": False, "error": f"Unsupported method {method}. Use one of {sorted(allowed)}"}
    return _request(method, path, body=body, query=query)


COMMON_METADATA_DESCRIPTION = (
    "Optional JSON metadata. Keep it generic and tenant-neutral: business_id, "
    "service_type, source_channel, client_id, external_ref, labels, notes."
)


def _object_schema(properties: Dict[str, Any], required: Optional[list[str]] = None) -> Dict[str, Any]:
    return {"type": "object", "properties": properties, "required": required or []}


SCHEMAS = {
    "calendar_status": _object_schema({}),
    "calendar_create_actor": _object_schema({"metadata": {"type": "object", "description": COMMON_METADATA_DESCRIPTION}}),
    "calendar_find_actor_by_metadata": _object_schema({"key": {"type": "string"}, "value": {"type": "string"}, "limit": {"type": "integer", "default": 20}, "skip": {"type": "integer", "default": 0}}, ["key", "value"]),
    "calendar_create_calendar": _object_schema({"actor_id": {"type": "string"}, "timezone": {"type": "string", "default": "UTC"}, "week_start": {"type": "string", "default": "Mon"}, "metadata": {"type": "object", "description": COMMON_METADATA_DESCRIPTION}}, ["actor_id"]),
    "calendar_find_calendar_by_metadata": _object_schema({"key": {"type": "string"}, "value": {"type": "string"}, "limit": {"type": "integer", "default": 20}, "skip": {"type": "integer", "default": 0}}, ["key", "value"]),
    "calendar_create_service": _object_schema({"metadata": {"type": "object", "description": COMMON_METADATA_DESCRIPTION}, "multi_person": {"type": "object", "description": "Optional backend-specific multi-person/group capacity options."}}),
    "calendar_find_service_by_metadata": _object_schema({"key": {"type": "string"}, "value": {"type": "string"}, "limit": {"type": "integer", "default": 20}, "skip": {"type": "integer", "default": 0}}, ["key", "value"]),
    "calendar_add_actor_to_service": _object_schema({"service_id": {"type": "string"}, "actor_id": {"type": "string"}, "availability": {"type": "object", "description": "Backend TimePlan object, e.g. schedule-based availability."}, "buffer_before": {"type": "integer"}, "buffer_after": {"type": "integer"}, "closest_booking_time": {"type": "integer"}, "furthest_booking_time": {"type": "integer"}}, ["service_id", "actor_id"]),
    "calendar_add_busy_calendar": _object_schema({"service_id": {"type": "string"}, "actor_id": {"type": "string"}, "calendar_id": {"type": "string"}, "provider": {"type": "string", "default": "Nettu"}}, ["service_id", "actor_id", "calendar_id"]),
    "calendar_find_availability": _object_schema({"service_id": {"type": "string"}, "start_date": {"type": "string", "description": "YYYY-MM-DD"}, "end_date": {"type": "string", "description": "YYYY-MM-DD"}, "duration_minutes": {"type": "integer"}, "timezone": {"type": "string", "default": "UTC"}, "interval_minutes": {"type": "integer", "default": 15}, "host_actor_ids": {"type": "array", "items": {"type": "string"}}}, ["service_id", "start_date", "end_date", "duration_minutes"]),
    "calendar_create_event": _object_schema({"actor_id": {"type": "string"}, "calendar_id": {"type": "string"}, "start_ts": {"type": "integer", "description": "Unix timestamp in milliseconds"}, "duration_minutes": {"type": "integer"}, "busy": {"type": "boolean", "default": True}, "metadata": {"type": "object", "description": COMMON_METADATA_DESCRIPTION}, "service_id": {"type": "string"}, "reminders": {"type": "array", "items": {"type": "object"}}, "recurrence": {"type": "object"}}, ["actor_id", "calendar_id", "start_ts", "duration_minutes"]),
    "calendar_block_time": _object_schema({"actor_id": {"type": "string"}, "calendar_id": {"type": "string"}, "start_ts": {"type": "integer", "description": "Unix timestamp in milliseconds"}, "duration_minutes": {"type": "integer"}, "reason": {"type": "string"}, "metadata": {"type": "object", "description": COMMON_METADATA_DESCRIPTION}}, ["actor_id", "calendar_id", "start_ts", "duration_minutes"]),
    "calendar_list_events": _object_schema({"calendar_id": {"type": "string"}, "start_ts": {"type": "integer"}, "end_ts": {"type": "integer"}, "actor_scope": {"type": "boolean", "default": False}}, ["calendar_id", "start_ts", "end_ts"]),
    "calendar_update_event": _object_schema({"event_id": {"type": "string"}, "start_ts": {"type": "integer"}, "duration_minutes": {"type": "integer"}, "busy": {"type": "boolean"}, "metadata": {"type": "object", "description": COMMON_METADATA_DESCRIPTION}}, ["event_id"]),
    "calendar_cancel_event": _object_schema({"event_id": {"type": "string"}}, ["event_id"]),
    "calendar_raw_request": _object_schema({"method": {"type": "string", "enum": ["GET", "POST", "PUT", "DELETE"]}, "path": {"type": "string"}, "body": {"type": "object"}, "query": {"type": "object"}}, ["method", "path"]),
}

HANDLERS = {
    "calendar_status": lambda args, **kw: calendar_status(),
    "calendar_create_actor": lambda args, **kw: calendar_create_actor(args.get("metadata")),
    "calendar_find_actor_by_metadata": lambda args, **kw: calendar_find_actor_by_metadata(args["key"], args["value"], args.get("limit", 20), args.get("skip", 0)),
    "calendar_create_calendar": lambda args, **kw: calendar_create_calendar(args["actor_id"], args.get("timezone", "UTC"), args.get("week_start", "Mon"), args.get("metadata")),
    "calendar_find_calendar_by_metadata": lambda args, **kw: calendar_find_calendar_by_metadata(args["key"], args["value"], args.get("limit", 20), args.get("skip", 0)),
    "calendar_create_service": lambda args, **kw: calendar_create_service(args.get("metadata"), args.get("multi_person")),
    "calendar_find_service_by_metadata": lambda args, **kw: calendar_find_service_by_metadata(args["key"], args["value"], args.get("limit", 20), args.get("skip", 0)),
    "calendar_add_actor_to_service": lambda args, **kw: calendar_add_actor_to_service(args["service_id"], args["actor_id"], args.get("availability"), args.get("buffer_before"), args.get("buffer_after"), args.get("closest_booking_time"), args.get("furthest_booking_time")),
    "calendar_add_busy_calendar": lambda args, **kw: calendar_add_busy_calendar(args["service_id"], args["actor_id"], args["calendar_id"], args.get("provider", "Nettu")),
    "calendar_find_availability": lambda args, **kw: calendar_find_availability(args["service_id"], args["start_date"], args["end_date"], args["duration_minutes"], args.get("timezone", "UTC"), args.get("interval_minutes", 15), args.get("host_actor_ids")),
    "calendar_create_event": lambda args, **kw: calendar_create_event(args["actor_id"], args["calendar_id"], args["start_ts"], args["duration_minutes"], args.get("busy", True), args.get("metadata"), args.get("service_id"), args.get("reminders"), args.get("recurrence")),
    "calendar_block_time": lambda args, **kw: calendar_block_time(args["actor_id"], args["calendar_id"], args["start_ts"], args["duration_minutes"], args.get("reason"), args.get("metadata")),
    "calendar_list_events": lambda args, **kw: calendar_list_events(args["calendar_id"], args["start_ts"], args["end_ts"], args.get("actor_scope", False)),
    "calendar_update_event": lambda args, **kw: calendar_update_event(args["event_id"], args.get("start_ts"), args.get("duration_minutes"), args.get("busy"), args.get("metadata")),
    "calendar_cancel_event": lambda args, **kw: calendar_cancel_event(args["event_id"]),
    "calendar_raw_request": lambda args, **kw: calendar_raw_request(args["method"], args["path"], args.get("body"), args.get("query")),
}

DESCRIPTIONS = {
    "calendar_status": "Check the configured scheduler backend status.",
    "calendar_create_actor": "Create a generic bookable actor/resource for the current tenant.",
    "calendar_find_actor_by_metadata": "Find generic actors/resources by metadata.",
    "calendar_create_calendar": "Create a calendar for a generic actor/resource.",
    "calendar_find_calendar_by_metadata": "Find calendars by metadata.",
    "calendar_create_service": "Create a bookable service type, such as consultation, demo, table reservation, or visit.",
    "calendar_find_service_by_metadata": "Find bookable services by metadata.",
    "calendar_add_actor_to_service": "Make an actor/resource bookable for a service.",
    "calendar_add_busy_calendar": "Attach a calendar as busy-time source for a service actor/resource.",
    "calendar_find_availability": "Find available booking slots for a service.",
    "calendar_create_event": "Create a scheduled event/booking/block on a calendar.",
    "calendar_block_time": "Block time on a calendar so it cannot be booked.",
    "calendar_list_events": "List events on a calendar within a timestamp window.",
    "calendar_update_event": "Update or reschedule an event.",
    "calendar_cancel_event": "Cancel/delete an event.",
    "calendar_raw_request": "Advanced escape hatch for backend endpoints not yet covered by canonical tools.",
}

for _name, _parameters in SCHEMAS.items():
    registry.register(
        name=_name,
        toolset="calendar",
        schema={"name": _name, "description": DESCRIPTIONS[_name], "parameters": _parameters},
        handler=HANDLERS[_name],
        check_fn=check_calendar_requirements,
        emoji="📅",
    )
