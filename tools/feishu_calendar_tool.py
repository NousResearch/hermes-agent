"""Feishu Calendar Tool -- manage Feishu/Lark calendar events via API.

Provides the following tools:
  - feishu_calendar_list        : List calendars or get primary calendar
  - feishu_calendar_events     : List events in a calendar
  - feishu_calendar_create_event : Create a calendar event
  - feishu_calendar_update_event : Update an existing event
  - feishu_calendar_delete_event : Delete an event
  - feishu_calendar_attendees  : List or add attendees to an event
"""

import json
import logging
from datetime import datetime, timezone, timedelta
from typing import Any, Optional

from tools.registry import registry, tool_error, tool_result

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Client helpers
# ---------------------------------------------------------------------------

_thread_local_client = None


def set_client(client) -> None:
    """Store a lark client for the current thread (called by feishu_comment)."""
    global _thread_local_client
    _thread_local_client = client


def _get_client():
    """Return the lark client for the current thread, or None."""
    return _thread_local_client


def _get_user_request_option():
    """Return a RequestOption with the current user's Feishu access token, or None."""
    try:
        from gateway.session_context import get_session_env
        from tools import feishu_oauth

        open_id = get_session_env("HERMES_SESSION_USER_ID", "")
        if not open_id:
            return None
        store = feishu_oauth.FeishuUserTokenStore()
        return store.get_request_option(open_id)
    except Exception:
        return None


def _check_feishu():
    """Return True if lark_oapi is installed."""
    try:
        import lark_oapi  # noqa: F401
        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# Time helpers
# ---------------------------------------------------------------------------

def _parse_iso8601(ts: str) -> Optional[datetime]:
    """Parse an ISO8601 timestamp string to datetime (UTC)."""
    if not ts:
        return None
    try:
        # Handle 'Z' suffix
        ts_clean = ts.replace("Z", "+00:00")
        dt = datetime.fromisoformat(ts_clean)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except (ValueError, OSError):
        return None


def _to_unix_ts(dt: datetime) -> str:
    """Convert a datetime to Unix timestamp string (seconds)."""
    return str(int(dt.timestamp()))


# ---------------------------------------------------------------------------
# Response formatters
# ---------------------------------------------------------------------------

def _format_calendar_list(data: dict) -> str:
    """Format calendar list response."""
    items = data.get("calendar_list", [])
    if not items:
        return "No calendars found."

    lines = ["Calendars:"]
    for item in items:
        cal = item.get("calendar", {})
        cal_id = cal.get("calendar_id", "")
        summary = cal.get("summary", "(no title)")
        alias = cal.get("summary_alias", "")
        cal_type = cal.get("type", "")
        line = f"  - [{cal_id}] {summary}"
        if alias:
            line += f" ({alias})"
        lines.append(line)
        if cal_type:
            lines.append(f"    type: {cal_type}")
    return "\n".join(lines)


def _format_event_list(data: dict) -> str:
    """Format event list response."""
    items = data.get("items", [])
    if not items:
        return "No events found."

    lines = ["Events:"]
    for event in items:
        event_id = event.get("event_id", "")
        summary = event.get("summary", "(no title)")
        status = event.get("status", "")
        start = event.get("start_time", {})
        end = event.get("end_time", {})

        start_str = ""
        end_str = ""
        if isinstance(start, dict):
            ts = start.get("timestamp", "")
            if ts:
                try:
                    dt = datetime.fromtimestamp(int(ts), tz=timezone.utc)
                    start_str = dt.strftime("%Y-%m-%d %H:%M")
                except (ValueError, OSError):
                    start_str = ts
        if isinstance(end, dict):
            ts = end.get("timestamp", "")
            if ts:
                try:
                    dt = datetime.fromtimestamp(int(ts), tz=timezone.utc)
                    end_str = dt.strftime("%Y-%m-%d %H:%M")
                except (ValueError, OSError):
                    end_str = ts

        line = f"  - [{event_id}] {summary}"
        if start_str and end_str:
            line += f" | {start_str} ~ {end_str}"
        if status:
            line += f" | {status}"
        lines.append(line)

    return "\n".join(lines)


def _format_event(data: dict) -> str:
    """Format single event response."""
    event = data.get("event", data)
    event_id = event.get("event_id", "")
    summary = event.get("summary", "(no title)")
    status = event.get("status", "")

    lines = [f"Event: {summary}"]
    if event_id:
        lines.append(f"  ID: {event_id}")
    if status:
        lines.append(f"  Status: {status}")

    # Format times
    for key in ("start_time", "end_time"):
        t = event.get(key, {})
        if isinstance(t, dict):
            ts = t.get("timestamp", "")
            if ts:
                try:
                    dt = datetime.fromtimestamp(int(ts), tz=timezone.utc)
                    lines.append(f"  {key}: {dt.strftime('%Y-%m-%d %H:%M')}")
                except (ValueError, OSError):
                    lines.append(f"  {key}: {ts}")

    # Format description
    desc = event.get("description", "")
    if desc:
        lines.append(
            f"  Description: {desc[:200]}" + ("..." if len(desc) > 200 else "")
        )

    # Format location
    loc = event.get("location", {})
    if isinstance(loc, dict):
        name = loc.get("name", "")
        if name:
            lines.append(f"  Location: {name}")

    return "\n".join(lines)


def _format_attendees(data: dict) -> str:
    """Format attendees list response."""
    items = data.get("items", [])
    if not items:
        return "No attendees found."

    lines = ["Attendees:"]
    for attendee in items:
        att_id = attendee.get("attendee_id", {})
        uid = ""
        if isinstance(att_id, dict):
            uid = att_id.get("user_id", "")
        name = attendee.get("display_name", uid or "(unknown)")
        att_type = attendee.get("type", "")
        rsvp = attendee.get("rsvp_status", "")
        line = f"  - {name}" + (f" ({uid})" if uid else "")
        lines.append(line)
        if att_type:
            lines.append(f"    type: {att_type}")
        if rsvp:
            lines.append(f"    rsvp: {rsvp}")

    return "\n".join(lines)


def _check_response_error(response) -> Optional[str]:
    """Check response for errors. Returns error message or None if OK."""
    code = getattr(response, "code", None)
    if code != 0:
        msg = getattr(response, "msg", "unknown error")
        return f"Calendar API error: code={code} msg={msg}"
    return None


# ---------------------------------------------------------------------------
# Tool schemas
# ---------------------------------------------------------------------------

FEISHU_CALENDAR_LIST_SCHEMA = {
    "name": "feishu_calendar_list",
    "description": (
        "List the user's calendars or get a specific calendar by ID. "
        "Use 'primary' as calendar_id to get the user's primary calendar."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "calendar_id": {
                "type": "string",
                "description": (
                    "Calendar ID. Use 'primary' for the user's primary calendar. "
                    "Omit to list all calendars the user has access to."
                ),
            },
        },
        "required": [],
    },
}

FEISHU_CALENDAR_EVENTS_SCHEMA = {
    "name": "feishu_calendar_events",
    "description": (
        "List events in a Feishu calendar within a time range. "
        "Defaults to now - 7 days to now + 30 days if start_time/end_time are not specified."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "calendar_id": {
                "type": "string",
                "description": "The calendar ID (use 'primary' for the user's primary calendar).",
            },
            "start_time": {
                "type": "string",
                "description": (
                    "Start of time range in ISO8601 format (e.g. '2026-04-01T00:00:00Z'). "
                    "Defaults to 7 days ago."
                ),
            },
            "end_time": {
                "type": "string",
                "description": (
                    "End of time range in ISO8601 format (e.g. '2026-04-30T23:59:59Z'). "
                    "Defaults to 30 days from now."
                ),
            },
            "limit": {
                "type": "integer",
                "description": "Maximum number of events to return (default 50, max 500). Note: Feishu API minimum is 50.",
                "minimum": 1,
                "maximum": 500,
                "default": 50,
            },
        },
        "required": ["calendar_id"],
    },
}

FEISHU_CALENDAR_CREATE_EVENT_SCHEMA = {
    "name": "feishu_calendar_create_event",
    "description": (
        "Create a new event in a Feishu calendar. "
        "All times must be in ISO8601 format with timezone (e.g. '2026-04-01T09:00:00+08:00')."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "calendar_id": {
                "type": "string",
                "description": "The calendar ID (use 'primary' for the user's primary calendar).",
            },
            "summary": {
                "type": "string",
                "description": "Event title/summary.",
            },
            "start_time": {
                "type": "string",
                "description": "Event start time in ISO8601 format (required).",
            },
            "end_time": {
                "type": "string",
                "description": "Event end time in ISO8601 format (required).",
            },
            "description": {
                "type": "string",
                "description": "Event description/notes.",
            },
            "location": {
                "type": "string",
                "description": "Event location name.",
            },
            "attendees": {
                "type": "array",
                "description": (
                    "List of attendees to invite. Each attendee should have 'type' "
                    "(e.g. 'user') and 'user_id' (e.g. 'ou_xxx')."
                ),
                "items": {
                    "type": "object",
                    "properties": {
                        "type": {"type": "string"},
                        "user_id": {"type": "string"},
                    },
                    "required": ["type", "user_id"],
                },
            },
        },
        "required": ["calendar_id", "summary", "start_time", "end_time"],
    },
}

FEISHU_CALENDAR_UPDATE_EVENT_SCHEMA = {
    "name": "feishu_calendar_update_event",
    "description": (
        "Update an existing calendar event. All fields are optional -- only provided "
        "fields will be updated (partial update supported)."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "calendar_id": {
                "type": "string",
                "description": "The calendar ID (use 'primary' for the user's primary calendar).",
            },
            "event_id": {
                "type": "string",
                "description": "The event ID to update.",
            },
            "summary": {
                "type": "string",
                "description": "New event title/summary.",
            },
            "start_time": {
                "type": "string",
                "description": "New start time in ISO8601 format.",
            },
            "end_time": {
                "type": "string",
                "description": "New end time in ISO8601 format.",
            },
            "description": {
                "type": "string",
                "description": "New event description.",
            },
            "location": {
                "type": "string",
                "description": "New event location.",
            },
        },
        "required": ["calendar_id", "event_id"],
    },
}

FEISHU_CALENDAR_DELETE_EVENT_SCHEMA = {
    "name": "feishu_calendar_delete_event",
    "description": "Delete a calendar event by ID.",
    "parameters": {
        "type": "object",
        "properties": {
            "calendar_id": {
                "type": "string",
                "description": "The calendar ID (use 'primary' for the user's primary calendar).",
            },
            "event_id": {
                "type": "string",
                "description": "The event ID to delete.",
            },
        },
        "required": ["calendar_id", "event_id"],
    },
}

FEISHU_CALENDAR_ATTENDEES_SCHEMA = {
    "name": "feishu_calendar_attendees",
    "description": (
        "List attendees of a calendar event, or add new attendees. "
        "Provide the 'attendees' parameter to add attendees; omit it to list existing attendees."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "calendar_id": {
                "type": "string",
                "description": "The calendar ID (use 'primary' for the user's primary calendar).",
            },
            "event_id": {
                "type": "string",
                "description": "The event ID.",
            },
            "attendees": {
                "type": "array",
                "description": (
                    "List of attendees to add. Each attendee should have 'type' "
                    "(e.g. 'user') and 'user_id' (e.g. 'ou_xxx')."
                ),
                "items": {
                    "type": "object",
                    "properties": {
                        "type": {"type": "string"},
                        "user_id": {"type": "string"},
                    },
                    "required": ["type", "user_id"],
                },
            },
            "list_only": {
                "type": "boolean",
                "description": (
                    "If True, only list attendees (do not add). "
                    "If False and attendees provided, add the attendees. Default True."
                ),
                "default": True,
            },
        },
        "required": ["calendar_id", "event_id"],
    },
}


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------

def _handle_feishu_calendar_list(args: dict, **kwargs) -> str:
    """Handle feishu_calendar_list tool calls."""
    client = _get_client()
    if client is None:
        return tool_error(
            "Feishu client not available (not in a Feishu comment context). "
            "Make sure this tool is called within an active Feishu session."
        )

    calendar_id = args.get("calendar_id", "").strip()

    try:
        from lark_oapi.api.calendar.v4 import (
            ListCalendarRequestBuilder,
            GetCalendarRequestBuilder,
        )
    except ImportError:
        return tool_error("lark_oapi not installed. Install with: pip install lark-oapi")

    if calendar_id == "primary" or not calendar_id:
        # Get primary calendar
        request = (
            GetCalendarRequestBuilder()
            .calendar_id("primary")
            .build()
        )
    else:
        # List all calendars
        request = (
            ListCalendarRequestBuilder()
            .page_size(50)
            .build()
        )

    response = client.request(request)
    err = _check_response_error(response)
    if err:
        return tool_error(err)

    data = getattr(response, "data", None)
    if data is None:
        return tool_error("No data returned from calendar API")

    formatted = _format_calendar_list(data if isinstance(data, dict) else {})
    return tool_result(success=True, content=formatted)


def _handle_feishu_calendar_events(args: dict, **kwargs) -> str:
    """Handle feishu_calendar_events tool calls."""
    calendar_id = args.get("calendar_id", "").strip()
    if not calendar_id:
        return tool_error("calendar_id is required")

    client = _get_client()
    if client is None:
        return tool_error("Feishu client not available")

    start_time_str = args.get("start_time", "").strip()
    end_time_str = args.get("end_time", "").strip()
    limit = args.get("limit", 50)
    try:
        limit = max(1, min(500, int(limit)))
    except (ValueError, TypeError):
        limit = 50

    now = datetime.now(timezone.utc)
    if start_time_str:
        start_dt = _parse_iso8601(start_time_str)
        if start_dt is None:
            return tool_error(
                f"Invalid start_time format: '{start_time_str}'. "
                "Use ISO8601 (e.g. '2026-04-01T00:00:00Z')."
            )
    else:
        start_dt = now - timedelta(days=7)

    if end_time_str:
        end_dt = _parse_iso8601(end_time_str)
        if end_dt is None:
            return tool_error(
                f"Invalid end_time format: '{end_time_str}'. "
                "Use ISO8601 (e.g. '2026-04-30T23:59:59Z')."
            )
    else:
        end_dt = now + timedelta(days=30)

    if start_dt >= end_dt:
        return tool_error("start_time must be before end_time")

    try:
        from lark_oapi.api.calendar.v4 import ListCalendarEventRequestBuilder
    except ImportError:
        return tool_error("lark_oapi not installed")

    request = (
        ListCalendarEventRequestBuilder()
        .calendar_id(calendar_id)
        .start_time(_to_unix_ts(start_dt))
        .end_time(_to_unix_ts(end_dt))
        .page_size(limit)
        .build()
    )

    response = client.request(request)
    err = _check_response_error(response)
    if err:
        return tool_error(err)

    data = getattr(response, "data", None)
    if data is None:
        return tool_error("No data returned from calendar API")

    formatted = _format_event_list(data if isinstance(data, dict) else {})
    return tool_result(success=True, content=formatted)


def _handle_feishu_calendar_create_event(args: dict, **kwargs) -> str:
    """Handle feishu_calendar_create_event tool calls."""
    calendar_id = args.get("calendar_id", "").strip()
    summary = args.get("summary", "").strip()
    start_time_str = args.get("start_time", "").strip()
    end_time_str = args.get("end_time", "").strip()
    description = args.get("description", "").strip()
    location = args.get("location", "").strip()
    attendees = args.get("attendees", [])

    # Validate required fields
    missing = []
    if not calendar_id:
        missing.append("calendar_id")
    if not summary:
        missing.append("summary")
    if not start_time_str:
        missing.append("start_time")
    if not end_time_str:
        missing.append("end_time")
    if missing:
        return tool_error(f"Missing required fields: {', '.join(missing)}")

    # Parse times
    start_dt = _parse_iso8601(start_time_str)
    if start_dt is None:
        return tool_error(
            f"Invalid start_time format: '{start_time_str}'. "
            "Use ISO8601 (e.g. '2026-04-01T09:00:00+08:00')."
        )

    end_dt = _parse_iso8601(end_time_str)
    if end_dt is None:
        return tool_error(
            f"Invalid end_time format: '{end_time_str}'. "
            "Use ISO8601 (e.g. '2026-04-01T10:00:00+08:00')."
        )

    if start_dt >= end_dt:
        return tool_error("start_time must be before end_time")

    client = _get_client()
    if client is None:
        return tool_error("Feishu client not available")

    try:
        from lark_oapi.api.calendar.v4 import (
            CreateCalendarEventRequestBuilder,
            CreateCalendarEventAttendeeRequestBuilder,
            CreateCalendarEventAttendeeRequestBodyBuilder,
            CalendarEventBuilder,
            EventTimeBuilder,
            EventLocationBuilder,
            CalendarEventAttendeeBuilder,
            CalendarEventAttendeeIdBuilder,
        )
    except ImportError:
        return tool_error("lark_oapi not installed")

    # Build event body
    event_builder = (
        CalendarEventBuilder()
        .summary(summary)
        .start_time(
            EventTimeBuilder()
            .time_stamp(_to_unix_ts(start_dt))
            .build()
        )
        .end_time(
            EventTimeBuilder()
            .time_stamp(_to_unix_ts(end_dt))
            .build()
        )
    )

    if description:
        event_builder.description(description)
    if location:
        event_builder.location(EventLocationBuilder().name(location).build())

    event_body = event_builder.build()

    # Build create request
    request = (
        CreateCalendarEventRequestBuilder()
        .calendar_id(calendar_id)
        .request_body(event_body)
        .build()
    )

    user_option = _get_user_request_option()
    response = client.request(request, user_option) if user_option else client.request(request)
    err = _check_response_error(response)
    if err:
        return tool_error(err)

    data = getattr(response, "data", None)
    if data is None:
        return tool_error("No data returned from calendar API")

    data_dict = data if isinstance(data, dict) else {"event": data}
    event_id = data_dict.get("event", {}).get("event_id", "") if isinstance(data_dict, dict) else ""

    # Add attendees if provided
    if attendees and event_id:
        att_items = []
        for att in attendees:
            att_type = att.get("type", "user")
            user_id = att.get("user_id", "")
            if user_id:
                att_id = CalendarEventAttendeeIdBuilder().user_id(user_id).build()
                att_item = (
                    CalendarEventAttendeeBuilder()
                    .attendee_id(att_id)
                    .type(att_type)
                    .build()
                )
                att_items.append(att_item)

        if att_items:
            att_request = (
                CreateCalendarEventAttendeeRequestBuilder()
                .calendar_id(calendar_id)
                .event_id(event_id)
                .request_body(
                    CreateCalendarEventAttendeeRequestBodyBuilder()
                    .attendees(att_items)
                    .build()
                )
                .build()
            )
            client.request(att_request)

    formatted = _format_event(data_dict)
    return tool_result(success=True, content=formatted)


def _handle_feishu_calendar_update_event(args: dict, **kwargs) -> str:
    """Handle feishu_calendar_update_event tool calls."""
    calendar_id = args.get("calendar_id", "").strip()
    event_id = args.get("event_id", "").strip()

    if not calendar_id:
        return tool_error("calendar_id is required")
    if not event_id:
        return tool_error("event_id is required")

    summary = args.get("summary", "").strip()
    start_time_str = args.get("start_time", "").strip()
    end_time_str = args.get("end_time", "").strip()
    description = args.get("description", "").strip()
    location = args.get("location", "").strip()

    has_update = bool(summary or start_time_str or end_time_str or description or location)
    if not has_update:
        return tool_error(
            "At least one field must be provided to update: "
            "summary, start_time, end_time, description, or location"
        )

    # Parse times
    start_dt = None
    end_dt = None
    if start_time_str:
        start_dt = _parse_iso8601(start_time_str)
        if start_dt is None:
            return tool_error(f"Invalid start_time format: '{start_time_str}'.")
    if end_time_str:
        end_dt = _parse_iso8601(end_time_str)
        if end_dt is None:
            return tool_error(f"Invalid end_time format: '{end_time_str}'.")
    if start_dt and end_dt and start_dt >= end_dt:
        return tool_error("start_time must be before end_time")

    client = _get_client()
    if client is None:
        return tool_error("Feishu client not available")

    try:
        from lark_oapi.api.calendar.v4 import (
            PatchCalendarEventRequestBuilder,
            CalendarEventBuilder,
            EventTimeBuilder,
            EventLocationBuilder,
        )
    except ImportError:
        return tool_error("lark_oapi not installed")

    event_builder = CalendarEventBuilder()

    if summary:
        event_builder.summary(summary)
    if start_dt:
        event_builder.start_time(
            EventTimeBuilder()
            .time_stamp(_to_unix_ts(start_dt))
            .build()
        )
    if end_dt:
        event_builder.end_time(
            EventTimeBuilder()
            .time_stamp(_to_unix_ts(end_dt))
            .build()
        )
    if description:
        event_builder.description(description)
    if location:
        event_builder.location(EventLocationBuilder().name(location).build())

    event_body = event_builder.build()

    request = (
        PatchCalendarEventRequestBuilder()
        .calendar_id(calendar_id)
        .event_id(event_id)
        .request_body(event_body)
        .build()
    )

    user_option = _get_user_request_option()
    response = client.request(request, user_option) if user_option else client.request(request)
    err = _check_response_error(response)
    if err:
        return tool_error(err)

    data = getattr(response, "data", None)
    if data is None:
        return tool_error("No data returned from calendar API")

    data_dict = data if isinstance(data, dict) else {"event": data}
    formatted = _format_event(data_dict)
    return tool_result(success=True, content=formatted)


def _handle_feishu_calendar_delete_event(args: dict, **kwargs) -> str:
    """Handle feishu_calendar_delete_event tool calls."""
    calendar_id = args.get("calendar_id", "").strip()
    event_id = args.get("event_id", "").strip()

    if not calendar_id:
        return tool_error("calendar_id is required")
    if not event_id:
        return tool_error("event_id is required")

    client = _get_client()
    if client is None:
        return tool_error("Feishu client not available")

    try:
        from lark_oapi.api.calendar.v4 import DeleteCalendarEventRequestBuilder
    except ImportError:
        return tool_error("lark_oapi not installed")

    request = (
        DeleteCalendarEventRequestBuilder()
        .calendar_id(calendar_id)
        .event_id(event_id)
        .build()
    )

    user_option = _get_user_request_option()
    response = client.request(request, user_option) if user_option else client.request(request)
    err = _check_response_error(response)
    if err:
        return tool_error(err)

    return tool_result(success=True, content="Event deleted successfully.")


def _handle_feishu_calendar_attendees(args: dict, **kwargs) -> str:
    """Handle feishu_calendar_attendees tool calls."""
    calendar_id = args.get("calendar_id", "").strip()
    event_id = args.get("event_id", "").strip()
    attendees = args.get("attendees", [])
    list_only = args.get("list_only", True)

    if not calendar_id:
        return tool_error("calendar_id is required")
    if not event_id:
        return tool_error("event_id is required")

    client = _get_client()
    if client is None:
        return tool_error("Feishu client not available")

    user_option = _get_user_request_option()

    try:
        from lark_oapi.api.calendar.v4 import (
            ListCalendarEventAttendeeRequestBuilder,
            CreateCalendarEventAttendeeRequestBuilder,
            CreateCalendarEventAttendeeRequestBodyBuilder,
            CalendarEventAttendeeBuilder,
            CalendarEventAttendeeIdBuilder,
        )
    except ImportError:
        return tool_error("lark_oapi not installed")

    if attendees and not list_only:
        # Add attendees
        att_items = []
        for att in attendees:
            att_type = att.get("type", "user")
            user_id = att.get("user_id", "")
            if user_id:
                att_id = CalendarEventAttendeeIdBuilder().user_id(user_id).build()
                att_item = (
                    CalendarEventAttendeeBuilder()
                    .attendee_id(att_id)
                    .type(att_type)
                    .build()
                )
                att_items.append(att_item)

        if not att_items:
            return tool_error("No valid attendees provided (each attendee needs user_id)")

        request = (
            CreateCalendarEventAttendeeRequestBuilder()
            .calendar_id(calendar_id)
            .event_id(event_id)
            .request_body(
                CreateCalendarEventAttendeeRequestBodyBuilder()
                .attendees(att_items)
                .build()
            )
            .build()
        )
        response = client.request(request, user_option) if user_option else client.request(request)
        err = _check_response_error(response)
        if err:
            return tool_error(err)
        data = getattr(response, "data", None)
        if data is None:
            return tool_error("No data returned from calendar API")
        formatted = _format_attendees(data if isinstance(data, dict) else {})
        return tool_result(success=True, content=formatted)

    # List attendees
    request = (
        ListCalendarEventAttendeeRequestBuilder()
        .calendar_id(calendar_id)
        .event_id(event_id)
        .page_size(100)
        .build()
    )

    response = client.request(request, user_option) if user_option else client.request(request)
    err = _check_response_error(response)
    if err:
        return tool_error(err)

    data = getattr(response, "data", None)
    if data is None:
        return tool_error("No data returned from calendar API")

    formatted = _format_attendees(data if isinstance(data, dict) else {})
    return tool_result(success=True, content=formatted)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

registry.register(
    name="feishu_calendar_list",
    toolset="feishu_calendar",
    schema=FEISHU_CALENDAR_LIST_SCHEMA,
    handler=_handle_feishu_calendar_list,
    check_fn=_check_feishu,
    requires_env=[],
    is_async=False,
    description="List Feishu calendars or get primary calendar",
    emoji="\U0001f4c5",
)

registry.register(
    name="feishu_calendar_events",
    toolset="feishu_calendar",
    schema=FEISHU_CALENDAR_EVENTS_SCHEMA,
    handler=_handle_feishu_calendar_events,
    check_fn=_check_feishu,
    requires_env=[],
    is_async=False,
    description="List events in a Feishu calendar",
    emoji="\U0001f4c5",
)

registry.register(
    name="feishu_calendar_create_event",
    toolset="feishu_calendar",
    schema=FEISHU_CALENDAR_CREATE_EVENT_SCHEMA,
    handler=_handle_feishu_calendar_create_event,
    check_fn=_check_feishu,
    requires_env=[],
    is_async=False,
    description="Create a new calendar event",
    emoji="\U0001f4c5",
)

registry.register(
    name="feishu_calendar_update_event",
    toolset="feishu_calendar",
    schema=FEISHU_CALENDAR_UPDATE_EVENT_SCHEMA,
    handler=_handle_feishu_calendar_update_event,
    check_fn=_check_feishu,
    requires_env=[],
    is_async=False,
    description="Update an existing calendar event",
    emoji="\U0001f4c5",
)

registry.register(
    name="feishu_calendar_delete_event",
    toolset="feishu_calendar",
    schema=FEISHU_CALENDAR_DELETE_EVENT_SCHEMA,
    handler=_handle_feishu_calendar_delete_event,
    check_fn=_check_feishu,
    requires_env=[],
    is_async=False,
    description="Delete a calendar event",
    emoji="\U0001f4c5",
)

registry.register(
    name="feishu_calendar_attendees",
    toolset="feishu_calendar",
    schema=FEISHU_CALENDAR_ATTENDEES_SCHEMA,
    handler=_handle_feishu_calendar_attendees,
    check_fn=_check_feishu,
    requires_env=[],
    is_async=False,
    description="List or add attendees to a calendar event",
    emoji="\U0001f4c5",
)
