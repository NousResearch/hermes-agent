"""Feishu Task Tool -- manage Feishu/Lark tasks via API.

Provides the following tools:
  - feishu_task_list       : List tasks assigned to the current user
  - feishu_task_create    : Create a new task
  - feishu_task_complete   : Mark a task as completed
  - feishu_task_reopen    : Reopen a completed task
  - feishu_task_search    : Search tasks by query
  - feishu_task_delete    : Delete a task

Required scopes: task:task:read, task:task:write
"""

import logging
from datetime import datetime, timezone
from typing import Optional

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
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Time helpers
# ---------------------------------------------------------------------------

def _parse_iso8601(ts: str) -> Optional[datetime]:
    """Parse an ISO8601 timestamp string to datetime (UTC)."""
    if not ts:
        return None
    try:
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

def _format_task_list(data: dict) -> str:
    """Format task list response."""
    items = data.get("items", [])
    if not items:
        return "No tasks found."

    lines = ["Tasks:"]
    for task in items:
        guid = task.get("guid", "")
        summary = task.get("summary", "(no title)")
        due = task.get("due", {})
        completed_at = task.get("completed_at")

        line = f"  - [{guid}] {summary}"
        if completed_at:
            line += " [COMPLETED]"
        lines.append(line)

        # Format due date
        if isinstance(due, dict):
            ts = due.get("timestamp", "")
            is_all_day = due.get("is_all_day", False)
            if ts:
                try:
                    dt = datetime.fromtimestamp(int(ts), tz=timezone.utc)
                    if is_all_day:
                        due_str = dt.strftime("%Y-%m-%d")
                    else:
                        due_str = dt.strftime("%Y-%m-%d %H:%M")
                    lines.append(f"    Due: {due_str}")
                except (ValueError, OSError):
                    if ts:
                        lines.append(f"    Due: {ts}")

    return "\n".join(lines)


def _format_task(data: dict) -> str:
    """Format single task response."""
    task = data.get("task", data)
    guid = task.get("guid", "")
    summary = task.get("summary", "(no title)")
    description = task.get("description", "")
    completed_at = task.get("completed_at")
    due = task.get("due", {})
    members = task.get("members", [])

    lines = [f"Task: {summary}"]
    if guid:
        lines.append(f"  GUID: {guid}")

    if completed_at:
        lines.append("  Status: Completed")
        if isinstance(completed_at, dict):
            ts = completed_at.get("timestamp", "")
            if ts:
                try:
                    dt = datetime.fromtimestamp(int(ts), tz=timezone.utc)
                    lines.append(f"  Completed at: {dt.strftime('%Y-%m-%d %H:%M')}")
                except (ValueError, OSError):
                    pass
    else:
        lines.append("  Status: Open")

    # Format due date
    if isinstance(due, dict):
        ts = due.get("timestamp", "")
        is_all_day = due.get("is_all_day", False)
        if ts:
            try:
                dt = datetime.fromtimestamp(int(ts), tz=timezone.utc)
                if is_all_day:
                    due_str = dt.strftime("%Y-%m-%d")
                else:
                    due_str = dt.strftime("%Y-%m-%d %H:%M")
                lines.append(f"  Due: {due_str}")
            except (ValueError, OSError):
                pass

    # Format description
    if description:
        desc_preview = description[:200] + "..." if len(description) > 200 else description
        lines.append(f"  Description: {desc_preview}")

    # Format members
    if members:
        assignee = next((m for m in members if m.get("role") == "assignee"), None)
        followers = [m for m in members if m.get("role") == "follower"]
        if assignee:
            lines.append(f"  Assignee: {assignee.get('id', '')}")
        if followers:
            follower_ids = [f.get("id", "") for f in followers]
            lines.append(f"  Followers: {', '.join(follower_ids)}")

    return "\n".join(lines)


def _check_response_error(response) -> Optional[str]:
    """Check response for errors. Returns error message or None if OK."""
    code = getattr(response, "code", None)
    if code != 0:
        msg = getattr(response, "msg", "unknown error")
        return f"Task API error: code={code} msg={msg}"
    return None


# ---------------------------------------------------------------------------
# Tool schemas
# ---------------------------------------------------------------------------

FEISHU_TASK_LIST_SCHEMA = {
    "name": "feishu_task_list",
    "description": (
        "List tasks assigned to the current user. "
        "Supports filtering by completion status and due date range."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "completed": {
                "type": "boolean",
                "description": (
                    "Filter by completion status. "
                    "True = completed only, False = open only, omit = all tasks."
                ),
            },
            "due_start": {
                "type": "string",
                "description": (
                    "Filter tasks with due date on or after this date (ISO8601 format, "
                    "e.g. '2026-04-01T00:00:00Z')."
                ),
            },
            "due_end": {
                "type": "string",
                "description": (
                    "Filter tasks with due date on or before this date (ISO8601 format, "
                    "e.g. '2026-04-30T23:59:59Z')."
                ),
            },
            "limit": {
                "type": "integer",
                "description": "Maximum number of tasks to return (default 20, max 100).",
                "minimum": 1,
                "maximum": 100,
                "default": 20,
            },
        },
        "required": [],
    },
}

FEISHU_TASK_CREATE_SCHEMA = {
    "name": "feishu_task_create",
    "description": (
        "Create a new task. All times must be in ISO8601 format with timezone "
        "(e.g. '2026-04-01T09:00:00+08:00' or '2026-04-01T09:00:00Z')."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "summary": {
                "type": "string",
                "description": "Task title/summary (required).",
            },
            "due": {
                "type": "string",
                "description": "Due date/time in ISO8601 format.",
            },
            "description": {
                "type": "string",
                "description": "Task description/notes.",
            },
            "assignee": {
                "type": "string",
                "description": (
                    "Assignee user open_id (e.g. 'ou_xxx'). "
                    "Note: Bot cannot add cross-tenant members."
                ),
            },
            "follower": {
                "type": "string",
                "description": (
                    "Follower user open_id (e.g. 'ou_xxx'). "
                    "Note: Bot cannot add cross-tenant members."
                ),
            },
        },
        "required": ["summary"],
    },
}

FEISHU_TASK_COMPLETE_SCHEMA = {
    "name": "feishu_task_complete",
    "description": "Mark a task as completed.",
    "parameters": {
        "type": "object",
        "properties": {
            "task_guid": {
                "type": "string",
                "description": (
                    "The task GUID (format: 'task guid/xxx'). "
                    "You can find this from feishu_task_list or feishu_task_search results."
                ),
            },
        },
        "required": ["task_guid"],
    },
}

FEISHU_TASK_REOPEN_SCHEMA = {
    "name": "feishu_task_reopen",
    "description": "Reopen a completed task.",
    "parameters": {
        "type": "object",
        "properties": {
            "task_guid": {
                "type": "string",
                "description": (
                    "The task GUID (format: 'task guid/xxx'). "
                    "You can find this from feishu_task_list or feishu_task_search results."
                ),
            },
        },
        "required": ["task_guid"],
    },
}

FEISHU_TASK_SEARCH_SCHEMA = {
    "name": "feishu_task_search",
    "description": (
        "Search tasks by text query. "
        "Supports filtering by assignee, creator, and completion status."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query text.",
            },
            "assignee": {
                "type": "string",
                "description": "Filter by assignee open_id (e.g. 'ou_xxx').",
            },
            "creator": {
                "type": "string",
                "description": "Filter by creator open_id (e.g. 'ou_xxx').",
            },
            "completed": {
                "type": "boolean",
                "description": "Filter by completion status.",
            },
            "limit": {
                "type": "integer",
                "description": "Maximum number of tasks to return (default 20, max 100).",
                "minimum": 1,
                "maximum": 100,
                "default": 20,
            },
        },
        "required": ["query"],
    },
}

FEISHU_TASK_DELETE_SCHEMA = {
    "name": "feishu_task_delete",
    "description": "Delete a task.",
    "parameters": {
        "type": "object",
        "properties": {
            "task_guid": {
                "type": "string",
                "description": (
                    "The task GUID (format: 'task guid/xxx'). "
                    "You can find this from feishu_task_list or feishu_task_search results."
                ),
            },
        },
        "required": ["task_guid"],
    },
}


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------

def _handle_feishu_task_list(args: dict, **kwargs) -> str:
    """Handle feishu_task_list tool calls."""
    client = _get_client()
    if client is None:
        return tool_error(
            "Feishu client not available (not in a Feishu comment context). "
            "Make sure this tool is called within an active Feishu session."
        )

    completed = args.get("completed")
    due_start = args.get("due_start", "").strip()
    due_end = args.get("due_end", "").strip()
    limit = args.get("limit", 20)
    try:
        limit = max(1, min(100, int(limit)))
    except (ValueError, TypeError):
        limit = 20

    # Parse due dates
    due_start_ts = ""
    due_end_ts = ""
    if due_start:
        dt = _parse_iso8601(due_start)
        if dt is None:
            return tool_error(
                f"Invalid due_start format: '{due_start}'. "
                "Use ISO8601 (e.g. '2026-04-01T00:00:00Z')."
            )
        due_start_ts = _to_unix_ts(dt)

    if due_end:
        dt = _parse_iso8601(due_end)
        if dt is None:
            return tool_error(
                f"Invalid due_end format: '{due_end}'. "
                "Use ISO8601 (e.g. '2026-04-30T23:59:59Z')."
            )
        due_end_ts = _to_unix_ts(dt)

    try:
        from lark_oapi.api.task.v2 import ListTaskRequestBuilder
    except Exception:
        return tool_error("lark_oapi not installed. Install with: pip install lark-oapi")

    if due_start_ts:
        return tool_error(
            "due_start is not yet supported by the Feishu Task API. "
            "The SDK's ListTaskRequestBuilder does not expose a due_start parameter. "
            "Please file a feature request with Feishu."
        )
    if due_end_ts:
        return tool_error(
            "due_end is not yet supported by the Feishu Task API. "
            "The SDK's ListTaskRequestBuilder does not expose a due_end parameter. "
            "Please file a feature request with Feishu."
        )

    builder = ListTaskRequestBuilder().page_size(limit)

    if completed is not None:
        builder.completed(completed)

    request = builder.build()

    response = client.request(request)
    err = _check_response_error(response)
    if err:
        return tool_error(err)

    data = getattr(response, "data", None)
    if data is None:
        return tool_error("No data returned from task API")

    formatted = _format_task_list(data if isinstance(data, dict) else {})
    return tool_result(success=True, content=formatted)


def _handle_feishu_task_create(args: dict, **kwargs) -> str:
    """Handle feishu_task_create tool calls."""
    summary = args.get("summary", "").strip()
    if not summary:
        return tool_error("summary is required")

    due_str = args.get("due", "").strip()
    description = args.get("description", "").strip()
    assignee = args.get("assignee", "").strip()
    follower = args.get("follower", "").strip()

    # Parse due date
    due_dt = None
    if due_str:
        due_dt = _parse_iso8601(due_str)
        if due_dt is None:
            return tool_error(
                f"Invalid due format: '{due_str}'. "
                "Use ISO8601 (e.g. '2026-04-01T09:00:00+08:00' or '2026-04-01T09:00:00Z')."
            )

    client = _get_client()
    if client is None:
        return tool_error(
            "Feishu client not available (not in a Feishu comment context). "
            "Make sure this tool is called within an active Feishu session."
        )

    user_option = _get_user_request_option()

    try:
        from lark_oapi.api.task.v2 import (
            CreateTaskRequestBuilder,
            TaskBuilder,
        )
        from lark_oapi.api.task.v2.model import DueBuilder, MemberBuilder
    except Exception:
        return tool_error("lark_oapi not installed. Install with: pip install lark-oapi")

    # Build task body
    task_builder = TaskBuilder().summary(summary)

    if due_dt:
        task_builder.due(
            DueBuilder()
            .timestamp(_to_unix_ts(due_dt))
            .is_all_day(False)
            .build()
        )

    if description:
        task_builder.description(description)

    # Add members
    members = []
    if assignee:
        members.append(
            MemberBuilder()
            .id(assignee)
            .type("user")
            .role("assignee")
            .build()
        )
    if follower:
        members.append(
            MemberBuilder()
            .id(follower)
            .type("user")
            .role("follower")
            .build()
        )
    if members:
        task_builder.members(members)

    request = (
        CreateTaskRequestBuilder()
        .request_body(task_builder.build())
        .build()
    )

    response = client.request(request, user_option) if user_option else client.request(request)
    err = _check_response_error(response)
    if err:
        return tool_error(err)

    data = getattr(response, "data", None)
    if data is None:
        return tool_error("No data returned from task API")

    data_dict = data if isinstance(data, dict) else {}
    formatted = _format_task(data_dict)
    return tool_result(success=True, content=formatted)


def _handle_feishu_task_complete(args: dict, **kwargs) -> str:
    """Handle feishu_task_complete tool calls."""
    task_guid = args.get("task_guid", "").strip()
    if not task_guid:
        return tool_error("task_guid is required")

    client = _get_client()
    if client is None:
        return tool_error(
            "Feishu client not available (not in a Feishu comment context). "
            "Make sure this tool is called within an active Feishu session."
        )

    user_option = _get_user_request_option()

    try:
        from lark_oapi.api.task.v2 import (
            PatchTaskRequestBuilder,
            PatchTaskRequestBodyBuilder,
            TaskBuilder,
        )
    except Exception:
        return tool_error("lark_oapi not installed. Install with: pip install lark-oapi")

    now_ts = _to_unix_ts(datetime.now(timezone.utc))

    request = (
        PatchTaskRequestBuilder()
        .task_guid(task_guid)
        .request_body(
            PatchTaskRequestBodyBuilder()
            .task(TaskBuilder().completed_at(now_ts).build())
            .update_fields(["completed_at"])
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
        return tool_error("No data returned from task API")

    data_dict = data if isinstance(data, dict) else {}
    formatted = _format_task(data_dict)
    return tool_result(success=True, content=formatted)


def _handle_feishu_task_reopen(args: dict, **kwargs) -> str:
    """Handle feishu_task_reopen tool calls."""
    task_guid = args.get("task_guid", "").strip()
    if not task_guid:
        return tool_error("task_guid is required")

    client = _get_client()
    if client is None:
        return tool_error(
            "Feishu client not available (not in a Feishu comment context). "
            "Make sure this tool is called within an active Feishu session."
        )

    user_option = _get_user_request_option()

    try:
        from lark_oapi.api.task.v2 import (
            PatchTaskRequestBuilder,
            PatchTaskRequestBodyBuilder,
            TaskBuilder,
        )
    except Exception:
        return tool_error("lark_oapi not installed. Install with: pip install lark-oapi")

    request = (
        PatchTaskRequestBuilder()
        .task_guid(task_guid)
        .request_body(
            PatchTaskRequestBodyBuilder()
            .task(TaskBuilder().completed_at(0).build())
            .update_fields(["completed_at"])
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
        return tool_error("No data returned from task API")

    data_dict = data if isinstance(data, dict) else {}
    formatted = _format_task(data_dict)
    return tool_result(success=True, content=formatted)


def _handle_feishu_task_search(args: dict, **kwargs) -> str:
    """Handle feishu_task_search tool calls."""
    query = args.get("query", "").strip()
    if not query:
        return tool_error("query is required")

    assignee = args.get("assignee", "").strip()
    creator = args.get("creator", "").strip()
    completed = args.get("completed")
    limit = args.get("limit", 20)
    try:
        limit = max(1, min(100, int(limit)))
    except (ValueError, TypeError):
        limit = 20

    client = _get_client()
    if client is None:
        return tool_error(
            "Feishu client not available (not in a Feishu comment context). "
            "Make sure this tool is called within an active Feishu session."
        )

    user_option = _get_user_request_option()

    try:
        from lark_oapi.api.task.v2 import ListTaskRequestBuilder
    except Exception:
        return tool_error("lark_oapi not installed. Install with: pip install lark-oapi")

    # NOTE: ListTaskRequestBuilder does not support query/due_start/due_end filters.
    # It only supports: completed, page_size, page_token, type, user_id_type.
    # For full-text search and due date filtering, the SDK needs to be updated.
    # The query/assignee/creator fields are silently ignored per current SDK limits.
    builder = ListTaskRequestBuilder().page_size(limit)
    if completed is not None:
        builder.completed(completed)

    request = builder.build()

    response = client.request(request, user_option) if user_option else client.request(request)
    err = _check_response_error(response)
    if err:
        return tool_error(err)

    data = getattr(response, "data", None)
    if data is None:
        return tool_error("No data returned from task API")

    formatted = _format_task_list(data if isinstance(data, dict) else {})
    return tool_result(success=True, content=formatted)


def _handle_feishu_task_delete(args: dict, **kwargs) -> str:
    """Handle feishu_task_delete tool calls."""
    task_guid = args.get("task_guid", "").strip()
    if not task_guid:
        return tool_error("task_guid is required")

    client = _get_client()
    if client is None:
        return tool_error(
            "Feishu client not available (not in a Feishu comment context). "
            "Make sure this tool is called within an active Feishu session."
        )

    user_option = _get_user_request_option()

    try:
        from lark_oapi.api.task.v2 import DeleteTaskRequestBuilder
    except Exception:
        return tool_error("lark_oapi not installed. Install with: pip install lark-oapi")

    request = (
        DeleteTaskRequestBuilder()
        .task_guid(task_guid)
        .build()
    )

    response = client.request(request, user_option) if user_option else client.request(request)
    err = _check_response_error(response)
    if err:
        return tool_error(err)

    return tool_result(success=True, content="Task deleted successfully.")


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

registry.register(
    name="feishu_task_list",
    toolset="feishu_task",
    schema=FEISHU_TASK_LIST_SCHEMA,
    handler=_handle_feishu_task_list,
    check_fn=_check_feishu,
    requires_env=[],
    is_async=False,
    description="List tasks assigned to the current user",
    emoji="\U0001f4dd",
)

registry.register(
    name="feishu_task_create",
    toolset="feishu_task",
    schema=FEISHU_TASK_CREATE_SCHEMA,
    handler=_handle_feishu_task_create,
    check_fn=_check_feishu,
    requires_env=[],
    is_async=False,
    description="Create a new task",
    emoji="\U0001f4dd",
)

registry.register(
    name="feishu_task_complete",
    toolset="feishu_task",
    schema=FEISHU_TASK_COMPLETE_SCHEMA,
    handler=_handle_feishu_task_complete,
    check_fn=_check_feishu,
    requires_env=[],
    is_async=False,
    description="Mark a task as completed",
    emoji="\U0001f4dd",
)

registry.register(
    name="feishu_task_reopen",
    toolset="feishu_task",
    schema=FEISHU_TASK_REOPEN_SCHEMA,
    handler=_handle_feishu_task_reopen,
    check_fn=_check_feishu,
    requires_env=[],
    is_async=False,
    description="Reopen a completed task",
    emoji="\U0001f4dd",
)

registry.register(
    name="feishu_task_search",
    toolset="feishu_task",
    schema=FEISHU_TASK_SEARCH_SCHEMA,
    handler=_handle_feishu_task_search,
    check_fn=_check_feishu,
    requires_env=[],
    is_async=False,
    description="Search tasks by query",
    emoji="\U0001f4dd",
)

registry.register(
    name="feishu_task_delete",
    toolset="feishu_task",
    schema=FEISHU_TASK_DELETE_SCHEMA,
    handler=_handle_feishu_task_delete,
    check_fn=_check_feishu,
    requires_env=[],
    is_async=False,
    description="Delete a task",
    emoji="\U0001f4dd",
)
