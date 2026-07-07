"""Flow State local task API tools.

Registers Hermes-callable tools that talk to Flow State's localhost Local Task
API. Flow State remains the source of truth; Hermes is only a client.
"""

import json
import logging
import os
import re
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

_DEFAULT_BASE_URL = "http://127.0.0.1:5577"
_FLOW_STATE_API_URL: str = ""
_FLOW_STATE_API_TOKEN: str = ""
_DATE_ONLY_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_VALID_STATUS_FILTERS = {"todo", "open", "done"}
_VALID_DUE_FILTERS = {"today", "overdue", "open"}
_VALID_TASK_STATUSES = {"todo", "done"}
_VALID_PRIORITIES = {"low", "medium", "high"}


def _get_env_value(key: str) -> Optional[str]:
    try:
        from hermes_cli.config import get_env_value

        return get_env_value(key)
    except Exception:
        return os.getenv(key)


def _get_config() -> tuple[str, str]:
    base_url = (
        _FLOW_STATE_API_URL
        or _get_env_value("FLOW_STATE_API_URL")
        or _get_env_value("FLOWSTATE_API_URL")
        or _DEFAULT_BASE_URL
    )
    token = (
        _FLOW_STATE_API_TOKEN
        or _get_env_value("FLOW_STATE_API_TOKEN")
        or _get_env_value("FLOWSTATE_API_TOKEN")
        or ""
    )
    return base_url.rstrip("/"), token


def _headers(token: str = "") -> Dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _is_valid_date_or_due_filter(value: str) -> bool:
    return value in _VALID_DUE_FILTERS or bool(_DATE_ONLY_RE.match(value))


def _compact_http_error(exc: urllib.error.HTTPError) -> str:
    try:
        raw = exc.read().decode("utf-8", errors="replace")
        payload = json.loads(raw) if raw else {}
        message = payload.get("error") if isinstance(payload, dict) else None
    except Exception:
        message = None
    if not message:
        message = exc.reason or "HTTP error"
    if exc.code == 401:
        return "Flow State Local Task API rejected the bearer token. Check FLOW_STATE_API_TOKEN."
    if exc.code == 503:
        return "Flow State Local Task API is running but is not signed in."
    return f"Flow State API returned {exc.code}: {message}"


def _request(method: str, path: str, body: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    base_url, token = _get_config()
    data = None if body is None else json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        f"{base_url}{path}",
        data=data,
        headers=_headers(token),
        method=method,
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        raise RuntimeError(_compact_http_error(exc)) from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(
            f"Flow State Local Task API is unavailable at {base_url}. "
            "Open Flow State and enable Local Task API, then check FLOW_STATE_API_URL."
        ) from exc
    except TimeoutError as exc:
        raise RuntimeError(f"Flow State Local Task API timed out at {base_url}.") from exc

    try:
        payload = json.loads(raw) if raw else {}
    except json.JSONDecodeError as exc:
        raise RuntimeError("Flow State Local Task API returned non-JSON data.") from exc
    if isinstance(payload, dict) and payload.get("error"):
        raise RuntimeError(str(payload["error"]))
    if not isinstance(payload, dict):
        raise RuntimeError("Flow State Local Task API returned an unexpected response.")
    return payload


def _tool_result(payload: Dict[str, Any]) -> str:
    return json.dumps({"result": payload})


def _tool_error(message: str) -> str:
    from tools.registry import tool_error

    return tool_error(message)


def _check_flowstate_available() -> bool:
    base_url, token = _get_config()
    if base_url != _DEFAULT_BASE_URL:
        return True
    return bool(token)


def _handle_health(args: dict, **kw) -> str:
    try:
        return _tool_result(_request("GET", "/api/health"))
    except Exception as exc:
        logger.error("flowstate_health error: %s", exc)
        return _tool_error(str(exc))


def _handle_list_tasks(args: dict, **kw) -> str:
    status = args.get("status")
    due = args.get("due")
    limit = args.get("limit")

    params = urllib.parse.urlencode(
        {
            key: value
            for key, value in {
                "status": status,
                "due": due,
                "limit": limit,
            }.items()
            if value not in (None, "")
        }
    )

    if status not in (None, "") and status not in _VALID_STATUS_FILTERS:
        return _tool_error("status must be todo|open|done")
    if due not in (None, "") and not _is_valid_date_or_due_filter(str(due)):
        return _tool_error("due must be today|overdue|open|YYYY-MM-DD")
    if limit not in (None, ""):
        try:
            n = int(limit)
        except (TypeError, ValueError):
            return _tool_error("limit must be an integer from 1 to 25")
        if n < 1 or n > 25:
            return _tool_error("limit must be an integer from 1 to 25")

    try:
        suffix = f"?{params}" if params else ""
        return _tool_result(_request("GET", f"/api/tasks{suffix}"))
    except Exception as exc:
        logger.error("flowstate_list_tasks error: %s", exc)
        return _tool_error(str(exc))


def _handle_create_task(args: dict, **kw) -> str:
    title = str(args.get("title") or "").strip()
    if not title:
        return _tool_error("title is required")

    priority = args.get("priority")
    if priority in ("", None):
        priority = None
    if priority is not None and priority not in _VALID_PRIORITIES:
        return _tool_error("priority must be low|medium|high or null")

    due_date = args.get("dueDate")
    if due_date in ("", None):
        due_date = None
    elif not _DATE_ONLY_RE.match(str(due_date)):
        return _tool_error("dueDate must be YYYY-MM-DD")

    project_id = args.get("projectId")
    if project_id in ("", None):
        project_id = None

    body = {
        "title": title,
        "description": str(args.get("description") or ""),
        "priority": priority,
        "dueDate": due_date,
        "projectId": project_id,
    }
    try:
        return _tool_result(_request("POST", "/api/tasks", body))
    except Exception as exc:
        logger.error("flowstate_create_task error: %s", exc)
        return _tool_error(str(exc))


def _handle_update_task(args: dict, **kw) -> str:
    task_id = str(args.get("id") or "").strip()
    if not task_id:
        return _tool_error("id is required")

    body: Dict[str, Any] = {}
    if "status" in args and args.get("status") not in ("", None):
        status = args["status"]
        if status not in _VALID_TASK_STATUSES:
            return _tool_error("status must be todo|done")
        body["status"] = status
    if "title" in args and args.get("title") not in ("", None):
        title = str(args["title"]).strip()
        if not title:
            return _tool_error("title cannot be empty")
        body["title"] = title
    if "priority" in args:
        priority = args.get("priority")
        if priority in ("", None):
            body["priority"] = None
        elif priority in _VALID_PRIORITIES:
            body["priority"] = priority
        else:
            return _tool_error("priority must be low|medium|high or null")
    if "dueDate" in args:
        due_date = args.get("dueDate")
        if due_date in ("", None):
            body["dueDate"] = None
        elif _DATE_ONLY_RE.match(str(due_date)):
            body["dueDate"] = due_date
        else:
            return _tool_error("dueDate must be YYYY-MM-DD")
    if "progress" in args and args.get("progress") not in ("", None):
        try:
            progress = float(args["progress"])
        except (TypeError, ValueError):
            return _tool_error("progress must be a number from 0 to 100")
        if progress < 0 or progress > 100:
            return _tool_error("progress must be a number from 0 to 100")
        body["progress"] = progress

    if not body:
        return _tool_error("provide at least one field to update")

    try:
        return _tool_result(_request("PATCH", f"/api/tasks/{urllib.parse.quote(task_id, safe='')}", body))
    except Exception as exc:
        logger.error("flowstate_update_task error: %s", exc)
        return _tool_error(str(exc))


def _handle_delete_task(args: dict, **kw) -> str:
    task_id = str(args.get("id") or "").strip()
    if not task_id:
        return _tool_error("id is required")
    try:
        return _tool_result(_request("DELETE", f"/api/tasks/{urllib.parse.quote(task_id, safe='')}"))
    except Exception as exc:
        logger.error("flowstate_delete_task error: %s", exc)
        return _tool_error(str(exc))


def _handle_current_timer(args: dict, **kw) -> str:
    try:
        return _tool_result(_request("GET", "/api/timer/current"))
    except Exception as exc:
        logger.error("flowstate_get_current_timer error: %s", exc)
        return _tool_error(str(exc))


FLOWSTATE_HEALTH_SCHEMA = {
    "name": "flowstate_health",
    "description": "Check whether the local Flow State API sidecar is reachable.",
    "parameters": {"type": "object", "properties": {}, "required": []},
}

FLOWSTATE_LIST_TASKS_SCHEMA = {
    "name": "flowstate_list_tasks",
    "description": (
        "List Flow State tasks. Flow State is the user's personal task app; "
        "it is not a project name. Use returned task ids for later updates."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "status": {"type": "string", "description": "Optional: todo, open, or done."},
            "due": {"type": "string", "description": "Optional: today, overdue, open, or YYYY-MM-DD."},
            "limit": {"type": "integer", "description": "Optional result cap, 1-25."},
        },
        "required": [],
    },
}

FLOWSTATE_CREATE_TASK_SCHEMA = {
    "name": "flowstate_create_task",
    "description": (
        "Create a personal task in Flow State. Omit projectId unless the user "
        "explicitly provided a known Flow State project id."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "title": {"type": "string", "description": "Task title."},
            "description": {"type": "string", "description": "Optional task description."},
            "priority": {"type": "string", "description": "Optional: low, medium, high, or null."},
            "dueDate": {"type": "string", "description": "Optional YYYY-MM-DD due date."},
            "projectId": {"type": "string", "description": "Optional known Flow State project id."},
        },
        "required": ["title"],
    },
}

FLOWSTATE_UPDATE_TASK_SCHEMA = {
    "name": "flowstate_update_task",
    "description": "Update an existing Flow State task by exact task id.",
    "parameters": {
        "type": "object",
        "properties": {
            "id": {"type": "string", "description": "Exact Flow State task id."},
            "status": {"type": "string", "description": "Optional: todo or done."},
            "title": {"type": "string", "description": "Optional new title."},
            "priority": {"type": "string", "description": "Optional: low, medium, high, or null."},
            "dueDate": {"type": "string", "description": "Optional YYYY-MM-DD or null."},
            "progress": {"type": "number", "description": "Optional progress from 0 to 100."},
        },
        "required": ["id"],
    },
}

FLOWSTATE_DELETE_TASK_SCHEMA = {
    "name": "flowstate_delete_task",
    "description": "Soft-delete an existing Flow State task by exact task id.",
    "parameters": {
        "type": "object",
        "properties": {"id": {"type": "string", "description": "Exact Flow State task id."}},
        "required": ["id"],
    },
}

FLOWSTATE_CURRENT_TIMER_SCHEMA = {
    "name": "flowstate_get_current_timer",
    "description": "Get the current Flow State timer session, if one is active.",
    "parameters": {"type": "object", "properties": {}, "required": []},
}


from tools.registry import registry

for _name, _schema, _handler in [
    ("flowstate_health", FLOWSTATE_HEALTH_SCHEMA, _handle_health),
    ("flowstate_list_tasks", FLOWSTATE_LIST_TASKS_SCHEMA, _handle_list_tasks),
    ("flowstate_create_task", FLOWSTATE_CREATE_TASK_SCHEMA, _handle_create_task),
    ("flowstate_update_task", FLOWSTATE_UPDATE_TASK_SCHEMA, _handle_update_task),
    ("flowstate_delete_task", FLOWSTATE_DELETE_TASK_SCHEMA, _handle_delete_task),
    ("flowstate_get_current_timer", FLOWSTATE_CURRENT_TIMER_SCHEMA, _handle_current_timer),
]:
    registry.register(
        name=_name,
        toolset="flowstate",
        schema=_schema,
        handler=_handler,
        check_fn=_check_flowstate_available,
        emoji="📋",
    )
