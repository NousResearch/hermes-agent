"""Feishu Task Tool -- task management via Feishu Task v2 API.

Provides five tools for managing Feishu tasks:
  - ``feishu_task_list``        -- list tasks (optionally filtered by tasklist/completion)
  - ``feishu_task_get``         -- get a single task by GUID
  - ``feishu_task_create``      -- create a new task
  - ``feishu_task_update``      -- update summary, due time, or completion state
  - ``feishu_task_add_comment`` -- add a comment to a task

Uses FeishuClient.for_user() (UAT) with scope ``task:task``.
"""

import json
import logging

from tools.feishu_oapi_client import (
    AppScopeMissingError,
    FeishuClient,
    NeedAuthorizationError,
    TOOLS_METADATA,
    UserAuthRequiredError,
    UserScopeInsufficientError,
    raise_for_feishu_errcode,
)
from tools.registry import registry, tool_error, tool_result

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# TOOLS_METADATA entries (scope declarations)
# ---------------------------------------------------------------------------

TOOLS_METADATA["feishu_task_list"] = {"identity": "user", "scopes": ["task:task"]}
TOOLS_METADATA["feishu_task_get"] = {"identity": "user", "scopes": ["task:task"]}
TOOLS_METADATA["feishu_task_create"] = {"identity": "user", "scopes": ["task:task"]}
TOOLS_METADATA["feishu_task_update"] = {"identity": "user", "scopes": ["task:task"]}
TOOLS_METADATA["feishu_task_add_comment"] = {"identity": "user", "scopes": ["task:task"]}


def _check_feishu():
    try:
        import lark_oapi  # noqa: F401
        return True
    except ImportError:
        return False


def _auth_error_message(exc: Exception) -> str:
    """Format semantic auth exceptions as tool_error strings."""
    if isinstance(exc, NeedAuthorizationError):
        return f"Need Feishu authorization: {exc}. Run 'hermes feishu-uat' to authorize."
    if isinstance(exc, AppScopeMissingError):
        return f"App scope missing: {exc}"
    if isinstance(exc, UserAuthRequiredError):
        return f"User authorization required: {exc}"
    if isinstance(exc, UserScopeInsufficientError):
        return f"User scope insufficient: {exc}"
    return str(exc)


def _get_user_client():
    """Return (client, error_str). error_str is None on success."""
    try:
        return FeishuClient.for_user(), None
    except NeedAuthorizationError as exc:
        return None, _auth_error_message(exc)
    except ValueError as exc:
        return None, f"Feishu configuration error: {exc}"


def _do_request(client, method, uri, *, paths=None, queries=None, body=None):
    """Build and execute a BaseRequest with UAT, return (code, msg, data).

    Supports GET, POST, and PATCH HTTP methods.
    """
    try:
        from lark_oapi import AccessTokenType, RequestOption
        from lark_oapi.core.enum import HttpMethod
        from lark_oapi.core.model.base_request import BaseRequest
    except ImportError as exc:
        raise RuntimeError("lark_oapi not installed") from exc

    method_upper = method.upper()
    if method_upper == "GET":
        http_method = HttpMethod.GET
    elif method_upper == "PATCH":
        http_method = HttpMethod.PATCH
    else:
        http_method = HttpMethod.POST

    builder = (
        BaseRequest.builder()
        .http_method(http_method)
        .uri(uri)
        .token_types({AccessTokenType.USER})
    )
    if paths:
        builder = builder.paths(paths)
    if queries:
        builder = builder.queries(queries)
    if body is not None:
        builder = builder.body(body)

    request = builder.build()

    opt = (
        RequestOption.builder()
        .user_access_token(client.access_token)
        .build()
    )
    response = client.sdk.request(request, opt)

    code = getattr(response, "code", None)
    msg = getattr(response, "msg", "")

    data: dict = {}
    raw = getattr(response, "raw", None)
    if raw and hasattr(raw, "content"):
        try:
            body_json = json.loads(raw.content)
            data = body_json.get("data", {})
            if code is None:
                code = body_json.get("code", -1)
            if not msg:
                msg = body_json.get("msg", "")
        except (json.JSONDecodeError, AttributeError):
            pass
    if not data:
        resp_data = getattr(response, "data", None)
        if isinstance(resp_data, dict):
            data = resp_data
        elif resp_data and hasattr(resp_data, "__dict__"):
            data = vars(resp_data)

    return (code if code is not None else -1), msg, data


# ---------------------------------------------------------------------------
# feishu_task_list
# ---------------------------------------------------------------------------

_TASK_LIST_URI = "/open-apis/task/v2/tasks"

FEISHU_TASK_LIST_SCHEMA = {
    "name": "feishu_task_list",
    "description": (
        "List Feishu tasks for the current user. "
        "Optionally filter by tasklist ID or completion state."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "tasklist_id": {
                "type": "string",
                "description": "Filter tasks by tasklist GUID. Omit to list all tasks.",
            },
            "completed": {
                "type": "boolean",
                "description": "If true, return only completed tasks; if false, only incomplete. Omit to return all.",
            },
            "page_size": {
                "type": "integer",
                "description": "Number of tasks per page (max 100, default 50).",
                "default": 50,
            },
            "page_token": {
                "type": "string",
                "description": "Pagination token for the next page.",
            },
        },
        "required": [],
    },
}


def _handle_task_list(args: dict, **kwargs) -> str:
    logger.info("feishu_task_list: tasklist_id=%s", args.get("tasklist_id", ""))
    client, err = _get_user_client()
    if err:
        return tool_error(err)

    queries = [("user_id_type", "open_id")]

    tasklist_id = (args.get("tasklist_id") or "").strip()
    if tasklist_id:
        queries.append(("tasklist_guid", tasklist_id))

    completed = args.get("completed")
    if completed is not None:
        queries.append(("completed", "true" if completed else "false"))

    page_size = args.get("page_size", 50)
    queries.append(("page_size", str(page_size)))

    page_token = (args.get("page_token") or "").strip()
    if page_token:
        queries.append(("page_token", page_token))

    code, msg, data = _do_request(client, "GET", _TASK_LIST_URI, queries=queries)
    if code != 0:
        try:
            raise_for_feishu_errcode(code, msg or "", api_name="feishu.task.list")
        except (AppScopeMissingError, UserAuthRequiredError, UserScopeInsufficientError, NeedAuthorizationError) as e:
            return tool_error(_auth_error_message(e))
        logger.warning("feishu_task_list failed: code=%d msg=%s", code, msg)
        return tool_error(f"List tasks failed: code={code} msg={msg}")

    logger.info("feishu_task_list: returned %d tasks", len((data or {}).get("items", [])))
    return tool_result(data)


# ---------------------------------------------------------------------------
# feishu_task_get
# ---------------------------------------------------------------------------

_TASK_GET_URI = "/open-apis/task/v2/tasks/:task_guid"

FEISHU_TASK_GET_SCHEMA = {
    "name": "feishu_task_get",
    "description": "Get details of a single Feishu task by its GUID.",
    "parameters": {
        "type": "object",
        "properties": {
            "task_id": {
                "type": "string",
                "description": "The task GUID.",
            },
        },
        "required": ["task_id"],
    },
}


def _handle_task_get(args: dict, **kwargs) -> str:
    task_id = (args.get("task_id") or "").strip()
    logger.info("feishu_task_get: task_id=%s", task_id)
    client, err = _get_user_client()
    if err:
        return tool_error(err)

    if not task_id:
        return tool_error("task_id is required")

    code, msg, data = _do_request(
        client, "GET", _TASK_GET_URI,
        paths={"task_guid": task_id},
        queries=[("user_id_type", "open_id")],
    )
    if code != 0:
        try:
            raise_for_feishu_errcode(code, msg or "", api_name="feishu.task.get")
        except (AppScopeMissingError, UserAuthRequiredError, UserScopeInsufficientError, NeedAuthorizationError) as e:
            return tool_error(_auth_error_message(e))
        logger.warning("feishu_task_get failed: code=%d msg=%s", code, msg)
        return tool_error(f"Get task failed: code={code} msg={msg}")

    logger.info("feishu_task_get: fetched task %s", task_id)
    return tool_result(data)


# ---------------------------------------------------------------------------
# feishu_task_create
# ---------------------------------------------------------------------------

_TASK_CREATE_URI = "/open-apis/task/v2/tasks"

FEISHU_TASK_CREATE_SCHEMA = {
    "name": "feishu_task_create",
    "description": "Create a new Feishu task with an optional due time, description, and member list.",
    "parameters": {
        "type": "object",
        "properties": {
            "summary": {
                "type": "string",
                "description": "Task title/summary (required).",
            },
            "due_time": {
                "type": "string",
                "description": "Due date-time in RFC3339 format, e.g. '2026-05-01T18:00:00+08:00'.",
            },
            "description": {
                "type": "string",
                "description": "Task description (plain text).",
            },
            "members": {
                "type": "array",
                "description": "List of assignee open_id strings.",
                "items": {"type": "string"},
            },
        },
        "required": ["summary"],
    },
}


def _handle_task_create(args: dict, **kwargs) -> str:
    summary = (args.get("summary") or "").strip()
    logger.info("feishu_task_create: summary=%r", summary)
    client, err = _get_user_client()
    if err:
        return tool_error(err)

    if not summary:
        return tool_error("summary is required")

    body: dict = {"summary": summary}

    due_time = (args.get("due_time") or "").strip()
    if due_time:
        body["due"] = {"timestamp": due_time, "is_all_day": False}

    description = (args.get("description") or "").strip()
    if description:
        body["description"] = description

    members = args.get("members")
    if members:
        body["members"] = [
            {"id": uid, "type": "user", "role": "assignee"}
            for uid in members
            if uid
        ]

    code, msg, data = _do_request(
        client, "POST", _TASK_CREATE_URI,
        queries=[("user_id_type", "open_id")],
        body=body,
    )
    if code != 0:
        try:
            raise_for_feishu_errcode(code, msg or "", api_name="feishu.task.create")
        except (AppScopeMissingError, UserAuthRequiredError, UserScopeInsufficientError, NeedAuthorizationError) as e:
            return tool_error(_auth_error_message(e))
        logger.warning("feishu_task_create failed: code=%d msg=%s", code, msg)
        return tool_error(f"Create task failed: code={code} msg={msg}")

    logger.info("feishu_task_create: created task summary=%r", summary)
    return tool_result(success=True, data=data)


# ---------------------------------------------------------------------------
# feishu_task_update
# ---------------------------------------------------------------------------

_TASK_UPDATE_URI = "/open-apis/task/v2/tasks/:task_guid"

FEISHU_TASK_UPDATE_SCHEMA = {
    "name": "feishu_task_update",
    "description": (
        "Update a Feishu task. Supports changing summary, due time, and completion state. "
        "Only provided fields are updated."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "task_id": {
                "type": "string",
                "description": "The task GUID to update.",
            },
            "summary": {
                "type": "string",
                "description": "New task title/summary.",
            },
            "due_time": {
                "type": "string",
                "description": "New due date-time in RFC3339 format, e.g. '2026-05-01T18:00:00+08:00'.",
            },
            "completed": {
                "type": "boolean",
                "description": "Set to true to mark task complete, false to reopen.",
            },
        },
        "required": ["task_id"],
    },
}


def _handle_task_update(args: dict, **kwargs) -> str:
    task_id = (args.get("task_id") or "").strip()
    logger.info("feishu_task_update: task_id=%s", task_id)
    client, err = _get_user_client()
    if err:
        return tool_error(err)

    if not task_id:
        return tool_error("task_id is required")

    task_patch: dict = {}
    update_fields = []

    summary = (args.get("summary") or "").strip()
    if summary:
        task_patch["summary"] = summary
        update_fields.append("summary")

    due_time = (args.get("due_time") or "").strip()
    if due_time:
        task_patch["due"] = {"timestamp": due_time, "is_all_day": False}
        update_fields.append("due")

    completed = args.get("completed")
    if completed is not None:
        task_patch["completed_at"] = "1" if completed else "0"
        update_fields.append("completed_at")

    if not task_patch:
        return tool_error("At least one of summary, due_time, or completed must be provided")

    body = {"task": task_patch, "update_fields": update_fields}

    code, msg, data = _do_request(
        client, "PATCH", _TASK_UPDATE_URI,
        paths={"task_guid": task_id},
        queries=[("user_id_type", "open_id")],
        body=body,
    )
    if code != 0:
        try:
            raise_for_feishu_errcode(code, msg or "", api_name="feishu.task.update")
        except (AppScopeMissingError, UserAuthRequiredError, UserScopeInsufficientError, NeedAuthorizationError) as e:
            return tool_error(_auth_error_message(e))
        logger.warning("feishu_task_update failed: code=%d msg=%s", code, msg)
        return tool_error(f"Update task failed: code={code} msg={msg}")

    logger.info("feishu_task_update: updated task %s fields=%s", task_id, update_fields)
    return tool_result(success=True, data=data)


# ---------------------------------------------------------------------------
# feishu_task_add_comment
# ---------------------------------------------------------------------------

_TASK_COMMENT_URI = "/open-apis/task/v2/tasks/:task_guid/comments"

FEISHU_TASK_ADD_COMMENT_SCHEMA = {
    "name": "feishu_task_add_comment",
    "description": "Add a comment to a Feishu task.",
    "parameters": {
        "type": "object",
        "properties": {
            "task_id": {
                "type": "string",
                "description": "The task GUID to comment on.",
            },
            "content": {
                "type": "string",
                "description": "The comment text content (plain text).",
            },
        },
        "required": ["task_id", "content"],
    },
}


def _handle_task_add_comment(args: dict, **kwargs) -> str:
    task_id = (args.get("task_id") or "").strip()
    logger.info("feishu_task_add_comment: task_id=%s", task_id)
    client, err = _get_user_client()
    if err:
        return tool_error(err)

    content = (args.get("content") or "").strip()
    if not task_id:
        return tool_error("task_id is required")
    if not content:
        return tool_error("content is required")

    body = {"content": content}

    code, msg, data = _do_request(
        client, "POST", _TASK_COMMENT_URI,
        paths={"task_guid": task_id},
        queries=[("user_id_type", "open_id")],
        body=body,
    )
    if code != 0:
        try:
            raise_for_feishu_errcode(code, msg or "", api_name="feishu.task.add_comment")
        except (AppScopeMissingError, UserAuthRequiredError, UserScopeInsufficientError, NeedAuthorizationError) as e:
            return tool_error(_auth_error_message(e))
        logger.warning("feishu_task_add_comment failed: code=%d msg=%s", code, msg)
        return tool_error(f"Add comment failed: code={code} msg={msg}")

    logger.info("feishu_task_add_comment: commented on task %s", task_id)
    return tool_result(success=True, data=data)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

registry.register(
    name="feishu_task_list",
    toolset="feishu_task",
    schema=FEISHU_TASK_LIST_SCHEMA,
    handler=_handle_task_list,
    check_fn=_check_feishu,
    requires_env=["FEISHU_APP_ID", "FEISHU_APP_SECRET"],
    is_async=False,
    description="List Feishu tasks",
    emoji="\U00002705",
)

registry.register(
    name="feishu_task_get",
    toolset="feishu_task",
    schema=FEISHU_TASK_GET_SCHEMA,
    handler=_handle_task_get,
    check_fn=_check_feishu,
    requires_env=["FEISHU_APP_ID", "FEISHU_APP_SECRET"],
    is_async=False,
    description="Get a Feishu task by GUID",
    emoji="\U0001f4cb",
)

registry.register(
    name="feishu_task_create",
    toolset="feishu_task",
    schema=FEISHU_TASK_CREATE_SCHEMA,
    handler=_handle_task_create,
    check_fn=_check_feishu,
    requires_env=["FEISHU_APP_ID", "FEISHU_APP_SECRET"],
    is_async=False,
    description="Create a new Feishu task",
    emoji="\U0000270f️",
)

registry.register(
    name="feishu_task_update",
    toolset="feishu_task",
    schema=FEISHU_TASK_UPDATE_SCHEMA,
    handler=_handle_task_update,
    check_fn=_check_feishu,
    requires_env=["FEISHU_APP_ID", "FEISHU_APP_SECRET"],
    is_async=False,
    description="Update a Feishu task",
    emoji="\U0001f4dd",
)

registry.register(
    name="feishu_task_add_comment",
    toolset="feishu_task",
    schema=FEISHU_TASK_ADD_COMMENT_SCHEMA,
    handler=_handle_task_add_comment,
    check_fn=_check_feishu,
    requires_env=["FEISHU_APP_ID", "FEISHU_APP_SECRET"],
    is_async=False,
    description="Add a comment to a Feishu task",
    emoji="\U0001f4ac",
)
