#!/usr/bin/env python3
"""Read-only Todoist Sync API tools.

Uses TODOIST_API_TOKEN (preferred) or TODOIST_TOKEN from the environment. These
wrappers are intentionally read-only: they list/probe tasks and never create,
update, complete, delete, or otherwise mutate Todoist data.
"""

from __future__ import annotations

import json
import logging
import os
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Optional

from tools.registry import registry

logger = logging.getLogger(__name__)

TODOIST_TOKEN_ENVS = ("TODOIST_API_TOKEN", "TODOIST_TOKEN")
TODOIST_SYNC_URL = "https://api.todoist.com/api/v1/sync"
TODOIST_TIMEOUT_SECONDS = 20
READ_ONLY_CAVEAT = (
    "Todoist access is read-only in this tool. Hermes must not create, update, "
    "complete, delete, or move Todoist tasks without explicit user approval and "
    "a separate write-capable workflow."
)


def _get_token() -> tuple[Optional[str], Optional[str]]:
    for name in TODOIST_TOKEN_ENVS:
        value = os.getenv(name, "").strip()
        if value:
            return value, name
    return None, None


def check_todoist_requirements() -> bool:
    token, _ = _get_token()
    return bool(token)


def _missing_secret_payload(tool: str) -> Dict[str, Any]:
    return {
        "success": False,
        "provider": "todoist",
        "tool": tool,
        "error": (
            "Missing TODOIST_API_TOKEN. Store the Todoist API token in "
            "Bitwarden Secrets Manager using the exact secret name "
            "TODOIST_API_TOKEN, then restart or reload Hermes."
        ),
        "missing_secret": "TODOIST_API_TOKEN",
        "caveat": READ_ONLY_CAVEAT,
    }


def _sync_request(resource_types: List[str]) -> Dict[str, Any]:
    token, token_env = _get_token()
    if not token:
        raise RuntimeError("missing_token")
    data = urllib.parse.urlencode(
        {
            "sync_token": "*",
            "resource_types": json.dumps(resource_types),
        }
    ).encode()
    req = urllib.request.Request(
        TODOIST_SYNC_URL,
        data=data,
        headers={"Authorization": f"Bearer {token}"},
    )
    with urllib.request.urlopen(req, timeout=TODOIST_TIMEOUT_SECONDS) as resp:
        payload = json.loads(resp.read().decode("utf-8", errors="replace"))
    payload["_token_env"] = token_env
    return payload


def _http_error_payload(tool: str, exc: urllib.error.HTTPError) -> Dict[str, Any]:
    try:
        body = exc.read().decode("utf-8", errors="replace")[:500]
    except Exception:
        body = ""
    return {
        "success": False,
        "provider": "todoist",
        "tool": tool,
        "status_code": exc.code,
        "error": f"Todoist Sync API request failed with HTTP {exc.code}.",
        "response_excerpt": body,
        "caveat": READ_ONLY_CAVEAT,
    }


def _request_error_payload(tool: str, exc: Exception) -> Dict[str, Any]:
    logger.warning("Todoist request failed: %s", exc)
    return {
        "success": False,
        "provider": "todoist",
        "tool": tool,
        "error": "Todoist Sync API request failed.",
        "error_type": type(exc).__name__,
        "caveat": READ_ONLY_CAVEAT,
    }


def _project_map(projects: List[Dict[str, Any]]) -> Dict[str, str]:
    return {str(project.get("id")): project.get("name", "") for project in projects}


def _normalize_task(item: Dict[str, Any], projects_by_id: Dict[str, str]) -> Dict[str, Any]:
    project_id = str(item.get("project_id") or "")
    return {
        "id": item.get("id"),
        "content": item.get("content"),
        "description": item.get("description"),
        "project_id": project_id,
        "project_name": projects_by_id.get(project_id, ""),
        "section_id": item.get("section_id"),
        "parent_id": item.get("parent_id"),
        "priority": item.get("priority"),
        "checked": bool(item.get("checked")),
        "is_deleted": bool(item.get("is_deleted")),
        "due": item.get("due"),
        "labels": item.get("labels", []),
        "child_order": item.get("child_order"),
    }


def todoist_read_only_probe() -> str:
    token, token_env = _get_token()
    if not token:
        return json.dumps(_missing_secret_payload("todoist_read_only_probe"))
    try:
        payload = _sync_request(["projects", "items", "labels", "sections"])
    except RuntimeError:
        return json.dumps(_missing_secret_payload("todoist_read_only_probe"))
    except urllib.error.HTTPError as exc:
        return json.dumps(_http_error_payload("todoist_read_only_probe", exc), ensure_ascii=False)
    except Exception as exc:
        return json.dumps(_request_error_payload("todoist_read_only_probe", exc), ensure_ascii=False)

    projects = payload.get("projects", []) if isinstance(payload.get("projects"), list) else []
    items = payload.get("items", []) if isinstance(payload.get("items"), list) else []
    labels = payload.get("labels", []) if isinstance(payload.get("labels"), list) else []
    sections = payload.get("sections", []) if isinstance(payload.get("sections"), list) else []
    return json.dumps(
        {
            "success": True,
            "provider": "todoist",
            "tool": "todoist_read_only_probe",
            "endpoint": TODOIST_SYNC_URL,
            "token_env": token_env,
            "counts": {
                "projects": len(projects),
                "items": len(items),
                "labels": len(labels),
                "sections": len(sections),
            },
            "sync_token_present": bool(payload.get("sync_token")),
            "projects": [
                {"id": project.get("id"), "name": project.get("name")}
                for project in projects[:10]
            ],
            "caveat": READ_ONLY_CAVEAT,
        },
        ensure_ascii=False,
    )


def todoist_list_tasks(*, limit: int = 25, include_completed: bool = False, project_name_contains: str = "") -> str:
    if not check_todoist_requirements():
        return json.dumps(_missing_secret_payload("todoist_list_tasks"))
    try:
        capped_limit = max(1, min(int(limit or 25), 100))
    except (TypeError, ValueError):
        capped_limit = 25
    try:
        payload = _sync_request(["projects", "items", "labels", "sections"])
    except RuntimeError:
        return json.dumps(_missing_secret_payload("todoist_list_tasks"))
    except urllib.error.HTTPError as exc:
        return json.dumps(_http_error_payload("todoist_list_tasks", exc), ensure_ascii=False)
    except Exception as exc:
        return json.dumps(_request_error_payload("todoist_list_tasks", exc), ensure_ascii=False)

    projects = payload.get("projects", []) if isinstance(payload.get("projects"), list) else []
    items = payload.get("items", []) if isinstance(payload.get("items"), list) else []
    projects_by_id = _project_map(projects)
    needle = project_name_contains.lower().strip()
    tasks: List[Dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        task = _normalize_task(item, projects_by_id)
        if not include_completed and task["checked"]:
            continue
        if task["is_deleted"]:
            continue
        if needle and needle not in task["project_name"].lower():
            continue
        tasks.append(task)
        if len(tasks) >= capped_limit:
            break

    return json.dumps(
        {
            "success": True,
            "provider": "todoist",
            "tool": "todoist_list_tasks",
            "tasks_count": len(tasks),
            "tasks": tasks,
            "caveat": READ_ONLY_CAVEAT,
        },
        ensure_ascii=False,
    )


TODOIST_READ_ONLY_PROBE_SCHEMA = {
    "name": "todoist_read_only_probe",
    "description": "Read-only Todoist Sync API probe. Verifies token and returns counts without mutating tasks.",
    "parameters": {"type": "object", "properties": {}},
}

TODOIST_LIST_TASKS_SCHEMA = {
    "name": "todoist_list_tasks",
    "description": "List Todoist tasks read-only using the current Sync API. Never creates/updates/completes/deletes tasks.",
    "parameters": {
        "type": "object",
        "properties": {
            "limit": {"type": "integer", "description": "Maximum tasks to return, 1-100.", "default": 25},
            "include_completed": {"type": "boolean", "description": "Include completed tasks.", "default": False},
            "project_name_contains": {"type": "string", "description": "Optional case-insensitive project name filter."},
        },
    },
}


def _handle_probe(args, **kwargs):
    return todoist_read_only_probe()


def _handle_list_tasks(args, **kwargs):
    return todoist_list_tasks(
        limit=args.get("limit", 25),
        include_completed=args.get("include_completed", False),
        project_name_contains=args.get("project_name_contains", ""),
    )


registry.register(
    name="todoist_read_only_probe",
    toolset="todoist",
    schema=TODOIST_READ_ONLY_PROBE_SCHEMA,
    handler=_handle_probe,
    check_fn=check_todoist_requirements,
    requires_env=["TODOIST_API_TOKEN"],
    emoji="✅",
    max_result_size_chars=20_000,
)

registry.register(
    name="todoist_list_tasks",
    toolset="todoist",
    schema=TODOIST_LIST_TASKS_SCHEMA,
    handler=_handle_list_tasks,
    check_fn=check_todoist_requirements,
    requires_env=["TODOIST_API_TOKEN"],
    emoji="✅",
    max_result_size_chars=40_000,
)
