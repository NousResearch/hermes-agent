#!/usr/bin/env python3
"""Session Management Tools — list, archive, rename, delete sessions and folders.

Requires the Hermes dashboard API to be running (http://127.0.0.1:9119).

All tools make direct HTTP requests to the Hermes REST API. No LLM calls,
no external dependencies beyond Python stdlib.
"""

import json
import logging
import os
import sys
import urllib.request
import urllib.error
import urllib.parse
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_DASHBOARD_BASE = "http://127.0.0.1:9119"
_SESSION_TOKEN_ENV = "HERMES_DASHBOARD_SESSION_TOKEN"
_SESSION_HEADER = "X-Hermes-Session-Token"


def _dashboard_token() -> str:
    """Return the token shared with the dashboard backend, when available."""
    token = os.environ.get(_SESSION_TOKEN_ENV, "").strip()
    if token:
        return token
    # Standalone dashboard runs generate the token in the already-loaded
    # web_server module instead of exporting it through the environment.
    server = sys.modules.get("hermes_cli.web_server")
    return str(getattr(server, "_SESSION_TOKEN", "") or "").strip()


def _api_headers(*, json_body: bool = False) -> dict[str, str]:
    headers = {"Content-Type": "application/json"} if json_body else {}
    token = _dashboard_token()
    if token:
        headers[_SESSION_HEADER] = token
    return headers


def _api_get(path: str) -> Optional[dict]:
    """GET request to the Hermes dashboard API."""
    try:
        req = urllib.request.Request(
            f"{_DASHBOARD_BASE}{path}", method="GET", headers=_api_headers()
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            return json.loads(resp.read())
    except (urllib.error.URLError, urllib.error.HTTPError, ConnectionRefusedError, OSError) as e:
        logger.debug("Dashboard API GET %s failed: %s", path, e)
        return None


def _api_patch(path: str, body: dict) -> Optional[dict]:
    """PATCH request to the Hermes dashboard API."""
    try:
        data = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(
            f"{_DASHBOARD_BASE}{path}", data=data, method="PATCH",
            headers=_api_headers(json_body=True),
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            return json.loads(resp.read())
    except (urllib.error.URLError, urllib.error.HTTPError, ConnectionRefusedError, OSError) as e:
        logger.debug("Dashboard API PATCH %s failed: %s", path, e)
        return None


def _api_delete(path: str) -> Optional[dict]:
    """DELETE request to the Hermes dashboard API."""
    try:
        req = urllib.request.Request(
            f"{_DASHBOARD_BASE}{path}", method="DELETE", headers=_api_headers()
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            return json.loads(resp.read())
    except (urllib.error.URLError, urllib.error.HTTPError, ConnectionRefusedError, OSError) as e:
        logger.debug("Dashboard API DELETE %s failed: %s", path, e)
        return None


def _api_post(path: str, body: dict) -> Optional[dict]:
    """POST request to the Hermes dashboard API."""
    try:
        data = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(
            f"{_DASHBOARD_BASE}{path}", data=data, method="POST",
            headers=_api_headers(json_body=True),
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            return json.loads(resp.read())
    except (urllib.error.URLError, urllib.error.HTTPError, ConnectionRefusedError, OSError) as e:
        logger.debug("Dashboard API POST %s failed: %s", path, e)
        return None


def _check_dashboard() -> bool:
    """Check that the dashboard is reachable and the caller is authenticated."""
    return bool(_dashboard_token()) and _api_get("/api/sessions?limit=1") is not None


# ── Tool handlers ────────────────────────────────────────────────────


def _session_list(limit: int = 20, archived: str = "exclude",
                  order: str = "recent", profile: str = "") -> str:
    params = f"?limit={limit}&archived={archived}&order={order}"
    if profile:
        params += f"&profile={urllib.parse.quote(profile)}"
    result = _api_get(f"/api/sessions{params}")
    if result is None:
        return json.dumps({"success": False, "error": "Cannot reach Hermes dashboard API"})
    sessions = result.get("sessions", [])
    display = []
    for s in sessions[:limit]:
        sid = s.get("id", "")[:20]
        title = (s.get("title") or "(untitled)")[:40]
        arch = " [archived]" if s.get("archived") else ""
        model = (s.get("model") or "?")[:18]
        source = s.get("source", "?")
        msg_count = s.get("message_count", 0)
        display.append(f"  {sid:22s} {title:42s} {arch:12s} {msg_count:4d}msgs {model:18s} {source}")
    return json.dumps({
        "success": True,
        "count": len(display),
        "sessions": display,
        "message": f"Found {len(display)} session(s). Use session_archive/rename/delete on a session by id."
    }, ensure_ascii=False)


def _session_archive(session_id: str, archived: bool = True, profile: str = "") -> str:
    path = f"/api/sessions/{urllib.parse.quote(session_id)}"
    body = {"archived": archived}
    if profile:
        body["profile"] = profile
    result = _api_patch(path, body)
    if result is None:
        return json.dumps({"success": False, "error": "Cannot reach Hermes dashboard API"})
    action = "archived" if archived else "unarchived"
    return json.dumps({
        "success": result.get("ok", False),
        "message": f"Session {session_id[:20]} {action}.",
    }, ensure_ascii=False)


def _session_rename(session_id: str, title: str, profile: str = "") -> str:
    path = f"/api/sessions/{urllib.parse.quote(session_id)}"
    body = {"title": title}
    if profile:
        body["profile"] = profile
    result = _api_patch(path, body)
    if result is None:
        return json.dumps({"success": False, "error": "Cannot reach Hermes dashboard API"})
    return json.dumps({
        "success": result.get("ok", False),
        "message": f"Session {session_id[:20]} renamed to '{title}'.",
    }, ensure_ascii=False)


def _session_delete(session_id: str, profile: str = "") -> str:
    path = f"/api/sessions/{urllib.parse.quote(session_id)}"
    if profile:
        path += f"?profile={urllib.parse.quote(profile)}"
    result = _api_delete(path)
    if result is None:
        return json.dumps({"success": False, "error": "Cannot reach Hermes dashboard API"})
    return json.dumps({
        "success": result.get("ok", False),
        "message": f"Session {session_id[:20]} deleted." if result.get("ok") else f"Failed to delete session {session_id[:20]}.",
    }, ensure_ascii=False)


def _folder_list(profile: str = "") -> str:
    path = "/api/session-folders"
    if profile:
        path += f"?profile={urllib.parse.quote(profile)}"
    result = _api_get(path)
    if result is None:
        return json.dumps({"success": False, "error": "Cannot reach Hermes dashboard API"})
    if isinstance(result, list):
        folders = result
    else:
        folders = []
    display = []
    for f in folders:
        display.append(f"  {f['id'][:18]:20s} {f['name']:30s} {f['session_count']} sessions")
    return json.dumps({
        "success": True,
        "count": len(display),
        "folders": display,
    }, ensure_ascii=False)


def _folder_create(name: str, profile: str = "") -> str:
    body = {"name": name}
    if profile:
        body["profile"] = profile
    result = _api_post("/api/session-folders", body)
    if result is None:
        return json.dumps({"success": False, "error": "Cannot reach Hermes dashboard API"})
    return json.dumps({
        "success": True,
        "message": f"Folder '{name}' created.",
        "folder": result,
    }, ensure_ascii=False)


def _folder_add(folder_id: str, session_ids: List[str], profile: str = "") -> str:
    body = {"session_ids": session_ids}
    if profile:
        body["profile"] = profile
    result = _api_post(f"/api/session-folders/{urllib.parse.quote(folder_id)}/sessions", body)
    if result is None:
        return json.dumps({"success": False, "error": "Cannot reach Hermes dashboard API"})
    return json.dumps({
        "success": True,
        "message": f"Added {result.get('count', 0)} session(s) to folder.",
    }, ensure_ascii=False)


def _folder_rename(folder_id: str, name: str, profile: str = "") -> str:
    body = {"name": name}
    if profile:
        body["profile"] = profile
    result = _api_patch(f"/api/session-folders/{urllib.parse.quote(folder_id)}", body)
    if result is None:
        return json.dumps({"success": False, "error": "Cannot reach Hermes dashboard API"})
    return json.dumps({
        "success": result.get("ok", False),
        "message": f"Folder renamed to '{name}'." if result.get("ok") else "Failed to rename folder.",
    }, ensure_ascii=False)


def _folder_delete(folder_id: str, profile: str = "") -> str:
    path = f"/api/session-folders/{urllib.parse.quote(folder_id)}"
    if profile:
        path += f"?profile={urllib.parse.quote(profile)}"
    result = _api_delete(path)
    if result is None:
        return json.dumps({"success": False, "error": "Cannot reach Hermes dashboard API"})
    return json.dumps({
        "success": result.get("ok", False),
        "message": "Folder deleted." if result.get("ok") else "Failed to delete folder. Sessions are not affected.",
    }, ensure_ascii=False)


# ── Schemas ──────────────────────────────────────────────────────────


SESSION_LIST_SCHEMA = {
    "name": "session_list",
    "description": "List Hermes sessions with optional filters for archive status and sort order. Use to find session IDs for archive/rename/delete operations. For content search, use the session_search tool instead.",
    "parameters": {
        "type": "object",
        "properties": {
            "limit": {"type": "integer", "description": "Max results (default 20).", "default": 20},
            "archived": {"type": "string", "enum": ["exclude", "include", "only"], "description": "Archive filter."},
            "order": {"type": "string", "enum": ["recent", "created"], "description": "Sort order."},
            "profile": {"type": "string", "description": "Target profile name (required for cross-profile operations)."},
        },
    },
}

SESSION_ARCHIVE_SCHEMA = {
    "name": "session_archive",
    "description": "Archive or unarchive a session. Archiving hides it from the default session list.",
    "parameters": {
        "type": "object",
        "properties": {
            "session_id": {"type": "string", "description": "Session ID to archive/unarchive."},
            "archived": {"type": "boolean", "description": "True=archive, False=unarchive.", "default": True},
            "profile": {"type": "string", "description": "Target profile name."},
        },
        "required": ["session_id"],
    },
}

SESSION_RENAME_SCHEMA = {
    "name": "session_rename",
    "description": "Rename a session to a new title.",
    "parameters": {
        "type": "object",
        "properties": {
            "session_id": {"type": "string", "description": "Session ID to rename."},
            "title": {"type": "string", "description": "New title for the session."},
            "profile": {"type": "string", "description": "Target profile name."},
        },
        "required": ["session_id", "title"],
    },
}

SESSION_DELETE_SCHEMA = {
    "name": "session_delete",
    "description": "Permanently delete a session and all its messages. WARNING: This cannot be undone. The session and ALL its messages will be permanently removed. Use with extreme caution.",
    "parameters": {
        "type": "object",
        "properties": {
            "session_id": {"type": "string", "description": "Session ID to delete."},
            "profile": {"type": "string", "description": "Target profile name."},
        },
        "required": ["session_id"],
    },
}

FOLDER_LIST_SCHEMA = {
    "name": "session_folder_list",
    "description": "List all session folders with per-folder session counts.",
    "parameters": {
        "type": "object",
        "properties": {
            "profile": {"type": "string", "description": "Target profile name."},
        },
    },
}

FOLDER_CREATE_SCHEMA = {
    "name": "session_folder_create",
    "description": "Create a new named folder for organizing sessions.",
    "parameters": {
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "Name for the new folder."},
            "profile": {"type": "string", "description": "Target profile name."},
        },
        "required": ["name"],
    },
}

FOLDER_ADD_SCHEMA = {
    "name": "session_folder_add",
    "description": "Add one or more sessions to a folder.",
    "parameters": {
        "type": "object",
        "properties": {
            "folder_id": {"type": "string", "description": "Folder ID (get from session_folder_list)."},
            "session_ids": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Session IDs to add to the folder.",
            },
            "profile": {"type": "string", "description": "Target profile name."},
        },
        "required": ["folder_id", "session_ids"],
    },
}

FOLDER_RENAME_SCHEMA = {
    "name": "session_folder_rename",
    "description": "Rename a session folder.",
    "parameters": {
        "type": "object",
        "properties": {
            "folder_id": {"type": "string", "description": "Folder ID to rename."},
            "name": {"type": "string", "description": "New name for the folder."},
            "profile": {"type": "string", "description": "Target profile name."},
        },
        "required": ["folder_id", "name"],
    },
}

FOLDER_DELETE_SCHEMA = {
    "name": "session_folder_delete",
    "description": "Delete a session folder. Sessions in the folder are NOT deleted — only the folder is removed.",
    "parameters": {
        "type": "object",
        "properties": {
            "folder_id": {"type": "string", "description": "Folder ID to delete."},
            "profile": {"type": "string", "description": "Target profile name."},
        },
        "required": ["folder_id"],
    },
}


# ── Registry ─────────────────────────────────────────────────────────


from tools.registry import registry, tool_error

registry.register(
    name="session_list",
    toolset="session_management",
    schema=SESSION_LIST_SCHEMA,
    handler=lambda args, **kw: _session_list(
        limit=args.get("limit", 20),
        archived=args.get("archived", "exclude"),
        order=args.get("order", "recent"),
        profile=args.get("profile", ""),
    ),
    check_fn=_check_dashboard,
    emoji="📋",
)

registry.register(
    name="session_archive",
    toolset="session_management",
    schema=SESSION_ARCHIVE_SCHEMA,
    handler=lambda args, **kw: _session_archive(
        session_id=args["session_id"],
        archived=args.get("archived", True),
        profile=args.get("profile", ""),
    ),
    check_fn=_check_dashboard,
    emoji="📦",
)

registry.register(
    name="session_rename",
    toolset="session_management",
    schema=SESSION_RENAME_SCHEMA,
    handler=lambda args, **kw: _session_rename(
        session_id=args["session_id"],
        title=args["title"],
        profile=args.get("profile", ""),
    ),
    check_fn=_check_dashboard,
    emoji="✏️",
)

registry.register(
    name="session_delete",
    toolset="session_management",
    schema=SESSION_DELETE_SCHEMA,
    handler=lambda args, **kw: _session_delete(
        session_id=args["session_id"],
        profile=args.get("profile", ""),
    ),
    check_fn=_check_dashboard,
    emoji="🗑️",
)

registry.register(
    name="session_folder_list",
    toolset="session_management",
    schema=FOLDER_LIST_SCHEMA,
    handler=lambda args, **kw: _folder_list(profile=args.get("profile", "")),
    check_fn=_check_dashboard,
    emoji="📁",
)

registry.register(
    name="session_folder_create",
    toolset="session_management",
    schema=FOLDER_CREATE_SCHEMA,
    handler=lambda args, **kw: _folder_create(name=args["name"], profile=args.get("profile", "")),
    check_fn=_check_dashboard,
    emoji="➕📁",
)

registry.register(
    name="session_folder_add",
    toolset="session_management",
    schema=FOLDER_ADD_SCHEMA,
    handler=lambda args, **kw: _folder_add(
        folder_id=args["folder_id"],
        session_ids=args["session_ids"],
        profile=args.get("profile", ""),
    ),
    check_fn=_check_dashboard,
    emoji="📎",
)

registry.register(
    name="session_folder_rename",
    toolset="session_management",
    schema=FOLDER_RENAME_SCHEMA,
    handler=lambda args, **kw: _folder_rename(
        folder_id=args["folder_id"],
        name=args["name"],
        profile=args.get("profile", ""),
    ),
    check_fn=_check_dashboard,
    emoji="✏️📁",
)

registry.register(
    name="session_folder_delete",
    toolset="session_management",
    schema=FOLDER_DELETE_SCHEMA,
    handler=lambda args, **kw: _folder_delete(
        folder_id=args["folder_id"],
        profile=args.get("profile", ""),
    ),
    check_fn=_check_dashboard,
    emoji="🗑️📁",
)
