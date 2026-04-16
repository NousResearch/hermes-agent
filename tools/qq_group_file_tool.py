"""QQ/NapCat group file management tool."""

from __future__ import annotations

import json
import os
from pathlib import Path

from gateway.config import Platform
from gateway.platforms.qq_napcat import (
    normalize_qq_napcat_local_path,
    resolve_qq_napcat_group_id,
)
from tools.group_scope_helpers import resolve_group_chat_id
from tools.registry import registry, tool_error
from tools.send_message_tool import _check_send_message, _error, _qq_napcat_call


QQ_GROUP_FILE_SCHEMA = {
    "name": "qq_group_file",
    "description": (
        "Manage QQ/NapCat group files and folders. Supports listing files/folders, uploading a local file, "
        "deleting a group file by file_id + busid, creating/deleting a folder, reading file system info, "
        "resolving a file URL, moving files, renaming files, forwarding files to another QQ group, "
        "searching by name, uniquely resolving a single match, and safe resolved file or folder actions. Target must be a QQ group."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": [
                    "list", "upload", "delete",
                    "create_folder", "delete_folder",
                    "system_info", "get_url", "move", "rename", "forward", "find", "resolve", "resolve_folder",
                    "get_url_resolved", "delete_resolved", "forward_resolved", "move_resolved", "rename_resolved",
                    "delete_folder_resolved",
                ],
                "description": "Operation to perform on QQ group files.",
            },
            "target": {
                "type": "string",
                "description": (
                    "QQ group target. Accepts 'group:123456', 'qq_napcat:group:123456', or "
                    "a numeric group id. If omitted, Hermes will try the QQ home channel."
                ),
            },
            "folder_id": {
                "type": "string",
                "description": "Optional QQ folder id. For list, lists that folder instead of root. For upload, uploads into that folder. For delete_folder, identifies the folder to delete.",
            },
            "folder_name": {
                "type": "string",
                "description": "Folder name to create when action='create_folder'.",
            },
            "parent_id": {
                "type": "string",
                "description": "Parent folder id for create_folder. NapCat currently supports only '/'.",
            },
            "file_path": {
                "type": "string",
                "description": "Local filesystem path to upload when action='upload'.",
            },
            "file_name": {
                "type": "string",
                "description": "Optional display name to use when uploading. Defaults to the local basename.",
            },
            "file_id": {
                "type": "string",
                "description": "QQ group file id to delete when action='delete'.",
            },
            "busid": {
                "type": "integer",
                "description": "NapCat busid for the file when action='delete' or action='get_url'.",
            },
            "target_dir": {
                "type": "string",
                "description": "Target folder id/path when action='move'.",
            },
            "new_name": {
                "type": "string",
                "description": "New file name when action='rename'.",
            },
            "current_parent_directory": {
                "type": "string",
                "description": "Current parent directory of the file when action='rename'.",
            },
            "target_group_id": {
                "type": "string",
                "description": "Destination QQ group when action='forward'. Accepts 'group:123456' or numeric id.",
            },
            "query": {
                "type": "string",
                "description": "Search text when action='find'. Matches file/folder names.",
            },
            "include_folders": {
                "type": "boolean",
                "description": "Whether action='find' should include folder matches in addition to files.",
            },
            "recursive": {
                "type": "boolean",
                "description": "Whether action='find' should recurse into nested folders. Defaults to true.",
            },
            "exact": {
                "type": "boolean",
                "description": "Whether action='find' should require exact case-insensitive name equality.",
            },
            "max_results": {
                "type": "integer",
                "description": "Maximum number of matches to return for action='find'. Defaults to 50.",
            },
        },
        "required": ["action"],
    },
}


def qq_group_file_tool(args, **kw):
    """Handle QQ/NapCat group file operations."""
    del kw

    action = str(args.get("action") or "").strip().lower()
    if not action:
        return tool_error("'action' is required")
    if action not in {
        "list", "upload", "delete",
        "create_folder", "delete_folder",
        "system_info", "get_url", "move", "rename", "forward", "find", "resolve", "resolve_folder",
        "get_url_resolved", "delete_resolved", "forward_resolved", "move_resolved", "rename_resolved",
        "delete_folder_resolved",
    }:
        return tool_error(
            "Unsupported action. Use 'list', 'upload', 'delete', 'create_folder', 'delete_folder', 'system_info', 'get_url', 'move', 'rename', 'forward', 'find', 'resolve', 'resolve_folder', 'get_url_resolved', 'delete_resolved', 'forward_resolved', 'move_resolved', 'rename_resolved', or 'delete_folder_resolved'"
        )

    from tools.interrupt import is_interrupted
    if is_interrupted():
        return tool_error("Interrupted")

    try:
        from gateway.config import load_gateway_config

        config = load_gateway_config()
    except Exception as exc:
        return json.dumps(_error(f"Failed to load gateway config: {exc}"), ensure_ascii=False)

    pconfig = config.platforms.get(Platform.QQ_NAPCAT)
    if not pconfig or not pconfig.enabled:
        return tool_error(
            "Platform 'qq_napcat' is not configured. Set up NapCat credentials in ~/.hermes/config.yaml or environment variables."
        )

    try:
        group_id = _resolve_group_target(args.get("target"), config)
    except Exception as exc:
        return json.dumps(_error(str(exc)), ensure_ascii=False)

    try:
        from model_tools import _run_async

        result = _run_async(_dispatch_group_file_action(action, pconfig.extra, group_id, args))
        return json.dumps(result, ensure_ascii=False)
    except Exception as exc:
        return json.dumps(_error(f"QQ group file action failed: {exc}"), ensure_ascii=False)


def _resolve_group_target(target, config) -> str:
    """Resolve an explicit target or QQ home channel into a group id string."""
    home = config.get_home_channel(Platform.QQ_NAPCAT)
    return resolve_group_chat_id(
        target,
        expected_platform="qq_napcat",
        explicit_resolver=lambda value: str(resolve_qq_napcat_group_id(value)),
        home_chat_id=(home.chat_id if home else None),
        missing_target_error=(
            "No QQ group target specified. Use target='group:<id>' or configure a QQ group home channel."
        ),
    )


def _parse_bool_arg(value, *, arg_name: str, default: bool):
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().casefold()
        if normalized in {"true", "1", "yes", "y", "on"}:
            return True
        if normalized in {"false", "0", "no", "n", "off"}:
            return False
    return _error(f"'{arg_name}' must be a boolean")


async def _dispatch_group_file_action(action: str, extra: dict, group_id: str, args: dict) -> dict:
    if action == "list":
        return await _list_group_files(extra, group_id, args.get("folder_id"))
    if action == "upload":
        return await _upload_group_file(extra, group_id, args)
    if action == "delete":
        return await _delete_group_file(extra, group_id, args)
    if action == "create_folder":
        return await _create_group_folder(extra, group_id, args)
    if action == "delete_folder":
        return await _delete_group_folder(extra, group_id, args)
    if action == "system_info":
        return await _get_group_file_system_info(extra, group_id)
    if action == "get_url":
        return await _get_group_file_url(extra, group_id, args)
    if action == "move":
        return await _move_group_file(extra, group_id, args)
    if action == "rename":
        return await _rename_group_file(extra, group_id, args)
    if action == "find":
        return await _find_group_files(extra, group_id, args)
    if action == "resolve":
        return await _resolve_group_entry(extra, group_id, args)
    if action == "resolve_folder":
        return await _resolve_group_folder(extra, group_id, args)
    if action == "get_url_resolved":
        return await _get_group_file_url_resolved(extra, group_id, args)
    if action == "delete_resolved":
        return await _delete_group_file_resolved(extra, group_id, args)
    if action == "delete_folder_resolved":
        return await _delete_group_folder_resolved(extra, group_id, args)
    if action == "forward_resolved":
        return await _forward_group_file_resolved(extra, group_id, args)
    if action == "move_resolved":
        return await _move_group_file_resolved(extra, group_id, args)
    if action == "rename_resolved":
        return await _rename_group_file_resolved(extra, group_id, args)
    return await _forward_group_file(extra, group_id, args)


async def _list_group_files(extra: dict, group_id: str, folder_id) -> dict:
    numeric_group_id = int(group_id)
    folder_value = str(folder_id or "").strip() or None
    if folder_value and folder_value != "/":
        data, error = await _qq_napcat_call(
            extra,
            "get_group_files_by_folder",
            {"group_id": numeric_group_id, "folder_id": folder_value},
        )
    else:
        data, error = await _qq_napcat_call(
            extra,
            "get_group_root_files",
            {"group_id": numeric_group_id},
        )
    if error:
        return error

    payload = data or {}
    return {
        "success": True,
        "platform": "qq_napcat",
        "action": "list",
        "group_id": str(numeric_group_id),
        "folder_id": folder_value,
        "files": payload.get("files") or [],
        "folders": payload.get("folders") or [],
        "raw_response": payload,
    }


async def _upload_group_file(extra: dict, group_id: str, args: dict) -> dict:
    raw_path = str(args.get("file_path") or "").strip()
    if not raw_path:
        return _error("'file_path' is required when action='upload'")

    local_path = normalize_qq_napcat_local_path(raw_path)
    if not os.path.exists(local_path):
        return _error(f"QQ group file not found: {local_path}")

    file_name = str(args.get("file_name") or "").strip() or Path(local_path).name
    folder_value = str(args.get("folder_id") or "").strip() or None
    api_folder_value = None if folder_value in (None, "/") else folder_value
    params = {
        "group_id": int(group_id),
        "file": local_path,
        "name": file_name,
    }
    if api_folder_value:
        params["folder"] = api_folder_value

    data, error = await _qq_napcat_call(extra, "upload_group_file", params)
    if error:
        return error

    return {
        "success": True,
        "platform": "qq_napcat",
        "action": "upload",
        "group_id": str(group_id),
        "folder_id": folder_value,
        "file_name": file_name,
        "file_path": local_path,
        "uploaded": True,
        "raw_response": data or {},
    }


async def _delete_group_file(extra: dict, group_id: str, args: dict) -> dict:
    file_id = str(args.get("file_id") or "").strip()
    if not file_id:
        return _error("'file_id' is required when action='delete'")

    busid = args.get("busid")
    if busid in (None, ""):
        return _error("'busid' is required when action='delete'")

    try:
        busid_value = int(busid)
    except (TypeError, ValueError):
        return _error("'busid' must be an integer")

    data, error = await _qq_napcat_call(
        extra,
        "delete_group_file",
        {"group_id": int(group_id), "file_id": file_id, "busid": busid_value},
    )
    if error:
        return error

    return {
        "success": True,
        "platform": "qq_napcat",
        "action": "delete",
        "group_id": str(group_id),
        "file_id": file_id,
        "busid": busid_value,
        "deleted": True,
        "raw_response": data or {},
    }


async def _create_group_folder(extra: dict, group_id: str, args: dict) -> dict:
    folder_name = str(args.get("folder_name") or "").strip()
    if not folder_name:
        return _error("'folder_name' is required when action='create_folder'")

    parent_id = str(args.get("parent_id") or "").strip() or "/"
    if parent_id != "/":
        return _error("NapCat group folder creation currently supports only the root parent_id '/'")

    data, error = await _qq_napcat_call(
        extra,
        "create_group_file_folder",
        {"group_id": int(group_id), "name": folder_name, "parent_id": parent_id},
    )
    if error:
        return error

    return {
        "success": True,
        "platform": "qq_napcat",
        "action": "create_folder",
        "group_id": str(group_id),
        "folder_name": folder_name,
        "parent_id": parent_id,
        "created": True,
        "raw_response": data or {},
    }


async def _delete_group_folder(extra: dict, group_id: str, args: dict) -> dict:
    folder_id = str(args.get("folder_id") or "").strip()
    if not folder_id:
        return _error("'folder_id' is required when action='delete_folder'")

    data, error = await _qq_napcat_call(
        extra,
        "delete_group_folder",
        {"group_id": int(group_id), "folder_id": folder_id},
    )
    if error:
        return error

    return {
        "success": True,
        "platform": "qq_napcat",
        "action": "delete_folder",
        "group_id": str(group_id),
        "folder_id": folder_id,
        "deleted": True,
        "raw_response": data or {},
    }


def _require_file_id(args: dict, *, action: str) -> str | dict:
    file_id = str(args.get("file_id") or "").strip()
    if not file_id:
        return _error(f"'file_id' is required when action='{action}'")
    return file_id


def _require_busid(args: dict, *, action: str) -> int | dict:
    busid = args.get("busid")
    if busid in (None, ""):
        return _error(f"'busid' is required when action='{action}'")
    try:
        return int(busid)
    except (TypeError, ValueError):
        return _error("'busid' must be an integer")


async def _get_group_file_system_info(extra: dict, group_id: str) -> dict:
    data, error = await _qq_napcat_call(
        extra,
        "get_group_file_system_info",
        {"group_id": int(group_id)},
    )
    if error:
        return error
    return {
        "success": True,
        "platform": "qq_napcat",
        "action": "system_info",
        "group_id": str(group_id),
        "raw_response": data or {},
    }


async def _get_group_file_url(extra: dict, group_id: str, args: dict) -> dict:
    file_id = _require_file_id(args, action="get_url")
    if isinstance(file_id, dict):
        return file_id
    busid = _require_busid(args, action="get_url")
    if isinstance(busid, dict):
        return busid

    data, error = await _qq_napcat_call(
        extra,
        "get_group_file_url",
        {"group_id": int(group_id), "file_id": file_id, "busid": busid},
    )
    if error:
        return error

    payload = data or {}
    return {
        "success": True,
        "platform": "qq_napcat",
        "action": "get_url",
        "group_id": str(group_id),
        "file_id": file_id,
        "busid": busid,
        "url": payload.get("url"),
        "raw_response": payload,
    }


async def _move_group_file(extra: dict, group_id: str, args: dict) -> dict:
    file_id = _require_file_id(args, action="move")
    if isinstance(file_id, dict):
        return file_id
    target_dir = str(args.get("target_dir") or "").strip()
    if not target_dir:
        return _error("'target_dir' is required when action='move'")

    data, error = await _qq_napcat_call(
        extra,
        "move_group_file",
        {"group_id": int(group_id), "file_id": file_id, "target_dir": target_dir},
    )
    if error:
        return error

    return {
        "success": True,
        "platform": "qq_napcat",
        "action": "move",
        "group_id": str(group_id),
        "file_id": file_id,
        "target_dir": target_dir,
        "moved": True,
        "raw_response": data or {},
    }


async def _rename_group_file(extra: dict, group_id: str, args: dict) -> dict:
    file_id = _require_file_id(args, action="rename")
    if isinstance(file_id, dict):
        return file_id
    current_parent_directory = str(args.get("current_parent_directory") or "").strip()
    if not current_parent_directory:
        return _error("'current_parent_directory' is required when action='rename'")
    new_name = str(args.get("new_name") or "").strip()
    if not new_name:
        return _error("'new_name' is required when action='rename'")

    data, error = await _qq_napcat_call(
        extra,
        "rename_group_file",
        {
            "group_id": int(group_id),
            "file_id": file_id,
            "current_parent_directory": current_parent_directory,
            "new_name": new_name,
        },
    )
    if error:
        return error

    return {
        "success": True,
        "platform": "qq_napcat",
        "action": "rename",
        "group_id": str(group_id),
        "file_id": file_id,
        "current_parent_directory": current_parent_directory,
        "new_name": new_name,
        "renamed": True,
        "raw_response": data or {},
    }


async def _forward_group_file(extra: dict, group_id: str, args: dict) -> dict:
    file_id = _require_file_id(args, action="forward")
    if isinstance(file_id, dict):
        return file_id
    raw_target_group_id = args.get("target_group_id")
    if not str(raw_target_group_id or "").strip():
        return _error("'target_group_id' is required when action='forward'")
    try:
        target_group_id = resolve_qq_napcat_group_id(raw_target_group_id)
    except Exception as exc:
        return _error(str(exc))

    data, error = await _qq_napcat_call(
        extra,
        "trans_group_file",
        {"group_id": int(group_id), "file_id": file_id, "target_group_id": target_group_id},
    )
    if error:
        return error

    return {
        "success": True,
        "platform": "qq_napcat",
        "action": "forward",
        "group_id": str(group_id),
        "file_id": file_id,
        "target_group_id": str(target_group_id),
        "forwarded": True,
        "raw_response": data or {},
    }


def _entry_name(entry: dict, *, entry_type: str) -> str:
    if entry_type == "folder":
        return str(
            entry.get("folder_name")
            or entry.get("name")
            or entry.get("folderName")
            or ""
        ).strip()
    return str(
        entry.get("file_name")
        or entry.get("name")
        or entry.get("fileName")
        or ""
    ).strip()


async def _list_group_directory_payload(extra: dict, group_id: str, folder_id: str | None):
    if folder_id in (None, "/", ""):
        return await _qq_napcat_call(
            extra,
            "get_group_root_files",
            {"group_id": int(group_id)},
        )
    return await _qq_napcat_call(
        extra,
        "get_group_files_by_folder",
        {"group_id": int(group_id), "folder_id": folder_id},
    )


def _matches_query(name: str, query: str, *, exact: bool) -> bool:
    left = str(name or "").casefold()
    right = str(query or "").casefold()
    if exact:
        return left == right
    return right in left


async def _find_group_files(extra: dict, group_id: str, args: dict) -> dict:
    result = await _search_group_entries(extra, group_id, args, action="find")
    if "error" in result:
        return result

    return {
        "success": True,
        "platform": "qq_napcat",
        "action": "find",
        "group_id": str(group_id),
        "query": result["query"],
        "include_folders": result["include_folders"],
        "recursive": result["recursive"],
        "exact": result["exact"],
        "match_count": len(result["matches"]),
        "searched_folder_ids": result["searched_folder_ids"],
        "matches": result["matches"],
        "truncated": result["truncated"],
    }


async def _search_group_entries(extra: dict, group_id: str, args: dict, *, action: str) -> dict:
    query = str(args.get("query") or "").strip()
    if not query:
        return _error(f"'query' is required when action='{action}'")

    include_files = _parse_bool_arg(args.get("include_files"), arg_name="include_files", default=True)
    if isinstance(include_files, dict):
        return include_files
    include_folders = _parse_bool_arg(args.get("include_folders"), arg_name="include_folders", default=False)
    if isinstance(include_folders, dict):
        return include_folders
    recursive = _parse_bool_arg(args.get("recursive"), arg_name="recursive", default=True)
    if isinstance(recursive, dict):
        return recursive
    exact = _parse_bool_arg(args.get("exact"), arg_name="exact", default=False)
    if isinstance(exact, dict):
        return exact

    raw_max_results = args.get("max_results", 50)
    try:
        max_results = int(raw_max_results)
    except (TypeError, ValueError):
        return _error("'max_results' must be an integer")
    if max_results <= 0:
        return _error("'max_results' must be greater than 0")
    search_limit = max_results + 1

    start_folder_id = str(args.get("folder_id") or "").strip() or "/"
    pending = [start_folder_id]
    visited: set[str] = set()
    searched_folder_ids: list[str] = []
    matches: list[dict] = []

    while pending and len(matches) < search_limit:
        current_folder_id = pending.pop(0)
        if current_folder_id in visited:
            continue
        visited.add(current_folder_id)
        searched_folder_ids.append(current_folder_id)

        data, error = await _list_group_directory_payload(extra, group_id, current_folder_id)
        if error:
            return error

        payload = data or {}
        files = payload.get("files") or []
        folders = payload.get("folders") or []

        if include_files:
            for entry in files:
                name = _entry_name(entry, entry_type="file")
                if not _matches_query(name, query, exact=exact):
                    continue
                matches.append(
                    {
                        "entry_type": "file",
                        "name": name,
                        "file_id": entry.get("file_id"),
                        "busid": entry.get("busid"),
                        "current_parent_directory": current_folder_id,
                        "raw_entry": entry,
                    }
                )
                if len(matches) >= search_limit:
                    break

        if len(matches) >= search_limit:
            break

        for entry in folders:
            folder_name = _entry_name(entry, entry_type="folder")
            folder_id = str(entry.get("folder_id") or "").strip()
            if include_folders and _matches_query(folder_name, query, exact=exact):
                matches.append(
                    {
                        "entry_type": "folder",
                        "name": folder_name,
                        "folder_id": folder_id,
                        "parent_id": current_folder_id,
                        "raw_entry": entry,
                    }
                )
                if len(matches) >= search_limit:
                    break

            if recursive and folder_id and folder_id not in visited and folder_id not in pending:
                pending.append(folder_id)

        if len(matches) >= search_limit:
            break

    truncated = len(matches) > max_results

    return {
        "query": query,
        "include_files": include_files,
        "include_folders": include_folders,
        "recursive": recursive,
        "exact": exact,
        "searched_folder_ids": searched_folder_ids,
        "matches": matches[:max_results],
        "truncated": truncated,
    }


async def _search_group_entries_for_resolution(
    extra: dict,
    group_id: str,
    args: dict,
    *,
    action: str,
    include_files: bool,
    include_folders: bool,
) -> dict:
    return await _search_group_entries(
        extra,
        group_id,
        {
            **args,
            "include_files": include_files,
            "include_folders": include_folders,
            # Resolve-style actions only need to distinguish 0/1/many.
            "max_results": 2,
        },
        action=action,
    )


async def _resolve_group_entry(extra: dict, group_id: str, args: dict) -> dict:
    resolve_args = dict(args)
    include_files = _parse_bool_arg(resolve_args.get("include_files"), arg_name="include_files", default=True)
    if isinstance(include_files, dict):
        return include_files
    include_folders = _parse_bool_arg(resolve_args.get("include_folders"), arg_name="include_folders", default=True)
    if isinstance(include_folders, dict):
        return include_folders
    search = await _search_group_entries_for_resolution(
        extra,
        group_id,
        resolve_args,
        action="resolve",
        include_files=include_files,
        include_folders=include_folders,
    )
    if "error" in search:
        return search

    matches = search["matches"]
    match_count = len(matches)
    inspection_hint = (
        "Use action='find' with include_folders=true for broader inspection."
        if search["include_folders"]
        else "Use action='find' for broader inspection."
    )
    candidate_hint = (
        "Refine the query or use action='find' with include_folders=true to inspect candidates."
        if search["include_folders"]
        else "Refine the query or use action='find' to inspect candidates."
    )

    if match_count == 1:
        return {
            "success": True,
            "platform": "qq_napcat",
            "action": "resolve",
            "group_id": str(group_id),
            "query": search["query"],
            "match_count": 1,
            "searched_folder_ids": search["searched_folder_ids"],
            "match": matches[0],
        }

    if match_count == 0:
        return {
            "error": (
                f"No QQ group file or folder matched '{search['query']}'. "
                f"{inspection_hint}"
            ),
            "group_id": str(group_id),
            "query": search["query"],
            "match_count": 0,
            "searched_folder_ids": search["searched_folder_ids"],
            "matches": [],
        }

    return {
        "error": (
            f"Multiple QQ group entries matched '{search['query']}'. "
            f"{candidate_hint}"
        ),
        "group_id": str(group_id),
        "query": search["query"],
        "match_count": match_count,
        "searched_folder_ids": search["searched_folder_ids"],
        "matches": matches,
        "truncated": search["truncated"],
    }


async def _resolve_unique_file_match(extra: dict, group_id: str, args: dict, *, action: str):
    resolved = await _resolve_group_entry(
        extra,
        group_id,
        {
            **args,
            "include_files": True,
            "include_folders": False,
        },
    )
    if "error" in resolved:
        return resolved

    match = resolved.get("match") or {}
    if match.get("entry_type") != "file":
        return _error(f"action='{action}' requires a file match")
    return {
        "resolved": resolved,
        "match": match,
    }


async def _resolve_unique_folder_match(extra: dict, group_id: str, args: dict, *, action: str):
    search = await _search_group_entries_for_resolution(
        extra,
        group_id,
        args,
        action=action,
        include_files=False,
        include_folders=True,
    )
    if "error" in search:
        return search

    folder_matches = [match for match in search["matches"] if match.get("entry_type") == "folder"]
    folder_match_count = len(folder_matches)

    if folder_match_count == 1:
        resolved = {
            "success": True,
            "platform": "qq_napcat",
            "action": "resolve",
            "group_id": str(group_id),
            "query": search["query"],
            "match_count": 1,
            "searched_folder_ids": search["searched_folder_ids"],
            "match": folder_matches[0],
        }
        return {
            "resolved": resolved,
            "match": folder_matches[0],
        }

    if folder_match_count == 0:
        if search["matches"]:
            return {
                "error": f"action='{action}' requires a folder match",
                "group_id": str(group_id),
                "query": search["query"],
                "match_count": len(search["matches"]),
                "searched_folder_ids": search["searched_folder_ids"],
                "matches": search["matches"],
                "truncated": search["truncated"],
            }
        return {
            "error": (
                f"No QQ group folder matched '{search['query']}'. "
                "Use action='find' with include_folders=true for broader inspection."
            ),
            "group_id": str(group_id),
            "query": search["query"],
            "match_count": 0,
            "searched_folder_ids": search["searched_folder_ids"],
            "matches": [],
        }

    resolved = {
        "error": (
            f"Multiple QQ group folders matched '{search['query']}'. "
            "Refine the query or use action='find' with include_folders=true to inspect candidates."
        ),
        "group_id": str(group_id),
        "query": search["query"],
        "match_count": folder_match_count,
        "searched_folder_ids": search["searched_folder_ids"],
        "matches": folder_matches,
        "truncated": search["truncated"],
    }
    return resolved


async def _resolve_group_folder(extra: dict, group_id: str, args: dict) -> dict:
    resolved_data = await _resolve_unique_folder_match(extra, group_id, args, action="resolve_folder")
    if "error" in resolved_data:
        return resolved_data

    resolved = dict(resolved_data["resolved"])
    resolved["action"] = "resolve_folder"
    return resolved


async def _get_group_file_url_resolved(extra: dict, group_id: str, args: dict) -> dict:
    resolved_data = await _resolve_unique_file_match(extra, group_id, args, action="get_url_resolved")
    if "error" in resolved_data:
        return resolved_data

    match = resolved_data["match"]
    result = await _get_group_file_url(
        extra,
        group_id,
        {
            **args,
            "file_id": match.get("file_id"),
            "busid": match.get("busid"),
        },
    )
    if "error" in result:
        return result
    result["action"] = "get_url_resolved"
    result["resolved_match"] = match
    return result


async def _delete_group_file_resolved(extra: dict, group_id: str, args: dict) -> dict:
    resolved_data = await _resolve_unique_file_match(extra, group_id, args, action="delete_resolved")
    if "error" in resolved_data:
        return resolved_data

    match = resolved_data["match"]
    result = await _delete_group_file(
        extra,
        group_id,
        {
            **args,
            "file_id": match.get("file_id"),
            "busid": match.get("busid"),
        },
    )
    if "error" in result:
        return result
    result["action"] = "delete_resolved"
    result["resolved_match"] = match
    return result


async def _forward_group_file_resolved(extra: dict, group_id: str, args: dict) -> dict:
    resolved_data = await _resolve_unique_file_match(extra, group_id, args, action="forward_resolved")
    if "error" in resolved_data:
        return resolved_data

    match = resolved_data["match"]
    result = await _forward_group_file(
        extra,
        group_id,
        {
            **args,
            "file_id": match.get("file_id"),
        },
    )
    if "error" in result:
        return result
    result["action"] = "forward_resolved"
    result["resolved_match"] = match
    return result


async def _move_group_file_resolved(extra: dict, group_id: str, args: dict) -> dict:
    resolved_data = await _resolve_unique_file_match(extra, group_id, args, action="move_resolved")
    if "error" in resolved_data:
        return resolved_data

    match = resolved_data["match"]
    result = await _move_group_file(
        extra,
        group_id,
        {
            **args,
            "file_id": match.get("file_id"),
        },
    )
    if "error" in result:
        return result
    result["action"] = "move_resolved"
    result["resolved_match"] = match
    return result


async def _rename_group_file_resolved(extra: dict, group_id: str, args: dict) -> dict:
    resolved_data = await _resolve_unique_file_match(extra, group_id, args, action="rename_resolved")
    if "error" in resolved_data:
        return resolved_data

    match = resolved_data["match"]
    result = await _rename_group_file(
        extra,
        group_id,
        {
            **args,
            "file_id": match.get("file_id"),
            "current_parent_directory": match.get("current_parent_directory"),
        },
    )
    if "error" in result:
        return result
    result["action"] = "rename_resolved"
    result["resolved_match"] = match
    return result


async def _delete_group_folder_resolved(extra: dict, group_id: str, args: dict) -> dict:
    resolved_data = await _resolve_unique_folder_match(extra, group_id, args, action="delete_folder_resolved")
    if "error" in resolved_data:
        return resolved_data

    match = resolved_data["match"]
    result = await _delete_group_folder(
        extra,
        group_id,
        {
            **args,
            "folder_id": match.get("folder_id"),
        },
    )
    if "error" in result:
        return result
    result["action"] = "delete_folder_resolved"
    result["resolved_match"] = match
    return result


registry.register(
    name="qq_group_file",
    toolset="messaging",
    schema=QQ_GROUP_FILE_SCHEMA,
    handler=qq_group_file_tool,
    check_fn=_check_send_message,
    emoji="🗂️",
)
