#!/usr/bin/env python3
"""Android device helper tools for the Hermes Android app runtime."""

from __future__ import annotations

import json
import os

from hermes_android.device_bridge import (
    list_shared_folder_entries,
    perform_accessibility_action,
    perform_system_action,
    read_accessibility_snapshot,
    read_device_capabilities,
    read_shared_folder_file,
    write_shared_folder_file,
)
from tools.registry import registry


def check_requirements() -> bool:
    return bool(os.getenv("HERMES_ANDROID_BOOTSTRAP", "").strip())


def android_device_status_tool(task_id: str | None = None) -> str:
    del task_id
    return json.dumps(read_device_capabilities(), ensure_ascii=False)


def android_system_action_tool(action: str, task_id: str | None = None) -> str:
    del task_id
    return json.dumps(perform_system_action(action), ensure_ascii=False)


def android_shared_folder_list_tool(
    relative_path: str = "",
    recursive: bool = False,
    limit: int = 100,
    task_id: str | None = None,
) -> str:
    del task_id
    return json.dumps(
        list_shared_folder_entries(relative_path, recursive=recursive, limit=limit),
        ensure_ascii=False,
    )


def android_shared_folder_read_tool(
    relative_path: str,
    max_chars: int = 100_000,
    task_id: str | None = None,
) -> str:
    del task_id
    return json.dumps(
        read_shared_folder_file(relative_path, max_chars=max_chars),
        ensure_ascii=False,
    )


def android_shared_folder_write_tool(
    relative_path: str,
    content: str,
    create_directories: bool = True,
    task_id: str | None = None,
) -> str:
    del task_id
    return json.dumps(
        write_shared_folder_file(relative_path, content, create_directories=create_directories),
        ensure_ascii=False,
    )


def android_ui_snapshot_tool(limit: int = 80, task_id: str | None = None) -> str:
    del task_id
    return json.dumps(read_accessibility_snapshot(limit=limit), ensure_ascii=False)


def android_ui_action_tool(
    action: str,
    text_contains: str = "",
    content_description_contains: str = "",
    view_id: str = "",
    package_name: str = "",
    value: str = "",
    index: int = 0,
    task_id: str | None = None,
) -> str:
    del task_id
    selector_values = [text_contains.strip(), content_description_contains.strip(), view_id.strip(), package_name.strip()]
    if not any(selector_values):
        return json.dumps(
            {"error": "android_ui_action requires at least one selector: text_contains, content_description_contains, view_id, or package_name"},
            ensure_ascii=False,
        )
    if action.strip().lower() == "set_text" and not value:
        return json.dumps({"error": "android_ui_action with action='set_text' requires a non-empty value"}, ensure_ascii=False)

    return json.dumps(
        perform_accessibility_action(
            action=action,
            text_contains=text_contains,
            content_description_contains=content_description_contains,
            view_id=view_id,
            package_name=package_name,
            value=value,
            index=index,
        ),
        ensure_ascii=False,
    )


registry.register(
    name="android_device_status",
    toolset="hermes-android-app",
    schema={
        "name": "android_device_status",
        "description": (
            "Inspect Hermes Android workspace and device capabilities. Returns Linux command-suite status, "
            "shared-folder grant status, accessibility-control availability, plus Wi-Fi/Bluetooth/USB/NFC, "
            "overlay, notification, and background-runtime state."
        ),
        "parameters": {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
    },
    handler=lambda args, **kwargs: android_device_status_tool(task_id=kwargs.get("task_id")),
    check_fn=check_requirements,
    description="Inspect Hermes Android workspace and expanded device capability status",
    emoji="📱",
)

registry.register(
    name="android_system_action",
    toolset="hermes-android-app",
    schema={
        "name": "android_system_action",
        "description": (
            "Run a high-level Android system action for Hermes app control surfaces. Supported actions: "
            "open_wifi_panel, open_bluetooth_settings, open_connected_devices_settings, open_nfc_settings, "
            "open_notification_settings, open_overlay_settings, open_accessibility_settings, "
            "start_background_runtime, stop_background_runtime."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "action": {"type": "string", "description": "Required Android system action name."},
            },
            "required": ["action"],
            "additionalProperties": False,
        },
    },
    handler=lambda args, **kwargs: android_system_action_tool(
        action=str(args.get("action", "")),
        task_id=kwargs.get("task_id"),
    ),
    check_fn=check_requirements,
    description="Open Android system panels or control Hermes background runtime",
    emoji="🛰️",
)

registry.register(
    name="android_shared_folder_list",
    toolset="hermes-android-app",
    schema={
        "name": "android_shared_folder_list",
        "description": "List files and directories inside the granted Android shared folder.",
        "parameters": {
            "type": "object",
            "properties": {
                "relative_path": {"type": "string", "description": "Optional path inside the granted folder. Blank means the folder root."},
                "recursive": {"type": "boolean", "description": "When true, recurse into subdirectories until the result limit is hit."},
                "limit": {"type": "integer", "description": "Maximum number of entries to return (1-200)."},
            },
            "additionalProperties": False,
        },
    },
    handler=lambda args, **kwargs: android_shared_folder_list_tool(
        relative_path=str(args.get("relative_path", "")),
        recursive=bool(args.get("recursive", False)),
        limit=int(args.get("limit", 100) or 100),
        task_id=kwargs.get("task_id"),
    ),
    check_fn=check_requirements,
    description="List granted Android shared-folder entries",
    emoji="📂",
)

registry.register(
    name="android_shared_folder_read",
    toolset="hermes-android-app",
    schema={
        "name": "android_shared_folder_read",
        "description": "Read a UTF-8 text file directly from the granted Android shared folder.",
        "parameters": {
            "type": "object",
            "properties": {
                "relative_path": {"type": "string", "description": "Required path to the text file inside the granted shared folder."},
                "max_chars": {"type": "integer", "description": "Safety limit for returned characters."},
            },
            "required": ["relative_path"],
            "additionalProperties": False,
        },
    },
    handler=lambda args, **kwargs: android_shared_folder_read_tool(
        relative_path=str(args.get("relative_path", "")),
        max_chars=int(args.get("max_chars", 100_000) or 100_000),
        task_id=kwargs.get("task_id"),
    ),
    check_fn=check_requirements,
    description="Read a text file from the granted Android shared folder",
    emoji="📄",
)

registry.register(
    name="android_shared_folder_write",
    toolset="hermes-android-app",
    schema={
        "name": "android_shared_folder_write",
        "description": "Write UTF-8 text directly into the granted Android shared folder.",
        "parameters": {
            "type": "object",
            "properties": {
                "relative_path": {"type": "string", "description": "Required path to the target text file inside the granted shared folder."},
                "content": {"type": "string", "description": "Full file content to write."},
                "create_directories": {"type": "boolean", "description": "Create any missing parent directories when true."},
            },
            "required": ["relative_path", "content"],
            "additionalProperties": False,
        },
    },
    handler=lambda args, **kwargs: android_shared_folder_write_tool(
        relative_path=str(args.get("relative_path", "")),
        content=str(args.get("content", "")),
        create_directories=bool(args.get("create_directories", True)),
        task_id=kwargs.get("task_id"),
    ),
    check_fn=check_requirements,
    description="Write a text file into the granted Android shared folder",
    emoji="📝",
)

registry.register(
    name="android_ui_snapshot",
    toolset="hermes-android-app",
    schema={
        "name": "android_ui_snapshot",
        "description": "Inspect the visible Android accessibility tree so Hermes can target on-screen controls.",
        "parameters": {
            "type": "object",
            "properties": {
                "limit": {"type": "integer", "description": "Maximum number of visible nodes to return (1-200)."},
            },
            "additionalProperties": False,
        },
    },
    handler=lambda args, **kwargs: android_ui_snapshot_tool(
        limit=int(args.get("limit", 80) or 80),
        task_id=kwargs.get("task_id"),
    ),
    check_fn=check_requirements,
    description="Inspect the current Android accessibility tree",
    emoji="👁️",
)

registry.register(
    name="android_ui_action",
    toolset="hermes-android-app",
    schema={
        "name": "android_ui_action",
        "description": "Perform a targeted Android accessibility action on a visible UI control.",
        "parameters": {
            "type": "object",
            "properties": {
                "action": {"type": "string", "description": "One of: click, long_click, focus, set_text, scroll_forward, scroll_backward."},
                "text_contains": {"type": "string", "description": "Match nodes whose visible text contains this value."},
                "content_description_contains": {"type": "string", "description": "Match nodes whose content description contains this value."},
                "view_id": {"type": "string", "description": "Match nodes whose Android view ID contains this value."},
                "package_name": {"type": "string", "description": "Optional package name filter for the active UI."},
                "value": {"type": "string", "description": "Required text payload when action is set_text."},
                "index": {"type": "integer", "description": "Zero-based match index when multiple nodes satisfy the selector."},
            },
            "required": ["action"],
            "additionalProperties": False,
        },
    },
    handler=lambda args, **kwargs: android_ui_action_tool(
        action=str(args.get("action", "")),
        text_contains=str(args.get("text_contains", "")),
        content_description_contains=str(args.get("content_description_contains", "")),
        view_id=str(args.get("view_id", "")),
        package_name=str(args.get("package_name", "")),
        value=str(args.get("value", "")),
        index=int(args.get("index", 0) or 0),
        task_id=kwargs.get("task_id"),
    ),
    check_fn=check_requirements,
    description="Perform a targeted Android accessibility action",
    emoji="🤖",
)
