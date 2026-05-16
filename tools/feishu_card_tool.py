"""Feishu Interactive Card Tool -- send Message Cards, files, and rich messages.

Provides:
  - feishu_send_card         : Send an interactive Message Card
  - feishu_send_message      : Send a plain text message
  - feishu_upload_file       : Upload a file and return the file key
  - feishu_send_file         : Send a file to a user or chat

Templates (CardElement helpers):
  - task_card       : Task notification card (pending/completed/overdue)
  - calendar_card  : Calendar/event invite card
  - generic_card   : Generic info card with field rows
  - card_div / card_text / card_hr / card_button : base element builders
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Optional

from tools.registry import registry, tool_error, tool_result

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

_thread_local_client = None


def set_client(client) -> None:
    """Store a lark client for the current thread."""
    global _thread_local_client
    _thread_local_client = client


def _load_env():
    env_path = os.path.expanduser("~/.hermes/.env")
    if os.path.exists(env_path):
        for line in open(env_path):
            if "=" in line and not line.startswith("#"):
                k, v = line.strip().split("=", 1)
                os.environ.setdefault(k, v)


def _get_client():
    if _thread_local_client:
        return _thread_local_client
    _load_env()
    import lark_oapi as lark

    return (
        lark.Client.builder()
        .app_id(os.environ.get("FEISHU_APP_ID", ""))
        .app_secret(os.environ.get("FEISHU_APP_SECRET", ""))
        .log_level(lark.LogLevel.WARNING)
        .build()
    )


# ---------------------------------------------------------------------------
# Core send functions
# ---------------------------------------------------------------------------


def _send_message_impl(
    receive_id: str, receive_id_type: str, msg_type: str, content: str | dict,
) -> dict:
    """Low-level send. Returns {"success": bool, "message_id": str, "error": str}."""
    try:
        from lark_oapi.api.im.v1 import CreateMessageRequest, CreateMessageRequestBody

        client = _get_client()
        content_str = json.dumps(content) if isinstance(content, dict) else content
        request = (
            CreateMessageRequest.builder()
            .receive_id_type(receive_id_type)
            .request_body(
                CreateMessageRequestBody.builder()
                .receive_id(receive_id)
                .msg_type(msg_type)
                .content(content_str)
                .build()
            )
            .build()
        )
        resp = client.im.v1.message.create(request)
        if not resp.success():
            return {"success": False, "error": resp.msg}
        return {"success": True, "message_id": resp.data.message_id}
    except Exception as e:
        logger.exception("send_message failed")
        return {"success": False, "error": str(e)}


# ---------------------------------------------------------------------------
# Card templates (base elements)
# ---------------------------------------------------------------------------


def _card_text(text: str, tag: str = "larkmd", **kwargs) -> dict:
    result: dict[str, Any] = {"tag": tag, "content": text}
    if kwargs.get("color"):
        result["color"] = kwargs["color"]
    if kwargs.get("bold"):
        result["bold"] = True
    return result


def _card_div(text: str = "", fields: list = None, **kwargs) -> dict:
    props: dict[str, Any] = {"tag": "div"}
    if fields:
        props["fields"] = fields
    if text:
        props["text"] = _card_text(text, **kwargs)
    return props


def _card_hr() -> dict:
    return {"tag": "hr"}


def _card_button(text: str, value: str, action_id: str) -> dict:
    return {
        "tag": "button",
        "text": _card_text(text, tag="plain_text"),
        "value": value,
        "action_id": action_id,
    }


def _build_card(header: dict, elements: list) -> dict:
    return {
        "schema": "2.0",
        "config": {"wide_screen_mode": True},
        "header": header,
        "elements": elements,
    }


# ---------------------------------------------------------------------------
# Card template builders
# ---------------------------------------------------------------------------


def task_card(
    task_title: str,
    status: str = "pending",
    due_date: str = "",
    assignee_name: str = "",
    task_url: str = "",
) -> dict:
    """Build a task notification card.

    Args:
        task_title: Task summary/title
        status: "pending" | "completed" | "overdue"
        due_date: Due date string (e.g. "2026-04-30")
        assignee_name: Name of person assigned
        task_url: Optional deep link to task
    """
    status_emoji = {"pending": "⏳", "completed": "✅", "overdue": "🔴"}.get(status, "📋")
    status_text = {
        "pending": "待处理",
        "completed": "已完成",
        "overdue": "已逾期",
    }.get(status, status)
    template_map = {"pending": "purple", "completed": "green", "overdue": "red"}

    header = {
        "title": _card_text(f"{status_emoji} {task_title}", tag="plain_text"),
        "subtitle": _card_text(f"状态: {status_text}", tag="plain_text"),
        "template": template_map.get(status, "blue"),
    }

    elements = []
    fields = []
    if assignee_name:
        fields.append({"tag": "field", "text": _card_text(f"👤 负责人: **{assignee_name}**")})
    if due_date:
        fields.append({"tag": "field", "text": _card_text(f"📅 截止: **{due_date}**")})
    if fields:
        elements.append({"tag": "div", "fields": fields, "text": None})

    elements.append(_card_hr())
    elements.append({"tag": "note", "elements": [_card_text(f"状态: {status_text}")]})

    if task_url:
        elements.append({
            "tag": "action",
            "actions": [{
                "tag": "button",
                "text": _card_text("打开任务", tag="plain_text"),
                "type": "primary",
                "value": task_url,
                "action_id": "open_task",
            }],
        })

    return _build_card(header, elements)


def calendar_card(
    event_title: str,
    start_time: str,
    end_time: str = "",
    location: str = "",
    attendees: list[str] | None = None,
    organizer: str = "",
    description: str = "",
) -> dict:
    """Build a calendar invite card.

    Args:
        event_title: Event summary
        start_time: Start time (e.g. "2026-04-30 09:00")
        end_time: End time (e.g. "2026-04-30 10:00")
        location: Location string
        attendees: List of attendee names
        organizer: Organizer name
        description: Markdown description body
    """
    header = {
        "title": _card_text(f"📅 {event_title}"),
        "template": "blue",
    }

    elements = []
    time_str = start_time + (f" ~ {end_time}" if end_time else "")
    elements.append({
        "tag": "div",
        "fields": [{"tag": "field", "text": _card_text(f"🕐 **时间**: {time_str}")}],
        "text": None,
    })
    if location:
        elements.append({
            "tag": "div",
            "fields": [{"tag": "field", "text": _card_text(f"📍 **地点**: {location}")}],
            "text": None,
        })
    if organizer:
        elements.append({
            "tag": "div",
            "fields": [{"tag": "field", "text": _card_text(f"👔 **组织者**: {organizer}")}],
            "text": None,
        })
    if attendees:
        elements.append({
            "tag": "div",
            "fields": [{"tag": "field", "text": _card_text(f"👥 **参与者**: {', '.join(attendees)}")}],
            "text": None,
        })

    elements.append(_card_hr())
    if description:
        elements.append({"tag": "div", "text": _card_text(description, tag="larkmd")})

    return _build_card(header, elements)


def generic_card(
    title: str,
    body_lines: list[tuple[str, str]] | None = None,
    description: str = "",
    status: str = "info",
    actions: list[dict] | None = None,
) -> dict:
    """Build a generic information card.

    Args:
        title: Card title
        body_lines: List of (label, value) tuples to show as field rows
        description: Markdown description body
        status: "info" | "success" | "warning" | "danger" — sets header color
        actions: List of action button dicts
    """
    template_map = {"info": "blue", "success": "green", "warning": "yellow", "danger": "red"}
    header = {"title": _card_text(title), "template": template_map.get(status, "blue")}

    elements = []
    if body_lines:
        fields = []
        for label, value in body_lines:
            fields.append({"tag": "field", "text": _card_text(f"**{label}**: {value}")})
            if len(fields) == 2:
                elements.append({"tag": "div", "fields": fields, "text": None})
                fields = []
        if fields:
            elements.append({"tag": "div", "fields": fields, "text": None})

    if description:
        elements.append({"tag": "div", "text": _card_text(description, tag="larkmd")})

    if actions:
        elements.append({"tag": "action", "actions": actions})

    return _build_card(header, elements)


# ---------------------------------------------------------------------------
# Tool schemas
# ---------------------------------------------------------------------------

FEISHU_SEND_CARD_SCHEMA = {
    "name": "feishu_send_card",
    "description": (
        "Send an interactive Feishu Message Card to a user or chat.\n\n"
        "Use this to send rich, interactive notifications with buttons, "
        "fields, and structured layouts — far richer than plain text.\n\n"
        "Args:\n"
        "  card (dict): Message Card JSON. Use task_card(), calendar_card(), "
        "or generic_card() helpers to build it, or provide raw card JSON.\n"
        "  receive_id (str): Feishu ID of the recipient — open_id, chat_id, or user_id.\n"
        "  receive_id_type (str, optional): Type of receive_id — "
        "'open_id' (default), 'chat_id', 'user_id', 'union_id', 'email'.\n"
        "Returns: {\"success\": bool, \"message_id\": str, \"error\": str}"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "card": {
                "type": "object",
                "description": "Message Card JSON dict. "
                "Use task_card(), calendar_card(), or generic_card() to build.",
            },
            "receive_id": {"type": "string", "description": "Recipient ID (open_id, chat_id, etc.)."},
            "receive_id_type": {
                "type": "string",
                "description": "Type of receive_id — open_id (default), chat_id, user_id, union_id, email.",
                "default": "open_id",
            },
        },
        "required": ["card", "receive_id"],
    },
}

FEISHU_SEND_MESSAGE_SCHEMA = {
    "name": "feishu_send_message",
    "description": (
        "Send a plain text message to a Feishu user or chat.\n\n"
        "For rich interactive cards, use feishu_send_card instead.\n\n"
        "Args:\n"
        "  text (str): Plain text content to send.\n"
        "  receive_id (str): Recipient ID.\n"
        "  receive_id_type (str, optional): Type — 'open_id' (default), 'chat_id', etc.\n"
        "Returns: {\"success\": bool, \"message_id\": str, \"error\": str}"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "text": {"type": "string", "description": "Plain text message content."},
            "receive_id": {"type": "string", "description": "Recipient ID."},
            "receive_id_type": {
                "type": "string",
                "description": "Type of receive_id — open_id (default), chat_id, etc.",
                "default": "open_id",
            },
        },
        "required": ["text", "receive_id"],
    },
}

FEISHU_UPLOAD_FILE_SCHEMA = {
    "name": "feishu_upload_file",
    "description": (
        "Upload a file to Feishu and get a file_key for subsequent use.\n\n"
        "Returns the file_key. To send the file to a user, use feishu_send_file instead.\n\n"
        "Args:\n"
        "  file_path (str): Absolute path to the local file.\n"
        "Returns: {\"file_key\": str, \"error\": str}"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "file_path": {"type": "string", "description": "Absolute path to the file to upload."},
        },
        "required": ["file_path"],
    },
}

FEISHU_SEND_FILE_SCHEMA = {
    "name": "feishu_send_file",
    "description": (
        "Send a file to a Feishu user or chat.\n\n"
        "Two-step: uploads the file to Feishu, then sends it as a file message.\n\n"
        "Args:\n"
        "  file_path (str): Absolute path to the local file.\n"
        "  receive_id (str): Recipient ID.\n"
        "  receive_id_type (str, optional): 'open_id' (default) or 'chat_id'.\n"
        "  file_name (str, optional): Display name for the file.\n"
        "Returns: {\"success\": bool, \"message_id\": str, \"error\": str}"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "file_path": {"type": "string", "description": "Absolute path to the file."},
            "receive_id": {"type": "string", "description": "Recipient ID."},
            "receive_id_type": {
                "type": "string",
                "description": "Type of receive_id — open_id (default), chat_id, etc.",
                "default": "open_id",
            },
            "file_name": {
                "type": "string",
                "description": "Optional display name for the file.",
            },
        },
        "required": ["file_path", "receive_id"],
    },
}


# ---------------------------------------------------------------------------
# Tool handlers
# ---------------------------------------------------------------------------


def _handle_feishu_send_card(args: dict, **kw) -> str:
    card = args.get("card")
    receive_id = args.get("receive_id", "").strip()
    receive_id_type = args.get("receive_id_type", "open_id").strip()

    if not card:
        return tool_error("card is required (Message Card JSON dict)")
    if not receive_id:
        return tool_error("receive_id is required")

    result = _send_message_impl(receive_id, receive_id_type, "interactive", card)
    if result.get("success"):
        return tool_result(success=True, content=f"Card sent. message_id={result['message_id']}")
    return tool_error(result.get("error", "Unknown error"))


def _handle_feishu_send_message(args: dict, **kw) -> str:
    text = args.get("text", "").strip()
    receive_id = args.get("receive_id", "").strip()
    receive_id_type = args.get("receive_id_type", "open_id").strip()

    if not text:
        return tool_error("text is required")
    if not receive_id:
        return tool_error("receive_id is required")

    result = _send_message_impl(receive_id, receive_id_type, "text", json.dumps({"text": text}))
    if result.get("success"):
        return tool_result(success=True, content=f"Message sent. message_id={result['message_id']}")
    return tool_error(result.get("error", "Unknown error"))


_FILE_TYPE_MAP = {
    ".pdf": "pdf",
    ".doc": "doc",
    ".docx": "doc",
    ".xls": "xls",
    ".xlsx": "xls",
    ".ppt": "ppt",
    ".pptx": "ppt",
    ".png": "image",
    ".jpg": "image",
    ".jpeg": "image",
    ".gif": "image",
    ".mp4": "mp4",
    ".mp3": "mp3",
}


def _resolve_file_type(file_path: str) -> str:
    ext = os.path.splitext(file_path)[-1].lower()
    return _FILE_TYPE_MAP.get(ext, "stream")


def _handle_feishu_upload_file(args: dict, **kw) -> str:
    file_path = args.get("file_path", "").strip()
    if not file_path:
        return tool_error("file_path is required")

    if not os.path.exists(file_path):
        return tool_error(f"File not found: {file_path}")

    try:
        from lark_oapi.api.im.v1 import CreateFileRequest, CreateFileRequestBody

        client = _get_client()
        file_name = os.path.basename(file_path)
        file_type = _resolve_file_type(file_path)

        with open(file_path, "rb") as f:
            request = (
                CreateFileRequest.builder()
                .request_body(
                    CreateFileRequestBody.builder()
                    .file_name(file_name)
                    .file_type(file_type)
                    .file(f)
                    .build()
                )
                .build()
            )
            resp = client.im.v1.create_file(request)

        if not resp.success():
            return tool_error(f"Upload failed: {resp.msg}")
        return tool_result(success=True, content=f"file_key={resp.data.file_key}")
    except Exception as e:
        logger.exception("upload_file crash")
        return tool_error(str(e))


def _handle_feishu_send_file(args: dict, **kw) -> str:
    file_path = args.get("file_path", "").strip()
    receive_id = args.get("receive_id", "").strip()
    receive_id_type = args.get("receive_id_type", "open_id").strip()
    file_name = args.get("file_name", "").strip() or None

    if not file_path:
        return tool_error("file_path is required")
    if not receive_id:
        return tool_error("receive_id is required")

    # Step 1: Upload
    upload_resp = _handle_feishu_upload_file({"file_path": file_path}, **kw)
    import re

    m = re.search(r"file_key=(\S+)", upload_resp)
    if not m:
        return upload_resp  # Error message already in tool_error format
    file_key = m.group(1)
    display_name = file_name or os.path.basename(file_path)

    # Step 2: Send
    content = json.dumps({"file_key": file_key, "file_name": display_name})
    result = _send_message_impl(receive_id, receive_id_type, "file", content)
    if result.get("success"):
        return tool_result(success=True, content=f"File sent. message_id={result['message_id']}")
    return tool_error(result.get("error", "Unknown error"))


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

for _name, _schema, _handler in [
    ("feishu_send_card", FEISHU_SEND_CARD_SCHEMA, _handle_feishu_send_card),
    ("feishu_send_message", FEISHU_SEND_MESSAGE_SCHEMA, _handle_feishu_send_message),
    ("feishu_upload_file", FEISHU_UPLOAD_FILE_SCHEMA, _handle_feishu_upload_file),
    ("feishu_send_file", FEISHU_SEND_FILE_SCHEMA, _handle_feishu_send_file),
]:
    registry.register(
        name=_name,
        toolset="feishu",
        schema=_schema,
        handler=_handler,
        check_fn=lambda _: True,
        emoji="📦",
    )