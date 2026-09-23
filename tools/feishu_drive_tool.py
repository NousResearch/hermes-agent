"""Feishu Drive Tools -- document comment operations via Feishu/Lark API.

List / reply-to / add document comments through the generic BaseRequest path (lazy SDK
import), sharing client/request plumbing with feishu_doc_tool via ``tools.feishu_lark``.
The lark client is injected per-thread by the feishu_comment event handler.
"""

import logging

from tools.feishu_lark import (  # noqa: F401  (set_client/get_client are imported by feishu_comment)
    _check_feishu,
    build_request,
    get_client,
    lark_call,
    set_client)
from tools.registry import registry, tool_error, tool_result

logger = logging.getLogger(__name__)


def _comment_op(keys, missing_msg, label, method, uri, queries=None, body=None, flag_success=False):
    """Handler factory: client check, then required ``keys`` (stripped) → lark_call → result.

    All keys except ``content`` are URI path params. ``queries(args)`` / ``body(args, values)``
    build the request pieces; ``flag_success`` adds ``success=True`` to the result (writes).
    """
    def _handler(args: dict, **kwargs) -> str:
        client = get_client()
        values = tuple(args.get(k, "").strip() for k in keys)
        if client is None:
            return tool_error("Feishu client not available")
        if not all(values):
            return tool_error(missing_msg)
        code, msg, data = lark_call(
            client, method, uri, paths={k: v for k, v in zip(keys, values) if k != "content"},
            queries=queries and queries(args), body=body and body(args, values))
        if code != 0:
            return tool_error(f"{label} failed: code={code} msg={msg}")
        return tool_result(success=True, data=data) if flag_success else tool_result(data)
    return _handler


def _file_type(args: dict) -> str:
    return args.get("file_type", "docx") or "docx"


def _paged_queries(args: dict, *extra) -> list:
    """Query params shared by the listing endpoints; page_token goes last (after any extra)."""
    queries = [
        ("file_type", _file_type(args)), ("user_id_type", "open_id"),
        ("page_size", str(args.get("page_size", 100))), *extra]
    page_token = args.get("page_token", "")
    if page_token:
        queries.append(("page_token", page_token))
    return queries


_FILE_TOKEN_PROP = {"type": "string", "description": "The document file token."}
_FILE_TYPE_PROP = {"type": "string", "description": "File type (default: docx).", "default": "docx"}
_PAGE_TOKEN_PROP = {"type": "string", "description": "Pagination token for next page."}
_COMMENTS_URI = "/open-apis/drive/v1/files/:file_token/comments"
_REPLIES_URI = "/open-apis/drive/v1/files/:file_token/comments/:comment_id/replies"
_ADD_COMMENT_URI = "/open-apis/drive/v1/files/:file_token/new_comments"


FEISHU_DRIVE_LIST_COMMENTS_SCHEMA = {
    "name": "feishu_drive_list_comments",
    "description": (
        "List comments on a Feishu document. "
        "Use is_whole=true to list whole-document comments only."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "file_token": _FILE_TOKEN_PROP,
            "file_type": _FILE_TYPE_PROP,
            "is_whole": {
                "type": "boolean",
                "description": "If true, only return whole-document comments.",
                "default": False,
            },
            "page_size": {
                "type": "integer",
                "description": "Number of comments per page (max 100).",
                "default": 100,
            },
            "page_token": _PAGE_TOKEN_PROP,
        },
        "required": ["file_token"],
    },
}


FEISHU_DRIVE_LIST_REPLIES_SCHEMA = {
    "name": "feishu_drive_list_comment_replies",
    "description": "List all replies in a comment thread on a Feishu document.",
    "parameters": {
        "type": "object",
        "properties": {
            "file_token": _FILE_TOKEN_PROP,
            "comment_id": {
                "type": "string",
                "description": "The comment ID to list replies for.",
            },
            "file_type": _FILE_TYPE_PROP,
            "page_size": {
                "type": "integer",
                "description": "Number of replies per page (max 100).",
                "default": 100,
            },
            "page_token": _PAGE_TOKEN_PROP,
        },
        "required": ["file_token", "comment_id"],
    },
}


FEISHU_DRIVE_REPLY_SCHEMA = {
    "name": "feishu_drive_reply_comment",
    "description": (
        "Reply to a local comment thread on a Feishu document. "
        "Use this for local (quoted-text) comments. "
        "For whole-document comments, use feishu_drive_add_comment instead."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "file_token": _FILE_TOKEN_PROP,
            "comment_id": {
                "type": "string",
                "description": "The comment ID to reply to.",
            },
            "content": {
                "type": "string",
                "description": "The reply text content (plain text only, no markdown).",
            },
            "file_type": _FILE_TYPE_PROP,
        },
        "required": ["file_token", "comment_id", "content"],
    },
}


FEISHU_DRIVE_ADD_COMMENT_SCHEMA = {
    "name": "feishu_drive_add_comment",
    "description": (
        "Add a new whole-document comment on a Feishu document. "
        "Use this for whole-document comments or as a fallback when "
        "reply_comment fails with code 1069302."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "file_token": _FILE_TOKEN_PROP,
            "content": {
                "type": "string",
                "description": "The comment text content (plain text only, no markdown).",
            },
            "file_type": _FILE_TYPE_PROP,
        },
        "required": ["file_token", "content"],
    },
}


_handle_list_comments = _comment_op(
    ("file_token",), "file_token is required", "List comments", "GET", _COMMENTS_URI,
    queries=lambda a: _paged_queries(a, *([("is_whole", "true")] if a.get("is_whole", False) else [])))
_handle_list_replies = _comment_op(
    ("file_token", "comment_id"), "file_token and comment_id are required", "List replies", "GET",
    _REPLIES_URI, queries=_paged_queries)
# Replies use the rich "content.elements[text_run]" body shape; file_type is a query param.
_handle_reply_comment = _comment_op(
    ("file_token", "comment_id", "content"), "file_token, comment_id, and content are required",
    "Reply comment", "POST", _REPLIES_URI, queries=lambda a: [("file_type", _file_type(a))],
    body=lambda a, v: {"content": {"elements": [{"type": "text_run", "text_run": {"text": v[-1]}}]}},
    flag_success=True)
# new_comments takes the flat "reply_elements[text]" shape with file_type in the body.
_handle_add_comment = _comment_op(
    ("file_token", "content"), "file_token and content are required", "Add comment", "POST",
    _ADD_COMMENT_URI,
    body=lambda a, v: {"file_type": _file_type(a), "reply_elements": [{"type": "text", "text": v[-1]}]},
    flag_success=True)


for _schema, _handler, _desc, _emoji in (
    (FEISHU_DRIVE_LIST_COMMENTS_SCHEMA, _handle_list_comments, "List document comments", "\U0001f4ac"),
    (FEISHU_DRIVE_LIST_REPLIES_SCHEMA, _handle_list_replies, "List comment replies", "\U0001f4ac"),
    (FEISHU_DRIVE_REPLY_SCHEMA, _handle_reply_comment, "Reply to a document comment", "\u2709\ufe0f"),
    (FEISHU_DRIVE_ADD_COMMENT_SCHEMA, _handle_add_comment, "Add a whole-document comment", "\u2709\ufe0f"),
):
    registry.register(
        name=_schema["name"], toolset="feishu_drive", schema=_schema, handler=_handler,
        check_fn=_check_feishu, requires_env=[], is_async=False, description=_desc, emoji=_emoji)


# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# Names external plugins imported from this module before the Sep 2026 decomposition.
# Internal code MUST NOT use these (scripts/check_compat_pointers.py fails CI if it does).
# The whole block is removed by reverting the commit that added it.
import json  # noqa: F401,E402
import threading  # noqa: F401,E402
# ---- END PLUGIN-COMPAT ----
