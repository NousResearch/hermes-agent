"""Feishu Document Tool -- read document content via Feishu/Lark API.

Provides ``feishu_doc_read`` for reading document content as plain text.
Uses the same lazy-import + BaseRequest pattern as feishu_comment.py.
"""

import json
import logging
import threading

from tools.registry import registry, tool_error, tool_result

logger = logging.getLogger(__name__)

# Thread-local storage for the lark client injected by feishu_comment handler.
_local = threading.local()


def set_client(client):
    """Store a lark client for the current thread (called by feishu_comment)."""
    _local.client = client


def get_client():
    """Return the lark client for the current thread, or None."""
    return getattr(_local, "client", None)


# ---------------------------------------------------------------------------
# feishu_doc_read
# ---------------------------------------------------------------------------

_RAW_CONTENT_URI = "/open-apis/docx/v1/documents/:document_id/raw_content"

FEISHU_DOC_READ_SCHEMA = {
    "name": "feishu_doc_read",
    "description": (
        "Read the full content of a Feishu/Lark document as plain text. "
        "Useful when you need more context beyond the quoted text in a comment."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "doc_token": {
                "type": "string",
                "description": "The document token (from the document URL or comment context).",
            },
        },
        "required": ["doc_token"],
    },
}


def _check_feishu():
    # Use ``importlib.util.find_spec`` — it checks whether ``lark_oapi``
    # is importable without actually executing its ``__init__``.
    # Executing the real import here costs ~5 seconds (the SDK eagerly
    # loads websockets, dispatcher, every api/v2 model) and this probe
    # fires at every ``hermes`` startup during tool-availability
    # evaluation.  Correctness is preserved because the actual tool
    # handler still does the real import when invoked.
    import importlib.util
    try:
        return importlib.util.find_spec("lark_oapi") is not None
    except (ImportError, ValueError):
        return False


def _handle_feishu_doc_read(args: dict, **kwargs) -> str:
    doc_token = args.get("doc_token", "").strip()
    if not doc_token:
        return tool_error("doc_token is required")

    client = get_client()
    if client is None:
        return tool_error("Feishu client not available (not in a Feishu comment context)")

    try:
        from lark_oapi import AccessTokenType
        from lark_oapi.core.enum import HttpMethod
        from lark_oapi.core.model.base_request import BaseRequest
    except ImportError:
        return tool_error("lark_oapi not installed")

    request = (
        BaseRequest.builder()
        .http_method(HttpMethod.GET)
        .uri(_RAW_CONTENT_URI)
        .token_types({AccessTokenType.TENANT})
        .paths({"document_id": doc_token})
        .build()
    )

    # Tool handlers run synchronously in a worker thread (no running event
    # loop), so call the blocking lark client directly.
    response = client.request(request)

    code = getattr(response, "code", None)
    if code != 0:
        msg = getattr(response, "msg", "unknown error")
        return tool_error(f"Failed to read document: code={code} msg={msg}")

    raw = getattr(response, "raw", None)
    if raw and hasattr(raw, "content"):
        try:
            body = json.loads(raw.content)
            content = body.get("data", {}).get("content", "")
            return tool_result(success=True, content=content)
        except (json.JSONDecodeError, AttributeError):
            pass

    # Fallback: try response.data
    data = getattr(response, "data", None)
    if data:
        if isinstance(data, dict):
            content = data.get("content", "")
        else:
            content = getattr(data, "content", str(data))
        return tool_result(success=True, content=content)

    return tool_error("No content returned from document API")


# ---------------------------------------------------------------------------
# feishu_doc_write
# ---------------------------------------------------------------------------

_CHILDREN_URI = "/open-apis/docx/v1/documents/:document_id/blocks/:block_id/children"
_BATCH_DELETE_URI = "/open-apis/docx/v1/documents/:document_id/blocks/:block_id/children/batch_delete"

FEISHU_DOC_WRITE_SCHEMA = {
    "name": "feishu_doc_write",
    "description": (
        "Write markdown content to a Feishu/Lark document. "
        "Supports replace mode (clears document first) and append mode. "
        "Supports headings (h1-h3), paragraphs, bold, italic, inline code, "
        "code blocks, unordered lists, dividers, and simple tables."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "doc_token": {
                "type": "string",
                "description": "The document token (from the document URL).",
            },
            "content": {
                "type": "string",
                "description": "Markdown content to write to the document.",
            },
            "mode": {
                "type": "string",
                "description": "'replace' clears existing content first, 'append' adds to end (default: 'replace').",
                "enum": ["replace", "append"],
                "default": "replace",
            },
        },
        "required": ["doc_token", "content"],
    },
}

# Block type constants
_BLOCK_TEXT = 2
_BLOCK_H1 = 3
_BLOCK_H2 = 4
_BLOCK_H3 = 5
_BLOCK_BULLET = 11
_BLOCK_CODE = 14
_BLOCK_DIVIDER = 22
_BLOCK_TABLE = 27


def _parse_inline(text: str) -> list:
    """Parse inline markdown formatting into Feishu text elements.

    Handles: **bold**, *italic*, `code`, and plain text.
    Returns a list of text_run element dicts.
    """
    import re

    elements = []
    # Pattern order matters: code first (backticks), then bold, then italic
    pattern = re.compile(r"(`[^`]+`|\*\*[^*]+\*\*|\*[^*]+\*|[^`*]+)")
    parts = pattern.findall(text)

    for part in parts:
        if part.startswith("`") and part.endswith("`"):
            elements.append({
                "text_run": {
                    "content": part[1:-1],
                    "text_element_style": {"inline_code": True},
                }
            })
        elif part.startswith("**") and part.endswith("**"):
            elements.append({
                "text_run": {
                    "content": part[2:-2],
                    "text_element_style": {"bold": True},
                }
            })
        elif part.startswith("*") and part.endswith("*"):
            elements.append({
                "text_run": {
                    "content": part[1:-1],
                    "text_element_style": {"italic": True},
                }
            })
        elif part:
            elements.append({
                "text_run": {"content": part, "text_element_style": {}}
            })

    return elements


def _make_text_block(content: str) -> dict:
    """Create a text paragraph block from markdown content."""
    elements = _parse_inline(content)
    if not elements:
        elements = [{"text_run": {"content": "", "text_element_style": {}}}]
    return {
        "block_type": _BLOCK_TEXT,
        "text": {"elements": elements, "style": {}},
    }


def _make_heading_block(content: str, level: int) -> dict:
    """Create a heading block. level: 1-3."""
    block_types = {1: _BLOCK_H1, 2: _BLOCK_H2, 3: _BLOCK_H3}
    heading_keys = {1: "heading1", 2: "heading2", 3: "heading3"}
    elements = _parse_inline(content)
    if not elements:
        elements = [{"text_run": {"content": "", "text_element_style": {}}}]
    return {
        "block_type": block_types[level],
        heading_keys[level]: {"elements": elements, "style": {}},
    }


def _make_bullet_block(content: str) -> dict:
    """Create a bullet list item block."""
    elements = _parse_inline(content)
    if not elements:
        elements = [{"text_run": {"content": "", "text_element_style": {}}}]
    return {
        "block_type": _BLOCK_BULLET,
        "bullet": {"elements": elements, "style": {}},
    }


def _make_code_block(code: str) -> dict:
    """Create a code block."""
    return {
        "block_type": _BLOCK_CODE,
        "code": {
            "elements": [
                {"text_run": {"content": code, "text_element_style": {}}}
            ],
            "style": {"language": 1},  # 1 = plain text
        },
    }


def _make_divider_block() -> dict:
    """Create a divider block."""
    return {"block_type": _BLOCK_DIVIDER, "divider": {}}


def _make_table_block(header_cells: list, row_cells_list: list) -> dict:
    """Create a simple table block.

    Args:
        header_cells: list of header cell strings
        row_cells_list: list of lists, each inner list is a row of cells
    """
    rows = []
    if header_cells:
        header_row = [
            [
                {"text_run": {"content": cell.strip(), "text_element_style": {"bold": True}}}
                for cell in header_cells
            ]
        ]
        rows.append(header_row[0])

    for row_cells in row_cells_list:
        rows.append([
            {"text_run": {"content": cell.strip(), "text_element_style": {}}}
            for cell in row_cells
        ])

    # Calculate column widths proportionally
    num_cols = len(rows[0]) if rows else 1
    col_width = max(100, 600 // num_cols)

    return {
        "block_type": _BLOCK_TABLE,
        "table": {
            "cells": rows,
            "property": {
                "column_size": [col_width] * num_cols,
                "column_width": [col_width] * num_cols,
            },
        },
    }


def _md_to_feishu_blocks(md_content: str) -> list:
    """Convert markdown content to a list of Feishu document blocks.

    Supports: headings (###), paragraphs, bold/italic/inline-code,
    code fences, dividers, unordered lists, and simple tables.
    """
    import re

    blocks = []
    lines = md_content.strip().split("\n")
    i = 0

    while i < len(lines):
        line = lines[i]

        # Skip empty lines
        if not line.strip():
            i += 1
            continue

        stripped = line.strip()

        # Code fence block (``` ... ```)
        if stripped.startswith("```"):
            code_lines = []
            i += 1
            while i < len(lines):
                if lines[i].strip().startswith("```"):
                    i += 1
                    break
                code_lines.append(lines[i])
                i += 1
            blocks.append(_make_code_block("\n".join(code_lines)))
            continue

        # Divider (---, ***, ___)
        if re.match(r"^[-*_]{3,}\s*$", stripped):
            blocks.append(_make_divider_block())
            i += 1
            continue

        # Heading (# ## ###)
        heading_match = re.match(r"^(#{1,3})\s+(.+)$", stripped)
        if heading_match:
            level = len(heading_match.group(1))
            text = heading_match.group(2)
            blocks.append(_make_heading_block(text, level))
            i += 1
            continue

        # Table detection: current line has | and next line is separator
        if (
            "|" in stripped
            and i + 2 < len(lines)
            and re.match(r"^[\s\|\-:]+$", lines[i + 1].strip())
        ):
            # Parse header
            header = [c.strip() for c in stripped.strip("|").split("|")]
            # Skip separator line
            i += 2
            # Parse data rows
            rows = []
            while i < len(lines) and "|" in lines[i]:
                rows.append(
                    [c.strip() for c in lines[i].strip().strip("|").split("|")]
                )
                i += 1
            blocks.append(_make_table_block(header, rows))
            continue

        # Unordered list (- or * followed by space)
        list_match = re.match(r"^[\s]*[-*]\s+(.+)$", stripped)
        if list_match:
            while i < len(lines):
                lm = re.match(r"^[\s]*[-*]\s+(.+)$", lines[i].strip())
                if not lm:
                    break
                blocks.append(_make_bullet_block(lm.group(1)))
                i += 1
            continue

        # Regular paragraph
        blocks.append(_make_text_block(stripped))
        i += 1

    return blocks


def _do_doc_request(client, method: str, uri: str, paths=None, queries=None, body=None):
    """Build and execute a Feishu doc API request. Returns (code, msg, data_dict)."""
    from lark_oapi import AccessTokenType
    from lark_oapi.core.enum import HttpMethod
    from lark_oapi.core.model.base_request import BaseRequest

    http_method = HttpMethod.GET if method == "GET" else (
        HttpMethod.POST if method == "POST" else HttpMethod.DELETE
    )

    builder = (
        BaseRequest.builder()
        .http_method(http_method)
        .uri(uri)
        .token_types({AccessTokenType.TENANT})
    )
    if paths:
        builder = builder.paths(paths)
    if queries:
        builder = builder.queries(queries)
    if body is not None:
        builder = builder.body(body)

    request = builder.build()
    response = client.request(request)

    code = getattr(response, "code", None)
    msg = getattr(response, "msg", "")

    # Parse response data
    data = {}
    raw = getattr(response, "raw", None)
    if raw and hasattr(raw, "content"):
        try:
            body_json = json.loads(raw.content)
            data = body_json.get("data", {})
        except (json.JSONDecodeError, AttributeError):
            pass
    if not data:
        resp_data = getattr(response, "data", None)
        if isinstance(resp_data, dict):
            data = resp_data
        elif resp_data and hasattr(resp_data, "__dict__"):
            data = vars(resp_data)

    return code, msg, data


def _get_child_block_ids(client, doc_token: str) -> list:
    """Get IDs of all direct child blocks of the document root."""
    all_ids = []
    page_token = None

    while True:
        queries = [("page_size", "50")]
        if page_token:
            queries.append(("page_token", page_token))

        code, msg, data = _do_doc_request(
            client, "GET", _CHILDREN_URI,
            paths={"document_id": doc_token, "block_id": doc_token},
            queries=queries,
        )
        if code != 0:
            logger.warning("Failed to list blocks for replace: code=%s msg=%s", code, msg)
            return []

        items = data.get("items", [])
        for item in items:
            bid = item.get("block_id", "")
            if bid:
                all_ids.append(bid)

        if not data.get("has_more"):
            break
        page_token = data.get("page_token", "")

    return all_ids


def _delete_blocks(client, doc_token: str, block_ids: list):
    """Batch delete blocks from the document root."""
    if not block_ids:
        return

    # Feishu batch delete: body is a list of block IDs
    code, msg, _data = _do_doc_request(
        client, "DELETE", _BATCH_DELETE_URI,
        paths={"document_id": doc_token, "block_id": doc_token},
        body=block_ids,
    )
    if code != 0:
        logger.warning("Batch delete blocks failed: code=%s msg=%s", code, msg)


def _create_blocks(client, doc_token: str, blocks: list):
    """Create blocks as children of the document root.

    Creates blocks in batches (max 50 per request as per Feishu API limits).
    """
    BATCH_SIZE = 50

    for start in range(0, len(blocks), BATCH_SIZE):
        batch = blocks[start : start + BATCH_SIZE]
        body = {"children": batch, "index": -1}  # -1 = append to end

        code, msg, _data = _do_doc_request(
            client, "POST", _CHILDREN_URI,
            paths={"document_id": doc_token, "block_id": doc_token},
            body=body,
        )
        if code != 0:
            return tool_error(
                f"Failed to create blocks (batch {start // BATCH_SIZE + 1}): "
                f"code={code} msg={msg}"
            )

    return None  # success


def _handle_feishu_doc_write(args: dict, **kwargs) -> str:
    doc_token = args.get("doc_token", "").strip()
    content = args.get("content", "")
    mode = args.get("mode", "replace")

    if not doc_token:
        return tool_error("doc_token is required")
    if not content:
        return tool_error("content is required")

    client = get_client()
    if client is None:
        return tool_error("Feishu client not available (not in a Feishu comment context)")

    try:
        from lark_oapi import AccessTokenType
        from lark_oapi.core.enum import HttpMethod
        from lark_oapi.core.model.base_request import BaseRequest
    except ImportError:
        return tool_error("lark_oapi not installed")

    # Convert markdown to Feishu blocks
    try:
        blocks = _md_to_feishu_blocks(content)
    except Exception as e:
        return tool_error(f"Failed to parse markdown: {e}")

    if not blocks:
        return tool_error("No content blocks generated from markdown")

    # Replace mode: delete existing content first
    if mode == "replace":
        existing_ids = _get_child_block_ids(client, doc_token)
        if existing_ids:
            _delete_blocks(client, doc_token, existing_ids)

    # Create new blocks
    error = _create_blocks(client, doc_token, blocks)
    if error:
        return error

    return tool_result(
        success=True,
        blocks_created=len(blocks),
        mode=mode,
        message=f"{'Replaced' if mode == 'replace' else 'Appended'} document with {len(blocks)} blocks",
    )


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

registry.register(
    name="feishu_doc_read",
    toolset="feishu_doc",
    schema=FEISHU_DOC_READ_SCHEMA,
    handler=_handle_feishu_doc_read,
    check_fn=_check_feishu,
    requires_env=[],
    is_async=False,
    description="Read Feishu document content",
    emoji="\U0001f4c4",
)

registry.register(
    name="feishu_doc_write",
    toolset="feishu_doc",
    schema=FEISHU_DOC_WRITE_SCHEMA,
    handler=_handle_feishu_doc_write,
    check_fn=_check_feishu,
    requires_env=[],
    is_async=False,
    description="Write markdown content to a Feishu document",
    emoji="\u270f\ufe0f",
)
