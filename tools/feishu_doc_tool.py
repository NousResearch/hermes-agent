"""Feishu Docx tools for standalone Open Platform document operations."""

from __future__ import annotations

from typing import Any

from tools.feishu_openapi import FeishuOpenAPIError, check_feishu_openapi_requirements, request_json
from tools.registry import registry, tool_error, tool_result

_DOC_RAW_CONTENT_URI = "/open-apis/docx/v1/documents/:document_id/raw_content"
_DOC_CREATE_URI = "/open-apis/docx/v1/documents"
_DOC_BLOCK_CHILDREN_URI = "/open-apis/docx/v1/documents/:document_id/blocks/:block_id/children"
_DOC_BLOCK_DELETE_CHILDREN_URI = "/open-apis/docx/v1/documents/:document_id/blocks/:block_id/children/batch_delete"
_DOC_BLOCK_LIST_URI = "/open-apis/docx/v1/documents/:document_id/blocks"
_DOC_BLOCK_CONVERT_URI = "/open-apis/docx/v1/documents/blocks/convert"


def _string_arg(args: dict[str, Any], name: str) -> str:
    return str(args.get(name) or "").strip()


def _handle_doc_read(args: dict[str, Any], **kwargs: Any) -> str:
    doc_token = _string_arg(args, "doc_token")
    if not doc_token:
        return tool_error("doc_token is required")
    try:
        data = request_json("GET", _DOC_RAW_CONTENT_URI, paths={"document_id": doc_token}, client=kwargs.get("client"))
        return tool_result({"success": True, "doc_token": doc_token, "content": data.get("content", ""), "data": data})
    except FeishuOpenAPIError as exc:
        return tool_error(str(exc), code=exc.code, data=exc.data)


def _handle_doc_create(args: dict[str, Any], **kwargs: Any) -> str:
    title = _string_arg(args, "title")
    if not title:
        return tool_error("title is required")
    folder_token = _string_arg(args, "folder_token")
    body: dict[str, Any] = {"title": title}
    if folder_token:
        body["folder_token"] = folder_token
    try:
        data = request_json("POST", _DOC_CREATE_URI, body=body, client=kwargs.get("client"))
        return tool_result({"success": True, "document": data.get("document", data), "data": data})
    except FeishuOpenAPIError as exc:
        return tool_error(str(exc), code=exc.code, data=exc.data)


def _paragraph_block(text: str) -> dict[str, Any]:
    return {
        "block_type": 2,
        "text": {
            "elements": [
                {
                    "text_run": {
                        "content": text,
                        "text_element_style": {},
                    }
                }
            ],
            "style": {},
        },
    }


def _handle_doc_append_text(args: dict[str, Any], **kwargs: Any) -> str:
    doc_token = _string_arg(args, "doc_token")
    text = str(args.get("text") or "")
    if not doc_token:
        return tool_error("doc_token is required")
    if not text.strip():
        return tool_error("text is required")
    parent_block_id = _string_arg(args, "parent_block_id") or doc_token
    lines = [line for line in text.splitlines() if line.strip()]
    if not lines:
        lines = [text]
    body = {"children": [_paragraph_block(line) for line in lines]}
    try:
        data = request_json(
            "POST",
            _DOC_BLOCK_CHILDREN_URI,
            paths={"document_id": doc_token, "block_id": parent_block_id},
            body=body,
            client=kwargs.get("client"),
        )
        return tool_result({"success": True, "doc_token": doc_token, "appended_blocks": data.get("children", data), "data": data})
    except FeishuOpenAPIError as exc:
        return tool_error(str(exc), code=exc.code, data=exc.data)


def _convert_markdown(markdown: str, *, client: Any | None = None) -> list[dict[str, Any]]:
    data = request_json(
        "POST",
        _DOC_BLOCK_CONVERT_URI,
        body={"content_type": "markdown", "content": markdown},
        client=client,
    )
    blocks = data.get("blocks", [])
    if not isinstance(blocks, list):
        return []
    # Feishu's convert endpoint returns first_level_block_ids as the canonical
    # document order. The blocks array can be non-deterministic, so reorder it
    # before insertion or rendered documents become scrambled.
    ordered_ids = data.get("first_level_block_ids", [])
    if isinstance(ordered_ids, list) and ordered_ids:
        by_id = {block.get("block_id"): block for block in blocks if isinstance(block, dict)}
        ordered = [by_id[block_id] for block_id in ordered_ids if block_id in by_id]
        remaining = [block for block in blocks if isinstance(block, dict) and block.get("block_id") not in set(ordered_ids)]
        return ordered + remaining
    return blocks


def _clear_document(doc_token: str, *, client: Any | None = None) -> int:
    data = request_json("GET", _DOC_BLOCK_LIST_URI, paths={"document_id": doc_token}, client=client)
    items = data.get("items", [])
    child_count = len([item for item in items if item.get("parent_id") == doc_token and item.get("block_type") != 1])
    if child_count:
        request_json(
            "DELETE",
            _DOC_BLOCK_DELETE_CHILDREN_URI,
            paths={"document_id": doc_token, "block_id": doc_token},
            body={"start_index": 0, "end_index": child_count},
            client=client,
        )
    return child_count


def _insert_blocks(doc_token: str, blocks: list[dict[str, Any]], *, parent_block_id: str | None = None, client: Any | None = None) -> dict[str, Any]:
    if not blocks:
        return {"children": []}
    block_id = parent_block_id or doc_token
    all_children: list[Any] = []
    last_data: dict[str, Any] = {}
    # Feishu limits one documentBlockChildren.create request to 50 children.
    for start in range(0, len(blocks), 50):
        data = request_json(
            "POST",
            _DOC_BLOCK_CHILDREN_URI,
            paths={"document_id": doc_token, "block_id": block_id},
            body={"children": blocks[start : start + 50]},
            client=client,
        )
        last_data = data
        children = data.get("children", [])
        if isinstance(children, list):
            all_children.extend(children)
    return {**last_data, "children": all_children}


def _handle_doc_append_markdown(args: dict[str, Any], **kwargs: Any) -> str:
    doc_token = _string_arg(args, "doc_token")
    markdown = str(args.get("markdown") or "")
    if not doc_token:
        return tool_error("doc_token is required")
    if not markdown.strip():
        return tool_error("markdown is required")
    client = kwargs.get("client")
    try:
        blocks = _convert_markdown(markdown, client=client)
        data = _insert_blocks(doc_token, blocks, parent_block_id=_string_arg(args, "parent_block_id") or None, client=client)
        return tool_result({"success": True, "doc_token": doc_token, "block_count": len(blocks), "data": data})
    except FeishuOpenAPIError as exc:
        return tool_error(str(exc), code=exc.code, data=exc.data)


def _handle_doc_replace_markdown(args: dict[str, Any], **kwargs: Any) -> str:
    doc_token = _string_arg(args, "doc_token")
    markdown = str(args.get("markdown") or "")
    if not doc_token:
        return tool_error("doc_token is required")
    if not markdown.strip():
        return tool_error("markdown is required")
    client = kwargs.get("client")
    try:
        blocks = _convert_markdown(markdown, client=client)
        deleted = _clear_document(doc_token, client=client)
        data = _insert_blocks(doc_token, blocks, client=client)
        return tool_result({"success": True, "doc_token": doc_token, "deleted_blocks": deleted, "block_count": len(blocks), "data": data})
    except FeishuOpenAPIError as exc:
        return tool_error(str(exc), code=exc.code, data=exc.data)


registry.register(
    name="feishu_doc_read",
    toolset="feishu",
    schema={
        "name": "feishu_doc_read",
        "description": "Read a Feishu/Lark Docx document as plain text raw content.",
        "parameters": {
            "type": "object",
            "properties": {"doc_token": {"type": "string", "description": "Docx document token from the URL."}},
            "required": ["doc_token"],
        },
    },
    handler=_handle_doc_read,
    check_fn=check_feishu_openapi_requirements,
    requires_env=["FEISHU_APP_ID", "FEISHU_APP_SECRET"],
    description="Read Feishu Docx document content",
    emoji="📄",
)

registry.register(
    name="feishu_doc_create",
    toolset="feishu",
    schema={
        "name": "feishu_doc_create",
        "description": "Create a Feishu/Lark Docx document. Requires document create permissions and folder access if folder_token is provided.",
        "parameters": {
            "type": "object",
            "properties": {
                "title": {"type": "string", "description": "Document title."},
                "folder_token": {"type": "string", "description": "Optional Drive folder token; this is not an IM file_key."},
            },
            "required": ["title"],
        },
    },
    handler=_handle_doc_create,
    check_fn=check_feishu_openapi_requirements,
    requires_env=["FEISHU_APP_ID", "FEISHU_APP_SECRET"],
    description="Create a Feishu Docx document",
    emoji="📄",
)

registry.register(
    name="feishu_doc_append_text",
    toolset="feishu",
    schema={
        "name": "feishu_doc_append_text",
        "description": "Append plain paragraph text to a Feishu/Lark Docx document. For formatted headings/lists, prefer feishu_doc_append_markdown or feishu_doc_replace_markdown.",
        "parameters": {
            "type": "object",
            "properties": {
                "doc_token": {"type": "string", "description": "Docx document token from the URL."},
                "text": {"type": "string", "description": "Plain text to append. Non-empty lines become paragraphs."},
                "parent_block_id": {"type": "string", "description": "Optional parent block ID. Defaults to doc_token/root."},
            },
            "required": ["doc_token", "text"],
        },
    },
    handler=_handle_doc_append_text,
    check_fn=check_feishu_openapi_requirements,
    requires_env=["FEISHU_APP_ID", "FEISHU_APP_SECRET"],
    description="Append plain text to Feishu Docx",
    emoji="📝",
)

registry.register(
    name="feishu_doc_append_markdown",
    toolset="feishu",
    schema={
        "name": "feishu_doc_append_markdown",
        "description": "Append Markdown as native Feishu Docx rich blocks so headings/lists render correctly.",
        "parameters": {
            "type": "object",
            "properties": {
                "doc_token": {"type": "string", "description": "Docx document token from the URL."},
                "markdown": {"type": "string", "description": "Markdown content to convert into Feishu Docx blocks."},
                "parent_block_id": {"type": "string", "description": "Optional parent block ID. Defaults to doc_token/root."},
            },
            "required": ["doc_token", "markdown"],
        },
    },
    handler=_handle_doc_append_markdown,
    check_fn=check_feishu_openapi_requirements,
    requires_env=["FEISHU_APP_ID", "FEISHU_APP_SECRET"],
    description="Append Markdown to Feishu Docx as native blocks",
    emoji="📝",
)

registry.register(
    name="feishu_doc_replace_markdown",
    toolset="feishu",
    schema={
        "name": "feishu_doc_replace_markdown",
        "description": "Replace a Feishu/Lark Docx document body with Markdown converted to native rich blocks. Use for formatted documents where headings/lists should render natively.",
        "parameters": {
            "type": "object",
            "properties": {
                "doc_token": {"type": "string", "description": "Docx document token from the URL."},
                "markdown": {"type": "string", "description": "Markdown content to convert into Feishu Docx blocks."},
            },
            "required": ["doc_token", "markdown"],
        },
    },
    handler=_handle_doc_replace_markdown,
    check_fn=check_feishu_openapi_requirements,
    requires_env=["FEISHU_APP_ID", "FEISHU_APP_SECRET"],
    description="Replace Feishu Docx body with formatted Markdown",
    emoji="📝",
)