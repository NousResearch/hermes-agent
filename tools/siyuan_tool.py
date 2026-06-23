"""SiYuan HTTP API tools for Hermes static knowledge operations."""

from __future__ import annotations

from typing import Any

from knowledge.adapters.siyuan import (
    SiYuanClient,
    SiYuanKnowledgeAdapter,
    append_block,
    export_markdown,
    load_config,
    run_sql,
    search_blocks,
    update_block,
)
from knowledge.policy import RoutePolicy
from knowledge.types import KnowledgeWriteRequest
from tools.registry import registry, tool_error, tool_result


def classify_write_policy(content_type: str) -> dict[str, Any]:
    decision = RoutePolicy().decide(content_type)
    return decision.to_dict()


def _client() -> SiYuanClient:
    return SiYuanClient(load_config())


def _handle_siyuan_write_policy(args: dict[str, Any], **_: Any) -> str:
    return tool_result(classify_write_policy(args.get("content_type", "")))


def _handle_siyuan_write_doc(args: dict[str, Any], **_: Any) -> str:
    try:
        request = KnowledgeWriteRequest(
            content_type="runbook",
            title=args.get("title", ""),
            content=args.get("markdown", ""),
            notebook=args.get("notebook", "") or "",
            path=args.get("path", "") or "",
            duplicate_policy=args.get("duplicate_policy", "create_new") or "create_new",
        )
        return tool_result(SiYuanKnowledgeAdapter().write(request).to_dict())
    except Exception as exc:
        return tool_error(str(exc), success=False)


def _handle_siyuan_append_doc(args: dict[str, Any], **_: Any) -> str:
    try:
        block_id = args.get("block_id")
        markdown = args.get("markdown", "")
        if not block_id:
            return tool_error("block_id is required", success=False)
        if not markdown.strip():
            return tool_error("markdown is required", success=False)
        return tool_result(append_block(_client(), block_id, markdown))
    except Exception as exc:
        return tool_error(str(exc), success=False)


def _handle_siyuan_search(args: dict[str, Any], **_: Any) -> str:
    try:
        query = (args.get("query") or "").strip()
        if not query:
            return tool_error("query is required", success=False)
        return tool_result(search_blocks(_client(), query, int(args.get("page") or 1)))
    except Exception as exc:
        return tool_error(str(exc), success=False)


def _handle_siyuan_export_doc(args: dict[str, Any], **_: Any) -> str:
    try:
        block_id = args.get("block_id") or args.get("id")
        if not block_id:
            return tool_error("block_id is required", success=False)
        return tool_result(export_markdown(_client(), block_id))
    except Exception as exc:
        return tool_error(str(exc), success=False)


def _handle_siyuan_update_block(args: dict[str, Any], **_: Any) -> str:
    try:
        block_id = args.get("block_id") or args.get("id")
        markdown = args.get("markdown", "")
        if not block_id:
            return tool_error("block_id is required", success=False)
        if not markdown.strip():
            return tool_error("markdown is required", success=False)
        return tool_result(update_block(_client(), block_id, markdown))
    except Exception as exc:
        return tool_error(str(exc), success=False)


def _handle_siyuan_sql(args: dict[str, Any], **_: Any) -> str:
    try:
        return tool_result(run_sql(_client(), args.get("stmt", ""), bool(args.get("allow_unsafe"))))
    except Exception as exc:
        return tool_error(str(exc), success=False)


SIYUAN_WRITE_POLICY_SCHEMA = {
    "name": "siyuan_write_policy",
    "description": "Classify whether content belongs in dynamic memory, static knowledge, or nowhere before writing knowledge.",
    "parameters": {"type": "object", "properties": {"content_type": {"type": "string"}}, "required": ["content_type"]},
}
SIYUAN_WRITE_DOC_SCHEMA = {
    "name": "siyuan_write_doc",
    "description": "Write curated static knowledge as a SiYuan document from Markdown. Do not use for raw chat logs or preferences.",
    "parameters": {"type": "object", "properties": {"title": {"type": "string"}, "markdown": {"type": "string"}, "notebook": {"type": "string"}, "path": {"type": "string"}}, "required": ["title", "markdown"]},
}
SIYUAN_APPEND_DOC_SCHEMA = {"name": "siyuan_append_doc", "description": "Append Markdown under an existing SiYuan block/document ID.", "parameters": {"type": "object", "properties": {"block_id": {"type": "string"}, "markdown": {"type": "string"}}, "required": ["block_id", "markdown"]}}
SIYUAN_SEARCH_SCHEMA = {"name": "siyuan_search", "description": "Search the SiYuan static knowledge base and return compact block matches.", "parameters": {"type": "object", "properties": {"query": {"type": "string"}, "page": {"type": "integer"}}, "required": ["query"]}}
SIYUAN_EXPORT_DOC_SCHEMA = {"name": "siyuan_export_doc", "description": "Export a SiYuan document/block as Markdown by block ID for audit or snapshots.", "parameters": {"type": "object", "properties": {"block_id": {"type": "string"}, "id": {"type": "string"}}, "required": []}}
SIYUAN_UPDATE_BLOCK_SCHEMA = {"name": "siyuan_update_block", "description": "Update an existing SiYuan block by ID using Markdown content.", "parameters": {"type": "object", "properties": {"block_id": {"type": "string"}, "id": {"type": "string"}, "markdown": {"type": "string"}}, "required": ["markdown"]}}
SIYUAN_SQL_SCHEMA = {"name": "siyuan_sql", "description": "Run a bounded SiYuan SQL query. Defaults to simple SELECT-only and auto-adds LIMIT 20.", "parameters": {"type": "object", "properties": {"stmt": {"type": "string"}, "allow_unsafe": {"type": "boolean"}}, "required": ["stmt"]}}


def _check_siyuan_available() -> bool:
    try:
        cfg = load_config()
        return bool(cfg.endpoint and cfg.api_token)
    except Exception:
        return False


registry.register(
    name="siyuan_write_policy",
    toolset="siyuan",
    schema=SIYUAN_WRITE_POLICY_SCHEMA,
    handler=_handle_siyuan_write_policy,
    check_fn=_check_siyuan_available,
    emoji="📚",
)
registry.register(
    name="siyuan_write_doc",
    toolset="siyuan",
    schema=SIYUAN_WRITE_DOC_SCHEMA,
    handler=_handle_siyuan_write_doc,
    check_fn=_check_siyuan_available,
    emoji="📚",
)
registry.register(
    name="siyuan_append_doc",
    toolset="siyuan",
    schema=SIYUAN_APPEND_DOC_SCHEMA,
    handler=_handle_siyuan_append_doc,
    check_fn=_check_siyuan_available,
    emoji="📚",
)
registry.register(
    name="siyuan_search",
    toolset="siyuan",
    schema=SIYUAN_SEARCH_SCHEMA,
    handler=_handle_siyuan_search,
    check_fn=_check_siyuan_available,
    emoji="📚",
)
registry.register(
    name="siyuan_export_doc",
    toolset="siyuan",
    schema=SIYUAN_EXPORT_DOC_SCHEMA,
    handler=_handle_siyuan_export_doc,
    check_fn=_check_siyuan_available,
    emoji="📚",
)
registry.register(
    name="siyuan_update_block",
    toolset="siyuan",
    schema=SIYUAN_UPDATE_BLOCK_SCHEMA,
    handler=_handle_siyuan_update_block,
    check_fn=_check_siyuan_available,
    emoji="📚",
)
registry.register(
    name="siyuan_sql",
    toolset="siyuan",
    schema=SIYUAN_SQL_SCHEMA,
    handler=_handle_siyuan_sql,
    check_fn=_check_siyuan_available,
    emoji="📚",
)
