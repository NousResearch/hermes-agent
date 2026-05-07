"""Feishu Drive tools for standalone Open Platform file and comment operations."""

from __future__ import annotations

from typing import Any

from tools.feishu_openapi import FeishuOpenAPIError, check_feishu_openapi_requirements, request_json
from tools.registry import registry, tool_error, tool_result

_META_URI = "/open-apis/drive/v1/metas/batch_query"
_SEARCH_URI = "/open-apis/drive/v1/files/search"
_CREATE_FOLDER_URI = "/open-apis/drive/v1/files/create_folder"
_COMMENTS_URI = "/open-apis/drive/v1/files/:file_token/comments"
_REPLIES_URI = "/open-apis/drive/v1/files/:file_token/comments/:comment_id/replies"


def _s(args: dict[str, Any], name: str) -> str:
    return str(args.get(name) or "").strip()


def _call(method: str, uri: str, *, paths=None, queries=None, body=None, **kwargs: Any) -> str:
    try:
        data = request_json(method, uri, paths=paths, queries=queries, body=body, client=kwargs.get("client"))
        return tool_result({"success": True, "data": data})
    except FeishuOpenAPIError as exc:
        return tool_error(str(exc), code=exc.code, data=exc.data)


def _get_meta(args: dict[str, Any], **kwargs: Any) -> str:
    file_token = _s(args, "file_token")
    file_type = _s(args, "file_type") or "docx"
    if not file_token:
        return tool_error("file_token is required; Drive tokens are not IM file_key values")
    body = {"request_docs": [{"doc_token": file_token, "doc_type": file_type}]}
    return _call("POST", _META_URI, body=body, **kwargs)


def _search_files(args: dict[str, Any], **kwargs: Any) -> str:
    query = _s(args, "query")
    if not query:
        return tool_error("query is required")
    body = {"search_key": query}
    for key in ("folder_token", "file_extension", "owner_id", "chat_id"):
        if args.get(key):
            body[key] = args[key]
    queries = {"page_size": args.get("page_size"), "page_token": args.get("page_token")}
    return _call("POST", _SEARCH_URI, queries=queries, body=body, **kwargs)


def _create_folder(args: dict[str, Any], **kwargs: Any) -> str:
    name = _s(args, "name")
    folder_token = _s(args, "folder_token")
    if not name:
        return tool_error("name is required")
    body = {"name": name}
    if folder_token:
        body["folder_token"] = folder_token
    return _call("POST", _CREATE_FOLDER_URI, body=body, **kwargs)


def _list_comments(args: dict[str, Any], **kwargs: Any) -> str:
    file_token = _s(args, "file_token")
    if not file_token:
        return tool_error("file_token is required; Drive tokens are not IM file_key values")
    queries = {
        "file_type": _s(args, "file_type") or "docx",
        "user_id_type": args.get("user_id_type") or "open_id",
        "is_whole": args.get("is_whole"),
        "page_size": args.get("page_size"),
        "page_token": args.get("page_token"),
    }
    return _call("GET", _COMMENTS_URI, paths={"file_token": file_token}, queries=queries, **kwargs)


def _reply_comment(args: dict[str, Any], **kwargs: Any) -> str:
    file_token = _s(args, "file_token")
    comment_id = _s(args, "comment_id")
    content = str(args.get("content") or "")
    if not file_token:
        return tool_error("file_token is required; Drive tokens are not IM file_key values")
    if not comment_id:
        return tool_error("comment_id is required")
    if not content.strip():
        return tool_error("content is required")
    queries = {"file_type": _s(args, "file_type") or "docx", "user_id_type": args.get("user_id_type") or "open_id"}
    body = {"reply": {"content": {"elements": [{"text_run": {"text": content}}]}}}
    return _call("POST", _REPLIES_URI, paths={"file_token": file_token, "comment_id": comment_id}, queries=queries, body=body, **kwargs)


def _schema(name: str, description: str, props: dict[str, Any], required: list[str]) -> dict[str, Any]:
    common = {
        "file_token": {"type": "string", "description": "Drive file token; not an IM message file_key."},
        "file_type": {"type": "string", "description": "Drive file type, default docx."},
        "page_size": {"type": "integer"},
        "page_token": {"type": "string"},
    }
    common.update(props)
    return {"name": name, "description": description, "parameters": {"type": "object", "properties": common, "required": required}}


_COMMON = dict(toolset="feishu", check_fn=check_feishu_openapi_requirements, requires_env=["FEISHU_APP_ID", "FEISHU_APP_SECRET"], emoji="🗂️")

for _name, _handler, _description, _props, _required in [
    ("feishu_drive_get_meta", _get_meta, "Get Drive metadata for a Feishu file token. Drive tokens are distinct from IM file_key values.", {}, ["file_token"]),
    ("feishu_drive_search_files", _search_files, "Search Feishu Drive files by keyword.", {"query": {"type": "string"}, "folder_token": {"type": "string"}, "file_extension": {"type": "string"}, "owner_id": {"type": "string"}, "chat_id": {"type": "string"}}, ["query"]),
    ("feishu_drive_create_folder", _create_folder, "Create a Feishu Drive folder.", {"name": {"type": "string"}, "folder_token": {"type": "string", "description": "Optional parent folder token."}}, ["name"]),
    ("feishu_drive_list_comments", _list_comments, "List comments on a Feishu Drive document.", {"is_whole": {"type": "boolean"}, "user_id_type": {"type": "string"}}, ["file_token"]),
    ("feishu_drive_reply_comment", _reply_comment, "Reply to a comment on a Feishu Drive document.", {"comment_id": {"type": "string"}, "content": {"type": "string"}, "user_id_type": {"type": "string"}}, ["file_token", "comment_id", "content"]),
]:
    registry.register(name=_name, handler=_handler, schema=_schema(_name, _description, _props, _required), description=_description, **_COMMON)