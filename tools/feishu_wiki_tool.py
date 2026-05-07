"""Feishu Wiki tools."""

from __future__ import annotations

from typing import Any

from tools.feishu_openapi import FeishuOpenAPIError, check_feishu_openapi_requirements, request_json
from tools.registry import registry, tool_error, tool_result

_SPACES_URI = "/open-apis/wiki/v2/spaces"
_NODES_URI = "/open-apis/wiki/v2/spaces/:space_id/nodes"
_NODE_URI = "/open-apis/wiki/v2/spaces/:space_id/nodes/:node_token"


def _s(args: dict[str, Any], name: str) -> str:
    return str(args.get(name) or "").strip()


def _call(method: str, uri: str, *, paths=None, queries=None, body=None, **kwargs: Any) -> str:
    try:
        data = request_json(method, uri, paths=paths, queries=queries, body=body, client=kwargs.get("client"))
        return tool_result({"success": True, "data": data})
    except FeishuOpenAPIError as exc:
        return tool_error(str(exc), code=exc.code, data=exc.data)


def _list_spaces(args: dict[str, Any], **kwargs: Any) -> str:
    queries = {"page_size": args.get("page_size"), "page_token": args.get("page_token")}
    return _call("GET", _SPACES_URI, queries=queries, **kwargs)


def _list_nodes(args: dict[str, Any], **kwargs: Any) -> str:
    space_id = _s(args, "space_id")
    if not space_id:
        return tool_error("space_id is required")
    queries = {
        "parent_node_token": args.get("parent_node_token"),
        "page_size": args.get("page_size"),
        "page_token": args.get("page_token"),
    }
    return _call("GET", _NODES_URI, paths={"space_id": space_id}, queries=queries, **kwargs)


def _get_node(args: dict[str, Any], **kwargs: Any) -> str:
    space_id = _s(args, "space_id")
    node_token = _s(args, "node_token")
    if not space_id:
        return tool_error("space_id is required")
    if not node_token:
        return tool_error("node_token is required")
    return _call("GET", _NODE_URI, paths={"space_id": space_id, "node_token": node_token}, **kwargs)


def _create_node(args: dict[str, Any], **kwargs: Any) -> str:
    space_id = _s(args, "space_id")
    parent_node_token = _s(args, "parent_node_token")
    obj_type = _s(args, "obj_type") or "docx"
    title = _s(args, "title")
    if not space_id:
        return tool_error("space_id is required")
    if not parent_node_token:
        return tool_error("parent_node_token is required")
    if not title:
        return tool_error("title is required")
    body = {"parent_node_token": parent_node_token, "node_type": "origin", "obj_type": obj_type, "title": title}
    if args.get("obj_token"):
        body["obj_token"] = args["obj_token"]
    return _call("POST", _NODES_URI, paths={"space_id": space_id}, body=body, **kwargs)


def _schema(name: str, description: str, props: dict[str, Any], required: list[str]) -> dict[str, Any]:
    common = {
        "space_id": {"type": "string", "description": "Wiki space ID."},
        "node_token": {"type": "string", "description": "Wiki node token."},
        "parent_node_token": {"type": "string", "description": "Parent Wiki node token."},
        "page_size": {"type": "integer"},
        "page_token": {"type": "string"},
    }
    common.update(props)
    return {"name": name, "description": description, "parameters": {"type": "object", "properties": common, "required": required}}


_COMMON = dict(toolset="feishu", check_fn=check_feishu_openapi_requirements, requires_env=["FEISHU_APP_ID", "FEISHU_APP_SECRET"], emoji="📚")

for _name, _handler, _description, _props, _required in [
    ("feishu_wiki_list_spaces", _list_spaces, "List Feishu Wiki spaces visible to the app.", {}, []),
    ("feishu_wiki_list_nodes", _list_nodes, "List Feishu Wiki nodes in a space, optionally under a parent node.", {}, ["space_id"]),
    ("feishu_wiki_get_node", _get_node, "Get metadata for a Feishu Wiki node.", {}, ["space_id", "node_token"]),
    ("feishu_wiki_create_node", _create_node, "Create a Feishu Wiki node, optionally linking an existing object token.", {"title": {"type": "string"}, "obj_type": {"type": "string", "description": "Object type, default docx."}, "obj_token": {"type": "string", "description": "Optional existing document/object token."}}, ["space_id", "parent_node_token", "title"]),
]:
    registry.register(name=_name, handler=_handler, schema=_schema(_name, _description, _props, _required), description=_description, **_COMMON)