"""Feishu Bitable/Base tools."""

from __future__ import annotations

from typing import Any

from tools.feishu_openapi import FeishuOpenAPIError, check_feishu_openapi_requirements, request_json
from tools.registry import registry, tool_error, tool_result

_APPS_URI = "/open-apis/bitable/v1/apps"
_TABLES_URI = "/open-apis/bitable/v1/apps/:app_token/tables"
_TABLE_URI = "/open-apis/bitable/v1/apps/:app_token/tables/:table_id"
_FIELDS_URI = "/open-apis/bitable/v1/apps/:app_token/tables/:table_id/fields"
_FIELD_URI = "/open-apis/bitable/v1/apps/:app_token/tables/:table_id/fields/:field_id"
_RECORDS_URI = "/open-apis/bitable/v1/apps/:app_token/tables/:table_id/records"
_RECORD_URI = "/open-apis/bitable/v1/apps/:app_token/tables/:table_id/records/:record_id"


def _s(args: dict[str, Any], name: str) -> str:
    return str(args.get(name) or "").strip()


def _require(args: dict[str, Any], *names: str) -> str | None:
    missing = [name for name in names if not _s(args, name)]
    return f"{', '.join(missing)} required" if missing else None


def _page_queries(args: dict[str, Any]) -> dict[str, Any]:
    return {"page_size": args.get("page_size"), "page_token": args.get("page_token")}


def _call(method: str, uri: str, *, args: dict[str, Any], paths: dict[str, Any], queries=None, body=None, **kwargs: Any) -> str:
    try:
        data = request_json(method, uri, paths=paths, queries=queries, body=body, client=kwargs.get("client"))
        return tool_result({"success": True, "data": data})
    except FeishuOpenAPIError as exc:
        return tool_error(str(exc), code=exc.code, data=exc.data)


def _create_app(args: dict[str, Any], **kwargs: Any) -> str:
    name = _s(args, "name")
    if not name:
        return tool_error("name is required")
    body = {"name": name}
    if args.get("folder_token"):
        body["folder_token"] = args["folder_token"]
    return _call("POST", _APPS_URI, args=args, paths={}, body=body, **kwargs)


def _list_tables(args: dict[str, Any], **kwargs: Any) -> str:
    if err := _require(args, "app_token"):
        return tool_error(err)
    return _call("GET", _TABLES_URI, args=args, paths={"app_token": _s(args, "app_token")}, queries=_page_queries(args), **kwargs)


def _update_table(args: dict[str, Any], **kwargs: Any) -> str:
    if err := _require(args, "app_token", "table_id", "name"):
        return tool_error(err)
    return _call("PATCH", _TABLE_URI, args=args, paths={"app_token": _s(args, "app_token"), "table_id": _s(args, "table_id")}, body={"name": _s(args, "name")}, **kwargs)


def _get_fields(args: dict[str, Any], **kwargs: Any) -> str:
    if err := _require(args, "app_token", "table_id"):
        return tool_error(err)
    return _call("GET", _FIELDS_URI, args=args, paths={"app_token": _s(args, "app_token"), "table_id": _s(args, "table_id")}, queries=_page_queries(args), **kwargs)


def _create_field(args: dict[str, Any], **kwargs: Any) -> str:
    if err := _require(args, "app_token", "table_id", "field_name", "type"):
        return tool_error(err)
    body: dict[str, Any] = {"field_name": _s(args, "field_name"), "type": args.get("type")}
    if args.get("property") is not None:
        body["property"] = args["property"]
    if args.get("description") is not None:
        body["description"] = args["description"]
    return _call("POST", _FIELDS_URI, args=args, paths={"app_token": _s(args, "app_token"), "table_id": _s(args, "table_id")}, body=body, **kwargs)


def _update_field(args: dict[str, Any], **kwargs: Any) -> str:
    if err := _require(args, "app_token", "table_id", "field_id", "field_name", "type"):
        return tool_error(err)
    body: dict[str, Any] = {"field_name": _s(args, "field_name"), "type": args.get("type")}
    if args.get("property") is not None:
        body["property"] = args["property"]
    if args.get("description") is not None:
        body["description"] = args["description"]
    return _call("PUT", _FIELD_URI, args=args, paths={"app_token": _s(args, "app_token"), "table_id": _s(args, "table_id"), "field_id": _s(args, "field_id")}, body=body, **kwargs)


def _search_records(args: dict[str, Any], **kwargs: Any) -> str:
    if err := _require(args, "app_token", "table_id"):
        return tool_error(err)
    body: dict[str, Any] = {}
    for key in ("view_id", "field_names", "sort", "filter", "automatic_fields"):
        if args.get(key) not in (None, ""):
            body[key] = args[key]
    return _call(
        "POST",
        _RECORDS_URI + "/search",
        args=args,
        paths={"app_token": _s(args, "app_token"), "table_id": _s(args, "table_id")},
        queries=_page_queries(args),
        body=body,
        **kwargs,
    )


def _create_record(args: dict[str, Any], **kwargs: Any) -> str:
    if err := _require(args, "app_token", "table_id"):
        return tool_error(err)
    fields = args.get("fields")
    if not isinstance(fields, dict) or not fields:
        return tool_error("fields must be a non-empty object")
    return _call("POST", _RECORDS_URI, args=args, paths={"app_token": _s(args, "app_token"), "table_id": _s(args, "table_id")}, body={"fields": fields}, **kwargs)


def _update_record(args: dict[str, Any], **kwargs: Any) -> str:
    if err := _require(args, "app_token", "table_id", "record_id"):
        return tool_error(err)
    fields = args.get("fields")
    if not isinstance(fields, dict) or not fields:
        return tool_error("fields must be a non-empty object")
    return _call("PUT", _RECORD_URI, args=args, paths={"app_token": _s(args, "app_token"), "table_id": _s(args, "table_id"), "record_id": _s(args, "record_id")}, body={"fields": fields}, **kwargs)


def _delete_record(args: dict[str, Any], **kwargs: Any) -> str:
    if err := _require(args, "app_token", "table_id", "record_id"):
        return tool_error(err)
    return _call("DELETE", _RECORD_URI, args=args, paths={"app_token": _s(args, "app_token"), "table_id": _s(args, "table_id"), "record_id": _s(args, "record_id")}, **kwargs)


def _schema(name: str, description: str, props: dict[str, Any], required: list[str]) -> dict[str, Any]:
    base = {
        "app_token": {"type": "string", "description": "Bitable app/base token."},
        "table_id": {"type": "string", "description": "Bitable table ID."},
        "record_id": {"type": "string", "description": "Bitable record ID."},
        "page_size": {"type": "integer", "description": "Page size for paginated list/search calls."},
        "page_token": {"type": "string", "description": "Pagination token."},
    }
    base.update(props)
    return {"name": name, "description": description, "parameters": {"type": "object", "properties": base, "required": required}}


_COMMON = dict(check_fn=check_feishu_openapi_requirements, requires_env=["FEISHU_APP_ID", "FEISHU_APP_SECRET"], emoji="🧮", toolset="feishu")

for _name, _handler, _description, _props, _required in [
    ("feishu_bitable_create_app", _create_app, "Create a Feishu Bitable/Base app.", {"name": {"type": "string"}, "folder_token": {"type": "string", "description": "Optional Drive folder token."}}, ["name"]),
    ("feishu_bitable_list_tables", _list_tables, "List tables in a Feishu Bitable/Base app.", {}, ["app_token"]),
    ("feishu_bitable_update_table", _update_table, "Rename/update a Feishu Bitable table.", {"name": {"type": "string"}}, ["app_token", "table_id", "name"]),
    ("feishu_bitable_get_fields", _get_fields, "List fields for a Feishu Bitable table.", {}, ["app_token", "table_id"]),
    ("feishu_bitable_create_field", _create_field, "Create a field in a Feishu Bitable table. Use field type codes from Feishu Bitable, e.g. 1 Text, 3 SingleSelect, 5 DateTime, 17 Attachment.", {"field_name": {"type": "string"}, "type": {"type": "integer"}, "property": {"type": "object"}, "description": {"type": "string"}}, ["app_token", "table_id", "field_name", "type"]),
    ("feishu_bitable_update_field", _update_field, "Update a Feishu Bitable field name/type/property.", {"field_id": {"type": "string"}, "field_name": {"type": "string"}, "type": {"type": "integer"}, "property": {"type": "object"}, "description": {"type": "string"}}, ["app_token", "table_id", "field_id", "field_name", "type"]),
    ("feishu_bitable_search_records", _search_records, "Search records in a Feishu Bitable table with optional filter/sort/view parameters.", {"view_id": {"type": "string"}, "field_names": {"type": "array", "items": {"type": "string"}}, "sort": {"type": "array"}, "filter": {"type": "object"}, "automatic_fields": {"type": "boolean"}}, ["app_token", "table_id"]),
    ("feishu_bitable_create_record", _create_record, "Create one record in a Feishu Bitable table.", {"fields": {"type": "object", "description": "Record field values keyed by field name."}}, ["app_token", "table_id", "fields"]),
    ("feishu_bitable_update_record", _update_record, "Update one record in a Feishu Bitable table.", {"fields": {"type": "object", "description": "Record field values keyed by field name."}}, ["app_token", "table_id", "record_id", "fields"]),
    ("feishu_bitable_delete_record", _delete_record, "Delete one record in a Feishu Bitable table. Requires an explicit record_id.", {}, ["app_token", "table_id", "record_id"]),
]:
    registry.register(name=_name, handler=_handler, schema=_schema(_name, _description, _props, _required), description=_description, **_COMMON)