"""Feishu Bitable Tools — CRUD operations on Feishu/Lark Bitable (多维表格).

Provides tools for listing tables, fields, records, searching records,
and batch creating/updating records.

Unlike feishu_doc_tool and feishu_drive_tool (which depend on thread-local
client injection from feishu_comment.py), these tools build their own lark
client from FEISHU_APP_ID / FEISHU_APP_SECRET env vars — so they work from
regular DMs, not just during comment event handling.
"""

import json
import logging
import os

from tools.registry import registry, tool_error, tool_result

logger = logging.getLogger(__name__)

# Module-level lazy client
_client = None


def _get_client():
    """Build and cache a lark Client from env vars."""
    global _client
    if _client is None:
        from lark_oapi import Client
        app_id = os.getenv("FEISHU_APP_ID", "")
        app_secret = os.getenv("FEISHU_APP_SECRET", "")
        if app_id.startswith("cli_"):
            from lark_oapi import FEISHU_DOMAIN
            domain = FEISHU_DOMAIN
        else:
            from lark_oapi import LARK_DOMAIN
            domain = LARK_DOMAIN
        _client = Client.builder().app_id(app_id).app_secret(app_secret).domain(domain).build()
    return _client


def _check_feishu():
    """Return True when lark_oapi is importable AND env vars are set."""
    import importlib.util
    try:
        return (
            importlib.util.find_spec("lark_oapi") is not None
            and bool(os.getenv("FEISHU_APP_ID"))
            and bool(os.getenv("FEISHU_APP_SECRET"))
        )
    except (ImportError, ValueError):
        return False


def _do_request(method, uri, paths=None, queries=None, body=None):
    """Build and execute a BaseRequest, return (code, msg, data_dict)."""
    from lark_oapi import AccessTokenType
    from lark_oapi.core.enum import HttpMethod
    from lark_oapi.core.model.base_request import BaseRequest

    client = _get_client()
    http_method = HttpMethod.GET if method == "GET" else HttpMethod.POST

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


# ---------------------------------------------------------------------------
# feishu_bitable_list_tables
# ---------------------------------------------------------------------------

_LIST_TABLES_URI = "/open-apis/bitable/v1/apps/:app_token/tables"

BITABLE_LIST_TABLES_SCHEMA = {
    "name": "feishu_bitable_list_tables",
    "description": (
        "List all data tables in a Feishu/Lark Bitable (多维表格). "
        "Returns table_id, name, and revision for each table."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "app_token": {
                "type": "string",
                "description": (
                    "The Bitable app token (from the Bitable URL: "
                    "https://xxx.feishu.cn/base/APP_TOKEN?table=..."
                ),
            },
            "page_size": {
                "type": "integer",
                "description": "Number of tables per page (max 100).",
                "default": 100,
            },
            "page_token": {
                "type": "string",
                "description": "Pagination token for next page.",
            },
        },
        "required": ["app_token"],
    },
}


def _handle_list_tables(args: dict, **kwargs) -> str:
    app_token = args.get("app_token", "").strip()
    if not app_token:
        return tool_error("app_token is required")

    page_size = args.get("page_size", 100)
    page_token = args.get("page_token", "")

    queries = [("page_size", str(page_size))]
    if page_token:
        queries.append(("page_token", page_token))

    code, msg, data = _do_request(
        "GET", _LIST_TABLES_URI,
        paths={"app_token": app_token},
        queries=queries,
    )
    if code != 0:
        return tool_error(f"List tables failed: code={code} msg={msg}")

    return tool_result(data)


# ---------------------------------------------------------------------------
# feishu_bitable_list_fields
# ---------------------------------------------------------------------------

_LIST_FIELDS_URI = "/open-apis/bitable/v1/apps/:app_token/tables/:table_id/fields"

BITABLE_LIST_FIELDS_SCHEMA = {
    "name": "feishu_bitable_list_fields",
    "description": (
        "List all fields (columns) in a Bitable data table. "
        "Returns field_id, field_name, type, and property for each field. "
        "Field types: 1=Text, 2=Number, 3=SingleSelect, 4=MultiSelect, "
        "5=DateTime, 7=Checkbox, 11=Attachment, 15=URL, 17=Location, "
        "18=AutoNumber, 20=Formula, 21=Link, 22=Lookup, "
        "23=CreatedTime, 24=ModifiedTime, 1001=User."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "app_token": {
                "type": "string",
                "description": "The Bitable app token.",
            },
            "table_id": {
                "type": "string",
                "description": "The table ID (from feishu_bitable_list_tables).",
            },
            "page_size": {
                "type": "integer",
                "description": "Number of fields per page (max 100).",
                "default": 100,
            },
            "page_token": {
                "type": "string",
                "description": "Pagination token for next page.",
            },
        },
        "required": ["app_token", "table_id"],
    },
}


def _handle_list_fields(args: dict, **kwargs) -> str:
    app_token = args.get("app_token", "").strip()
    table_id = args.get("table_id", "").strip()
    if not app_token or not table_id:
        return tool_error("app_token and table_id are required")

    page_size = args.get("page_size", 100)
    page_token = args.get("page_token", "")

    queries = [("page_size", str(page_size))]
    if page_token:
        queries.append(("page_token", page_token))

    code, msg, data = _do_request(
        "GET", _LIST_FIELDS_URI,
        paths={"app_token": app_token, "table_id": table_id},
        queries=queries,
    )
    if code != 0:
        return tool_error(f"List fields failed: code={code} msg={msg}")

    return tool_result(data)


# ---------------------------------------------------------------------------
# feishu_bitable_list_records
# ---------------------------------------------------------------------------

_LIST_RECORDS_URI = "/open-apis/bitable/v1/apps/:app_token/tables/:table_id/records"

BITABLE_LIST_RECORDS_SCHEMA = {
    "name": "feishu_bitable_list_records",
    "description": (
        "List records (rows) from a Bitable data table. "
        "Supports filtering, sorting, and field projection. "
        "Use filter for simple conditions, e.g. 'CurrentValue.[status] = \"done\"'. "
        "Returns items with record_id and fields."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "app_token": {
                "type": "string",
                "description": "The Bitable app token.",
            },
            "table_id": {
                "type": "string",
                "description": "The table ID.",
            },
            "filter": {
                "type": "string",
                "description": (
                    "Optional filter expression. "
                    "Simple: 'CurrentValue.[field_name] = \"value\"'. "
                    "Use feishu_bitable_search_records for AND/OR/parentheses."
                ),
            },
            "sort": {
                "type": "string",
                "description": (
                    "Optional JSON array of sort specs, "
                    "e.g. '[{\"field_name\":\"created_at\",\"desc\":true}]'."
                ),
            },
            "field_names": {
                "type": "string",
                "description": (
                    "Optional JSON array of field names to return, "
                    "e.g. '[\"name\",\"status\"]'. Omitting returns all fields."
                ),
            },
            "page_size": {
                "type": "integer",
                "description": "Records per page (default 100, max 500).",
                "default": 100,
            },
            "page_token": {
                "type": "string",
                "description": "Pagination token for next page.",
            },
        },
        "required": ["app_token", "table_id"],
    },
}


def _handle_list_records(args: dict, **kwargs) -> str:
    app_token = args.get("app_token", "").strip()
    table_id = args.get("table_id", "").strip()
    if not app_token or not table_id:
        return tool_error("app_token and table_id are required")

    page_size = args.get("page_size", 100)
    page_token = args.get("page_token", "")

    queries = [("page_size", str(page_size))]
    if page_token:
        queries.append(("page_token", page_token))

    # Optional filter / sort / field_names (only add if provided)
    filter_expr = args.get("filter", "").strip()
    if filter_expr:
        queries.append(("filter", filter_expr))

    sort_expr = args.get("sort", "").strip()
    if sort_expr:
        queries.append(("sort", sort_expr))

    field_names = args.get("field_names", "").strip()
    if field_names:
        queries.append(("field_names", field_names))

    code, msg, data = _do_request(
        "GET", _LIST_RECORDS_URI,
        paths={"app_token": app_token, "table_id": table_id},
        queries=queries,
    )
    if code != 0:
        return tool_error(f"List records failed: code={code} msg={msg}")

    return tool_result(data)


# ---------------------------------------------------------------------------
# feishu_bitable_search_records
# ---------------------------------------------------------------------------

_SEARCH_RECORDS_URI = "/open-apis/bitable/v1/apps/:app_token/tables/:table_id/records/search"

BITABLE_SEARCH_RECORDS_SCHEMA = {
    "name": "feishu_bitable_search_records",
    "description": (
        "Search records in a Bitable table with complex filter expressions. "
        "Supports AND, OR, parentheses. "
        "Example filter: 'CurrentValue.[status] = \"done\" AND CurrentValue.[priority] > 3'."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "app_token": {
                "type": "string",
                "description": "The Bitable app token.",
            },
            "table_id": {
                "type": "string",
                "description": "The table ID.",
            },
            "filter": {
                "type": "string",
                "description": (
                    "Filter expression supporting AND/OR/parentheses. "
                    "E.g. 'CurrentValue.[status] = \"done\" AND CurrentValue.[priority] > 3'."
                ),
            },
            "sort": {
                "type": "string",
                "description": (
                    "Optional JSON sort array, "
                    "e.g. '[{\"field_name\":\"created_at\",\"desc\":true}]'."
                ),
            },
            "field_names": {
                "type": "string",
                "description": (
                    "Optional JSON array of field names to return."
                ),
            },
            "page_size": {
                "type": "integer",
                "description": "Records per page (default 100, max 500).",
                "default": 100,
            },
            "page_token": {
                "type": "string",
                "description": "Pagination token for next page.",
            },
        },
        "required": ["app_token", "table_id"],
    },
}


def _handle_search_records(args: dict, **kwargs) -> str:
    app_token = args.get("app_token", "").strip()
    table_id = args.get("table_id", "").strip()
    if not app_token or not table_id:
        return tool_error("app_token and table_id are required")

    body: dict = {}

    filter_expr = args.get("filter", "").strip()
    if filter_expr:
        body["filter"] = filter_expr

    sort_expr = args.get("sort", "").strip()
    if sort_expr:
        try:
            body["sort"] = json.loads(sort_expr)
        except json.JSONDecodeError:
            return tool_error(f"sort is not valid JSON: {sort_expr}")

    field_names = args.get("field_names", "").strip()
    if field_names:
        try:
            body["field_names"] = json.loads(field_names)
        except json.JSONDecodeError:
            return tool_error(f"field_names is not valid JSON: {field_names}")

    page_size = args.get("page_size", 100)
    page_token = args.get("page_token", "")

    body["page_size"] = page_size
    if page_token:
        body["page_token"] = page_token

    code, msg, data = _do_request(
        "POST", _SEARCH_RECORDS_URI,
        paths={"app_token": app_token, "table_id": table_id},
        body=body,
    )
    if code != 0:
        return tool_error(f"Search records failed: code={code} msg={msg}")

    return tool_result(data)


# ---------------------------------------------------------------------------
# feishu_bitable_create_records
# ---------------------------------------------------------------------------

_CREATE_RECORDS_URI = "/open-apis/bitable/v1/apps/:app_token/tables/:table_id/records/batch_create"

BITABLE_CREATE_RECORDS_SCHEMA = {
    "name": "feishu_bitable_create_records",
    "description": (
        "Batch create records (rows) in a Bitable table. "
        "Pass records as a JSON array of {fields: {field_name: value}} objects. "
        "Returns the created records with their record_ids."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "app_token": {
                "type": "string",
                "description": "The Bitable app token.",
            },
            "table_id": {
                "type": "string",
                "description": "The table ID.",
            },
            "records": {
                "type": "string",
                "description": (
                    "JSON string: array of {fields: {field_name: value}} objects. "
                    "Example: '[{\"fields\":{\"name\":\"Alice\",\"status\":\"active\"}}]'."
                ),
            },
        },
        "required": ["app_token", "table_id", "records"],
    },
}


def _handle_create_records(args: dict, **kwargs) -> str:
    app_token = args.get("app_token", "").strip()
    table_id = args.get("table_id", "").strip()
    records_raw = args.get("records", "").strip()
    if not app_token or not table_id or not records_raw:
        return tool_error("app_token, table_id, and records are required")

    try:
        records = json.loads(records_raw)
    except json.JSONDecodeError as e:
        return tool_error(f"records is not valid JSON: {e}")

    if not isinstance(records, list):
        return tool_error("records must be a JSON array")

    body = {"records": records}

    code, msg, data = _do_request(
        "POST", _CREATE_RECORDS_URI,
        paths={"app_token": app_token, "table_id": table_id},
        body=body,
    )
    if code != 0:
        return tool_error(f"Create records failed: code={code} msg={msg}")

    return tool_result(success=True, data=data)


# ---------------------------------------------------------------------------
# feishu_bitable_update_records
# ---------------------------------------------------------------------------

_UPDATE_RECORDS_URI = "/open-apis/bitable/v1/apps/:app_token/tables/:table_id/records/batch_update"

BITABLE_UPDATE_RECORDS_SCHEMA = {
    "name": "feishu_bitable_update_records",
    "description": (
        "Batch update records (rows) in a Bitable table. "
        "Pass records as a JSON array of {record_id, fields: {field_name: value}} objects. "
        "Record IDs can be obtained from feishu_bitable_list_records or "
        "feishu_bitable_search_records."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "app_token": {
                "type": "string",
                "description": "The Bitable app token.",
            },
            "table_id": {
                "type": "string",
                "description": "The table ID.",
            },
            "records": {
                "type": "string",
                "description": (
                    "JSON string: array of {record_id, fields: {field_name: value}} objects. "
                    "Example: '[{\"record_id\":\"recXXXX\",\"fields\":{\"status\":\"done\"}}]'."
                ),
            },
        },
        "required": ["app_token", "table_id", "records"],
    },
}


def _handle_update_records(args: dict, **kwargs) -> str:
    app_token = args.get("app_token", "").strip()
    table_id = args.get("table_id", "").strip()
    records_raw = args.get("records", "").strip()
    if not app_token or not table_id or not records_raw:
        return tool_error("app_token, table_id, and records are required")

    try:
        records = json.loads(records_raw)
    except json.JSONDecodeError as e:
        return tool_error(f"records is not valid JSON: {e}")

    if not isinstance(records, list):
        return tool_error("records must be a JSON array")

    body = {"records": records}

    code, msg, data = _do_request(
        "POST", _UPDATE_RECORDS_URI,
        paths={"app_token": app_token, "table_id": table_id},
        body=body,
    )
    if code != 0:
        return tool_error(f"Update records failed: code={code} msg={msg}")

    return tool_result(success=True, data=data)


# ---------------------------------------------------------------------------
# feishu_bitable_create_table
# ---------------------------------------------------------------------------

_CREATE_TABLE_URI = "/open-apis/bitable/v1/apps/:app_token/tables"

BITABLE_CREATE_TABLE_SCHEMA = {
    "name": "feishu_bitable_create_table",
    "description": (
        "Create a new data table in a Feishu Bitable. "
        "Returns the new table_id and name."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "app_token": {
                "type": "string",
                "description": "The Bitable app token.",
            },
            "name": {
                "type": "string",
                "description": "Name for the new table.",
            },
        },
        "required": ["app_token", "name"],
    },
}


def _handle_create_table(args: dict, **kwargs) -> str:
    app_token = args.get("app_token", "").strip()
    name = args.get("name", "").strip()
    if not app_token or not name:
        return tool_error("app_token and name are required")

    body = {"table": {"name": name}}

    code, msg, data = _do_request(
        "POST", _CREATE_TABLE_URI,
        paths={"app_token": app_token},
        body=body,
    )
    if code != 0:
        return tool_error(f"Create table failed: code={code} msg={msg}")

    return tool_result(success=True, data=data)


# ---------------------------------------------------------------------------
# feishu_bitable_create_field
# ---------------------------------------------------------------------------

_CREATE_FIELD_URI = "/open-apis/bitable/v1/apps/:app_token/tables/:table_id/fields"

BITABLE_CREATE_FIELD_SCHEMA = {
    "name": "feishu_bitable_create_field",
    "description": (
        "Create a new field (column) in a Bitable table. "
        "Supported types with their property format:\n"
        "  1=Text: {}\n"
        '  3=SingleSelect: {"options":[{"name":"a","color":0}]}\n'
        '  5=DateTime: {"date_formatter":"yyyy/MM/dd"}\n'
        '  11=User: {"multiple":false}\n'
        '  21=Link: {"table_id":"tblXXX","is_multiple":true,"back_field_name":"..."}\n'
        '  22=Lookup: {"link_field_id":"fldXXX","target_field_id":"fldXXX",'
        '"records_limit":1,"accumulate_type":"max"}\n'
        "Colors for select options: 0=gray,1=blue,2=green,3=yellow,4=red,5=purple."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "app_token": {
                "type": "string",
                "description": "The Bitable app token.",
            },
            "table_id": {
                "type": "string",
                "description": "The table ID.",
            },
            "field_name": {
                "type": "string",
                "description": "Name for the new field.",
            },
            "type": {
                "type": "integer",
                "description": (
                    "Field type number: 1=Text, 3=SingleSelect, 5=DateTime, "
                    "11=User, 21=Link, 22=Lookup."
                ),
            },
            "property": {
                "type": "string",
                "description": (
                    "JSON property object specific to the field type. "
                    "See description for format per type."
                ),
            },
        },
        "required": ["app_token", "table_id", "field_name", "type"],
    },
}


def _handle_create_field(args: dict, **kwargs) -> str:
    app_token = args.get("app_token", "").strip()
    table_id = args.get("table_id", "").strip()
    field_name = args.get("field_name", "").strip()
    try:
        field_type = int(args.get("type", 0))
    except (TypeError, ValueError):
        return tool_error("type must be an integer")

    if not app_token or not table_id or not field_name:
        return tool_error("app_token, table_id, and field_name are required")
    if not field_type:
        return tool_error("type is required")

    body: dict = {"field_name": field_name, "type": field_type}

    property_raw = args.get("property", "").strip()
    if property_raw:
        try:
            body["property"] = json.loads(property_raw)
        except json.JSONDecodeError as e:
            return tool_error(f"property is not valid JSON: {e}")

    code, msg, data = _do_request(
        "POST", _CREATE_FIELD_URI,
        paths={"app_token": app_token, "table_id": table_id},
        body=body,
    )
    if code != 0:
        return tool_error(f"Create field failed: code={code} msg={msg}")

    return tool_result(success=True, data=data)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

registry.register(
    name="feishu_bitable_list_tables",
    toolset="feishu_bitable",
    schema=BITABLE_LIST_TABLES_SCHEMA,
    handler=_handle_list_tables,
    check_fn=_check_feishu,
    requires_env=[],
    is_async=False,
    description="List Bitable data tables",
    emoji="\U0001f4ca",
)

registry.register(
    name="feishu_bitable_list_fields",
    toolset="feishu_bitable",
    schema=BITABLE_LIST_FIELDS_SCHEMA,
    handler=_handle_list_fields,
    check_fn=_check_feishu,
    requires_env=[],
    is_async=False,
    description="List Bitable fields (columns)",
    emoji="\U0001f4ca",
)

registry.register(
    name="feishu_bitable_list_records",
    toolset="feishu_bitable",
    schema=BITABLE_LIST_RECORDS_SCHEMA,
    handler=_handle_list_records,
    check_fn=_check_feishu,
    requires_env=[],
    is_async=False,
    description="List Bitable records (rows)",
    emoji="\U0001f4ca",
)

registry.register(
    name="feishu_bitable_search_records",
    toolset="feishu_bitable",
    schema=BITABLE_SEARCH_RECORDS_SCHEMA,
    handler=_handle_search_records,
    check_fn=_check_feishu,
    requires_env=[],
    is_async=False,
    description="Search Bitable records with complex filters",
    emoji="\U0001f50d",
)

registry.register(
    name="feishu_bitable_create_records",
    toolset="feishu_bitable",
    schema=BITABLE_CREATE_RECORDS_SCHEMA,
    handler=_handle_create_records,
    check_fn=_check_feishu,
    requires_env=[],
    is_async=False,
    description="Batch create Bitable records",
    emoji="\u2795",
)

registry.register(
    name="feishu_bitable_update_records",
    toolset="feishu_bitable",
    schema=BITABLE_UPDATE_RECORDS_SCHEMA,
    handler=_handle_update_records,
    check_fn=_check_feishu,
    requires_env=[],
    is_async=False,
    description="Batch update Bitable records",
    emoji="\u270f\ufe0f",
)

registry.register(
    name="feishu_bitable_create_table",
    toolset="feishu_bitable",
    schema=BITABLE_CREATE_TABLE_SCHEMA,
    handler=_handle_create_table,
    check_fn=_check_feishu,
    requires_env=[],
    is_async=False,
    description="Create a new Bitable data table",
    emoji="\U0001f4ca",
)

registry.register(
    name="feishu_bitable_create_field",
    toolset="feishu_bitable",
    schema=BITABLE_CREATE_FIELD_SCHEMA,
    handler=_handle_create_field,
    check_fn=_check_feishu,
    requires_env=[],
    is_async=False,
    description="Create a new Bitable field (column)",
    emoji="\u2795",
)

# ---------------------------------------------------------------------------
# feishu_bitable_delete_field
# ---------------------------------------------------------------------------

_DELETE_FIELD_URI = "/open-apis/bitable/v1/apps/:app_token/tables/:table_id/fields/:field_id"

BITABLE_DELETE_FIELD_SCHEMA = {
    "name": "feishu_bitable_delete_field",
    "description": (
        "Delete a field (column) from a Bitable table. "
        "Cannot delete the primary index field."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "app_token": {
                "type": "string",
                "description": "The Bitable app token.",
            },
            "table_id": {
                "type": "string",
                "description": "The table ID.",
            },
            "field_id": {
                "type": "string",
                "description": "The field ID to delete.",
            },
        },
        "required": ["app_token", "table_id", "field_id"],
    },
}


def _handle_delete_field(args: dict, **kwargs) -> str:
    app_token = args.get("app_token", "").strip()
    table_id = args.get("table_id", "").strip()
    field_id = args.get("field_id", "").strip()
    if not app_token or not table_id or not field_id:
        return tool_error("app_token, table_id, and field_id are required")

    code, msg, data = _do_request(
        "DELETE", _DELETE_FIELD_URI,
        paths={"app_token": app_token, "table_id": table_id, "field_id": field_id},
    )
    if code != 0:
        return tool_error(f"Delete field failed: code={code} msg={msg}")

    return tool_result(success=True)


# ---------------------------------------------------------------------------
# feishu_bitable_update_field (rename)
# ---------------------------------------------------------------------------

_UPDATE_FIELD_URI = "/open-apis/bitable/v1/apps/:app_token/tables/:table_id/fields/:field_id"

BITABLE_UPDATE_FIELD_SCHEMA = {
    "name": "feishu_bitable_update_field",
    "description": (
        "Update a field's name or type in a Bitable table. "
        "Both field_name and type are required per Feishu API."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "app_token": {
                "type": "string",
                "description": "The Bitable app token.",
            },
            "table_id": {
                "type": "string",
                "description": "The table ID.",
            },
            "field_id": {
                "type": "string",
                "description": "The field ID to update.",
            },
            "field_name": {
                "type": "string",
                "description": "New field name.",
            },
            "type": {
                "type": "integer",
                "description": "Field type number (must match current type).",
            },
        },
        "required": ["app_token", "table_id", "field_id", "field_name", "type"],
    },
}


def _handle_update_field(args: dict, **kwargs) -> str:
    app_token = args.get("app_token", "").strip()
    table_id = args.get("table_id", "").strip()
    field_id = args.get("field_id", "").strip()
    field_name = args.get("field_name", "").strip()
    try:
        field_type = int(args.get("type", 0))
    except (TypeError, ValueError):
        return tool_error("type must be an integer")

    if not all([app_token, table_id, field_id, field_name, field_type]):
        return tool_error("app_token, table_id, field_id, field_name, and type are required")

    body = {"field_name": field_name, "type": field_type}

    code, msg, data = _do_request(
        "PUT", _UPDATE_FIELD_URI,
        paths={"app_token": app_token, "table_id": table_id, "field_id": field_id},
        body=body,
    )
    if code != 0:
        return tool_error(f"Update field failed: code={code} msg={msg}")

    return tool_result(success=True, data=data)


# ---------------------------------------------------------------------------
# feishu_bitable_list_views
# ---------------------------------------------------------------------------

_LIST_VIEWS_URI = "/open-apis/bitable/v1/apps/:app_token/tables/:table_id/views"

BITABLE_LIST_VIEWS_SCHEMA = {
    "name": "feishu_bitable_list_views",
    "description": (
        "List all views on a Bitable data table. "
        "Returns view_id, view_name, and view_type for each view."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "app_token": {
                "type": "string",
                "description": "The Bitable app token.",
            },
            "table_id": {
                "type": "string",
                "description": "The table ID.",
            },
            "page_size": {
                "type": "integer",
                "description": "Number of views per page (max 100).",
                "default": 100,
            },
            "page_token": {
                "type": "string",
                "description": "Pagination token for next page.",
            },
        },
        "required": ["app_token", "table_id"],
    },
}


def _handle_list_views(args: dict, **kwargs) -> str:
    app_token = args.get("app_token", "").strip()
    table_id = args.get("table_id", "").strip()
    if not app_token or not table_id:
        return tool_error("app_token and table_id are required")

    page_size = args.get("page_size", 100)
    page_token = args.get("page_token", "")

    queries = [("page_size", str(page_size))]
    if page_token:
        queries.append(("page_token", page_token))

    code, msg, data = _do_request(
        "GET", _LIST_VIEWS_URI,
        paths={"app_token": app_token, "table_id": table_id},
        queries=queries,
    )
    if code != 0:
        return tool_error(f"List views failed: code={code} msg={msg}")

    return tool_result(data)


# ---------------------------------------------------------------------------
# feishu_bitable_create_view
# ---------------------------------------------------------------------------

_CREATE_VIEW_URI = "/open-apis/bitable/v1/apps/:app_token/tables/:table_id/views"

BITABLE_CREATE_VIEW_SCHEMA = {
    "name": "feishu_bitable_create_view",
    "description": (
        "Create a new view on a Bitable table. "
        "Supported view types: grid (表格), kanban (看板), gallery (画册), "
        "gantt (甘特), form (表单). Note: view configuration (group/filter/sort) "
        "cannot be set via API — configure manually in the Feishu UI after creation."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "app_token": {
                "type": "string",
                "description": "The Bitable app token.",
            },
            "table_id": {
                "type": "string",
                "description": "The table ID.",
            },
            "view_name": {
                "type": "string",
                "description": "Name for the new view.",
            },
            "view_type": {
                "type": "string",
                "description": (
                    "View type: 'grid', 'kanban', 'gallery', 'gantt', or 'form'."
                ),
            },
        },
        "required": ["app_token", "table_id", "view_name", "view_type"],
    },
}


def _handle_create_view(args: dict, **kwargs) -> str:
    app_token = args.get("app_token", "").strip()
    table_id = args.get("table_id", "").strip()
    view_name = args.get("view_name", "").strip()
    view_type = args.get("view_type", "").strip()

    if not all([app_token, table_id, view_name, view_type]):
        return tool_error("app_token, table_id, view_name, and view_type are required")

    valid_types = {"grid", "kanban", "gallery", "gantt", "form"}
    if view_type not in valid_types:
        return tool_error(
            f"view_type must be one of: {', '.join(sorted(valid_types))}"
        )

    body = {"view_name": view_name, "view_type": view_type}

    code, msg, data = _do_request(
        "POST", _CREATE_VIEW_URI,
        paths={"app_token": app_token, "table_id": table_id},
        body=body,
    )
    if code != 0:
        return tool_error(f"Create view failed: code={code} msg={msg}")

    return tool_result(success=True, data=data)


# ---------------------------------------------------------------------------
# feishu_bitable_delete_view
# ---------------------------------------------------------------------------

_DELETE_VIEW_URI = "/open-apis/bitable/v1/apps/:app_token/tables/:table_id/views/:view_id"

BITABLE_DELETE_VIEW_SCHEMA = {
    "name": "feishu_bitable_delete_view",
    "description": (
        "Delete a view from a Bitable table. "
        "A table must always have at least one view remaining."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "app_token": {
                "type": "string",
                "description": "The Bitable app token.",
            },
            "table_id": {
                "type": "string",
                "description": "The table ID.",
            },
            "view_id": {
                "type": "string",
                "description": "The view ID to delete.",
            },
        },
        "required": ["app_token", "table_id", "view_id"],
    },
}


def _handle_delete_view(args: dict, **kwargs) -> str:
    app_token = args.get("app_token", "").strip()
    table_id = args.get("table_id", "").strip()
    view_id = args.get("view_id", "").strip()
    if not app_token or not table_id or not view_id:
        return tool_error("app_token, table_id, and view_id are required")

    code, msg, data = _do_request(
        "DELETE", _DELETE_VIEW_URI,
        paths={"app_token": app_token, "table_id": table_id, "view_id": view_id},
    )
    if code != 0:
        return tool_error(f"Delete view failed: code={code} msg={msg}")

    return tool_result(success=True)


# ---------------------------------------------------------------------------
# Registration of new tools
# ---------------------------------------------------------------------------

registry.register(
    name="feishu_bitable_delete_field",
    toolset="feishu_bitable",
    schema=BITABLE_DELETE_FIELD_SCHEMA,
    handler=_handle_delete_field,
    check_fn=_check_feishu,
    requires_env=[],
    is_async=False,
    description="Delete a Bitable field (column)",
    emoji="\U0001f5d1\ufe0f",
)

registry.register(
    name="feishu_bitable_update_field",
    toolset="feishu_bitable",
    schema=BITABLE_UPDATE_FIELD_SCHEMA,
    handler=_handle_update_field,
    check_fn=_check_feishu,
    requires_env=[],
    is_async=False,
    description="Update a Bitable field name",
    emoji="\u270f\ufe0f",
)

registry.register(
    name="feishu_bitable_list_views",
    toolset="feishu_bitable",
    schema=BITABLE_LIST_VIEWS_SCHEMA,
    handler=_handle_list_views,
    check_fn=_check_feishu,
    requires_env=[],
    is_async=False,
    description="List views on a Bitable table",
    emoji="\U0001f4ca",
)

registry.register(
    name="feishu_bitable_create_view",
    toolset="feishu_bitable",
    schema=BITABLE_CREATE_VIEW_SCHEMA,
    handler=_handle_create_view,
    check_fn=_check_feishu,
    requires_env=[],
    is_async=False,
    description="Create a new Bitable view",
    emoji="\U0001f4ca",
)

registry.register(
    name="feishu_bitable_delete_view",
    toolset="feishu_bitable",
    schema=BITABLE_DELETE_VIEW_SCHEMA,
    handler=_handle_delete_view,
    check_fn=_check_feishu,
    requires_env=[],
    is_async=False,
    description="Delete a Bitable view",
    emoji="\U0001f5d1\ufe0f",
)
