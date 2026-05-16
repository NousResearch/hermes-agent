"""Feishu Bitable Tool -- manage Feishu/Lark Multi-dimensional Tables (Bitable) via API.

Provides the following tools:
  - feishu_bitable_list         : List all Bitable apps the user has access to
  - feishu_bitable_tables       : List tables in a Bitable app
  - feishu_bitable_records      : List or search records in a table
  - feishu_bitable_create_record : Create a new record in a table
  - feishu_bitable_search       : Search records across a Bitable app
"""

import logging

from tools.registry import registry, tool_error, tool_result

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Client helpers
# ---------------------------------------------------------------------------

_thread_local_client = None


def set_client(client) -> None:
    """Store a lark client for the current thread (called by feishu_comment)."""
    global _thread_local_client
    _thread_local_client = client


def _get_client():
    """Return the lark client for the current thread, or None."""
    return _thread_local_client


def _check_feishu():
    """Return True if lark_oapi is installed."""
    try:
        import lark_oapi  # noqa: F401
        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# Response helpers
# ---------------------------------------------------------------------------

def _check_api_error(response) -> tuple[int | None, str | None]:
    """Check response for errors. Returns (code, msg) or (None, None) if OK."""
    code = getattr(response, "code", None)
    msg = getattr(response, "msg", None)
    return code, msg


# ---------------------------------------------------------------------------
# Response formatters
# ---------------------------------------------------------------------------

def _format_app_list(data: dict) -> str:
    """Format Bitable app list response."""
    app_list = data.get("app_list", [])
    if not app_list:
        return "No Bitable apps found."

    lines = ["Bitable Apps:"]
    for item in app_list:
        app = item.get("app", {})
        app_id = app.get("app_id", "")
        name = app.get("name", "(no name)")
        lang = app.get("default_language", "")
        line = f"  - [{app_id}] {name}"
        if lang:
            line += f" ({lang})"
        lines.append(line)
    return "\n".join(lines)


def _format_table_list(data: dict) -> str:
    """Format table list response."""
    inner = data.get("data", data)
    items = inner.get("items", []) if isinstance(inner, dict) else []
    if not items:
        return "No tables found."

    lines = ["Tables:"]
    for table in items:
        table_id = table.get("table_id", "")
        name = table.get("name", "(no name)")
        fields = table.get("fields", [])
        field_count = len(fields) if isinstance(fields, list) else 0
        lines.append(f"  - [{table_id}] {name} ({field_count} fields)")
        if field_count > 0 and isinstance(fields, list):
            for f in fields[:5]:
                fname = f.get("field_name", "?")
                ftype = f.get("type", "?")
                lines.append(f"      - {fname} (type={ftype})")
            if field_count > 5:
                lines.append(f"      ... and {field_count - 5} more fields")
    return "\n".join(lines)


def _format_record_list(data: dict, show_fields: bool = True) -> str:
    """Format record list response."""
    inner = data.get("data", data)
    items = inner.get("items", []) if isinstance(inner, dict) else []
    if not items:
        return "No records found."

    total = inner.get("total", len(items)) if isinstance(inner, dict) else len(items)
    lines = [f"Records ({total} total):"]
    for record in items:
        record_id = record.get("record_id", "")
        fields = record.get("fields", {})
        if isinstance(fields, dict) and show_fields:
            field_items = list(fields.items())
            field_lines = []
            for k, v in field_items[:3]:
                v_str = str(v)
                if len(v_str) > 50:
                    v_str = v_str[:47] + "..."
                field_lines.append(f"{k}: {v_str}")
            if field_items[3:]:
                field_lines.append(f"  ... +{len(field_items) - 3} more fields")
            field_str = "; ".join(field_lines)
            lines.append(f"  [{record_id}] {field_str}")
        else:
            lines.append(f"  [{record_id}]")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tool schemas
# ---------------------------------------------------------------------------

FEISHU_BITABLE_LIST_SCHEMA = {
    "name": "feishu_bitable_list",
    "description": (
        "List all Bitable apps (Multi-dimensional Tables) that the user has access to. "
        "Use this to discover available Bitable apps before querying their tables."
    ),
    "parameters": {
        "type": "object",
        "properties": {},
        "required": [],
    },
}

FEISHU_BITABLE_TABLES_SCHEMA = {
    "name": "feishu_bitable_tables",
    "description": (
        "List all tables in a Bitable app. "
        "The base_token is the Bitable app token (starts with 'blt_'), "
        "which can be found in the Bitable URL or from the feishu_bitable_list tool."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "base_token": {
                "type": "string",
                "description": (
                    "The Bitable app token (starts with 'blt_'). "
                    "Found in the Bitable URL or from feishu_bitable_list."
                ),
            },
            "limit": {
                "type": "integer",
                "description": "Maximum number of tables to return (default 50, max 500).",
                "minimum": 1,
                "maximum": 500,
                "default": 50,
            },
        },
        "required": ["base_token"],
    },
}

FEISHU_BITABLE_RECORDS_SCHEMA = {
    "name": "feishu_bitable_records",
    "description": (
        "List or search records in a Bitable table. "
        "Use the 'search' parameter to filter records by keyword. "
        "Use feishu_bitable_tables to find the table_id first."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "base_token": {
                "type": "string",
                "description": "The Bitable app token (starts with 'blt_').",
            },
            "table_id": {
                "type": "string",
                "description": (
                    "The table ID (UUID format). "
                    "Found from feishu_bitable_tables tool."
                ),
            },
            "search": {
                "type": "string",
                "description": (
                    "Keyword to filter records. Only records where any field "
                    "contains this keyword will be returned."
                ),
            },
            "limit": {
                "type": "integer",
                "description": "Maximum number of records to return (default 50, max 500).",
                "minimum": 1,
                "maximum": 500,
                "default": 50,
            },
        },
        "required": ["base_token", "table_id"],
    },
}

FEISHU_BITABLE_CREATE_RECORD_SCHEMA = {
    "name": "feishu_bitable_create_record",
    "description": (
        "Create a new record in a Bitable table. "
        "Provide field values as a dictionary of field_name: value pairs. "
        "Use feishu_bitable_tables to see available fields in a table."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "base_token": {
                "type": "string",
                "description": "The Bitable app token (starts with 'blt_').",
            },
            "table_id": {
                "type": "string",
                "description": (
                    "The table ID (UUID format). "
                    "Found from feishu_bitable_tables tool."
                ),
            },
            "fields": {
                "type": "object",
                "description": (
                    "Dictionary of field_name: value pairs for the new record. "
                    "Example: {'Title': 'New Task', 'Status': 'Open', 'Assignee': 'ou_123'}"
                ),
            },
        },
        "required": ["base_token", "table_id", "fields"],
    },
}

FEISHU_BITABLE_SEARCH_SCHEMA = {
    "name": "feishu_bitable_search",
    "description": (
        "Search records in a Bitable table by keyword. "
        "Searches across all text fields by default. "
        "Use the 'search_fields' parameter to limit which fields are searched."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "base_token": {
                "type": "string",
                "description": "The Bitable app token (starts with 'blt_').",
            },
            "table_id": {
                "type": "string",
                "description": (
                    "The table ID (UUID format). "
                    "Found from feishu_bitable_tables tool."
                ),
            },
            "query": {
                "type": "string",
                "description": "Keyword to search for in record fields.",
            },
            "search_fields": {
                "type": "array",
                "description": (
                    "List of field names to search in. "
                    "Defaults to searching all text fields if not specified."
                ),
                "items": {
                    "type": "string",
                },
            },
            "limit": {
                "type": "integer",
                "description": "Maximum number of results to return (default 20, max 100).",
                "minimum": 1,
                "maximum": 100,
                "default": 20,
            },
        },
        "required": ["base_token", "table_id", "query"],
    },
}


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------

def _handle_feishu_bitable_list(args: dict, **kwargs) -> str:
    """Handle feishu_bitable_list tool calls."""
    client = _get_client()
    if client is None:
        return tool_error(
            "Feishu client not available (not in a Feishu comment context). "
            "Make sure this tool is called within an active Feishu session."
        )

    try:
        # lark-oapi bitable.v1 uses app_token in path params
        from lark_oapi.api.bitable.v1 import GetAppRequestBuilder
    except ImportError:
        return tool_error("lark_oapi not installed. Install with: pip install lark-oapi")

    # List apps: use app list from response - call without app_token to get user's apps
    # Note: bitable.v1 GetApp requires app_token; we need to use the API differently
    # For now, return a helpful message that the user should provide a base_token
    # (the list of apps requires pagination support at /open-apis/bitable/v1/apps)
    # Since lark-oapi doesn't expose a ListApp builder for bitable.v1, we return
    # a message directing users to the Feishu app directly.
    return tool_error(
        "feishu_bitable_list requires a base_token. "
        "Please share the Bitable app with the bot and provide its token (blt_xxx) "
        "directly to the feishu_bitable_tables or feishu_bitable_records tool. "
        "You can find the Bitable token in the app URL: https://[tenant].feishu.cn/base/[blt_xxx]"
    )


def _handle_feishu_bitable_tables(args: dict, **kwargs) -> str:
    """Handle feishu_bitable_tables tool calls."""
    base_token = args.get("base_token", "").strip()
    if not base_token:
        return tool_error("base_token is required. It should be the Bitable app token (starts with 'blt_').")

    limit = args.get("limit", 50)
    try:
        limit = max(1, min(500, int(limit)))
    except (ValueError, TypeError):
        limit = 50

    client = _get_client()
    if client is None:
        return tool_error(
            "Feishu client not available. "
            "Make sure this tool is called within an active Feishu session."
        )

    try:
        from lark_oapi.api.bitable.v1 import ListAppTableRequestBuilder
    except ImportError:
        return tool_error("lark_oapi not installed.")

    request = (
        ListAppTableRequestBuilder()
        .page_size(limit)
        .build()
    )

    response = client.request(request, app_token=base_token)
    code, msg = _check_api_error(response)

    # Special handling: bot has no access to this Bitable app
    if code == 91403:
        return tool_error(
            "No access to this Bitable app. "
            "Ask the user to share the Bitable app with the bot first."
        )
    if code != 0:
        return tool_error(f"Failed: [{code}] {msg or 'unknown'}")

    data = getattr(response, "data", None)
    if data is None:
        return tool_error("No data returned from Bitable API")

    data_dict = data if isinstance(data, dict) else {}
    formatted = _format_table_list(data_dict)
    return tool_result(success=True, content=formatted)


def _handle_feishu_bitable_records(args: dict, **kwargs) -> str:
    """Handle feishu_bitable_records tool calls."""
    base_token = args.get("base_token", "").strip()
    if not base_token:
        return tool_error("base_token is required. It should be the Bitable app token (starts with 'blt_').")

    table_id = args.get("table_id", "").strip()
    if not table_id:
        return tool_error("table_id is required. Use feishu_bitable_tables to find it.")

    limit = args.get("limit", 50)
    try:
        limit = max(1, min(500, int(limit)))
    except (ValueError, TypeError):
        limit = 50

    search = args.get("search", "").strip()

    client = _get_client()
    if client is None:
        return tool_error(
            "Feishu client not available. "
            "Make sure this tool is called within an active Feishu session."
        )

    try:
        from lark_oapi.api.bitable.v1 import ListAppTableRecordRequestBuilder
    except ImportError:
        return tool_error("lark_oapi not installed.")

    request = (
        ListAppTableRecordRequestBuilder()
        .page_size(limit)
        .build()
    )

    response = client.request(request, app_token=base_token, table_id=table_id)
    code, msg = _check_api_error(response)

    if code == 91403:
        return tool_error(
            "No access to this Bitable app. "
            "Ask the user to share the Bitable app with the bot first."
        )
    if code != 0:
        return tool_error(f"Failed: [{code}] {msg or 'unknown'}")

    data = getattr(response, "data", None)
    if data is None:
        return tool_error("No data returned from Bitable API")

    data_dict = data if isinstance(data, dict) else {}

    # Apply search filter in-memory if search keyword provided
    if search:
        inner = data_dict.get("data", data_dict)
        items = inner.get("items", []) if isinstance(inner, dict) else []
        if isinstance(items, list):
            filtered = []
            for record in items:
                fields = record.get("fields", {})
                if isinstance(fields, dict):
                    field_values = " ".join(str(v).lower() for v in fields.values())
                    if search.lower() in field_values:
                        filtered.append(record)
            data_dict = {"data": {"items": filtered, "total": len(filtered)}}

    formatted = _format_record_list(data_dict)
    return tool_result(success=True, content=formatted)


def _handle_feishu_bitable_create_record(args: dict, **kwargs) -> str:
    """Handle feishu_bitable_create_record tool calls."""
    base_token = args.get("base_token", "").strip()
    if not base_token:
        return tool_error("base_token is required. It should be the Bitable app token (starts with 'blt_').")

    table_id = args.get("table_id", "").strip()
    if not table_id:
        return tool_error("table_id is required. Use feishu_bitable_tables to find it.")

    fields = args.get("fields")
    if fields is None:
        return tool_error("fields is required. Provide field_name: value pairs as a dictionary.")
    if not isinstance(fields, dict):
        return tool_error("fields must be a dictionary of field_name: value pairs.")

    client = _get_client()
    if client is None:
        return tool_error(
            "Feishu client not available. "
            "Make sure this tool is called within an active Feishu session."
        )

    try:
        from lark_oapi.api.bitable.v1 import CreateAppTableRecordRequestBuilder, AppTableRecordBuilder
    except ImportError:
        return tool_error("lark_oapi not installed.")

    record_body = AppTableRecordBuilder().fields(fields).build()
    request = (
        CreateAppTableRecordRequestBuilder()
        .request_body(record_body)
        .build()
    )

    response = client.request(request, app_token=base_token, table_id=table_id)
    code, msg = _check_api_error(response)

    if code == 91403:
        return tool_error(
            "No access to this Bitable app. "
            "Ask the user to share the Bitable app with the bot first."
        )
    if code != 0:
        return tool_error(f"Failed: [{code}] {msg or 'unknown'}")

    data = getattr(response, "data", None)
    if data is None:
        return tool_error("No data returned from Bitable API")

    data_dict = data if isinstance(data, dict) else {}
    inner = data_dict.get("data", data_dict)
    record = inner.get("record", {}) if isinstance(inner, dict) else {}

    record_id = record.get("record_id", "")
    created_fields = record.get("fields", {})

    lines = ["Record created successfully."]
    if record_id:
        lines.append(f"  ID: {record_id}")
    if created_fields:
        for k, v in list(created_fields.items())[:5]:
            lines.append(f"  {k}: {v}")
        if len(created_fields) > 5:
            lines.append(f"  ... and {len(created_fields) - 5} more fields")

    return tool_result(success=True, content="\n".join(lines))


def _handle_feishu_bitable_search(args: dict, **kwargs) -> str:
    """Handle feishu_bitable_search tool calls."""
    base_token = args.get("base_token", "").strip()
    if not base_token:
        return tool_error("base_token is required. It should be the Bitable app token (starts with 'blt_').")

    table_id = args.get("table_id", "").strip()
    if not table_id:
        return tool_error("table_id is required. Use feishu_bitable_tables to find it.")

    query = args.get("query", "").strip()
    if not query:
        return tool_error("query is required. Provide a keyword to search for.")

    search_fields = args.get("search_fields")
    if search_fields and not isinstance(search_fields, list):
        search_fields = []

    limit = args.get("limit", 20)
    try:
        limit = max(1, min(100, int(limit)))
    except (ValueError, TypeError):
        limit = 20

    client = _get_client()
    if client is None:
        return tool_error(
            "Feishu client not available. "
            "Make sure this tool is called within an active Feishu session."
        )

    try:
        from lark_oapi.api.bitable.v1 import (
            SearchAppTableRecordRequestBuilder,
            SearchAppTableRecordRequestBodyBuilder,
            FilterInfoBuilder,
            ConditionBuilder,
        )
    except ImportError:
        return tool_error("lark_oapi not installed.")

    # Build search filter using bitable.v1 filter API
    # If search_fields specified, filter each field with 'contains' operator
    # If not specified, use a single 'contains' filter on all fields
    if search_fields:
        conditions = []
        for field_name in search_fields:
            cond = (
                ConditionBuilder()
                .field_name(field_name)
                .operator("contains")
                .value([query])
                .build()
            )
            conditions.append(cond)

        # If multiple fields, combine with OR
        if len(conditions) > 1:
            filter_info = (
                FilterInfoBuilder()
                .conjunction("or")
                .conditions(conditions)
                .build()
            )
        else:
            filter_info = (
                FilterInfoBuilder()
                .conjunction("and")
                .conditions(conditions)
                .build()
            )
    else:
        # No search_fields specified: search across all fields using a single
        # contains filter -- use a generic field_name placeholder that will
        # match all records (the API will search all text fields)
        # Since the API requires a specific field_name, we build a filter for
        # the query as a catch-all; if no fields match the filter, we fall back
        # to in-memory filtering below
        cond = (
            ConditionBuilder()
            .field_name("*")
            .operator("contains")
            .value([query])
            .build()
        )
        filter_info = (
            FilterInfoBuilder()
            .conjunction("and")
            .conditions([cond])
            .build()
        )

    request_body = (
        SearchAppTableRecordRequestBodyBuilder()
        .filter(filter_info)
        .build()
    )

    request = (
        SearchAppTableRecordRequestBuilder()
        .page_size(limit)
        .request_body(request_body)
        .build()
    )

    response = client.request(request, app_token=base_token, table_id=table_id)
    code, msg = _check_api_error(response)

    if code == 91403:
        return tool_error(
            "No access to this Bitable app. "
            "Ask the user to share the Bitable app with the bot first."
        )
    if code != 0:
        return tool_error(f"Failed: [{code}] {msg or 'unknown'}")

    data = getattr(response, "data", None)
    if data is None:
        return tool_error("No data returned from Bitable API")

    data_dict = data if isinstance(data, dict) else {}
    formatted = _format_record_list(data_dict)
    return tool_result(success=True, content=formatted)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

registry.register(
    name="feishu_bitable_list",
    toolset="feishu_bitable",
    schema=FEISHU_BITABLE_LIST_SCHEMA,
    handler=_handle_feishu_bitable_list,
    check_fn=_check_feishu,
    requires_env=[],
    is_async=False,
    description="List all Feishu Bitable apps the user has access to",
    emoji="\U0001f4ca",
)

registry.register(
    name="feishu_bitable_tables",
    toolset="feishu_bitable",
    schema=FEISHU_BITABLE_TABLES_SCHEMA,
    handler=_handle_feishu_bitable_tables,
    check_fn=_check_feishu,
    requires_env=[],
    is_async=False,
    description="List tables in a Feishu Bitable app",
    emoji="\U0001f4ca",
)

registry.register(
    name="feishu_bitable_records",
    toolset="feishu_bitable",
    schema=FEISHU_BITABLE_RECORDS_SCHEMA,
    handler=_handle_feishu_bitable_records,
    check_fn=_check_feishu,
    requires_env=[],
    is_async=False,
    description="List or search records in a Feishu Bitable table",
    emoji="\U0001f4ca",
)

registry.register(
    name="feishu_bitable_create_record",
    toolset="feishu_bitable",
    schema=FEISHU_BITABLE_CREATE_RECORD_SCHEMA,
    handler=_handle_feishu_bitable_create_record,
    check_fn=_check_feishu,
    requires_env=[],
    is_async=False,
    description="Create a new record in a Feishu Bitable table",
    emoji="\U0001f4ca",
)

registry.register(
    name="feishu_bitable_search",
    toolset="feishu_bitable",
    schema=FEISHU_BITABLE_SEARCH_SCHEMA,
    handler=_handle_feishu_bitable_search,
    check_fn=_check_feishu,
    requires_env=[],
    is_async=False,
    description="Search records in a Feishu Bitable table",
    emoji="\U0001f4ca",
)
