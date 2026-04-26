"""Feishu Bitable Tool -- multidimensional table (bitable) operations via Feishu/Lark API.

Provides six tools for interacting with Feishu Bitable (多维表格):
  - ``feishu_bitable_list_apps``      list accessible bitable apps
  - ``feishu_bitable_list_tables``    list tables inside an app
  - ``feishu_bitable_list_records``   list records in a table
  - ``feishu_bitable_search_records`` search records with filter conditions
  - ``feishu_bitable_create_record``  create a new record
  - ``feishu_bitable_update_record``  update an existing record

All tools use UAT (user_access_token) via FeishuClient.for_user() and require
the bitable:app scope.
"""

import logging

from tools.feishu_oapi_client import (
    AppScopeMissingError,
    FeishuClient,
    NeedAuthorizationError,
    TOOLS_METADATA,
    UserAuthRequiredError,
    UserScopeInsufficientError,
    raise_for_feishu_errcode,
)
from tools.registry import registry, tool_error, tool_result

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# TOOLS_METADATA — declare scopes and identity for all bitable tools
# ---------------------------------------------------------------------------

_BITABLE_TOOLS = [
    "feishu_bitable_list_apps",
    "feishu_bitable_list_tables",
    "feishu_bitable_list_records",
    "feishu_bitable_search_records",
    "feishu_bitable_create_record",
    "feishu_bitable_update_record",
]

for _tool_name in _BITABLE_TOOLS:
    TOOLS_METADATA[_tool_name] = {
        "identity": "user",
        "scopes": ["bitable:app"],
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _check_feishu():
    """Check if lark_oapi is available."""
    try:
        import lark_oapi  # noqa: F401
        return True
    except ImportError:
        return False


def _auth_error_message(exc: Exception) -> str:
    """Format semantic auth exceptions as tool_error strings."""
    if isinstance(exc, NeedAuthorizationError):
        return f"Need Feishu authorization: {exc}. Run 'hermes feishu-uat' to authorize."
    if isinstance(exc, AppScopeMissingError):
        return f"App scope missing: {exc}"
    if isinstance(exc, UserAuthRequiredError):
        return f"User authorization required: {exc}"
    if isinstance(exc, UserScopeInsufficientError):
        return f"User scope insufficient: {exc}"
    return str(exc)


def _get_user_client():
    """Return (client, error_str). error_str is None on success."""
    try:
        return FeishuClient.for_user(), None
    except NeedAuthorizationError as exc:
        return None, _auth_error_message(exc)
    except ValueError as exc:
        return None, f"Feishu configuration error: {exc}"


# ---------------------------------------------------------------------------
# feishu_bitable_list_apps
# ---------------------------------------------------------------------------

_LIST_APPS_URI = "/open-apis/drive/v1/files"

FEISHU_BITABLE_LIST_APPS_SCHEMA = {
    "name": "feishu_bitable_list_apps",
    "description": (
        "List Feishu Bitable apps (multidimensional tables) accessible to the current user. "
        "Returns files of type 'bitable' from the user's Drive."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "folder_token": {
                "type": "string",
                "description": "Folder token to list from (default: user's root space).",
            },
            "page_size": {
                "type": "integer",
                "description": "Number of files per page (max 200, default 50).",
                "default": 50,
            },
            "page_token": {
                "type": "string",
                "description": "Pagination token for next page.",
            },
        },
        "required": [],
    },
}


def _handle_bitable_list_apps(args: dict, **kwargs) -> str:
    """Handler for feishu_bitable_list_apps."""
    logger.info("feishu_bitable_list_apps: folder_token=%s", args.get("folder_token", ""))
    client, err = _get_user_client()
    if err:
        return tool_error(err)

    folder_token = args.get("folder_token", "") or ""
    page_size = args.get("page_size", 50) or 50
    page_token = args.get("page_token", "") or ""

    queries = [("page_size", str(page_size))]
    if folder_token:
        queries.append(("folder_token", folder_token))
    if page_token:
        queries.append(("page_token", page_token))

    code, msg, data = client.do_request(
        "GET", _LIST_APPS_URI,
        queries=queries,
        use_uat=True,
    )
    if code != 0:
        try:
            raise_for_feishu_errcode(code, msg or "", api_name="feishu.bitable.list_apps")
        except (AppScopeMissingError, UserAuthRequiredError, UserScopeInsufficientError, NeedAuthorizationError) as e:
            return tool_error(_auth_error_message(e))
        logger.warning("feishu_bitable_list_apps failed: code=%d msg=%s", code, msg)
        return tool_error(f"List bitable apps failed: code={code} msg={msg}")

    files = data.get("files", [])
    bitable_apps = [f for f in files if f.get("type") == "bitable"]
    logger.info("feishu_bitable_list_apps: returned %d bitable apps", len(bitable_apps))

    return tool_result({
        "apps": bitable_apps,
        "has_more": data.get("has_more", False),
        "page_token": data.get("next_page_token", data.get("page_token", "")),
    })


# ---------------------------------------------------------------------------
# feishu_bitable_list_tables
# ---------------------------------------------------------------------------

_LIST_TABLES_URI = "/open-apis/bitable/v1/apps/:app_token/tables"

FEISHU_BITABLE_LIST_TABLES_SCHEMA = {
    "name": "feishu_bitable_list_tables",
    "description": "List all data tables inside a Feishu Bitable app.",
    "parameters": {
        "type": "object",
        "properties": {
            "app_token": {
                "type": "string",
                "description": "The bitable app token.",
            },
            "page_size": {
                "type": "integer",
                "description": "Number of tables per page (max 100, default 50).",
                "default": 50,
            },
            "page_token": {
                "type": "string",
                "description": "Pagination token for next page.",
            },
        },
        "required": ["app_token"],
    },
}


def _handle_bitable_list_tables(args: dict, **kwargs) -> str:
    """Handler for feishu_bitable_list_tables."""
    app_token = args.get("app_token", "").strip()
    logger.info("feishu_bitable_list_tables: app_token=%s", app_token)
    client, err = _get_user_client()
    if err:
        return tool_error(err)

    if not app_token:
        return tool_error("app_token is required")

    page_size = args.get("page_size", 50) or 50
    page_token = args.get("page_token", "") or ""

    queries = [("page_size", str(page_size))]
    if page_token:
        queries.append(("page_token", page_token))

    code, msg, data = client.do_request(
        "GET", _LIST_TABLES_URI,
        paths={"app_token": app_token},
        queries=queries,
        use_uat=True,
    )
    if code != 0:
        try:
            raise_for_feishu_errcode(code, msg or "", api_name="feishu.bitable.list_tables")
        except (AppScopeMissingError, UserAuthRequiredError, UserScopeInsufficientError, NeedAuthorizationError) as e:
            return tool_error(_auth_error_message(e))
        logger.warning("feishu_bitable_list_tables failed: code=%d msg=%s", code, msg)
        return tool_error(f"List tables failed: code={code} msg={msg}")

    tables = data.get("items", [])
    logger.info("feishu_bitable_list_tables: app=%s returned %d tables", app_token, len(tables))

    return tool_result({
        "tables": tables,
        "has_more": data.get("has_more", False),
        "page_token": data.get("page_token", ""),
    })


# ---------------------------------------------------------------------------
# feishu_bitable_list_records
# ---------------------------------------------------------------------------

_SEARCH_RECORDS_URI = "/open-apis/bitable/v1/apps/:app_token/tables/:table_id/records/search"

FEISHU_BITABLE_LIST_RECORDS_SCHEMA = {
    "name": "feishu_bitable_list_records",
    "description": (
        "List records in a Feishu Bitable table. "
        "Returns all records without filter conditions. "
        "Use feishu_bitable_search_records to filter by field values."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "app_token": {
                "type": "string",
                "description": "The bitable app token.",
            },
            "table_id": {
                "type": "string",
                "description": "The data table ID.",
            },
            "page_size": {
                "type": "integer",
                "description": "Number of records per page (max 500, default 50).",
                "default": 50,
            },
            "page_token": {
                "type": "string",
                "description": "Pagination token for next page.",
            },
        },
        "required": ["app_token", "table_id"],
    },
}


def _handle_bitable_list_records(args: dict, **kwargs) -> str:
    """Handler for feishu_bitable_list_records."""
    app_token = args.get("app_token", "").strip()
    table_id = args.get("table_id", "").strip()
    logger.info("feishu_bitable_list_records: app_token=%s table_id=%s", app_token, table_id)
    client, err = _get_user_client()
    if err:
        return tool_error(err)

    if not app_token or not table_id:
        return tool_error("app_token and table_id are required")

    page_size = args.get("page_size", 50) or 50
    page_token = args.get("page_token", "") or ""

    queries = [
        ("user_id_type", "open_id"),
        ("page_size", str(page_size)),
    ]
    if page_token:
        queries.append(("page_token", page_token))

    code, msg, data = client.do_request(
        "POST", _SEARCH_RECORDS_URI,
        paths={"app_token": app_token, "table_id": table_id},
        queries=queries,
        body={},
        use_uat=True,
    )
    if code != 0:
        try:
            raise_for_feishu_errcode(code, msg or "", api_name="feishu.bitable.list_records")
        except (AppScopeMissingError, UserAuthRequiredError, UserScopeInsufficientError, NeedAuthorizationError) as e:
            return tool_error(_auth_error_message(e))
        logger.warning("feishu_bitable_list_records failed: code=%d msg=%s", code, msg)
        return tool_error(f"List records failed: code={code} msg={msg}")

    records = data.get("items", [])
    logger.info(
        "feishu_bitable_list_records: app=%s table=%s returned %d records",
        app_token, table_id, len(records),
    )

    return tool_result({
        "records": records,
        "has_more": data.get("has_more", False),
        "page_token": data.get("page_token", ""),
        "total": data.get("total"),
    })


# ---------------------------------------------------------------------------
# feishu_bitable_search_records
# ---------------------------------------------------------------------------

FEISHU_BITABLE_SEARCH_RECORDS_SCHEMA = {
    "name": "feishu_bitable_search_records",
    "description": (
        "Search records in a Feishu Bitable table using filter conditions. "
        "Filter example: {\"conjunction\": \"and\", \"conditions\": [{\"field_name\": \"Status\", \"operator\": \"is\", \"value\": [\"Done\"]}]}"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "app_token": {
                "type": "string",
                "description": "The bitable app token.",
            },
            "table_id": {
                "type": "string",
                "description": "The data table ID.",
            },
            "filter": {
                "type": "object",
                "description": (
                    "Filter conditions object with 'conjunction' (and/or) and 'conditions' array. "
                    "Each condition: {field_name, operator, value}. "
                    "Operators: is, isNot, contains, doesNotContain, isEmpty, isNotEmpty, "
                    "isGreater, isGreaterEqual, isLess, isLessEqual."
                ),
            },
            "page_size": {
                "type": "integer",
                "description": "Number of records per page (max 500, default 50).",
                "default": 50,
            },
            "page_token": {
                "type": "string",
                "description": "Pagination token for next page.",
            },
        },
        "required": ["app_token", "table_id", "filter"],
    },
}


def _handle_bitable_search_records(args: dict, **kwargs) -> str:
    """Handler for feishu_bitable_search_records."""
    app_token = args.get("app_token", "").strip()
    table_id = args.get("table_id", "").strip()
    logger.info("feishu_bitable_search_records: app_token=%s table_id=%s", app_token, table_id)
    client, err = _get_user_client()
    if err:
        return tool_error(err)

    if not app_token or not table_id:
        return tool_error("app_token and table_id are required")

    filter_obj = args.get("filter")
    if not filter_obj or not isinstance(filter_obj, dict):
        return tool_error("filter is required and must be an object with conjunction and conditions")

    page_size = args.get("page_size", 50) or 50
    page_token = args.get("page_token", "") or ""

    queries = [
        ("user_id_type", "open_id"),
        ("page_size", str(page_size)),
    ]
    if page_token:
        queries.append(("page_token", page_token))

    # Auto-fix isEmpty/isNotEmpty conditions that lack value field
    conditions = filter_obj.get("conditions", [])
    fixed_conditions = []
    for cond in conditions:
        if cond.get("operator") in ("isEmpty", "isNotEmpty") and "value" not in cond:
            logger.debug(
                "feishu_bitable_search_records: auto-adding value=[] for %s operator",
                cond.get("operator"),
            )
            cond = {**cond, "value": []}
        fixed_conditions.append(cond)

    body = {"filter": {**filter_obj, "conditions": fixed_conditions}}

    code, msg, data = client.do_request(
        "POST", _SEARCH_RECORDS_URI,
        paths={"app_token": app_token, "table_id": table_id},
        queries=queries,
        body=body,
        use_uat=True,
    )
    if code != 0:
        try:
            raise_for_feishu_errcode(code, msg or "", api_name="feishu.bitable.search_records")
        except (AppScopeMissingError, UserAuthRequiredError, UserScopeInsufficientError, NeedAuthorizationError) as e:
            return tool_error(_auth_error_message(e))
        logger.warning("feishu_bitable_search_records failed: code=%d msg=%s", code, msg)
        return tool_error(f"Search records failed: code={code} msg={msg}")

    records = data.get("items", [])
    logger.info(
        "feishu_bitable_search_records: app=%s table=%s returned %d records",
        app_token, table_id, len(records),
    )

    return tool_result({
        "records": records,
        "has_more": data.get("has_more", False),
        "page_token": data.get("page_token", ""),
        "total": data.get("total"),
    })


# ---------------------------------------------------------------------------
# feishu_bitable_create_record
# ---------------------------------------------------------------------------

_CREATE_RECORD_URI = "/open-apis/bitable/v1/apps/:app_token/tables/:table_id/records"

FEISHU_BITABLE_CREATE_RECORD_SCHEMA = {
    "name": "feishu_bitable_create_record",
    "description": (
        "Create a new record in a Feishu Bitable table. "
        "Pass field values as a key-value object where keys are field names."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "app_token": {
                "type": "string",
                "description": "The bitable app token.",
            },
            "table_id": {
                "type": "string",
                "description": "The data table ID.",
            },
            "fields": {
                "type": "object",
                "description": (
                    "Record fields as key-value pairs. "
                    "Value types by field type: text=string, number=number, "
                    "single_select=string (option name), multi_select=string[] (option names), "
                    "date=number (ms timestamp), checkbox=boolean, person=[{id: 'ou_xxx'}]."
                ),
            },
        },
        "required": ["app_token", "table_id", "fields"],
    },
}


def _handle_bitable_create_record(args: dict, **kwargs) -> str:
    """Handler for feishu_bitable_create_record."""
    app_token = args.get("app_token", "").strip()
    table_id = args.get("table_id", "").strip()
    logger.info("feishu_bitable_create_record: app_token=%s table_id=%s", app_token, table_id)
    client, err = _get_user_client()
    if err:
        return tool_error(err)

    if not app_token or not table_id:
        return tool_error("app_token and table_id are required")

    fields = args.get("fields")
    if not fields or not isinstance(fields, dict):
        return tool_error("fields is required and must be a non-empty object")
    if len(fields) == 0:
        return tool_error("fields cannot be empty")

    code, msg, data = client.do_request(
        "POST", _CREATE_RECORD_URI,
        paths={"app_token": app_token, "table_id": table_id},
        queries=[("user_id_type", "open_id")],
        body={"fields": fields},
        use_uat=True,
    )
    if code != 0:
        try:
            raise_for_feishu_errcode(code, msg or "", api_name="feishu.bitable.create_record")
        except (AppScopeMissingError, UserAuthRequiredError, UserScopeInsufficientError, NeedAuthorizationError) as e:
            return tool_error(_auth_error_message(e))
        logger.warning("feishu_bitable_create_record failed: code=%d msg=%s", code, msg)
        return tool_error(f"Create record failed: code={code} msg={msg}")

    record = data.get("record", {})
    logger.info(
        "feishu_bitable_create_record: app=%s table=%s created record %s",
        app_token, table_id, record.get("record_id", "unknown"),
    )

    return tool_result(success=True, data={"record": record})


# ---------------------------------------------------------------------------
# feishu_bitable_update_record
# ---------------------------------------------------------------------------

_UPDATE_RECORD_URI = "/open-apis/bitable/v1/apps/:app_token/tables/:table_id/records/:record_id"

FEISHU_BITABLE_UPDATE_RECORD_SCHEMA = {
    "name": "feishu_bitable_update_record",
    "description": (
        "Update an existing record in a Feishu Bitable table. "
        "Only the fields provided will be updated; other fields remain unchanged."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "app_token": {
                "type": "string",
                "description": "The bitable app token.",
            },
            "table_id": {
                "type": "string",
                "description": "The data table ID.",
            },
            "record_id": {
                "type": "string",
                "description": "The record ID to update.",
            },
            "fields": {
                "type": "object",
                "description": (
                    "Fields to update as key-value pairs. "
                    "Only specified fields are changed; others remain untouched."
                ),
            },
        },
        "required": ["app_token", "table_id", "record_id", "fields"],
    },
}


def _handle_bitable_update_record(args: dict, **kwargs) -> str:
    """Handler for feishu_bitable_update_record."""
    app_token = args.get("app_token", "").strip()
    table_id = args.get("table_id", "").strip()
    record_id = args.get("record_id", "").strip()
    logger.info(
        "feishu_bitable_update_record: app_token=%s table_id=%s record_id=%s",
        app_token, table_id, record_id,
    )
    client, err = _get_user_client()
    if err:
        return tool_error(err)

    if not app_token or not table_id or not record_id:
        return tool_error("app_token, table_id, and record_id are required")

    fields = args.get("fields")
    if not fields or not isinstance(fields, dict):
        return tool_error("fields is required and must be a non-empty object")
    if len(fields) == 0:
        return tool_error("fields cannot be empty")

    code, msg, data = client.do_request(
        "PUT", _UPDATE_RECORD_URI,
        paths={"app_token": app_token, "table_id": table_id, "record_id": record_id},
        queries=[("user_id_type", "open_id")],
        body={"fields": fields},
        use_uat=True,
    )
    if code != 0:
        try:
            raise_for_feishu_errcode(code, msg or "", api_name="feishu.bitable.update_record")
        except (AppScopeMissingError, UserAuthRequiredError, UserScopeInsufficientError, NeedAuthorizationError) as e:
            return tool_error(_auth_error_message(e))
        logger.warning("feishu_bitable_update_record failed: code=%d msg=%s", code, msg)
        return tool_error(f"Update record failed: code={code} msg={msg}")

    record = data.get("record", {})
    logger.info(
        "feishu_bitable_update_record: app=%s table=%s updated record %s",
        app_token, table_id, record_id,
    )

    return tool_result(success=True, data={"record": record})


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

registry.register(
    name="feishu_bitable_list_apps",
    toolset="feishu_bitable",
    schema=FEISHU_BITABLE_LIST_APPS_SCHEMA,
    handler=_handle_bitable_list_apps,
    check_fn=_check_feishu,
    requires_env=[],
    is_async=False,
    description="List accessible Feishu Bitable apps",
    emoji="\U0001f4ca",
)

registry.register(
    name="feishu_bitable_list_tables",
    toolset="feishu_bitable",
    schema=FEISHU_BITABLE_LIST_TABLES_SCHEMA,
    handler=_handle_bitable_list_tables,
    check_fn=_check_feishu,
    requires_env=[],
    is_async=False,
    description="List tables in a Feishu Bitable app",
    emoji="\U0001f4ca",
)

registry.register(
    name="feishu_bitable_list_records",
    toolset="feishu_bitable",
    schema=FEISHU_BITABLE_LIST_RECORDS_SCHEMA,
    handler=_handle_bitable_list_records,
    check_fn=_check_feishu,
    requires_env=[],
    is_async=False,
    description="List records in a Feishu Bitable table",
    emoji="\U0001f4cb",
)

registry.register(
    name="feishu_bitable_search_records",
    toolset="feishu_bitable",
    schema=FEISHU_BITABLE_SEARCH_RECORDS_SCHEMA,
    handler=_handle_bitable_search_records,
    check_fn=_check_feishu,
    requires_env=[],
    is_async=False,
    description="Search records by filter conditions in a Feishu Bitable table",
    emoji="\U0001f50d",
)

registry.register(
    name="feishu_bitable_create_record",
    toolset="feishu_bitable",
    schema=FEISHU_BITABLE_CREATE_RECORD_SCHEMA,
    handler=_handle_bitable_create_record,
    check_fn=_check_feishu,
    requires_env=[],
    is_async=False,
    description="Create a new record in a Feishu Bitable table",
    emoji="➕",
)

registry.register(
    name="feishu_bitable_update_record",
    toolset="feishu_bitable",
    schema=FEISHU_BITABLE_UPDATE_RECORD_SCHEMA,
    handler=_handle_bitable_update_record,
    check_fn=_check_feishu,
    requires_env=[],
    is_async=False,
    description="Update an existing record in a Feishu Bitable table",
    emoji="✏️",
)
