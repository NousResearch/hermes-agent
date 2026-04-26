"""Feishu Search Tool -- message and global search via Feishu/Lark API.

Provides:
  ``feishu_search_message``  -- search messages (POST /search/v2/message)
  ``feishu_search_global``   -- global content search (POST /search/v2/search)

Uses FeishuClient.for_user() with UAT (user_access_token) identity.
Requires scope: search:search
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
# TOOLS_METADATA entries
# ---------------------------------------------------------------------------

TOOLS_METADATA["feishu_search_message"] = {
    "identity": "user",
    "scopes": ["search:search"],
}

TOOLS_METADATA["feishu_search_global"] = {
    "identity": "user",
    "scopes": ["search:search"],
}


def _check_feishu():
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


# ---------------------------------------------------------------------------
# feishu_search_message
# ---------------------------------------------------------------------------

_SEARCH_MESSAGE_URI = "/open-apis/search/v2/message"

FEISHU_SEARCH_MESSAGE_SCHEMA = {
    "name": "feishu_search_message",
    "description": (
        "Search Feishu messages by keyword. "
        "Supports filtering by sender, chat, and time range."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search keyword.",
            },
            "from_ids": {
                "type": "array",
                "description": "Filter by sender open_id list (optional).",
                "items": {"type": "string"},
            },
            "chat_ids": {
                "type": "array",
                "description": "Filter by chat_id list (optional).",
                "items": {"type": "string"},
            },
            "start_time": {
                "type": "string",
                "description": "Search start time as Unix timestamp string (optional, e.g. '1609459200').",
            },
            "end_time": {
                "type": "string",
                "description": "Search end time as Unix timestamp string (optional, e.g. '1640995200').",
            },
            "page_size": {
                "type": "integer",
                "description": "Number of results per page (default: 20, max: 100).",
                "default": 20,
            },
            "page_token": {
                "type": "string",
                "description": "Pagination token for next page.",
            },
        },
        "required": ["query"],
    },
}


def _handle_search_message(args: dict, **kwargs) -> str:
    """Handler for feishu_search_message tool.

    Args:
        args: Tool arguments from user/model.
        **kwargs: Additional keyword arguments.

    Returns:
        JSON string (tool_error or tool_result).
    """
    query = args.get("query", "").strip()
    if not query:
        return tool_error("query is required")

    logger.info("feishu_search_message: query=%r", query)

    from_ids = args.get("from_ids") or []
    chat_ids = args.get("chat_ids") or []
    start_time = args.get("start_time", "")
    end_time = args.get("end_time", "")
    page_size = args.get("page_size", 20)
    page_token = args.get("page_token", "")

    try:
        client = FeishuClient.for_user()
    except NeedAuthorizationError as exc:
        return tool_error(_auth_error_message(exc))
    except ValueError as exc:
        return tool_error(f"Feishu configuration error: {exc}")

    body: dict = {
        "query": query,
        "page_size": page_size,
    }
    if from_ids:
        body["from_ids"] = from_ids
    if chat_ids:
        body["chat_ids"] = chat_ids
    if start_time:
        body["start_time"] = start_time
    if end_time:
        body["end_time"] = end_time
    if page_token:
        body["page_token"] = page_token

    try:
        code, msg, data = client.do_request(
            "POST",
            _SEARCH_MESSAGE_URI,
            body=body,
            use_uat=True,
        )
    except RuntimeError as exc:
        return tool_error(str(exc))

    if code != 0:
        try:
            raise_for_feishu_errcode(code, msg or "", api_name="feishu.search.message")
        except (AppScopeMissingError, UserAuthRequiredError, UserScopeInsufficientError, NeedAuthorizationError) as e:
            return tool_error(_auth_error_message(e))
        logger.warning("feishu_search_message failed: code=%d msg=%s", code, msg)
        return tool_error(f"Search message failed: code={code} msg={msg}")

    result_count = len(data.get("items", []))
    logger.info("feishu_search_message: found %d results for query=%r", result_count, query)
    return tool_result(data)


# ---------------------------------------------------------------------------
# feishu_search_global
# ---------------------------------------------------------------------------

_SEARCH_GLOBAL_URI = "/open-apis/search/v2/search"

FEISHU_SEARCH_GLOBAL_SCHEMA = {
    "name": "feishu_search_global",
    "description": (
        "Perform a global search across Feishu content (docs, messages, wikis, etc.). "
        "Returns ranked results from the user's accessible content."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search keyword.",
            },
            "page_size": {
                "type": "integer",
                "description": "Number of results per page (default: 20, max: 50).",
                "default": 20,
            },
            "page_token": {
                "type": "string",
                "description": "Pagination token for next page.",
            },
        },
        "required": ["query"],
    },
}


def _handle_search_global(args: dict, **kwargs) -> str:
    """Handler for feishu_search_global tool.

    Args:
        args: Tool arguments from user/model.
        **kwargs: Additional keyword arguments.

    Returns:
        JSON string (tool_error or tool_result).
    """
    query = args.get("query", "").strip()
    if not query:
        return tool_error("query is required")

    page_size = args.get("page_size", 20)
    page_token = args.get("page_token", "")

    logger.info("feishu_search_global: query=%r page_size=%d", query, page_size)

    try:
        client = FeishuClient.for_user()
    except NeedAuthorizationError as exc:
        return tool_error(_auth_error_message(exc))
    except ValueError as exc:
        return tool_error(f"Feishu configuration error: {exc}")

    body: dict = {
        "query": query,
        "page_size": page_size,
    }
    if page_token:
        body["page_token"] = page_token

    try:
        code, msg, data = client.do_request(
            "POST",
            _SEARCH_GLOBAL_URI,
            body=body,
            use_uat=True,
        )
    except RuntimeError as exc:
        return tool_error(str(exc))

    if code != 0:
        try:
            raise_for_feishu_errcode(code, msg or "", api_name="feishu.search.global")
        except (AppScopeMissingError, UserAuthRequiredError, UserScopeInsufficientError, NeedAuthorizationError) as e:
            return tool_error(_auth_error_message(e))
        logger.warning("feishu_search_global failed: code=%d msg=%s", code, msg)
        return tool_error(f"Global search failed: code={code} msg={msg}")

    result_count = len(data.get("items", []))
    logger.info("feishu_search_global: found %d results for query=%r", result_count, query)
    return tool_result(data)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

registry.register(
    name="feishu_search_message",
    toolset="feishu_search",
    schema=FEISHU_SEARCH_MESSAGE_SCHEMA,
    handler=_handle_search_message,
    check_fn=_check_feishu,
    requires_env=[],
    is_async=False,
    description="Search Feishu messages by keyword",
    emoji="\U0001f50d",
)

registry.register(
    name="feishu_search_global",
    toolset="feishu_search",
    schema=FEISHU_SEARCH_GLOBAL_SCHEMA,
    handler=_handle_search_global,
    check_fn=_check_feishu,
    requires_env=[],
    is_async=False,
    description="Global search across Feishu content",
    emoji="\U0001f310",
)
