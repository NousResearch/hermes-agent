"""Feishu Wiki Tool -- knowledge base search and node retrieval via Feishu/Lark API.

Provides ``feishu_wiki_search`` for searching wiki nodes and
``feishu_wiki_get_node`` for retrieving a single wiki node by token.
Uses FeishuClient.for_user() (UAT) with scope wiki:wiki:readonly.
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

TOOLS_METADATA["feishu_wiki_search"] = {
    "identity": "user",
    "scopes": ["wiki:wiki:readonly"],
}

TOOLS_METADATA["feishu_wiki_get_node"] = {
    "identity": "user",
    "scopes": ["wiki:wiki:readonly"],
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
# feishu_wiki_search
# ---------------------------------------------------------------------------

_WIKI_SEARCH_URI = "/open-apis/wiki/v1/nodes/search"

FEISHU_WIKI_SEARCH_SCHEMA = {
    "name": "feishu_wiki_search",
    "description": (
        "Search wiki nodes in Feishu knowledge base by keyword. "
        "Returns a list of matching nodes with their tokens, titles, and metadata."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search keyword or phrase.",
            },
            "page_size": {
                "type": "integer",
                "description": "Number of results per page (max 50, default 10).",
                "default": 10,
            },
        },
        "required": ["query"],
    },
}


def _handle_wiki_search(args: dict, **kwargs) -> str:
    """Handler for feishu_wiki_search tool.

    Args:
        args: Tool arguments from user/model.
        **kwargs: Additional keyword arguments.

    Returns:
        JSON string (tool_error or tool_result).
    """
    query = args.get("query", "").strip()
    if not query:
        return tool_error("query is required")

    page_size = args.get("page_size", 10)
    if not isinstance(page_size, int) or page_size < 1:
        page_size = 10
    if page_size > 50:
        page_size = 50

    logger.info("feishu_wiki_search: query=%r page_size=%d", query, page_size)

    try:
        feishu = FeishuClient.for_user()
    except NeedAuthorizationError as exc:
        return tool_error(_auth_error_message(exc))
    except Exception as exc:
        return tool_error(f"Feishu client unavailable: {exc}")

    body = {
        "query": query,
        "page_size": page_size,
    }

    try:
        code, msg, data = feishu.do_request(
            "POST",
            _WIKI_SEARCH_URI,
            body=body,
            use_uat=True,
        )
    except Exception as exc:
        return tool_error(f"Request failed: {exc}")

    if code != 0:
        try:
            raise_for_feishu_errcode(code, msg or "", api_name="feishu.wiki.search")
        except (AppScopeMissingError, UserAuthRequiredError, UserScopeInsufficientError, NeedAuthorizationError) as e:
            return tool_error(_auth_error_message(e))
        logger.warning("feishu_wiki_search failed: code=%d msg=%s", code, msg)
        return tool_error(f"Wiki search failed: code={code} msg={msg}")

    logger.info(
        "feishu_wiki_search: returned %d items",
        len(data.get("items", [])),
    )
    return tool_result(data)


# ---------------------------------------------------------------------------
# feishu_wiki_get_node
# ---------------------------------------------------------------------------

_WIKI_GET_NODE_URI = "/open-apis/wiki/v2/spaces/get_node"

FEISHU_WIKI_GET_NODE_SCHEMA = {
    "name": "feishu_wiki_get_node",
    "description": (
        "Get metadata for a Feishu wiki node by its node token. "
        "Returns node info including title, obj_token, space_id, and type."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "node_token": {
                "type": "string",
                "description": "The wiki node token to retrieve.",
            },
        },
        "required": ["node_token"],
    },
}


def _handle_wiki_get_node(args: dict, **kwargs) -> str:
    """Handler for feishu_wiki_get_node tool.

    Args:
        args: Tool arguments from user/model.
        **kwargs: Additional keyword arguments.

    Returns:
        JSON string (tool_error or tool_result).
    """
    node_token = args.get("node_token", "").strip()
    if not node_token:
        return tool_error("node_token is required")

    logger.info("feishu_wiki_get_node: node_token=%r", node_token)

    try:
        feishu = FeishuClient.for_user()
    except NeedAuthorizationError as exc:
        return tool_error(_auth_error_message(exc))
    except Exception as exc:
        return tool_error(f"Feishu client unavailable: {exc}")

    queries = [("token", node_token), ("obj_type", "wiki")]

    try:
        code, msg, data = feishu.do_request(
            "GET",
            _WIKI_GET_NODE_URI,
            queries=queries,
            use_uat=True,
        )
    except Exception as exc:
        return tool_error(f"Request failed: {exc}")

    if code != 0:
        try:
            raise_for_feishu_errcode(code, msg or "", api_name="feishu.wiki.get_node")
        except (AppScopeMissingError, UserAuthRequiredError, UserScopeInsufficientError, NeedAuthorizationError) as e:
            return tool_error(_auth_error_message(e))
        logger.warning("feishu_wiki_get_node failed: code=%d msg=%s", code, msg)
        return tool_error(f"Get wiki node failed: code={code} msg={msg}")

    logger.info("feishu_wiki_get_node: retrieved node %r", node_token)
    return tool_result(data)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

registry.register(
    name="feishu_wiki_search",
    toolset="feishu_wiki",
    schema=FEISHU_WIKI_SEARCH_SCHEMA,
    handler=_handle_wiki_search,
    check_fn=_check_feishu,
    requires_env=[],
    is_async=False,
    description="Search Feishu wiki nodes by keyword",
    emoji="\U0001f50d",
)

registry.register(
    name="feishu_wiki_get_node",
    toolset="feishu_wiki",
    schema=FEISHU_WIKI_GET_NODE_SCHEMA,
    handler=_handle_wiki_get_node,
    check_fn=_check_feishu,
    requires_env=[],
    is_async=False,
    description="Get Feishu wiki node metadata by token",
    emoji="\U0001f4c4",
)
