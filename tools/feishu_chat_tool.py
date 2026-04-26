"""Feishu Chat Tool -- group chat info and member listing via Feishu/Lark API.

Provides:
  ``feishu_chat_get_info``      -- get details for a specific chat group
  ``feishu_chat_list_members``  -- list members in a chat group

Uses FeishuClient.for_user() with UAT (user_access_token) identity.
Requires scope: im:chat:readonly
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

TOOLS_METADATA["feishu_chat_get_info"] = {
    "identity": "user",
    "scopes": ["im:chat:readonly"],
}

TOOLS_METADATA["feishu_chat_list_members"] = {
    "identity": "user",
    "scopes": ["im:chat:readonly"],
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
# feishu_chat_get_info
# ---------------------------------------------------------------------------

_CHAT_GET_URI = "/open-apis/im/v1/chats/:chat_id"

FEISHU_CHAT_GET_INFO_SCHEMA = {
    "name": "feishu_chat_get_info",
    "description": (
        "Get detailed information about a Feishu group chat. "
        "Returns chat name, description, avatar, owner, member count, and permission settings."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "chat_id": {
                "type": "string",
                "description": "The chat group ID (e.g. oc_xxx).",
            },
            "user_id_type": {
                "type": "string",
                "description": "User ID type for returned members (default: open_id).",
                "default": "open_id",
            },
        },
        "required": ["chat_id"],
    },
}


def _handle_chat_get_info(args: dict, **kwargs) -> str:
    """Handler for feishu_chat_get_info tool.

    Args:
        args: Tool arguments from user/model.
        **kwargs: Additional keyword arguments.

    Returns:
        JSON string (tool_error or tool_result).
    """
    chat_id = args.get("chat_id", "").strip()
    if not chat_id:
        return tool_error("chat_id is required")

    logger.info("feishu_chat_get_info: chat_id=%s", chat_id)

    user_id_type = args.get("user_id_type", "open_id") or "open_id"

    try:
        client = FeishuClient.for_user()
    except NeedAuthorizationError as exc:
        return tool_error(_auth_error_message(exc))
    except ValueError as exc:
        return tool_error(f"Feishu configuration error: {exc}")

    try:
        code, msg, data = client.do_request(
            "GET",
            _CHAT_GET_URI,
            paths={"chat_id": chat_id},
            queries=[("user_id_type", user_id_type)],
            use_uat=True,
        )
    except RuntimeError as exc:
        return tool_error(str(exc))

    if code != 0:
        try:
            raise_for_feishu_errcode(code, msg or "", api_name="feishu.chat.get_info")
        except (AppScopeMissingError, UserAuthRequiredError, UserScopeInsufficientError, NeedAuthorizationError) as e:
            return tool_error(_auth_error_message(e))
        logger.warning("feishu_chat_get_info failed: code=%d msg=%s", code, msg)
        return tool_error(f"Get chat info failed: code={code} msg={msg}")

    logger.info("feishu_chat_get_info: retrieved info for chat_id=%s", chat_id)
    return tool_result(data)


# ---------------------------------------------------------------------------
# feishu_chat_list_members
# ---------------------------------------------------------------------------

_CHAT_MEMBERS_URI = "/open-apis/im/v1/chats/:chat_id/members"

FEISHU_CHAT_LIST_MEMBERS_SCHEMA = {
    "name": "feishu_chat_list_members",
    "description": (
        "List members in a Feishu group chat. "
        "Returns member IDs and names. Bot members are not included."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "chat_id": {
                "type": "string",
                "description": "The chat group ID (e.g. oc_xxx).",
            },
            "page_size": {
                "type": "integer",
                "description": "Number of members per page (default: 20, max: 100).",
                "default": 20,
            },
            "page_token": {
                "type": "string",
                "description": "Pagination token for next page.",
            },
            "member_id_type": {
                "type": "string",
                "description": "Member ID type (default: open_id).",
                "default": "open_id",
            },
        },
        "required": ["chat_id"],
    },
}


def _handle_chat_list_members(args: dict, **kwargs) -> str:
    """Handler for feishu_chat_list_members tool.

    Args:
        args: Tool arguments from user/model.
        **kwargs: Additional keyword arguments.

    Returns:
        JSON string (tool_error or tool_result).
    """
    chat_id = args.get("chat_id", "").strip()
    if not chat_id:
        return tool_error("chat_id is required")

    logger.info("feishu_chat_list_members: chat_id=%s", chat_id)

    page_size = args.get("page_size", 20)
    page_token = args.get("page_token", "")
    member_id_type = args.get("member_id_type", "open_id") or "open_id"

    try:
        client = FeishuClient.for_user()
    except NeedAuthorizationError as exc:
        return tool_error(_auth_error_message(exc))
    except ValueError as exc:
        return tool_error(f"Feishu configuration error: {exc}")

    queries = [
        ("member_id_type", member_id_type),
        ("page_size", str(page_size)),
    ]
    if page_token:
        queries.append(("page_token", page_token))

    try:
        code, msg, data = client.do_request(
            "GET",
            _CHAT_MEMBERS_URI,
            paths={"chat_id": chat_id},
            queries=queries,
            use_uat=True,
        )
    except RuntimeError as exc:
        return tool_error(str(exc))

    if code != 0:
        try:
            raise_for_feishu_errcode(code, msg or "", api_name="feishu.chat.list_members")
        except (AppScopeMissingError, UserAuthRequiredError, UserScopeInsufficientError, NeedAuthorizationError) as e:
            return tool_error(_auth_error_message(e))
        logger.warning("feishu_chat_list_members failed: code=%d msg=%s", code, msg)
        return tool_error(f"List chat members failed: code={code} msg={msg}")

    member_count = len(data.get("items", []))
    logger.info(
        "feishu_chat_list_members: found %d members for chat_id=%s",
        member_count, chat_id,
    )
    return tool_result(data)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

registry.register(
    name="feishu_chat_get_info",
    toolset="feishu_chat",
    schema=FEISHU_CHAT_GET_INFO_SCHEMA,
    handler=_handle_chat_get_info,
    check_fn=_check_feishu,
    requires_env=[],
    is_async=False,
    description="Get Feishu group chat details",
    emoji="\U0001f4ac",
)

registry.register(
    name="feishu_chat_list_members",
    toolset="feishu_chat",
    schema=FEISHU_CHAT_LIST_MEMBERS_SCHEMA,
    handler=_handle_chat_list_members,
    check_fn=_check_feishu,
    requires_env=[],
    is_async=False,
    description="List Feishu group chat members",
    emoji="\U0001f465",
)
