"""Feishu IM User Tool -- send and reply to IM messages as the user via UAT.

Provides two tools:
  ``feishu_im_send_message_as_user``  -- POST /im/v1/messages?receive_id_type=...
  ``feishu_im_reply_message_as_user`` -- POST /im/v1/messages/:message_id/reply

Both use user_access_token (UAT) so messages are sent under the user's identity,
not the bot's. The FeishuClient.for_user() factory loads the UAT from disk.

Required scope: im:message:send_as_user
"""

import json
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

TOOLS_METADATA["feishu_im_send_message_as_user"] = {
    "identity": "user",
    "scopes": ["im:message:send_as_user"],
}

TOOLS_METADATA["feishu_im_reply_message_as_user"] = {
    "identity": "user",
    "scopes": ["im:message:send_as_user"],
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


def _do_user_request(method, uri, *, paths=None, queries=None, body=None):
    """Build a UAT BaseRequest via FeishuClient.for_user() and execute it.

    Args:
        method: HTTP method string "GET" or "POST".
        uri: Feishu open-api URI.
        paths: Path parameter substitutions dict.
        queries: List of (key, value) query parameter tuples.
        body: JSON body dict (POST only).

    Returns:
        Tuple of (code, msg, data_dict).

    Raises:
        NeedAuthorizationError: If UAT is missing or expired.
        RuntimeError: If lark_oapi is not installed.
    """
    client = FeishuClient.for_user()
    return client.do_request(
        method,
        uri,
        paths=paths,
        queries=queries,
        body=body,
        use_uat=True,
    )


# ---------------------------------------------------------------------------
# feishu_im_send_message_as_user
# ---------------------------------------------------------------------------

_SEND_MESSAGE_URI = "/open-apis/im/v1/messages"

FEISHU_IM_SEND_MESSAGE_AS_USER_SCHEMA = {
    "name": "feishu_im_send_message_as_user",
    "description": (
        "Send an IM message as the user (not the bot) via user_access_token. "
        "Supports sending to a user, group chat, or open_id. "
        "msg_type can be 'text' or 'post'. For 'text', content should be "
        "a JSON string like '{\"text\": \"hello\"}'. "
        "Requires scope: im:message:send_as_user."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "receive_id_type": {
                "type": "string",
                "description": (
                    "Type of receive_id. One of: open_id, user_id, union_id, "
                    "email, chat_id."
                ),
            },
            "receive_id": {
                "type": "string",
                "description": "The recipient's ID (open_id, chat_id, email, etc.).",
            },
            "msg_type": {
                "type": "string",
                "description": (
                    "Message type. Common values: 'text', 'post', 'image', "
                    "'interactive'. Typically 'text' for plain messages."
                ),
            },
            "content": {
                "type": "string",
                "description": (
                    "Message content as a JSON string. For msg_type='text': "
                    "'{\"text\": \"your message\"}'. "
                    "For msg_type='post': a Feishu post rich-text JSON string."
                ),
            },
        },
        "required": ["receive_id_type", "receive_id", "msg_type", "content"],
    },
}


def _handle_send_message_as_user(args: dict, **kwargs) -> str:
    """Handler for feishu_im_send_message_as_user tool.

    Args:
        args: Tool arguments from user/model.
        **kwargs: Additional keyword arguments.

    Returns:
        JSON string (tool_error or tool_result).
    """
    receive_id_type = args.get("receive_id_type", "").strip()
    receive_id = args.get("receive_id", "").strip()
    msg_type = args.get("msg_type", "").strip()
    content = args.get("content", "").strip()

    logger.info(
        "feishu_im_send_message_as_user: receive_id_type=%s receive_id=%s msg_type=%s",
        receive_id_type, receive_id, msg_type,
    )

    if not receive_id_type:
        return tool_error("receive_id_type is required")
    if not receive_id:
        return tool_error("receive_id is required")
    if not msg_type:
        return tool_error("msg_type is required")
    if not content:
        return tool_error("content is required")

    # Validate content is valid JSON
    try:
        json.loads(content)
    except json.JSONDecodeError as exc:
        return tool_error(f"content must be a valid JSON string: {exc}")

    body = {
        "receive_id": receive_id,
        "msg_type": msg_type,
        "content": content,
    }

    try:
        code, msg, data = _do_user_request(
            "POST",
            _SEND_MESSAGE_URI,
            queries=[("receive_id_type", receive_id_type)],
            body=body,
        )
    except NeedAuthorizationError as exc:
        return tool_error(_auth_error_message(exc))
    except RuntimeError as exc:
        return tool_error(str(exc))

    if code != 0:
        try:
            raise_for_feishu_errcode(code, msg or "", api_name="feishu.im.send_message_as_user")
        except (AppScopeMissingError, UserAuthRequiredError, UserScopeInsufficientError, NeedAuthorizationError) as e:
            return tool_error(_auth_error_message(e))
        logger.warning("feishu_im_send_message_as_user failed: code=%d msg=%s", code, msg)
        return tool_error(f"Send message failed: code={code} msg={msg}")

    logger.info("feishu_im_send_message_as_user: sent message to %s", receive_id)
    return tool_result(success=True, data=data)


# ---------------------------------------------------------------------------
# feishu_im_reply_message_as_user
# ---------------------------------------------------------------------------

_REPLY_MESSAGE_URI = "/open-apis/im/v1/messages/:message_id/reply"

FEISHU_IM_REPLY_MESSAGE_AS_USER_SCHEMA = {
    "name": "feishu_im_reply_message_as_user",
    "description": (
        "Reply to an existing IM message as the user (not the bot) via "
        "user_access_token. The reply appears under the original message thread. "
        "msg_type can be 'text' or 'post'. For 'text', content should be "
        "a JSON string like '{\"text\": \"hello\"}'. "
        "Requires scope: im:message:send_as_user."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "message_id": {
                "type": "string",
                "description": "The message ID to reply to (e.g. om_xxxxxxxx).",
            },
            "msg_type": {
                "type": "string",
                "description": (
                    "Message type. Common values: 'text', 'post', 'image', "
                    "'interactive'. Typically 'text' for plain replies."
                ),
            },
            "content": {
                "type": "string",
                "description": (
                    "Reply content as a JSON string. For msg_type='text': "
                    "'{\"text\": \"your reply\"}'. "
                    "For msg_type='post': a Feishu post rich-text JSON string."
                ),
            },
        },
        "required": ["message_id", "msg_type", "content"],
    },
}


def _handle_reply_message_as_user(args: dict, **kwargs) -> str:
    """Handler for feishu_im_reply_message_as_user tool.

    Args:
        args: Tool arguments from user/model.
        **kwargs: Additional keyword arguments.

    Returns:
        JSON string (tool_error or tool_result).
    """
    message_id = args.get("message_id", "").strip()
    msg_type = args.get("msg_type", "").strip()
    content = args.get("content", "").strip()

    logger.info(
        "feishu_im_reply_message_as_user: message_id=%s msg_type=%s",
        message_id, msg_type,
    )

    if not message_id:
        return tool_error("message_id is required")
    if not msg_type:
        return tool_error("msg_type is required")
    if not content:
        return tool_error("content is required")

    # Validate content is valid JSON
    try:
        json.loads(content)
    except json.JSONDecodeError as exc:
        return tool_error(f"content must be a valid JSON string: {exc}")

    body = {
        "msg_type": msg_type,
        "content": content,
    }

    try:
        code, msg, data = _do_user_request(
            "POST",
            _REPLY_MESSAGE_URI,
            paths={"message_id": message_id},
            body=body,
        )
    except NeedAuthorizationError as exc:
        return tool_error(_auth_error_message(exc))
    except RuntimeError as exc:
        return tool_error(str(exc))

    if code != 0:
        try:
            raise_for_feishu_errcode(code, msg or "", api_name="feishu.im.reply_message_as_user")
        except (AppScopeMissingError, UserAuthRequiredError, UserScopeInsufficientError, NeedAuthorizationError) as e:
            return tool_error(_auth_error_message(e))
        logger.warning("feishu_im_reply_message_as_user failed: code=%d msg=%s", code, msg)
        return tool_error(f"Reply message failed: code={code} msg={msg}")

    logger.info("feishu_im_reply_message_as_user: replied to message %s", message_id)
    return tool_result(success=True, data=data)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

registry.register(
    name="feishu_im_send_message_as_user",
    toolset="feishu_im_user",
    schema=FEISHU_IM_SEND_MESSAGE_AS_USER_SCHEMA,
    handler=_handle_send_message_as_user,
    check_fn=_check_feishu,
    requires_env=[],
    is_async=False,
    description="Send IM message as user (UAT)",
    emoji="\U0001f4e8",
)

registry.register(
    name="feishu_im_reply_message_as_user",
    toolset="feishu_im_user",
    schema=FEISHU_IM_REPLY_MESSAGE_AS_USER_SCHEMA,
    handler=_handle_reply_message_as_user,
    check_fn=_check_feishu,
    requires_env=[],
    is_async=False,
    description="Reply to IM message as user (UAT)",
    emoji="\U0001f4e8",
)
