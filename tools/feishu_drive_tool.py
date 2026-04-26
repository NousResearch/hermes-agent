"""Feishu Drive Tools -- document comment operations via Feishu/Lark API.

Provides tools for listing, replying to, and adding document comments.
Uses the same lazy-import + BaseRequest pattern as feishu_comment.py.
The lark client is injected per-thread by the comment event handler.
"""

import json
import logging
import threading

from tools.feishu_oapi_client import (
    AppScopeMissingError,
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

TOOLS_METADATA["feishu_drive_list_comments"] = {
    "identity": "user",
    "scopes": ["drive:drive:readonly"],
}
TOOLS_METADATA["feishu_drive_list_comment_replies"] = {
    "identity": "user",
    "scopes": ["drive:drive:readonly"],
}
TOOLS_METADATA["feishu_drive_reply_comment"] = {
    "identity": "user",
    "scopes": ["drive:drive"],
}
TOOLS_METADATA["feishu_drive_add_comment"] = {
    "identity": "user",
    "scopes": ["drive:drive"],
}

# Thread-local storage for the lark client injected by feishu_comment handler.
_local = threading.local()


def set_client(client):
    """Store a lark client for the current thread (called by feishu_comment)."""
    _local.client = client


def get_client():
    """Return the lark client for the current thread, or None."""
    return getattr(_local, "client", None)


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


def _do_request(client, method, uri, paths=None, queries=None, body=None):
    """Build and execute a BaseRequest, return (code, msg, data_dict)."""
    from lark_oapi import AccessTokenType
    from lark_oapi.core.enum import HttpMethod
    from lark_oapi.core.model.base_request import BaseRequest

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

    # Tool handlers run synchronously in a worker thread (no running event
    # loop), so call the blocking lark client directly.
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


def _do_request_uat(fc, method, uri, paths=None, queries=None, body=None):
    """Build and execute a BaseRequest with UAT, return (code, msg, data_dict).

    Args:
        fc: FeishuClient instance with access_token set (from FeishuClient.for_user()).
        method: HTTP method string "GET" or "POST".
        uri: Feishu open-api URI.
        paths: Path parameter substitutions dict.
        queries: List of (key, value) query parameter tuples.
        body: JSON body dict (POST only).

    Returns:
        Tuple of (code, msg, data_dict).
    """
    from lark_oapi import AccessTokenType, RequestOption
    from lark_oapi.core.enum import HttpMethod
    from lark_oapi.core.model.base_request import BaseRequest

    http_method = HttpMethod.GET if method == "GET" else HttpMethod.POST

    builder = (
        BaseRequest.builder()
        .http_method(http_method)
        .uri(uri)
        .token_types({AccessTokenType.USER})
    )
    if paths:
        builder = builder.paths(paths)
    if queries:
        builder = builder.queries(queries)
    if body is not None:
        builder = builder.body(body)

    request = builder.build()
    opt = (
        RequestOption.builder()
        .user_access_token(fc.access_token)
        .build()
    )
    response = fc.sdk.request(request, opt)

    code = getattr(response, "code", None)
    msg = getattr(response, "msg", "")

    # Parse response data
    data = {}
    raw = getattr(response, "raw", None)
    if raw and hasattr(raw, "content"):
        try:
            body_json = json.loads(raw.content)
            data = body_json.get("data", {})
            if code is None:
                code = body_json.get("code", -1)
            if not msg:
                msg = body_json.get("msg", "")
        except (json.JSONDecodeError, AttributeError):
            pass
    if not data:
        resp_data = getattr(response, "data", None)
        if isinstance(resp_data, dict):
            data = resp_data
        elif resp_data and hasattr(resp_data, "__dict__"):
            data = vars(resp_data)

    return (code if code is not None else -1), msg, data


def _resolve_client_for_uat(use_uat: bool):
    """Return (fc_or_client, error_str) depending on use_uat flag.

    If use_uat=True, loads FeishuClient.for_user() and returns (fc, None).
    If use_uat=False, returns (thread_local_client, None) or (None, error_msg).
    """
    if use_uat:
        try:
            from tools.feishu_oapi_client import FeishuClient, NeedAuthorizationError
        except ImportError:
            return None, "feishu_oapi_client not available"
        try:
            fc = FeishuClient.for_user()
            return fc, None
        except NeedAuthorizationError as exc:
            return None, f"UAT not available: {exc}"
        except ValueError as exc:
            return None, f"Feishu client config error: {exc}"
    else:
        client = get_client()
        if client is None:
            return None, "Feishu client not available"
        return client, None


# ---------------------------------------------------------------------------
# feishu_drive_list_comments
# ---------------------------------------------------------------------------

_LIST_COMMENTS_URI = "/open-apis/drive/v1/files/:file_token/comments"

FEISHU_DRIVE_LIST_COMMENTS_SCHEMA = {
    "name": "feishu_drive_list_comments",
    "description": (
        "List comments on a Feishu document. "
        "Use is_whole=true to list whole-document comments only."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "file_token": {
                "type": "string",
                "description": "The document file token.",
            },
            "file_type": {
                "type": "string",
                "description": "File type (default: docx).",
                "default": "docx",
            },
            "is_whole": {
                "type": "boolean",
                "description": "If true, only return whole-document comments.",
                "default": False,
            },
            "page_size": {
                "type": "integer",
                "description": "Number of comments per page (max 100).",
                "default": 100,
            },
            "page_token": {
                "type": "string",
                "description": "Pagination token for next page.",
            },
            "use_uat": {
                "type": "boolean",
                "description": (
                    "If true, use user_access_token (UAT) identity instead of "
                    "tenant identity. Requires a valid UAT on disk."
                ),
                "default": False,
            },
        },
        "required": ["file_token"],
    },
}


def _handle_list_comments(args: dict, **kwargs) -> str:
    use_uat = bool(args.get("use_uat", False))
    file_token = args.get("file_token", "").strip()
    logger.info("feishu_drive_list_comments: file_token=%s use_uat=%s", file_token, use_uat)
    resolved, err = _resolve_client_for_uat(use_uat)
    if err:
        return tool_error(err)

    if not file_token:
        return tool_error("file_token is required")

    file_type = args.get("file_type", "docx") or "docx"
    is_whole = args.get("is_whole", False)
    page_size = args.get("page_size", 100)
    page_token = args.get("page_token", "")

    queries = [
        ("file_type", file_type),
        ("user_id_type", "open_id"),
        ("page_size", str(page_size)),
    ]
    if is_whole:
        queries.append(("is_whole", "true"))
    if page_token:
        queries.append(("page_token", page_token))

    if use_uat:
        code, msg, data = _do_request_uat(
            resolved, "GET", _LIST_COMMENTS_URI,
            paths={"file_token": file_token},
            queries=queries,
        )
    else:
        code, msg, data = _do_request(
            resolved, "GET", _LIST_COMMENTS_URI,
            paths={"file_token": file_token},
            queries=queries,
        )
    if code != 0:
        try:
            raise_for_feishu_errcode(code, msg or "", api_name="feishu.drive.list_comments")
        except (AppScopeMissingError, UserAuthRequiredError, UserScopeInsufficientError, NeedAuthorizationError) as e:
            return tool_error(_auth_error_message(e))
        logger.warning("feishu_drive_list_comments failed: code=%d msg=%s", code, msg)
        return tool_error(f"List comments failed: code={code} msg={msg}")

    return tool_result(data)


# ---------------------------------------------------------------------------
# feishu_drive_list_comment_replies
# ---------------------------------------------------------------------------

_LIST_REPLIES_URI = "/open-apis/drive/v1/files/:file_token/comments/:comment_id/replies"

FEISHU_DRIVE_LIST_REPLIES_SCHEMA = {
    "name": "feishu_drive_list_comment_replies",
    "description": "List all replies in a comment thread on a Feishu document.",
    "parameters": {
        "type": "object",
        "properties": {
            "file_token": {
                "type": "string",
                "description": "The document file token.",
            },
            "comment_id": {
                "type": "string",
                "description": "The comment ID to list replies for.",
            },
            "file_type": {
                "type": "string",
                "description": "File type (default: docx).",
                "default": "docx",
            },
            "page_size": {
                "type": "integer",
                "description": "Number of replies per page (max 100).",
                "default": 100,
            },
            "page_token": {
                "type": "string",
                "description": "Pagination token for next page.",
            },
            "use_uat": {
                "type": "boolean",
                "description": (
                    "If true, use user_access_token (UAT) identity instead of "
                    "tenant identity. Requires a valid UAT on disk."
                ),
                "default": False,
            },
        },
        "required": ["file_token", "comment_id"],
    },
}


def _handle_list_replies(args: dict, **kwargs) -> str:
    use_uat = bool(args.get("use_uat", False))
    file_token = args.get("file_token", "").strip()
    comment_id = args.get("comment_id", "").strip()
    logger.info("feishu_drive_list_comment_replies: file_token=%s comment_id=%s use_uat=%s", file_token, comment_id, use_uat)
    resolved, err = _resolve_client_for_uat(use_uat)
    if err:
        return tool_error(err)

    if not file_token or not comment_id:
        return tool_error("file_token and comment_id are required")

    file_type = args.get("file_type", "docx") or "docx"
    page_size = args.get("page_size", 100)
    page_token = args.get("page_token", "")

    queries = [
        ("file_type", file_type),
        ("user_id_type", "open_id"),
        ("page_size", str(page_size)),
    ]
    if page_token:
        queries.append(("page_token", page_token))

    if use_uat:
        code, msg, data = _do_request_uat(
            resolved, "GET", _LIST_REPLIES_URI,
            paths={"file_token": file_token, "comment_id": comment_id},
            queries=queries,
        )
    else:
        code, msg, data = _do_request(
            resolved, "GET", _LIST_REPLIES_URI,
            paths={"file_token": file_token, "comment_id": comment_id},
            queries=queries,
        )
    if code != 0:
        try:
            raise_for_feishu_errcode(code, msg or "", api_name="feishu.drive.list_comment_replies")
        except (AppScopeMissingError, UserAuthRequiredError, UserScopeInsufficientError, NeedAuthorizationError) as e:
            return tool_error(_auth_error_message(e))
        logger.warning("feishu_drive_list_comment_replies failed: code=%d msg=%s", code, msg)
        return tool_error(f"List replies failed: code={code} msg={msg}")

    return tool_result(data)


# ---------------------------------------------------------------------------
# feishu_drive_reply_comment
# ---------------------------------------------------------------------------

_REPLY_COMMENT_URI = "/open-apis/drive/v1/files/:file_token/comments/:comment_id/replies"

FEISHU_DRIVE_REPLY_SCHEMA = {
    "name": "feishu_drive_reply_comment",
    "description": (
        "Reply to a local comment thread on a Feishu document. "
        "Use this for local (quoted-text) comments. "
        "For whole-document comments, use feishu_drive_add_comment instead."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "file_token": {
                "type": "string",
                "description": "The document file token.",
            },
            "comment_id": {
                "type": "string",
                "description": "The comment ID to reply to.",
            },
            "content": {
                "type": "string",
                "description": "The reply text content (plain text only, no markdown).",
            },
            "file_type": {
                "type": "string",
                "description": "File type (default: docx).",
                "default": "docx",
            },
            "use_uat": {
                "type": "boolean",
                "description": (
                    "If true, use user_access_token (UAT) identity instead of "
                    "tenant identity. Requires a valid UAT on disk."
                ),
                "default": False,
            },
        },
        "required": ["file_token", "comment_id", "content"],
    },
}


def _handle_reply_comment(args: dict, **kwargs) -> str:
    use_uat = bool(args.get("use_uat", False))
    file_token = args.get("file_token", "").strip()
    comment_id = args.get("comment_id", "").strip()
    logger.info("feishu_drive_reply_comment: file_token=%s comment_id=%s use_uat=%s", file_token, comment_id, use_uat)
    resolved, err = _resolve_client_for_uat(use_uat)
    if err:
        return tool_error(err)

    content = args.get("content", "").strip()
    if not file_token or not comment_id or not content:
        return tool_error("file_token, comment_id, and content are required")

    file_type = args.get("file_type", "docx") or "docx"

    body = {
        "content": {
            "elements": [
                {
                    "type": "text_run",
                    "text_run": {"text": content},
                }
            ]
        }
    }

    if use_uat:
        code, msg, data = _do_request_uat(
            resolved, "POST", _REPLY_COMMENT_URI,
            paths={"file_token": file_token, "comment_id": comment_id},
            queries=[("file_type", file_type)],
            body=body,
        )
    else:
        code, msg, data = _do_request(
            resolved, "POST", _REPLY_COMMENT_URI,
            paths={"file_token": file_token, "comment_id": comment_id},
            queries=[("file_type", file_type)],
            body=body,
        )
    if code != 0:
        try:
            raise_for_feishu_errcode(code, msg or "", api_name="feishu.drive.reply_comment")
        except (AppScopeMissingError, UserAuthRequiredError, UserScopeInsufficientError, NeedAuthorizationError) as e:
            return tool_error(_auth_error_message(e))
        logger.warning("feishu_drive_reply_comment failed: code=%d msg=%s", code, msg)
        return tool_error(f"Reply comment failed: code={code} msg={msg}")

    return tool_result(success=True, data=data)


# ---------------------------------------------------------------------------
# feishu_drive_add_comment
# ---------------------------------------------------------------------------

_ADD_COMMENT_URI = "/open-apis/drive/v1/files/:file_token/new_comments"

FEISHU_DRIVE_ADD_COMMENT_SCHEMA = {
    "name": "feishu_drive_add_comment",
    "description": (
        "Add a new whole-document comment on a Feishu document. "
        "Use this for whole-document comments or as a fallback when "
        "reply_comment fails with code 1069302."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "file_token": {
                "type": "string",
                "description": "The document file token.",
            },
            "content": {
                "type": "string",
                "description": "The comment text content (plain text only, no markdown).",
            },
            "file_type": {
                "type": "string",
                "description": "File type (default: docx).",
                "default": "docx",
            },
            "use_uat": {
                "type": "boolean",
                "description": (
                    "If true, use user_access_token (UAT) identity instead of "
                    "tenant identity. Requires a valid UAT on disk."
                ),
                "default": False,
            },
        },
        "required": ["file_token", "content"],
    },
}


def _handle_add_comment(args: dict, **kwargs) -> str:
    use_uat = bool(args.get("use_uat", False))
    file_token = args.get("file_token", "").strip()
    logger.info("feishu_drive_add_comment: file_token=%s use_uat=%s", file_token, use_uat)
    resolved, err = _resolve_client_for_uat(use_uat)
    if err:
        return tool_error(err)

    content = args.get("content", "").strip()
    if not file_token or not content:
        return tool_error("file_token and content are required")

    file_type = args.get("file_type", "docx") or "docx"

    body = {
        "file_type": file_type,
        "reply_elements": [
            {"type": "text", "text": content},
        ],
    }

    if use_uat:
        code, msg, data = _do_request_uat(
            resolved, "POST", _ADD_COMMENT_URI,
            paths={"file_token": file_token},
            body=body,
        )
    else:
        code, msg, data = _do_request(
            resolved, "POST", _ADD_COMMENT_URI,
            paths={"file_token": file_token},
            body=body,
        )
    if code != 0:
        try:
            raise_for_feishu_errcode(code, msg or "", api_name="feishu.drive.add_comment")
        except (AppScopeMissingError, UserAuthRequiredError, UserScopeInsufficientError, NeedAuthorizationError) as e:
            return tool_error(_auth_error_message(e))
        logger.warning("feishu_drive_add_comment failed: code=%d msg=%s", code, msg)
        return tool_error(f"Add comment failed: code={code} msg={msg}")

    return tool_result(success=True, data=data)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

registry.register(
    name="feishu_drive_list_comments",
    toolset="feishu_drive",
    schema=FEISHU_DRIVE_LIST_COMMENTS_SCHEMA,
    handler=_handle_list_comments,
    check_fn=_check_feishu,
    requires_env=[],
    is_async=False,
    description="List document comments",
    emoji="\U0001f4ac",
)

registry.register(
    name="feishu_drive_list_comment_replies",
    toolset="feishu_drive",
    schema=FEISHU_DRIVE_LIST_REPLIES_SCHEMA,
    handler=_handle_list_replies,
    check_fn=_check_feishu,
    requires_env=[],
    is_async=False,
    description="List comment replies",
    emoji="\U0001f4ac",
)

registry.register(
    name="feishu_drive_reply_comment",
    toolset="feishu_drive",
    schema=FEISHU_DRIVE_REPLY_SCHEMA,
    handler=_handle_reply_comment,
    check_fn=_check_feishu,
    requires_env=[],
    is_async=False,
    description="Reply to a document comment",
    emoji="\u2709\ufe0f",
)

registry.register(
    name="feishu_drive_add_comment",
    toolset="feishu_drive",
    schema=FEISHU_DRIVE_ADD_COMMENT_SCHEMA,
    handler=_handle_add_comment,
    check_fn=_check_feishu,
    requires_env=[],
    is_async=False,
    description="Add a whole-document comment",
    emoji="\u2709\ufe0f",
)
