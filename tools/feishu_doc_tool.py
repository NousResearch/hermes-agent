"""Feishu Document Tool -- read document content via Feishu/Lark API.

Provides ``feishu_doc_read`` for reading document content as plain text.
Uses the same lazy-import + BaseRequest pattern as feishu_comment.py.
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

TOOLS_METADATA["feishu_doc_read"] = {
    "identity": "user",
    "scopes": ["docx:document:readonly"],
}

# Thread-local storage for the lark client injected by feishu_comment handler.
_local = threading.local()


def set_client(client):
    """Store a lark client for the current thread (called by feishu_comment)."""
    _local.client = client


def get_client():
    """Return the lark client for the current thread, or None."""
    return getattr(_local, "client", None)


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
# feishu_doc_read
# ---------------------------------------------------------------------------

_RAW_CONTENT_URI = "/open-apis/docx/v1/documents/:document_id/raw_content"

FEISHU_DOC_READ_SCHEMA = {
    "name": "feishu_doc_read",
    "description": (
        "Read the full content of a Feishu/Lark document as plain text. "
        "Useful when you need more context beyond the quoted text in a comment."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "doc_token": {
                "type": "string",
                "description": "The document token (from the document URL or comment context).",
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
        "required": ["doc_token"],
    },
}


def _check_feishu():
    try:
        import lark_oapi  # noqa: F401
        return True
    except ImportError:
        return False


def _handle_feishu_doc_read(args: dict, **kwargs) -> str:
    doc_token = args.get("doc_token", "").strip()
    if not doc_token:
        return tool_error("doc_token is required")

    use_uat = bool(args.get("use_uat", False))
    logger.info("feishu_doc_read: doc_token=%s use_uat=%s", doc_token, use_uat)

    try:
        from lark_oapi import AccessTokenType, RequestOption
        from lark_oapi.core.enum import HttpMethod
        from lark_oapi.core.model.base_request import BaseRequest
    except ImportError:
        return tool_error("lark_oapi not installed")

    if use_uat:
        # UAT path: build client from disk token
        try:
            from tools.feishu_oapi_client import FeishuClient
        except ImportError:
            return tool_error("feishu_oapi_client not available")
        try:
            fc = FeishuClient.for_user()
        except NeedAuthorizationError as exc:
            return tool_error(_auth_error_message(exc))
        except ValueError as exc:
            return tool_error(f"Feishu client config error: {exc}")

        request = (
            BaseRequest.builder()
            .http_method(HttpMethod.GET)
            .uri(_RAW_CONTENT_URI)
            .token_types({AccessTokenType.USER})
            .paths({"document_id": doc_token})
            .build()
        )
        opt = (
            RequestOption.builder()
            .user_access_token(fc.access_token)
            .build()
        )
        response = fc.sdk.request(request, opt)
    else:
        # Tenant path: use thread-local client (original behaviour)
        client = get_client()
        if client is None:
            return tool_error("Feishu client not available (not in a Feishu comment context)")

        request = (
            BaseRequest.builder()
            .http_method(HttpMethod.GET)
            .uri(_RAW_CONTENT_URI)
            .token_types({AccessTokenType.TENANT})
            .paths({"document_id": doc_token})
            .build()
        )
        # Tool handlers run synchronously in a worker thread (no running event
        # loop), so call the blocking lark client directly.
        response = client.request(request)

    code = getattr(response, "code", None)
    msg = getattr(response, "msg", "unknown error")

    if code != 0:
        try:
            raise_for_feishu_errcode(code, msg or "", api_name="feishu.docx.raw_content")
        except (AppScopeMissingError, UserAuthRequiredError, UserScopeInsufficientError, NeedAuthorizationError) as e:
            return tool_error(_auth_error_message(e))
        logger.warning("feishu_doc_read failed: code=%s msg=%s", code, msg)
        return tool_error(f"Failed to read document: code={code} msg={msg}")

    raw = getattr(response, "raw", None)
    if raw and hasattr(raw, "content"):
        try:
            body = json.loads(raw.content)
            content = body.get("data", {}).get("content", "")
            logger.info("feishu_doc_read: read %d chars from doc_token=%s", len(content), doc_token)
            return tool_result(success=True, content=content)
        except (json.JSONDecodeError, AttributeError):
            pass

    # Fallback: try response.data
    data = getattr(response, "data", None)
    if data:
        if isinstance(data, dict):
            content = data.get("content", "")
        else:
            content = getattr(data, "content", str(data))
        logger.info("feishu_doc_read: read %d chars (fallback) from doc_token=%s", len(content), doc_token)
        return tool_result(success=True, content=content)

    return tool_error("No content returned from document API")


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

registry.register(
    name="feishu_doc_read",
    toolset="feishu_doc",
    schema=FEISHU_DOC_READ_SCHEMA,
    handler=_handle_feishu_doc_read,
    check_fn=_check_feishu,
    requires_env=[],
    is_async=False,
    description="Read Feishu document content",
    emoji="\U0001f4c4",
)
