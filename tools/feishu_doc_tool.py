"""Feishu Document Tool -- read and create documents via Feishu/Lark API.

Provides ``feishu_doc_read`` for reading document content as plain text,
and supports "create" action to create new documents.
Uses the same lazy-import + BaseRequest pattern as feishu_comment.py.
"""

import json
import logging
import threading

from tools.registry import registry, tool_error, tool_result

logger = logging.getLogger(__name__)

# Thread-local storage for the lark client injected by feishu_comment handler.
_local = threading.local()


def set_client(client):
    """Store a lark client for the current thread (called by feishu_comment)."""
    _local.client = client


def get_client():
    """Return the lark client for the current thread, or None."""
    return getattr(_local, "client", None)


# ---------------------------------------------------------------------------
# feishu_doc_read
# ---------------------------------------------------------------------------

_RAW_CONTENT_URI = "/open-apis/docx/v1/documents/:document_id/raw_content"
_CREATE_DOCUMENT_URI = "/open-apis/docx/v1/documents"

FEISHU_DOC_READ_SCHEMA = {
    "name": "feishu_doc_read",
    "description": (
        "Read the full content of a Feishu/Lark document as plain text, "
        "or create a new Feishu/Lark document. "
        "Use action='read' (default) to read an existing document by doc_token. "
        "Use action='create' to create a new document with a title and optional owner_open_id."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["read", "create"],
                "description": "Action to perform: 'read' (default) or 'create'.",
            },
            "doc_token": {
                "type": "string",
                "description": "The document token (from the document URL or comment context). Required for action='read'.",
            },
            "title": {
                "type": "string",
                "description": "Title for the new document. Required for action='create'.",
            },
            "owner_open_id": {
                "type": "string",
                "description": "The open_id of the document owner. Optional for action='create'.",
            },
        },
        "required": ["action"],
    },
}


def _check_feishu():
    # Use ``importlib.util.find_spec`` — it checks whether ``lark_oapi``
    # is importable without actually executing its ``__init__``.
    # Executing the real import here costs ~5 seconds (the SDK eagerly
    # loads websockets, dispatcher, every api/v2 model) and this probe
    # fires at every ``hermes`` startup during tool-availability
    # evaluation.  Correctness is preserved because the actual tool
    # handler still does the real import when invoked.
    import importlib.util
    try:
        return importlib.util.find_spec("lark_oapi") is not None
    except (ImportError, ValueError):
        return False


def _handle_feishu_doc_read(args: dict, **kwargs) -> str:
    action = args.get("action", "read").strip().lower()
    doc_token = args.get("doc_token", "").strip()
    title = args.get("title", "").strip()
    owner_open_id = args.get("owner_open_id", "").strip()

    client = get_client()
    if client is None:
        return tool_error("Feishu client not available (not in a Feishu comment context)")

    try:
        from lark_oapi import AccessTokenType
        from lark_oapi.core.enum import HttpMethod
        from lark_oapi.core.model.base_request import BaseRequest
    except ImportError:
        return tool_error("lark_oapi not installed")

    if action == "create":
        if not title:
            return tool_error("title is required for action='create'")
        try:
            from lark_oapi.api.docx.v1.model.create_document_request_body import (
                CreateDocumentRequestBody,
            )
        except ImportError:
            return tool_error("lark_oapi not installed")

        body = (
            CreateDocumentRequestBody.builder()
            .title(title)
            .build()
        )
        request = (
            BaseRequest.builder()
            .http_method(HttpMethod.POST)
            .uri(_CREATE_DOCUMENT_URI)
            .token_types({AccessTokenType.TENANT})
            .request_body(body)
            .build()
        )
        response = client.request(request)
        code = getattr(response, "code", None)
        if code != 0:
            msg = getattr(response, "msg", "unknown error")
            return tool_error(f"Failed to create document: code={code} msg={msg}")
        data = getattr(response, "data", None)
        if data and isinstance(data, dict):
            doc = data.get("document", {})
            doc_id = doc.get("document_id", "unknown")
            return tool_result(
                success=True,
                content=f"Document created successfully. doc_token={doc_id}, title={title}",
            )
        return tool_result(success=True, content="Document created successfully.")

    if action == "read":
        if not doc_token:
            return tool_error("doc_token is required for action='read'")

        request = (
            BaseRequest.builder()
            .http_method(HttpMethod.GET)
            .uri(_RAW_CONTENT_URI)
            .token_types({AccessTokenType.TENANT})
            .paths({"document_id": doc_token})
            .build()
        )

        response = client.request(request)

        code = getattr(response, "code", None)
        if code != 0:
            msg = getattr(response, "msg", "unknown error")
            return tool_error(f"Failed to read document: code={code} msg={msg}")

        raw = getattr(response, "raw", None)
        if raw and hasattr(raw, "content"):
            try:
                body = json.loads(raw.content)
                content = body.get("data", {}).get("content", "")
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
            return tool_result(success=True, content=content)

        return tool_error("No content returned from document API")

    return tool_error(f"Unknown action: {action}. Use 'read' or 'create'.")


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
