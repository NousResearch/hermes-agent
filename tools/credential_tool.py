"""Opaque credential request/management tool.

This tool intentionally never accepts or returns plaintext credential values.
Users enter/update values through `hermes credentials set/update`, which uses a
masked terminal prompt outside model chat.
"""

from __future__ import annotations

import json
from typing import Any, Dict

from agent.credential_store import (
    CredentialStoreError,
    delete_credential,
    list_credentials,
    request_credential,
    revoke_credential,
)
from tools.registry import registry, tool_error


SCHEMA = {
    "name": "credential",
    "description": (
        "Request and manage opaque credential references without seeing secret "
        "values. Use operation=request when a password/token/API key is needed; "
        "never ask the user to paste secrets in chat. Values are entered via a "
        "separate masked `hermes credentials set` UI. Returns only credential refs."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "operation": {
                "type": "string",
                "enum": ["request", "list", "revoke", "delete", "status"],
                "description": "request creates/returns a pending opaque ref; revoke/delete/status operate by credential_ref.",
            },
            "name": {
                "type": "string",
                "description": "Human-readable credential name for request, e.g. github-pat or client-api-token.",
            },
            "credential_type": {
                "type": "string",
                "description": "Short type identifier such as api_key, token, password, oauth_refresh_token. Defaults to secret.",
            },
            "credential_ref": {
                "type": "string",
                "description": "Opaque credential reference returned by request/list. Never contains plaintext.",
            },
        },
        "required": ["operation"],
        "additionalProperties": False,
    },
}


def _handle(args: Dict[str, Any], **_: Any) -> str:
    op = str(args.get("operation") or "").strip().lower()
    try:
        if op == "request":
            record = request_credential(
                str(args.get("name") or ""),
                str(args.get("credential_type") or "secret"),
            )
            return json.dumps({"success": True, "credential": record}, ensure_ascii=False)
        if op == "list":
            return json.dumps({"success": True, "credentials": list_credentials()}, ensure_ascii=False)
        if op == "status":
            ref = str(args.get("credential_ref") or "")
            for record in list_credentials():
                if record.get("ref") == ref:
                    return json.dumps({"success": True, "credential": record}, ensure_ascii=False)
            return tool_error("credential reference not found")
        if op == "revoke":
            record = revoke_credential(str(args.get("credential_ref") or ""))
            return json.dumps({"success": True, "credential": record}, ensure_ascii=False)
        if op == "delete":
            record = delete_credential(str(args.get("credential_ref") or ""))
            return json.dumps({"success": True, "credential": record}, ensure_ascii=False)
        return tool_error("operation must be one of request, list, status, revoke, delete")
    except CredentialStoreError as exc:
        return tool_error(str(exc))


registry.register(
    name="credential",
    toolset="credentials",
    schema=SCHEMA,
    handler=_handle,
    requires_env=[],
    emoji="🔐",
)
