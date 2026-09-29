"""Credential Tool — agent-facing interface to the encrypted credential store.

The agent may use this tool to request, list, or delete credentials. The tool
NEVER returns plaintext values -- the model gets opaque refs and metadata,
and resolves values only inside trusted execution code (execute_code).

Actions:
  request  -- Ask the user to provide a credential value via masked prompt.
             Returns {"stored": true, "name": "..."}. Value never enters context.
  list     -- Show credential names and metadata (never values).
  status   -- Check whether named credentials exist.
  delete   -- Remove a credential by name.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from hermes_constants import get_hermes_home

logger = logging.getLogger(__name__)

# Schema for the credential tool
CREDENTIAL_SCHEMA = {
    "name": "credentials",
    "description": (
        "Manage credentials (API keys, tokens, passwords) that the agent needs "
        "to do its work. The agent may REQUEST a new credential from the user, "
        "LIST or check existence of stored credentials, or DELETE them. "
        "Values are encrypted at rest and NEVER returned to the agent -- they "
        "are resolved only inside trusted execution code (execute_code) where "
        "the variable is set before running your script. "
        "Use this for task-specific secrets like Supabase service keys, Stripe "
        "secret keys, database passwords, or third-party API tokens that your "
        "project needs.\n\n"
        "REQUEST: When you need a credential the user hasn't provided yet, call "
        "credentials(action='request', name='...', description='...'). This "
        "triggers a masked input prompt for the user. The value is stored "
        "encrypted and never enters your context.\n\n"
        "LIST: Call credentials(action='list') to see stored credential names. "
        "Use this before requesting to avoid duplicates, or to remind yourself "
        "what's available.\n\n"
        "STATUS: Call credentials(action='status', names=['...']) to check if "
        "specific credentials exist before trying to use them.\n\n"
        "DELETE: Call credentials(action='delete', name='...') to remove a "
        "credential that is no longer needed.\n\n"
        "USE inside execute_code: Once stored, reference a credential by setting "
        "its name as a Python variable before your script runs. "
        "The value is injected into the execution context -- do not log or print it."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["request", "list", "status", "delete"],
                "description": "What to do with credentials.",
            },
            "name": {
                "type": "string",
                "description": (
                    "Credential name (e.g. 'supabase_service_key', "
                    "'stripe_secret_key'). Required for request, status, and delete. "
                    "Must be descriptive and at least 3 characters. "
                    "For request: this becomes the lookup key."
                ),
            },
            "names": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of credential names for status check.",
            },
            "description": {
                "type": "string",
                "description": (
                    "Human-readable description of what this credential is for "
                    "(e.g. 'Supabase service role key for database admin'). "
                    "Shown to the user in the request prompt. Only used with "
                    "action='request'."
                ),
            },
            "instructions": {
                "type": "string",
                "description": (
                    "Optional instructions on where to find this credential "
                    "(e.g. 'Find this at https://supabase.com/dashboard/project/xxx/settings/api'). "
                    "Shown to the user in the request prompt. Only used with "
                    "action='request'."
                ),
            },
            "overwrite": {
                "type": "boolean",
                "description": (
                    "When true and the credential already exists, replaces it. "
                    "When false (default), a duplicate name returns an error. "
                    "Only used with action='request'."
                ),
            },
        },
        "required": ["action"],
    },
}


def check_credential_requirements() -> bool:
    """Always available -- no external requirements."""
    return True


# -- The credential store is lazy-imported inside handlers so this module can
# -- be imported at tool-discovery time before HERMES_HOME is resolved. The
# -- store resolves get_hermes_home() at first use, which is after the agent
# -- loop is running and HERMES_HOME is stable.

def _store():
    from agent.credential_store import get_store
    return get_store()


def _request_credential(
    name: str,
    description: Optional[str] = None,
    instructions: Optional[str] = None,
    overwrite: bool = False,
) -> str:
    """Request a credential from the user via masked prompt."""
    from agent.credential_store import CredentialStore

    # Validate name early
    store = _store()
    try:
        store._validate_name(name)
    except ValueError as exc:
        return json.dumps({"success": False, "error": str(exc)}, ensure_ascii=False)

    # Check if already exists
    existing = store.get(name)
    if existing is not None and not overwrite:
        return json.dumps({
            "success": True,
            "already_exists": True,
            "name": name,
            "message": f"Credential '{name}' is already stored. "
                       f"Use overwrite=true if you need to replace it.",
        }, ensure_ascii=False)

    # Prompt the user with a masked input
    try:
        from hermes_cli.secret_prompt import masked_secret_prompt
        prompt = f"Enter value for credential '{name}'"
        if description:
            prompt = f"{description} ({name})"
        if instructions:
            prompt += f"\n  Where to find it: {instructions}"
        prompt += "\n  Value (hidden): "
        value = masked_secret_prompt(prompt)
    except (ImportError, Exception):
        # Fallback: plain getpass
        import getpass
        prompt = f"Enter value for '{name}' (hidden): "
        try:
            value = getpass.getpass(prompt)
        except (EOFError, KeyboardInterrupt):
            value = ""

    if not value:
        return json.dumps(
            {"success": False, "error": "User cancelled or provided an empty value."},
            ensure_ascii=False,
        )

    result = store.set(name, value.strip(), overwrite=overwrite)
    return json.dumps({
        "success": result["success"],
        "stored": True,
        "name": name,
        "message": f"Credential '{name}' has been stored. "
                   f"Reference it in execute_code by setting it as a variable.",
    }, ensure_ascii=False)


def _list_credentials() -> str:
    """Return metadata for all stored credentials (names only, never values)."""
    store = _store()
    entries = store.list()

    # Register each credential's value with the redactor so it's scrubbed
    # from any output path even if a caller accidentally logs or echoes it.
    try:
        from agent.redact import register_secret
        for entry in entries:
            value = store.get(entry["name"])
            if value:
                register_secret(value)
    except Exception:
        pass

    if not entries:
        return json.dumps(
            {"success": True, "entries": [], "message": "No credentials stored."},
            ensure_ascii=False,
        )

    return json.dumps(
        {"success": True, "entries": entries, "count": len(entries)},
        ensure_ascii=False,
    )


def _status_credentials(names: List[str]) -> str:
    """Check whether named credentials exist. Returns names only, never values."""
    store = _store()
    configured = []
    missing = []
    for name in names:
        value = store.get(name)
        if value is not None:
            configured.append(name)
            # Register with redactor
            try:
                from agent.redact import register_secret
                register_secret(value)
            except Exception:
                pass
        else:
            missing.append(name)

    return json.dumps({
        "success": True,
        "configured": configured,
        "missing": missing,
    }, ensure_ascii=False)


def _delete_credential(name: str) -> str:
    """Delete a credential by name."""
    store = _store()
    result = store.delete(name)
    return json.dumps(result, ensure_ascii=False)


def credential_tool(
    action: str = "list",
    name: Optional[str] = None,
    names: Optional[List[str]] = None,
    description: Optional[str] = None,
    instructions: Optional[str] = None,
    overwrite: bool = False,
) -> str:
    """Dispatch credential tool actions."""
    if action == "request":
        if not name:
            return json.dumps(
                {"success": False, "error": "name is required for request action."},
                ensure_ascii=False,
            )
        return _request_credential(
            name=name,
            description=description,
            instructions=instructions,
            overwrite=overwrite,
        )

    elif action == "list":
        return _list_credentials()

    elif action == "status":
        if not names:
            return json.dumps(
                {"success": False, "error": "names list is required for status action."},
                ensure_ascii=False,
            )
        return _status_credentials(names)

    elif action == "delete":
        if not name:
            return json.dumps(
                {"success": False, "error": "name is required for delete action."},
                ensure_ascii=False,
            )
        return _delete_credential(name)

    else:
        return json.dumps(
            {"success": False, "error": f"Unknown action: {action}. Use request, list, status, or delete."},
            ensure_ascii=False,
        )


# --- Registry ---
from tools.registry import registry

registry.register(
    name="credentials",
    toolset="credential",
    schema=CREDENTIAL_SCHEMA,
    handler=lambda args, **kw: credential_tool(
        action=args.get("action", "list"),
        name=args.get("name"),
        names=args.get("names"),
        description=args.get("description"),
        instructions=args.get("instructions"),
        overwrite=bool(args.get("overwrite", False)),
    ),
    check_fn=check_credential_requirements,
    emoji="🔐",
)