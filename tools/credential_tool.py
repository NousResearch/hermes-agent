"""Credential tool — agent-facing interface to the encrypted credential store.

The agent may REQUEST a credential from the user, LIST stored names, check
STATUS, or DELETE. Values are encrypted at rest and never returned to the
model: ``list``/``status`` are metadata-only, and a value reaches execution
only through execute_code's ``credentials=`` parameter (where it is registered
with the redactor before it can be echoed).

Actions:
  request — ask the user for a value (masked prompt, interactive terminals only)
  list    — stored names + timestamps (never values)
  status  — which of the given names exist (never values)
  delete  — remove a credential by name
"""

from __future__ import annotations

import json
import logging
import sys
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

CREDENTIAL_SCHEMA = {
    "name": "credentials",
    "description": (
        "Manage credentials (API keys, tokens, passwords) the agent needs for its work. "
        "Values are encrypted at rest and NEVER returned to you: you get names and "
        "metadata, and a value is available only as a variable inside execute_code "
        "when you pass its name in that call's `credentials` parameter.\n\n"
        "REQUEST: when you need a credential the user has not provided, call "
        "credentials(action='request', name='...', description='...'). On an interactive "
        "terminal the user is prompted with masked input (the value never enters this "
        "context); otherwise you get an error telling you how the user can store it "
        "with `/creds set <name>` in the CLI.\n\n"
        "LIST: credentials(action='list') — check what exists before requesting, to "
        "avoid duplicates.\n\n"
        "STATUS: credentials(action='status', names=['...']) — check specific names "
        "before relying on them.\n\n"
        "DELETE: credentials(action='delete', name='...') — remove a credential that "
        "is no longer needed.\n\n"
        "USE: reference a stored credential by passing its name in execute_code's "
        "`credentials` list; the decrypted value is bound to a variable of that name "
        "for the script only. Never read or write ~/.hermes/credentials/ directly "
        "(the file tools refuse those paths)."
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
                    "Credential name, e.g. 'supabase_service_key'. Required for "
                    "request and delete. Letters, digits, underscores and hyphens; "
                    "at least 3 characters; generic names like 'token' are rejected."
                ),
            },
            "names": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of credential names. Required for action='status'.",
            },
            "description": {
                "type": "string",
                "description": (
                    "What this credential is for (e.g. 'Supabase service role key for "
                    "database admin'). Shown to the user in the request prompt; only "
                    "used with action='request'."
                ),
            },
            "instructions": {
                "type": "string",
                "description": (
                    "Where the user can find it (e.g. 'Supabase dashboard → Project "
                    "Settings → Database'). Shown in the request prompt; only used "
                    "with action='request'."
                ),
            },
            "overwrite": {
                "type": "boolean",
                "description": (
                    "With action='request': replace an existing credential instead of "
                    "reporting it as already stored. Default false."
                ),
            },
            "backend": {
                "type": "string",
                "enum": ["auto", "local", "bitwarden", "onepassword"],
                "description": (
                    "Where to store the value (action='request'). 'auto' (default) uses "
                    "the secret platform that `hermes secrets` has set up — Bitwarden "
                    "Secrets Manager or 1Password — and falls back to the local encrypted "
                    "store when neither is available; if BOTH are available, 'auto' is "
                    "ambiguous and this call fails asking you to pick one (the user "
                    "decides which they want). 'local' forces the local store. Values "
                    "saved to a platform are NOT copied locally — only an index entry is."
                ),
            },
        },
        "required": ["action"],
    },
}


def check_credential_requirements() -> bool:
    """Always available — no external requirements."""
    return True


def _store():
    # Lazy so tool discovery can import this module before HERMES_HOME is stable;
    # get_store() resolves the home per call, so the right profile's store is used.
    from agent.credential_store import get_store

    return get_store()


def _error(message: str) -> str:
    return json.dumps({"success": False, "error": message}, ensure_ascii=False)


def _interactive_terminal() -> bool:
    """True when a masked prompt can actually reach the user (CLI/TUI stdin+stdout)."""
    try:
        return bool(sys.stdin.isatty() and sys.stdout.isatty())
    except (AttributeError, OSError):
        return False


def _request_credential(
    name: str,
    description: Optional[str] = None,
    instructions: Optional[str] = None,
    overwrite: bool = False,
    backend: str = "auto",
) -> str:
    """Prompt the user for a value on an interactive terminal; refuse otherwise."""
    from agent.credential_store import CredentialStoreError, validate_name

    try:
        validate_name(name)
    except ValueError as exc:
        return _error(str(exc))

    store = _store()
    try:
        already = name in {entry["name"] for entry in store.list()}
    except CredentialStoreError as exc:
        return _error(str(exc))
    if already and not overwrite:
        return json.dumps({
            "success": True,
            "already_exists": True,
            "name": name,
            "message": f"Credential '{name}' is already stored. Use overwrite=true to replace it.",
        }, ensure_ascii=False)

    if not _interactive_terminal():
        return _error(
            f"No interactive terminal here to collect '{name}' safely. Ask the user to run "
            f"`/creds set {name}` in the Hermes CLI (masked prompt, value stays out of this "
            f"context), then check status again."
        )

    prompt = f"Enter value for credential '{name}'"
    if description:
        prompt = f"{description} ({name})"
    if instructions:
        prompt += f"\n  Where to find it: {instructions}"
    prompt += "\n  Value (hidden): "
    try:
        from hermes_cli.secret_prompt import masked_secret_prompt

        value = masked_secret_prompt(prompt)
    except (EOFError, KeyboardInterrupt):
        value = ""
    except ImportError:
        import getpass

        try:
            value = getpass.getpass(prompt)
        except (EOFError, KeyboardInterrupt):
            value = ""

    if not value:
        return _error("User cancelled or provided an empty value.")

    try:
        result = _store_credential(name, value.strip(), backend=backend, overwrite=overwrite)
    except (ValueError, CredentialStoreError) as exc:
        return _error(str(exc))
    if not result.get("success"):
        return _error(str(result.get("error") or "Failed to store credential."))
    _register(value.strip())
    message = (f"Credential '{name}' stored in {_backend_display(result)}. Pass its name in "
               f"execute_code's `credentials` parameter to use it.")
    for warning in result.get("warnings", []):
        message += f" Note: {warning}"
    return json.dumps({
        "success": True,
        "stored": True,
        "name": name,
        "backend": result.get("backend", "local"),
        "message": message,
    }, ensure_ascii=False)


def _store_credential(name: str, value: str, *, backend: str, overwrite: bool) -> dict:
    """save_credential with ambiguity surfaced as an actionable tool error."""
    from agent.credential_backends import (
        AmbiguousBackendError,
        BackendError,
        get_backend,
        save_credential,
    )

    try:
        return save_credential(name, value, backend=backend, overwrite=overwrite)
    except AmbiguousBackendError as exc:
        choices = ", ".join(exc.names)
        return {"success": False, "error":
                f"More than one secret platform is available ({choices}). Ask the user which "
                f"one to use, then retry with backend set to one of them (or 'local')."}
    except BackendError as exc:  # explicit --backend that cannot be used right now
        return {"success": False, "error": str(exc)}


def _backend_display(result: dict) -> str:
    from agent.credential_backends import LOCAL, get_backend

    name = str(result.get("backend") or LOCAL)
    if name == LOCAL:
        return "the local encrypted store"
    return get_backend(name).display_name


def _register(value: str) -> None:
    """Register a value that just left the store so output paths scrub it."""
    from agent.redact import register_secret

    register_secret(value)


def _list_credentials() -> str:
    """Names + timestamps only — never decrypts, so no plaintext crosses this path."""
    from agent.credential_store import CredentialStoreError

    try:
        entries = _store().list()
    except CredentialStoreError as exc:
        return _error(str(exc))
    if not entries:
        return json.dumps(
            {"success": True, "entries": [], "count": 0, "message": "No credentials stored."},
            ensure_ascii=False,
        )
    return json.dumps({"success": True, "entries": entries, "count": len(entries)},
                      ensure_ascii=False)


def _status_credentials(names: list[str]) -> str:
    """Existence check against the metadata listing — never decrypts."""
    from agent.credential_store import CredentialStoreError

    try:
        stored = {entry["name"] for entry in _store().list()}
    except CredentialStoreError as exc:
        return _error(str(exc))
    configured = [name for name in names if name in stored]
    missing = [name for name in names if name not in stored]
    return json.dumps({"success": True, "configured": configured, "missing": missing},
                      ensure_ascii=False)


def _delete_credential(name: str) -> str:
    from agent.credential_store import CredentialStoreError

    try:
        result = _store().delete(name)
    except CredentialStoreError as exc:
        return _error(str(exc))
    return json.dumps(result, ensure_ascii=False)


def credential_tool(
    action: str = "list",
    name: Optional[str] = None,
    names: Optional[list[str]] = None,
    description: Optional[str] = None,
    instructions: Optional[str] = None,
    overwrite: bool = False,
    backend: str = "auto",
) -> str:
    """Dispatch credential tool actions."""
    if action == "request":
        if not name:
            return _error("name is required for the request action.")
        return _request_credential(name, description=description, instructions=instructions,
                                   overwrite=overwrite, backend=backend)
    if action == "list":
        return _list_credentials()
    if action == "status":
        if not names:
            return _error("names list is required for the status action.")
        return _status_credentials(names)
    if action == "delete":
        if not name:
            return _error("name is required for the delete action.")
        return _delete_credential(name)
    return _error(f"Unknown action: {action}. Use request, list, status, or delete.")


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
        backend=str(args.get("backend") or "auto"),
    ),
    check_fn=check_credential_requirements,
    emoji="🔐",
)