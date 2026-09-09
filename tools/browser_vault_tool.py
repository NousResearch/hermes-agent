#!/usr/bin/env python3
"""Vault-backed model-blind browser autofill tools.

Two model-facing tools, gated on the local vault having at least one item
(zero schema cost otherwise, same ``check_fn`` pattern as the Home Assistant
tools):

- ``browser_vault_list``  → handles + metadata (for logins this includes the
  identifier — it is NOT a secret; the agent types it itself). Passwords are
  never returned.
- ``browser_vault_fill``  → server-side fill of ONLY the password field of
  the CURRENT page's login form from a vault handle. The password is
  resolved locally, the page origin must EXACTLY match the item's bound
  origin (pre-checked AND re-asserted synchronously inside the fill script),
  the field is chosen by the ported login-control classifier, injection runs
  exclusively over the supervisor CDP WebSocket (never argv), and the tool
  result reports only ``{filled_fields, kind, origin, success}`` — the
  password never appears in tool results, logs, or the session DB, and its
  exact bytes are registered with the browser-result redaction boundary so
  no later browser tool call can echo them back to the model.

Ported design from Merit-Systems/OpenInstinct (MIT): opaque-handle vault
autofill (kernel-login-autofill.ts / fill_from_vault.ts).
"""

from __future__ import annotations

import json
import secrets
import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Availability check
# ---------------------------------------------------------------------------

def _check_vault_available() -> bool:
    """Schema-gate: the tools appear only when the local vault has items or an external manager is enabled."""
    try:
        from agent.vault_backends import enabled_backends
        from agent.vault_store import get_vault_store
        return get_vault_store().has_items() or any(b.needs_unlock for b in enabled_backends())
    except Exception:
        return False


# ---------------------------------------------------------------------------
# JS evaluation plumbing (server-side; results never carry secret values)
# ---------------------------------------------------------------------------

def _eval_js(task_id: str, expression: str) -> Dict[str, Any]:
    """Evaluate NON-SECRET JS on the current page (inspection, origin reads).

    Prefers the supervisor's persistent CDP WebSocket, falls back to the
    agent-browser CLI ``eval`` command. Never use this for expressions that
    embed secret values — the fallback places the expression in subprocess
    argv. Use :func:`_eval_js_secret` for secret-bearing expressions.
    """
    try:
        from tools.browser_supervisor import SUPERVISOR_REGISTRY

        supervisor = SUPERVISOR_REGISTRY.get(task_id)
        if supervisor is not None:
            sup = supervisor.evaluate_runtime(expression)
            if sup.get("ok"):
                return {"success": True, "result": sup.get("result")}
            err = str(sup.get("error") or "")
            if "supervisor" not in err.lower():
                return {"success": False, "error": err}
    except ImportError:
        pass
    except Exception as exc:  # pragma: no cover — defensive
        logger.debug("vault fill: supervisor eval unavailable (%s)", exc)

    from tools.browser_tool import _last_session_key
    from tools.browser_tool_session import _run_browser_command

    effective = _last_session_key(task_id)
    result = _run_browser_command(effective, "eval", [expression])
    if not result.get("success"):
        return {"success": False, "error": result.get("error", "eval failed")}
    return {"success": True, "result": result.get("data", {}).get("result")}


def _eval_js_secret(task_id: str, expression: str) -> Dict[str, Any]:
    """Evaluate a SECRET-BEARING JS expression. Supervisor CDP-WS only.

    Fails closed: there is deliberately NO fallback to the agent-browser CLI
    ``eval`` path, because that places the expression — and therefore the
    credential bytes — in subprocess argv, visible to any process listing.
    When no supervisor session is available the caller gets a typed refusal
    (``error_type='supervisor_required'``) and nothing is written.
    """
    try:
        from tools.browser_supervisor import SUPERVISOR_REGISTRY

        supervisor = SUPERVISOR_REGISTRY.get(task_id)
    except ImportError:
        supervisor = None
    except Exception as exc:  # pragma: no cover — defensive
        logger.debug("vault fill: supervisor registry unavailable (%s)", exc)
        supervisor = None

    if supervisor is None:
        return {
            "success": False,
            "error_type": "supervisor_required",
            "error": (
                "Vault fill requires the supervised browser session (direct "
                "CDP WebSocket). The fallback eval path would place the "
                "credential in subprocess argv, so it is never used for "
                "secrets. Start the browser through the Hermes-managed "
                "session and retry."
            ),
        }

    sup = supervisor.evaluate_runtime(expression)
    if sup.get("ok"):
        return {"success": True, "result": sup.get("result")}
    return {
        "success": False,
        "error_type": "supervisor_required"
        if "supervisor" in str(sup.get("error") or "").lower()
        else "eval_failed",
        "error": str(sup.get("error") or "eval failed"),
    }


def _parse_json_result(raw: Any) -> Any:
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            return raw
    return raw


def _current_page_origin(task_id: str) -> Optional[str]:
    res = _eval_js(task_id, "window.location.href")
    if not res.get("success"):
        return None
    href = str(res.get("result") or "").strip().strip('"').strip("'")
    if not href or href == "about:blank":
        return None
    try:
        from agent.vault_store import normalize_origin

        return normalize_origin(href)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------

def browser_vault_list() -> str:
    """List login handles + metadata across every enabled backend. Passwords are never included.

    A locked external manager contributes no items; instead it is reported under ``locked`` so the
    agent knows to call browser_vault_fill (which prompts the user to unlock) or tell the user.
    """
    from agent.vault_backends import enabled_backends
    from agent.vault_backends.unlock import can_prompt_here

    items, locked, errors = [], [], []
    for backend in enabled_backends():
        if backend.needs_unlock and not backend.is_unlocked():
            locked.append({"backend": backend.name, "display_name": backend.display_name,
                           "unlock": "browser_vault_unlock" if can_prompt_here() else "unavailable_in_this_session"})
            continue
        try:
            metas = backend.list_items()
        except Exception as exc:
            errors.append({"backend": backend.name, "error": str(exc)[:200]})
            continue
        for meta in metas:
            entry = {"handle": meta.id, "backend": backend.name, "label": meta.label, "kind": meta.kind,
                     "origin": meta.origin, "available": meta.kind == "login"}
            if meta.identifier:
                entry["identifier"] = meta.identifier
                entry["identifier_type"] = meta.identifier_type
            items.append(entry)
    out: Dict[str, Any] = {"success": True, "items": items}
    if locked:
        out["locked"] = locked
    if errors:
        out["errors"] = errors
    return json.dumps(out, ensure_ascii=False)


def browser_vault_unlock(backend_name: str) -> str:
    """Ask the user (via the surface's masked prompt) to unlock an external manager for this session."""
    from agent.vault_backends import enabled_backends
    from agent.vault_backends.unlock import can_prompt_here, get_unlock_prompt_callback

    backend = next((b for b in enabled_backends() if b.name == backend_name and b.needs_unlock), None)
    if backend is None:
        return json.dumps({"success": False, "error": f"No unlockable vault backend named {backend_name!r}."})
    if backend.is_unlocked():
        return json.dumps({"success": True, "backend": backend.name, "already_unlocked": True})
    if not can_prompt_here():
        return json.dumps({"success": False, "error_type": "unlock_unavailable",
                           "error": (f"{backend.display_name} is locked and this session cannot prompt for the "
                                     "master password (headless/cron/API). Unlock it from an interactive Hermes "
                                     "session or the Desktop app first.")})
    prompt = get_unlock_prompt_callback()
    master = prompt(backend.name, backend.display_name) if prompt else ""
    if not master:
        return json.dumps({"success": False, "error_type": "unlock_cancelled",
                           "error": f"The user declined to unlock {backend.display_name}."})
    try:
        backend.unlock(master)  # type: ignore[attr-defined]
    except Exception as exc:
        return json.dumps({"success": False, "error_type": "unlock_failed", "error": str(exc)[:300]})
    finally:
        del master
    return json.dumps({"success": True, "backend": backend.name})


def browser_vault_fill(handle: str, task_id: Optional[str] = None) -> str:
    """Fill the current page's password field from a vault handle.

    Password-only: the identifier is agent-visible metadata (see
    browser_vault_list) and is typed by the agent via normal input tools.
    The password is resolved server-side and injected via in-page JS over
    the supervisor CDP WebSocket; the result reports only counts/metadata.
    """
    from agent.redact import register_vault_redaction_value
    from agent.vault_login_classifier import (
        ClassifiedLoginControl,
        LoginControl,
        build_fill_js,
        build_inspection_js,
        classify_login_control,
        select_password_fill,
    )
    from agent.vault_backends import UnlockRequired, backend_for_handle
    from agent.vault_store import scrub_secret_from_text

    effective_task_id = task_id or "default"
    backend = backend_for_handle(handle)
    if backend is not None and backend.needs_unlock and not backend.is_unlocked():
        unlocked = json.loads(browser_vault_unlock(backend.name))
        if not unlocked.get("success"):
            return json.dumps(unlocked)

    try:
        meta = backend.get_meta(handle) if backend is not None else None
    except UnlockRequired:
        return json.dumps({"success": False, "error_type": "unlock_required",
                           "error": f"{backend.display_name} locked again; call browser_vault_unlock."})
    if meta is None:
        return json.dumps(
            {
                "success": False,
                "error": (
                    f"No vault item with handle {handle!r}. Use browser_vault_list. "
                    "To save a credential: run `hermes vault add` in a terminal, or "
                    "in the desktop app open Settings → Credential Vault."
                ),
            }
        )
    if meta.kind != "login":
        return json.dumps(
            {"success": False, "error": f"Vault item {handle!r} is kind={meta.kind!r}; only login items can be filled in Phase 1."}
        )

    # ── Origin binding pre-check (cheap early exit; the authoritative check
    # runs synchronously inside the fill script itself) ──────────────────────
    page_origin = _current_page_origin(effective_task_id)
    if not page_origin:
        return json.dumps(
            {"success": False, "error": "Could not determine the current page origin. Navigate to the login page first."}
        )
    if page_origin != meta.origin:
        return json.dumps(
            {
                "success": False,
                "error_type": "origin_mismatch",
                "error": (
                    f"Refused: current page origin ({page_origin}) does not match "
                    f"the vault item's bound origin ({meta.origin}). Vault fills "
                    "only run on the exact origin the credential was saved for."
                ),
            }
        )

    # ── Inspect + classify page controls ────────────────────────────────────
    nonce = secrets.token_hex(8)  # binds this fill to THIS inspection's stamps
    inspect = _eval_js(effective_task_id, build_inspection_js(nonce))
    if not inspect.get("success"):
        return json.dumps(
            {"success": False, "error": f"Could not inspect page inputs: {inspect.get('error', 'eval failed')}"}
        )
    raw_controls = _parse_json_result(inspect.get("result"))
    if isinstance(raw_controls, str):
        raw_controls = _parse_json_result(raw_controls)
    if not isinstance(raw_controls, list):
        return json.dumps({"success": False, "error": "Page input inspection returned no usable controls."})

    classified: list[ClassifiedLoginControl] = []
    for raw in raw_controls:
        if not isinstance(raw, dict):
            continue
        result = classify_login_control(LoginControl.from_dict(raw))
        if result is not None:
            classified.append(result)
    if not classified:
        return json.dumps({"success": False, "error": "No login form fields were found on the current page."})

    # ── Resolve secret and fill (secret never enters any logged string) ─────
    try:
        password = backend.resolve_password(handle)
    except UnlockRequired:
        return json.dumps({"success": False, "error_type": "unlock_required",
                           "error": f"{backend.display_name} locked again; call browser_vault_unlock."})
    secret = {"password": password}
    fills = select_password_fill(classified, password)
    if not fills:
        return json.dumps(
            {"success": False, "error": "No fillable password field matched (is there a password field on this page?)."}
        )

    # Register the secret bytes with the model-egress redaction boundary
    # BEFORE they touch the page: any later browser_* result (including
    # browser_cdp Runtime.evaluate reads) that echoes them is scrubbed.
    register_vault_redaction_value(password)

    try:
        fill_result = _eval_js_secret(
            effective_task_id, build_fill_js(fills, expected_origin=str(meta.origin), nonce=nonce)
        )
    except Exception as exc:
        # Strip any secret material from exception text before surfacing.
        return json.dumps(
            {"success": False, "error": scrub_secret_from_text(str(exc), secret)}
        )
    if not fill_result.get("success"):
        err = scrub_secret_from_text(str(fill_result.get("error") or "fill failed"), secret)
        out = {"success": False, "error": err}
        if fill_result.get("error_type"):
            out["error_type"] = fill_result["error_type"]
        return json.dumps(out)

    parsed = _parse_json_result(fill_result.get("result"))
    if isinstance(parsed, str):
        parsed = _parse_json_result(parsed)
    if isinstance(parsed, dict) and parsed.get("refused") == "origin_changed":
        return json.dumps(
            {
                "success": False,
                "error_type": "origin_changed",
                "error": (
                    "Refused: the page navigated away from the bound origin "
                    f"({meta.origin}) before the fill could run "
                    f"(now on {parsed.get('found') or 'unknown'}). "
                    "Nothing was written."
                ),
            }
        )
    filled = parsed.get("filled", 0) if isinstance(parsed, dict) else 0

    return json.dumps(
        {
            "success": bool(filled),
            "filled_fields": int(filled),
            "backend": backend.name,
            "kind": meta.kind,
            "origin": meta.origin,
        }
    )


# ---------------------------------------------------------------------------
# Schemas + registration
# ---------------------------------------------------------------------------

BROWSER_VAULT_LIST_SCHEMA = {
    "name": "browser_vault_list",
    "description": (
        "List saved website logins as handles with metadata (label, backend, bound origin, and the "
        "identifier + identifier_type so you can type the username yourself with fill_input). "
        "Passwords are NEVER returned. Sources: the local Hermes vault plus any enabled password "
        "manager (1Password, Bitwarden). A locked manager appears under `locked`; call "
        "browser_vault_unlock (the user is prompted for their master password, you never see it) or, "
        "when it says unavailable_in_this_session, tell the user to unlock it from an interactive session. "
        "Workflow: fill_input the identifier, then browser_vault_fill with the handle."
    ),
    "input_schema": {"type": "object", "properties": {}, "required": []},
}

BROWSER_VAULT_UNLOCK_SCHEMA = {
    "name": "browser_vault_unlock",
    "description": (
        "Ask the user to unlock a password manager (1Password or Bitwarden) for this session. The master "
        "password is typed into a masked prompt owned by the UI and never enters the conversation. "
        "Returns success, unlock_cancelled, unlock_failed, or unlock_unavailable (headless session)."
    ),
    "input_schema": {
        "type": "object",
        "properties": {"backend": {"type": "string", "enum": ["onepassword", "bitwarden"],
                                   "description": "Backend name from browser_vault_list `locked`."}},
        "required": ["backend"],
    },
}

BROWSER_VAULT_FILL_SCHEMA = {
    "name": "browser_vault_fill",
    "description": (
        "Fill ONLY the password field of the CURRENT browser page's login form from a vault handle "
        "(see browser_vault_list). Type the identifier/username yourself first with fill_input, then "
        "call this. The password is resolved from the local vault or the password manager and injected "
        "server-side; it never appears in the conversation. Refused unless the page origin exactly "
        "matches the credential's bound origin (re-checked atomically at fill time). If the manager is "
        "locked the user is prompted to unlock first."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "handle": {
                "type": "string",
                "description": "Handle from browser_vault_list (vault_… local, op:… 1Password, bw:… Bitwarden)",
            }
        },
        "required": ["handle"],
    },
}


def _handle_vault_list(args: Dict[str, Any], **kwargs) -> str:
    return browser_vault_list()


def _handle_vault_unlock(args: Dict[str, Any], **kwargs) -> str:
    return browser_vault_unlock(str(args.get("backend") or ""))


def _handle_vault_fill(args: Dict[str, Any], **kwargs) -> str:
    return browser_vault_fill(
        handle=str(args.get("handle") or ""), task_id=kwargs.get("task_id")
    )


from tools.registry import registry  # noqa: E402

registry.register(
    name="browser_vault_list",
    toolset="browser",
    schema=BROWSER_VAULT_LIST_SCHEMA,
    handler=_handle_vault_list,
    check_fn=_check_vault_available,
    emoji="🔐",
)

registry.register(
    name="browser_vault_unlock",
    toolset="browser",
    schema=BROWSER_VAULT_UNLOCK_SCHEMA,
    handler=_handle_vault_unlock,
    check_fn=_check_vault_available,
    emoji="🔐",
)

registry.register(
    name="browser_vault_fill",
    toolset="browser",
    schema=BROWSER_VAULT_FILL_SCHEMA,
    handler=_handle_vault_fill,
    check_fn=_check_vault_available,
    emoji="🔐",
)
