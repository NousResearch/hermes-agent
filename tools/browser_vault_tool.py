#!/usr/bin/env python3
"""Vault-backed model-blind browser autofill tools.

One model-facing tool, ``browser_vault``, deferred behind tool_search (only its catalog line is
paid per request) and available wherever a browser is. Its actions:

- ``list``  → handles + metadata (for logins this includes the
  identifier — it is NOT a secret; the agent types it itself). Passwords are
  never returned.
- ``fill``  → server-side fill of the CURRENT page from a vault
  handle: the password field for logins, card fields for payment items (after
  the user confirms), address fields for address items. The secret is
  resolved locally, the page origin must EXACTLY match the item's bound
  origin (pre-checked AND re-asserted synchronously inside the fill script),
  the field is chosen by the ported login-control classifier, injection runs
  exclusively over the supervisor CDP WebSocket (never argv) — or, for a plugin browser with no CDP
  endpoint, through its registered ``BrowserPageBackend`` (Camofox's REST evaluate) — and the tool
  result reports only ``{filled_fields, kind, origin, success}`` — the
  password never appears in tool results, logs, or the session DB, and its
  exact bytes are registered with the browser-result redaction boundary so
  no later browser tool call can echo them back to the model.
- ``enter_code`` → a one-time code for the current page: minted from a saved
  login's authenticator key (same origin binding as ``fill``, checked before
  the code is generated) or typed by the user into their surface's prompt.

Ported design from Merit-Systems/OpenInstinct (MIT): opaque-handle vault
autofill (kernel-login-autofill.ts / fill_from_vault.ts).
"""

from __future__ import annotations

import json
import secrets
import logging
from typing import Any, Dict, Optional

from tools.browser_vault_context import plugin_errors_as_refusal, saved_origins, snapshot_refusal, write_backend_code

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Availability check
# ---------------------------------------------------------------------------

def _check_vault_available() -> bool:
    """Schema-gate: the vault rides with the browser. An empty vault still needs save_login so the
    agent can offer to remember a login the first time it meets a form; hiding the tool until an item
    exists meant nobody ever discovered the feature. Deferral keeps that presence nearly free."""
    from agent.browser_page_registry import list_providers
    from tools.browser_session_cdp import get_cdp_override_raw
    from tools.browser_use_cli import browser_exec_available
    return bool(browser_exec_available() or get_cdp_override_raw() or list_providers())


# ---------------------------------------------------------------------------
# JS evaluation plumbing (server-side; results never carry secret values)
# ---------------------------------------------------------------------------

# A secret-bearing script runs inside this wrapper on a page backend, so a page exception is answered
# in-page and its message (which may quote the script) never reaches the backend's error path or logs.
_PAGE_FILL_FAILED = {"hermes_vault_eval_failed": True}


def _page_backend(task_id: str):
    """The plugin browser holding this task's live page (e.g. Camofox), or None for the CDP session."""
    from agent.browser_page_registry import backend_for_task
    return backend_for_task(task_id)


def _page_eval(backend: Any, task_id: str, expression: str, *, secret: bool) -> dict[str, Any]:
    """Evaluate on a plugin page backend. On the secret path every failure gets fixed text."""
    if secret:
        expression = ("(() => { try { return (" + expression + "); } catch (_) { return "
                      + json.dumps(_PAGE_FILL_FAILED) + "; } })()")
    try:
        result = backend.evaluate(task_id, expression, secret=secret)
    except Exception as exc:  # health: allow BLE001 -- plugin failure becomes a typed refusal
        detail = "" if secret else f": {str(exc)[:200]}"
        return {"success": False, "error_type": "eval_failed",
                "error": f"The {backend.name} browser could not run the vault script{detail}"}
    if secret and _parse_json_result(result) == _PAGE_FILL_FAILED:
        return {"success": False, "error_type": "eval_failed",
                "error": f"The {backend.name} browser could not run the vault script"}
    return {"success": True, "result": result}


def _eval_js(task_id: str, expression: str) -> dict[str, Any]:
    """Evaluate NON-SECRET JS on the current page (inspection, origin reads): on the plugin browser
    holding the task's page, else over the task's CDP supervisor. Without either the call reports it."""
    page = _page_backend(task_id)
    if page is not None:
        return _page_eval(page, task_id, expression, secret=False)
    supervisor = _ensure_supervisor(task_id)
    if supervisor is None:
        return {"success": False, "error": "no supervised browser session for this task; open the page with "
                                           "browser_exec first"}
    sup = supervisor.evaluate_runtime(expression)
    if sup.get("ok"):
        return {"success": True, "result": sup.get("result")}
    return {"success": False, "error": str(sup.get("error") or "evaluate failed")}


def _ensure_supervisor(task_id: str):
    """The supervisor for ``task_id`` (browser_exec attaches one on every call), attaching one on demand to
    the task's session CDP endpoint when none is running. None when no endpoint is known; a fill then
    refuses rather than touching argv."""
    from tools.browser_supervisor import SUPERVISOR_REGISTRY

    supervisor = SUPERVISOR_REGISTRY.get(task_id)
    if supervisor is not None:
        return supervisor
    from tools.browser_session import session_for_task
    from tools.browser_session_cdp import get_dialog_policy_config, resolve_cdp_override

    cdp_url = str(session_for_task(task_id).get("cdp_url") or "")
    if not cdp_url:
        return None
    policy, timeout_s = get_dialog_policy_config()
    try:
        return SUPERVISOR_REGISTRY.get_or_start(task_id=task_id, cdp_url=resolve_cdp_override(cdp_url),
                                                dialog_policy=policy, dialog_timeout_s=timeout_s)
    except Exception as exc:  # health: allow BLE001 -- a failed attach is reported as supervisor_required
        logger.debug("vault fill: supervisor attach to the task's session failed (%s)", exc)
        return None


def _eval_js_secret(task_id: str, expression: str) -> dict[str, Any]:
    """Evaluate a SECRET-BEARING JS expression. Supervisor CDP-WS only.

    Fails closed: there is deliberately NO subprocess path, because that would
    place the expression — and therefore the credential bytes — in argv,
    visible to any process listing. When no supervisor session is available
    the caller gets a typed refusal (``error_type='supervisor_required'``) and
    nothing is written. A plugin browser holding the task's page (no CDP endpoint) carries the
    expression over its own transport instead; see ``agent.browser_page_backend``.
    """
    page = _page_backend(task_id)
    supervisor = None
    if page is None:
        try:
            supervisor = _ensure_supervisor(task_id)
        except Exception as exc:
            logger.debug("vault fill: supervisor unavailable (%s)", exc)

    if page is None and supervisor is None:
        return {
            "success": False,
            "error_type": "supervisor_required",
            "error": (
                "Vault fill requires the supervised browser session (direct "
                "CDP WebSocket), and this task has none. Open the page with "
                "browser_exec and retry."
            ),
        }

    # Re-admit at the WRITE. The handler-level fence (_fenced_page_op) admitted before a possibly
    # human-length prompt (enter_code waits for the user's code); a takeover during that wait must
    # refuse here, before the credential lands in a page the human is now typing into. The outer
    # epoch check only discards the result, and a fill is a side effect, not a result.
    if _bot_desktop_browser_session(task_id):
        from tools.bot_desktop import lease as _bd_lease
        try:
            _bd_lease.assert_agent_may_act()
        except _bd_lease.HumanHasControl as exc:
            return {"success": False, "error_type": "human_has_control", "error": str(exc)}

    if page is not None:
        return _page_eval(page, task_id, expression, secret=True)
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


def _deep_probe(selector: str) -> str:
    """A JS expression truthy when ``selector`` matches in the document or any open shadow root (the
    inspection and fill scripts reach shadow-root forms, so tab selection must too)."""
    return ("((s) => { const w = (r) => r.querySelector(s) || Array.from(r.querySelectorAll('*'))"
            ".some((e) => e.shadowRoot && w(e.shadowRoot)); return !!w(document); })(" + json.dumps(selector) + ")")


# Per kind: a JS probe that is truthy on a tab holding the form this kind fills.
_TAB_PROBES = {
    "login": _deep_probe("input[type=password]"),
    "payment": _deep_probe("input[autocomplete^=cc-], [name*=card i], [placeholder*=card i], [name*=cvc i], [name*=cvv i]"),
    "address": _deep_probe("input[autocomplete^=address-], [autocomplete=postal-code], [name*=address i], [name*=zip i], [name*=postal i]"),
}


def _focus_bound_origin(task_id: str, origin: str, kind: str) -> Optional[str]:
    """Point the supervisor's page session at the open tab on ``origin`` that holds a ``kind`` form
    (browser_exec sessions open their own tabs, so the tab the supervisor attached to first is rarely the
    login page). Returns the origin when a tab was focused, else None (caller falls back to the current page).
    A plugin page backend's current page IS the tab the agent drove, so there is nothing to focus."""
    if _page_backend(task_id) is not None:
        return None
    try:
        supervisor = _ensure_supervisor(task_id)
    except Exception:
        supervisor = None
    if supervisor is None:
        return None
    focused = supervisor.focus_page(origin, accept=_TAB_PROBES.get(kind))
    return (origin or focused.get("url")) if focused.get("ok") else None


def _bound_page_origin(task_id: str, meta: Any, probe_kind: str) -> str | dict[str, Any]:
    """The origin guard every fill of a stored secret (password, one-time code, card, address) passes
    BEFORE the backend resolves any value: the page origin when it is one of the item's saved origins,
    else the refusal to return. Manager items may bind several websites; matching is exact and
    nothing wildcard or parent-domain is inferred. The caller hands the returned origin to ``build_fill_js``,
    which re-asserts it inside the page immediately before the write (navigation between check and fill)."""
    allowed = saved_origins(meta)
    if not allowed:
        return {"success": False, "error_type": "no_origin",
                "error": f"Vault item {meta.id!r} has no bound origin; {meta.kind} items are filled only on "
                         "the site they were saved for."}
    page_origin = next((o for o in (_focus_bound_origin(task_id, c, probe_kind) for c in allowed) if o), None)
    page_origin = page_origin or _current_page_origin(task_id)
    if not page_origin:
        return {"success": False,
                "error": "Could not determine the current page origin. Navigate to the login page first."}
    if page_origin not in allowed:
        return {"success": False, "error_type": "origin_mismatch",
                "error": (f"Refused: current page origin ({page_origin}) does not match the vault item's bound "
                          f"origin(s) ({', '.join(allowed)}). Vault fills only run on the exact origin(s) the "
                          "credential was saved for.")}
    return page_origin


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------

def browser_vault_list() -> str:
    """List login handles + metadata across every enabled backend. Passwords are never included.

    A locked external manager contributes no items; instead it is reported under ``locked`` so the
    agent knows to call fill (which prompts the user to unlock) or tell the user.
    """
    from agent.vault_backends import enabled_backends
    from agent.vault_backends.unlock import can_prompt_here

    items, locked, errors = [], [], []
    for backend in enabled_backends():
        if backend.needs_unlock and not backend.is_unlocked():
            locked.append({"backend": backend.name, "display_name": backend.display_name,
                           "unlock": "browser_vault action=unlock" if can_prompt_here() else "unavailable_in_this_session"})
            continue
        try:
            metas = backend.list_items()
        except Exception as exc:
            errors.append({"backend": backend.name, "error": str(exc)[:200]})
            continue
        for meta in metas:
            entry = {"handle": meta.id, "backend": backend.name, "label": meta.label, "kind": meta.kind,
                     "origin": meta.origin, "available": meta.kind == "login" or bool(meta.origin)}
            if len(meta.allowed_origins) > 1:
                entry["allowed_origins"] = list(meta.allowed_origins)
            if meta.has_otp or backend.needs_unlock:
                entry["two_factor"] = "automatic" if meta.has_otp else "automatic if the manager stores a TOTP seed, else the user is asked"
            if meta.identifier:
                entry["identifier"] = meta.identifier
                entry["identifier_type"] = meta.identifier_type
            items.append(entry)
    out: dict[str, Any] = {"success": True, "items": items}
    if not items:
        out["hint"] = ("No saved logins. On a login page, call browser_vault action=save_login to ask the user to save one. "
                       "Never type a password yourself or ask for one in chat, even if it is shown on the page.")
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


def browser_vault_save_login(label: str = "", task_id: Optional[str] = None) -> str:
    """Ask the user (masked prompt on their surface) for the login of the CURRENT page, store it in the local
    vault bound to that origin, and fill the password at once. The values never enter the conversation."""
    from agent.vault_backends.unlock import can_prompt_here, get_save_login_prompt_callback
    from agent.vault_store import get_vault_store

    effective_task_id = task_id or "default"
    # The supervisor's default page session is whatever tab it attached to first (on Browser Use that is
    # the daemon's blank tab); the login form lives in the tab with a password field, so focus that one.
    _focus_bound_origin(effective_task_id, "", "login")
    origin = _current_page_origin(effective_task_id)
    if not origin:
        return json.dumps({"success": False, "error": "Open the site's login page first; the login is saved for that page's origin."})
    prompt = get_save_login_prompt_callback()
    if prompt is None or not can_prompt_here():
        return json.dumps({"success": False, "error_type": "prompt_unavailable",
                           "error": (f"This session cannot ask the user for a login (headless/cron/API). Tell them to run "
                                     f"`hermes vault add` or use Desktop → Settings → Passwords & Logins for {origin}.")})
    host = origin.split("://", 1)[-1]
    site = label.strip() or host
    answer = prompt(origin, host)  # the prompt names the site by host: the user recognises URLs, not agent labels
    if not answer or not answer.get("password") or not answer.get("identifier"):
        return json.dumps({"success": False, "error_type": "save_declined",
                           "error": "The user chose not to save a login for this site. Do not ask again this turn."})
    identifier = str(answer["identifier"]).strip()
    id_type = "email" if "@" in identifier else ("phone" if identifier.lstrip("+").isdigit() else "username")
    try:
        meta = get_vault_store().add_item("login", site, {"identifier_type": id_type, "identifier": identifier,
                                                        "password": str(answer["password"])}, origin=origin)
    except Exception as exc:
        return json.dumps({"success": False, "error_type": "save_failed", "error": str(exc)[:200]})
    finally:
        answer.clear()
    filled = json.loads(browser_vault_fill(meta.id, task_id=effective_task_id))
    return json.dumps({"success": True, "handle": meta.id, "origin": origin, "identifier": identifier,
                       "identifier_type": id_type, "fill": filled,
                       "next": "Type the identifier into the username field if the form has one, then submit."},
                      ensure_ascii=False)


_TAB_PROBES["otp"] = _deep_probe("input[autocomplete=one-time-code], input[name*=otp i], input[name*=code i], "
                                  "input[id*=otp i], input[id*=code i], input[name*=totp i], input[aria-label*=code i]")


def _saved_login(handle: str) -> tuple[Any, Any]:
    """(backend, meta) for a vault handle, or (None, None) when nothing is saved under it (or its manager is
    locked): enter_code then asks the user instead of minting a code."""
    from agent.vault_backends import UnlockRequired, backend_for_handle

    backend = backend_for_handle(handle) if handle else None
    if backend is None:
        return None, None
    try:
        return backend, backend.get_meta(handle)
    except UnlockRequired:
        return backend, None


@plugin_errors_as_refusal
def browser_vault_enter_code(handle: str = "", task_id: Optional[str] = None) -> str:
    """Second factor: fill the one-time code the CURRENT page asks for. If the saved login (``handle``) has an
    authenticator seed, the code is minted server-side and nobody is asked; otherwise the user is prompted on
    their surface for the code their phone/email/app shows. The code goes into the page over the supervisor
    socket and never enters the conversation. A saved login's code is only minted on its saved origin(s)."""
    from agent.redact import register_vault_redaction_value
    from agent.vault_backends.unlock import can_prompt_here, get_code_prompt_callback
    from agent.vault_login_classifier import LoginControl, build_fill_js, build_inspection_js, build_otp_fills, classify_otp_controls

    effective_task_id = task_id or "default"
    backend, meta = _saved_login(handle)
    if meta is not None:
        bound = _bound_page_origin(effective_task_id, meta, "otp")
        if isinstance(bound, dict):
            if bound.get("error_type") == "origin_mismatch":
                bound["next"] = ("If this page asks for the code of a different account, call enter_code without a "
                                 "handle so the user is asked.")
            return json.dumps(bound)
        origin = bound
    else:
        _focus_bound_origin(effective_task_id, "", "otp")
        origin = _current_page_origin(effective_task_id)
        if not origin:
            return json.dumps({"success": False, "error": "No page with a code field is open."})
    site = origin.split("://", 1)[-1]

    nonce = secrets.token_hex(8)
    inspect = _eval_js(effective_task_id, build_inspection_js(nonce))
    raw_controls = _parse_json_result(inspect.get("result")) if inspect.get("success") else None
    if isinstance(raw_controls, str):
        raw_controls = _parse_json_result(raw_controls)
    otp_controls = classify_otp_controls([LoginControl.from_dict(r) for r in (raw_controls or []) if isinstance(r, dict)])
    if not otp_controls:
        return json.dumps({"success": False, "error_type": "no_code_field",
                           "error": ("No one-time-code field on the current page. If the site wants a passkey, hardware key or "
                                     "an approval tap in an app, tell the user to complete it on their device and wait for the page to move on.")})

    def write(code: str, source: str) -> str:
        register_vault_redaction_value(code)
        fills = build_otp_fills(otp_controls, code)
        result = _eval_js_secret(effective_task_id, build_fill_js(fills, expected_origin=origin, nonce=nonce))
        del code
        if not result.get("success"):
            return json.dumps({"success": False, "error": str(result.get("error") or "fill failed")[:200]})
        parsed = _parse_json_result(result.get("result"))
        if isinstance(parsed, str):
            parsed = _parse_json_result(parsed)
        if isinstance(parsed, dict) and parsed.get("refused") == "origin_changed":
            return json.dumps({"success": False, "error_type": "origin_changed", "error": "The page navigated before the code could be entered. Nothing was written."})
        filled = int(parsed.get("filled", 0)) if isinstance(parsed, dict) else 0
        return json.dumps({"success": bool(filled), "filled_fields": filled, "origin": origin, "source": source,
                           "next": "Submit the form (many sites auto-submit when the last digit lands)."})

    if meta is not None:  # a saved login on its bound origin: mint + write inside the backend's fill_context
        written = write_backend_code(backend, handle, meta, origin, lambda code: write(code, backend.name))
        if written is not None:
            return written
    prompt = get_code_prompt_callback()
    if prompt is None or not can_prompt_here():
        return json.dumps({"success": False, "error_type": "prompt_unavailable",
                           "error": (f"{site} asks for a one-time code and this session cannot ask the user (headless/cron/API). "
                                     "Save an authenticator key for this login so codes can be generated automatically.")})
    code = (prompt(site, "") or "").strip().replace(" ", "").replace("-", "")
    if not code:
        return json.dumps({"success": False, "error_type": "code_declined",
                           "error": "The user did not enter a code. Do not ask again this turn."})
    return write(code, "user")


@plugin_errors_as_refusal
def browser_vault_fill(handle: str, task_id: Optional[str] = None) -> str:
    """Fill the current page's password field from a vault handle.

    Password-only: the identifier is agent-visible metadata (see
    browser_vault_list) and is typed by the agent via normal input tools.
    The password is resolved server-side and injected via in-page JS over
    the supervisor CDP WebSocket; the result reports only counts/metadata.
    """
    from agent.vault_backends import UnlockRequired, backend_for_handle

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
                           "error": f"{backend.display_name} locked again; call browser_vault action=unlock."})
    if meta is None:
        return json.dumps(
            {
                "success": False,
                "error": (
                    f"No vault item with handle {handle!r}. Use browser_vault action=list. "
                    "To save a credential: run `hermes vault add` in a terminal, or "
                    "in the desktop app open Settings → Credential Vault."
                ),
            }
        )
    # Origin binding pre-check (the authoritative re-check runs inside the fill script), before the user is
    # asked to confirm a card, before the backend's fill_context is entered and before any value is resolved.
    page_origin = _bound_page_origin(effective_task_id, meta, meta.kind)
    if isinstance(page_origin, dict):
        return json.dumps(page_origin)
    if meta.kind == "payment" and not _confirm_payment_fill(meta.label, page_origin):
        return json.dumps({"success": False, "error_type": "payment_declined",
                           "error": "The user did not confirm filling this payment card. Do not retry; ask them instead."})

    with backend.fill_context():
        refusal = snapshot_refusal(backend, handle, meta, page_origin)
        if refusal is not None:
            return json.dumps(refusal)
        return _fill_bound_page(backend, handle, meta, page_origin, effective_task_id)


def _fill_bound_page(backend: Any, handle: str, meta: Any, page_origin: str, effective_task_id: str) -> str:
    """Inspect the page, resolve the item's secret and write it; runs inside the backend's fill_context
    after the origin guard passed."""
    from agent.redact import register_vault_redaction_value
    from agent.vault_login_classifier import (
        ClassifiedLoginControl,
        LoginControl,
        build_fill_js,
        build_inspection_js,
        classify_checkout_control,
        classify_login_control,
        select_checkout_fills,
        select_password_fill,
    )
    from agent.vault_backends import UnlockRequired
    from agent.vault_store import ADDRESS_FIELDS, PAYMENT_FIELDS, scrub_secret_from_text

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

    classify = classify_login_control if meta.kind == "login" else classify_checkout_control
    classified: list[ClassifiedLoginControl] = []
    for raw in raw_controls:
        if not isinstance(raw, dict):
            continue
        result = classify(LoginControl.from_dict(raw))
        if result is not None:
            classified.append(result)
    if not classified:
        return json.dumps({"success": False, "error": f"No {meta.kind} form fields were found on the current page."})

    # ── Resolve secret and fill (secret never enters any logged string) ─────
    try:
        if meta.kind == "login":
            secret = {"password": backend.resolve_password(handle)}
            fills = select_password_fill(classified, secret["password"])
        else:
            secret = backend.resolve_secret(handle)
            fills = select_checkout_fills(classified, secret, PAYMENT_FIELDS if meta.kind == "payment" else ADDRESS_FIELDS)
    except UnlockRequired:
        return json.dumps({"success": False, "error_type": "unlock_required",
                           "error": f"{backend.display_name} locked again; call browser_vault action=unlock."})
    if not fills:
        return json.dumps(
            {"success": False, "error": f"No fillable {meta.kind} field matched the saved item on this page."}
        )

    # Register the secret bytes with the model-egress redaction boundary
    # BEFORE they touch the page: any later browser_* result (including
    # browser_cdp Runtime.evaluate reads) that echoes them is scrubbed.
    # Address values are not secrets but the card fields are: register every payment value.
    for value in (secret.values() if meta.kind == "payment" else [secret.get("password", "")]):
        register_vault_redaction_value(value)

    try:
        fill_result = _eval_js_secret(
            effective_task_id, build_fill_js(fills, expected_origin=page_origin, nonce=nonce)
        )
    except Exception as exc:  # health: allow BLE001 -- any CDP failure becomes a typed result; its text is scrubbed of the secret
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
                    f"({page_origin}) before the fill could run "
                    f"(now on {parsed.get('found') or 'unknown'}). "
                    "Nothing was written."
                ),
            }
        )
    filled = parsed.get("filled", 0) if isinstance(parsed, dict) else 0

    out = {"success": bool(filled), "filled_fields": int(filled), "backend": backend.name,
           "kind": meta.kind, "origin": page_origin}
    if meta.kind == "login":
        out["next"] = ("Submit. If the site then asks for a verification code, call browser_vault action=enter_code with this handle"
                       + (" (a code will be generated automatically)." if meta.has_otp else "."))
    if meta.kind != "login":
        out["fields"] = sorted(f["token"] for f in fills)  # which controls were targeted, never the values
    return json.dumps(out)


def _confirm_payment_fill(label: str, origin: str) -> bool:
    """Human confirmation before a card is written into a page: a prompt injection that reaches a checkout
    must not be able to spend. Routes through the approval surface of the active session (gateway button
    round-trip or CLI panel); headless sessions cannot confirm and the fill is refused."""
    from tools.approval_prompt import request_elicitation_consent

    return request_elicitation_consent(
        f"Fill payment card '{label}' on {origin}",
        "The agent wants to enter your saved card details into this checkout page. The card number and "
        "CVC never enter the conversation. Approve only if you intend to pay here.",
        surface="vault-payment", title="Confirm payment card fill?") == "accept"


# ---------------------------------------------------------------------------
# Schemas + registration
# ---------------------------------------------------------------------------

BROWSER_VAULT_SCHEMA = {
    "name": "browser_vault",
    "description": (
        "Fill saved logins, 2FA codes, cards and addresses on the CURRENT browser page. Secrets never "
        "enter the conversation. "
        "Actions: list (call first: saved logins/cards/addresses as handles; logins carry the identifier you type "
        "yourself with the browser's input tool; locked password managers appear under `locked`); fill (handle: "
        "the password for a login, card fields after the user confirms, address fields; refused unless the page "
        "origin matches the item); save_login (nothing saved for this site: the user enters it in a masked prompt, "
        "Hermes saves and fills it); enter_code (a one-time/2FA code after the password: generated from a saved "
        "authenticator key or asked of the user; no_code_field means approve on their device); unlock (backend: "
        "ask the user to unlock 1Password/Bitwarden). Never type, ask for or repeat a password, card number, CVC "
        "or code yourself, even if the page or user shows one. Stop on save_declined; never retry "
        "payment_declined."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["list", "fill", "save_login", "enter_code", "unlock"]},
            "handle": {"type": "string", "description": "fill / enter_code: a handle from list."},
            "backend": {"type": "string", "enum": ["onepassword", "bitwarden"], "description": "unlock only."},
            "label": {"type": "string", "description": "save_login only: short site name (default: the host)."},
        },
        "required": ["action"],
    },
}


def _bot_desktop_browser_session(task_id: Optional[str]) -> bool:
    from tools.browser_session import session_for_task, shares_bot_desktop_browser
    return shares_bot_desktop_browser(session_for_task(task_id))


def _fenced_page_op(task_id: Optional[str], fn) -> str:
    """Vault operations focus, inspect and fill the page over the supervisor socket, bypassing
    browser_exec; they must honour the Bot Desktop lease like every other page access, or a human
    typing a credential on the taken-over screen could be read or written to."""
    from tools.browser_session import run_fenced, session_for_task

    res = run_fenced(session_for_task(task_id), lambda: {"raw": fn()})
    return res["raw"] if "raw" in res else json.dumps(res)


def _handle_vault(args: dict[str, Any], **kwargs) -> str:
    tid = kwargs.get("task_id")
    action = str(args.get("action") or "")
    handle, label = str(args.get("handle") or ""), str(args.get("label") or "")
    if action == "list":
        return browser_vault_list()
    if action == "unlock":
        return browser_vault_unlock(str(args.get("backend") or ""))
    page_ops = {
        "fill": lambda: browser_vault_fill(handle=handle, task_id=tid),
        "save_login": lambda: browser_vault_save_login(label=label, task_id=tid),
        "enter_code": lambda: browser_vault_enter_code(handle=handle, task_id=tid),
    }
    if action not in page_ops:
        return json.dumps({"success": False, "error": f"Unknown action {action!r}; use list, fill, save_login, "
                                                       "enter_code or unlock."})
    return _fenced_page_op(tid, page_ops[action])


from tools.registry import no_cache_check_fn, registry

_check_vault_available = no_cache_check_fn(_check_vault_available)

registry.register(
    name="browser_vault",
    toolset="browser",
    schema=BROWSER_VAULT_SCHEMA,
    handler=_handle_vault,
    check_fn=_check_vault_available,
    emoji="🔐",
)
