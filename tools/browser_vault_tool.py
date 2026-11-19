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
import time
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Dict, Iterator, Optional

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

# Read-back pacing. A framework commits on its own schedule, so one read is a guess: 500 ms (~30
# frames) is where a real commit has happened or is not coming. Polling beats a single sleep because a
# mask or a late revert can write the value twice.
# The read-back is a sample, so a revert shorter than the gap between two samples is not seen. 10 ms
# is under one frame; on a remote browser the CDP round trip is the real gap. A watcher inside the page
# would not close it: an input listener never fires on a programmatic revert, MutationObserver does not
# see the value property, a value-setter hook fights React's own tracker on that property, and a script
# that waits needs a backend that awaits promises (a plugin page backend's evaluate does not promise
# to) and a page whose timers run (a background tab's run about once a second).
_FILL_VERIFY_INTERVAL_S = 0.01
_FILL_VERIFY_CEILING_S = 0.5
# A value counts as kept only once every field has read back right for this long without a miss:
# a framework can show the value for a frame and revert it on its next commit.
_FILL_VERIFY_STABLE_S = 0.2

# The CDP page session the running fill validated. Every script of that fill goes to it: the
# inspection, the write, each focus check, the read-back that carries the expected values, and the
# cleanup. The supervisor refuses once it has moved to another session (focus_page, a reconnect), so
# no script of the fill runs on a page the fill never inspected. None = no fill is running.
_fill_session: ContextVar[Optional[str]] = ContextVar("vault_fill_session", default=None)

# Per-key pacing for a masked field. A mask processes keystrokes as they arrive, so a value typed
# one character at a time (60 ms apart, roughly a fast human) survives where a bulk insert loses
# characters. Only used after a bulk insert read back as a mismatch.
_TYPE_PER_KEY_DELAY_S = 0.06


def _page_backend(task_id: str):
    """The page this operation runs on when it is not the task's CDP session: the Desktop in-app browser
    page pinned for a ``target='preview'`` operation, else the plugin browser holding this task's live page
    (e.g. Camofox), else None."""
    from agent.browser_page_registry import backend_for_task
    from tools.browser_vault_preview import active_page
    return active_page() or backend_for_task(task_id)


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


@dataclass(frozen=True)
class _CheckedPage:
    """A page origin and the CDP page session it was checked on. ``session`` is None when there is
    no session to check: a plugin page backend owns its page, and without a supervisor there is no CDP."""

    origin: str
    session: Optional[str] = None


@contextmanager
def _pinned_page_session(task_id: str, checked: Optional[str]) -> Iterator[Optional[dict[str, Any]]]:
    """Pin every eval inside this block to ``checked``, the page session the origin check ran on.

    The pin is that session, not the one the supervisor holds when the block is entered: a
    reconnect or a ``focus_page`` in between (a payment prompt can last minutes) moves the
    supervisor to a tab the origin check never looked at. A plugin page backend owns its page and
    has no CDP session, so there is nothing to pin.
    Yields the refusal to return when the supervisor no longer holds the checked session (it moved
    or is reconnecting, or it is gone), or when there was a supervisor but no session to check.
    """
    supervisor = None if _page_backend(task_id) is not None else _ensure_supervisor(task_id)
    attached = supervisor.current_page_session() if supervisor is not None else None
    cdp = supervisor is not None or checked is not None
    refusal = None
    if cdp and (checked is None or attached != checked):
        refusal = {"success": False, "error_type": "supervisor_required",
                   "error": "The browser moved to another tab or is reconnecting since the page origin "
                            "was checked. Nothing was written; retry in a moment."}
    token = _fill_session.set(checked)
    try:
        yield refusal
    finally:
        _fill_session.reset(token)


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
    sup = supervisor.evaluate_runtime(expression, session_id=_fill_session.get())
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
    from tools.browser_vault_preview import active_page
    if active_page() is None and _bot_desktop_browser_session(task_id):
        from tools.bot_desktop import lease as _bd_lease
        try:
            _bd_lease.assert_agent_may_act()
        except _bd_lease.HumanHasControl as exc:
            return {"success": False, "error_type": "human_has_control", "error": str(exc)}

    if page is not None:
        return _page_eval(page, task_id, expression, secret=True)
    try:
        sup = supervisor.evaluate_runtime(expression, session_id=_fill_session.get())
    except Exception as exc:  # health: allow BLE001 -- its text can quote the expression, i.e. the secret
        logger.debug("vault fill: secret-bearing evaluate raised %s", type(exc).__name__)
        return {"success": False, "error_type": "eval_failed",
                "error": "The browser could not run the vault script."}
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


def _focus_bound_origin(task_id: str, origin: str, kind: str) -> Optional[_CheckedPage]:
    """Point the supervisor's page session at the open tab on ``origin`` that holds a ``kind`` form
    (browser_exec sessions open their own tabs, so the tab the supervisor attached to first is rarely the
    login page). Returns the origin (its URL for ``origin=""``) and the session ``focus_page`` attached
    when a tab was focused, else None (caller falls back to the current page).
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
    if not focused.get("ok"):
        return None
    return _CheckedPage(origin or str(focused.get("url") or ""), focused.get("session_id"))


def _current_page_check(task_id: str) -> Optional[_CheckedPage]:
    """The current page's origin, read on the session the supervisor holds now and returned with it
    (the read is pinned to it, so a reconnect during the read cannot answer for another tab)."""
    supervisor = None if _page_backend(task_id) is not None else _ensure_supervisor(task_id)
    session = supervisor.current_page_session() if supervisor is not None else None
    token = _fill_session.set(session)
    try:
        origin = _current_page_origin(task_id)
    finally:
        _fill_session.reset(token)
    return _CheckedPage(origin, session) if origin else None


def _bound_page_origin(task_id: str, meta: Any, probe_kind: str) -> _CheckedPage | dict[str, Any]:
    """The origin guard every fill of a stored secret (password, one-time code, card, address) passes
    BEFORE the backend resolves any value: the checked page (origin + the session it was checked on)
    when its origin is one of the item's saved origins, else the refusal to return. Manager items may
    bind several websites; matching is exact and nothing wildcard or parent-domain is inferred. The
    caller pins the checked session and hands the origin to ``build_fill_js``, which re-asserts it
    inside the page immediately before the write (navigation between check and fill)."""
    allowed = saved_origins(meta)
    if not allowed:
        return {"success": False, "error_type": "no_origin",
                "error": f"Vault item {meta.id!r} has no bound origin; {meta.kind} items are filled only on "
                         "the site they were saved for."}
    page = next((p for p in (_focus_bound_origin(task_id, c, probe_kind) for c in allowed) if p), None)
    page = page or _current_page_check(task_id)
    if page is None:
        return {"success": False,
                "error": "Could not determine the current page origin. Navigate to the login page first."}
    if page.origin not in allowed:
        return {"success": False, "error_type": "origin_mismatch",
                "error": (f"Refused: current page origin ({page.origin}) does not match the vault item's bound "
                          f"origin(s) ({', '.join(allowed)}). Vault fills only run on the exact origin(s) the "
                          "credential was saved for.")}
    return page


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
    effective_task_id = task_id or "default"
    backend, meta = _saved_login(handle)
    if meta is not None:
        page = _bound_page_origin(effective_task_id, meta, "otp")
        if isinstance(page, dict):
            if page.get("error_type") == "origin_mismatch":
                page["next"] = ("If this page asks for the code of a different account, call enter_code without a "
                                "handle so the user is asked.")
            return json.dumps(page)
    else:
        _focus_bound_origin(effective_task_id, "", "otp")
        page = _current_page_check(effective_task_id)
        if page is None:
            return json.dumps({"success": False, "error": "No page with a code field is open."})
    with _pinned_page_session(effective_task_id, page.session) as unpinned:
        if unpinned is not None:
            return json.dumps(unpinned)
        return _enter_code_on_page(backend, handle, meta, page.origin, effective_task_id)


def _enter_code_on_page(backend: Any, handle: str, meta: Any, origin: str, effective_task_id: str) -> str:
    """Inspect the page for its code field and write the code (minted, else asked of the user)."""
    from agent.redact import register_vault_redaction_value
    from agent.vault_backends.unlock import can_prompt_here, get_code_prompt_callback
    from agent.vault_login_classifier import LoginControl, build_fill_js, build_inspection_js, build_otp_fills, classify_otp_controls

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
    if prompt is None or not can_prompt_here(prompt):
        return json.dumps({"success": False, "error_type": "prompt_unavailable",
                           "error": (f"{site} asks for a one-time code and this session cannot ask the user (headless/cron/API). "
                                     "Save an authenticator key for this login so codes can be generated automatically.")})
    code = (prompt(site, "") or "").strip().replace(" ", "").replace("-", "")
    if not code:
        return json.dumps({"success": False, "error_type": "code_declined",
                           "error": "The user did not enter a code. Do not ask again this turn."})
    return write(code, "user")


@plugin_errors_as_refusal
def browser_vault_fill(handle: str, task_id: Optional[str] = None, *, with_identifier: bool = True) -> str:
    """Fill the current page's login from a vault handle: the identifier and the password.

    Both fields go through the same path, so both are typed as trusted browser input and read
    back. That matters on a framework-controlled form: a normal input tool writes the value into
    the DOM without React/Vue/MUI accepting it, and the page then re-renders the field empty.
    Pass ``with_identifier=False`` to write the password only.
    The password is resolved server-side and never enters a result; the result reports only
    counts and metadata.
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
    page = _bound_page_origin(effective_task_id, meta, meta.kind)
    if isinstance(page, dict):
        return json.dumps(page)
    if meta.kind == "payment" and not _confirm_payment_fill(meta.label, page.origin):
        return json.dumps({"success": False, "error_type": "payment_declined",
                           "error": "The user did not confirm filling this payment card. Do not retry; ask them instead."})

    with backend.fill_context(), _pinned_page_session(effective_task_id, page.session) as unpinned:
        refusal = unpinned or snapshot_refusal(backend, handle, meta, page.origin)
        if refusal is not None:
            return json.dumps(refusal)
        return _fill_bound_page(backend, handle, meta, page.origin, effective_task_id, with_identifier)


def _identifier_fill(classified: list, raw_controls: list[dict[str, Any]],
                     password_fill: list[dict[str, Any]], identifier: str) -> list[dict[str, Any]]:
    """The username/email/phone control for a user-chosen login.

    Prefers a classified control in the password field's form; on an identifier-only step (no
    password field yet) it is the whole fill. When nothing was classified, falls back to the single
    other text box in the password's form — a phone entry on a framework form often carries no
    autocomplete, name, id or placeholder at all, so there is nothing to score.
    """
    if not identifier:
        return []
    anchor = password_fill[0]["index"] if password_fill else None
    form = next((c.control.form_index for c in classified if c.control.index == anchor), None)
    if form is None:
        raw_anchor = next((r for r in raw_controls if int(r.get("index", -1) or -1) == anchor), None)
        form = raw_anchor.get("formIndex", raw_anchor.get("form_index")) if raw_anchor else None

    fields = [c for c in classified if c.token in ("username", "email", "tel")]
    if fields:
        best = min(fields, key=lambda c: (c.control.form_index != form, -c.score, c.control.index))
        return [{"index": best.control.index, "token": best.token, "value": identifier}]

    from agent.vault_login_classifier import LoginControl

    taken = {f["index"] for f in password_fill}
    candidates = []
    for raw in raw_controls:
        if not isinstance(raw, dict):
            continue
        try:
            control = LoginControl.from_dict(raw)
        except (TypeError, ValueError):
            continue
        if control.index in taken or control.type not in ("text", "tel", "email"):
            continue
        if control.form_index != form:
            continue
        candidates.append(control)
    if len(candidates) != 1:  # ambiguous: fill nothing rather than guess
        return []
    return [{"index": candidates[0].index, "token": "username", "value": identifier}]


def _trusted_typing_fill(supervisor: Any, task_id: str, type_fills: list[dict[str, Any]],
                         js_fills: list[dict[str, Any]], page_origin: str, nonce: str,
                         secret: dict[str, Any]) -> dict[str, Any]:
    """Fill the text/password fields with CDP TRUSTED typing (``Input.insertText``).

    A framework-controlled input (React/Vue/MUI) accepts a trusted input event and rejects a
    JS value write: the page sees the same ``input`` event a human keystroke produces.
    ``Input.insertText`` types into whatever element holds focus, so the focus is asserted
    immediately before every insert (``build_focus_js``) — once per field, and before every key of
    a key-by-key retype — and every insert is pinned to the page session the fill started on; the
    value is read back after it
    (``build_fill_verify_js``). A focus that drifted, or a value the page reverted, aborts the
    whole fill: nothing further is typed and every inspection stamp is removed.

    ``select`` / checkbox / radio controls keep the JS path (``build_fill_js``): they are not
    text inputs and trusted typing cannot address them.

    Returns ``{"ok": True, "filled": n, "landed": n, "reverted": n}`` or
    ``{"ok": False, "success": False, "error_type": ..., "error": ...}``.
    """
    from agent.vault_login_classifier import (
        build_clear_js,
        build_fill_js,
        build_fill_verify_js,
        build_focus_js,
        build_origin_check_js,
        build_unstamp_js,
    )
    from agent.vault_store import scrub_secret_from_text

    def _abort(error_type: str, message: str) -> dict[str, Any]:
        # Remove every stamp: a stamp left behind would let a later read address the control
        # this fill just targeted.
        _eval_js_secret(task_id, build_unstamp_js(nonce))
        return {"ok": False, "success": False, "error_type": error_type, "filled_fields": 0,
                "error": scrub_secret_from_text(message, secret)}

    def _script(expression: str) -> Any:
        res = _eval_js_secret(task_id, expression)
        if not res.get("success"):
            return None
        parsed = _parse_json_result(res.get("result"))
        if isinstance(parsed, str):
            parsed = _parse_json_result(parsed)
        return parsed

    # Every keystroke goes to the page session this fill validated, the one its focus checks ran on.
    # If another tool moves the supervisor to a different page session (focus_page, a reconnect), the
    # next keystroke is refused rather than typed into a page the focus check never looked at.
    pinned = _fill_session.get()
    if not pinned:
        return _abort("fill_unverified", "The browser has no attached page to type into; nothing was written.")

    def _insert(text: str) -> Optional[dict[str, Any]]:
        typed = supervisor.insert_text(text, session_id=pinned)
        if typed.get("ok"):
            return None
        return _abort("fill_unverified",
                      "The browser refused the trusted input for one field: "
                      f"{scrub_secret_from_text(str(typed.get('error') or 'input failed'), secret)}")

    # The page that holds the stamps must still be the bound origin: the supervisor's tab can
    # have navigated between the inspection and the first keystroke.
    origin = _script(build_origin_check_js(page_origin))
    if not isinstance(origin, dict) or origin.get("refused") == "origin_changed":
        found = origin.get("found") if isinstance(origin, dict) else None
        return _abort("origin_changed",
                      f"Refused: the page navigated away from the bound origin ({page_origin}) "
                      f"before the value could be typed (now on {found or 'unknown'}). Nothing was written.")

    filled = 0
    # Non-text controls (select, checkbox, radio) keep the JS path.
    if js_fills:
        js_result = _script(build_fill_js(js_fills, expected_origin=page_origin, nonce=nonce))
        if not isinstance(js_result, dict):
            return _abort("fill_unverified", "The page could not be written to; nothing was filled.")
        if js_result.get("refused") == "origin_changed":
            return _abort("origin_changed",
                          f"Refused: the page navigated away from the bound origin ({page_origin}) "
                          f"before the fill could run (now on {js_result.get('found') or 'unknown'}). "
                          "Nothing was written.")
        filled += int(js_result.get("filled", 0) or 0)

    def _type_field(f: dict[str, Any]) -> Optional[dict[str, Any]]:
        """Type one field, read it back, and retype key by key when a mask mangled it.

        Returns ``None`` when the field holds exactly the value we typed, or a finished refusal.
        """
        def _verify() -> Any:
            return _script(build_fill_verify_js([f], nonce=nonce, expected_origin=page_origin, unstamp=False))

        def _holds(checked: Any) -> bool:
            return (isinstance(checked, dict)
                    and int(checked.get("landed", 0) or 0) == 1
                    and not int(checked.get("reverted", 0) or 0))

        def _assert_focus(*, refocus: bool = True) -> Optional[dict[str, Any]]:
            focus = _script(build_focus_js([f], nonce, expected_origin=page_origin, refocus=refocus))
            if isinstance(focus, dict) and focus.get("focused") == f["index"]:
                return None
            if isinstance(focus, dict) and focus.get("refused") == "origin_changed":
                return _abort("origin_changed",
                              f"Refused: the page navigated away from the bound origin ({page_origin}) "
                              f"while the value was being typed (now on {focus.get('found') or 'unknown'}). "
                              "Nothing further was typed.")
            tag = focus.get("tag") if isinstance(focus, dict) else None
            found = focus.get("found") if isinstance(focus, dict) else None
            return _abort("focus_lost",
                          f"Refused: focus moved off the target field (tag {tag or 'unknown'}, "
                          f"focus now on {found or 'nothing'}) before the value could be typed. "
                          "Nothing further was typed.")

        aborted = _assert_focus() or _insert(f["value"])
        if aborted:
            return aborted
        verified = _verify()
        if not isinstance(verified, dict):
            return _abort("fill_unverified",
                          "The page could not be read back after the value was typed; "
                          "treating the fill as failed.")
        if _holds(verified):
            return None

        # A bulk insert lands wrong in a masked field (phone, date, card): the mask drops or
        # reformats characters as they stream in. Clear it and type the value one key at a time,
        # the way a person does, then read it back again.
        # Focus is asserted before EVERY key: a page can move it after the first one, and the rest
        # of the value would land in whatever took it. Only the first key may (re)focus the field.
        _script(build_clear_js([f], nonce, expected_origin=page_origin))
        for position, char in enumerate(str(f["value"])):
            aborted = _assert_focus(refocus=position == 0) or _insert(char)
            if aborted:
                return aborted
            time.sleep(_TYPE_PER_KEY_DELAY_S)
        verified = _verify()
        if _holds(verified):
            return None
        return _abort("fill_reverted",
                      "The page rejected the value: the field did not hold it after it was "
                      "typed key by key. Nothing is filled.")

    for f in type_fills:
        aborted = _type_field(f)
        if aborted:
            return aborted

    return {"ok": True, "filled": filled + len(type_fills)}


def _typing_channel(task_id: str) -> tuple[Any, Any, Any]:
    """The trusted-typing channel for a task: ``(supervisor, insert_text, page_backend)``.

    A plugin or preview page backend owns the page and has no CDP channel: asking the supervisor for
    one would attach a managed browser to a preview-only session. When no supervisor is reachable the
    first two are None and the caller stays on the JS path.
    """
    page_backend = _page_backend(task_id)
    if page_backend is not None:
        return None, None, page_backend
    try:
        supervisor = _ensure_supervisor(task_id)
    except Exception as exc:  # health: allow BLE001 -- a missing supervisor is not a fill failure
        logger.debug("vault fill: supervisor unavailable for trusted typing (%s)", exc)
        return None, None, None
    return supervisor, getattr(supervisor, "insert_text", None), None


def _js_write(task_id: str, fills: list[dict[str, Any]], page_origin: str, nonce: str,
              secret: dict[str, str]) -> dict[str, Any]:
    """Write the fields with the JS fill script (no CDP channel: page/plugin backends).

    Returns ``{"success": True}`` or a refusal dict ready to be serialised: the caller does not have
    to interpret anything. Every error string is scrubbed of the secret before it leaves.
    """
    from agent.vault_login_classifier import build_fill_js
    from agent.vault_store import scrub_secret_from_text

    fill_result = _eval_js_secret(task_id, build_fill_js(fills, expected_origin=page_origin, nonce=nonce))
    if not fill_result.get("success"):
        out = {"success": False, "error": scrub_secret_from_text(str(fill_result.get("error") or "fill failed"), secret)}
        if fill_result.get("error_type"):
            out["error_type"] = fill_result["error_type"]
        return out
    parsed = _parse_json_result(fill_result.get("result"))
    if isinstance(parsed, str):
        parsed = _parse_json_result(parsed)
    if isinstance(parsed, dict) and parsed.get("refused") == "origin_changed":
        return {"success": False, "error_type": "origin_changed",
                "error": ("Refused: the page navigated away from the bound origin "
                          f"({page_origin}) before the fill could run "
                          f"(now on {parsed.get('found') or 'unknown'}). Nothing was written.")}
    wrote = int(parsed.get("filled", 0) or 0) if isinstance(parsed, dict) else 0
    if not wrote:
        return {"success": False, "error_type": "fill_not_written", "filled_fields": 0,
                "error": "No field accepted the write: nothing was filled on this page."}
    return {"success": True}


def _read_back(task_id: str, fills: list[dict[str, Any]], nonce: str, page_origin: str) -> dict[str, Any]:
    """Poll the filled controls until the page has kept every value for ``_FILL_VERIFY_STABLE_S``.

    A miss resets the window; a fill that has not landed in full by ``_FILL_VERIFY_CEILING_S`` is
    reported as reverted, so the longest wait is the ceiling plus one stability window.

    The fill's own count is a claim about the write; this is the claim about the page. A controlled
    input (React/Vue/MUI) reverts a value it did not accept, and a re-rendered field loses its stamp.
    Polling rather than sleeping once: a mask reformat or a late revert writes twice, and the second
    write is the one that matters. Counts only — no value leaves the page. Each poll re-asserts
    ``page_origin`` in the page: a login that submits on input can land on the next page mid-poll.
    """
    from agent.vault_login_classifier import build_fill_verify_js, build_unstamp_js

    expected = len(fills)
    started = time.monotonic()
    held_since: Optional[float] = None
    landed = 0
    reverted = 0
    while True:
        time.sleep(_FILL_VERIFY_INTERVAL_S)
        verify = _eval_js_secret(task_id, build_fill_verify_js(fills, nonce=nonce, expected_origin=page_origin,
                                                               unstamp=False))
        verified = _parse_json_result(verify.get("result")) if verify.get("success") else None
        if isinstance(verified, str):
            verified = _parse_json_result(verified)
        if isinstance(verified, dict) and verified.get("refused") == "origin_changed":
            return {"success": False, "error_type": "origin_changed", "filled_fields": 0,
                    "error": (f"The page left the bound origin ({page_origin}) before the fill could be read "
                              f"back (now on {verified.get('found') or 'unknown'}); treating it as failed.")}
        if not isinstance(verified, dict):
            _eval_js(task_id, build_unstamp_js())
            return {"success": False, "error_type": "fill_unverified", "filled_fields": 0,
                    "error": "The page could not be read back; treating the fill as failed rather than claiming it landed."}
        landed = int(verified.get("landed", 0) or 0)
        reverted = int(verified.get("reverted", 0) or 0)
        now = time.monotonic()
        if landed < expected:
            held_since = None
            if now - started >= _FILL_VERIFY_CEILING_S:
                break
            continue
        held_since = now if held_since is None else held_since
        if now - held_since >= _FILL_VERIFY_STABLE_S:
            break
    _eval_js(task_id, build_unstamp_js())  # stamps were kept across the polls; drop them once
    if landed < expected:
        return {"success": False, "error_type": "fill_reverted", "filled_fields": 0,
                "error": (f"The page did not keep the value: {reverted} of {expected} field(s) reverted "
                          "or were re-rendered. Nothing is filled.")}
    return {"success": True, "landed": landed}


def _fill_bound_page(backend: Any, handle: str, meta: Any, page_origin: str, effective_task_id: str,
                     with_identifier: bool = False) -> str:
    """Inspect the page, resolve the item's secret and write it; runs inside the backend's fill_context
    after the origin guard passed. ``with_identifier`` also writes the login's (non-secret) identifier."""
    from agent.redact import register_vault_redaction_value
    from agent.vault_login_classifier import (
        ClassifiedLoginControl,
        LoginControl,
        build_fill_js,
        build_fill_verify_js,
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
            if with_identifier:
                fills += _identifier_fill(classified, raw_controls, fills, str(meta.identifier or ""))
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

    # Trusted typing (CDP ``Input.insertText``) is preferred when the task's CDP supervisor is
    # available: a controlled input (React/Vue/MUI) accepts a trusted input event and rejects a
    # JS value write. ``Input.insertText`` types into whatever holds focus, so the focus is
    # asserted immediately before every single keystroke (build_focus_js) and the value is read
    # back after it. A page backend (plugin browser) has no CDP channel: that path stays on JS.
    supervisor, insert_text, page_backend = _typing_channel(effective_task_id)
    # The trusted path types into the SUPERVISOR's page: only use it when that page is the page the
    # inspection stamped (no plugin page backend in front of it).
    trusted = bool(callable(insert_text)) and page_backend is None
    type_by_index = {c.control.index: c.control.type for c in classified}
    type_fills = [f for f in fills
                  if f["index"] in type_by_index
                  and type_by_index[f["index"]] not in ("select", "checkbox", "radio")]
    js_fills = [f for f in fills if f not in type_fills]

    if trusted and type_fills:
        typed = _trusted_typing_fill(supervisor, effective_task_id, type_fills, js_fills,
                                     page_origin, nonce, secret)
        if not typed.get("ok"):
            return json.dumps({k: v for k, v in typed.items() if k != "ok"})
    else:
        written = _js_write(effective_task_id, fills, page_origin, nonce, secret)
        if not written.get("success"):
            return json.dumps(written)

    checked = _read_back(effective_task_id, fills, nonce, page_origin)
    if not checked.get("success"):
        return json.dumps(checked)
    landed = int(checked["landed"])


    out = {"success": True, "filled_fields": landed, "backend": backend.name,
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
        "Actions: list (call first: saved logins/cards/addresses as handles; locked password managers appear "
        "under `locked`); fill (handle: a login writes the identifier and the password, card fields after the "
        "user confirms, address fields; refused unless the page "
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
            "target": {"type": "string", "enum": ["browser", "preview"],
                       "description": "fill / save_login / enter_code: 'preview' = the Desktop in-app browser "
                                      "page beside the chat, 'browser' = the managed automation browser."},
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
    target = args.get("target") or "browser"
    if target == "preview":
        # The in-app browser is not the Bot Desktop browser; its window owns the page and refuses a
        # changed tab or document itself. Never retried on another browser.
        from tools.browser_vault_preview import run_preview_vault
        return run_preview_vault(kwargs.get("preview_callback"), page_ops[action])
    if target != "browser":
        return json.dumps({"success": False, "error_type": "invalid_target",
                           "error": "target must be 'browser' or 'preview'."})
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
