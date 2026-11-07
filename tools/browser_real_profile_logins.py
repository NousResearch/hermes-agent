"""Read the real-profile browser's saved logins for the vault's read-only ``browser:`` backend.

NO decryption happens here. Two decryption-free reads, both from the already-running real-profile
browser Hermes owns (``tools.browser_session_real_profile.real_profile_cdp``), the same mechanism
PR #150283 used for cookies:

- **list metadata** — ``origin_url`` and ``username_value`` are PLAINTEXT columns in the profile's
  ``Login Data`` SQLite (only ``password_value`` is encrypted); reading them is not decryption, the
  same category as reading a cookie's name/domain. They come from the Hermes-owned snapshot copy the
  real-profile consent already makes (``real_profile_copy_dir``), opened read-only.
- **resolve a password** — Chrome's DevTools protocol exposes NO command that reads a saved password
  (``Autofill.trigger`` is cards/addresses only). The reachable mechanism is the browser's own
  autofill: over CDP we open the item's saved login URL in the real-profile browser, dispatch a click
  gesture on the sign-in form, and Chrome's own PasswordAutofillAgent fills the password into the DOM
  (the browser decrypts its OWN store with the OS keychain). We read that value back over
  ``Runtime.evaluate``. The agent never sees it: the fill tool registers it with the redaction
  boundary before it touches a page, exactly like every other backend's ``resolve_password``.

Chrome autofills only on an EXACT-origin page and only after a user-style gesture, so the password is
resolved against the real-profile browser on the credential's true origin, never a page we fabricate.
Two safety checks happen in the same read that pulls the value back, before the caller ever sees it:
the final page's origin must equal the handle's saved origin (a root that redirects to another site
with a saved login must not leak THAT site's password), and, when the handle carries a username, the
autofilled username must match it (a site with several saved accounts autofills its preferred one).
"""

from __future__ import annotations

import contextlib
import itertools
import json
import logging
import sqlite3
import time
from pathlib import Path
from typing import Any, Iterator, Optional

logger = logging.getLogger(__name__)

_CDP_TIMEOUT_S = 20.0
# Bounded wait for a usable sign-in form to render: some pages mount their form a second or two
# after navigation (SPA hydration, late JS), so a single fixed sleep + one inspection misses them.
_FORM_WAIT_S = 6.0
# Bounded wait for Chrome's PasswordAutofillAgent to fill the field after the trusted gesture.
_AUTOFILL_WAIT_S = 6.0
# Poll cadence while waiting for the form / autofill.
_POLL_INTERVAL_S = 0.15


# ── list: plaintext metadata from the snapshot's Login Data (no decryption) ───────────────────────

def _login_data_paths(copy_dir: str) -> list[Path]:
    """The snapshot's login stores. Chrome keeps profile-scoped logins in ``Login Data`` and, when
    signed into a Google account, account-scoped ones in ``Login Data For Account``; both carry
    plaintext origin/username. The snapshot copies both into the copy's ``Default``."""
    base = Path(copy_dir) / "Default"
    return [p for p in (base / "Login Data", base / "Login Data For Account") if p.is_file()]


def login_data_present(copy_dir: str) -> bool:
    """True when the snapshot has at least one ``Login Data`` store to read. Used by the listing path
    to decide whether a snapshot still needs preparing before the source can show anything."""
    return bool(_login_data_paths(copy_dir))


def read_login_metadata(copy_dir: str) -> list[dict[str, str]]:
    """``[{origin_url, username}, …]`` for every non-blocklisted saved login, deduped on (origin_url,
    username). ``origin_url`` is the full saved sign-in URL (path preserved — a login saved at
    ``https://example.com/account/login`` resolves there, not at the bare root). Reads plaintext
    columns only; never the encrypted ``password_value``."""
    seen: set[tuple[str, str]] = set()
    out: list[dict[str, str]] = []
    for db in _login_data_paths(copy_dir):
        try:
            # mode=ro WITHOUT immutable=1: the real browser mutating its own store commits changes
            # into a WAL file first, and immutable=1 tells SQLite to ignore the WAL, so logins saved
            # or changed this session would read back stale/missing. Read-only still prevents writes.
            # Build the file: URI from the path with pathlib so a '#', '?' or space in any directory
            # component is percent-escaped — a bare f"file:{db}" truncates at the first '#' and the
            # query string, silently opening the wrong (or an empty) database.
            con = sqlite3.connect(f"{db.as_uri()}?mode=ro", uri=True)
        except sqlite3.Error as exc:
            logger.debug("real-profile logins: cannot open %s: %s", db, exc)
            continue
        try:
            rows = con.execute(
                "SELECT origin_url, username_value FROM logins "
                "WHERE blacklisted_by_user = 0 AND origin_url <> ''"
            ).fetchall()
        except sqlite3.Error as exc:
            logger.debug("real-profile logins: cannot query %s: %s", db, exc)
            rows = []
        finally:
            con.close()
        for origin_url, username in rows:
            url, user = str(origin_url or ""), str(username or "")
            key = (url, user)
            if url and key not in seen:
                seen.add(key)
                out.append({"origin_url": url, "username": user})
    return out


# ── resolve: the browser autofills its own password; we read it over CDP (no decryption) ──────────

def _cdp_call(ws, msg_id: int, method: str, params: Optional[dict] = None) -> dict:
    ws.send(json.dumps({"id": msg_id, "method": method, "params": params or {}}))
    deadline = time.monotonic() + _CDP_TIMEOUT_S
    while time.monotonic() < deadline:
        msg = json.loads(ws.recv(timeout=_CDP_TIMEOUT_S))
        if msg.get("id") == msg_id:
            if "error" in msg:
                raise RuntimeError(f"{method} failed: {msg['error'].get('message', msg['error'])}")
            return msg.get("result", {})
    raise RuntimeError(f"{method} timed out")


@contextlib.contextmanager
def _page_target(cdp_url: str, target_url: str) -> Iterator[str]:
    """Open ``target_url`` in a fresh tab of the real-profile browser and yield that tab's CDP URL.
    The tab is ALWAYS closed over ``Target.closeTarget`` on exit (success or failure) so repeated
    fills never leak live pages into the shared real-profile browser."""
    from websockets.sync.client import connect

    from agent.proxy_bypass import loopback_connect_kwargs
    target_id: Optional[str] = None
    with connect(cdp_url, open_timeout=_CDP_TIMEOUT_S, close_timeout=2, max_size=None,
                 **loopback_connect_kwargs(cdp_url)) as ws:
        created = _cdp_call(ws, 1, "Target.createTarget", {"url": target_url})
        target_id = created.get("targetId")
    # The per-tab debugger URL is derived from the browser-level URL's host.
    base = cdp_url.rsplit("/devtools/", 1)[0]
    try:
        yield f"{base}/devtools/page/{target_id}"
    finally:
        if target_id:
            try:
                with connect(cdp_url, open_timeout=_CDP_TIMEOUT_S, close_timeout=2, max_size=None,
                             **loopback_connect_kwargs(cdp_url)) as ws:
                    _cdp_call(ws, 99, "Target.closeTarget", {"targetId": target_id})
            except Exception as exc:  # health: allow BLE001 -- best-effort tab cleanup must never mask the resolve result or its error
                logger.debug("real-profile logins: could not close autofill tab %s: %s", target_id, exc)


# Shared JS: locate the sign-in form's password field and the username field TIED to it, rather than
# guessing the username by a name heuristic (a plain <input type=text name=login> that Chrome fills
# correctly is otherwise invisible). The username is the nearest "texty" input that PRECEDES the
# autofilled password field inside the same form (Chrome's own pairing), falling back to the last
# texty input before it in the document, or the first filled texty input.
_FORM_JS_HELPER = (
    "const texty = el => {"
    "  if (!el || el.tagName !== 'INPUT') return false;"
    "  const t = (el.type || 'text').toLowerCase();"
    "  return !['password','hidden','submit','button','checkbox','radio','image','file','range','color'].includes(t);"
    "};"
    "const pwFields = Array.from(document.querySelectorAll('input[type=password]'))"
    "  .filter(p => p.offsetParent !== null || p.getClientRects().length);"
    "const pw = pwFields.find(p => p.value) || pwFields[0] || null;"
    "const usernameFor = p => {"
    "  if (!p) return null;"
    "  const form = p.form;"
    "  const scope = form ? Array.from(form.elements) : Array.from(document.querySelectorAll('input'));"
    "  const idx = scope.indexOf(p);"
    "  const before = scope.slice(0, idx < 0 ? scope.length : idx).filter(texty);"
    "  if (before.length) return before[before.length - 1];"
    "  const filled = scope.filter(e => texty(e) && e.value);"
    "  if (filled.length) return filled[0];"
    "  const any = scope.filter(texty);"
    "  return any.length ? any[0] : null;"
    "};"
)

# JS that returns the viewport-centre coordinates of the field to click (the username field tied to
# the sign-in form when present, else the password field), so the resolver can land a TRUSTED CDP
# mouse click that makes Chrome's PasswordAutofillAgent offer+fill the saved credential. Returns ''
# while no password field has rendered yet, so the resolver can POLL for a late-mounting form.
_FIELD_POINT_JS = (
    "(() => {"
    + _FORM_JS_HELPER +
    " if (!pw) return '';"
    " const target = usernameFor(pw) || pw;"
    " const r = target.getBoundingClientRect();"
    " return JSON.stringify({x: r.x + r.width / 2, y: r.y + r.height / 2});"
    "})()"
)

# Reads the autofilled password AND the username TIED to its form AND the page's own origin in ONE
# evaluate, so the origin that guards the read is the SAME page the value came from (no redirect can
# slip in between two calls).
_READ_PASSWORD_VALUE_JS = (
    "(() => {"
    + _FORM_JS_HELPER +
    " const u = usernameFor(pw);"
    " return JSON.stringify({password: pw ? String(pw.value || '') : '',"
    "   username: u ? String(u.value || '') : '', origin: String(location.origin || '')}); })()"
)


def _same_origin(page_origin: str, saved_origin: str) -> bool:
    """True when the live page's ``location.origin`` is the handle's saved origin, compared through
    the vault's own normalizer (so ``:443`` / case differences never cause a false mismatch)."""
    from agent.vault_store import VaultError, normalize_origin
    try:
        return normalize_origin(page_origin) == normalize_origin(saved_origin)
    except VaultError:
        return False


def resolve_login_password(cdp_url: str, nav_url: str, origin: str, username: str) -> Optional[str]:
    """Return the saved password for the login by letting the real-profile browser autofill it and
    reading the field value over CDP. ``nav_url`` is the exact saved sign-in URL to open (path
    preserved); ``origin`` is the handle's normalized origin the final page must still be on;
    ``username`` is the saved account (may be the empty string for a blank-username credential).
    ``None`` when nothing autofills, the final page drifted to a different origin, or the filled
    account cannot be positively tied to the handle's identity. The browser decrypts its own store;
    this code never decrypts.

    Three refusals protect against the wrong secret coming back:

    - **origin recheck** — the page is read together with its ``location.origin``; if the saved URL
      redirected to a different site (which may have its own saved login), the origin no longer
      matches ``origin`` and we refuse, so another site's password can never be returned for this
      handle.
    - **account match (named handle)** — when the handle carries a username, the autofilled username
      must equal it; a site with several saved accounts autofills its preferred one, so an empty or
      different filled username is refused rather than returning a DIFFERENT account's password.
    - **account match (blank handle)** — an EMPTY saved username is a real identity, not a wildcard.
      Chrome, resolving a blank-username login on an origin that also has a named account, may
      autofill the NAMED account's password instead. We therefore require the autofilled username to
      be empty too; any non-empty filled username means Chrome handed back a different (named)
      account, and we refuse rather than leak it.

    Both the form discovery and the autofill read are POLLED up to bounded deadlines, so a form that
    mounts a second or two after navigation (SPA hydration) and a fill that lands slightly after the
    trusted click still resolve, instead of a single fixed sleep missing them.
    """
    from websockets.sync.client import connect

    from agent.proxy_bypass import loopback_connect_kwargs
    ids = itertools.count(1)
    with _page_target(cdp_url, nav_url) as page_url:
        with connect(page_url, open_timeout=_CDP_TIMEOUT_S, close_timeout=2, max_size=None,
                     **loopback_connect_kwargs(page_url)) as ws:
            _cdp_call(ws, next(ids), "Page.enable")
            _cdp_call(ws, next(ids), "Runtime.enable")
            # Poll for a usable sign-in form (password field present) up to _FORM_WAIT_S: late-mounting
            # forms would be missed by a single fixed sleep + one inspection.
            xy = _wait_for_field_point(ws, ids)
            if xy is None:
                return None
            # A TRUSTED click (not a scripted DOM event) is what makes Chrome offer+fill the credential.
            for ev in ("mousePressed", "mouseReleased"):
                _cdp_call(ws, next(ids), "Input.dispatchMouseEvent",
                          {"type": ev, "x": xy["x"], "y": xy["y"], "button": "left", "clickCount": 1})
            # Poll for autofill completion (a filled password) up to _AUTOFILL_WAIT_S.
            filled = _wait_for_autofill(ws, ids)
    if not isinstance(filled, dict):
        return None
    password = filled.get("password")
    if not password:
        return None
    filled_username = str(filled.get("username") or "")
    # Refuse if the saved URL redirected off its origin: another site's autofill is not this login's.
    if not _same_origin(str(filled.get("origin") or ""), origin):
        return None
    # Account match: the autofilled identity must be POSITIVELY the handle's. An empty saved username
    # is a real identity (not a wildcard), so a blank handle requires an empty filled username too —
    # a non-empty one means a multi-account origin handed back a NAMED account's password.
    if filled_username != username:
        return None
    return password


def _wait_for_field_point(ws, ids: itertools.count) -> Optional[dict]:
    """Poll the page until a sign-in form's clickable field has rendered (bounded by _FORM_WAIT_S),
    returning its viewport-centre point, or None if none appears in time."""
    deadline = time.monotonic() + _FORM_WAIT_S
    while True:
        point = _cdp_call(ws, next(ids), "Runtime.evaluate", {
            "expression": _FIELD_POINT_JS, "returnByValue": True}).get("result", {}).get("value")
        if point:
            try:
                return json.loads(point)
            except (json.JSONDecodeError, ValueError):
                return None
        if time.monotonic() >= deadline:
            return None
        time.sleep(_POLL_INTERVAL_S)


def _wait_for_autofill(ws, ids: itertools.count) -> Optional[dict]:
    """Poll the page until Chrome has autofilled a password (bounded by _AUTOFILL_WAIT_S), returning
    the parsed {password, username, origin} record. Returns the LAST read (which may have an empty
    password) so the caller's refusals still run; None only on an unreadable/invalid read."""
    deadline = time.monotonic() + _AUTOFILL_WAIT_S
    last: Optional[dict] = None
    while True:
        read = _cdp_call(ws, next(ids), "Runtime.evaluate",
                         {"expression": _READ_PASSWORD_VALUE_JS, "returnByValue": True})
        raw = read.get("result", {}).get("value")
        if isinstance(raw, str):
            try:
                parsed = json.loads(raw)
            except (json.JSONDecodeError, ValueError):
                return None
            last = parsed
            if isinstance(parsed, dict) and parsed.get("password"):
                return parsed
        if time.monotonic() >= deadline:
            return last
        time.sleep(_POLL_INTERVAL_S)
