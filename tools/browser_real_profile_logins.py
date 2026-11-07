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
import json
import logging
import sqlite3
import time
from pathlib import Path
from typing import Any, Iterator, Optional

logger = logging.getLogger(__name__)

_CDP_TIMEOUT_S = 20.0
# Chrome fills the password only after a real gesture lands on the form; give its agent a moment.
_AUTOFILL_SETTLE_S = 2.0


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
            con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
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


# JS that returns the viewport-centre coordinates of the sign-in form's username/password control, so
# the resolver can land a TRUSTED CDP mouse click on it. Chrome's PasswordAutofillAgent fills saved
# credentials only on a trusted interaction (synthetic DOM events are ignored), so the click must come
# over Input.dispatchMouseEvent, not a scripted MouseEvent.
_FIELD_POINT_JS = (
    "(() => {"
    " const el = document.querySelector("
    "   'input[autocomplete=username], input[type=email], input[name*=user i], input[name*=email i]')"
    "   || document.querySelector('input[type=password]');"
    " if (!el) return '';"
    " const r = el.getBoundingClientRect();"
    " return JSON.stringify({x: r.x + r.width / 2, y: r.y + r.height / 2});"
    "})()"
)

# Reads the autofilled values AND the page's own origin in ONE evaluate, so the origin that guards
# the read is the SAME page the value came from (no redirect can slip in between two calls).
_READ_PASSWORD_VALUE_JS = (
    "(() => { const p = document.querySelector('input[type=password]');"
    " const u = document.querySelector("
    "   'input[autocomplete=username], input[type=email], input[name*=user i], input[name*=email i]');"
    " return JSON.stringify({password: p ? String(p.value || '') : '',"
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
    ``username`` is the saved account. ``None`` when nothing autofills, the final page drifted to a
    different origin, or the filled account does not match. The browser decrypts its own store; this
    code never decrypts.

    Two refusals protect against the wrong secret coming back:

    - **origin recheck** — the page is read together with its ``location.origin``; if the saved URL
      redirected to a different site (which may have its own saved login), the origin no longer
      matches ``origin`` and we refuse, so another site's password can never be returned for this
      handle.
    - **account match** — when the handle carries a username, the autofilled username must equal it;
      a site with several saved accounts autofills its preferred one, so an empty or different filled
      username is refused rather than returning a DIFFERENT account's password.
    """
    from websockets.sync.client import connect

    from agent.proxy_bypass import loopback_connect_kwargs
    with _page_target(cdp_url, nav_url) as page_url:
        with connect(page_url, open_timeout=_CDP_TIMEOUT_S, close_timeout=2, max_size=None,
                     **loopback_connect_kwargs(page_url)) as ws:
            _cdp_call(ws, 1, "Page.enable")
            _cdp_call(ws, 2, "Runtime.enable")
            time.sleep(1.0)  # let the navigated page settle before touching the form
            point = _cdp_call(ws, 3, "Runtime.evaluate", {
                "expression": _FIELD_POINT_JS, "returnByValue": True}).get("result", {}).get("value")
            if not point:
                return None
            xy = json.loads(point)
            # A TRUSTED click (not a scripted DOM event) is what makes Chrome offer+fill the credential.
            for ev in ("mousePressed", "mouseReleased"):
                _cdp_call(ws, 4, "Input.dispatchMouseEvent",
                          {"type": ev, "x": xy["x"], "y": xy["y"], "button": "left", "clickCount": 1})
            time.sleep(_AUTOFILL_SETTLE_S)
            read = _cdp_call(ws, 5, "Runtime.evaluate",
                             {"expression": _READ_PASSWORD_VALUE_JS, "returnByValue": True})
    raw = read.get("result", {}).get("value")
    if not isinstance(raw, str):
        return None
    try:
        filled = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return None
    password = filled.get("password")
    if not password:
        return None
    # Refuse if the saved URL redirected off its origin: another site's autofill is not this login's.
    if not _same_origin(str(filled.get("origin") or ""), origin):
        return None
    # Refuse a multi-account origin that autofilled a different (or no) account than the handle names.
    if username and filled.get("username") != username:
        return None
    return password
