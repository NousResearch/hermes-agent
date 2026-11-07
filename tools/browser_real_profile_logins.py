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
  autofill: over CDP we open the item's exact origin in the real-profile browser, dispatch a click
  gesture on the sign-in form, and Chrome's own PasswordAutofillAgent fills the password into the DOM
  (the browser decrypts its OWN store with the OS keychain). We read that value back over
  ``Runtime.evaluate``. The agent never sees it: the fill tool registers it with the redaction
  boundary before it touches a page, exactly like every other backend's ``resolve_password``.

Chrome autofills only on an EXACT-origin page and only after a user-style gesture, so the password is
resolved against the real-profile browser on the credential's true origin, never a page we fabricate.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
from pathlib import Path
from typing import Any, Optional

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


def read_login_metadata(copy_dir: str) -> list[dict[str, str]]:
    """``[{origin, username}, …]`` for every non-blocklisted saved login, deduped on (origin,
    username). Reads plaintext columns only; never the encrypted ``password_value``."""
    seen: set[tuple[str, str]] = set()
    out: list[dict[str, str]] = []
    for db in _login_data_paths(copy_dir):
        try:
            con = sqlite3.connect(f"file:{db}?mode=ro&immutable=1", uri=True)
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
            origin, user = str(origin_url or ""), str(username or "")
            key = (origin, user)
            if origin and key not in seen:
                seen.add(key)
                out.append({"origin": origin, "username": user})
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


def _page_cdp_url(cdp_url: str, target_url: str) -> str:
    """Open ``target_url`` in a fresh tab of the real-profile browser and return that tab's CDP URL."""
    from websockets.sync.client import connect

    from agent.proxy_bypass import loopback_connect_kwargs
    with connect(cdp_url, open_timeout=_CDP_TIMEOUT_S, close_timeout=2, max_size=None,
                 **loopback_connect_kwargs(cdp_url)) as ws:
        created = _cdp_call(ws, 1, "Target.createTarget", {"url": target_url})
        target_id = created.get("targetId")
    # The per-tab debugger URL is derived from the browser-level URL's host.
    base = cdp_url.rsplit("/devtools/", 1)[0]
    return f"{base}/devtools/page/{target_id}"


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

_READ_PASSWORD_VALUE_JS = (
    "(() => { const p = document.querySelector('input[type=password]');"
    " return p ? String(p.value || '') : ''; })()"
)


def resolve_login_password(cdp_url: str, origin: str, username: str) -> Optional[str]:
    """Return the saved password for ``(origin, username)`` by letting the real-profile browser
    autofill it on ``origin`` and reading the field value over CDP. ``None`` when nothing autofills
    (the credential is gone, the origin blocks forms, or several accounts need a chooser). The browser
    decrypts its own store; this code never decrypts."""
    from websockets.sync.client import connect

    from agent.proxy_bypass import loopback_connect_kwargs
    page_url = _page_cdp_url(cdp_url, origin)
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
    value = read.get("result", {}).get("value")
    return value if isinstance(value, str) and value else None
