"""Vault page operations on the Hermes Desktop in-app browser (the preview ``<webview>``).

The vault code in ``tools.browser_vault_tool`` owns classification, origin/nonce checks, secret
resolution and redaction registration. This module only carries its scripts to ONE page the Desktop
window pinned for the operation: an opaque target the renderer minted for the tab (``open``), every
``evaluate`` bound to it, and ``close`` at the end. The renderer refuses an evaluate once that tab's
document changed, its window lost the session, or the request was cancelled.

Two callers share it:

- the agent's ``browser_vault`` tool (``target='preview'``), whose transport is the session's
  ``drive_preview_callback`` (a ``preview.act`` server request answered by the window showing the chat);
- the user's key button in the Browser bar (``vault.preview.fill``), whose renderer mints the target
  first and whose transport is the RPC's own socket.

It plugs into the vault's ``BrowserPageBackend`` seam, so core keeps the secret contract (exception text
on the secret path is discarded, failures get fixed text). A preview failure never falls back to
another browser.
"""

from __future__ import annotations

import json
from contextvars import ContextVar
from typing import Any, Callable

from agent.browser_page_backend import BrowserPageBackend

PreviewCallback = Callable[[dict], Any]

_UNAVAILABLE = ("The in-app browser page could not complete the secure vault operation. Keep this chat and its "
                "login page open in an updated Hermes Desktop, then retry.")


def _unavailable() -> dict:
    return {"success": False, "error_type": "preview_unavailable", "error": _UNAVAILABLE}


def _request(callback: PreviewCallback, operation: str, **params) -> dict:
    try:
        raw = callback({"action": "vault", "vault": {"operation": operation, **params}})
        answer = json.loads(raw) if isinstance(raw, str) else raw
    except Exception:  # health: allow BLE001 -- IPC/guest exception text can echo the evaluated script
        return _unavailable()
    return answer if isinstance(answer, dict) and answer.get("success") is True else _unavailable()


class _PreviewPage(BrowserPageBackend):
    """The pinned preview page as the vault's page backend for the duration of one operation."""

    def __init__(self, callback: PreviewCallback, target: str) -> None:
        self._callback, self._target = callback, target

    @property
    def name(self) -> str:
        return "in-app browser"

    def has_page(self, task_id: str) -> bool:
        return True

    def evaluate(self, task_id: str, expression: str, *, secret: bool) -> Any:
        answer = _request(self._callback, "evaluate", target=self._target, expression=expression)
        if answer.get("success") is not True:
            raise RuntimeError(_UNAVAILABLE)
        return answer.get("result")


_page: ContextVar[_PreviewPage | None] = ContextVar("vault_preview_page", default=None)


def active_page() -> _PreviewPage | None:
    """The preview page bound to the running vault operation, or None (managed browser / plugin page)."""
    return _page.get()


def run_preview_vault(callback: PreviewCallback | None, operation: Callable[[], str], *, target: str = "") -> str:
    """Run one vault operation against a pinned preview page. ``target`` given = the renderer already
    pinned the page (user path); empty = ask the window showing the session to pin its active tab."""
    if callback is None:
        return json.dumps(_unavailable())
    if not target:
        opened = _request(callback, "open")
        target = str(opened.get("target") or "")
        if not target:
            return json.dumps(_unavailable())
    token = _page.set(_PreviewPage(callback, target))
    try:
        return operation()
    finally:
        _page.reset(token)
        _request(callback, "close", target=target)


def _redact(value: Any, depth: int = 0) -> Any:
    from agent.redact import REDACT_WALK_DEPTH_MARKER, REDACT_WALK_MAX_DEPTH, redact_registered_vault_values

    if depth >= REDACT_WALK_MAX_DEPTH:
        return REDACT_WALK_DEPTH_MARKER
    if isinstance(value, str):
        return redact_registered_vault_values(value)
    if isinstance(value, list):
        return [_redact(v, depth + 1) for v in value]
    if isinstance(value, dict):
        return {(_redact(k, depth + 1) if isinstance(k, str) else k): _redact(v, depth + 1) for k, v in value.items()}
    return value


def redact_preview_result(raw: str) -> str:
    """Preview reads and action inventories cross the same vault-secret egress boundary as browser results:
    a page that echoes a filled password into its text, a label or an error never hands it to the model."""
    try:
        value = json.loads(raw)
    except (ValueError, TypeError):
        from agent.redact import redact_registered_vault_values
        return redact_registered_vault_values(str(raw))
    return json.dumps(_redact(value), ensure_ascii=False)
