"""Request-local transport for vault operations on a Desktop preview.

The existing vault code owns classification, origin/nonce checks and secrets.
This bridge only binds its evaluations to the window's selected guest page.
It never retries an unavailable preview through a managed browser.
"""

from __future__ import annotations

import json
from contextvars import ContextVar
from typing import Callable


_evaluator: ContextVar[Callable[[str], dict] | None] = ContextVar("vault_preview_evaluator", default=None)


def preview_evaluator() -> Callable[[str], dict] | None:
    return _evaluator.get()


def _unavailable() -> dict:
    return {"success": False, "error_type": "preview_unavailable",
            "error": "The selected preview could not complete the secure vault operation. "
                     "Keep this chat and its login page open in an updated Hermes Desktop, then retry."}


def _request(callback: Callable, operation: str, **params) -> dict:
    try:
        raw = callback({"action": "vault", "vault": {"operation": operation, **params}})
        answer = json.loads(raw) if isinstance(raw, str) else raw
    except Exception:
        # IPC/guest exceptions can echo the evaluated source, including a secret.
        # Neither the exception nor the request belongs in logs or tool output.
        return _unavailable()
    return answer if isinstance(answer, dict) and answer.get("success") is True else _unavailable()


def run_preview_vault(callback: Callable | None, operation: Callable[[], str]) -> str:
    if callback is None:
        return json.dumps(_unavailable())
    opened = _request(callback, "open")
    target = opened.get("target")
    if not opened.get("success") or not isinstance(target, str) or not target:
        return json.dumps(_unavailable())

    token = _evaluator.set(lambda expression: _request(
        callback, "evaluate", target=target, expression=expression))
    try:
        return operation()
    finally:
        _evaluator.reset(token)
        _request(callback, "close", target=target)


def redact_preview_result(raw: str) -> str:
    """Preview text and interaction inventories share the browser egress policy."""
    from tools.browser_tool_snapshot import _redact_browser_output

    try:
        value = json.loads(raw)
    except (ValueError, TypeError):
        value = {"text": str(raw)}
    return json.dumps(_redact_browser_output(value), ensure_ascii=False)
