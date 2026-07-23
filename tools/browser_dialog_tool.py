"""Agent-facing tool: respond to a native JS dialog captured by the CDP supervisor.

This tool is response-only — the agent first reads ``pending_dialogs`` from
``browser_snapshot`` output, then calls ``browser_dialog(action=...)`` to
accept or dismiss.

Gated on the same ``_browser_cdp_check`` as ``browser_cdp`` so it only
appears when a CDP endpoint is reachable (Browserbase with a
``connectUrl``, local Chromium-family browser via ``/browser connect``, or
``browser.cdp_url`` set in config).

See ``website/docs/developer-guide/browser-supervisor.md`` for the full
design.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, Optional

from tools.browser_supervisor import SUPERVISOR_REGISTRY
from tools.registry import registry

logger = logging.getLogger(__name__)


BROWSER_DIALOG_SCHEMA: Dict[str, Any] = {
    "name": "browser_dialog",
    "description": (
        "Respond to a native JavaScript dialog (alert / confirm / prompt / "
        "beforeunload) that is currently blocking the page.\n\n"
        "**Workflow:** call ``browser_snapshot`` first — if a dialog is open, "
        "it appears in the ``pending_dialogs`` field with ``id``, ``type``, "
        "and ``message``. Then call this tool with ``action='accept'`` or "
        "``action='dismiss'``.\n\n"
        "**Prompt dialogs:** pass ``prompt_text`` to supply the response "
        "string. Ignored for alert/confirm/beforeunload.\n\n"
        "**Multiple dialogs:** if more than one dialog is queued (rare — "
        "happens when a second dialog fires while the first is still open), "
        "pass ``dialog_id`` from the snapshot to disambiguate.\n\n"
        "**Availability:** only present when a CDP-capable backend is "
        "attached — Browserbase sessions, local Chromium-family browser via "
        "``/browser connect``, or ``browser.cdp_url`` in config.yaml. "
        "Not available on Camofox (REST-only) or the default Playwright "
        "local browser (CDP port is hidden)."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["accept", "dismiss"],
                "description": (
                    "'accept' clicks OK / returns the prompt text. "
                    "'dismiss' clicks Cancel / returns null from prompt(). "
                    "For ``beforeunload`` dialogs: 'accept' allows the "
                    "navigation, 'dismiss' keeps the page."
                ),
            },
            "prompt_text": {
                "type": "string",
                "description": (
                    "Response string for a ``prompt()`` dialog. Ignored for "
                    "other dialog types. Defaults to empty string."
                ),
            },
            "dialog_id": {
                "type": "string",
                "description": (
                    "Specific dialog to respond to, from "
                    "``browser_snapshot.pending_dialogs[].id``. Required "
                    "only when multiple dialogs are queued."
                ),
            },
        },
        "required": ["action"],
    },
}


def _iter_frame_entries(node: Any):
    if isinstance(node, dict):
        yield node
        for value in node.values():
            yield from _iter_frame_entries(value)
    elif isinstance(node, (list, tuple)):
        for item in node:
            yield from _iter_frame_entries(item)


def _dialog_frame_url(snapshot: Any, dialog_id: Optional[str]) -> Optional[str]:
    pending = list(getattr(snapshot, "pending_dialogs", ()) or ())
    if dialog_id:
        dialog = next((d for d in pending if getattr(d, "id", None) == dialog_id), None)
    elif len(pending) == 1:
        dialog = pending[0]
    else:
        return None
    if dialog is None:
        return None

    frame_id = getattr(dialog, "frame_id", None)
    frame_tree = getattr(snapshot, "frame_tree", {}) or {}
    fallback_top_url = None
    for frame in _iter_frame_entries(frame_tree):
        url = frame.get("url")
        if frame.get("frame_id") == frame_id and url:
            return str(url)
        if fallback_top_url is None and frame.get("url"):
            fallback_top_url = str(frame.get("url"))
    return fallback_top_url


def _blocked_private_dialog_url(snapshot: Any, dialog_id: Optional[str], effective_task_id: str) -> Optional[str]:
    try:
        from tools.browser_tool import (  # type: ignore[import-not-found]
            _eval_ssrf_guard_active,
            _is_always_blocked_url,
            _is_safe_url,
        )
    except Exception as exc:  # pragma: no cover - defensive import guard
        logger.debug("browser_dialog private-page guard unavailable: %s", exc)
        return None

    if not _eval_ssrf_guard_active(effective_task_id):
        return None
    url = _dialog_frame_url(snapshot, dialog_id)
    if url and (_is_always_blocked_url(url) or not _is_safe_url(url)):
        return url
    return None


def browser_dialog(
    action: str,
    prompt_text: Optional[str] = None,
    dialog_id: Optional[str] = None,
    task_id: Optional[str] = None,
) -> str:
    """Respond to a pending dialog on the active task's CDP supervisor."""
    effective_task_id = task_id or "default"
    supervisor = SUPERVISOR_REGISTRY.get(effective_task_id)
    if supervisor is None:
        return json.dumps(
            {
                "success": False,
                "error": (
                    "No CDP supervisor is attached to this task. Either the "
                    "browser backend doesn't expose CDP (Camofox, default "
                    "Playwright) or no browser session has been started yet. "
                    "Call browser_navigate or /browser connect first."
                ),
            }
        )

    snapshot = supervisor.snapshot()
    blocked_url = _blocked_private_dialog_url(snapshot, dialog_id, effective_task_id)
    if blocked_url:
        return json.dumps(
            {
                "success": False,
                "error": (
                    "Blocked browser_dialog on a private/internal page: "
                    f"{blocked_url}"
                ),
            }
        )

    result = supervisor.respond_to_dialog(
        action=action,
        prompt_text=prompt_text,
        dialog_id=dialog_id,
    )
    if result.get("ok"):
        return json.dumps(
            {
                "success": True,
                "action": action,
                "dialog": result.get("dialog", {}),
            }
        )
    return json.dumps({"success": False, "error": result.get("error", "unknown error")})


def _browser_dialog_check() -> bool:
    """Gate: same as ``browser_cdp`` — only offered when CDP is reachable.

    Kept identical so the two tools appear and disappear together. The
    supervisor itself is started lazily by ``browser_navigate`` /
    ``/browser connect`` / Browserbase session creation, so a reachable
    CDP URL is enough to commit to showing the tool.
    """
    try:
        from tools.browser_cdp_tool import _browser_cdp_check  # type: ignore[import-not-found]
    except Exception as exc:  # pragma: no cover — defensive
        logger.debug("browser_dialog check: browser_cdp_tool import failed: %s", exc)
        return False
    return _browser_cdp_check()


registry.register(
    name="browser_dialog",
    toolset="browser-cdp",
    schema=BROWSER_DIALOG_SCHEMA,
    handler=lambda args, **kw: browser_dialog(
        action=args.get("action", ""),
        prompt_text=args.get("prompt_text"),
        dialog_id=args.get("dialog_id"),
        task_id=kw.get("task_id"),
    ),
    check_fn=_browser_dialog_check,
    emoji="💬",
)
