"""Agent-facing tool: respond to a native JS dialog captured by the CDP supervisor.

Response-only: the agent reads ``pending_dialogs`` from ``browser_snapshot``,
then calls ``browser_dialog(action=...)``. Gated on ``_browser_cdp_check`` so it
appears together with ``browser_cdp``. Design: ``website/docs/developer-guide/browser-supervisor.md``.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Optional

from tools.browser_supervisor import SUPERVISOR_REGISTRY
from tools.registry import registry

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


def browser_dialog(
    action: str,
    prompt_text: Optional[str] = None,
    dialog_id: Optional[str] = None,
    task_id: Optional[str] = None,
) -> str:
    """Respond to a pending dialog on the active task's CDP supervisor."""
    supervisor = SUPERVISOR_REGISTRY.get(task_id or "default")
    if supervisor is None:
        return json.dumps({
            "success": False,
            "error": (
                "No CDP supervisor is attached to this task. Either the "
                "browser backend doesn't expose CDP (Camofox, default "
                "Playwright) or no browser session has been started yet. "
                "Call browser_navigate or /browser connect first."
            ),
        })
    result = supervisor.respond_to_dialog(action=action, prompt_text=prompt_text, dialog_id=dialog_id)
    if result.get("ok"):
        return json.dumps({"success": True, "action": action, "dialog": result.get("dialog", {})})
    return json.dumps({"success": False, "error": result.get("error", "unknown error")})


def _browser_dialog_check() -> bool:
    """Gate: same as ``browser_cdp`` so the two tools appear/disappear together."""
    from tools.browser_cdp_tool import _browser_cdp_check

    return _browser_cdp_check()


registry.register(
    name="browser_dialog",
    toolset="browser-cdp",
    schema=BROWSER_DIALOG_SCHEMA,
    handler=lambda args, **kw: browser_dialog(action=args.get("action", ""), prompt_text=args.get("prompt_text"),
                                              dialog_id=args.get("dialog_id"), task_id=kw.get("task_id")),
    check_fn=_browser_dialog_check,
    emoji="💬",
)


# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# Names external plugins imported from this module before the Sep 2026 decomposition.
# Internal code MUST NOT use these (scripts/check_compat_pointers.py fails CI if it does).
# The whole block is removed by reverting the commit that added it.
import logging  # noqa: F401,E402


_PLUGIN_COMPAT_LAZY = {
    'logger': ('tools.approval', 'logger'),
}


def __getattr__(name):  # PEP 562 — lazy so no import cycles
    target = _PLUGIN_COMPAT_LAZY.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib
    from hermes_cli.plugin_compat import warn_once
    warn_once(__name__, name, *target)
    return getattr(importlib.import_module(target[0]), target[1])
# ---- END PLUGIN-COMPAT ----
