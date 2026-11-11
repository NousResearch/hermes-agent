"""Clipboard action handlers for the computer_use tool (rows in ``tool._ACTIONS``)."""

from __future__ import annotations

import json
from typing import Any

from tools.computer_use.backend import ActionResult

_CLIPBOARD_PAYLOAD_KEYS = ("text", "image_path", "file_path")


def _clipboard_response(res: ActionResult) -> str:
    # No verify->escalate verdict: the clipboard has no on-screen effect to re-capture. ``types`` (always) and
    # ``text`` (read; None when the clipboard holds no plain text) come straight from the driver payload.
    optional = {k: v for k, v in (("message", res.message), ("code", res.code)) if v}
    return json.dumps({"ok": res.ok, "action": res.action, **optional,
                       **{k: res.meta[k] for k in ("types", "text") if k in res.meta}})


def _do_clipboard_read(backend, action, args, **_):
    return _clipboard_response(backend.clipboard_read())


def _do_clipboard_write(backend, action, args, **_):
    given = {k: args[k] for k in _CLIPBOARD_PAYLOAD_KEYS if args.get(k) is not None}
    if len(given) != 1:
        return json.dumps({"error": "clipboard_write takes exactly one of `text`, `image_path` (absolute path), "
                                    "`file_path` (absolute path)"})
    return _clipboard_response(backend.clipboard_write(**given))


def _summarize_clipboard_write(action: str, args: dict[str, Any], fg: str) -> str:
    if (text := args.get("text")) is not None:
        return f"replace the clipboard with text {text[:60]!r}" + ("..." if len(text) > 60 else "")
    return f"replace the clipboard with {args.get('image_path') or args.get('file_path')!r}"
