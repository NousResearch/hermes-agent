"""Clipboard side of the cua-driver backend (mixed into ``CuaDriverBackend``).

cua-driver 0.21+ ships ``clipboard_read`` / ``clipboard_write`` tools; this exposes them through the
``ComputerUseBackend`` contract. Neither touches the sticky input target, so no capture is required first.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from tools.computer_use.backend import ActionResult

_UNSUPPORTED_MSG = ("The connected cua-driver does not provide the `{name}` tool — update it "
                    "(`hermes computer-use install`) or use type/key to move text through the UI instead.")


class _ClipboardMixin:
    def _clipboard_refusal(self, name: str) -> Optional[ActionResult]:
        # Only a completed tools/list can prove absence; before discovery the call itself reports the real error.
        if self._session.capabilities_discovered and not self._session._has_tool(name):
            return ActionResult(ok=False, action=name, code="clipboard_unsupported", message=_UNSUPPORTED_MSG.format(name=name))
        return None

    def clipboard_read(self) -> ActionResult:
        """Types + plain text of the system clipboard (``meta.types``, ``meta.text``; text is ``None`` when the
        clipboard holds no plain text)."""
        if (refusal := self._clipboard_refusal("clipboard_read")) is not None:
            return refusal
        return self._action("clipboard_read", {"include_text": True})

    def clipboard_write(self, *, text: Optional[str] = None, image_path: Optional[str] = None,
                        file_path: Optional[str] = None) -> ActionResult:
        """Replace the clipboard with exactly one value; the driver returns the resulting ``types`` for read-back."""
        if (refusal := self._clipboard_refusal("clipboard_write")) is not None:
            return refusal
        payload: Dict[str, Any] = {k: v for k, v in (("text", text), ("image_path", image_path), ("file_path", file_path))
                                   if v is not None}
        return self._action("clipboard_write", payload)
