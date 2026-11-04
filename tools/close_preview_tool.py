"""Close the Hermes desktop GUI's preview pane, or one tab (``preview.close``).
Registration lives in `desktop_preview`. The renderer drops the matching tab — or
the whole pane when no url is given — only for the window that asked."""

from tools import desktop_ui
from tools.open_preview_tool import _normalize_target


def close_preview_tool(url: str = "") -> str:
    """Ask the desktop GUI to close the preview pane, or the tab for ``url``."""
    target = _normalize_target(url or "")
    return desktop_ui.emit_or_error(
        "preview.close", {"url": target}, "Failed to close the preview pane: ",
        "The preview pane is only available in the Hermes desktop app.", {"success": True, "url": target},
    )
