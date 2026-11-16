"""Region zoom and exact menu invocation for the cua-driver backend (mixed into ``CuaDriverBackend``).

cua-driver 0.21+ ships ``zoom`` (a padded, at most 500 px wide JPEG crop of a window region; ``from_zoom=true`` on a
later click maps zoom-image pixels back to the window) and ``invoke_menu`` (walks an exact application-menu path
through the accessibility API and fails closed on a missing, ambiguous or disabled segment, never via pixels).
Both act on the sticky target set by the last capture/focus_app.
"""

from __future__ import annotations

from typing import Any

from tools.computer_use.backend import ActionResult, CaptureResult
from tools.computer_use.cua_backend_capture import _image_from_tool_result
from tools.computer_use.cua_backend_input import _NO_TARGET_MSG

_ZOOM_NOTE = ("zoomed crop of region {region} with 20% padding; to click something seen here, pass its pixel "
              "coordinate from THIS image with from_zoom=true")


class _ZoomMenuMixin:
    def zoom(self, region: tuple[float, float, float, float]) -> CaptureResult:
        if self._active_window_id is None:
            return CaptureResult(mode="vision", width=0, height=0, window_title=_NO_TARGET_MSG)
        x1, y1, x2, y2 = region
        args: dict[str, Any] = {"window_id": self._active_window_id, "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                                "session": self._session_id}
        if self._active_pid is not None:
            args["pid"] = self._active_pid  # required by the driver for from_zoom translation
        out = self._session.call_tool("zoom", args)  # transport errors surface like capture's (tool-level error)
        image, mime = _image_from_tool_result(out)
        if out.get("isError") or not image:
            return CaptureResult(mode="vision", width=0, height=0, window_title=str(out.get("data") or "zoom failed"))
        sc = out.get("structuredContent") or {}
        return CaptureResult(mode="vision", width=int(sc.get("width") or 0), height=int(sc.get("height") or 0),
                             png_b64=image, image_mime_type=mime or "image/jpeg", app=self._last_app or "",
                             png_bytes_len=len(image) * 3 // 4, note=_ZOOM_NOTE.format(region=list(region)))

    def invoke_menu(self, path: list[str]) -> ActionResult:
        if self._active_pid is None or self._active_window_id is None:
            return ActionResult(ok=False, action="invoke_menu", message=_NO_TARGET_MSG)
        res = self._action("invoke_menu", {"pid": self._active_pid, "window_id": self._active_window_id,
                                           "path": list(path)})
        refusal = res.meta.get("refusal") if isinstance(res.meta, dict) else None
        if isinstance(refusal, dict) and refusal.get("code") and res.code is None:
            res.code = str(refusal["code"])  # e.g. menu_path_unavailable: the model must not retry by pixels
        return res
