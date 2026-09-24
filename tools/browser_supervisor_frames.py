"""Frame / OOPIF target tracking half of the CDP supervisor.

Maintains ``CDPSupervisor._frames`` from ``Page.frame*`` and ``Target.*``
events and renders the bounded ``frame_tree`` snapshot payload. ``FrameInfo``
entries for OOPIFs carry the child CDP session id so ``browser_cdp(frame_id=)``
can route calls into the iframe over the supervisor's live socket.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, replace
from typing import Any, Callable, Dict, List, Optional, Tuple

# Logger-name parity with the origin module (records must look unchanged).
logger = logging.getLogger("tools.browser_supervisor")

# Snapshot caps for frame_tree — keep payloads bounded on ad-heavy pages.
FRAME_TREE_MAX_ENTRIES = 30
FRAME_TREE_MAX_OOPIF_DEPTH = 2

_AUTO_ATTACH_PARAMS = {"autoAttach": True, "waitForDebuggerOnStart": False, "flatten": True}


@dataclass
class FrameInfo:
    """One frame in the page's frame tree. ``is_oopif`` frames have their own CDP
    target (reachable via ``cdp_session_id``); same-origin / srcdoc iframes share
    the parent process (``is_oopif=False``, ``cdp_session_id=None``)."""

    frame_id: str
    url: str
    origin: str
    parent_frame_id: Optional[str]
    is_oopif: bool
    cdp_session_id: Optional[str] = None
    name: str = ""

    def to_dict(self) -> Dict[str, Any]:
        d = {"frame_id": self.frame_id, "url": self.url, "origin": self.origin, "is_oopif": self.is_oopif}
        optional = (("session_id", self.cdp_session_id), ("parent_frame_id", self.parent_frame_id), ("name", self.name))
        d.update({k: v for k, v in optional if v})
        return d


class FrameTrackingMixin:
    """Frame-tree bookkeeping for ``CDPSupervisor`` (event handlers run on its loop)."""

    async def _enable_page_domains(self, session_id: Optional[str], *, timeout: float) -> None:
        """Page.enable + Runtime.enable + nested auto-attach on one session."""
        await self._cdp("Page.enable", session_id=session_id, timeout=timeout)
        await self._cdp("Runtime.enable", session_id=session_id, timeout=timeout)
        await self._cdp("Target.setAutoAttach", _AUTO_ATTACH_PARAMS, session_id=session_id, timeout=timeout)

    def _on_frame_attached(self, params: Dict[str, Any], session_id: Optional[str]) -> None:
        frame_id = params.get("frameId")
        if frame_id:
            self._set_frame(FrameInfo(frame_id=frame_id, url="", origin="", parent_frame_id=params.get("parentFrameId"),
                                      is_oopif=False, cdp_session_id=session_id))

    def _set_frame(self, frame: FrameInfo) -> None:
        with self._state_lock:
            self._frames[frame.frame_id] = frame

    def _on_frame_navigated(self, params: Dict[str, Any], session_id: Optional[str]) -> None:
        frame = params.get("frame") or {}
        frame_id = frame.get("id")
        if not frame_id:
            return
        with self._state_lock:
            old = self._frames.get(frame_id) or FrameInfo(frame_id, "", "", None, False, session_id)
            self._frames[frame_id] = FrameInfo(
                frame_id=frame_id, url=str(frame.get("url") or ""),
                origin=str(frame.get("securityOrigin") or frame.get("origin") or ""),
                parent_frame_id=frame.get("parentId") or old.parent_frame_id, is_oopif=old.is_oopif,
                cdp_session_id=old.cdp_session_id, name=str(frame.get("name") or old.name),
            )

    def _on_frame_detached(self, params: Dict[str, Any], session_id: Optional[str]) -> None:
        """Drop a frame only when it's truly gone. ``reason="swap"`` = migrating processes
        (e.g. promoted to an OOPIF) — dropping would hide the iframe. Even with ``remove``
        the parent only knows the child left ITS process; a live child session means it's
        still alive, so keep it until Target.detached + a later frameDetached clear it."""
        frame_id = params.get("frameId")
        if not frame_id or str(params.get("reason") or "remove").lower() == "swap":
            return
        with self._state_lock:
            old = self._frames.get(frame_id)
            if not (old and old.is_oopif and old.cdp_session_id):
                self._frames.pop(frame_id, None)

    async def _on_target_attached(self, params: Dict[str, Any], session_id: Optional[str] = None) -> None:
        info = params.get("targetInfo") or {}
        sid = params.get("sessionId")
        target_type = info.get("type")
        if not sid or target_type not in {"iframe", "worker"}:
            return
        if target_type == "iframe":
            # Record the frame with its OOPIF session id for interaction routing;
            # origin is filled by frameNavigated on the child session.
            target_id = info.get("targetId")
            with self._state_lock:
                old = self._frames.get(target_id)
                self._frames[target_id] = FrameInfo(
                    frame_id=target_id, url=str(info.get("url") or ""), origin="", is_oopif=True, cdp_session_id=sid,
                    parent_frame_id=old.parent_frame_id if old else None, name=str(info.get("title") or (old.name if old else "")),
                )
        # Enable child domains off-loop: awaiting the replies here would deadlock
        # because only the reader can resolve those Futures.
        asyncio.create_task(self._enable_child_domains(sid))

    async def _enable_child_domains(self, sid: str) -> None:
        """Enable Page+Runtime (+nested setAutoAttach) and the dialog bridge on a child session."""
        try:
            await self._enable_page_domains(sid, timeout=3.0)
        except Exception as e:
            logger.debug("child session %s setup failed: %s", sid[:16], e)
        await self._install_dialog_bridge(sid)

    def _on_target_detached(self, params: Dict[str, Any], session_id: Optional[str] = None) -> None:
        """Clear the session binding of frames on a detached child session. Frames are
        deliberately NOT dropped: Browserbase fires transient detaches during page transitions
        while the iframe is still visible; ``Page.frameDetached`` cleans up if it truly goes away."""
        sid = params.get("sessionId")
        if not sid:
            return
        with self._state_lock:
            self._frames.update({fid: replace(f, cdp_session_id=None) for fid, f in self._frames.items()
                                 if f.cdp_session_id == sid})

    def _build_frame_tree_locked(self) -> Dict[str, Any]:
        """Capped frame_tree payload (must hold state lock). Top frame = one with
        no parent, preferring oopif=False; BFS from it, capped by
        FRAME_TREE_MAX_ENTRIES and FRAME_TREE_MAX_OOPIF_DEPTH for OOPIF branches."""
        frames = self._frames
        tops = [f for f in frames.values() if not f.parent_frame_id]
        top = next((f for f in tops if not f.is_oopif), tops[0] if tops else None)
        if top is None:
            return {"top": None, "children": [], "truncated": False}

        children: List[Dict[str, Any]] = []
        truncated = False
        queue: List[Tuple[FrameInfo, int]] = [(f, 1) for f in frames.values() if f.parent_frame_id == top.frame_id]
        visited = {top.frame_id}
        while queue and len(children) < FRAME_TREE_MAX_ENTRIES:
            frame, depth = queue.pop(0)
            if frame.frame_id in visited:
                continue
            visited.add(frame.frame_id)
            if frame.is_oopif and depth > FRAME_TREE_MAX_OOPIF_DEPTH:
                truncated = True
                continue
            children.append(frame.to_dict())
            queue.extend((f, depth + 1) for f in frames.values()
                         if f.parent_frame_id == frame.frame_id and f.frame_id not in visited)
        return {"top": top.to_dict(), "children": children, "truncated": truncated or bool(queue)}

    # CDP event → handler(self, params, session_id); merged into CDPSupervisor._EVENT_HANDLERS.
    EVENT_HANDLERS: Dict[str, Callable[..., Any]] = {
        "Page.frameAttached": _on_frame_attached,
        "Page.frameNavigated": _on_frame_navigated,
        "Page.frameDetached": _on_frame_detached,
        "Target.attachedToTarget": _on_target_attached,
        "Target.detachedFromTarget": _on_target_detached,
    }
