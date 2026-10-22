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
    # CDP changes this for every committed document navigation, even when an
    # OOPIF keeps its frame id and target session.  It is internal route
    # provenance, not model-facing frame-tree metadata.
    loader_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        d = {"frame_id": self.frame_id, "url": self.url, "origin": self.origin, "is_oopif": self.is_oopif}
        optional = (("session_id", self.cdp_session_id), ("parent_frame_id", self.parent_frame_id), ("name", self.name))
        d.update({k: v for k, v in optional if v})
        return d


class FrameTrackingMixin:
    """Frame-tree bookkeeping for ``CDPSupervisor`` (event handlers run on its loop)."""

    async def _select_child_document(self, sid: str, accept: str, *, timeout: float) -> Optional[dict[str, Any]]:
        """Select one owned OOPIF and bind its session, loader and isolated principal."""
        from websockets.exceptions import WebSocketException

        # A vault form may live in a cross-origin OOPIF. Only
        # inspect frame ids returned by THIS top-level page's
        # frame tree: arbitrary attached iframe targets must
        # never become a credential destination.
        tree = await self._cdp("Page.getFrameTree", session_id=sid, timeout=timeout)

        def _child_ids(node: Dict[str, Any]) -> dict[str, None]:
            ids: dict[str, None] = {}
            for child in node.get("childFrames") or []:
                child_frame = child.get("frame") or {}
                frame_id = str(child_frame.get("id") or "")
                if frame_id:
                    ids[frame_id] = None
                ids.update(_child_ids(child))
            return ids

        top_tree = (tree.get("result") or {}).get("frameTree") or {}
        top_frame_id = str((top_tree.get("frame") or {}).get("id") or "")
        frame_ids_by_owner = _child_ids(top_tree)
        # Chrome's top-level Page.getFrameTree omits remote
        # OOPIFs on some versions.  Page.frameAttached is
        # delivered on the top session before Target attaches
        # the OOPIF, so use only its descendants of this exact
        # top frame—not every attached iframe target.
        with self._state_lock:
            all_frames = dict(self._frames)

        def _belongs_to_selected_page(candidate_id: str) -> bool:
            seen = set()
            current = all_frames.get(candidate_id)
            while current and current.parent_frame_id and current.frame_id not in seen:
                seen.add(current.frame_id)
                if current.parent_frame_id == top_frame_id:
                    return True
                current = all_frames.get(current.parent_frame_id)
            return False

        for candidate_id in all_frames:
            if _belongs_to_selected_page(candidate_id):
                frame_ids_by_owner.setdefault(candidate_id, None)
        frame_ids = list(frame_ids_by_owner)
        child_session = None
        child_origin = ""
        with self._state_lock:
            frames = {fid: self._frames.get(fid) for fid in frame_ids}
        for frame_id in frame_ids:
            frame = frames.get(frame_id)
            if frame is None or not frame.is_oopif or not frame.cdp_session_id:
                continue
            child_tree = await self._cdp(
                "Page.getFrameTree", session_id=frame.cdp_session_id, timeout=timeout,
            )
            child_root = ((child_tree.get("result") or {}).get("frameTree") or {}).get("frame") or {}
            loader_id = str(child_root.get("loaderId") or "")
            # Bind the route to the selected OOPIF document;
            # missing/changed loader provenance fails closed.
            if str(child_root.get("id") or "") != frame_id or not loader_id:
                continue
            with self._state_lock:
                current = self._frames.get(frame_id)
                if current is None or current.cdp_session_id != frame.cdp_session_id:
                    continue
                self._frames[frame_id] = replace(current, loader_id=loader_id)
            child_probe = await self._cdp(
                "Runtime.evaluate", {"expression": accept, "returnByValue": True},
                session_id=frame.cdp_session_id, timeout=timeout,
            )
            if child_probe.get("result", {}).get("result", {}).get("value"):
                # The effective principal can be opaque despite
                # an HTTPS location/securityOrigin. Probe in an
                # isolated world so page code cannot replace
                # self.origin to grant itself vault authority.
                try:
                    world = await self._cdp(
                        "Page.createIsolatedWorld", {"frameId": frame_id, "worldName": "hermes-vault-origin"},
                        session_id=frame.cdp_session_id, timeout=timeout,
                    )
                    origin_probe = await self._cdp(
                        "Runtime.evaluate", {"expression": "self.origin", "returnByValue": True,
                                             "contextId": world["result"]["executionContextId"]},
                        session_id=frame.cdp_session_id, timeout=timeout,
                    )
                    child_origin = str(origin_probe.get("result", {}).get("result", {}).get("value") or "")
                except (KeyError, TypeError, RuntimeError, TimeoutError, WebSocketException):
                    # Failed provenance cannot become a request
                    # to try another credential destination.
                    child_origin = ""
                # Even an empty origin selects this document;
                # the vault refuses it without sibling fallback.
                child_session = frame.cdp_session_id
                child_loader_id = loader_id
                break
        if child_session is None:
            return None
        with self._state_lock:
            frame = self._frames.get(frame_id)
        if frame is None or frame.cdp_session_id != child_session:
            return None
        return {"frame_origin": child_origin,
                "route": {"page_session_id": sid, "frame_id": frame_id,
                          "frame_session_id": child_session, "frame_loader_id": child_loader_id}}

    async def _enable_page_domains(self, session_id: Optional[str], *, timeout: float) -> None:
        """Page.enable + Runtime.enable + nested auto-attach on one session."""
        await self._cdp("Page.enable", session_id=session_id, timeout=timeout)
        await self._cdp("Runtime.enable", session_id=session_id, timeout=timeout)
        await self._cdp("Target.setAutoAttach", _AUTO_ATTACH_PARAMS, session_id=session_id, timeout=timeout)

    def _on_frame_attached(self, params: dict[str, Any], session_id: Optional[str]) -> None:
        frame_id = params.get("frameId")
        if frame_id:
            self._set_frame(FrameInfo(frame_id=frame_id, url="", origin="", parent_frame_id=params.get("parentFrameId"),
                                      is_oopif=False, cdp_session_id=session_id))

    def _set_frame(self, frame: FrameInfo) -> None:
        with self._state_lock:
            self._frames[frame.frame_id] = frame

    def _on_frame_navigated(self, params: dict[str, Any], session_id: Optional[str]) -> None:
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
                loader_id=str(frame.get("loaderId") or ""),
            )

    def _on_frame_detached(self, params: dict[str, Any], session_id: Optional[str]) -> None:
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

    async def _on_target_attached(self, params: dict[str, Any], session_id: Optional[str] = None) -> None:
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

    def _on_target_detached(self, params: dict[str, Any], session_id: Optional[str] = None) -> None:
        """Clear the session binding of frames on a detached child session. Frames are
        deliberately NOT dropped: Browserbase fires transient detaches during page transitions
        while the iframe is still visible; ``Page.frameDetached`` cleans up if it truly goes away."""
        sid = params.get("sessionId")
        if not sid:
            return
        with self._state_lock:
            self._frames.update({fid: replace(f, cdp_session_id=None) for fid, f in self._frames.items()
                                 if f.cdp_session_id == sid})

    def _build_frame_tree_locked(self) -> dict[str, Any]:
        """Capped frame_tree payload (must hold state lock). Top frame = one with
        no parent, preferring oopif=False; BFS from it, capped by
        FRAME_TREE_MAX_ENTRIES and FRAME_TREE_MAX_OOPIF_DEPTH for OOPIF branches."""
        frames = self._frames
        tops = [f for f in frames.values() if not f.parent_frame_id]
        top = next((f for f in tops if not f.is_oopif), tops[0] if tops else None)
        if top is None:
            return {"top": None, "children": [], "truncated": False}

        children: list[dict[str, Any]] = []
        truncated = False
        queue: list[tuple[FrameInfo, int]] = [(f, 1) for f in frames.values() if f.parent_frame_id == top.frame_id]
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
    EVENT_HANDLERS: dict[str, Callable[..., Any]] = {
        "Page.frameAttached": _on_frame_attached,
        "Page.frameNavigated": _on_frame_navigated,
        "Page.frameDetached": _on_frame_detached,
        "Target.attachedToTarget": _on_target_attached,
        "Target.detachedFromTarget": _on_target_detached,
    }
