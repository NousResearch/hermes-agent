"""Reverse WebSocket broker for Desktop-managed local Computer Use.

This is the productised path for Hermes Desktop connected to a remote backend:

- Desktop starts a local loopback `hermes computer-use bridge` sidecar next to
  the app.
- Desktop opens an authenticated WebSocket back to the remote backend.
- The remote backend's normal `computer_use` tool calls this module, which sends
  request frames over that WebSocket and waits for Desktop to proxy them to the
  local sidecar.

The remote backend never needs to dial the user's laptop directly, so this works
behind NAT and avoids requiring manual SSH reverse tunnels.
"""
from __future__ import annotations

import asyncio
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
import json
import logging
import threading
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple

from tools.computer_use.backend import ActionResult, CaptureResult, ComputerUseBackend
from tools.computer_use.bridge import (  # shared allow-list + payload codecs
    _BACKEND_METHODS,
    action_from_payload,
    capture_from_payload,
)

_log = logging.getLogger(__name__)
_DEFAULT_TIMEOUT_S = 30.0
_MAX_FRAME_BYTES = 20 * 1024 * 1024


class _PendingCall:
    def __init__(self) -> None:
        self.event = threading.Event()
        self.future: Optional[asyncio.Future[Any]] = None
        self.result: Any = None
        self.error: Optional[str] = None


@dataclass(frozen=True)
class DesktopBridgeScope:
    """Trusted broker key for one authenticated principal and Hermes profile."""

    provider: str
    principal: str
    profile: str

    def validate(self) -> "DesktopBridgeScope":
        if not self.provider or not self.principal or not self.profile:
            raise RuntimeError(
                "Desktop Computer Use bridge scope is incomplete; "
                "authenticated principal and profile are required"
            )
        return self


@dataclass
class _BridgeConnection:
    ws: Any
    loop: asyncio.AbstractEventLoop
    client_id: str
    connected_at: float
    pending: Dict[str, _PendingCall] = field(default_factory=dict)


_CALLER_SCOPE: ContextVar[Optional[DesktopBridgeScope]] = ContextVar(
    "desktop_bridge_caller_scope",
    default=None,
)


def set_desktop_bridge_caller(
    principal: Optional[Tuple[str, str]], *, profile: Optional[str] = None
) -> Token:
    """Bind one immutable, verified bridge scope for an agent execution."""
    scope = None
    if principal is not None:
        provider, subject = principal
        if profile is None:
            try:
                from hermes_cli.profiles import get_active_profile_name

                profile = get_active_profile_name() or "default"
            except Exception:
                profile = ""
        scope = DesktopBridgeScope(
            str(provider).strip(),
            str(subject).strip(),
            str(profile).strip(),
        ).validate()
    return _CALLER_SCOPE.set(scope)


def reset_desktop_bridge_caller(token: Token) -> None:
    _CALLER_SCOPE.reset(token)


def current_desktop_bridge_scope() -> DesktopBridgeScope:
    """Resolve the exact trusted caller scope for the current agent context."""
    scope = _CALLER_SCOPE.get()
    if scope is None:
        raise RuntimeError(
            "Desktop Computer Use bridge caller is not authenticated in this context"
        )
    return scope.validate()


class DesktopBridgeBroker:
    """Thread-safe, fail-closed broker for scoped Desktop bridge clients."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._connections: Dict[DesktopBridgeScope, Dict[str, _BridgeConnection]] = {}

    def _scope(self, scope: Optional[DesktopBridgeScope]) -> DesktopBridgeScope:
        return (scope or current_desktop_bridge_scope()).validate()

    def _matching_connection(
        self, scope: Optional[DesktopBridgeScope]
    ) -> Tuple[DesktopBridgeScope, _BridgeConnection]:
        resolved = self._scope(scope)
        with self._lock:
            matches = list(self._connections.get(resolved, {}).values())
        if not matches:
            raise RuntimeError(
                "Desktop Computer Use bridge is not connected for the "
                "authenticated principal and profile"
            )
        if len(matches) != 1:
            raise RuntimeError(
                "Desktop Computer Use bridge scope is ambiguous; "
                "close duplicate Desktop connections and retry"
            )
        return resolved, matches[0]

    def is_connected(self, scope: Optional[DesktopBridgeScope] = None) -> bool:
        try:
            self._matching_connection(scope)
            return True
        except RuntimeError:
            return False

    def connection_info(
        self, scope: Optional[DesktopBridgeScope] = None
    ) -> Dict[str, Any]:
        try:
            resolved, connection = self._matching_connection(scope)
        except RuntimeError as exc:
            with self._lock:
                total = sum(len(items) for items in self._connections.values())
            return {
                "connected": False,
                "connection_count": total,
                "error": str(exc),
            }
        with self._lock:
            return {
                "connected": True,
                "client_id": connection.client_id,
                "connected_at": connection.connected_at,
                "pending": len(connection.pending),
                "profile": resolved.profile,
            }

    async def handle_ws(self, ws: Any, scope: DesktopBridgeScope) -> None:
        """Accept and service a Desktop bridge WebSocket until it disconnects."""
        scope = self._scope(scope)
        await ws.accept()
        connection = _BridgeConnection(
            ws=ws,
            loop=asyncio.get_running_loop(),
            client_id=uuid.uuid4().hex,
            connected_at=time.time(),
        )

        with self._lock:
            self._connections.setdefault(scope, {})[connection.client_id] = connection

        _log.info(
            "desktop computer_use bridge connected client=%s profile=%s",
            connection.client_id,
            scope.profile,
        )
        try:
            while True:
                raw = await ws.receive_text()
                if len(raw.encode("utf-8", errors="ignore")) > _MAX_FRAME_BYTES:
                    await ws.close(code=1009, reason="frame too large")
                    return
                try:
                    message = json.loads(raw)
                except json.JSONDecodeError:
                    _log.warning("desktop computer_use bridge sent invalid JSON")
                    continue
                self._handle_message(connection, message)
        except Exception as exc:
            # Starlette raises WebSocketDisconnect on normal close. Avoid importing
            # it just for an isinstance; debug is enough for expected disconnects.
            _log.debug("desktop computer_use bridge disconnected: %s", exc)
        finally:
            self._disconnect(scope, connection)

    def _handle_message(self, connection: _BridgeConnection, message: Any) -> None:
        if not isinstance(message, dict):
            return
        call_id = str(message.get("id") or "")
        if not call_id:
            return
        with self._lock:
            pending = connection.pending.get(call_id)
        if pending is None:
            return
        if message.get("ok") is False:
            pending.error = str(message.get("error") or "Desktop bridge call failed")
        else:
            pending.result = message.get("result")
        if pending.future is not None and not pending.future.done():
            if pending.error:
                pending.future.set_exception(RuntimeError(pending.error))
            else:
                pending.future.set_result(pending.result)
        pending.event.set()

    def _disconnect(
        self, scope: DesktopBridgeScope, connection: _BridgeConnection
    ) -> None:
        with self._lock:
            scoped = self._connections.get(scope)
            if scoped is None or scoped.get(connection.client_id) is not connection:
                return
            scoped.pop(connection.client_id, None)
            if not scoped:
                self._connections.pop(scope, None)
            pending = list(connection.pending.values())
            connection.pending.clear()
        for call in pending:
            call.error = "Desktop Computer Use bridge disconnected"
            if call.future is not None and not call.future.done():
                call.future.set_exception(RuntimeError(call.error))
            call.event.set()
        _log.info(
            "desktop computer_use bridge disconnected client=%s profile=%s",
            connection.client_id,
            scope.profile,
        )

    def _make_pending(
        self, scope: Optional[DesktopBridgeScope]
    ) -> Tuple[str, _PendingCall, _BridgeConnection]:
        _resolved, connection = self._matching_connection(scope)
        with self._lock:
            call_id = uuid.uuid4().hex
            pending = _PendingCall()
            connection.pending[call_id] = pending
        return call_id, pending, connection

    def request(
        self,
        payload: Dict[str, Any],
        timeout: Optional[float] = None,
        scope: Optional[DesktopBridgeScope] = None,
    ) -> Any:
        """Send one request frame to Desktop and synchronously wait for the reply."""
        call_id, pending, connection = self._make_pending(scope)

        try:
            running_loop = asyncio.get_running_loop()
        except RuntimeError:
            running_loop = None
        if running_loop is connection.loop:
            with self._lock:
                connection.pending.pop(call_id, None)
            raise RuntimeError("Desktop Computer Use bridge sync request cannot run on the WebSocket event loop")

        frame = {"id": call_id, **payload}
        try:
            future = asyncio.run_coroutine_threadsafe(
                connection.ws.send_text(json.dumps(frame)), connection.loop
            )
            future.result(timeout=5)
            if not pending.event.wait(timeout or _DEFAULT_TIMEOUT_S):
                raise RuntimeError("Desktop Computer Use bridge timed out")
            if pending.error:
                raise RuntimeError(pending.error)
            return pending.result
        finally:
            with self._lock:
                connection.pending.pop(call_id, None)

    async def request_async(
        self,
        payload: Dict[str, Any],
        timeout: Optional[float] = None,
        scope: Optional[DesktopBridgeScope] = None,
    ) -> Any:
        """Send one request frame from the WebSocket event loop and await its reply."""
        call_id, pending, connection = self._make_pending(scope)
        pending.future = asyncio.get_running_loop().create_future()
        frame = {"id": call_id, **payload}
        try:
            await connection.ws.send_text(json.dumps(frame))
            return await asyncio.wait_for(pending.future, timeout or _DEFAULT_TIMEOUT_S)
        finally:
            with self._lock:
                connection.pending.pop(call_id, None)


_BROKER = DesktopBridgeBroker()


def desktop_bridge_connected(scope: Optional[DesktopBridgeScope] = None) -> bool:
    return _BROKER.is_connected(scope)


def desktop_bridge_info(
    scope: Optional[DesktopBridgeScope] = None,
) -> Dict[str, Any]:
    return _BROKER.connection_info(scope)


async def handle_desktop_bridge_ws(
    ws: Any, *, provider: str, principal: str, profile: str
) -> None:
    scope = DesktopBridgeScope(provider, principal, profile).validate()
    await _BROKER.handle_ws(ws, scope)


def _offline_status(message: str) -> Dict[str, Any]:
    return {
        "platform": "desktop-bridge",
        "platform_supported": True,
        "installed": False,
        "version": None,
        "ready": False,
        "can_grant": False,
        "checks": [
            {
                "label": "Desktop bridge",
                "status": "failed",
                "message": message,
            }
        ],
        "source": {
            "note": "Remote backend is waiting for a Hermes Desktop local Computer Use bridge.",
        },
        "error": message,
        "accessibility": None,
        "screen_recording": None,
        "screen_recording_capturable": None,
        "bridge": {"kind": "desktop", "connected": False},
    }


def desktop_bridge_computer_use_status(timeout: float = 5.0) -> Dict[str, Any]:
    """Return the local Desktop host's Computer Use status via the live bridge."""
    if not desktop_bridge_connected():
        info = desktop_bridge_info()
        return _offline_status(
            str(info.get("error") or "Desktop Computer Use bridge is not connected")
        )
    try:
        result = _BROKER.request({"type": "status"}, timeout=timeout)
        return _status_from_bridge_result(result)
    except Exception as exc:
        return _offline_status(f"Desktop Computer Use bridge unavailable: {exc}")


async def desktop_bridge_computer_use_status_async(timeout: float = 5.0) -> Dict[str, Any]:
    """Async status variant for FastAPI routes running on the bridge WS loop."""
    if not desktop_bridge_connected():
        info = desktop_bridge_info()
        return _offline_status(
            str(info.get("error") or "Desktop Computer Use bridge is not connected")
        )
    try:
        result = await _BROKER.request_async({"type": "status"}, timeout=timeout)
        return _status_from_bridge_result(result)
    except Exception as exc:
        return _offline_status(f"Desktop Computer Use bridge unavailable: {exc}")


def _status_from_bridge_result(result: Any) -> Dict[str, Any]:
    if not isinstance(result, dict):
        raise RuntimeError("Desktop bridge returned invalid status payload")
    status = dict(result)
    checks = list(status.get("checks") or [])
    checks.insert(
        0,
        {
            "label": "Desktop bridge",
            "status": "ok",
            "message": "Connected through Hermes Desktop on the local machine.",
        },
    )
    status["checks"] = checks
    source = status.get("source") if isinstance(status.get("source"), dict) else {}
    status["source"] = {
        **source,
        "note": "Computer Use is routed through the connected Hermes Desktop app on this machine.",
    }
    status["bridge"] = {"kind": "desktop", **desktop_bridge_info()}
    return status


class DesktopComputerUseBridgeBackend(ComputerUseBackend):
    """ComputerUseBackend that calls the live Hermes Desktop bridge WebSocket."""

    def __init__(self, timeout: Optional[float] = None) -> None:
        self.timeout = float(timeout or _DEFAULT_TIMEOUT_S)

    def start(self) -> None:
        if not desktop_bridge_connected():
            info = desktop_bridge_info()
            raise RuntimeError(
                str(info.get("error") or "Desktop Computer Use bridge is not connected")
            )
        self._request("status", timeout=5.0)

    def stop(self) -> None:
        return None

    def is_available(self) -> bool:
        return desktop_bridge_connected()

    def _request(self, request_type: str, **payload: Any) -> Any:
        return _BROKER.request({"type": request_type, **payload}, timeout=self.timeout)

    def _call(self, method: str, args: Dict[str, Any]) -> Any:
        if method not in _BACKEND_METHODS:
            raise RuntimeError(f"unsupported Desktop bridge method: {method}")
        return self._request("computer-use", method=method, args=args)

    def capture(self, mode: str = "som", app: Optional[str] = None) -> CaptureResult:
        result = self._call("capture", {"mode": mode, "app": app})
        if not isinstance(result, dict):
            raise RuntimeError("Desktop bridge capture returned invalid payload")
        return capture_from_payload(result)

    def click(
        self,
        *,
        element: Optional[int] = None,
        x: Optional[int] = None,
        y: Optional[int] = None,
        button: str = "left",
        click_count: int = 1,
        modifiers: Optional[List[str]] = None,
    ) -> ActionResult:
        result = self._call("click", {
            "element": element,
            "x": x,
            "y": y,
            "button": button,
            "click_count": click_count,
            "modifiers": modifiers,
        })
        return action_from_payload(dict(result or {}))

    def drag(
        self,
        *,
        from_element: Optional[int] = None,
        to_element: Optional[int] = None,
        from_xy: Optional[Tuple[int, int]] = None,
        to_xy: Optional[Tuple[int, int]] = None,
        button: str = "left",
        modifiers: Optional[List[str]] = None,
    ) -> ActionResult:
        result = self._call("drag", {
            "from_element": from_element,
            "to_element": to_element,
            "from_xy": list(from_xy) if from_xy else None,
            "to_xy": list(to_xy) if to_xy else None,
            "button": button,
            "modifiers": modifiers,
        })
        return action_from_payload(dict(result or {}))

    def scroll(
        self,
        *,
        direction: str,
        amount: int = 3,
        element: Optional[int] = None,
        x: Optional[int] = None,
        y: Optional[int] = None,
        modifiers: Optional[List[str]] = None,
    ) -> ActionResult:
        result = self._call("scroll", {
            "direction": direction,
            "amount": amount,
            "element": element,
            "x": x,
            "y": y,
            "modifiers": modifiers,
        })
        return action_from_payload(dict(result or {}))

    def type_text(self, text: str) -> ActionResult:
        result = self._call("type_text", {"text": text})
        return action_from_payload(dict(result or {}))

    def key(self, keys: str) -> ActionResult:
        result = self._call("key", {"keys": keys})
        return action_from_payload(dict(result or {}))

    def list_apps(self) -> List[Dict[str, Any]]:
        result = self._call("list_apps", {})
        apps = result.get("apps") if isinstance(result, dict) else result
        return [dict(app) for app in apps or [] if isinstance(app, dict)]

    def focus_app(self, app: str, raise_window: bool = False) -> ActionResult:
        result = self._call("focus_app", {"app": app, "raise_window": raise_window})
        return action_from_payload(dict(result or {}))

    def set_value(self, value: str, element: Optional[int] = None) -> ActionResult:
        result = self._call("set_value", {"value": value, "element": element})
        return action_from_payload(dict(result or {}))

    def wait(self, seconds: float) -> ActionResult:
        result = self._call("wait", {"seconds": seconds})
        return action_from_payload(dict(result or {}))
