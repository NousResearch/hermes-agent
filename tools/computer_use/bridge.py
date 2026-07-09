"""HTTP bridge backend for remote Computer Use.

This module is intentionally small and dependency-free. It lets a Hermes
backend whose tools run on one machine forward ``computer_use`` operations to a
local desktop host running ``hermes computer-use bridge``.

Typical shape for a safe remote Desktop setup:

1. Mac/local desktop:
     HERMES_COMPUTER_USE_BRIDGE_TOKEN=... \
       hermes computer-use bridge --host 127.0.0.1 --port 8765
2. Remote backend host:
     ssh -N -L 18765:127.0.0.1:8765 mac-host
     HERMES_COMPUTER_USE_BACKEND=bridge \
     HERMES_COMPUTER_USE_BRIDGE_URL=http://127.0.0.1:18765 \
     HERMES_COMPUTER_USE_BRIDGE_TOKEN=... \
       hermes serve ...

The bridge is an authenticated actuator surface. Do not bind it to a public
interface without a tunnel/VPN and an explicit token.
"""

from __future__ import annotations

import json
import os
import re
import secrets
import sys
import threading
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable, Dict, List, Optional, Tuple
from urllib.parse import urlparse

from tools.computer_use.backend import (
    ActionResult,
    CaptureResult,
    ComputerUseBackend,
    UIElement,
)

_DEFAULT_BRIDGE_PORT = 8765
_MAX_BODY_BYTES = 64 * 1024 * 1024
_TOKEN_HEADER = "X-Hermes-Computer-Use-Bridge-Token"
_BACKEND_METHODS = {
    "capture",
    "click",
    "drag",
    "scroll",
    "type_text",
    "key",
    "wait",
    "list_apps",
    "focus_app",
    "set_value",
}
_LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1", "[::1]"}
_BLOCKED_KEY_COMBOS = {
    frozenset({"cmd", "shift", "backspace"}),
    frozenset({"cmd", "option", "backspace"}),
    frozenset({"cmd", "ctrl", "q"}),
    frozenset({"cmd", "shift", "q"}),
    frozenset({"cmd", "option", "shift", "q"}),
    frozenset({"win", "l"}),
    frozenset({"ctrl", "option", "delete"}),
    frozenset({"ctrl", "option", "del"}),
    frozenset({"option", "f4"}),
}
_KEY_ALIASES = {
    "command": "cmd",
    "control": "ctrl",
    "alt": "option",
    "⌘": "cmd",
    "⌥": "option",
    "windows": "win",
    "super": "win",
    "meta": "win",
}
_BLOCKED_TYPE_PATTERNS = [
    re.compile(r"curl\s+[^|]*\|\s*bash", re.IGNORECASE),
    re.compile(r"curl\s+[^|]*\|\s*sh", re.IGNORECASE),
    re.compile(r"wget\s+[^|]*\|\s*bash", re.IGNORECASE),
    re.compile(r"\bsudo\s+rm\s+-[rf]", re.IGNORECASE),
    re.compile(r"\brm\s+-rf\s+/\s*$", re.IGNORECASE),
    re.compile(r":\s*\(\)\s*\{\s*:\|:\s*&\s*\}", re.IGNORECASE),
]


def _load_config_value(key: str) -> Optional[str]:
    """Best-effort read of ``computer_use.<key>`` from config.yaml."""
    try:
        from hermes_cli.config import load_config

        cfg = load_config() or {}
        cu = cfg.get("computer_use") or {}
        value = cu.get(key)
        return str(value).strip() if value is not None else None
    except Exception:
        return None


def bridge_url_from_env_or_config() -> str:
    return (
        os.environ.get("HERMES_COMPUTER_USE_BRIDGE_URL")
        or _load_config_value("bridge_url")
        or _load_config_value("remote_bridge_url")
        or ""
    ).strip().rstrip("/")


def bridge_token_from_env_or_config() -> str:
    return (
        os.environ.get("HERMES_COMPUTER_USE_BRIDGE_TOKEN")
        or _load_config_value("bridge_token")
        or _load_config_value("remote_bridge_token")
        or ""
    ).strip()


def bridge_backend_configured() -> bool:
    """True when enough config exists for the remote bridge backend to appear."""
    return bool(bridge_url_from_env_or_config() and bridge_token_from_env_or_config())


def _element_to_payload(element: UIElement) -> Dict[str, Any]:
    return {
        "index": element.index,
        "role": element.role,
        "label": element.label,
        "bounds": list(element.bounds),
        "app": element.app,
        "pid": element.pid,
        "window_id": element.window_id,
        "attributes": element.attributes,
        "element_token": element.element_token,
    }


def _element_from_payload(data: Dict[str, Any]) -> UIElement:
    bounds = data.get("bounds") or (0, 0, 0, 0)
    if not isinstance(bounds, (list, tuple)) or len(bounds) != 4:
        bounds = (0, 0, 0, 0)
    return UIElement(
        index=int(data.get("index") or 0),
        role=str(data.get("role") or ""),
        label=str(data.get("label") or ""),
        bounds=tuple(int(v or 0) for v in bounds),
        app=str(data.get("app") or ""),
        pid=int(data.get("pid") or 0),
        window_id=int(data.get("window_id") or 0),
        attributes=dict(data.get("attributes") or {}),
        element_token=(
            str(data.get("element_token"))
            if data.get("element_token") is not None
            else None
        ),
    )


def capture_to_payload(capture: CaptureResult) -> Dict[str, Any]:
    return {
        "mode": capture.mode,
        "width": capture.width,
        "height": capture.height,
        "png_b64": capture.png_b64,
        "elements": [_element_to_payload(e) for e in capture.elements],
        "app": capture.app,
        "window_title": capture.window_title,
        "png_bytes_len": capture.png_bytes_len,
        "image_mime_type": capture.image_mime_type,
    }


def capture_from_payload(data: Dict[str, Any]) -> CaptureResult:
    return CaptureResult(
        mode=str(data.get("mode") or "som"),
        width=int(data.get("width") or 0),
        height=int(data.get("height") or 0),
        png_b64=data.get("png_b64"),
        elements=[
            _element_from_payload(e)
            for e in data.get("elements") or []
            if isinstance(e, dict)
        ],
        app=str(data.get("app") or ""),
        window_title=str(data.get("window_title") or ""),
        png_bytes_len=int(data.get("png_bytes_len") or 0),
        image_mime_type=data.get("image_mime_type"),
    )


def action_to_payload(result: ActionResult) -> Dict[str, Any]:
    return {
        "ok": result.ok,
        "action": result.action,
        "message": result.message,
        "capture": capture_to_payload(result.capture) if result.capture else None,
        "meta": result.meta,
    }


def action_from_payload(data: Dict[str, Any]) -> ActionResult:
    capture_payload = data.get("capture")
    return ActionResult(
        ok=bool(data.get("ok")),
        action=str(data.get("action") or ""),
        message=str(data.get("message") or ""),
        capture=(
            capture_from_payload(capture_payload)
            if isinstance(capture_payload, dict)
            else None
        ),
        meta=dict(data.get("meta") or {}),
    )


def bridge_computer_use_status(
    url: Optional[str] = None,
    token: Optional[str] = None,
    timeout: Optional[float] = None,
) -> Dict[str, Any]:
    """Return the local desktop status exposed by a configured bridge.

    Used by Desktop's `/api/tools/computer-use/status` endpoint when the
    backend is configured with `HERMES_COMPUTER_USE_BACKEND=bridge`, so the UI
    reports the Mac/local actuator readiness rather than the remote server's
    local `cua-driver` state.
    """
    try:
        backend = HttpComputerUseBridgeBackend(url=url, token=token, timeout=timeout)
        data = backend._request("GET", "/v1/status")
        status = data.get("status")
        if isinstance(status, dict):
            return status
        raise RuntimeError("bridge returned invalid status payload")
    except Exception as exc:
        configured = bool(
            (url or bridge_url_from_env_or_config())
            and (token if token is not None else bridge_token_from_env_or_config())
        )
        return {
            "platform": "bridge",
            "platform_supported": True,
            "installed": configured,
            "version": None,
            "ready": False,
            "can_grant": False,
            "checks": [
                {
                    "label": "bridge",
                    "status": "failed",
                    "message": str(exc),
                }
            ],
            "source": None,
            "error": f"computer_use bridge unavailable: {exc}",
            "accessibility": None,
            "screen_recording": None,
            "screen_recording_capturable": None,
        }


class HttpComputerUseBridgeBackend(ComputerUseBackend):
    """ComputerUseBackend that forwards calls to an authenticated HTTP bridge."""

    def __init__(
        self,
        url: Optional[str] = None,
        token: Optional[str] = None,
        timeout: Optional[float] = None,
    ) -> None:
        self.url = (url or bridge_url_from_env_or_config()).strip().rstrip("/")
        self.token = token if token is not None else bridge_token_from_env_or_config()
        self.timeout = float(
            timeout
            if timeout is not None
            else os.environ.get("HERMES_COMPUTER_USE_BRIDGE_TIMEOUT", "30")
        )
        if not self.url:
            raise RuntimeError("HERMES_COMPUTER_USE_BRIDGE_URL is required")
        if not self.token:
            raise RuntimeError("HERMES_COMPUTER_USE_BRIDGE_TOKEN is required")

    def start(self) -> None:
        # Fail early with a clear auth/connectivity error instead of waiting for
        # the first model-issued tool call.
        self._request("GET", "/v1/status")

    def stop(self) -> None:
        return None

    def is_available(self) -> bool:
        try:
            self._request("GET", "/v1/status")
            return True
        except Exception:
            return False

    def _request(
        self,
        method: str,
        path: str,
        payload: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        body = None
        headers = {
            "Accept": "application/json",
            "Authorization": f"Bearer {self.token}",
            _TOKEN_HEADER: self.token,
        }
        if payload is not None:
            body = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"
        request = urllib.request.Request(
            self.url + path,
            data=body,
            headers=headers,
            method=method,
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                raw = response.read(_MAX_BODY_BYTES + 1)
        except urllib.error.HTTPError as exc:
            detail = ""
            try:
                detail = exc.read(4096).decode("utf-8", errors="replace")
            except Exception:
                pass
            raise RuntimeError(
                f"computer_use bridge HTTP {exc.code}: {detail or exc.reason}"
            ) from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"computer_use bridge unreachable: {exc.reason}") from exc
        if len(raw) > _MAX_BODY_BYTES:
            raise RuntimeError("computer_use bridge response too large")
        try:
            data = json.loads(raw.decode("utf-8"))
        except Exception as exc:
            raise RuntimeError("computer_use bridge returned invalid JSON") from exc
        if not isinstance(data, dict):
            raise RuntimeError("computer_use bridge returned a non-object response")
        if data.get("ok") is False:
            raise RuntimeError(str(data.get("error") or "computer_use bridge call failed"))
        return data

    def _call(self, method: str, args: Dict[str, Any]) -> Any:
        if method not in _BACKEND_METHODS:
            raise RuntimeError(f"unsupported bridge method: {method}")
        data = self._request("POST", "/v1/computer-use", {"method": method, "args": args})
        return data.get("result")

    def capture(self, mode: str = "som", app: Optional[str] = None) -> CaptureResult:
        result = self._call("capture", {"mode": mode, "app": app})
        if not isinstance(result, dict):
            raise RuntimeError("bridge capture returned invalid payload")
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
        if isinstance(result, dict):
            apps = result.get("apps")
        else:
            apps = result
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


def _make_local_backend() -> ComputerUseBackend:
    """Create the local actuator backend used by the bridge server.

    Do not call tools.computer_use.tool._get_backend() here: the bridge server
    may itself be launched with HERMES_COMPUTER_USE_BACKEND=bridge in its
    environment, which would recurse. The server side must always control the
    local machine directly.
    """
    from tools.computer_use.cua_backend import CuaDriverBackend

    backend = CuaDriverBackend()
    backend.start()
    return backend


def _json_response(handler: BaseHTTPRequestHandler, status: int, payload: Dict[str, Any]) -> None:
    raw = json.dumps(payload).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(raw)))
    handler.end_headers()
    handler.wfile.write(raw)


def _is_loopback_host(host: str) -> bool:
    return host in _LOOPBACK_HOSTS or host.startswith("127.")


def _authorised(handler: BaseHTTPRequestHandler, token: str) -> bool:
    auth = handler.headers.get("Authorization", "")
    header_token = handler.headers.get(_TOKEN_HEADER, "")
    supplied = ""
    if auth.lower().startswith("bearer "):
        supplied = auth[7:].strip()
    if not supplied:
        supplied = header_token.strip()
    return bool(supplied) and secrets.compare_digest(supplied, token)


def make_bridge_handler(
    *,
    token: str,
    backend_factory: Callable[[], ComputerUseBackend] = _make_local_backend,
):
    backend_lock = threading.Lock()
    call_lock = threading.Lock()
    backend_holder: Dict[str, Optional[ComputerUseBackend]] = {"backend": None}

    def get_backend() -> ComputerUseBackend:
        with backend_lock:
            backend = backend_holder["backend"]
            if backend is None:
                backend = backend_factory()
                backend_holder["backend"] = backend
            return backend

    class ComputerUseBridgeHandler(BaseHTTPRequestHandler):
        server_version = "HermesComputerUseBridge/1"

        def log_message(self, fmt: str, *args: Any) -> None:  # pragma: no cover - noise
            return None

        def _require_auth(self) -> bool:
            if _authorised(self, token):
                return True
            _json_response(self, 401, {"ok": False, "error": "unauthorised"})
            return False

        def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
            path = urlparse(self.path).path
            if path == "/healthz":
                _json_response(self, 200, {
                    "ok": True,
                    "service": "hermes-computer-use-bridge",
                    "platform": sys.platform,
                })
                return
            if path == "/v1/status":
                if not self._require_auth():
                    return
                from tools.computer_use.permissions import computer_use_status

                _json_response(self, 200, {"ok": True, "status": computer_use_status()})
                return
            _json_response(self, 404, {"ok": False, "error": "not found"})

        def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
            path = urlparse(self.path).path
            if path != "/v1/computer-use":
                _json_response(self, 404, {"ok": False, "error": "not found"})
                return
            if not self._require_auth():
                return
            try:
                length = int(self.headers.get("Content-Length") or "0")
            except ValueError:
                _json_response(self, 400, {"ok": False, "error": "bad Content-Length"})
                return
            if length < 0 or length > _MAX_BODY_BYTES:
                _json_response(self, 413, {"ok": False, "error": "request too large"})
                return
            try:
                body = self.rfile.read(length)
                request = json.loads(body.decode("utf-8")) if body else {}
            except Exception:
                _json_response(self, 400, {"ok": False, "error": "invalid JSON"})
                return
            if not isinstance(request, dict):
                _json_response(self, 400, {"ok": False, "error": "JSON body must be an object"})
                return
            method = str(request.get("method") or "")
            args = request.get("args") or {}
            if method not in _BACKEND_METHODS or not isinstance(args, dict):
                _json_response(self, 400, {"ok": False, "error": "bad method or args"})
                return
            try:
                with call_lock:
                    result = _dispatch_backend_method(get_backend(), method, args)
            except ValueError as exc:
                _json_response(self, 400, {"ok": False, "error": str(exc)})
                return
            except Exception as exc:
                _json_response(self, 500, {"ok": False, "error": str(exc)})
                return
            _json_response(self, 200, {"ok": True, "result": result})

    return ComputerUseBridgeHandler


def _blocked_type_pattern(text: str) -> Optional[str]:
    for pattern in _BLOCKED_TYPE_PATTERNS:
        if pattern.search(text):
            return pattern.pattern
    return None


def _canon_key_combo(keys: str) -> frozenset:
    parts = [p.strip().lower() for p in re.split(r"\s*\+\s*", keys) if p.strip()]
    return frozenset(_KEY_ALIASES.get(p, p) for p in parts)


def _blocked_key_combo(keys: str) -> Optional[List[str]]:
    combo = _canon_key_combo(keys)
    for blocked in _BLOCKED_KEY_COMBOS:
        if blocked.issubset(combo) and len(blocked) <= len(combo):
            return sorted(blocked)
    return None


def _dispatch_backend_method(backend: ComputerUseBackend, method: str, args: Dict[str, Any]) -> Any:
    if method == "capture":
        return capture_to_payload(
            backend.capture(
                mode=str(args.get("mode") or "som"),
                app=args.get("app") if args.get("app") is not None else None,
            )
        )
    if method == "click":
        return action_to_payload(backend.click(
            element=args.get("element"),
            x=args.get("x"),
            y=args.get("y"),
            button=str(args.get("button") or "left"),
            click_count=int(args.get("click_count") or 1),
            modifiers=args.get("modifiers"),
        ))
    if method == "drag":
        from_xy = args.get("from_xy")
        to_xy = args.get("to_xy")
        return action_to_payload(backend.drag(
            from_element=args.get("from_element"),
            to_element=args.get("to_element"),
            from_xy=tuple(from_xy) if isinstance(from_xy, (list, tuple)) else None,
            to_xy=tuple(to_xy) if isinstance(to_xy, (list, tuple)) else None,
            button=str(args.get("button") or "left"),
            modifiers=args.get("modifiers"),
        ))
    if method == "scroll":
        return action_to_payload(backend.scroll(
            direction=str(args.get("direction") or "down"),
            amount=int(args.get("amount") or 3),
            element=args.get("element"),
            x=args.get("x"),
            y=args.get("y"),
            modifiers=args.get("modifiers"),
        ))
    if method == "type_text":
        text = str(args.get("text") or "")
        blocked = _blocked_type_pattern(text)
        if blocked:
            raise ValueError(f"blocked pattern in type text: {blocked!r}")
        return action_to_payload(backend.type_text(text))
    if method == "key":
        keys = str(args.get("keys") or "")
        blocked = _blocked_key_combo(keys)
        if blocked:
            raise ValueError(f"blocked key combo: {blocked}")
        return action_to_payload(backend.key(keys))
    if method == "wait":
        return action_to_payload(backend.wait(float(args.get("seconds") or 0)))
    if method == "list_apps":
        return {"apps": backend.list_apps()}
    if method == "focus_app":
        return action_to_payload(backend.focus_app(
            str(args.get("app") or ""),
            raise_window=bool(args.get("raise_window")),
        ))
    if method == "set_value":
        return action_to_payload(backend.set_value(
            str(args.get("value") or ""),
            element=args.get("element"),
        ))
    raise RuntimeError(f"unsupported bridge method: {method}")


def run_bridge_server(
    *,
    host: str = "127.0.0.1",
    port: int = _DEFAULT_BRIDGE_PORT,
    token: Optional[str] = None,
    token_env: str = "HERMES_COMPUTER_USE_BRIDGE_TOKEN",
    allow_non_loopback: bool = False,
) -> int:
    token_value = (token or os.environ.get(token_env) or "").strip()
    if not token_value:
        print(
            "Computer Use bridge refused to start: provide --token or set "
            f"{token_env}. This HTTP service can control the desktop.",
            file=sys.stderr,
        )
        return 2
    if not allow_non_loopback and not _is_loopback_host(host):
        print(
            "Computer Use bridge refused non-loopback bind without "
            "--allow-non-loopback. Prefer 127.0.0.1 plus SSH/VPN tunnel.",
            file=sys.stderr,
        )
        return 2

    handler = make_bridge_handler(token=token_value)
    httpd = ThreadingHTTPServer((host, int(port)), handler)
    print(
        f"Hermes Computer Use bridge listening on http://{host}:{port} "
        "(token required)",
        flush=True,
    )
    try:
        httpd.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt:
        return 130
    finally:
        httpd.server_close()
    return 0
