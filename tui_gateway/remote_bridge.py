"""Opt-in WebSocket listener for remote control of a live TUI gateway.

The normal TUI process model is Node/Ink talking to ``tui_gateway.entry`` over
stdio.  This module adds a *second* backend-only listener inside that same
Python process so another client (mobile/web/native) can attach to the live
in-memory sessions over the same JSON-RPC protocol used by stdio.

The listener is deliberately off by default.  Binding anywhere other than
loopback requires an explicit bearer token, and WebSocket upgrades enforce Host
and Origin checks to keep browser DNS-rebinding attacks out of localhost-bound
bridges.
"""

from __future__ import annotations

import hmac
import importlib
import os
import sys
import threading
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence
from urllib.parse import urlsplit

_DEFAULT_HOST = "127.0.0.1"
_DEFAULT_PORT = 8769
_DEFAULT_PATH = "/api/tui/ws"
_TOKEN_HEADER = "x-hermes-tui-remote-token"

_LOOPBACK_HOST_VALUES: frozenset[str] = frozenset({"localhost", "127.0.0.1", "::1"})
_ALL_INTERFACE_HOST_VALUES: frozenset[str] = frozenset({"0.0.0.0", "::"})


class RemoteBridgeConfigError(RuntimeError):
    """Configuration refused because it would expose the TUI unsafely."""


@dataclass(frozen=True)
class RemoteBridgeConfig:
    enabled: bool = False
    host: str = _DEFAULT_HOST
    port: int = _DEFAULT_PORT
    path: str = _DEFAULT_PATH
    token: str = ""
    trusted_origins: tuple[str, ...] = field(default_factory=tuple)

    def validate(self) -> None:
        if not self.enabled:
            return
        if not self.host.strip():
            raise RemoteBridgeConfigError("tui_remote_bridge.host must not be empty")
        if not (1 <= int(self.port) <= 65535):
            raise RemoteBridgeConfigError("tui_remote_bridge.port must be between 1 and 65535")
        if not self.path.startswith("/") or "?" in self.path or "#" in self.path:
            raise RemoteBridgeConfigError(
                "tui_remote_bridge.path must be an absolute path without query or fragment"
            )
        if not _is_loopback_bind(self.host) and not self.token:
            raise RemoteBridgeConfigError(
                "tui_remote_bridge.token is required when host is not loopback"
            )

    def public_info(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "host": self.host,
            "port": self.port,
            "path": self.path,
            "requires_token": bool(self.token),
            "trusted_origins": list(self.trusted_origins),
            "url": f"ws://{_url_host(self.host)}:{self.port}{self.path}",
        }


@dataclass
class RemoteBridgeHandle:
    config: RemoteBridgeConfig
    server: Any
    thread: threading.Thread

    def public_info(self) -> dict[str, Any]:
        return self.config.public_info()

    def stop(self) -> None:
        try:
            self.server.should_exit = True
        except Exception:
            pass


def _url_host(host: str) -> str:
    return f"[{host}]" if ":" in host and not host.startswith("[") else host


def _truthy(value: Any) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _as_int(value: Any, default: int) -> int:
    try:
        return int(str(value).strip())
    except Exception:
        return default


def _as_origins(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        parts = value.split(",")
    elif isinstance(value, Sequence):
        parts = [str(v) for v in value]
    else:
        parts = [str(value)]
    normalized = []
    for part in parts:
        origin = _normalize_origin(part)
        if origin:
            normalized.append(origin)
    return tuple(dict.fromkeys(normalized))


def _cfg_bool(node: Mapping[str, Any], key: str, default: bool) -> bool:
    if key not in node:
        return default
    return _truthy(node.get(key))


def resolve_remote_bridge_config(
    cfg: Mapping[str, Any] | None = None,
    environ: Mapping[str, str] | None = None,
) -> RemoteBridgeConfig:
    """Resolve bridge settings from config + environment overrides.

    Config surface::

        tui_remote_bridge:
          enabled: false
          host: 127.0.0.1
          port: 8769
          path: /api/tui/ws
          token: ""
          trusted_origins: []

    Environment overrides use ``HERMES_TUI_REMOTE_BRIDGE_*``.  The legacy-ish
    short toggle ``HERMES_TUI_REMOTE=1`` is also accepted for quick manual
    testing, but the full name is preferred for scripts.
    """
    if cfg is None:
        try:
            from hermes_cli.config import read_raw_config

            cfg = read_raw_config() or {}
        except Exception:
            cfg = {}
    environ = environ or os.environ

    node = cfg.get("tui_remote_bridge", {}) if isinstance(cfg, Mapping) else {}
    if not isinstance(node, Mapping):
        node = {}

    enabled = _cfg_bool(node, "enabled", False)
    env_enabled = (
        environ.get("HERMES_TUI_REMOTE_BRIDGE")
        or environ.get("HERMES_TUI_REMOTE_CONTROL")
        or environ.get("HERMES_TUI_REMOTE")
    )
    if env_enabled is not None and str(env_enabled).strip():
        enabled = _truthy(env_enabled)

    host = str(environ.get("HERMES_TUI_REMOTE_BRIDGE_HOST") or node.get("host") or _DEFAULT_HOST).strip()
    port = _as_int(environ.get("HERMES_TUI_REMOTE_BRIDGE_PORT") or node.get("port"), _DEFAULT_PORT)
    path = str(environ.get("HERMES_TUI_REMOTE_BRIDGE_PATH") or node.get("path") or _DEFAULT_PATH).strip()
    token = str(environ.get("HERMES_TUI_REMOTE_BRIDGE_TOKEN") or node.get("token") or "").strip()
    origins = _as_origins(
        environ.get("HERMES_TUI_REMOTE_BRIDGE_ORIGINS")
        if environ.get("HERMES_TUI_REMOTE_BRIDGE_ORIGINS") is not None
        else node.get("trusted_origins")
    )

    return RemoteBridgeConfig(
        enabled=enabled,
        host=host,
        port=port,
        path=path or _DEFAULT_PATH,
        token=token,
        trusted_origins=origins,
    )


def _host_only(host_header: str) -> str:
    """Strip optional port/brackets from a Host header-like value."""
    h = (host_header or "").strip()
    if not h:
        return ""
    if h.startswith("["):
        close = h.find("]")
        return h[1:close].lower() if close != -1 else h.strip("[]").lower()
    return (h.rsplit(":", 1)[0] if ":" in h else h).lower()


def _is_loopback_bind(host: str) -> bool:
    return host.strip().lower() in _LOOPBACK_HOST_VALUES


def is_accepted_host(host_header: str, bound_host: str) -> bool:
    """Return True when the Host header targets the bound interface.

    This mirrors the dashboard's Host guard: loopback binds only accept loopback
    hostnames, explicit non-loopback binds require an exact host match, and
    0.0.0.0/:: all-interface binds accept any Host because the operator has
    explicitly opted into network exposure (token still required by validate()).
    """
    host = _host_only(host_header)
    if not host:
        return False

    bound = (bound_host or "").strip().lower()
    if bound in _ALL_INTERFACE_HOST_VALUES:
        return True
    if bound in _LOOPBACK_HOST_VALUES:
        return host in _LOOPBACK_HOST_VALUES
    return host == bound


def _normalize_origin(raw: str) -> str:
    raw = (raw or "").strip().rstrip("/")
    if not raw:
        return ""
    try:
        parsed = urlsplit(raw)
    except Exception:
        return ""
    if not parsed.scheme or not parsed.netloc:
        return ""
    return f"{parsed.scheme.lower()}://{parsed.netloc.lower()}"


def _origin_host(raw: str) -> str:
    try:
        return (urlsplit(raw).hostname or "").lower()
    except Exception:
        return ""


def is_accepted_origin(
    origin: str,
    *,
    bound_host: str,
    host_header: str,
    trusted_origins: Sequence[str] = (),
) -> bool:
    """Validate browser Origin for WebSocket upgrades.

    Native/mobile WebSocket stacks commonly omit ``Origin``; those are accepted
    after Host/token checks.  Browser upgrades with Origin are accepted only
    when the origin is explicitly trusted or matches the endpoint host boundary.
    """
    if not origin:
        return True

    normalized = _normalize_origin(origin)
    if not normalized:
        return False
    trusted = {_normalize_origin(o) for o in trusted_origins if _normalize_origin(o)}
    if normalized in trusted:
        return True

    origin_host = _origin_host(normalized)
    if not origin_host:
        return False

    bound = (bound_host or "").strip().lower()
    if bound in _LOOPBACK_HOST_VALUES:
        return origin_host in _LOOPBACK_HOST_VALUES
    if bound in _ALL_INTERFACE_HOST_VALUES:
        # For 0.0.0.0/::, the actual request Host is the only useful browser
        # boundary we can validate at this layer.
        return origin_host == _host_only(host_header)
    return origin_host == bound


def _mapping_get(mapping: Any, name: str, default: str = "") -> str:
    try:
        return str(mapping.get(name, default) or "")
    except Exception:
        return default


def _extract_token(ws: Any) -> str:
    query = getattr(ws, "query_params", {}) or {}
    headers = getattr(ws, "headers", {}) or {}

    query_token = _mapping_get(query, "token")
    if query_token:
        return query_token

    header_token = _mapping_get(headers, _TOKEN_HEADER)
    if header_token:
        return header_token

    auth = _mapping_get(headers, "authorization")
    if auth.lower().startswith("bearer "):
        return auth[7:].strip()
    return ""


async def authorize_ws(ws: Any, config: RemoteBridgeConfig) -> bool:
    """Validate Host/Origin/token before accepting the WebSocket."""
    headers = getattr(ws, "headers", {}) or {}
    host_header = _mapping_get(headers, "host")
    origin = _mapping_get(headers, "origin")

    close_code = 4403
    ok = is_accepted_host(host_header, config.host) and is_accepted_origin(
        origin,
        bound_host=config.host,
        host_header=host_header,
        trusted_origins=config.trusted_origins,
    )

    if ok and config.token:
        close_code = 4401
        ok = hmac.compare_digest(_extract_token(ws).encode(), config.token.encode())

    if not ok:
        try:
            await ws.close(code=close_code)
        except Exception:
            pass
        return False
    return True


async def handle_remote_ws(ws: Any, config: RemoteBridgeConfig) -> None:
    if not await authorize_ws(ws, config):
        return

    from tui_gateway.ws import BRIDGE_ALLOWED_METHODS, handle_ws

    await handle_ws(ws, allowed_methods=BRIDGE_ALLOWED_METHODS)


def build_app(config: RemoteBridgeConfig) -> Any:
    """Build a tiny FastAPI app exposing only health + the bridge WS."""
    fastapi = importlib.import_module("fastapi")
    FastAPI = getattr(fastapi, "FastAPI")
    # Resolve the class eagerly so FastAPI can inspect the endpoint signature,
    # but keep the import dynamic so base installs without dashboard deps do not
    # trigger static import diagnostics.
    _websocket_type = getattr(fastapi, "WebSocket")

    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)

    @app.get("/healthz")
    async def healthz() -> dict[str, Any]:
        return {"ok": True, "remote_bridge": config.public_info()}

    async def remote_ws(ws: Any) -> None:
        await handle_remote_ws(ws, config)

    remote_ws.__annotations__["ws"] = _websocket_type
    app.websocket(config.path)(remote_ws)

    return app


def _ensure_server_deps() -> Any:
    try:
        importlib.import_module("fastapi")
        return importlib.import_module("uvicorn")
    except ImportError:
        from tools.lazy_deps import ensure

        ensure("tool.dashboard")
        importlib.import_module("fastapi")
        return importlib.import_module("uvicorn")


def start_remote_bridge(config: RemoteBridgeConfig | None = None) -> RemoteBridgeHandle | None:
    """Start the bridge listener in a daemon thread when enabled."""
    config = config or resolve_remote_bridge_config()
    if not config.enabled:
        return None
    config.validate()

    uvicorn = _ensure_server_deps()
    app = build_app(config)
    server_config = uvicorn.Config(
        app,
        host=config.host,
        port=int(config.port),
        log_level="warning",
        lifespan="off",
    )
    uvicorn_server = uvicorn.Server(server_config)
    thread = threading.Thread(
        target=uvicorn_server.run,
        name="hermes-tui-remote-bridge",
        daemon=True,
    )
    thread.start()
    return RemoteBridgeHandle(config=config, server=uvicorn_server, thread=thread)


def start_remote_bridge_if_enabled() -> RemoteBridgeHandle | None:
    try:
        return start_remote_bridge()
    except Exception as exc:
        print(f"[tui-remote-bridge] not started: {exc}", file=sys.stderr, flush=True)
        return None
