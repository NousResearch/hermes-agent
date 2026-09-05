"""Interactive-session launcher for the authenticated CUA host bridge.

Runs on any OS with a cua-driver installation: Linux (Xvfb or real X session),
macOS, or Windows (interactive user session). The bridge wraps a local
cua-driver MCP session and exposes it over authenticated streamable HTTP so
a remote Hermes agent gateway can drive the desktop.
"""

from __future__ import annotations

import contextlib
import importlib
import os
import sys
from collections.abc import AsyncIterator, Sequence
from typing import Any, AsyncContextManager

from tools.computer_use.host_validation import validate_security_allowlists

_CUA_REMOTE_TOKEN_ENV = "HERMES_CUA_REMOTE_TOKEN"
_CUA_PERMISSION_MODE_ENV = "CUA_DRIVER_PERMISSION_MODE"
_CUA_BYPASS_APPROVALS_ENV = "CUA_DRIVER_DANGEROUSLY_BYPASS_APPROVALS"
# Fail-closed gate: a non-loopback plaintext HTTP bind exposes the bearer token
# to network observers; HTTPS via a reverse proxy is the production path, and
# the env var is the explicit "I accept the risk" acknowledgement.
_BRIDGE_ALLOW_PLAINTEXT_ENV = "HERMES_CUA_BRIDGE_ALLOW_PLAINTEXT"
_LOOPBACK_BINDS = frozenset({"127.0.0.1", "localhost", "::1", "[::1]"})
# MCP recommendation for interactive sessions: long model turns must not be
# reaped mid-call (screenshots, UI waits). Matches create_host_bridge_app's
# default so the bridge behaves the same however it is launched.
_DEFAULT_SESSION_IDLE_TIMEOUT_SECONDS = 1800


def _validate_standard_permission_environment() -> None:
    permission_mode = os.environ.get(_CUA_PERMISSION_MODE_ENV, "").strip().lower()
    if permission_mode not in {"", "standard"}:
        raise RuntimeError("the computer-use bridge requires standard permission mode")

    bypass = os.environ.get(_CUA_BYPASS_APPROVALS_ENV, "").strip().lower()
    if bypass not in {"", "0", "false", "no", "off"}:
        raise RuntimeError("the computer-use bridge does not allow bypassing approvals")


def _ensure_interactive_session() -> None:
    """On Windows, refuse to run in Session 0 (services session — no desktop).
    On Linux, warn if DISPLAY is unset (cua-driver needs an X server or Xvfb).
    On macOS, the interactive session is the norm."""
    if sys.platform == "win32":
        import ctypes
        session_id = ctypes.c_ulong()
        kernel32 = getattr(ctypes, "windll").kernel32
        if not kernel32.ProcessIdToSessionId(os.getpid(), ctypes.byref(session_id)):
            raise OSError("could not determine the current Windows session")
        if int(session_id.value) == 0:
            raise RuntimeError(
                "the computer-use bridge cannot run in Windows Session 0; "
                "launch it from the interactive signed-in user's session"
            )
    elif sys.platform.startswith("linux"):
        if not os.environ.get("DISPLAY"):
            raise RuntimeError(
                "DISPLAY is not set — the computer-use bridge needs an X server. "
                "Start Xvfb (e.g. `Xvfb :99 -screen 0 1920x1080x24` and `export DISPLAY=:99`)"
            )


def _build_child_session_context() -> AsyncContextManager[Any]:
    """Build a standard-mode local cua-driver stdio session context."""
    # Import from the defining modules: _resolve_mcp_invocation and
    # cua_driver_install_hint are not facade attributes (in-tree compat
    # pointers are off-limits), only resolve_cua_driver_cmd is re-exported.
    from tools.computer_use.cua_backend import cua_driver_child_env
    from tools.computer_use.cua_backend_driver import (
        _resolve_mcp_invocation,
        cua_driver_install_hint,
        resolve_cua_driver_cmd,
    )
    from tools.environments.local import _sanitize_subprocess_env

    driver_cmd = resolve_cua_driver_cmd()
    if not driver_cmd:
        raise RuntimeError(cua_driver_install_hint())
    command, args = _resolve_mcp_invocation(driver_cmd)
    child_env = _sanitize_subprocess_env(cua_driver_child_env())
    child_env.pop(_CUA_REMOTE_TOKEN_ENV, None)
    child_env.pop(_CUA_BYPASS_APPROVALS_ENV, None)
    child_env[_CUA_PERMISSION_MODE_ENV] = "standard"
    # Security parity with the standalone launcher: the third-party driver
    # must never phone home from a bridge-spawned session.
    child_env["CUA_DRIVER_RS_TELEMETRY_ENABLED"] = "0"

    return _cua_driver_session_context(
        command=command,
        args=args,
        env=child_env,
    )


def _create_host_bridge_app(**kwargs: Any) -> Any:
    from tools.computer_use.host_bridge import create_host_bridge_app

    return create_host_bridge_app(**kwargs)


@contextlib.asynccontextmanager
async def _cua_driver_session_context(
    *,
    command: str,
    args: Sequence[str],
    env: dict[str, str],
) -> AsyncIterator[Any]:
    """Own driver stdio and ClientSession contexts in the bridge lifespan task."""
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    params = StdioServerParameters(command=command, args=list(args), env=env)
    async with stdio_client(params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            yield session


def _serve_app(app: Any, host: str, port: int) -> None:
    """Run the bridge app. Tiny seam so tests can monkeypatch without uvicorn."""
    import uvicorn

    uvicorn.run(app, host=host, port=port, log_level="info")


def _ensure_bind_security(bind: str) -> None:
    """Refuse non-loopback plaintext binds unless explicitly acknowledged.

    The bridge guards a desktop-control session with a bearer token; over
    plaintext HTTP on a routable interface that token travels in cleartext.
    Fail closed: HTTPS via a reverse proxy is the supported path, and
    HERMES_CUA_BRIDGE_ALLOW_PLAINTEXT=1 is the explicit risk acknowledgement.
    """
    if bind in _LOOPBACK_BINDS or os.environ.get(_BRIDGE_ALLOW_PLAINTEXT_ENV) == "1":
        return
    raise RuntimeError(
        f"refusing to serve the computer-use bridge over plaintext HTTP on non-loopback bind {bind!r}; "
        f"terminate TLS with a reverse proxy (recommended) or set {_BRIDGE_ALLOW_PLAINTEXT_ENV}=1 "
        "to acknowledge that the bearer token may be observed on the network"
    )


def run_host_bridge(
    *,
    allowed_hosts: Sequence[str],
    allowed_origins: Sequence[str],
    port: int,
    bind: str = "127.0.0.1",
) -> None:
    """Run the authenticated CUA host bridge on the local machine.

    Wraps a local cua-driver session and exposes it over authenticated
    streamable HTTP so a remote Hermes agent can drive this desktop.
    Works on Linux (Xvfb/X11), macOS, and Windows (interactive session).
    """
    _ensure_interactive_session()
    _validate_standard_permission_environment()
    _ensure_bind_security(bind)
    hosts, origins = validate_security_allowlists(allowed_hosts, allowed_origins)
    if not 1 <= int(port) <= 65535:
        raise RuntimeError("bridge port must be between 1 and 65535")

    token = os.environ.get(_CUA_REMOTE_TOKEN_ENV, "")
    try:
        token_bytes = token.encode("ascii")
    except UnicodeEncodeError as exc:
        raise RuntimeError(f"{_CUA_REMOTE_TOKEN_ENV} must contain only ASCII characters") from exc
    if len(token_bytes) < 32:
        raise RuntimeError(f"{_CUA_REMOTE_TOKEN_ENV} must contain at least 32 bytes")
    os.environ.pop(_CUA_REMOTE_TOKEN_ENV, None)

    from tools.lazy_deps import ensure as _lazy_ensure

    _lazy_ensure("tool.computer_use", prompt=False)
    importlib.invalidate_caches()
    child_context = _build_child_session_context()
    app = _create_host_bridge_app(
        child_session_context=child_context,
        bearer_token=token,
        allowed_hosts=hosts,
        allowed_origins=origins,
        session_idle_timeout=_DEFAULT_SESSION_IDLE_TIMEOUT_SECONDS,
    )
    _serve_app(app, bind, int(port))