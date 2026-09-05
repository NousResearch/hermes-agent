"""Standalone CUA host bridge launcher — no Hermes dependencies.

Runs on any machine with cua-driver + Python 3.11+. Starts Xvfb (if no DISPLAY),
spawns cua-driver in MCP stdio mode, and wraps it behind the authenticated
streamable-HTTP bridge so a remote Hermes agent can drive this desktop.

Usage:
    export HERMES_CUA_REMOTE_TOKEN=$(python3 -c "import secrets; print(secrets.token_hex(32))")
    export CUA_DRIVER_PERMISSION_MODE=standard
    python3 host_bridge_standalone.py --port 8765 \
        --allowed-hosts localhost,127.0.0.1 \
        --allowed-origins http://localhost:8765,http://127.0.0.1:8765
"""

from __future__ import annotations

import argparse
import contextlib
import os
import shutil
import subprocess
import sys
from collections.abc import AsyncIterator, Sequence
from typing import Any, AsyncContextManager


def _ensure_xvfb() -> str:
    """Start Xvfb if no DISPLAY is set. Returns the DISPLAY value."""
    display = os.environ.get("DISPLAY", "")
    if display:
        print(f"Using existing DISPLAY={display}")
        return display

    display = ":99"
    os.environ["DISPLAY"] = display
    # Start Xvfb if not already running
    xvfb = shutil.which("Xvfb")
    if not xvfb:
        raise RuntimeError("DISPLAY is not set and Xvfb is not installed (apt-get install xvfb)")
    proc = subprocess.Popen(
        [xvfb, display, "-screen", "0", "1920x1080x24"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    print(f"Started Xvfb on {display} (PID {proc.pid})")

    # Start openbox window manager
    openbox = shutil.which("openbox")
    if openbox:
        subprocess.Popen(
            [openbox, "--replace"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        print("Started openbox window manager")

    # Wait a moment for X to be ready
    import time
    time.sleep(2)
    return display


def _resolve_cua_driver() -> str:
    """Find cua-driver binary."""
    # Check PATH first, then common locations
    path = shutil.which("cua-driver")
    if path:
        return path
    for candidate in [
        os.path.expanduser("~/.local/bin/cua-driver"),
        "/usr/local/bin/cua-driver",
    ]:
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate
    raise RuntimeError(
        "cua-driver not found. Install: curl -fsSL "
        "https://raw.githubusercontent.com/trycua/cua/main/libs/cua-driver/scripts/install.sh | bash"
    )


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


def _build_child_session_context(driver_cmd: str) -> AsyncContextManager[Any]:
    """Build a standard-mode local cua-driver stdio session context."""
    env = dict(os.environ)
    # Strip sensitive vars
    for key in ["HERMES_CUA_REMOTE_TOKEN", "CUA_DRIVER_DANGEROUSLY_BYPASS_APPROVALS"]:
        env.pop(key, None)
    env["CUA_DRIVER_PERMISSION_MODE"] = "standard"
    env["CUA_DRIVER_RS_TELEMETRY_ENABLED"] = "0"

    return _cua_driver_session_context(
        command=driver_cmd,
        args=["mcp"],
        env=env,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Standalone CUA host bridge")
    parser.add_argument("--port", type=int, required=True, help="Port to listen on")
    parser.add_argument("--bind", default="127.0.0.1", help="Bind address (default 127.0.0.1)")
    parser.add_argument("--allowed-hosts", required=True, help="Comma-separated allowed Host headers")
    parser.add_argument("--allowed-origins", required=True, help="Comma-separated allowed Origin headers")
    parser.add_argument("--session-idle-timeout", type=int, default=300, help="Session idle timeout in seconds")
    args = parser.parse_args()

    # Ensure X display
    _ensure_xvfb()

    # Validate permission mode
    permission_mode = os.environ.get("CUA_DRIVER_PERMISSION_MODE", "").strip().lower()
    if permission_mode not in {"", "standard"}:
        raise RuntimeError("the computer-use bridge requires standard permission mode")
    bypass = os.environ.get("CUA_DRIVER_DANGEROUSLY_BYPASS_APPROVALS", "").strip().lower()
    if bypass not in {"", "0", "false", "no", "off"}:
        raise RuntimeError("the computer-use bridge does not allow bypassing approvals")

    # Validate token
    token = os.environ.get("HERMES_CUA_REMOTE_TOKEN", "")
    try:
        token_bytes = token.encode("ascii")
    except UnicodeEncodeError as exc:
        raise RuntimeError("HERMES_CUA_REMOTE_TOKEN must contain only ASCII characters") from exc
    if len(token_bytes) < 32:
        raise RuntimeError("HERMES_CUA_REMOTE_TOKEN must contain at least 32 bytes")
    # Don't pop the token — the standalone script may need it for reference

    # Resolve cua-driver
    driver_cmd = _resolve_cua_driver()
    print(f"Using cua-driver: {driver_cmd}")

    # Import the bridge (must be in same directory or PYTHONPATH)
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from host_validation import validate_security_allowlists
    from host_bridge import create_host_bridge_app

    hosts, origins = validate_security_allowlists(
        args.allowed_hosts.split(","),
        args.allowed_origins.split(","),
    )

    child_context = _build_child_session_context(driver_cmd)
    app = create_host_bridge_app(
        child_session_context=child_context,
        bearer_token=token,
        allowed_hosts=hosts,
        allowed_origins=origins,
        session_idle_timeout=args.session_idle_timeout,
    )

    import uvicorn
    print(f"Starting host bridge on {args.bind}:{args.port}")
    uvicorn.run(app, host=args.bind, port=args.port, log_level="info")


if __name__ == "__main__":
    main()