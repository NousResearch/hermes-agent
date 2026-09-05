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
import time
from collections.abc import AsyncIterator, Mapping, Sequence
from typing import Any, AsyncContextManager

# Fail-closed gate, mirrored from host_bridge_cli.py (this launcher is
# deliberately dependency-free, so the policy is duplicated, not imported):
# a non-loopback plaintext HTTP bind exposes the bearer token to network
# observers; HTTPS via a reverse proxy is the production path and the env var
# is the explicit "I accept the risk" acknowledgement.
_BRIDGE_ALLOW_PLAINTEXT_ENV = "HERMES_CUA_BRIDGE_ALLOW_PLAINTEXT"
_LOOPBACK_BINDS = frozenset({"127.0.0.1", "localhost", "::1", "[::1]"})
# :99-:109 stays clear of a real desktop session's :0 and of nested X servers.
_DISPLAY_RANGE = range(99, 110)
_XVFB_READY_TIMEOUT_SECONDS = 15.0
_XVFB_READY_POLL_INTERVAL_SECONDS = 0.2

# Xvfb/openbox need only the standard process environment; everything else from
# the parent (API keys, shell opts, ...) is dropped. LC_* travels as a prefix.
_PASSTHROUGH_ENV_VARS = frozenset({
    "PATH", "HOME", "DISPLAY", "LANG", "TMPDIR", "USER", "LOGNAME", "SHELL",
    "TERM", "XAUTHORITY", "DBUS_SESSION_BUS_ADDRESS", "XDG_RUNTIME_DIR",
    "XDG_CONFIG_HOME", "XDG_DATA_HOME", "XDG_CACHE_HOME",
    "CUA_DRIVER_PERMISSION_MODE", "CUA_DRIVER_RS_TELEMETRY_ENABLED",
})
_PASSTHROUGH_ENV_PREFIXES = ("LC_",)
_SECRET_ENV_VARS = ("HERMES_CUA_REMOTE_TOKEN", "CUA_DRIVER_DANGEROUSLY_BYPASS_APPROVALS")

# X-stack procs this launcher started; terminated after the bridge exits.
_spawned_procs: list[subprocess.Popen] = []
# True only when *we* set DISPLAY (it was unset at launch); guards restore.
_display_env_modified = False


def _sanitize_standalone_env(env: Mapping[str, str]) -> dict[str, str]:
    """Child-process env without secrets: standard vars + LC_* only.

    The parent keeps its own token (it builds the bridge verifier); a spawned
    child must never inherit it — /proc/<pid>/environ is world-readable to the
    same user and token-free children keep process listings clean.
    """
    child = {
        key: value
        for key, value in env.items()
        if key in _PASSTHROUGH_ENV_VARS or key.startswith(_PASSTHROUGH_ENV_PREFIXES)
    }
    for key in _SECRET_ENV_VARS:
        # Belt and braces: these are not in the passthrough set, but an explicit
        # pop documents the invariant and survives future list edits.
        child.pop(key, None)
    return child


def _pick_free_display(*, lock_dir: str = "/tmp", socket_dir: str = "/tmp/.X11-unix") -> str:
    """First display in :99-:109 with neither a lock file nor a socket."""
    for candidate in _DISPLAY_RANGE:
        if os.path.exists(os.path.join(lock_dir, f".X{candidate}-lock")):
            continue
        if os.path.exists(os.path.join(socket_dir, f"X{candidate}")):
            continue
        return f":{candidate}"
    raise RuntimeError(
        f"no free display found in range :{_DISPLAY_RANGE.start}-:{_DISPLAY_RANGE.stop - 1}; "
        "stop the stale Xvfb instances or clear their /tmp/.X*-lock files"
    )


def _wait_for_x_ready(display: str, child_env: dict[str, str]) -> None:
    """Poll xdpyinfo until the X server answers, bounded by 15s."""
    xdpyinfo = shutil.which("xdpyinfo")
    if not xdpyinfo:
        # No probe available; give X the same grace period the old sleep(2) did.
        time.sleep(2)
        return
    deadline = time.monotonic() + _XVFB_READY_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        probe = subprocess.run(
            [xdpyinfo, "-display", display],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=child_env,
        )
        if probe.returncode == 0:
            return
        time.sleep(_XVFB_READY_POLL_INTERVAL_SECONDS)
    raise RuntimeError(f"X server on {display} did not become ready within {_XVFB_READY_TIMEOUT_SECONDS:.0f}s")


def _ensure_xvfb() -> str:
    """Ensure an X server exists; returns the DISPLAY value.

    Linux-only: macOS/Windows drive their native display stacks, so an unset
    DISPLAY there is not an error and nothing is spawned.
    """
    global _display_env_modified
    display = os.environ.get("DISPLAY", "")
    if display:
        print(f"Using existing DISPLAY={display}")
        return display
    if not sys.platform.startswith("linux"):
        return display

    xvfb = shutil.which("Xvfb")
    if not xvfb:
        raise RuntimeError("DISPLAY is not set and Xvfb is not installed (apt-get install xvfb)")
    display = _pick_free_display()
    _display_env_modified = True
    os.environ["DISPLAY"] = display
    # Children must never see the bearer token (see _sanitize_standalone_env);
    # sanitize AFTER the DISPLAY set above so X clients (openbox) inherit it.
    child_env = _sanitize_standalone_env(os.environ)
    proc = subprocess.Popen(
        [xvfb, display, "-screen", "0", "1920x1080x24"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=child_env,
    )
    _spawned_procs.append(proc)
    print(f"Started Xvfb on {display} (PID {proc.pid})")

    openbox = shutil.which("openbox")
    if openbox:
        wm = subprocess.Popen(
            [openbox, "--replace"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=child_env,
        )
        _spawned_procs.append(wm)
        print("Started openbox window manager")

    _wait_for_x_ready(display, child_env)
    return display


def _shutdown_xvfb() -> None:
    """Terminate the X-stack procs we started and restore the DISPLAY env."""
    global _display_env_modified
    for proc in _spawned_procs:
        with contextlib.suppress(Exception):
            proc.terminate()
            proc.wait(timeout=5)
    _spawned_procs.clear()
    if _display_env_modified:
        # We set DISPLAY ourselves; the launcher found it unset.
        os.environ.pop("DISPLAY", None)
        _display_env_modified = False


def _ensure_bind_security(bind: str) -> None:
    """Refuse non-loopback plaintext binds unless explicitly acknowledged."""
    if bind in _LOOPBACK_BINDS or os.environ.get(_BRIDGE_ALLOW_PLAINTEXT_ENV) == "1":
        return
    raise RuntimeError(
        f"refusing to serve the computer-use bridge over plaintext HTTP on non-loopback bind {bind!r}; "
        f"terminate TLS with a reverse proxy (recommended) or set {_BRIDGE_ALLOW_PLAINTEXT_ENV}=1 "
        "to acknowledge that the bearer token may be observed on the network"
    )


def _split_list_arg(value: str) -> list[str]:
    """Split a comma-separated CLI arg, trimming whitespace and dropping empties."""
    return [entry.strip() for entry in value.split(",") if entry.strip()]


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
    # Strip sensitive vars so the driver child never sees the bridge token or
    # approval bypass; keep telemetry off for the third-party binary.
    for key in ["HERMES_CUA_REMOTE_TOKEN", "CUA_DRIVER_DANGEROUSLY_BYPASS_APPROVALS"]:
        env.pop(key, None)
    env["CUA_DRIVER_PERMISSION_MODE"] = "standard"
    env["CUA_DRIVER_RS_TELEMETRY_ENABLED"] = "0"

    return _cua_driver_session_context(
        command=driver_cmd,
        args=["mcp"],
        env=env,
    )


def _serve_app(app: Any, host: str, port: int) -> None:
    """Run the bridge app. Tiny seam so tests can monkeypatch without uvicorn."""
    import uvicorn

    uvicorn.run(app, host=host, port=port, log_level="info")


def main() -> None:
    parser = argparse.ArgumentParser(description="Standalone CUA host bridge")
    parser.add_argument("--port", type=int, required=True, help="Port to listen on")
    parser.add_argument("--bind", default="127.0.0.1", help="Bind address (default 127.0.0.1)")
    parser.add_argument("--allowed-hosts", required=True,
                        help="Comma-separated allowed Host headers (include the port, e.g. host:8765 — "
                             "the Host header carries it and comparison is exact)")
    parser.add_argument("--allowed-origins", required=True, help="Comma-separated allowed Origin headers")
    # MCP recommendation for interactive sessions: long model turns must not be
    # reaped mid-call. Matches create_host_bridge_app's default.
    parser.add_argument("--session-idle-timeout", type=int, default=1800, help="Session idle timeout in seconds")
    args = parser.parse_args()

    # Validate everything cheap and fail-closed BEFORE spawning any X stack, so
    # a config error can never leave an orphaned Xvfb behind.
    _ensure_bind_security(args.bind)
    if not 1 <= int(args.port) <= 65535:
        raise RuntimeError("bridge port must be between 1 and 65535")
    permission_mode = os.environ.get("CUA_DRIVER_PERMISSION_MODE", "").strip().lower()
    if permission_mode not in {"", "standard"}:
        raise RuntimeError("the computer-use bridge requires standard permission mode")
    bypass = os.environ.get("CUA_DRIVER_DANGEROUSLY_BYPASS_APPROVALS", "").strip().lower()
    if bypass not in {"", "0", "false", "no", "off"}:
        raise RuntimeError("the computer-use bridge does not allow bypassing approvals")

    token = os.environ.get("HERMES_CUA_REMOTE_TOKEN", "")
    try:
        token_bytes = token.encode("ascii")
    except UnicodeEncodeError as exc:
        raise RuntimeError("HERMES_CUA_REMOTE_TOKEN must contain only ASCII characters") from exc
    if len(token_bytes) < 32:
        raise RuntimeError("HERMES_CUA_REMOTE_TOKEN must contain at least 32 bytes")
    # Don't pop the token — the standalone script may need it for reference

    try:
        _ensure_xvfb()
        driver_cmd = _resolve_cua_driver()
        print(f"Using cua-driver: {driver_cmd}")

        # Import the bridge (must be in same directory or PYTHONPATH)
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from host_validation import validate_security_allowlists
        from host_bridge import create_host_bridge_app

        hosts, origins = validate_security_allowlists(
            _split_list_arg(args.allowed_hosts),
            _split_list_arg(args.allowed_origins),
        )

        child_context = _build_child_session_context(driver_cmd)
        app = create_host_bridge_app(
            child_session_context=child_context,
            bearer_token=token,
            allowed_hosts=hosts,
            allowed_origins=origins,
            session_idle_timeout=args.session_idle_timeout,
        )

        print(f"Starting host bridge on {args.bind}:{args.port}")
        _serve_app(app, args.bind, int(args.port))
    finally:
        _shutdown_xvfb()


if __name__ == "__main__":
    main()