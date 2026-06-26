"""Standalone gateway WebSocket host — the durable anchor.

The tui_gateway already serves the full session over WebSocket via
``tui_gateway.ws.handle_ws`` (it reuses ``server.dispatch`` verbatim — same RPC
methods, slash commands, approval/clarify/sudo flows, and agent events as the
stdio path). Today that handler is only mounted inside the heavyweight dashboard
web server. This module is the minimal, local-TUI equivalent: a tiny uvicorn app
that mounts just ``/api/ws`` so the orchestrator can run the gateway as its own
process, independent of any renderer.

Auth: a single multi-use ``internal`` credential (from
``HERMES_TUI_WS_INTERNAL_CREDENTIAL``) gates the socket — the same model the
dashboard uses for server-spawned children that must survive reconnects. Bound
to loopback only; the credential prevents another local user's process from
attaching to this session.

Env contract (set by the orchestrator):
    HERMES_TUI_WS_HOST                bind host (default 127.0.0.1)
    HERMES_TUI_WS_PORT                bind port (required; orchestrator picks one)
    HERMES_TUI_WS_INTERNAL_CREDENTIAL multi-use credential the renderer presents
"""

from __future__ import annotations

import logging
import os
import sys

_log = logging.getLogger(__name__)


def _bool_env(name: str, default: bool = False) -> bool:
    raw = (os.environ.get(name) or "").strip().lower()
    if not raw:
        return default
    return raw not in {"0", "false", "no", "off"}


def build_app(internal_credential: str):
    """Build the minimal Starlette/FastAPI app mounting /api/ws.

    Imported lazily so the orchestrator module stays importable in unit tests
    that never start a real host.
    """
    try:
        from starlette.applications import Starlette
        from starlette.routing import WebSocketRoute
        from starlette.websockets import WebSocket
    except ImportError as exc:  # pragma: no cover - starlette is on the install path
        raise RuntimeError(
            "tui_gateway.ws_host requires starlette/uvicorn (already a Hermes "
            "dashboard dependency). Install the web extra."
        ) from exc

    from tui_gateway.ws import handle_ws

    async def _ws_endpoint(ws: "WebSocket") -> None:
        # Gate on the multi-use internal credential. Loopback-only bind plus a
        # process-lifetime credential is the same threat model the dashboard
        # uses for its server-spawned PTY child (ws_tickets.consume_internal_credential).
        presented = ws.query_params.get("internal", "")
        if not internal_credential or presented != internal_credential:
            await ws.close(code=4401)  # 4401: app-level unauthorized
            return
        await handle_ws(ws)

    return Starlette(routes=[WebSocketRoute("/api/ws", _ws_endpoint)])


def main(argv: list[str] | None = None) -> int:
    host = os.environ.get("HERMES_TUI_WS_HOST", "127.0.0.1")
    port_raw = os.environ.get("HERMES_TUI_WS_PORT", "")
    cred = os.environ.get("HERMES_TUI_WS_INTERNAL_CREDENTIAL", "")

    if not port_raw.isdigit():
        print("tui_gateway.ws_host: HERMES_TUI_WS_PORT must be set to a port", file=sys.stderr)
        return 64  # EX_USAGE
    if not cred:
        print("tui_gateway.ws_host: HERMES_TUI_WS_INTERNAL_CREDENTIAL must be set", file=sys.stderr)
        return 64
    port = int(port_raw)

    try:
        import uvicorn
    except ImportError:  # pragma: no cover
        print("tui_gateway.ws_host: uvicorn not installed", file=sys.stderr)
        return 69  # EX_UNAVAILABLE

    app = build_app(cred)
    # log_level kept quiet: the gateway's own logging covers session events; we
    # don't want uvicorn access logs racing the TUI on the same terminal.
    uvicorn.run(app, host=host, port=port, log_level="warning", ws="websockets")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
