"""The owner side of cooperative session attach.

``hermes --tui --resume <id>`` on a session another local runtime owns does not start a competing
runtime: ``hermes_cli/shared_session_attach.py`` reads the owner's lease, and when its metadata
carries ``shared_runtime_url`` it asks that origin for ``/api/session-attach`` and dials the
returned WebSocket instead. The TUI then becomes a second viewer of the SAME live session
(``methods_session._resume_reuse_live`` attaches it alongside the clients already streaming), so a
Desktop chat and a terminal see each other's turns live.

This module is the one place the runtime says what it advertises. The web server registers its
loopback origin after it binds (``web_server.start_server``); ``session_lifecycle`` reads it when
it writes a lease. A runtime that never registered advertises nothing and keeps today's refusal.
"""
from __future__ import annotations

import ipaddress
from urllib.parse import urlsplit

_origin: str | None = None


def advertise_shared_runtime(origin: str | None) -> None:
    """Record the loopback ``http://host:port`` other local clients may attach through.

    Only an explicit loopback origin is accepted — the attach client refuses anything else
    (``_local_origin``), and a non-loopback bind must never hand its credential to the network.
    ``None`` withdraws the advertisement.
    """
    global _origin
    if origin is None:
        _origin = None
        return
    parts = urlsplit(origin)
    try:
        loopback = ipaddress.ip_address(parts.hostname or "").is_loopback
    except ValueError:
        loopback = False
    if parts.scheme != "http" or not loopback or not parts.port or parts.path not in ("", "/") or parts.query:
        raise ValueError(f"shared runtime origin must be an explicit loopback http origin, got {origin!r}")
    _origin = f"http://{parts.hostname}:{parts.port}"


def shared_runtime_origin() -> str | None:
    """The advertised origin, or None when this runtime does not accept attachments."""
    return _origin
