"""Size caps for the TUI gateway's stdio transport and its command fields.

Two layers, both env-overridable for ops debugging:

1. **Stdio frame cap** (``HERMES_TUI_MAX_FRAME_BYTES``) — ``entry.py`` and
   ``slash_worker.py`` read their JSON-RPC line through :func:`read_frame`, which
   never holds more than the cap in memory: an over-cap line is drained in
   cap-sized chunks and discarded, then reported once. The default matches the
   WebSocket transport (uvicorn ``ws_max_size`` =
   ``hermes_cli.web_server_boundary._DESKTOP_ATTACHMENT_WS_MAX_BYTES``), so an
   attachment-bearing RPC that is legal over WS is legal over stdio too. The WS
   leg needs no extra check here: uvicorn rejects over-size frames before the
   app ever sees them.

2. **Command-field cap** (``HERMES_TUI_MAX_FIELD_BYTES``) — ``command.dispatch``
   (``name``/``arg``), ``slash.exec`` and ``shell.exec`` (``command``) reject a
   field whose UTF-8 size exceeds the cap with a normal JSON-RPC 4000 error,
   before it is resolved, forwarded to the slash worker, or handed to a shell.

Limits are read at call time, so tests and operators can change the env vars
without reloading this module.
"""

from __future__ import annotations

import os

# Same budget as the WebSocket leg (see module docstring). Kept as a literal so
# this module stays import-light; tests pin it to the boundary constant.
_DEFAULT_FRAME_BYTES = 384 * 1024 * 1024
# Comfortably above any typed slash command or pasted argument; well below a
# frame. (Linux caps a single execve argument at 128 KiB anyway.)
_DEFAULT_FIELD_BYTES = 1024 * 1024


def _env_positive_int(name: str, default: int) -> int:
    try:
        value = int(os.environ.get(name, "") or default)
    except ValueError:
        return default
    return value if value > 0 else default


def max_frame_bytes() -> int:
    return _env_positive_int("HERMES_TUI_MAX_FRAME_BYTES", _DEFAULT_FRAME_BYTES)


def max_field_bytes() -> int:
    return _env_positive_int("HERMES_TUI_MAX_FIELD_BYTES", _DEFAULT_FIELD_BYTES)


class FrameTooLarge(ValueError):
    """An stdio JSON-RPC line exceeded :func:`max_frame_bytes` (already drained)."""

    def __init__(self, size: int, limit: int) -> None:
        super().__init__(f"frame too large: {size} characters (limit {limit})")
        self.size = size
        self.limit = limit


def read_frame(stream) -> str:
    """Read one newline-terminated line from a text ``stream`` with bounded memory.

    Returns ``""`` on EOF, like ``readline()``. Raises :class:`FrameTooLarge` for a line
    longer than the cap *after* consuming the rest of it, so the next call starts on the
    next request instead of parsing the tail of the oversized one. The overflow is
    discarded chunk by chunk, never accumulated.

    The cap is applied in characters (``TextIOWrapper.readline(size)``): a line of N
    UTF-8 bytes never has more than N characters, so every frame within the byte budget
    passes.
    """
    limit = max_frame_bytes()
    line = stream.readline(limit + 1)
    if len(line) <= limit:
        return line
    size = len(line)
    del line
    while True:
        chunk = stream.readline(limit)
        if not chunk:
            break
        size += len(chunk)
        if chunk.endswith("\n"):
            break
    raise FrameTooLarge(size, limit)


def field_error(name: str, value) -> str | None:
    """Return an error message when string ``value`` exceeds the field cap, else ``None``.

    Non-string values are left to the handler's own validation.
    """
    if not isinstance(value, str):
        return None
    limit = max_field_bytes()
    # Cheap pre-check: a str of L code points encodes to at most 4L bytes.
    if len(value) * 4 <= limit:
        return None
    size = len(value.encode("utf-8", errors="surrogatepass"))
    if size <= limit:
        return None
    return f"{name} too long: {size} bytes (limit {limit})"
