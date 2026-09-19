"""MCP wire-frame logging tap for live client<->server JSON-RPC diagnostics (mcpsnoop pattern).

Logs every request, response, and notification exchanged between Hermes and MCP servers
to ~/.hermes/logs/mcp/<server>.jsonl with direction, UTC timestamp, and raw payload.
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Tuple

logger = logging.getLogger("tools.mcp_wire_log")

_log_file_locks: dict[str, threading.Lock] = {}
_global_lock = threading.Lock()


def is_wire_log_enabled(server_name: str, config: Optional[dict] = None) -> bool:
    """Return True if wire-frame logging is enabled for this server or globally.

    Precedence:
    1. HERMES_MCP_WIRE_LOG environment variable ("1", "true", "yes", "on" vs "0", "false", "no", "off")
    2. Server-specific config: ``mcp_servers.<name>.wire_log``
    3. Global config: ``mcp.wire_log`` in config.yaml
    Default: False (zero model-tool footprint, zero cache impact, zero overhead).
    """
    env_val = os.environ.get("HERMES_MCP_WIRE_LOG", "").strip().lower()
    if env_val in ("1", "true", "yes", "on"):
        return True
    if env_val in ("0", "false", "no", "off"):
        return False

    if config and isinstance(config, dict) and config.get("wire_log") is not None:
        return bool(config["wire_log"])

    try:
        from hermes_cli.config import load_config_readonly
        app_cfg = load_config_readonly() or {}
        mcp_section = app_cfg.get("mcp")
        if isinstance(mcp_section, dict) and mcp_section.get("wire_log") is not None:
            return bool(mcp_section["wire_log"])
        srv_cfg = (app_cfg.get("mcp_servers") or {}).get(server_name)
        if isinstance(srv_cfg, dict) and srv_cfg.get("wire_log") is not None:
            return bool(srv_cfg["wire_log"])
    except Exception:
        pass

    return False


def _sanitize_filename(name: str) -> str:
    """Sanitize server name for safe filesystem path."""
    return re.sub(r"[^\w\-.]", "_", name)


def get_mcp_wire_log_dir() -> Path:
    """Return ~/.hermes/logs/mcp directory, creating it if needed."""
    try:
        from hermes_constants import get_hermes_home
        base = get_hermes_home()
    except Exception:
        base = Path.home() / ".hermes"
    log_dir = Path(base) / "logs" / "mcp"
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir


def get_mcp_wire_log_path(server_name: str) -> Path:
    """Return the Path to ~/.hermes/logs/mcp/<server>.jsonl."""
    return get_mcp_wire_log_dir() / f"{_sanitize_filename(server_name)}.jsonl"


def _extract_frame_data(item: Any) -> Tuple[Any, Optional[Any]]:
    """Extract serializable (frame, metadata) from an SDK item or model."""
    msg = getattr(item, "message", item)
    metadata = getattr(item, "metadata", None)

    if isinstance(msg, Exception):
        frame = {"error": type(msg).__name__, "message": str(msg)}
    elif hasattr(msg, "model_dump"):
        try:
            frame = msg.model_dump(by_alias=True, mode="json")
        except Exception:
            frame = json.loads(msg.model_dump_json(by_alias=True))
    elif isinstance(msg, dict):
        frame = msg
    elif isinstance(msg, str):
        try:
            frame = json.loads(msg)
        except Exception:
            frame = {"raw": msg}
    else:
        frame = {"raw": repr(msg)}

    meta_dict = None
    if metadata is not None:
        if hasattr(metadata, "model_dump"):
            try:
                meta_dict = metadata.model_dump(by_alias=True, mode="json")
            except Exception:
                meta_dict = str(metadata)
        elif isinstance(metadata, dict):
            meta_dict = metadata
        else:
            meta_dict = str(metadata)

    return frame, meta_dict


def log_wire_frame(server_name: str, direction: str, item: Any) -> None:
    """Append a JSON-RPC wire-frame record to the server's sidecar log.

    Direction is 'send' (client -> server) or 'recv' (server -> client).
    Fail-safe: logging exceptions are swallowed so diagnostic logging never
    disrupts connection handling.
    """
    try:
        frame, metadata = _extract_frame_data(item)
        entry: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "server": server_name,
            "direction": direction,
            "frame": frame,
        }
        if metadata:
            entry["metadata"] = metadata

        line = json.dumps(entry, ensure_ascii=False) + "\n"
        log_path = get_mcp_wire_log_path(server_name)

        with _global_lock:
            if server_name not in _log_file_locks:
                _log_file_locks[server_name] = threading.Lock()
            file_lock = _log_file_locks[server_name]

        with file_lock:
            with open(log_path, "a", encoding="utf-8", buffering=1) as f:
                f.write(line)
    except Exception:
        logger.debug("Failed to write MCP wire frame for %s", server_name, exc_info=True)


class _WireTapReceiveStream:
    """Async stream wrapper tapping incoming frames (recv)."""

    def __init__(self, server_name: str, stream: Any) -> None:
        self._server_name = server_name
        self._stream = stream

    def __getattr__(self, name: str) -> Any:
        return getattr(self._stream, name)

    async def __aenter__(self) -> _WireTapReceiveStream:
        if hasattr(self._stream, "__aenter__"):
            await self._stream.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> Any:
        if hasattr(self._stream, "__aexit__"):
            return await self._stream.__aexit__(exc_type, exc_val, exc_tb)
        return None

    def __aiter__(self) -> _WireTapReceiveStream:
        return self

    async def receive(self) -> Any:
        item = await self._stream.receive()
        log_wire_frame(self._server_name, "recv", item)
        return item

    async def __anext__(self) -> Any:
        try:
            return await self.receive()
        except Exception as exc:
            if type(exc).__name__ in ("EndOfStream", "StopAsyncIteration"):
                raise StopAsyncIteration from None
            raise

    def close(self) -> None:
        if hasattr(self._stream, "close"):
            self._stream.close()

    async def aclose(self) -> None:
        if hasattr(self._stream, "aclose"):
            await self._stream.aclose()


class _WireTapSendStream:
    """Async stream wrapper tapping outgoing frames (send)."""

    def __init__(self, server_name: str, stream: Any) -> None:
        self._server_name = server_name
        self._stream = stream

    def __getattr__(self, name: str) -> Any:
        return getattr(self._stream, name)

    async def __aenter__(self) -> _WireTapSendStream:
        if hasattr(self._stream, "__aenter__"):
            await self._stream.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> Any:
        if hasattr(self._stream, "__aexit__"):
            return await self._stream.__aexit__(exc_type, exc_val, exc_tb)
        return None

    async def send(self, item: Any) -> None:
        log_wire_frame(self._server_name, "send", item)
        await self._stream.send(item)

    def send_nowait(self, item: Any) -> Any:
        log_wire_frame(self._server_name, "send", item)
        return self._stream.send_nowait(item)

    def close(self) -> None:
        if hasattr(self._stream, "close"):
            self._stream.close()

    async def aclose(self) -> None:
        if hasattr(self._stream, "aclose"):
            await self._stream.aclose()


def tap_streams(
    server_name: str, read_stream: Any, write_stream: Any, config: Optional[dict] = None
) -> Tuple[Any, Any]:
    """Transparently tap read and write streams if wire logging is enabled.

    Zero overhead when disabled: returns the underlying streams untouched.
    """
    if not is_wire_log_enabled(server_name, config):
        return read_stream, write_stream

    return (
        _WireTapReceiveStream(server_name, read_stream),
        _WireTapSendStream(server_name, write_stream),
    )
