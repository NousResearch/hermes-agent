"""Native Windows Named Pipe transport for cua-driver IPC.

Replaces slow subprocess spawning (`cua-driver call ...`) with direct Windows Named Pipe IPC
over `\\\\.\\pipe\\cua-driver` (or the daemon socket), matching the proven low-latency transport
used in OpenAI Codex / ZCode (`NativePipeComputerUseTransport`).
"""

from __future__ import annotations

import base64
import contextlib
import ctypes
import json
import logging
import os
import struct
import sys
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

from tools.computer_use.cua_backend_parse import _tool_envelope

logger = logging.getLogger("tools.computer_use.cua_backend")

_DEFAULT_WINDOWS_PIPE_NAME = r"\\.\pipe\cua-driver"
_FRAME_HEADER_BYTES = 4


class PipeTransportError(Exception):
    """Base exception for native named pipe transport errors."""


class PipePreDispatchError(RuntimeError, PipeTransportError):
    """Failure occurred before bytes were written to the pipe (e.g. connection refused, busy, or timed out)."""


class PipePostDispatchError(RuntimeError, PipeTransportError):
    """Failure occurred during or after dispatching bytes to the pipe. Mutating tools must not be automatically replayed."""


class PipePostDispatchTimeoutError(PipePostDispatchError, TimeoutError):
    """Timeout occurred waiting for response after bytes were dispatched to the pipe."""


def get_computer_use_pipe_path(embedded_daemon: Optional[Any] = None) -> Optional[str]:
    """Resolve the Windows Named Pipe path for cua-driver.

    Checks:
    1. Embedded daemon socket path (if active on Windows).
    2. SKY_CUA_NATIVE_PIPE_DIRECTORY (ZCode / Codex env var).
    3. CUA_DRIVER_PIPE_PATH / CUA_DRIVER_SOCKET env vars.
    4. Default \\\\.\\pipe\\cua-driver.
    """
    if sys.platform != "win32":
        return None

    if embedded_daemon is not None:
        socket_path = getattr(embedded_daemon, "socket_path", None)
        if isinstance(socket_path, str) and socket_path.startswith(r"\\"):
            return socket_path

    for env_var in ("SKY_CUA_NATIVE_PIPE_DIRECTORY", "CUA_DRIVER_PIPE_PATH", "CUA_DRIVER_SOCKET"):
        val = os.environ.get(env_var, "").strip()
        if val:
            if not val.startswith(r"\\"):
                val = rf"\\.\pipe\{val}"
            return val

    return _DEFAULT_WINDOWS_PIPE_NAME


def is_named_pipe_available(pipe_path: Optional[str] = None) -> bool:
    """Check if the named pipe exists and is accepting connections without blocking."""
    if sys.platform != "win32":
        return False
    target_pipe = pipe_path or get_computer_use_pipe_path()
    if not target_pipe:
        return False
    try:
        # WaitNamedPipeW returns non-zero if an instance of the pipe is available
        res = ctypes.windll.kernel32.WaitNamedPipeW(target_pipe, 0)
        if res:
            return True
        last_error = ctypes.GetLastError()
        # 231 = ERROR_PIPE_BUSY, 121 = ERROR_SEM_TIMEOUT (pipe exists and is listening)
        return last_error in (231, 121)
    except Exception as exc:
        logger.debug("Named pipe check error on %s: %s", target_pipe, exc)
        return False


def encode_message_frame(payload_bytes: bytes, endianness: str = "le") -> bytes:
    """4-byte length prefix framing matching NativePipeComputerUseTransport."""
    fmt = "<I" if endianness == "le" else ">I"
    header = struct.pack(fmt, len(payload_bytes))
    return header + payload_bytes


def decode_message_frame(buffer: bytes, endianness: str = "le") -> Tuple[List[bytes], bytes]:
    """Decode 4-byte length prefixed frames from buffer."""
    fmt = "<I" if endianness == "le" else ">I"
    messages = []
    offset = 0
    buf_len = len(buffer)
    while buf_len - offset >= _FRAME_HEADER_BYTES:
        payload_len = struct.unpack_from(fmt, buffer, offset)[0]
        frame_len = _FRAME_HEADER_BYTES + payload_len
        if buf_len - offset < frame_len:
            break
        messages.append(buffer[offset + _FRAME_HEADER_BYTES: offset + frame_len])
        offset += frame_len
    return messages, buffer[offset:]


class NativePipeComputerUseTransport:
    """Direct Windows Named Pipe transport client for cua-driver.

    Bypasses subprocess spawn latency (~150-300ms on Windows Defender) and process-leak
    races by communicating directly over named pipes with 2-5ms round-trip latency.
    """

    def __init__(
        self,
        pipe_path: Optional[str] = None,
        framing: str = "auto",
        session_id: Optional[str] = None,
        timeout: float = 30.0,
    ) -> None:
        if sys.platform != "win32":
            raise RuntimeError("NativePipeComputerUseTransport is only available on Windows")

        self.pipe_path = pipe_path or get_computer_use_pipe_path() or _DEFAULT_WINDOWS_PIPE_NAME
        self.framing = framing  # "auto", "line", or "length_prefix"
        self.session_id = session_id
        self.timeout = timeout
        self.next_request_id = 1
        self._lock = threading.Lock()

    @classmethod
    def is_available(cls, pipe_path: Optional[str] = None) -> bool:
        """Check whether the native pipe daemon is ready to connect."""
        return is_named_pipe_available(pipe_path)

    @classmethod
    def create(
        cls,
        pipe_path: Optional[str] = None,
        framing: str = "auto",
        session_id: Optional[str] = None,
        timeout: float = 30.0,
    ) -> NativePipeComputerUseTransport:
        resolved = pipe_path or get_computer_use_pipe_path()
        if not resolved:
            raise RuntimeError("Computer Use native pipe path is unavailable")
        if not cls.is_available(resolved):
            raise RuntimeError(f"Computer Use native pipe {resolved} is not available")
        return cls(pipe_path=resolved, framing=framing, session_id=session_id, timeout=timeout)

    def _open_handle(self, timeout_sec: float = 5.0):
        """Open a file handle to the named pipe with retry on ERROR_PIPE_BUSY."""
        import _winapi

        deadline = time.monotonic() + timeout_sec
        while time.monotonic() < deadline:
            try:
                handle = _winapi.CreateFile(
                    self.pipe_path,
                    _winapi.GENERIC_READ | _winapi.GENERIC_WRITE,
                    0,
                    0,
                    _winapi.OPEN_EXISTING,
                    0,
                    0,
                )
                return handle
            except OSError as exc:
                # 231 = ERROR_PIPE_BUSY
                if exc.winerror == 231:
                    wait_time_ms = int(min(1000, max(50, (deadline - time.monotonic()) * 1000)))
                    ctypes.windll.kernel32.WaitNamedPipeW(self.pipe_path, wait_time_ms)
                    continue
                raise PipePreDispatchError(
                    f"Failed to connect to cua-driver named pipe {self.pipe_path}: {exc}"
                ) from exc

        raise PipePreDispatchError(
            f"Timed out connecting to cua-driver named pipe {self.pipe_path} after {timeout_sec:.1f}s"
        )

    def _read_all_line(self, handle, deadline: Optional[float] = None) -> bytes:
        """Read until newline delimiter from pipe handle respecting deadline."""
        import _winapi

        effective_deadline = deadline if deadline is not None else (time.monotonic() + self.timeout)
        buf = bytearray()
        avail = ctypes.c_ulong(0)
        kernel32 = ctypes.windll.kernel32
        while not buf.endswith(b"\n"):
            if time.monotonic() >= effective_deadline:
                raise TimeoutError(
                    f"Timed out waiting for response from cua-driver pipe {self.pipe_path}"
                )
            res = kernel32.PeekNamedPipe(handle, None, 0, None, ctypes.byref(avail), None)
            if not res:
                err = ctypes.GetLastError()
                if err == 109:  # ERROR_BROKEN_PIPE (pipe closed by peer)
                    break
                raise OSError(f"PeekNamedPipe failed with error {err}")
            if avail.value > 0:
                to_read = min(65536, avail.value)
                chunk, _ = _winapi.ReadFile(handle, to_read)
                if not chunk:
                    break
                buf.extend(chunk)
            else:
                time.sleep(0.002)
        return bytes(buf)

    def _read_length_prefix(self, handle, deadline: Optional[float] = None) -> bytes:
        """Read 4-byte LE length prefix followed by payload respecting deadline."""
        import _winapi

        effective_deadline = deadline if deadline is not None else (time.monotonic() + self.timeout)
        hdr = bytearray()
        avail = ctypes.c_ulong(0)
        kernel32 = ctypes.windll.kernel32
        while len(hdr) < _FRAME_HEADER_BYTES:
            if time.monotonic() >= effective_deadline:
                raise TimeoutError(
                    f"Timed out waiting for header from cua-driver pipe {self.pipe_path}"
                )
            res = kernel32.PeekNamedPipe(handle, None, 0, None, ctypes.byref(avail), None)
            if not res:
                err = ctypes.GetLastError()
                if err == 109:
                    break
                raise OSError(f"PeekNamedPipe failed with error {err}")
            if avail.value > 0:
                to_read = min(_FRAME_HEADER_BYTES - len(hdr), avail.value)
                chunk, _ = _winapi.ReadFile(handle, to_read)
                if not chunk:
                    break
                hdr.extend(chunk)
            else:
                time.sleep(0.002)

        if len(hdr) < _FRAME_HEADER_BYTES:
            return b""
        payload_len = struct.unpack("<I", hdr)[0]
        body = bytearray()
        while len(body) < payload_len:
            if time.monotonic() >= effective_deadline:
                raise TimeoutError(
                    f"Timed out waiting for payload body from cua-driver pipe {self.pipe_path}"
                )
            res = kernel32.PeekNamedPipe(handle, None, 0, None, ctypes.byref(avail), None)
            if not res:
                err = ctypes.GetLastError()
                if err == 109:
                    break
                raise OSError(f"PeekNamedPipe failed with error {err}")
            if avail.value > 0:
                to_read = min(min(65536, payload_len - len(body)), avail.value)
                chunk, _ = _winapi.ReadFile(handle, to_read)
                if not chunk:
                    break
                body.extend(chunk)
            else:
                time.sleep(0.002)
        return bytes(body)

    def get_metadata(self, timeout: float = 5.0) -> Dict[str, Any]:
        """Fetch driver daemon metadata."""
        import _winapi

        deadline = time.monotonic() + timeout
        handle = self._open_handle(timeout_sec=timeout)
        try:
            req = {"method": "metadata"}
            data = (json.dumps(req) + "\n").encode("utf-8")
            _winapi.WriteFile(handle, data)
            resp_bytes = self._read_all_line(handle, deadline=deadline)
            if not resp_bytes:
                return {}
            parsed = json.loads(resp_bytes.decode("utf-8", errors="replace").strip())
            return parsed.get("result", {}) if isinstance(parsed, dict) else {}
        finally:
            with contextlib.suppress(Exception):
                _winapi.CloseHandle(handle)

    def request(
        self, method: str, params: Dict[str, Any], options: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Codex / ZCode compatible request API."""
        return self.call_tool(method, params, timeout=self.timeout)

    def call_tool(
        self,
        name: str,
        args: Dict[str, Any],
        timeout: float = 30.0,
    ) -> Dict[str, Any]:
        """Invoke a tool via named pipe and return the normalized tool envelope shape.

        Returns {data, images, structuredContent, isError}.
        """
        import _winapi

        with self._lock:
            call_args = dict(args)
            shot_file = call_args.get("screenshot_out_file")

            session_to_use = call_args.get("session") or self.session_id
            req_payload: Dict[str, Any] = {
                "method": "call",
                "name": name,
                "args": call_args,
                "observation_origin": "direct",
                "client_kind": "python_sdk",
            }
            if session_to_use:
                req_payload["session_id"] = session_to_use

            deadline = time.monotonic() + timeout
            try:
                handle = self._open_handle(timeout_sec=min(5.0, timeout))
            except Exception as exc:
                if isinstance(exc, PipePreDispatchError):
                    raise
                raise PipePreDispatchError(
                    f"Failed opening handle for {name} on pipe {self.pipe_path}: {exc}"
                ) from exc

            try:
                encoded_str = json.dumps(req_payload)
                use_length_prefix = (
                    self.framing == "length_prefix"
                    or (self.framing == "auto" and bool(os.environ.get("SKY_CUA_NATIVE_PIPE_DIRECTORY")))
                )
                try:
                    if use_length_prefix:
                        frame = encode_message_frame(encoded_str.encode("utf-8"), "le")
                        _winapi.WriteFile(handle, frame)
                        raw_resp = self._read_length_prefix(handle, deadline=deadline)
                    else:
                        data = (encoded_str + "\n").encode("utf-8")
                        _winapi.WriteFile(handle, data)
                        raw_resp = self._read_all_line(handle, deadline=deadline)
                except TimeoutError as exc:
                    raise PipePostDispatchTimeoutError(
                        f"Timed out waiting for response from cua-driver pipe {self.pipe_path} for tool {name}: {exc}"
                    ) from exc
                except Exception as exc:
                    raise PipePostDispatchError(
                        f"Post-dispatch transport error on cua-driver pipe {self.pipe_path} for tool {name}: {exc}"
                    ) from exc

                if not raw_resp:
                    raise PipePostDispatchError(
                        f"cua-driver named pipe returned empty response for tool {name}"
                    )

                try:
                    parsed = json.loads(raw_resp.decode("utf-8", errors="replace").strip())
                except Exception as exc:
                    raise PipePostDispatchError(
                        f"cua-driver named pipe returned unparseable JSON for tool {name}: {exc}"
                    ) from exc

                return self._normalize_response(parsed, name, shot_file)
            finally:
                with contextlib.suppress(Exception):
                    _winapi.CloseHandle(handle)

    def _normalize_response(
        self, parsed: Any, name: str, shot_file: Optional[str]
    ) -> Dict[str, Any]:
        """Convert daemon response into Hermes tool envelope shape."""
        if not isinstance(parsed, dict):
            return _tool_envelope(None, [], None, False)

        # Standard JSON-RPC 2.0 error check: {"jsonrpc": "2.0", "error": {...}}
        if "error" in parsed and ("ok" not in parsed or parsed.get("ok") is False):
            err_obj = parsed["error"]
            err_msg = err_obj.get("message", str(err_obj)) if isinstance(err_obj, dict) else str(err_obj)
            exit_code = err_obj.get("code", 1) if isinstance(err_obj, dict) else parsed.get("exit_code", 1)
            return _tool_envelope(
                err_msg,
                [],
                {"ok": False, "error": err_msg, "exit_code": exit_code},
                True,
            )

        # Check for top-level error
        is_error = (
            parsed.get("ok") is False
            or parsed.get("isError") is True
            or parsed.get("is_error") is True
        )

        if not parsed.get("ok", True) and "error" in parsed:
            err_msg = str(parsed.get("error", "Unknown error"))
            return _tool_envelope(
                err_msg,
                [],
                {"ok": False, "error": err_msg, "exit_code": parsed.get("exit_code", 1)},
                True,
            )

        result = parsed.get("result")
        if not isinstance(result, dict):
            return _tool_envelope(str(result) if result is not None else None, [], parsed, is_error)

        content = result.get("content", [])
        structured = result.get("structuredContent") or result
        if result.get("isError") is True or result.get("is_error") is True:
            is_error = True

        data: Any = None
        images: List[str] = []
        text_chunks: List[str] = []

        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict):
                    ptype = part.get("type")
                    if ptype == "text":
                        text_chunks.append(part.get("text", "") or "")
                    elif ptype == "image" and part.get("data"):
                        images.append(part["data"])

        if text_chunks:
            joined = "\n".join(t for t in text_chunks if t)
            try:
                data = json.loads(joined) if joined.strip().startswith(("{", "[")) else joined
            except Exception:
                data = joined

        # Check for screenshot in structuredContent or shot_file
        shot = None
        if isinstance(structured, dict):
            shot = structured.get("screenshot_png_b64") or structured.get("png_b64")
            fpath = structured.get("screenshot_file_path") or shot_file
            if not shot and fpath and os.path.exists(fpath):
                try:
                    with open(fpath, "rb") as fh:
                        shot = base64.b64encode(fh.read()).decode("ascii")
                except Exception as exc:
                    logger.debug("Failed reading screenshot file %s: %s", fpath, exc)

        if shot and not images:
            images = [shot]

        if data is None and isinstance(structured, dict):
            markdown = structured.get("tree_markdown")
            elem_count = structured.get("element_count")
            if markdown is not None and elem_count is not None:
                data = f"{elem_count} elements\n{markdown}"
            elif markdown is not None:
                data = markdown

        return _tool_envelope(data, images, structured, is_error)

    def close(self) -> None:
        """Close transport."""
        pass
