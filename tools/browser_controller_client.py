"""Gateway-side client for the dedicated no-secret browser controller."""

from __future__ import annotations

import base64
import hashlib
import os
import socket
import stat
import threading
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

from gateway.browser_controller_protocol import (
    MAX_ARTIFACT_BYTES,
    MAX_REQUEST_BYTES,
    MAX_RESPONSE_BYTES,
    PeerCredentials,
    PROTOCOL_VERSION,
    receive_frame,
    send_frame,
    unix_peer_credentials,
    validate_response,
)


CLIENT_CONFIG_SCHEMA = "hermes-browser-controller-client.v1"
MAX_CLIENT_ARTIFACTS = 8
MAX_CLIENT_ARTIFACT_BYTES = 32 * 1024 * 1024
_controller_requirement_lock = threading.Lock()
_controller_required = False


class BrowserControllerClientError(RuntimeError):
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


def _exact(value: Any, keys: frozenset[str]) -> dict[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(k, str) for k in value):
        raise BrowserControllerClientError("browser_controller_client_config_invalid")
    result = dict(value)
    if set(result) != keys:
        raise BrowserControllerClientError("browser_controller_client_config_invalid")
    return result


def _path(value: Any, code: str) -> Path:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise BrowserControllerClientError(code)
    path = Path(value)
    if not path.is_absolute() or str(path) != value:
        raise BrowserControllerClientError(code)
    return path


def _integer(value: Any, *, low: int, high: int, code: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
        raise BrowserControllerClientError(code)
    return value


@dataclass(frozen=True)
class BrowserControllerClientConfig:
    socket_path: Path
    server_uid: int
    artifact_root: Path
    connect_timeout_seconds: int
    request_timeout_seconds: int

    @classmethod
    def from_mapping(cls, value: Any) -> "BrowserControllerClientConfig":
        raw = _exact(
            value,
            frozenset(
                {
                    "schema",
                    "socket_path",
                    "server_uid",
                    "artifact_root",
                    "connect_timeout_seconds",
                    "request_timeout_seconds",
                }
            ),
        )
        if raw["schema"] != CLIENT_CONFIG_SCHEMA:
            raise BrowserControllerClientError(
                "browser_controller_client_config_schema_invalid"
            )
        return cls(
            socket_path=_path(
                raw["socket_path"], "browser_controller_client_socket_invalid"
            ),
            server_uid=_integer(
                raw["server_uid"],
                low=0,
                high=2**31 - 1,
                code="browser_controller_client_server_uid_invalid",
            ),
            artifact_root=_path(
                raw["artifact_root"],
                "browser_controller_client_artifact_root_invalid",
            ),
            connect_timeout_seconds=_integer(
                raw["connect_timeout_seconds"],
                low=1,
                high=30,
                code="browser_controller_client_connect_timeout_invalid",
            ),
            request_timeout_seconds=_integer(
                raw["request_timeout_seconds"],
                low=5,
                high=300,
                code="browser_controller_client_request_timeout_invalid",
            ),
        )


def activate_browser_controller_required() -> None:
    """Irreversibly forbid every non-controller browser path in this process."""

    global _controller_required
    with _controller_requirement_lock:
        _controller_required = True


def browser_controller_required() -> bool:
    with _controller_requirement_lock:
        return _controller_required


def _raw_controller_mapping() -> tuple[bool, Any]:
    required = browser_controller_required()
    try:
        if required:
            from hermes_cli.config import (
                attest_pinned_effective_config_projection,
                effective_config_projection_is_pinned,
                load_config,
            )

            if (
                not effective_config_projection_is_pinned()
                or attest_pinned_effective_config_projection() is None
            ):
                raise BrowserControllerClientError(
                    "browser_controller_effective_config_not_pinned"
                )
            config = load_config()
        else:
            from hermes_cli.config import read_raw_config

            config = read_raw_config()
    except BrowserControllerClientError:
        raise
    except Exception as exc:
        # Preserve generic Hermes behavior when no controller configuration can
        # be observed. Production latches the controller before any model/tool
        # surface exists, so read errors there are never interpreted as
        # absence and cannot re-enable PATH/npx/local/cloud/CDP execution.
        if required:
            raise BrowserControllerClientError(
                "browser_controller_client_config_unavailable"
            ) from exc
        return False, None
    if not isinstance(config, Mapping):
        if required:
            raise BrowserControllerClientError(
                "browser_controller_client_config_invalid"
            )
        return False, None
    browser = config.get("browser")
    if not isinstance(browser, Mapping) or "controller" not in browser:
        if required:
            raise BrowserControllerClientError(
                "browser_controller_client_config_missing"
            )
        return False, None
    return True, browser.get("controller")


def controller_mode_requested() -> bool:
    """True when config requests the controller, even if that config drifted."""

    if browser_controller_required():
        return True
    try:
        requested, _raw = _raw_controller_mapping()
        return requested
    except BrowserControllerClientError:
        return True


def load_controller_client_config() -> BrowserControllerClientConfig | None:
    requested, raw = _raw_controller_mapping()
    if not requested:
        return None
    return BrowserControllerClientConfig.from_mapping(raw)


def _session_identity(task_id: str) -> str:
    try:
        from gateway.session_context import get_session_env

        durable = get_session_env("HERMES_SESSION_ID") or ""
        epoch = get_session_env("HERMES_CAPABILITY_EPOCH_SHA256") or ""
    except Exception:
        durable = ""
        epoch = ""
    fallback = str(task_id or "default")
    if not durable:
        durable = fallback
    if not epoch:
        epoch = fallback
    material = (
        b"hermes-browser-controller-session-v1\x00"
        + durable.encode("utf-8", errors="strict")
        + b"\x00"
        + epoch.encode("utf-8", errors="strict")
    )
    return hashlib.sha256(material).hexdigest()


class BrowserControllerClient:
    def __init__(
        self,
        config: BrowserControllerClientConfig,
        session_id_sha256: str,
        *,
        peer_getter: Callable[[socket.socket], PeerCredentials] = unix_peer_credentials,
    ) -> None:
        self.config = config
        self.session_id_sha256 = session_id_sha256
        self.peer_getter = peer_getter
        self._socket: socket.socket | None = None
        self._lock = threading.Lock()
        self._closed = False
        self._artifact_lock = threading.Lock()
        self._artifacts: dict[Path, int] = {}
        self._artifact_bytes = 0

    def _request(self, value: Mapping[str, Any]) -> dict[str, Any]:
        sock = self._socket
        if sock is None:
            raise BrowserControllerClientError("browser_controller_not_connected")
        request_id = str(value["request_id"])
        try:
            send_frame(sock, value, maximum=MAX_REQUEST_BYTES)
            response = receive_frame(sock, maximum=MAX_RESPONSE_BYTES)
            return validate_response(response, request_id)
        except BrowserControllerClientError:
            raise
        except Exception as exc:
            raise BrowserControllerClientError(
                "browser_controller_transport_failed"
            ) from exc

    def connect(self) -> None:
        with self._lock:
            if self._socket is not None:
                return
            if self._closed:
                raise BrowserControllerClientError("browser_controller_client_closed")
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            sock.settimeout(self.config.connect_timeout_seconds)
            try:
                sock.connect(str(self.config.socket_path))
                try:
                    peer = self.peer_getter(sock)
                except Exception as exc:
                    raise BrowserControllerClientError(
                        "browser_controller_server_peer_unavailable"
                    ) from exc
                if peer.uid != self.config.server_uid:
                    raise BrowserControllerClientError(
                        "browser_controller_server_peer_forbidden"
                    )
                sock.settimeout(self.config.request_timeout_seconds)
                self._socket = sock
                request_id = str(uuid.uuid4())
                response = self._request(
                    {
                        "version": PROTOCOL_VERSION,
                        "request_id": request_id,
                        "op": "session.open",
                        "session_id_sha256": self.session_id_sha256,
                    }
                )
                if response["status"] != "ok" or response["result"] != {
                    "session": "ready"
                }:
                    raise BrowserControllerClientError(
                        "browser_controller_session_open_failed"
                    )
            except Exception:
                self._socket = None
                sock.close()
                raise

    @staticmethod
    def _controller_command(command: str, args: list[str]) -> tuple[str, list[str]]:
        if command == "eval":
            if args == ["window.location.href"]:
                return "current_url", []
            raise BrowserControllerClientError("browser_controller_command_forbidden")
        if command == "close":
            return "session.close", []
        if command == "record":
            raise BrowserControllerClientError("browser_controller_command_forbidden")
        if command == "screenshot":
            # browser_tool historically supplies an output path.  The path is
            # intentionally not serialized; the controller writes into its
            # private session root and returns bounded PNG bytes instead.
            flags = [item for item in args if item in {"--annotate", "--full"}]
            non_flags = [item for item in args if item not in {"--annotate", "--full"}]
            if len(non_flags) > 1 or any(item.startswith("-") for item in non_flags):
                raise BrowserControllerClientError(
                    "browser_controller_screenshot_args_invalid"
                )
            return command, flags
        return command, list(args)

    def command(self, command: str, args: list[str]) -> dict[str, Any]:
        try:
            mapped, normalized_args = self._controller_command(command, args)
            if mapped == "session.close":
                self.close()
                return {"success": True, "data": {}}
            with self._lock:
                if self._socket is None:
                    # connect() owns the same lock, so release before entering.
                    pass
            if self._socket is None:
                self.connect()
            with self._lock:
                request_id = str(uuid.uuid4())
                response = self._request(
                    {
                        "version": PROTOCOL_VERSION,
                        "request_id": request_id,
                        "op": "command",
                        "command": mapped,
                        "args": normalized_args,
                    }
                )
            if response["status"] == "error":
                return {"success": False, "error": response["error"]}
            result = dict(response["result"])
            artifact = result.pop("artifact", None)
            if artifact is not None:
                path = self._materialize_artifact(artifact)
                data = result.get("data")
                if not isinstance(data, Mapping):
                    data = {}
                result["data"] = {**dict(data), "path": str(path)}
            return result
        except BrowserControllerClientError as exc:
            return {"success": False, "error": exc.code}
        except Exception:
            return {"success": False, "error": "browser_controller_request_failed"}

    def _secure_artifact_root(self) -> Path:
        root = self.config.artifact_root
        try:
            root.mkdir(mode=0o700, parents=True, exist_ok=True)
            state = root.lstat()
            resolved = root.resolve(strict=True)
        except OSError as exc:
            raise BrowserControllerClientError(
                "browser_controller_artifact_root_unavailable"
            ) from exc
        if (
            resolved != root
            or stat.S_ISLNK(state.st_mode)
            or not stat.S_ISDIR(state.st_mode)
            or stat.S_IMODE(state.st_mode) & 0o022
            or state.st_uid != os.geteuid()  # windows-footgun: ok — POSIX AF_UNIX controller boundary
        ):
            raise BrowserControllerClientError(
                "browser_controller_artifact_root_invalid"
            )
        return root

    def _materialize_artifact(self, value: Any) -> Path:
        if not isinstance(value, Mapping) or set(value) != {
            "encoding",
            "media_type",
            "sha256",
            "size",
            "data",
        }:
            raise BrowserControllerClientError("browser_controller_artifact_invalid")
        if value["encoding"] != "base64" or value["media_type"] != "image/png":
            raise BrowserControllerClientError("browser_controller_artifact_invalid")
        size = value["size"]
        if isinstance(size, bool) or not isinstance(size, int) or not 8 <= size <= MAX_ARTIFACT_BYTES:
            raise BrowserControllerClientError("browser_controller_artifact_invalid")
        digest = value["sha256"]
        encoded = value["data"]
        if not isinstance(digest, str) or not isinstance(encoded, str):
            raise BrowserControllerClientError("browser_controller_artifact_invalid")
        try:
            payload = base64.b64decode(encoded, validate=True)
        except (ValueError, TypeError) as exc:
            raise BrowserControllerClientError("browser_controller_artifact_invalid") from exc
        if (
            len(payload) != size
            or hashlib.sha256(payload).hexdigest() != digest
            or not payload.startswith(b"\x89PNG\r\n\x1a\n")
        ):
            raise BrowserControllerClientError("browser_controller_artifact_invalid")
        with self._artifact_lock:
            if self._closed:
                raise BrowserControllerClientError(
                    "browser_controller_client_closed"
                )
            if (
                len(self._artifacts) >= MAX_CLIENT_ARTIFACTS
                or self._artifact_bytes + size > MAX_CLIENT_ARTIFACT_BYTES
            ):
                raise BrowserControllerClientError(
                    "browser_controller_artifact_quota_exceeded"
                )
            root = self._secure_artifact_root()
            path = root / f"browser-{uuid.uuid4().hex}.png"
            flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
            if hasattr(os, "O_NOFOLLOW"):
                flags |= os.O_NOFOLLOW
            fd = os.open(path, flags, 0o600)
            try:
                view = memoryview(payload)
                while view:
                    written = os.write(fd, view)
                    if written <= 0:
                        raise OSError("short artifact write")
                    view = view[written:]
                os.fsync(fd)
            except Exception:
                try:
                    path.unlink()
                except OSError:
                    pass
                raise
            finally:
                os.close(fd)
            self._artifacts[path] = size
            self._artifact_bytes += size
            return path

    def _cleanup_artifacts(self) -> None:
        with self._artifact_lock:
            paths = tuple(self._artifacts)
            self._artifacts.clear()
            self._artifact_bytes = 0
        for path in paths:
            try:
                path.unlink()
            except FileNotFoundError:
                pass
            except OSError:
                pass

    def close(self) -> None:
        with self._lock:
            sock, self._socket = self._socket, None
            already_closed = self._closed
            self._closed = True
        if not already_closed and sock is not None:
            request_id = str(uuid.uuid4())
            try:
                send_frame(
                    sock,
                    {
                        "version": PROTOCOL_VERSION,
                        "request_id": request_id,
                        "op": "session.close",
                    },
                    maximum=MAX_REQUEST_BYTES,
                )
                response = receive_frame(sock, maximum=MAX_RESPONSE_BYTES)
                validate_response(response, request_id)
            except Exception:
                pass
            finally:
                sock.close()
        self._cleanup_artifacts()


_pool_lock = threading.Lock()
_clients: dict[tuple[BrowserControllerClientConfig, str], BrowserControllerClient] = {}


def _pooled_client(
    config: BrowserControllerClientConfig,
    task_id: str,
) -> tuple[tuple[BrowserControllerClientConfig, str], BrowserControllerClient]:
    identity = _session_identity(task_id)
    key = (config, identity)
    with _pool_lock:
        client = _clients.get(key)
        if client is None:
            client = BrowserControllerClient(config, identity)
            _clients[key] = client
        return key, client


def maybe_run_browser_controller_command(
    task_id: str,
    command: str,
    args: list[str],
) -> dict[str, Any] | None:
    """Return ``None`` only when controller mode is genuinely absent."""

    try:
        config = load_controller_client_config()
    except BrowserControllerClientError as exc:
        return {"success": False, "error": exc.code}
    except Exception:
        return {
            "success": False,
            "error": "browser_controller_client_config_unavailable",
        }
    if config is None:
        return None
    key, client = _pooled_client(config, task_id)
    result = client.command(command, args)
    if result.get("error") in {
        "browser_controller_transport_failed",
        "browser_controller_not_connected",
        "browser_controller_session_open_failed",
    }:
        with _pool_lock:
            if _clients.get(key) is client:
                _clients.pop(key, None)
        client.close()
    return result


def close_browser_controller_session(task_id: str) -> bool:
    try:
        config = load_controller_client_config()
    except Exception:
        return controller_mode_requested()
    if config is None:
        return False
    identity = _session_identity(task_id)
    key = (config, identity)
    with _pool_lock:
        client = _clients.pop(key, None)
    if client is not None:
        client.close()
    return True


def close_all_browser_controller_sessions() -> None:
    with _pool_lock:
        clients = list(_clients.values())
        _clients.clear()
    for client in clients:
        client.close()


__all__ = [
    "BrowserControllerClient",
    "BrowserControllerClientConfig",
    "BrowserControllerClientError",
    "CLIENT_CONFIG_SCHEMA",
    "MAX_CLIENT_ARTIFACTS",
    "MAX_CLIENT_ARTIFACT_BYTES",
    "activate_browser_controller_required",
    "browser_controller_required",
    "close_all_browser_controller_sessions",
    "close_browser_controller_session",
    "controller_mode_requested",
    "load_controller_client_config",
    "maybe_run_browser_controller_command",
]
