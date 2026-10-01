"""``share_group_file``: hand one local file to the current Group Chat.

The file is copied into the turn's private outbox. The room owner publishes it
as one member message when the turn completes, or discards it when the turn is
stopped or fails. Outside a Group Chat turn the tool refuses.
"""

from __future__ import annotations

import base64
import binascii
import json
import logging
import os
import sys
from pathlib import Path, PurePosixPath

from gateway.hosted_room_artifacts import (
    RoomArtifactError,
    RoomArtifactScope,
    open_room_artifact_path,
    validate_open_room_artifact_path,
)
from gateway.hosted_room_attachments import MAX_ATTACHMENT_BYTES
from gateway.session_hosted_output import current_output_binding
from tools.registry import registry


logger = logging.getLogger(__name__)


_PRIVATE_ROOM_STORAGE_NAMES = frozenset({
    "hosted-room-artifact-outbox",
    "hosted-room-attachments",
    "roomlink-attachment-spool",
})
_REMOTE_SENSITIVE_DIRECTORIES = frozenset({
    ".aws",
    ".azure",
    ".docker",
    ".gnupg",
    ".kube",
    ".ssh",
    "browser-profile",
    "mcp-tokens",
})
_REMOTE_SENSITIVE_FILES = frozenset({
    ".anthropic_oauth.json",
    ".env",
    "auth.json",
    "auth.lock",
    "bws_cache.json",
    "google_oauth.json",
    "webhook_subscriptions.json",
})
_REMOTE_FILE_MARKER = "HERMES_ROOM_FILE_V1:"
_REMOTE_FILE_READER = r'''
import base64
import json
import os
import stat
import sys

marker = "HERMES_ROOM_FILE_V1:"

def emit(value):
    print(marker + json.dumps(value, separators=(",", ":")))

try:
    candidate = os.path.abspath(os.path.expanduser(sys.argv[1]))
    limit = int(sys.argv[2])
    if os.name == "nt":
        current = os.path.splitdrive(candidate)[0] + os.sep
        original = None
        for part in [item for item in candidate[len(current):].split(os.sep) if item]:
            current = os.path.join(current, part)
            info = os.lstat(current)
            attributes = getattr(info, "st_file_attributes", 0)
            reparse = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
            if stat.S_ISLNK(info.st_mode) or (reparse and attributes & reparse):
                raise OSError("link")
            original = info
        descriptor = os.open(candidate, os.O_RDONLY | getattr(os, "O_BINARY", 0))
        current = os.path.splitdrive(candidate)[0] + os.sep
        for part in [item for item in candidate[len(current):].split(os.sep) if item]:
            current = os.path.join(current, part)
            info = os.lstat(current)
            attributes = getattr(info, "st_file_attributes", 0)
            reparse = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
            if stat.S_ISLNK(info.st_mode) or (reparse and attributes & reparse):
                raise OSError("link")
        if original is None or not os.path.samestat(original, os.fstat(descriptor)):
            raise OSError("changed")
    else:
        flags = os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
        directory_flags = flags | os.O_DIRECTORY
        parts = [item for item in candidate.split(os.sep) if item]
        directory = os.open(os.sep, directory_flags)
        try:
            for part in parts[:-1]:
                next_directory = os.open(part, directory_flags, dir_fd=directory)
                os.close(directory)
                directory = next_directory
            descriptor = os.open(
                parts[-1],
                flags | getattr(os, "O_NONBLOCK", 0),
                dir_fd=directory,
            )
        finally:
            os.close(directory)
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or not 0 < info.st_size <= limit:
            raise OSError("not-bounded-regular")
        chunks = []
        remaining = info.st_size
        while remaining:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                raise OSError("short-read")
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.fstat(descriptor).st_size != info.st_size:
            raise OSError("changed")
    finally:
        os.close(descriptor)
    emit({"ok": True, "data": base64.b64encode(b"".join(chunks)).decode("ascii")})
except Exception:
    emit({"ok": False})
'''


SHARE_GROUP_FILE_SCHEMA = {
    "name": "share_group_file",
    "description": (
        "Share one local file with the current Group Chat. The file is copied "
        "into private room storage and becomes available to the user and the "
        "other Bots in this Group Chat. Never use it for credentials or secrets."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Absolute path of the file to share.",
            },
            "name": {
                "type": "string",
                "description": "Optional filename shown in the Group Chat.",
            },
        },
        "required": ["path"],
    },
}

_UNAVAILABLE = "File sharing is available only during a Group Chat turn."
_UNSAFE_PATH = (
    "That file cannot be shared. Move it to the workspace or a Hermes media folder and try again."
)


def _requested_file_path(value: str) -> Path:
    candidate = str(value or "").strip()
    if len(candidate) >= 2 and candidate[0] == candidate[-1] and candidate[0] in "`\"'":
        candidate = candidate[1:-1].strip()
    candidate = candidate.lstrip("`\"'").rstrip("`\"',.;:)}]")
    return Path(candidate).expanduser()


def _canonical_macos_alias_path(path: Path) -> Path:
    """Rewrite only macOS' OS-owned root aliases before no-follow open."""

    if sys.platform != "darwin":
        return path
    normalized = os.path.normpath(str(path))
    for alias, target in (("/tmp", "/private/tmp"), ("/var", "/private/var")):
        if normalized == alias or normalized.startswith(alias + "/"):
            return Path(target + normalized[len(alias):])
    return path


def _portable_path_parts(path: Path | str) -> tuple[str, ...]:
    normalized = str(path).replace("\\", "/")
    return tuple(part.casefold() for part in PurePosixPath(normalized).parts)


def _is_private_room_storage_path(path: Path | str) -> bool:
    return any(part in _PRIVATE_ROOM_STORAGE_NAMES for part in _portable_path_parts(path))


def _is_sensitive_remote_path(path: Path | str) -> bool:
    parts = _portable_path_parts(path)
    if not parts:
        return True
    if any(part in _REMOTE_SENSITIVE_DIRECTORIES for part in parts):
        return True
    if parts[-1] in _REMOTE_SENSITIVE_FILES:
        return True
    return any(
        parts[index:index + 2] in ((".config", "gh"), (".config", "gcloud"))
        for index in range(len(parts) - 1)
    )


def _require_publishable_name(name: str) -> None:
    """A credential-shaped name would be masked in the stored turn result."""
    from agent.redact import redact_sensitive_text

    if redact_sensitive_text(name) != name:
        raise RoomArtifactError(
            "That file name looks like a credential. Pass a different name and try again."
        )


def _store_open_group_file(
    *,
    binding,
    path: Path,
    descriptor: int,
    session_key: str,
    name: str | None,
) -> dict[str, object]:
    from agent.file_safety import get_read_block_error
    from gateway.platforms.base import validate_media_delivery_path
    from hermes_constants import get_default_hermes_root, get_hermes_home

    safe_path = validate_media_delivery_path(str(path), session_key=session_key)
    if safe_path is None:
        raise RoomArtifactError(_UNSAFE_PATH)
    resolved = Path(safe_path)
    if _is_private_room_storage_path(resolved):
        raise RoomArtifactError("Private Group Chat storage cannot be shared.")
    active_home = Path(get_hermes_home()).resolve(strict=False)
    profiles_root = Path(get_default_hermes_root()).resolve(strict=False) / "profiles"
    try:
        resolved.relative_to(profiles_root)
    except ValueError:
        pass
    else:
        try:
            resolved.relative_to(active_home)
        except ValueError as exc:
            raise RoomArtifactError(
                "Files owned by another Hermes profile cannot be shared."
            ) from exc
        if active_home == profiles_root.parent:
            raise RoomArtifactError(
                "Files owned by another Hermes profile cannot be shared."
            )
    if get_read_block_error(str(resolved)):
        raise RoomArtifactError(
            "Hermes credential and internal state files cannot be shared."
        )
    resolved = validate_open_room_artifact_path(resolved, descriptor)
    _require_publishable_name(name or resolved.name)
    return binding.outbox().put_open_file(
        scope=binding.scope,
        descriptor=descriptor,
        source_name=resolved.name,
        name=name,
    )


def _store_backend_group_file(
    *,
    binding,
    path: Path,
    task_id: str,
    name: str | None,
) -> dict[str, object]:
    """Copy bytes from a non-host terminal backend through its file adapter."""

    from agent.file_safety import get_read_block_error
    from tools.file_tools import _get_file_ops
    from tools.file_tools_paths import _resolve_path_for_task

    resolved = _resolve_path_for_task(str(path), task_id)
    if _is_private_room_storage_path(Path(str(resolved))):
        raise RoomArtifactError("Private Group Chat storage cannot be shared.")
    if _is_sensitive_remote_path(resolved) or get_read_block_error(str(resolved)):
        raise RoomArtifactError(
            "Hermes credential and internal state files cannot be shared."
        )
    _require_publishable_name(name or Path(str(resolved)).name)
    data = _read_backend_file_bytes_nofollow(
        _get_file_ops(task_id),
        str(resolved),
    )
    return binding.outbox().put_bytes(
        scope=binding.scope,
        data=data,
        source_name=Path(str(resolved)).name,
        name=name,
    )


def _read_backend_file_bytes_nofollow(file_ops, path: str) -> bytes:
    """Read one bounded backend file without following any path component."""

    python = "python3" if file_ops._has_command("python3") else "python"
    command = " ".join((
        python,
        "-c",
        file_ops._escape_shell_arg(_REMOTE_FILE_READER),
        file_ops._escape_shell_arg(path),
        str(MAX_ATTACHMENT_BYTES),
    ))
    result = file_ops._exec(command, timeout=60)
    marker_at = result.stdout.rfind(_REMOTE_FILE_MARKER)
    if result.exit_code != 0 or marker_at < 0:
        raise RoomArtifactError(
            "That file cannot be shared from the active execution environment."
        )
    line = result.stdout[marker_at + len(_REMOTE_FILE_MARKER):].splitlines()[0]
    try:
        payload = json.loads(line)
        if payload != {"ok": False} and payload.get("ok") is True:
            data = base64.b64decode(payload["data"], validate=True)
            if 0 < len(data) <= MAX_ATTACHMENT_BYTES:
                return data
    except (KeyError, TypeError, ValueError, binascii.Error, json.JSONDecodeError):
        pass
    raise RoomArtifactError(
        "That file cannot be shared from the active execution environment."
    )


def _shared(stored) -> str:
    return json.dumps({
        "ok": True,
        "artifact_id": stored["artifact_id"],
        "name": stored["name"],
        "size": stored["size"],
        "sha256": stored["sha256"],
        "message": f"{stored['name']} will be shared with this Group Chat when your turn completes.",
    })


def share_group_file(
    path: str,
    *,
    name: str | None = None,
    task_id: str = "default",
) -> str:
    """Copy a safe file into the current hosted-room output outbox."""

    binding = current_output_binding()
    if binding is None or not isinstance(binding.scope, RoomArtifactScope):
        return json.dumps({"ok": False, "error": _UNAVAILABLE})
    try:
        from gateway.platforms.base import validate_media_delivery_path
        from gateway.session_context import get_session_env
        from tools.file_tools_paths import _terminal_env_type_for_task

        requested = _requested_file_path(path)
        if not requested.is_absolute():
            raise RoomArtifactError(_UNSAFE_PATH)
        if name is not None:
            _require_publishable_name(str(name))
        if _terminal_env_type_for_task(task_id) != "local":
            return _shared(_store_backend_group_file(
                binding=binding,
                path=requested,
                task_id=task_id,
                name=name,
            ))
        session_key = get_session_env("HERMES_SESSION_KEY", "")
        safe_path = validate_media_delivery_path(
            str(requested),
            session_key=session_key,
        )
        if safe_path is None:
            raise RoomArtifactError(_UNSAFE_PATH)
        open_candidate = _canonical_macos_alias_path(requested)
        with open_room_artifact_path(open_candidate) as (opened_path, descriptor):
            stored = _store_open_group_file(
                binding=binding,
                path=opened_path,
                descriptor=descriptor,
                session_key=session_key,
                name=name,
            )
        return _shared(stored)
    except RoomArtifactError as exc:
        return json.dumps({"ok": False, "error": str(exc)})
    except (OSError, RuntimeError, ValueError):
        logger.warning("Group Chat file sharing failed", exc_info=True)
        return json.dumps({
            "ok": False,
            "error": "That file could not be shared. Check the file and try again.",
        })


def _handle_share_group_file(args, **kwargs):
    return share_group_file(
        args.get("path", ""),
        name=args.get("name"),
        task_id=kwargs.get("task_id") or "default",
    )


registry.register(
    name="share_group_file",
    toolset="bot_room",
    schema=SHARE_GROUP_FILE_SCHEMA,
    handler=_handle_share_group_file,
    emoji="📎",
)


__all__ = [
    "SHARE_GROUP_FILE_SCHEMA",
    "share_group_file",
]
