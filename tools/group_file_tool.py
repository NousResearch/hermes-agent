"""Stage a file for the current group chat attempt; publish only with its settled reply.
Media-delivery and read-guard policy is narrowed to exclude every profile's private files.
Remote backends validate the source descriptor; the tool is scoped to bot_room sessions."""

from __future__ import annotations

import os
import stat
from contextlib import suppress
from pathlib import Path

from tools.registry import registry, tool_error, tool_result

_UNAVAILABLE = "File sharing requires a bot turn on the gateway that owns this group chat."
_UNSAFE_PATH = "That file cannot be shared. Share a file from your workspace or a Hermes media folder."
_PRIVATE_STORAGE = "hosted-room-attachments"
# Private trees of every Hermes home that a share never reads, whichever profile runs the turn:
# main's read-guarded directories (credential vault, MCP tokens, browser-profile copy) and its
# skill-hub cache, which the read guard covers for the active home and the root only, plus the
# group chat file store itself.
_PRIVATE_TREES = (("vault",), ("mcp-tokens",), ("browser-profile",), ("skills", ".hub"), (_PRIVATE_STORAGE,))


def _hermes_homes(turn) -> list[Path]:
    """Every Hermes home on this host, resolved: the active one, the root and each of its profiles
    (main's credential-denylist homes), plus the room owner's when it runs elsewhere."""
    from gateway.platforms.base import _credential_home_roots
    from hermes_constants import get_default_hermes_root, get_hermes_home
    root = get_default_hermes_root()
    homes = [get_hermes_home(), root, *_credential_home_roots(), Path(turn.db_path).parent]
    if turn.transport and turn.transport.get("source_home"):
        homes.append(Path(str(turn.transport["source_home"])))
    with suppress(OSError):
        homes.extend(path for path in (root / "profiles").iterdir() if path.is_dir())
    resolved: list[Path] = []
    for home in homes:
        with suppress(OSError, RuntimeError, ValueError):
            resolved.append(home.expanduser().resolve())
    return list(dict.fromkeys(resolved))


def _private(path: str, homes: list[Path]) -> bool:
    """Whether a host path, judged by its target, is a secret or private Hermes file: main's read
    guard, or a credential file or private tree of any Hermes home."""
    from agent.file_safety import _CREDENTIAL_FILE_NAMES, get_read_block_error
    from gateway.platforms.base import _ROOT_CREDENTIAL_PATHS
    if get_read_block_error(path):
        return True
    from hermes_constants import get_hermes_home
    target = Path(path).resolve()
    active = get_hermes_home().resolve()
    if any(home != active and target.is_relative_to(home) and not target.is_relative_to(active)
           for home in homes):
        return True
    if any(home != active and home.is_relative_to(active) and target.is_relative_to(home)
           for home in homes):
        return True
    if _PRIVATE_STORAGE in target.parts:
        return True
    for home in homes:
        with suppress(OSError, RuntimeError, ValueError):
            if any(target.is_relative_to((home / name).resolve())
                   for name in (*_CREDENTIAL_FILE_NAMES, *_ROOT_CREDENTIAL_PATHS)) or any(
                    target.is_relative_to(home.joinpath(*tree).resolve()) for tree in _PRIVATE_TREES):
                return True
    return False


def _open_regular(path: str) -> int:
    """Open the validated absolute path without following replaced parent directories."""
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_BINARY", 0)
    if not hasattr(os, "O_DIRECTORY"):
        return os.open(path, flags)
    parts = Path(path).parts
    directory = os.open(parts[0], flags | os.O_DIRECTORY)
    try:
        for part in parts[1:-1]:
            child = os.open(part, flags | os.O_DIRECTORY, dir_fd=directory)
            os.close(directory)
            directory = child
        return os.open(parts[-1], flags | os.O_NONBLOCK, dir_fd=directory)
    finally:
        os.close(directory)


def _read_bounded(path: str, limit: int) -> bytes:
    """A regular file's bytes, opened without following a final symlink and fenced by size."""
    descriptor = _open_regular(path)
    try:
        info = os.fstat(descriptor)
        if info.st_nlink != 1:
            raise ValueError(_UNSAFE_PATH)
        if not stat.S_ISREG(info.st_mode) or not 0 < info.st_size <= limit:
            raise ValueError(f"A shared file must be a non-empty file of at most {limit // 1_000_000} MB.")
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            data = handle.read(limit + 1)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if len(data) != info.st_size or any(getattr(info, field) != getattr(after, field) for field in (
            "st_dev", "st_ino", "st_mode", "st_nlink", "st_size", "st_mtime_ns", "st_ctime_ns")):
        raise ValueError("That file changed while it was being shared. Try again.")
    return data


def _read_file(path: str, homes: list[Path], limit: int) -> bytes:
    """Read the validated source, locally or on the active terminal backend."""
    from gateway.platforms.base import validate_media_delivery_path
    from gateway.session_context import get_session_env
    safe = validate_media_delivery_path(path, session_key=get_session_env("HERMES_SESSION_KEY", ""))
    if safe is not None:
        if _private(safe, homes):
            raise ValueError(_UNSAFE_PATH)
        return _read_bounded(safe, limit)
    from tools.group_file_remote import read_remote_file
    return read_remote_file(path, limit)


def share_group_file(path: str, name: str | None = None) -> str:
    from gateway.hosted_room_attachments import MAX_ATTACHMENT_BYTES
    from gateway.hosted_room_outputs import current_turn, stage
    turn = current_turn()
    if turn is None:
        return tool_error(_UNAVAILABLE)
    requested = str(path or "").strip()
    if not requested or not os.path.isabs(os.path.expanduser(requested)):
        return tool_error("Give the absolute path of the file to share.")
    try:
        data = _read_file(requested, _hermes_homes(turn), MAX_ATTACHMENT_BYTES)
        # Keep the requested name even when the source resolves through an alias.
        shared_name = name or os.path.basename(requested.replace("\\", "/").rstrip("/"))
        stored = stage(turn, name=shared_name.strip(), data=data)
    except ValueError as exc:
        return tool_error(str(exc))
    except OSError:
        return tool_error("That file could not be read. Check the path and try again.")
    return tool_result(shared=True, name=stored["name"], size=stored["size"], sha256=stored["sha256"],
                       message=f"{stored['name']} will be shared with this group chat when your turn completes.")


SHARE_GROUP_FILE_SCHEMA = {
    "name": "share_group_file",
    "description": (
        "Share one file with the current group chat. It is published with your reply when your turn "
        "completes, and the user and the other bots can open that exact version; it is never published "
        "if your turn is stopped or fails. Never share credentials or secrets."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Absolute path of the file to share."},
            "name": {"type": "string", "description": "Optional file name shown in the group chat."},
        },
        "required": ["path"],
    },
}


registry.register(
    name="share_group_file",
    toolset="bot_room",
    schema=SHARE_GROUP_FILE_SCHEMA,
    handler=lambda args, **kw: share_group_file(args.get("path", ""), name=args.get("name")),
    emoji="📎",
)
