"""Track short-lived provenance for scripts downloaded by terminal commands.

The dangerous-command detector is intentionally stateless. This module adds the
small amount of session-scoped state needed to connect a successful ``curl`` or
``wget`` download with a later shell/direct execution of that exact path.
"""

from __future__ import annotations

import ntpath
import posixpath
import re
import shlex
import threading
import time
from collections import OrderedDict

_DOWNLOADED_SCRIPT_KEY = "execute recently downloaded script"
_DOWNLOADED_SCRIPT_DESCRIPTION = "execute a script downloaded earlier in this session"
_DOWNLOAD_TTL_SECONDS = 10 * 60
_MAX_TRACKED_PATHS_PER_SESSION = 32

_lock = threading.Lock()
_downloads: dict[str, OrderedDict[str, float]] = {}
_SHELLS = frozenset({"bash", "dash", "ksh", "sh", "zsh"})
_ASSIGNMENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")
_DYNAMIC_PATH_CHARS = frozenset("$`*?[")


def _segments(command: str) -> list[list[str]]:
    """Return quote-aware top-level command argv segments, or no segments."""
    raw_segments: list[str] = []
    start = 0
    quote: str | None = None
    escaped = False
    index = 0
    while index < len(command):
        ch = command[index]
        if escaped:
            escaped = False
        elif ch == "\\" and quote != "'":
            escaped = True
        elif quote:
            if ch == quote:
                quote = None
        elif ch in "'\"":
            quote = ch
        elif ch in ";\n&|":
            raw_segments.append(command[start:index])
            if index + 1 < len(command) and command[index + 1] == ch and ch in "&|":
                index += 1
            start = index + 1
        index += 1
    if quote:
        return []
    raw_segments.append(command[start:])

    parsed: list[list[str]] = []
    for segment in raw_segments:
        try:
            words = shlex.split(segment, posix=True)
        except ValueError:
            return []
        if words:
            parsed.append(words)
    return parsed


def _basename(word: str) -> str:
    return word.replace("\\", "/").rsplit("/", 1)[-1].lower()


def _unwrap(words: list[str]) -> list[str]:
    words = list(words)
    while words and _ASSIGNMENT_RE.match(words[0]):
        words.pop(0)
    if words and _basename(words[0]) == "sudo":
        words.pop(0)
        while words and words[0].startswith("-"):
            option = words.pop(0)
            if option in {"-u", "-g", "-h", "-p", "-C", "-T", "-r", "-t"} and words:
                words.pop(0)
        while words and _ASSIGNMENT_RE.match(words[0]):
            words.pop(0)
    if words and _basename(words[0]) == "env":
        words.pop(0)
        while words and (_ASSIGNMENT_RE.match(words[0]) or words[0] in {"-i", "--ignore-environment"}):
            words.pop(0)
    return words


def _literal_path(path: str, cwd: str) -> str | None:
    if not path or any(ch in path for ch in _DYNAMIC_PATH_CHARS) or path.startswith("~"):
        return None
    windows = bool(re.match(r"^[A-Za-z]:[\\/]", path)) or "\\" in cwd
    pathmod = ntpath if windows else posixpath
    candidate = path if pathmod.isabs(path) else pathmod.join(cwd, path)
    normalized = pathmod.normpath(candidate)
    return pathmod.normcase(normalized) if windows else normalized


def _download_targets(command: str, cwd: str) -> set[str]:
    targets: set[str] = set()
    for raw_words in _segments(command):
        words = _unwrap(raw_words)
        if not words:
            continue
        program = _basename(words[0])
        args = words[1:]
        target: str | None = None
        for index, arg in enumerate(args):
            if program == "curl":
                if arg in {"-o", "--output"} and index + 1 < len(args):
                    target = args[index + 1]
                    break
                if arg.startswith("--output="):
                    target = arg.split("=", 1)[1]
                    break
                if arg.startswith("-o") and len(arg) > 2:
                    target = arg[2:]
                    break
            elif program == "wget":
                if arg in {"-O", "--output-document"} and index + 1 < len(args):
                    target = args[index + 1]
                    break
                if arg.startswith("--output-document="):
                    target = arg.split("=", 1)[1]
                    break
                if arg.startswith("-O") and len(arg) > 2:
                    target = arg[2:]
                    break
        literal = _literal_path(target or "", cwd)
        if literal:
            targets.add(literal)
    return targets


def _execution_targets(command: str, cwd: str) -> set[str]:
    targets: set[str] = set()
    for raw_words in _segments(command):
        words = _unwrap(raw_words)
        if not words:
            continue
        program = _basename(words[0])
        candidate: str | None = None
        if program in _SHELLS:
            for arg in words[1:]:
                if arg == "--":
                    continue
                if arg in {"-c", "--command"} or arg.startswith("-c"):
                    candidate = None
                    break
                if arg.startswith("-"):
                    continue
                candidate = arg
                break
        elif words[0].startswith(("./", "../", "/", ".\\", "..\\")) or re.match(
            r"^[A-Za-z]:[\\/]", words[0]
        ):
            candidate = words[0]
        literal = _literal_path(candidate or "", cwd)
        if literal:
            targets.add(literal)
    return targets


def _prune_locked(session_key: str, now: float) -> OrderedDict[str, float]:
    entries = _downloads.setdefault(session_key, OrderedDict())
    for path, expires_at in list(entries.items()):
        if expires_at <= now:
            entries.pop(path, None)
    if not entries:
        _downloads.pop(session_key, None)
        return OrderedDict()
    return entries


def downloaded_script_finding(
    command: str, *, session_key: str, cwd: str, now: float | None = None
) -> tuple[str, str] | None:
    """Return an approval finding when *command* executes a recent download."""
    current = time.monotonic() if now is None else now
    executions = _execution_targets(command, cwd)
    if not executions:
        return None
    downloads_in_command = _download_targets(command, cwd)
    key = str(session_key or "default")
    with _lock:
        tracked = set(_prune_locked(key, current))
    if executions & (tracked | downloads_in_command):
        return (_DOWNLOADED_SCRIPT_KEY, _DOWNLOADED_SCRIPT_DESCRIPTION)
    return None


def record_successful_command(
    command: str, *, session_key: str, cwd: str, exit_code: int, now: float | None = None
) -> None:
    """Update provenance after a successful foreground terminal command."""
    if exit_code != 0:
        return
    current = time.monotonic() if now is None else now
    key = str(session_key or "default")
    downloads = _download_targets(command, cwd)
    executions = _execution_targets(command, cwd)
    with _lock:
        entries = _prune_locked(key, current)
        if downloads:
            entries = _downloads.setdefault(key, entries)
            for path in downloads:
                entries[path] = current + _DOWNLOAD_TTL_SECONDS
                entries.move_to_end(path)
            while len(entries) > _MAX_TRACKED_PATHS_PER_SESSION:
                entries.popitem(last=False)
        for path in executions:
            entries.pop(path, None)
        if not entries:
            _downloads.pop(key, None)


def clear_session(session_key: str) -> None:
    """Drop provenance when terminal session state is torn down."""
    with _lock:
        _downloads.pop(str(session_key or "default"), None)
