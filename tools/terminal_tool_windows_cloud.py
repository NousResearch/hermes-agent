"""Windows Cloud Files guard helpers for terminal traversal (#97898).

The terminal is intentionally arbitrary shell. This module only recognizes
well-known recursive discovery commands and reasons lexically about their
roots; it never stats candidate cloud files, because that access can itself
hydrate a placeholder.
"""

from __future__ import annotations

import ntpath
import re
import shlex
from typing import Mapping, Sequence


_CLOUD_ROOT_ENV_VARS = (
    "OneDrive",
    "OneDriveConsumer",
    "OneDriveCommercial",
    "OneDriveBusiness",
    "iCloudDrive",
    "I_CLOUD_DRIVE_LOCATION",
)
_ICLOUD_PROFILE_SUBDIRS = (
    "iCloudDrive",
    "iCloud Photos",
    r"Pictures\iCloud Photos",
)

_DU_VALUE_OPTIONS = frozenset({
    "-B", "--block-size", "-d", "--max-depth", "--exclude",
    "--exclude-from", "-t", "--threshold", "--files0-from",
})
_RG_VALUE_OPTIONS = frozenset({
    "-g", "--glob", "-t", "--type", "-T", "--type-not", "-e", "--regexp",
    "-f", "--file", "--iglob", "--ignore-file", "--max-depth", "--max-count",
    "-m", "-A", "-B", "-C", "--context", "--sort", "--sortr", "--encoding",
    "--engine", "-r", "--replace",
})
_GREP_VALUE_OPTIONS = frozenset({
    "-e", "--regexp", "-f", "--file", "-m", "--max-count",
    "-A", "--after-context", "-B", "--before-context", "-C", "--context",
    "--include", "--exclude", "--exclude-from", "--exclude-dir",
})


def _env_get(environ: Mapping[str, object], name: str) -> str:
    """Case-insensitive Windows environment lookup."""
    wanted = name.lower()
    for key, value in environ.items():
        if str(key).lower() == wanted and value:
            return str(value)
    return ""


def _native_windows_path(value: str) -> str:
    """Git-Bash/MSYS drive spelling to native Windows spelling."""
    match = re.match(r"^/(?:(?:cygdrive|mnt)/)?([A-Za-z])(?:/(.*))?$", value)
    if match:
        tail = (match.group(2) or "").replace("/", "\\")
        return f"{match.group(1)}:\\{tail}" if tail else f"{match.group(1)}:\\"
    return value.replace("/", "\\")


def _traversal_anchor(raw: str, *, cwd: str, profile: str) -> str | None:
    """Absolute lexical root a recursive command can descend beneath.

    Glob operands use the directory prefix before the first wildcard:
    './*' therefore means the current directory, while 'src/*.py' means
    'src'. No filesystem calls are made.
    """
    raw = raw.strip()
    if not raw:
        return None
    if raw == "~" or raw.startswith(("~/", "~\\")):
        raw = profile + raw[1:]
    else:
        for prefix in ("$HOME", "${HOME}", "$USERPROFILE", "${USERPROFILE}", "%USERPROFILE%"):
            if raw == prefix or raw.startswith((prefix + "/", prefix + "\\")):
                raw = profile + raw[len(prefix):]
                break

    wildcard = min(
        (index for index in (raw.find("*"), raw.find("?"), raw.find("[")) if index >= 0),
        default=-1,
    )
    if wildcard >= 0:
        prefix = raw[:wildcard]
        raw = (
            prefix.rstrip("/\\") or "."
            if prefix.endswith(("/", "\\"))
            else ntpath.dirname(prefix) or "."
        )

    native = _native_windows_path(raw)
    base = _native_windows_path(cwd)
    if not ntpath.isabs(native):
        native = ntpath.join(base, native)
    return ntpath.normcase(ntpath.normpath(native))


_NATIVE_UNQUOTED_WINDOWS_PATH = re.compile(
    r"(?<![\w\"'])([A-Za-z]:\\[^\s;&|]*)"
)


def _preserve_native_windows_path_backslashes(command: str) -> str:
    """Protect unquoted native path separators from POSIX shlex escaping."""
    def protect(match: re.Match[str]) -> str:
        value = match.group(1)
        chars: list[str] = []
        for index, char in enumerate(value):
            if (
                char == "\\"
                and index + 1 < len(value)
                and value[index + 1] not in ";&|#'\"\\$"
            ):
                chars.append("\\\\")
            else:
                chars.append(char)
        return "".join(chars)

    return _NATIVE_UNQUOTED_WINDOWS_PATH.sub(protect, command)


def _shell_segments(command: str) -> list[list[str]]:
    """Tokenize simple shell command units while respecting quoted words."""
    try:
        lexer = shlex.shlex(
            _preserve_native_windows_path_backslashes(command),
            posix=True,
            punctuation_chars=";&|",
        )
        lexer.whitespace_split = True
        lexer.commenters = "#"
        segments: list[list[str]] = []
        current: list[str] = []
        for token in lexer:
            if token and all(char in ";&|" for char in token):
                if current:
                    segments.append(current)
                    current = []
            else:
                current.append(token)
        if current:
            segments.append(current)
        return segments
    except ValueError:
        return []


def _positionals(args: Sequence[str], value_options: frozenset[str]) -> list[str]:
    result: list[str] = []
    index = 0
    positional_only = False
    while index < len(args):
        token = args[index]
        if positional_only:
            result.append(token)
            index += 1
            continue
        if token == "--":
            positional_only = True
            index += 1
            continue
        option = token.split("=", 1)[0]
        if option in value_options and "=" not in token:
            index += 2
            continue
        if token.startswith("-") and token != "-":
            index += 1
            continue
        result.append(token)
        index += 1
    return result


def _recursive_roots(tokens: Sequence[str]) -> list[str]:
    """Traversal roots for the recursive command in one shell segment."""
    if not tokens:
        return []
    name = tokens[0].replace("\\", "/").rsplit("/", 1)[-1].lower()
    args = list(tokens[1:])

    if name == "find":
        roots = []
        for token in args:
            if token == "--":
                continue
            if token.startswith("-") or token in {"!", "(", ")", ","}:
                break
            roots.append(token)
        return roots or ["."]

    if name == "du":
        return _positionals(args, _DU_VALUE_OPTIONS) or ["."]

    if name == "rg":
        positionals = _positionals(args, _RG_VALUE_OPTIONS)
        if "--files" in args:
            return positionals or ["."]
        explicit_pattern = any(
            token in {"-e", "--regexp"} or token.startswith("--regexp=")
            for token in args
        )
        roots = positionals if explicit_pattern else positionals[1:]
        return roots or ["."]

    if name == "grep":
        recursive = any(
            token in {"-r", "-R", "--recursive"}
            or (
                token.startswith("-")
                and not token.startswith("--")
                and any(flag in token[1:] for flag in "rR")
            )
            for token in args
        )
        if not recursive:
            return []
        positionals = _positionals(args, _GREP_VALUE_OPTIONS)
        explicit_pattern = any(
            token in {"-e", "--regexp", "-f", "--file"}
            or token.startswith(("--regexp=", "--file="))
            for token in args
        )
        roots = positionals if explicit_pattern else positionals[1:]
        return roots or ["."]

    return []


def cloud_placeholder_traversal_reason(
    command: str, *, cwd: str, environ: Mapping[str, object]
) -> str | None:
    """Why a Windows-local command risks broad Cloud Files hydration.

    The user profile itself (or any ancestor) is always protected because
    iCloud does not reliably publish a root environment variable. Below the
    profile, only configured OneDrive/iCloud roots are protected. An explicit
    command rooted at or inside a cloud directory is allowed.
    """
    profile = _env_get(environ, "USERPROFILE")
    if not profile:
        profile = _env_get(environ, "HOMEDRIVE") + _env_get(environ, "HOMEPATH")
    profile_root = _traversal_anchor(profile, cwd=cwd, profile=profile) if profile else None
    if not profile_root:
        return None

    cloud_roots: list[str] = []
    candidates = [
        _env_get(environ, name) for name in _CLOUD_ROOT_ENV_VARS
    ] + [ntpath.join(profile, subdir) for subdir in _ICLOUD_PROFILE_SUBDIRS]
    for value in candidates:
        if not value or not ntpath.isabs(_native_windows_path(value)):
            continue
        root = _traversal_anchor(value, cwd=cwd, profile=profile)
        if root and root not in cloud_roots:
            cloud_roots.append(root)

    segment_cwd = cwd
    for segment in _shell_segments(command):
        if not segment:
            continue
        name = segment[0].replace("\\", "/").rsplit("/", 1)[-1].lower()
        if name == "cd" and len(segment) >= 2 and not segment[1].startswith("-"):
            changed = _traversal_anchor(segment[1], cwd=segment_cwd, profile=profile)
            if changed:
                segment_cwd = changed
            continue

        for raw_root in _recursive_roots(segment):
            target = _traversal_anchor(raw_root, cwd=segment_cwd, profile=profile)
            if not target:
                continue
            try:
                if ntpath.commonpath([target, profile_root]) == target:
                    return (
                        f"recursive {name} traversal rooted at {raw_root!r} would sweep "
                        "the Windows user profile"
                    )
            except ValueError:
                pass

            for cloud_root in cloud_roots:
                try:
                    crosses_cloud_root = (
                        ntpath.commonpath([target, cloud_root]) == target
                        and target != cloud_root
                    )
                except ValueError:
                    crosses_cloud_root = False
                if crosses_cloud_root:
                    return (
                        f"recursive {name} traversal rooted at {raw_root!r} would cross "
                        f"cloud-sync root {cloud_root!r}"
                    )
    return None
