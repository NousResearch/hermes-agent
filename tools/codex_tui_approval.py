"""Approval bridge primitives for Hermes-managed native Codex TUI PTYs.

This is intentionally *not* a general Codex attachment mechanism.  A native
Codex TUI exposes its approval requests only inside its terminal; unrelated
TUI processes have no structured endpoint Hermes can attach to.  The bridge
works by adding process-local key bindings when Hermes itself launches a
Codex PTY, then recognizing Codex's approval screen in that owned PTY.
"""

from __future__ import annotations

import os
import re
import shlex
import shutil
import subprocess
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional

from tools.ansi_strip import sanitize_display_text


# Codex 0.146 is the first version against which Hermes verifies these public
# keymap paths.  Older/unknown versions retain completely native behavior.
_MIN_KEYMAP_VERSION = (0, 146, 0)

APPROVE_ONCE_KEY = "\x07"       # ctrl-g
APPROVE_SESSION_KEY = "\x0f"    # ctrl-o
DENY_KEY = "\x18"               # ctrl-x

_KEYMAP_OVERRIDES = (
    'tui.keymap.approval.approve="ctrl-g"',
    'tui.keymap.approval.approve_for_session="ctrl-o"',
    'tui.keymap.approval.deny="ctrl-x"',
)

_APPROVAL_HEADERS = {
    "Would you like to run the following command?": "command execution",
    "Would you like to make the following edits?": "file changes",
    "Would you like to grant these permissions?": "permission grant",
}
_READY_MARKERS = (
    "Yes, just this once",
    "No, continue without",
)
_SHELL_PUNCTUATION = frozenset({";", "&", "|", "<", ">", "(", ")"})
_NON_INTERACTIVE_SUBCOMMANDS = frozenset(
    {
        "exec",
        "review",
        "mcp",
        "plugin",
        "app-server",
        "mcp-server",
        "remote-control",
        "login",
        "logout",
        "completion",
        "update",
        "doctor",
        "sandbox",
        "debug",
        "features",
        "apply",
        "cloud",
        "remote",
    }
)


def _is_discord_session_key(session_key: str) -> bool:
    """Recognize canonical gateway keys without importing gateway modules."""
    parts = str(session_key or "").split(":")
    return len(parts) >= 4 and parts[2].lower() == "discord"


def _parse_version(text: str) -> Optional[tuple[int, int, int]]:
    match = re.search(r"\b(\d+)\.(\d+)(?:\.(\d+))?\b", text or "")
    if not match:
        return None
    return tuple(int(part or 0) for part in match.groups())


@lru_cache(maxsize=16)
def _supported_codex_executable(executable: str) -> Optional[str]:
    """Return a verified local Codex path without session or network I/O."""
    resolved = executable if os.path.isabs(executable) else shutil.which(executable)
    if not resolved:
        return None
    try:
        result = subprocess.run(
            [resolved, "--version"],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    version = _parse_version(f"{result.stdout}\n{result.stderr}")
    if result.returncode != 0 or version is None or version < _MIN_KEYMAP_VERSION:
        return None
    return os.path.realpath(resolved)


def _split_direct_command(command: str) -> Optional[list[str]]:
    """Return argv only for a single, direct shell command.

    Shell composition is rejected because rewriting a compound command could
    change its meaning, and because Hermes could no longer prove which Codex
    process owns the PTY.
    """
    if not command or "\n" in command or "\r" in command:
        return None
    try:
        lexer = shlex.shlex(command, posix=True, punctuation_chars=";&|<>()")
        lexer.whitespace_split = True
        tokens = list(lexer)
    except ValueError:
        return None
    if not tokens or any(
        token and set(token).issubset(_SHELL_PUNCTUATION) for token in tokens
    ):
        return None
    return tokens


def _is_native_codex_tui(argv: list[str]) -> bool:
    if not argv or os.path.basename(argv[0]).lower() not in {"codex", "codex.exe"}:
        return False
    # Do not compete with an explicit user keymap override.
    if any("tui.keymap.approval." in token for token in argv[1:]):
        return False

    # Find the first positional token while skipping the value-bearing global
    # options Codex accepts before its optional initial prompt.
    value_options = {
        "-c", "--config", "-m", "--model", "-p", "--profile", "-s",
        "--sandbox", "-a", "--ask-for-approval", "-C", "--cd", "--add-dir",
        "--image", "--oss-provider",
    }
    i = 1
    while i < len(argv):
        token = argv[i]
        if token == "--":
            i += 1
            break
        if token in value_options:
            i += 2
            continue
        if token.startswith("-"):
            i += 1
            continue
        break
    return i >= len(argv) or argv[i].lower() not in _NON_INTERACTIVE_SUBCOMMANDS


def prepare_managed_codex_tui_command(
    command: str,
    session_key: str,
    *,
    approval_sink_available: bool,
) -> tuple[str, bool]:
    """Add per-process approval bindings when this PTY is safely bridgeable."""
    if not approval_sink_available or not _is_discord_session_key(session_key):
        return command, False
    argv = _split_direct_command(command)
    if not argv or not _is_native_codex_tui(argv):
        return command, False
    resolved_executable = _supported_codex_executable(argv[0])
    if not resolved_executable:
        return command, False

    # CLI overrides have higher precedence than config.toml and affect only
    # this child.  Put them before the existing argv so an initial prompt is
    # never mistaken for an option value.
    # Launch the exact binary we probed, rather than allowing a login-shell
    # alias/function named ``codex`` to receive approval keystrokes.
    bridged_argv = [resolved_executable]
    for override in _KEYMAP_OVERRIDES:
        bridged_argv.extend(("-c", override))
    bridged_argv.extend(argv[1:])
    return shlex.join(bridged_argv), True


@dataclass(frozen=True)
class CodexApprovalPrompt:
    kind: str
    command: str
    description: str


class CodexTuiApprovalDetector:
    """Recognize a fully-rendered Codex approval prompt in PTY output."""

    _MAX_TEXT = 32_000

    def __init__(self) -> None:
        self._text = ""
        self._scan_offset = 0
        self._pending = False

    def feed(self, chunk: str) -> Optional[CodexApprovalPrompt]:
        if not chunk:
            return None
        clean = sanitize_display_text(chunk).replace("\r\n", "\n").replace("\r", "\n")
        self._text += clean
        if len(self._text) > self._MAX_TEXT:
            removed = len(self._text) - self._MAX_TEXT
            self._text = self._text[removed:]
            self._scan_offset = max(0, self._scan_offset - removed)
        if self._pending:
            return None

        segment = self._text[self._scan_offset:]
        candidates = [
            (segment.rfind(header), header, kind)
            for header, kind in _APPROVAL_HEADERS.items()
        ]
        header_at, header, kind = max(candidates, key=lambda item: item[0])
        if header_at < 0:
            return None
        body = segment[header_at + len(header):]
        marker_positions = [body.find(marker) for marker in _READY_MARKERS]
        marker_positions = [pos for pos in marker_positions if pos >= 0]
        if not marker_positions:
            return None

        body = body[: min(marker_positions)]
        lines = [line.strip() for line in body.splitlines()]
        lines = [line for line in lines if line]
        detail = "\n".join(lines).strip()
        if len(detail) > 1800:
            detail = detail[:1785] + "... [truncated]"
        display = detail or f"Codex requested {kind} in its native TUI"
        self._pending = True
        return CodexApprovalPrompt(
            kind=kind,
            command=display,
            description=f"Native Codex TUI requested {kind}",
        )

    def mark_resolved(self) -> None:
        """Ignore prior screen redraws and begin looking for the next prompt."""
        self._scan_offset = len(self._text)
        self._pending = False


def key_for_choice(choice: Optional[str]) -> str:
    """Map Hermes approval scope onto Codex's process-local choices."""
    if choice == "once":
        return APPROVE_ONCE_KEY
    if choice in {"session", "always"}:
        # Hermes must never turn a chat approval into a global Codex rule.
        return APPROVE_SESSION_KEY
    return DENY_KEY
