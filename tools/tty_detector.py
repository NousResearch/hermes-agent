#!/usr/bin/env python3
"""
Adaptive TTY Detection — identify interactive terminal sessions.

Purpose: Detect when command output is from an interactive terminal program
(vim, less, ssh, top, etc.) so compression can be safely skipped. Interactive
output contains ANSI escape codes and full-screen terminal UI that our
compressors cannot parse.

Detection strategy (in priority order):
1. ANSI escape scan  — most reliable; interactive programs emit VT100 codes
2. Command blocklist  — known interactive programs by name
3. Output size heuristic — interactive output is typically small (<5KB)
"""

import re
import shlex
from typing import Literal

# -----------------------------------------------------------------------------
# ANSI Escape Sequence Detection
# -----------------------------------------------------------------------------
# Regex patterns for common interactive terminal control sequences.
# These are emitted by vim, less, top, tmux, ssh, and other full-screen programs.

ANSI_ESCAPE_PATTERNS = [
    # Clear screen / clear line
    re.compile(r"\x1b\[[0-9;]*[HJ]"),   # DECSC/DECRC: cursor position queries
    # Cursor movement: ESC [ <n> ; <n> H (position), ESC [ <n> A/B/C/D (move)
    re.compile(r"\x1b\[[0-9;]*[HfABCDH]"),
    # Erase functions: ESC [ <n> J (erase in display), ESC [ <n> P (erase chars)
    re.compile(r"\x1b\[[0-9;]*[JP]"),
    # SGR (Select Graphic Rendition) — colors, bold, underline
    re.compile(r"\x1b\[[0-9;]*m"),
    # Screen mode changes: ESC [ ? <n> h/l (DEC private mode set/reset)
    re.compile(r"\x1b\[\?[0-9;]*[hl]"),
    # tmux control sequences: ESC ] 0 ; <title> BEL (set window title)
    re.compile(r"\x1b\]0;"),
    # Backspace cursor movement
    re.compile(r"\x08\x1b\[[0-9;]*[ABCD]"),
    # Any raw ESC followed by a letter (vim uses many ESC-letter sequences)
    re.compile(r"\x1b[a-zA-Z]"),
]

# Threshold: interactive programs produce relatively small output.
# Full-screen programs redraw the entire screen (typically <10KB per frame).
# Batch output (pytest, git diff) is always large when compressed.
INTERACTIVE_SIZE_THRESHOLD = 5 * 1024  # 5 KB


def contains_ansi_escape_codes(data: str) -> bool:
    """
    Return True if the string contains ANSI/VT100 terminal escape sequences.

    Scans for the most common control sequences used by interactive programs:
    - Cursor positioning and movement
    - Clear screen / erase functions
    - SGR (color/bold/underline)
    - DEC private mode sequences (tmux, screen)
    - Window title sequences (OSC)

    Only matches genuine escape sequences (ESC + printable chars).
    Skips raw ESC bytes that might appear in binary data.
    """
    for pattern in ANSI_ESCAPE_PATTERNS:
        if pattern.search(data):
            return True
    return False


# -----------------------------------------------------------------------------
# Interactive Command Blocklist
# -----------------------------------------------------------------------------
# Known interactive terminal programs. Commands matching these names will have
# their output passed through uncompressed regardless of output content.

# Format: (normalized_cmd, extra_args_indicating_interactive)
# extra_args_indicating_interactive: list of substrings that, if found in the
# command line after the program name, confirm interactivity.
INTERACTIVE_COMMANDS: list[tuple[str, list[str]]] = [
    # Full-screen text editors
    ("vim", []),
    ("nvim", []),
    ("vi", []),
    ("nano", []),
    ("micro", []),
    ("emacs", ["-nw"]),
    # File viewers / pagers (interactive by nature)
    ("less", []),
    ("more", []),
    ("most", []),
    # Remote access (interactive sessions)
    ("ssh", []),
    ("mosh", []),
    ("telnet", []),
    # System monitors
    ("top", []),
    ("htop", []),
    ("btm", []),
    ("bashtop", []),
    ("bpytop", []),
    ("glances", []),
    ("nmon", []),
    ("vmstat", []),
    ("iostat", []),
    # Terminal multiplexers
    ("tmux", []),
    ("screen", []),
    ("byobu", []),
    # Chat / IM clients
    ("irssi", []),
    ("weechat", []),
    ("bitlbee", []),
    ("pidgin", []),
    # News / email readers
    ("tin", []),
    ("slrn", []),
    ("mutt", []),
    ("neomutt", []),
    ("alpine", []),
    # Web browsers
    ("links", []),
    ("elinks", []),
    ("lynx", []),
    ("w3m", []),
    # File managers
    ("mc", []),          # Midnight Commander
    ("ranger", []),
    ("nnn", []),
    ("vifm", []),
    ("lf", []),
    # Git TUI
    ("tig", []),
    # SQL clients
    ("mysql", []),
    ("psql", []),
    ("sqlite3", []),
    ("mongosh", []),
    # Debuggers / REPLs
    ("gdb", []),
    ("lldb", []),
    ("pdb", []),
    ("ipdb", []),
    ("pry", []),
    ("node", []),         # Node REPL (interactive by default)
    ("python", ["-i"]),
    ("python3", ["-i"]),
    ("ruby", ["-i"]),
    ("perl", []),
    # Docker / K8s interactives
    ("docker", ["-it"]),
    ("kubectl", ["-it"]),
    ("podman", ["run", "-it"]),
    # Shell itself (when invoked interactively — less common)
    ("bash", ["-i"]),
    ("zsh", ["-i"]),
    ("sh", ["-i"]),
]

# Fast lookup set for commands that are always interactive (no args needed)
ALWAYS_INTERACTIVE: set[str] = {
    cmd for cmd, _ in INTERACTIVE_COMMANDS
}

# Commands that are only interactive when certain flags are present
FLAG_GATED_INTERACTIVE: dict[str, list[str]] = {
    cmd: args for cmd, args in INTERACTIVE_COMMANDS if args
}


def _normalize_cmd_for_match(command: str) -> str:
    """Get the base command name for blocklist matching."""
    cmd = command.strip()
    # Remove sudo prefix
    if cmd.startswith("sudo "):
        cmd = cmd[5:]
    # Only strip leading path components (e.g. /usr/bin/vim → vim)
    # Do NOT strip all / segments — args like /var/log/syslog contain / legitimately
    if cmd.startswith("/"):
        # Absolute path: take the last component
        cmd = cmd.rstrip("/").split("/")[-1]
    # Now extract just the base command (first token), ignore all flags/args
    try:
        parts = shlex.split(cmd)
    except ValueError:
        # Unbalanced quotes — fall back to first whitespace-delimited word
        parts = cmd.split()
    return parts[0] if parts else cmd


def _cmd_has_interactive_flag(command: str, expected_flags: list[str]) -> bool:
    """
    Check if command line contains any of the expected interactive flags.

    Only flags that start with '-' are considered. Positional subcommands
    (e.g., 'docker run', 'ssh remote-host') are ignored.
    """
    cmd = command.strip()
    if cmd.startswith("sudo "):
        cmd = cmd[5:]
    try:
        parts = shlex.split(cmd)
    except ValueError:
        parts = cmd.split()
    if len(parts) < 2:
        return False
    # Only check args that look like flags (start with -)
    args = parts[1:]
    for flag in expected_flags:
        if flag.startswith("-") and flag in args:
            return True
        # Handle short flags: --flag or -f
        if flag.startswith("--") and any(arg.startswith("--" + flag.lstrip("-")) for arg in args):
            return True
    return False


def is_interactive_command(command: str) -> bool:
    """
    Return True if the command is a known interactive terminal program.

    Checks:
    1. Is the base command name in the always-interactive blocklist?
    2. Does it have interactive-gating flags (e.g., `python -i`, `docker run -it`)?
    """
    normalized = _normalize_cmd_for_match(command)
    # Check always-interactive set first (O(1) lookup)
    if normalized in ALWAYS_INTERACTIVE:
        # For flag-gated commands, verify the flag is actually present
        if normalized in FLAG_GATED_INTERACTIVE:
            return _cmd_has_interactive_flag(command, FLAG_GATED_INTERACTIVE[normalized])
        return True
    return False


# -----------------------------------------------------------------------------
# Combined Adaptive Detection
# -----------------------------------------------------------------------------

DetectionReason = Literal["ansi", "blocklist", "size", "none"]


def should_skip_compression(
    command: str,
    stdout: str,
    stderr: str,
) -> tuple[bool, DetectionReason]:
    """
    Determine whether compression should be skipped for this command output.

    Checks (in priority order):
    1. ANSI escape codes in output → definitely interactive (most reliable signal)
    2. Command is in the interactive blocklist → skip

    Returns (should_skip, reason). If should_skip is False, compression is safe.
    """
    combined = stdout + stderr

    # 1. ANSI escape codes — most reliable interactive signal
    if contains_ansi_escape_codes(combined):
        return True, "ansi"

    # 2. Command blocklist
    if is_interactive_command(command):
        return True, "blocklist"

    # 3. Output size heuristic — interactive programs produce small output
    # Batch commands (git diff, pytest, etc.) produce large output; compression is safe
    if len(combined) < INTERACTIVE_SIZE_THRESHOLD:
        return True, "size"

    return False, "none"
