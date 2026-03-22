"""Cross-platform shell quoting utility.

On Unix: uses shlex.quote() (POSIX sh quoting).
On Windows: wraps in double quotes with proper escaping for cmd.exe.

Usage:
    from tools.shell_quote import shell_quote
    cmd = f"whisper {shell_quote(input_path)} --model {shell_quote(model)}"
"""

import shlex
import sys


def shell_quote(s: str) -> str:
    """Quote a string for safe interpolation into a shell command.

    Uses the appropriate quoting strategy for the current platform:
    - Unix: shlex.quote() (single-quote wrapping)
    - Windows: double-quote wrapping with cmd.exe metachar escaping
    """
    if sys.platform != "win32":
        return shlex.quote(s)
    return _win_cmd_quote(s)


def _win_cmd_quote(s: str) -> str:
    """Quote a string for safe use in cmd.exe.

    cmd.exe metacharacters: & | < > ^ " % !
    Strategy: wrap in double quotes, escape internal double quotes with
    backslash, and escape % with %% (environment variable expansion).
    """
    if not s:
        return '""'
    # If the string has no special chars, return as-is
    _CMD_META = set('&|<>^"% !\t')
    if not any(c in _CMD_META for c in s):
        return s
    # Escape internal double quotes and percent signs
    escaped = s.replace('"', '\\"').replace("%", "%%")
    return f'"{escaped}"'
