"""Terminal environment for PTY-mode background sessions: pagers and window size (#7321)."""

import shutil

# The historical fixed window, kept wherever there is no terminal to copy (gateway, cron, CI).
# ``shutil.get_terminal_size`` takes and returns (columns, lines).
_HEADLESS_SIZE = (120, 30)


def pty_terminal_defaults(env: dict[str, str]) -> tuple[int, int]:
    """Fill *env* in for a child on a real PTY; return the ``(rows, cols)`` to spawn it with.

    A PTY is a real TTY, so pager-happy tools (git log/diff, man) WILL page and hang waiting
    for `q` — default them to cat, honoring any pager the user set. The window copies the real
    terminal instead of a fixed 30x120, and the child is told the same size: plenty of TUIs read
    COLUMNS/LINES before falling back to ioctl(TIOCGWINSZ). ``get_terminal_size`` already prefers
    those env vars, and its fallback is the historical 30x120, so a headless host behaves exactly
    as before. ``setdefault`` throughout: a value the caller passed in still wins.
    """
    env.setdefault("GIT_PAGER", "cat")
    env.setdefault("PAGER", "cat")
    cols, rows = shutil.get_terminal_size(fallback=_HEADLESS_SIZE)
    env.setdefault("COLUMNS", str(cols))
    env.setdefault("LINES", str(rows))
    return rows, cols
