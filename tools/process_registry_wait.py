"""How long ``process_manage action=wait`` may block for one call.

Shared by :meth:`ProcessRegistry.wait`, which enforces the window, and by the tool's
``deadline_floor`` declaration, which lets the generic tool deadline clear that same window
instead of cutting the wait short (see ``ToolEntry.deadline_floor``).
"""

import os


def max_wait_timeout() -> int:
    """``TERMINAL_TIMEOUT``: the default wait, and the cap on a requested one."""
    try:
        return int(os.getenv("TERMINAL_TIMEOUT", "180"))
    except (ValueError, TypeError):
        return 180


def process_deadline_floor(args: dict) -> float | None:
    """Only ``wait`` blocks; it mirrors :meth:`ProcessRegistry.wait` — the requested window
    clamped by ``TERMINAL_TIMEOUT``, or that default when none was requested."""
    if args.get("action") != "wait":
        return None
    max_timeout = float(max_wait_timeout())
    requested = args.get("timeout")
    if requested is None:
        return max_timeout
    try:
        requested = float(requested)
    except (TypeError, ValueError):
        return max_timeout
    return min(requested, max_timeout) if requested > 0 else None
