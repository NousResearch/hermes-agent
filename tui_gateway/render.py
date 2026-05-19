"""Safe TUI rendering fallback helpers.

The gateway deliberately does not import optional renderer modules from the
agent package. The TUI process already has a client-side markdown renderer, and
late Python imports from model-writable package paths can execute planted code.
"""

from __future__ import annotations


def render_message(text: str, cols: int = 80) -> str | None:
    """Return ``None`` so the TUI uses its built-in message renderer."""
    return None


def render_diff(text: str, cols: int = 80) -> str | None:
    """Return ``None`` so the TUI uses its built-in diff renderer."""
    return None


def make_stream_renderer(cols: int = 80):
    """Return ``None`` to disable Python-side streaming render hooks."""
    return None
