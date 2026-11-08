"""Rendering bridge — routes TUI content through Python-side renderers.

When agent.rich_output exists, its functions are used. When it doesn't,
everything returns None and the TUI falls back to its own markdown.tsx.
"""

from __future__ import annotations

import importlib


def _rich(name: str, *args, cols: int):
    """Call ``agent.rich_output.<name>(*args, cols=cols)``; retry without ``cols`` for older
    signatures; None when the module is missing or the renderer fails."""
    try:
        fn = getattr(importlib.import_module("agent.rich_output"), name)
    except (ImportError, AttributeError):
        return None
    try:
        return fn(*args, cols=cols)
    except TypeError:
        return fn(*args)
    except Exception:
        return None


def render_message(text: str, cols: int = 80) -> str | None:
    return _rich("format_response", text, cols=cols)


def render_diff(text: str, cols: int = 80) -> str | None:
    return _rich("render_diff", text, cols=cols)


def make_stream_renderer(cols: int = 80):
    return _rich("StreamingRenderer", cols=cols)
