"""Rendering bridge — routes TUI content through Python-side renderers.

When agent.rich_output exists, its functions are used. When it doesn't,
everything returns None and the TUI falls back to its own markdown.tsx.
"""

from __future__ import annotations

import importlib
import inspect
import logging

logger = logging.getLogger(__name__)


def _accepts_cols(fn) -> bool:
    """Whether ``fn`` takes the ``cols`` keyword (explicitly, or via ``**kwargs``).

    An unintrospectable callable is assumed to take it; the caller's exception
    handling then decides."""
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return True
    return "cols" in params or any(
        p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()
    )


def _rich(name: str, *args, cols: int):
    """Call ``agent.rich_output.<name>(*args, cols=cols)`` — without ``cols`` for an older
    signature; None when the module is missing or the renderer fails. The ``cols`` decision
    is made from the signature, never by catching a ``TypeError`` (which a renderer's own
    body may raise) and re-invoking it."""
    try:
        fn = getattr(importlib.import_module("agent.rich_output"), name)
    except (ImportError, AttributeError):
        return None
    try:
        return fn(*args, cols=cols) if _accepts_cols(fn) else fn(*args)
    except Exception:
        logger.debug("rich_output.%s failed; TUI falls back to markdown", name, exc_info=True)
        return None


def render_message(text: str, cols: int = 80) -> str | None:
    return _rich("format_response", text, cols=cols)


def render_diff(text: str, cols: int = 80) -> str | None:
    return _rich("render_diff", text, cols=cols)


def make_stream_renderer(cols: int = 80):
    return _rich("StreamingRenderer", cols=cols)
