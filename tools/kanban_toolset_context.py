"""Explicit Kanban selection during one model-schema build.

The registry's ordinary availability cache is profile-wide, while a gateway
can build schemas for several platforms in the same profile concurrently.
Carry only the explicit selection through a ContextVar, never process env.
"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Iterable, Iterator, Optional

_requested: ContextVar[Optional[bool]] = ContextVar("kanban_toolset_requested", default=None)


def kanban_toolset_requested() -> Optional[bool]:
    """None outside schema assembly; otherwise whether Kanban was named explicitly."""
    return _requested.get()


@contextmanager
def scoped_kanban_toolset_selection(toolsets: Optional[Iterable[str]]) -> Iterator[None]:
    """An all/default selection is not an explicit workflow opt-in."""
    token = _requested.set("kanban" in (toolsets or ()))
    try:
        yield
    finally:
        _requested.reset(token)
