"""Origin-module lookup shared by the ``tools.browser_tool_*`` extraction modules.

Extracted code reads facade-owned state (caches, locks, session tables, facade-defined
helpers) *through* ``tools.browser_tool`` and must not import it at import time (cycle) —
hence the lazy :func:`origin_module` and the :data:`origin` proxy.
"""

import sys
from types import ModuleType

_ORIGIN_NAME = "tools.browser_tool"


class _NamespaceView:
    """Live view over a module namespace whose module object is gone (a test purged
    ``sys.modules``): the old code still runs with its own globals, so hit *those*."""

    __slots__ = ("_g",)

    def __init__(self, g):
        object.__setattr__(self, "_g", g)

    def __getattr__(self, name):
        try:
            return self._g[name]
        except KeyError:
            raise AttributeError(name) from None

    def __setattr__(self, name, value):
        self._g[name] = value


def _module_for_globals(g: dict):
    """The module object whose namespace is ``g`` (or a live view over it)."""
    import tools

    for mod in (sys.modules.get(_ORIGIN_NAME), getattr(tools, "browser_tool", None)):
        if mod is not None and mod.__dict__ is g:
            return mod
    return _NamespaceView(g)


def origin_module(_depth: int = 2):
    """The ``tools.browser_tool`` instance the *calling* moved function belongs to, mirroring
    what in-file code would see: (1) the nearest enclosing frame executing origin code (the
    exact module copy, even after a test purged/reloaded ``sys.modules``); (2) an origin
    module referenced from a calling frame's globals; (3) ``sys.modules`` / ``tools``
    package attribute / fresh import. ``_depth`` is the frame of the moved function's caller."""
    try:
        start = sys._getframe(_depth)
    except ValueError:  # called directly by the interpreter (atexit callback)
        start = None
    frame = start
    while frame is not None:
        if frame.f_globals.get("__name__") == _ORIGIN_NAME:
            return _module_for_globals(frame.f_globals)
        frame = frame.f_back
    frame = start
    while frame is not None:
        for value in list(frame.f_globals.values()):
            if isinstance(value, ModuleType) and getattr(value, "__name__", None) == _ORIGIN_NAME:
                return value
        frame = frame.f_back
    mod = sys.modules.get(_ORIGIN_NAME)
    if mod is None:
        import tools

        mod = getattr(tools, "browser_tool", None)
    if mod is None:
        import tools.browser_tool as mod
    return mod


class _OriginProxy:
    """Module-level ``_bt`` stand-in: every attribute read/write forwards to the origin
    module resolved at that moment."""

    __slots__ = ()

    def __getattr__(self, name):
        return getattr(origin_module(3), name)

    def __setattr__(self, name, value):
        setattr(origin_module(3), name, value)


origin = _OriginProxy()
