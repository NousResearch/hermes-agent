"""Seam for the server.py handler/helper split. server.py's JSON-RPC handlers and helpers close
over its module globals (``_sessions``, ``_ok``, ``_err``, ...). Split modules define their code
normally and server.py calls :func:`bind_module` at the end of its own import, once every global
exists: bodies are re-created with ``types.FunctionType`` against server.py's namespace, so they
stay byte-identical and ``global X`` keeps mutating server.py state. No import cycle: split
modules never import server at module level — server passes itself in."""

import contextlib
import types

# contextlib.contextmanager wraps the generator; rebind the generator (found via
# __wrapped__) and re-wrap, otherwise only the wrapper would see server globals.
_CM_HELPER_CODE = contextlib.contextmanager(lambda: (yield)).__code__


def rebind(fn, g: dict, _seen=None):
    """Copy ``fn`` with globals ``g``; closure cells holding same-module functions are rebound too
    (so handlers produced by import-time decorator factories keep working)."""
    _seen = {} if _seen is None else _seen
    if id(fn) in _seen:
        return _seen[id(fn)]
    wrapped = getattr(fn, "__wrapped__", None)
    if wrapped is not None and fn.__code__ is _CM_HELPER_CODE:
        return contextlib.contextmanager(rebind(wrapped, g, _seen))
    closure = fn.__closure__
    if closure:
        def _cell(cell):
            try:
                val = cell.cell_contents
            except ValueError:  # empty cell
                return cell
            if isinstance(val, types.FunctionType) and val.__module__ == fn.__module__:
                return types.CellType(rebind(val, g, _seen))
            return cell
        closure = tuple(_cell(c) for c in closure)
    real = types.FunctionType(fn.__code__, g, fn.__name__, fn.__defaults__, closure)
    real.__kwdefaults__ = fn.__kwdefaults__
    real.__doc__ = fn.__doc__
    real.__dict__.update(fn.__dict__)
    _seen[id(fn)] = real
    return real


class HandlerRegistry:
    """Deferred @method registrar used by the split modules."""

    def __init__(self) -> None:
        self._pending: list[tuple[str, types.FunctionType]] = []

    def method(self, name: str):
        """Drop-in for server.py's ``@method`` decorator (defers registration)."""
        def dec(fn):
            self._pending.append((name, fn))
            return fn
        return dec

    def profile_scoped(self, fn):
        """Drop-in for server.py's ``@_profile_scoped`` (applied at install)."""
        fn._hermes_profile_scoped = True
        return fn

    def install(self, server) -> None:
        """Rebind pending handlers onto ``server``'s globals and register them."""
        g = vars(server)
        for name, fn in self._pending:
            real = rebind(fn, g)
            if getattr(fn, "_hermes_profile_scoped", False):
                real = server._profile_scoped(real)
            server._methods[name] = real


_PLUMBING = {"HandlerRegistry", "method", "_profile_scoped", "register", "rebind", "logger"}


def bind_module(module_globals: dict, server, *, skip=()) -> None:
    """Publish everything a split module defines onto ``server``, rebound to its globals.
    ``module_globals`` is the caller's ``globals()`` (not ``sys.modules[__name__]``: tests that
    ``patch.dict(sys.modules)`` around the server import drop the submodule entries). Functions
    are rebound; classes get their methods rebound in place; dispatch tables (dict/tuple/list of
    this module's functions) get their values rebound; other values are copied as-is. Imported
    modules/functions, dunders and registry plumbing are skipped; finally ``_registry`` installs."""
    g = vars(server)
    mod_name = module_globals["__name__"]
    seen: dict = {}

    def _own_fn(v):
        return isinstance(v, types.FunctionType) and v.__module__ == mod_name

    def _rebind_in(v):
        if _own_fn(v):
            return rebind(v, g, seen)
        if isinstance(v, dict):
            return {k: _rebind_in(x) for k, x in v.items()}
        return type(v)(_rebind_in(x) for x in v) if isinstance(v, (tuple, list)) else v

    def _has_own_fn(v):
        items = v.values() if isinstance(v, dict) else v if isinstance(v, (tuple, list)) else None
        return _own_fn(v) if items is None else any(_has_own_fn(x) for x in items)

    for name, obj in list(module_globals.items()):
        if (name.startswith("__") or name in _PLUMBING or name in skip
                or isinstance(obj, (types.ModuleType, HandlerRegistry))):
            continue
        if isinstance(obj, types.FunctionType):
            if obj.__module__ == mod_name:
                obj = rebind(obj, g, seen)
            elif name == obj.__name__:
                continue  # plain import; server has its own (``_alias = other.fn`` publishes as-is)
        elif isinstance(obj, (dict, tuple, list)) and _has_own_fn(obj):
            obj = module_globals[name] = _rebind_in(obj)  # keep the split module's own view in sync
        elif isinstance(obj, type):
            if obj.__module__ != mod_name:
                continue
            for attr, val in list(vars(obj).items()):
                if isinstance(val, types.FunctionType):
                    setattr(obj, attr, rebind(val, g))
                elif isinstance(val, (staticmethod, classmethod)):
                    setattr(obj, attr, type(val)(rebind(val.__func__, g)))
        prev = g.get(name)
        if isinstance(prev, types.FunctionType) and isinstance(obj, types.FunctionType):
            owner = getattr(prev, "_hermes_split_module", None)
            if owner and owner != mod_name:
                raise RuntimeError(
                    f"split-module name collision: {mod_name}.{name} would overwrite {owner}.{name}"
                )
            obj._hermes_split_module = mod_name
        setattr(server, name, obj)
    registry = module_globals.get("_registry")
    if isinstance(registry, HandlerRegistry):
        registry.install(server)
