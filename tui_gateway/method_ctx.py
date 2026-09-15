"""Seam for the server.py facade / sibling split.

server.py is the gateway's public namespace (``_sessions``, ``_ok``, ``_err``, every published helper);
the ``methods_*`` and ``session_*`` siblings are ordinary modules that reach that namespace through
 ``srv`` — a real ``from tui_gateway import server as srv`` placed at the *tail* of each sibling, after
every definition. Attribute access keeps facade state late-bound (``monkeypatch.setattr(server, ...)``
still works) and ty sees server.py's real types. The tail placement is what breaks the import cycle in
both directions: server.py imports the siblings at the end of its own import — once every global
exists — and when a sibling is imported first, its own tail import runs server.py, whose tail then
finds the sibling fully defined. Each sibling's ``register(server)`` calls :func:`bind_module`, which
publishes what the sibling defines onto the facade and installs its ``@method`` handlers. Nothing is
re-created against another module's globals: what ruff and ty resolve is what runs.
"""

import types




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

    def install(self, server, module_globals: dict | None = None) -> None:
        """Register the pending handlers on ``server``. Modules that skip ``bind_module`` pass their
        ``globals()`` so the contract models they import are published too."""
        if module_globals is not None:
            bind_facade(module_globals, server)
            publish_imported_classes(server, module_globals)
        for name, fn in self._pending:
            real = server._profile_scoped(fn) if getattr(fn, "_hermes_profile_scoped", False) else fn
            server.register_method(name, real)


_PLUMBING = {"HandlerRegistry", "method", "_profile_scoped", "register", "logger", "srv", "_"}  # "_": anonymous @method handlers


def bind_facade(module_globals: dict, server) -> None:
    """Point the sibling's ``srv`` at the server instance that is registering it. The tail
    ``from tui_gateway import server as srv`` already did that in any real process; this matters
    when a test re-imports ``tui_gateway.server`` after ``patch.dict(sys.modules)`` dropped the
    first import — the package attribute still names the old instance, so the sibling would
    answer RPCs against the wrong ``_sessions``."""
    if "srv" in module_globals:
        module_globals["srv"] = server


def publish_imported_classes(server, module_globals: dict) -> None:
    """Publish every class ``module_globals`` imported from elsewhere onto ``server`` (see ``publish_imported_class``)."""
    mod_name = module_globals.get("__name__", "")
    for name, obj in module_globals.items():
        if isinstance(obj, type) and obj.__module__ != mod_name:
            publish_imported_class(server, mod_name, name, obj)


def publish_imported_class(server, mod_name: str, name: str, obj: type) -> None:
    """Publish a class a split module imported (contract models, mostly) onto ``server`` so every
    sibling and test can reach it as ``server.Name``. A different object already published under
    the same name is a real clash, not a harmless re-import."""
    prev = getattr(server, name, None)
    if prev is not None and prev is not obj:
        raise RuntimeError(f"split-module name collision: {mod_name} imports {name} but server already binds a different {name}")
    setattr(server, name, obj)


def bind_module(module_globals: dict, server, *, skip=()) -> None:
    """Publish everything a split module defines onto ``server`` (functions, classes, tables, values)
    and install its ``_registry`` handlers. ``module_globals`` is the caller's ``globals()`` (not
    ``sys.modules[__name__]``: tests that ``patch.dict(sys.modules)`` around the server import drop
    the submodule entries). Imported modules/functions, dunders and registry plumbing are skipped;
    imported classes are published so ``server.Name`` resolves for every sibling."""
    mod_name = module_globals["__name__"]
    bind_facade(module_globals, server)
    for name, obj in list(module_globals.items()):
        if (name.startswith("__") or name in _PLUMBING or name in skip
                or isinstance(obj, (types.ModuleType, HandlerRegistry))):
            continue
        if isinstance(obj, types.FunctionType) and obj.__module__ != mod_name and name == obj.__name__:
            continue  # plain import; server has its own (``_alias = other.fn`` publishes as-is)
        if isinstance(obj, type) and obj.__module__ != mod_name:
            publish_imported_class(server, mod_name, name, obj)
            continue
        prev = vars(server).get(name)
        if isinstance(obj, types.FunctionType):
            owner = getattr(prev, "_hermes_split_module", None) if isinstance(prev, types.FunctionType) else None
            if owner and owner != mod_name:
                raise RuntimeError(
                    f"split-module name collision: {mod_name}.{name} would overwrite {owner}.{name}"
                )
            obj._hermes_split_module = mod_name  # stamped on every publish, so the NEXT owner is caught
        setattr(server, name, obj)
    registry = module_globals.get("_registry")
    if isinstance(registry, HandlerRegistry):
        registry.install(server)
