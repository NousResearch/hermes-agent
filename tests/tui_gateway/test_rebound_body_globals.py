"""Bodies rebound onto ``tui_gateway.server`` must not read imported names from its globals.

``method_ctx.bind_module`` re-creates every split module's functions against ``server``'s namespace, so
a body that reads a name its own module IMPORTED (a stdlib module, a class) finds nothing there: the
import is not a ``FunctionType``/registry entry, so it is never published (``bind_module`` skips
modules on purpose — one split module's ``sqlite3`` must not shadow another's). The read only fails at
CALL time, which is why neither import nor ruff sees it: ``scripts/code_health/config.py`` excludes
``tui_gateway/*`` from F821 exactly because body names are injected.

#152709: ``_live_count_field`` catches ``sqlite3.Error`` and answers ``{}`` so a locked store degrades
the roster field instead of the whole ``profiles.list``. Rebound, ``sqlite3`` resolved to nothing, so
the except clause raised ``NameError`` — the very failure it was written to absorb.
"""

from __future__ import annotations

import builtins
import dis
import sqlite3
import types

from tui_gateway import server


def _walk_codes(code: types.CodeType):
    yield code
    for const in code.co_consts:
        if isinstance(const, types.CodeType):
            yield from _walk_codes(const)


def _server_global_bodies():
    """Every function that runs on ``server``'s globals: the rebound split-module copies plus its own."""
    namespace = vars(server)
    for name, obj in list(namespace.items()):
        if isinstance(obj, types.FunctionType) and obj.__globals__ is namespace:
            yield name, obj


def test_rebound_bodies_only_read_names_the_server_namespace_publishes():
    namespace = vars(server)
    unresolved: dict[str, set[str]] = {}
    for name, fn in _server_global_bodies():
        for code in _walk_codes(fn.__code__):
            for ins in dis.get_instructions(code):
                # LOAD_GLOBAL only: attribute reads (``db.display_message_count``) are not globals.
                if ins.opname != "LOAD_GLOBAL":
                    continue
                target = ins.argval
                if target in namespace or hasattr(builtins, target):
                    continue
                unresolved.setdefault(target, set()).add(f"{name}.{code.co_name}")
    assert not unresolved, (
        "these rebound bodies read a name server.py does not publish; import it inside the body "
        f"(bind_module never publishes imported modules): {dict(sorted(unresolved.items()))}"
    )


class _LockedCountDB:
    """A store whose read raises the sqlite error ``_live_count_field`` is meant to absorb."""

    def display_message_count(self, session_id: str) -> int:
        raise sqlite3.OperationalError("database is locked")


def test_live_count_field_answers_empty_when_the_store_errors():
    assert server._live_count_field(_LockedCountDB(), "sid") == {}
