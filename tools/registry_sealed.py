"""Read-only storage for the tool registry once the hosted boot has frozen the registries.

Guarding ``register``/``deregister`` is not enough to make a registration immutable: a caller
that holds the schema dict a getter returned, a ``ToolEntry``, or one of the registry's maps can
still change it in place, and in a shared process that change reaches every tenant's agent.
At the freeze, :func:`seal` replaces each registration's payload with a detached deep copy built
from the containers below; their mutators raise :class:`registry_freeze.RegistryFrozen` while the
registries are frozen.

A sealed container is still a ``dict``/``list``/``set`` (JSON encoding, ``isinstance`` checks and
reads are unchanged), and ``copy.copy``/``copy.deepcopy`` of one give plain, writable containers,
so code that copies a schema before rewriting it (provider sanitizers) keeps working. A local
install never freezes and never seals.
"""

from __future__ import annotations

import copy
from typing import Any

from registry_freeze import refuse_if_frozen

STORAGE = "tool registry storage"


def _refusing(base: type, names: tuple[str, ...]) -> dict[str, Any]:
    def wrap(name: str):
        method = getattr(base, name)

        def guarded(self, *args, **kwargs):
            refuse_if_frozen(STORAGE)
            return method(self, *args, **kwargs)

        guarded.__name__ = name
        return guarded

    return {name: wrap(name) for name in names}


def _plain(value: Any, memo: dict | None = None) -> Any:
    """A writable deep copy: sealed containers become their plain base type."""
    if isinstance(value, dict):
        return {copy.deepcopy(k, memo): _plain(v, memo) for k, v in value.items()}
    if isinstance(value, list):
        return [_plain(v, memo) for v in value]
    if isinstance(value, set):
        return {copy.deepcopy(v, memo) for v in value}
    return copy.deepcopy(value, memo)


# ``__init__`` re-runs construction on a live container (``sealed.__init__({...})`` replaces or
# extends its contents), so it is refused like any other mutator; :func:`seal` builds through the
# base type's ``__init__`` instead.
_DICT_MUTATORS = ("__init__", "__setitem__", "__delitem__", "__ior__", "clear", "pop", "popitem",
                  "setdefault", "update")
_LIST_MUTATORS = ("__init__", "__setitem__", "__delitem__", "__iadd__", "__imul__", "append", "extend",
                  "insert", "pop", "remove", "clear", "sort", "reverse")
_SET_MUTATORS = ("__init__", "__ior__", "__iand__", "__isub__", "__ixor__", "add", "discard", "remove", "pop",
                 "clear", "update", "difference_update", "intersection_update",
                 "symmetric_difference_update")


def _copy_hooks() -> dict[str, Any]:
    # No instance ``__dict__`` (``__slots__ = ()``): nothing can be attached to a sealed container,
    # and ``__class__`` cannot be swapped back to the plain base type.
    # Copies and pickles come back as the plain base type: rebuilding a sealed container item by
    # item would go through the refusing mutators.
    return {"__slots__": (),
            "__copy__": lambda self: type(self).__mro__[1](self),
            "__deepcopy__": lambda self, memo: _plain(self, memo),
            "__reduce_ex__": lambda self, protocol: (type(self).__mro__[1], (type(self).__mro__[1](self),))}


SealedDict = type("SealedDict", (dict,), {**_refusing(dict, _DICT_MUTATORS), **_copy_hooks()})
SealedList = type("SealedList", (list,), {**_refusing(list, _LIST_MUTATORS), **_copy_hooks()})
SealedSet = type("SealedSet", (set,), {**_refusing(set, _SET_MUTATORS), **_copy_hooks()})


def _build(sealed: type, contents: Any) -> Any:
    base = sealed.__mro__[1]
    obj = base.__new__(sealed)
    base.__init__(obj, contents)  # the sealed type's own __init__ refuses while frozen
    return obj


def seal(value: Any) -> Any:
    """A detached deep copy of *value* whose dicts, lists and sets refuse changes while frozen.
    Anything else (strings, numbers, callables, entries) is kept as is.

    The containers' own methods refuse; calling a base type's method unbound on one
    (``dict.update(sealed, ...)``) is a deliberate bypass of the type, like ``object.__setattr__``
    on a frozen dataclass, and is not defended against."""
    if isinstance(value, dict):
        return _build(SealedDict, {k: seal(v) for k, v in value.items()})
    if isinstance(value, list):
        return _build(SealedList, [seal(v) for v in value])
    if isinstance(value, set):
        return _build(SealedSet, value)
    return value
