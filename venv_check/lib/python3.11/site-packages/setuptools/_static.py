from functools import wraps
from typing import TypeVar

import packaging.specifiers

from .warnings import SetuptoolsDeprecationWarning


class Static:
    """
    Wrapper for built-in object types that are allow setuptools to identify
    static core metadata (in opposition to ``Dynamic``, as defined :pep:`643`).

    The trick is to mark values with :class:`Static` when they come from
    ``pyproject.toml`` or ``setup.cfg``, so if any plugin overwrite the value
    with a built-in, setuptools will be able to recognise the change.

    We inherit from built-in classes, so that we don't need to change the existing
    code base to deal with the new types.
    We also should strive for immutability objects to avoid changes after the
    initial parsing.
    """

    _mutated_: bool = False  # TODO: Remove after deprecation warning is solved


def _prevent_modification(target: type, method: str, copying: str) -> None:
    """
    Because setuptools is very flexible we cannot fully prevent
    plugins and user customizations from modifying static values that were
    parsed from config files.
    But we can attempt to block "in-place" mutations and identify when they
    were done.
    """
    fn = getattr(target, method, None)
    if fn is None:
        return

    @wraps(fn)
    def _replacement(self: Static, *args, **kwargs):
        # TODO: After deprecation period raise NotImplementedError instead of warning
        #       which obviated the existence and checks of the `_mutated_` attribute.
        self._mutated_ = True
        SetuptoolsDeprecationWarning.emit(
            "Direct modification of value will be disallowed",
            f"""
            In an effort to implement PEP 643, direct/in-place changes of static values
            that come from configuration files are deprecated.
            If you need to modify this value, please first create a copy with {copying}
            and make sure conform to all relevant standards when overriding setuptools
            functionality (https://packaging.python.org/en/latest/specifications/).
            """,
            due_date=(2025, 10, 10),  # Initially introduced in 2024-09-06
        )
        return fn(self, *args, **kwargs)

    _replacement.__doc__ = ""  # otherwise doctest may fail.
    setattr(target, method, _replacement)


class Str(str, Static):
    pass


class Tuple(tuple, Static):
    pass


class List(list, Static):
    """
    :meta private:
    >>> x = List([1, 2, 3])
    >>> is_static(x)
    True
    >>> x += [0]  # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
    SetuptoolsDeprecationWarning: Direct modification ...
    >>> is_static(x)  # no longer static after modification
    False
    >>> y = list(x)
    >>> y.clear()
    >>> y
    []
    >>> y == x
    False
    >>> is_static(List(y))
    True
    """


# Make `List` immutable-ish
# (certain places of setuptools/distutils issue a warn if we use tuple instead of list)
for _method in (
    '__delitem__',
    '__iadd__',
    '__setitem__',
    'append',
    'clear',
    'extend',
    'insert',
    'remove',
    'reverse',
    'pop',
):
    _prevent_modification(List, _method, "`list(value)`")


class Dict(dict, Static):
    """
    :meta private:
    >>> x = Dict({'a': 1, 'b': 2})
    >>> is_static(x)
    True
    >>> x['c'] = 0  # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
    SetuptoolsDeprecationWarning: Direct modification ...
    >>> x._mutated_
    True
    >>> is_static(x)  # no longer static after modification
    False
    >>> y = dict(x)
    >>> y.popitem()
    ('b', 2)
    >>> y == x
    False
    >>> is_static(Dict(y))
    True
    """


# Make `Dict` immutable-ish (we cannot inherit from types.MappingProxyType):
for _method in (
    '__delitem__',
    '__ior__',
    '__setitem__',
    'clear',
    'pop',
    'popitem',
    'setdefault',
    'update',
):
    _prevent_modification(Dict, _method, "`dict(value)`")


class SpecifierSet(packaging.specifiers.SpecifierSet, Static):
    """Not exactly a built-in type but useful for ``requires-python``"""


T = TypeVar("T")


def noop(value: T) -> T:
    """
    >>> noop(42)
    42
    """
    return value


_CONVERSIONS = {str: Str, tuple: Tuple, list: List, dict: Dict}


def attempt_conversion(value: T) -> T:
    """
    >>> is_static(attempt_conversion("hello"))
    True
    >>> is_static(object())
    False
    """
    return _CONVERSIONS.get(type(value), noop)(value)  # type: ignore[call-overload]


def is_static(value: object) -> bool:
    """
    >>> is_static(a := Dict({'a': 1}))
    True
    >>> is_static(dict(a))
    False
    >>> is_static(b := List([1, 2, 3]))
    True
    >>> is_static(list(b))
    False
    """
    return isinstance(value, Static) and not value._mutated_


EMPTY_LIST = List()
EMPTY_DICT = Dict()
