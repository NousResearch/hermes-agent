from __future__ import annotations

from collections.abc import Iterable
from typing import TypeVar

_IterableT = TypeVar("_IterableT", bound="Iterable[str]")


def consolidate_linker_args(args: _IterableT) -> _IterableT | str:
    """
    Ensure the return value is a string for backward compatibility.

    Retain until at least 2025-04-31. See pypa/distutils#246
    """

    if not all(arg.startswith('-Wl,') for arg in args):
        return args
    return '-Wl,' + ','.join(arg.removeprefix('-Wl,') for arg in args)
