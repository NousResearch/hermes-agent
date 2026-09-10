from __future__ import annotations

import sys
from collections.abc import Callable, Generator
from contextlib import contextmanager
from functools import update_wrapper
from threading import Lock
from typing import ContextManager, TypeVar, overload

if sys.version_info >= (3, 10):
    from typing import ParamSpec
else:
    from typing_extensions import ParamSpec

P = ParamSpec("P")
T = TypeVar("T")

type_checks_suppressed = 0
type_checks_suppress_lock = Lock()


@overload
def suppress_type_checks(func: Callable[P, T]) -> Callable[P, T]: ...


@overload
def suppress_type_checks() -> ContextManager[None]: ...


def suppress_type_checks(
    func: Callable[P, T] | None = None,
) -> Callable[P, T] | ContextManager[None]:
    """
    Temporarily suppress all type checking.

    This function has two operating modes, based on how it's used:

    #. as a context manager (``with suppress_type_checks(): ...``)
    #. as a decorator (``@suppress_type_checks``)

    When used as a context manager, :func:`check_type` and any automatically
    instrumented functions skip the actual type checking. These context managers can be
    nested.

    When used as a decorator, all type checking is suppressed while the function is
    running.

    Type checking will resume once no more context managers are active and no decorated
    functions are running.

    Both operating modes are thread-safe.

    """

    def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
        global type_checks_suppressed

        with type_checks_suppress_lock:
            type_checks_suppressed += 1

        assert func is not None
        try:
            return func(*args, **kwargs)
        finally:
            with type_checks_suppress_lock:
                type_checks_suppressed -= 1

    def cm() -> Generator[None, None, None]:
        global type_checks_suppressed

        with type_checks_suppress_lock:
            type_checks_suppressed += 1

        try:
            yield
        finally:
            with type_checks_suppress_lock:
                type_checks_suppressed -= 1

    if func is None:
        # Context manager mode
        return contextmanager(cm)()
    else:
        # Decorator mode
        update_wrapper(wrapper, func)
        return wrapper
