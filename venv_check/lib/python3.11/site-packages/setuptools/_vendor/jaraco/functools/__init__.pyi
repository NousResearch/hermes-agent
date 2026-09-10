from collections.abc import Callable, Hashable, Iterator
from functools import partial
from operator import methodcaller
import sys
from typing import (
    Any,
    Generic,
    Protocol,
    TypeVar,
    overload,
)

if sys.version_info >= (3, 10):
    from typing import Concatenate, ParamSpec
else:
    from typing_extensions import Concatenate, ParamSpec

_P = ParamSpec('_P')
_R = TypeVar('_R')
_T = TypeVar('_T')
_R1 = TypeVar('_R1')
_R2 = TypeVar('_R2')
_V = TypeVar('_V')
_S = TypeVar('_S')
_R_co = TypeVar('_R_co', covariant=True)

class _OnceCallable(Protocol[_P, _R]):
    saved_result: _R
    reset: Callable[[], None]
    def __call__(self, *args: _P.args, **kwargs: _P.kwargs) -> _R: ...

class _ProxyMethodCacheWrapper(Protocol[_R_co]):
    cache_clear: Callable[[], None]
    def __call__(self, *args: Hashable, **kwargs: Hashable) -> _R_co: ...

class _MethodCacheWrapper(Protocol[_R_co]):
    def cache_clear(self) -> None: ...
    def __call__(self, *args: Hashable, **kwargs: Hashable) -> _R_co: ...

# `compose()` overloads below will cover most use cases.

@overload
def compose(
    __func1: Callable[[_R], _T],
    __func2: Callable[_P, _R],
    /,
) -> Callable[_P, _T]: ...
@overload
def compose(
    __func1: Callable[[_R], _T],
    __func2: Callable[[_R1], _R],
    __func3: Callable[_P, _R1],
    /,
) -> Callable[_P, _T]: ...
@overload
def compose(
    __func1: Callable[[_R], _T],
    __func2: Callable[[_R2], _R],
    __func3: Callable[[_R1], _R2],
    __func4: Callable[_P, _R1],
    /,
) -> Callable[_P, _T]: ...
def once(func: Callable[_P, _R]) -> _OnceCallable[_P, _R]: ...
def method_cache(
    method: Callable[..., _R],
    cache_wrapper: Callable[[Callable[..., _R]], _MethodCacheWrapper[_R]] = ...,
) -> _MethodCacheWrapper[_R] | _ProxyMethodCacheWrapper[_R]: ...
def apply(
    transform: Callable[[_R], _T]
) -> Callable[[Callable[_P, _R]], Callable[_P, _T]]: ...
def result_invoke(
    action: Callable[[_R], Any]
) -> Callable[[Callable[_P, _R]], Callable[_P, _R]]: ...
def invoke(
    f: Callable[_P, _R], /, *args: _P.args, **kwargs: _P.kwargs
) -> Callable[_P, _R]: ...

class Throttler(Generic[_R]):
    last_called: float
    func: Callable[..., _R]
    max_rate: float
    def __init__(
        self, func: Callable[..., _R] | Throttler[_R], max_rate: float = ...
    ) -> None: ...
    def reset(self) -> None: ...
    def __call__(self, *args: Any, **kwargs: Any) -> _R: ...
    def __get__(self, obj: Any, owner: type[Any] | None = ...) -> Callable[..., _R]: ...

def first_invoke(
    func1: Callable[..., Any], func2: Callable[_P, _R]
) -> Callable[_P, _R]: ...

method_caller: Callable[..., methodcaller]

def retry_call(
    func: Callable[..., _R],
    cleanup: Callable[..., None] = ...,
    retries: int | float = ...,
    trap: type[BaseException] | tuple[type[BaseException], ...] = ...,
) -> _R: ...
def retry(
    cleanup: Callable[..., None] = ...,
    retries: int | float = ...,
    trap: type[BaseException] | tuple[type[BaseException], ...] = ...,
) -> Callable[[Callable[..., _R]], Callable[..., _R]]: ...
def print_yielded(func: Callable[_P, Iterator[Any]]) -> Callable[_P, None]: ...
def pass_none(
    func: Callable[Concatenate[_T, _P], _R]
) -> Callable[Concatenate[_T, _P], _R]: ...
def assign_params(
    func: Callable[..., _R], namespace: dict[str, Any]
) -> partial[_R]: ...
def save_method_args(
    method: Callable[Concatenate[_S, _P], _R]
) -> Callable[Concatenate[_S, _P], _R]: ...
def except_(
    *exceptions: type[BaseException], replace: Any = ..., use: Any = ...
) -> Callable[[Callable[_P, Any]], Callable[_P, Any]]: ...
def identity(x: _T) -> _T: ...
def bypass_when(
    check: _V, *, _op: Callable[[_V], Any] = ...
) -> Callable[[Callable[[_T], _R]], Callable[[_T], _T | _R]]: ...
def bypass_unless(
    check: Any,
) -> Callable[[Callable[[_T], _R]], Callable[[_T], _T | _R]]: ...
