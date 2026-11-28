"""Coalesce concurrent ``plugins.manage update`` requests for one plugin in one profile."""
from concurrent.futures import Future
from threading import Lock
from typing import Any, Callable

from hermes_constants import hermes_home_key

_lock = Lock()
_pending: dict[tuple[str, str], Future] = {}


def join_update(name: str, operation: Callable[[], Any]) -> Any:
    """Run *operation*, or wait for the one already running for this profile and plugin.

    A client that timed out and retried gets the running update's answer instead of starting a
    second fetch against the same staging tree. The first request owns the decision: a joiner
    never changes ``accept_capabilities``/``force``, so a refusal comes back unchanged and must
    be submitted again. A call made after the update finished runs afresh.
    """
    key = (hermes_home_key(), name)
    with _lock:
        future = _pending.get(key)
        owner = future is None
        if owner:
            future = _pending[key] = Future()
    if not owner:
        return future.result()
    try:
        result = operation()
        future.set_result(result)
        return result
    except BaseException as exc:
        future.set_exception(exc)
        raise
    finally:
        with _lock:
            del _pending[key]
