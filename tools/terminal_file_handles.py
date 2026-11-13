"""Bounded, process-local post-turn transports; never sandbox discovery or file bytes.

A fixed 15-minute lease (not extended by reads), at most 128 qualified task keys.
Only per-turn cleanup retains a handle; session cleanup removes it, even with no
active environment. Callers hold terminal_tool._env_lock for all operations.
"""
from collections import OrderedDict
import time

_MAX_HANDLES = 128
_HANDLE_TTL = 15 * 60
_handles = OrderedDict()


def discard(key):
    _handles.pop(key, None)


def retain(key, env):
    _prune()
    discard(key)
    _handles[key] = (time.monotonic() + _HANDLE_TTL, env)
    while len(_handles) > _MAX_HANDLES:
        _handles.popitem(last=False)


def _prune():
    now = time.monotonic()
    for key, (expires, _env) in list(_handles.items()):
        if now >= expires:
            discard(key)


def resolve(key):
    from tools import terminal_tool
    _prune()
    active = terminal_tool._active_environments.get(key)
    if active is not None:
        discard(key)
        return active
    entry = _handles.get(key)
    return entry[1] if entry else None
