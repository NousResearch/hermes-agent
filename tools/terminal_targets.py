"""Explicit, profile-scoped terminal routes supplied by trusted host providers.

Providers authorize both caller identities and return an existing lease registered
under a synthetic target identity, never a parent/task alias. This is cooperative
local transport routing, not a sandbox or an approval exemption.
"""
from __future__ import annotations

import threading
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path

from hermes_constants import get_hermes_home
from hermes_cli.session_execution import SessionExecutionError, SessionExecutionLease, TargetSelection

_lock = threading.RLock()
_resolvers: dict[tuple[str, str], tuple] = {}


def _home(home=None):
    return str(Path(home if home is not None else get_hermes_home()).resolve())


def _name(name):
    if not isinstance(name, str) or not name or name != name.strip() or "\0" in name:
        raise ValueError("terminal target must be a nonempty exact name")
    return name


def register_terminal_target_resolver(name, resolver, *, hermes_home=None, selector=None):
    """Register a named provider; return an idempotent generation-safe disposer.

    ``resolver(command=, session_id=, task_id=)`` must authorize the caller
    before returning its target-only SessionExecutionLease. None/failure refuses
    the call; it never selects the ordinary terminal. Re-registration replaces
    the same profile/name only. Registration itself must not provision a target.
    Every provider must supply a non-provisioning ``selector`` with the same
    keyword arguments, returning a non-provisioning TargetSelection. Its check
    binds owner/kind/incarnation and authority; realization runs after consent.
    """
    key = (_home(hermes_home), _name(name))
    if not callable(resolver):
        raise TypeError("terminal target resolver must be callable")
    if selector is not None and not callable(selector):
        raise TypeError("terminal target selector must be callable")
    entry = (object(), resolver, selector)
    with _lock:
        _resolvers[key] = entry

    def dispose():
        with _lock:
            if _resolvers.get(key) is entry:
                del _resolvers[key]
    return dispose


def resolve_terminal_target(name, *, command, session_id=None, task_id=None):
    """Resolve only the explicit name in this profile; never fall back."""
    return select_terminal_target(name, command=command, session_id=session_id, task_id=task_id).realize()


def select_terminal_target(name, *, command, session_id=None, task_id=None):
    """Select without starting compute; refuse providers without a selector."""
    home = _home()
    try:
        key = (home, _name(name))
        with _lock:
            entry = _resolvers.get(key)
        if entry is None:
            raise SessionExecutionError("Unknown terminal target")
        if entry[2] is None:
            raise SessionExecutionError("terminal provider requires an explicit non-provisioning selector")
        selected = entry[2](command=command, session_id=session_id, task_id=task_id)
        if not isinstance(selected, TargetSelection):
            raise SessionExecutionError("invalid terminal target selection")

        def check():
            with _lock:
                current = _resolvers.get(key)
            if _home() != home or current is not entry:
                raise SessionExecutionError("terminal target provider changed during operation")
            selected.check()

        def realize():
            check()
            lease = selected.realize(before_start=check)
            if not isinstance(lease, SessionExecutionLease) or lease.home != home:
                raise SessionExecutionError("terminal target unavailable for this caller")
            lease.check()
            check()
            return lease

        check()
        return TargetSelection(realize, check)
    except SessionExecutionError:
        raise
    except Exception as exc:
        raise SessionExecutionError("terminal target selection failed") from exc


def access_epoch(lease):
    """Read operation authority separately from the reusable resource lease."""
    lease.check()
    reader = lease.context.terminal_access_epoch
    if reader is None:
        return None
    value = reader()
    if type(value) is not int or value < 0:
        raise SessionExecutionError("invalid terminal target control epoch")
    return value


_operation = ContextVar("terminal_target_operation", default=None)


@contextmanager
def terminal_operation(lease, epoch, check=None):
    """Per-call authority; never attach a pending approval to a cached shell."""
    outer = _operation.get()
    if outer is not None and outer[0] is lease:
        check_terminal_operation(lease)
        yield
        return
    token = _operation.set((lease, epoch, check))
    try:
        check_terminal_operation(lease)
        yield
    finally:
        _operation.reset(token)


def check_terminal_operation(lease):
    operation = _operation.get()
    if operation is None or operation[0] is not lease:
        return
    if operation[2] is not None:
        operation[2]()
    if access_epoch(lease) != operation[1]:
        raise SessionExecutionError("Target control changed during operation; request a fresh operation")
