"""Profile-scoped, trusted target selection for computer_use only.

Providers own session/alias authorization and register leases under target-only
identities. Absence of a provider permits legacy routing; provider failure never
does. Registration advertises capability without allocating a desktop.
"""
from __future__ import annotations

import threading
from pathlib import Path

from hermes_constants import get_hermes_home
from hermes_cli.session_execution import SessionExecutionError, SessionExecutionLease, TargetSelection

UNBOUND = object()
_lock = threading.RLock()
_resolvers: dict[str, dict[str, tuple]] = {}


def _home(home=None):
    return str(Path(home if home is not None else get_hermes_home()).resolve())


def register_target_resolver(name, resolver, *, hermes_home=None, selector=None):
    """Install a trusted resolver; return an idempotent, generation-safe disposer.

    A second distinct provider is ambiguous and denies resolution rather than
    choosing by registration order. Re-registering the same name replaces it.
    ``selector(session_id=, task_id=)`` returns a non-provisioning TargetSelection;
    every provider must supply it for execution. Missing selectors refuse before
    invoking the resolver; ordinary no-provider execution remains supported.
    """
    if not isinstance(name, str) or not name or name != name.strip() or "\0" in name:
        raise ValueError("resolver name must be a nonempty exact string")
    if not callable(resolver):
        raise TypeError("resolver must be callable")
    if selector is not None and not callable(selector):
        raise TypeError("target selector must be callable")
    home, entry = _home(hermes_home), (object(), resolver, selector)
    with _lock:
        _resolvers.setdefault(home, {})[name] = entry

    def dispose():
        with _lock:
            providers = _resolvers.get(home, {})
            if providers.get(name) is entry:
                del providers[name]
                if not providers:
                    _resolvers.pop(home, None)
    return dispose


def has_target_resolver():
    """Configured capability only; never invoke a provider or start a target."""
    with _lock:
        return bool(_resolvers.get(_home()))


def resolve_target_context(session_id, task_id=None):
    """Return an authorized lease, or UNBOUND only when no provider is installed."""
    selection, targeted = select_target_context(session_id, task_id)
    return selection.realize() if targeted else UNBOUND


def select_target_context(session_id, task_id=None):
    """Bind before approval. Every provider MUST register an inert selector.

    Selectors return TargetSelection with captured authority and realization.
    No backend/cache lookup and no second resolution after approval.
    """
    from hermes_cli.session_execution import resolve_session_execution_context
    from tools.computer_use.session_context import read_access_epoch
    home = _home()
    with _lock:
        entries = tuple(_resolvers.get(home, {}).values())
    if len(entries) > 1:
        raise SessionExecutionError("ambiguous computer-use target providers")
    try:
        provider = entries[0] if entries else None
        if provider:
            if provider[2] is None:
                raise SessionExecutionError("computer-use provider requires an explicit non-provisioning selector")
            selected = provider[2](session_id=session_id, task_id=task_id)
            if not isinstance(selected, TargetSelection):
                raise SessionExecutionError("invalid computer-use target selection")
        else:
            lease = resolve_session_execution_context(session_id=session_id, task_id=task_id)
            epoch = read_access_epoch(lease)
            def check_lease():
                if not provider and resolve_session_execution_context(session_id=session_id, task_id=task_id) is not lease:
                    raise SessionExecutionError("computer-use execution source changed during approval")
                if read_access_epoch(lease) != epoch:
                    raise SessionExecutionError("computer-use control epoch changed during approval")
            selected = TargetSelection(lambda **kw: lease, check_lease)

        def check():
            with _lock:
                current = tuple(_resolvers.get(home, {}).values())
            if _home() != home or current != entries:
                raise SessionExecutionError("computer-use target provider changed during operation")
            selected.check()

        def realize():
            check()
            lease = selected.realize(before_start=check)
            if lease is not None or provider:
                if not isinstance(lease, SessionExecutionLease) or lease.home != home:
                    raise SessionExecutionError("computer-use target unavailable for this caller")
                lease.check()
            check()
            return lease

        check()
        return TargetSelection(realize, check), bool(provider)
    except SessionExecutionError:
        raise
    except Exception as exc:
        raise SessionExecutionError("computer-use target selection failed") from exc
