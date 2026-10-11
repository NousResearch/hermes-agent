"""Optional execution-environment observations, without importing Relay in tools."""

from __future__ import annotations

import contextvars
import logging
from contextlib import contextmanager
from typing import Any

logger = logging.getLogger(__name__)
_OBSERVER: contextvars.ContextVar[Any] = contextvars.ContextVar(
    "hermes_environment_observer", default=None
)


@contextmanager
def observe_environment(observer: Any):
    """Bind the owning Hermes session across copied tool callback contexts."""
    token = _OBSERVER.set(observer)
    try:
        yield
    finally:
        _OBSERVER.reset(token)


def capture_observer(environment: Any = None) -> Any:
    """Capture ownership before asynchronous cleanup loses the tool context."""
    return getattr(environment, "_relay_lifecycle_observer", None)


def environment_provisioning(environment: Any) -> Any:
    """Bind one physical owner before the backend starts provisioning."""
    owner = getattr(environment, "_relay_lifecycle_observer", None)
    if owner is not None:
        return owner
    observer = _OBSERVER.get()
    if observer is None:
        return None
    try:
        owner = observer.provision(environment)
        if owner is not None:
            environment._relay_lifecycle_observer = owner
        return owner
    except Exception as exc:
        logger.debug("Environment provisioning observation failed (%s)", type(exc).__name__, exc_info=True)
        return None


def prepare_cleanup(environment: Any) -> Any:
    """Reserve the captured owner's lifetime before starting a cleanup worker."""
    owner = capture_observer(environment)
    if owner is None:
        return None
    try:
        return owner.prepare_cleanup() or owner
    except Exception as exc:
        logger.debug("Environment cleanup observation failed (%s)", type(exc).__name__, exc_info=True)
        return owner


def finish_observation(owner: Any, *, reason: str) -> None:
    """End observation without changing or masking the backend's result."""
    if owner is None:
        return
    try:
        owner.finish(reason=reason)
    except Exception as exc:
        logger.debug("Environment observation finish failed (%s)", type(exc).__name__, exc_info=True)


def environment_acquired(environment: Any) -> Any:
    """Observe a resolved cached handle; preserve the caller's return value."""
    observer = _OBSERVER.get()
    if observer is not None:
        try:
            observer.acquire(environment)
        except Exception as exc:
            logger.debug("Environment observation failed (%s)", type(exc).__name__, exc_info=True)
    return environment


def environment_released(environment: Any, *, reason: str = "release") -> None:
    """Close logical associations, not the backing container's lifetime."""
    for lease in tuple(getattr(environment, "_relay_environment_leases", ())):
        try:
            lease.close(reason)
        except Exception as exc:
            logger.debug("Environment release observation failed (%s)", type(exc).__name__, exc_info=True)


def record_transition(
    environment: Any, action: str, outcome: str, container_id: str | None,
    *, state_after: str | None = None, evidence: str = "operation_result",
    previous_container_id: str | None = None, error_class: str | None = None,
    observer: Any = None,
) -> None:
    """Forward checked backend facts; observation failure never fails a tool."""
    observer = observer or capture_observer(environment)
    if observer is None:
        return
    try:
        observer.transition(
            environment, action=action, outcome=outcome, container_id=container_id,
            state_after=state_after, evidence=evidence,
            previous_container_id=previous_container_id, error_class=error_class,
        )
    except Exception as exc:
        logger.debug("Environment transition observation failed (%s)", type(exc).__name__, exc_info=True)
