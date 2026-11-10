"""Which background processes run inside a terminal environment, for its teardown.

A live process belongs to the environment object it was spawned through (``env_ref``).
A handoff changes its ``task_id`` to the new owner's cache key without moving the process.
Only checkpoint-recovered processes without an ``env_ref`` fall back to their saved ``task_id``.
``owner_task_id`` is the raw spawning id, not an environment identity.
"""

from typing import Any, Optional


def _in_env(session, key: str, env: Optional[Any]) -> bool:
    if session.env_ref is not None:
        return env is not None and session.env_ref is env
    return session.task_id == key


def has_running_in_env(registry, key: str, env: Optional[Any] = None) -> bool:
    """Whether a still-running process lives in the environment cached under ``key`` (or is
    polled through ``env``). Tearing that environment down would kill or orphan it."""
    return registry._any_running(lambda s: _in_env(s, key, env))


def settle_env_closed(registry, env: Any) -> int:
    """Settle every running process polled through ``env`` as lost, before ``env`` is closed.

    A sandbox process is observed only through its spawning environment object; once that
    object is closed the poller can no longer read its output or exit code, and some backends
    report the failure as output rather than raising, so the poller would retry forever.
    Marking the session exited stops the poller at its next check and publishes the result.
    A process that exits between the snapshot and its settlement keeps its own result.
    Returns the number of sessions settled as lost."""
    if env is None:
        return 0
    with registry._lock:
        sessions = [s for s in registry._running.values() if s.env_ref is env and not s.exited]
    return sum(registry._finish_exited(s, -1, reason="lost", source="env_closed") for s in sessions)
