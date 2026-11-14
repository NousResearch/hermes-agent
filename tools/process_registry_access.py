"""Which caller may act on a ``process_manage`` session (poll/log/wait/kill/write/submit/close).

A session id is a routing handle, not a capability (SECURITY.md §2.6): another conversation that
learns a live id must not read, kill, or type into that process. The rule mirrors ``list_sessions``
and the retained-result loader so a process the caller can list is one it can act on.
"""

from __future__ import annotations

from contextlib import suppress


def caller_session_key() -> str:
    """The current gateway/CLI session key, or "" outside one (#29177)."""
    with suppress(Exception):
        from tools.approval_context import get_current_session_key
        return get_current_session_key(default="") or ""
    return ""


def session_action_allowed(session, task_id: str, session_key: str) -> bool:
    """Whether the caller may poll/log/wait/kill/write/submit/close ``session``.

    Ownership is the spawning task (``owner_task_id``, or the container
    ``task_id`` it collapsed to) or the gateway ``session_key`` — a
    session-scoped process started by a sibling task is still this
    conversation's to manage (#29177), including through a compression resume.
    A caller carrying no identity at all is trusted internal plumbing rather
    than a model call (model calls always pass task_id); a session carrying no
    identity (unowned/detached) has no owner to violate.
    """
    owner = session.owner_task_id or session.task_id
    if task_id and task_id in (owner, session.task_id):
        return True
    if session_key and session.session_key:
        if session_key == session.session_key:
            return True
        from tools.process_registry_results import _owns_result
        if _owns_result(session_key, session.session_key):
            return True
    return not (task_id or session_key) or not (owner or session.session_key)


def caller_may_act(session_id: str, task_id) -> bool:
    """False only for a session that exists and belongs to someone else; the caller then answers
    with the same ``not_found`` shape as a miss, so foreign ids are not enumerable."""
    from tools.process_registry import process_registry
    session = process_registry.get(session_id)
    return session is None or session_action_allowed(session, str(task_id or ""), caller_session_key())
