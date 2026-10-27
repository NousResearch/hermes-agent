"""Privacy-bounded conversation anchors for Kanban cards.

These fields are evidence, not delivery authorization. Notification routing stays
in the canonical subscription matcher; arbitrary tool or message metadata is not
copied into the board or future worker prompts.
"""
from contextlib import contextmanager
from contextvars import ContextVar
from urllib.parse import urlencode


_executor_origin = ContextVar("kanban_executor_origin", default=None)


@contextmanager
def bind_conversation_origin(origin):
    token = _executor_origin.set(origin)
    try:
        yield
    finally:
        _executor_origin.reset(token)


def executor_origin_session_id() -> str | None:
    """Only the executor binding may nominate a source session for tool dispatch."""
    from hermes_cli.profiles import current_profile_name

    origin = _executor_origin.get()
    if origin and origin.get("profile") == current_profile_name("default"):
        return origin.get("session_id")
    return None


def conversation_origin(session_id: str | None) -> dict | None:
    if not session_id:
        return None
    from gateway.session_context import get_session_env as env
    from hermes_cli.profiles import current_profile_name

    profile = current_profile_name("default")
    trusted = _executor_origin.get()
    if trusted and trusted.get("session_id") == session_id and trusted.get("profile") == profile:
        return dict(trusted)
    origin = {"profile": profile, "session_id": session_id,
              "session_link": "/chat?" + urlencode({"resume": session_id, "profile": profile})}
    if env("HERMES_SESSION_ID", "") != session_id:
        return origin
    for field, name in (
        ("platform", "PLATFORM"), ("chat_id", "CHAT_ID"), ("thread_id", "THREAD_ID"),
        ("message_id", "MESSAGE_ID"), ("scope_id", "SCOPE_ID"), ("parent_chat_id", "PARENT_CHAT_ID"),
        ("transport_profile", "TRANSPORT_PROFILE"),
    ):
        if value := env("HERMES_SESSION_" + name, ""):
            origin[field] = str(value)
    return origin
