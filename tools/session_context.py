"""Asyncio-task-local session context.

Replaces the process-wide ``os.environ["HERMES_SESSION_*"]`` pattern where
cross-task isolation matters (e.g. filtering a tool result by the calling
user). ``ContextVar`` values follow each asyncio task independently, so
concurrent message handlers can't clobber each other's session context.

Callers should treat env vars as a fallback for non-async contexts (CLI,
scripts, legacy tests). When running inside the gateway, read from the
ContextVars here first.
"""

from __future__ import annotations

from contextvars import ContextVar
from typing import Optional

session_platform: ContextVar[Optional[str]] = ContextVar("hermes_session_platform", default=None)
session_chat_id: ContextVar[Optional[str]] = ContextVar("hermes_session_chat_id", default=None)
session_chat_name: ContextVar[Optional[str]] = ContextVar("hermes_session_chat_name", default=None)
session_thread_id: ContextVar[Optional[str]] = ContextVar("hermes_session_thread_id", default=None)


def set_session(
    *,
    platform: Optional[str],
    chat_id: Optional[str],
    chat_name: Optional[str] = None,
    thread_id: Optional[str] = None,
) -> None:
    session_platform.set(platform)
    session_chat_id.set(chat_id)
    session_chat_name.set(chat_name)
    session_thread_id.set(thread_id)


def clear_session() -> None:
    session_platform.set(None)
    session_chat_id.set(None)
    session_chat_name.set(None)
    session_thread_id.set(None)


def get_chat_id() -> Optional[str]:
    return session_chat_id.get()


def get_platform() -> Optional[str]:
    return session_platform.get()


def get_chat_name() -> Optional[str]:
    return session_chat_name.get()


def get_thread_id() -> Optional[str]:
    return session_thread_id.get()
