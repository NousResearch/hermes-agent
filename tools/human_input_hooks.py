"""Plugin observers for every point where the agent blocks waiting on a human (#132333).

``on_human_input_request`` fires right before a prompt is shown (sudo password, clarify question,
dangerous-command approval) and ``on_human_input_resolved`` fires exactly once when the wait ends,
with the same ``request_id`` and an ``outcome``. Observers only: return values are ignored and a
failing plugin never affects the prompt. Payloads never carry what the human typed (password,
answers), and ``prompt`` is force-redacted so credentials in a command never reach a plugin.
"""

from __future__ import annotations

import contextlib
import logging
import os
import uuid
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class HumanInputRequest:
    """Handle yielded by :func:`human_input_request`; set ``outcome`` before the block exits."""

    request_id: str
    outcome: str = ""


def _redacted(text: str) -> str:
    try:
        from agent.redact import redact_sensitive_text
        return redact_sensitive_text(str(text or ""), force=True)
    except Exception:
        logger.debug("human-input hook prompt redaction failed", exc_info=True)
        return ""  # never fall back to the raw text


def _identity() -> dict:
    try:
        from gateway.session_context import get_session_env
        from tools import approval_context
        session_id = approval_context._approval_session_id.get() or get_session_env("HERMES_SESSION_ID")
        platform = (os.getenv("HERMES_PLATFORM") or get_session_env("HERMES_SESSION_PLATFORM")
                    or get_session_env("HERMES_SESSION_SOURCE") or "cli")
        return {"session_id": session_id, "session_key": approval_context.get_current_session_key(default=""),
                "platform": platform}
    except Exception:  # a broken lookup must not take the human prompt down with it
        logger.debug("human-input hook identity lookup failed", exc_info=True)
        return {"session_id": "", "session_key": "", "platform": ""}


def _fire(hook_name: str, payload: dict) -> None:
    try:
        from hermes_cli.lifecycle import invoke_hook
        invoke_hook(hook_name, **payload)
    except Exception:  # observability must never break a human prompt
        logger.debug("%s dispatch failed", hook_name, exc_info=True)


@contextlib.contextmanager
def human_input_request(kind: str, *, prompt: str = "", session_key: str | None = None, **extra):
    """Fire the request hook, run the block (the human wait), then fire the resolved hook.
    An exception escaping the block, or a block that never sets ``outcome``, resolves as ``error``."""
    payload = {"kind": kind, "request_id": uuid.uuid4().hex, **_identity(), "prompt": _redacted(prompt), **extra}
    if session_key is not None:
        payload["session_key"] = session_key
    request = HumanInputRequest(payload["request_id"])
    _fire("on_human_input_request", payload)
    try:
        yield request
    finally:
        _fire("on_human_input_resolved", {**payload, "outcome": str(request.outcome or "error")})
