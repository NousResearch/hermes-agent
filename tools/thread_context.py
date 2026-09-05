"""Propagate agent-turn context into worker threads that dispatch Hermes tools.

A bare ``threading.Thread`` / ``ThreadPoolExecutor`` worker starts with an empty
``contextvars.Context`` and no thread-local approval/sudo callbacks, so tool dispatch inside it
silently loses the approval ContextVars (gateway sessions then auto-approve dangerous commands)
and the CLI callbacks (``prompt_dangerous_approval`` cannot reach the user, GHSA-qg5c-hvr5-hjgr).
Call :func:`propagate_context_to_thread` **on the parent thread** (it snapshots at call time) and
use the result as the worker target; callbacks are installed for the worker's lifetime and
always cleared on exit.
"""

from __future__ import annotations

import contextvars
import logging
from typing import Any, Awaitable, Callable

logger = logging.getLogger(__name__)


def _callback_api():
    """Resolve the terminal_tool callback getters/setters (lazy: terminal_tool imports
    tools.approval at load, so a top-level import risks a cycle for tools.approval callers)."""
    from tools import terminal_tool as tt

    return (tt._get_approval_callback, tt._get_sudo_password_callback,
            tt.set_approval_callback, tt.set_sudo_password_callback)


def propagate_context_to_thread(target: Callable) -> Callable:
    """Wrap *target* to run with the *current* thread's ContextVars and approval/sudo callbacks.

    Fail-closed: if callback installation raises they stay ``None`` — dangerous commands are then
    denied by ``prompt_dangerous_approval`` and the gateway approval queue blocks.
    """
    ctx = contextvars.copy_context()
    # (setter, parent callback) pairs; None when the callback API could not be captured.
    installs = None
    try:
        get_approval, get_sudo, set_approval, set_sudo = _callback_api()
        installs = ((set_approval, get_approval()), (set_sudo, get_sudo()))
    except Exception:
        logger.debug("Could not capture parent approval/sudo callbacks", exc_info=True)

    def _runner(*args, **kwargs):
        def _inner():
            if installs is None:
                return target(*args, **kwargs)
            try:
                for setter, cb in installs:
                    if cb is not None:
                        setter(cb)
            except Exception:
                logger.debug("Failed to install propagated approval/sudo callbacks; "
                             "dangerous-command approval will fail closed", exc_info=True)
            try:
                return target(*args, **kwargs)
            finally:
                try:
                    for setter, _cb in installs:
                        setter(None)
                except Exception:
                    logger.debug("Failed to clear propagated approval/sudo callbacks",
                                 exc_info=True)

        return ctx.run(_inner)

    return _runner


def propagate_callbacks_to_async_thread(
    awaitable: Awaitable[Any],
) -> Awaitable[Any]:
    """Keep parent approval/sudo callbacks installed while *awaitable* runs.

    Call this on the parent thread immediately before submitting the returned
    awaitable with :func:`asyncio.run_coroutine_threadsafe`. That asyncio API
    carries the caller's ContextVars onto the destination loop; this helper
    supplies the thread-local half of the same security boundary and clears it
    when the turn finishes.
    """

    parent_approval_cb = parent_sudo_cb = None
    setters = None
    try:
        get_approval, get_sudo, set_approval, set_sudo = _callback_api()
        parent_approval_cb = get_approval()
        parent_sudo_cb = get_sudo()
        setters = (set_approval, set_sudo)
    except Exception:
        logger.debug("Could not capture parent approval/sudo callbacks", exc_info=True)

    async def _runner():
        if setters is not None:
            set_approval, set_sudo = setters
            try:
                set_approval(parent_approval_cb)
                set_sudo(parent_sudo_cb)
            except Exception:
                logger.debug(
                    "Failed to install propagated approval/sudo callbacks; "
                    "dangerous-command approval will fail closed",
                    exc_info=True,
                )
        try:
            return await awaitable
        finally:
            if setters is not None:
                set_approval, set_sudo = setters
                try:
                    set_approval(None)
                    set_sudo(None)
                except Exception:
                    logger.debug(
                        "Failed to clear propagated approval/sudo callbacks",
                        exc_info=True,
                    )
    return _runner()
