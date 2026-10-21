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
from typing import Callable

logger = logging.getLogger(__name__)


def _callback_api():
    """(getter, setter) pairs for every thread-local prompt callback a tool may need mid-dispatch
    (lazy: terminal_tool imports tools.approval at load, so a top-level import risks a cycle).
    Add a new per-thread prompt here — a callback missing from this table is silently absent on
    every parallel/timeout worker, so the tool believes nobody can answer."""
    from agent.vault_backends import unlock as vault_unlock
    from tools import terminal_tool as tt

    return ((tt._get_approval_callback, tt.set_approval_callback),
            (tt._get_sudo_password_callback, tt.set_sudo_password_callback),
            (vault_unlock.get_unlock_prompt_callback, vault_unlock.set_unlock_prompt_callback),
            (vault_unlock.get_save_login_prompt_callback, vault_unlock.set_save_login_prompt_callback),
            (vault_unlock.get_code_prompt_callback, vault_unlock.set_code_prompt_callback))


def propagate_context_to_thread(target: Callable) -> Callable:
    """Wrap *target* to run with the *current* thread's ContextVars and per-thread prompt callbacks
    (approval, sudo, password-manager unlock).

    Fail-closed: if callback installation raises they stay ``None`` — dangerous commands are then
    denied by ``prompt_dangerous_approval`` and the gateway approval queue blocks.
    """
    ctx = contextvars.copy_context()
    # (setter, parent callback) pairs; None when the callback API could not be captured.
    installs = None
    try:
        installs = tuple((setter, getter()) for getter, setter in _callback_api())
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
