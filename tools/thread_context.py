"""Propagate agent-turn context into worker threads that dispatch Hermes tools.

A bare ``threading.Thread`` / ``ThreadPoolExecutor`` worker starts with an
empty ``contextvars.Context`` and no thread-local approval callback.
Tool dispatch inside such a thread therefore silently loses:

  * the approval *session/platform* ContextVars (``tools.approval`` /
    ``gateway.session_context``) — so gateway sessions fall into
    ``check_dangerous_command``'s non-interactive auto-approve branch and
    dangerous commands run without prompting (#33057, #30882);
  * the thread-local CLI approval callback (``tools.terminal_tool``) —
    so ``prompt_dangerous_approval`` cannot reach the user
    (GHSA-qg5c-hvr5-hjgr, #15216).

This helper factors out that capture/install/clear lifecycle so the several
places that fan tool dispatch onto worker threads (``agent.tool_executor`` and
the ``execute_code`` RPC threads) share one audited implementation instead of
divergent copies.

Usage — call :func:`propagate_context_to_thread` **on the parent thread**
(it snapshots the parent's ContextVars and callback at call time) and use the
returned callable as the worker's target::

    t = threading.Thread(target=propagate_context_to_thread(loop_fn), args=(...))
    # or
    executor.submit(propagate_context_to_thread(worker_fn), *args)

The approval callback is installed for the worker's lifetime and **always
cleared on exit**, so a recycled thread never holds a stale reference to a
disposed CLI instance.
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
    every parallel/timeout worker, so the tool believes nobody can answer.

    Excludes the sudo-password callback: Hermes removed the SUDO_PASSWORD-piping mechanism
    (Standing Exclusion 1/2), so there is no ``set_sudo_password_callback`` to propagate.
    """
    from agent.vault_backends import unlock as vault_unlock
    from tools import terminal_tool as tt

    return ((tt._get_approval_callback, tt.set_approval_callback),
            (vault_unlock.get_unlock_prompt_callback, vault_unlock.set_unlock_prompt_callback),
            (vault_unlock.get_save_login_prompt_callback, vault_unlock.set_save_login_prompt_callback),
            (vault_unlock.get_code_prompt_callback, vault_unlock.set_code_prompt_callback))


def propagate_context_to_thread(target: Callable) -> Callable:
    """Wrap *target* to run with the *current* thread's ContextVars and per-thread prompt callbacks
    (approval, password-manager unlock).

    Fail-closed: if callback installation raises they stay ``None`` — dangerous commands are then
    denied by ``prompt_dangerous_approval`` and the gateway approval queue blocks.
    """
    ctx = contextvars.copy_context()
    # (setter, parent callback) pairs; None when the callback API could not be captured.
    installs = None
    try:
        installs = tuple((setter, getter()) for getter, setter in _callback_api())
    except Exception:
        logger.debug("Could not capture parent approval/vault callbacks", exc_info=True)

    def _runner(*args, **kwargs):
        def _inner():
            if installs is None:
                return target(*args, **kwargs)
            try:
                for setter, cb in installs:
                    if cb is not None:
                        setter(cb)
            except Exception:
                logger.debug("Failed to install propagated approval/vault callbacks; "
                             "dangerous-command approval will fail closed", exc_info=True)
            try:
                return target(*args, **kwargs)
            finally:
                try:
                    for setter, _cb in installs:
                        setter(None)
                except Exception:
                    logger.debug("Failed to clear propagated approval/vault callbacks",
                                 exc_info=True)

        return ctx.run(_inner)

    return _runner
