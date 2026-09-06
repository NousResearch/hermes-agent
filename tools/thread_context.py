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
    """Resolve the terminal_tool callback getter/setter.

    Imported lazily: ``tools.terminal_tool`` imports ``tools.approval`` at
    module load, so a top-level import here would risk an import cycle for
    callers that live in ``tools.approval``.
    """
    from tools.terminal_tool import (
        _get_approval_callback,
        set_approval_callback,
    )
    return (
        _get_approval_callback,
        set_approval_callback,
    )


def propagate_context_to_thread(target: Callable) -> Callable:
    """Wrap *target* for execution on a worker thread with the *current*
    thread's ContextVars and approval callback propagated.

    Call this on the parent thread; pass the returned callable as the
    thread/executor target.  The returned callable forwards its positional
    and keyword arguments to *target* and returns its result.

    Fail-closed: if callback installation raises, the callback is left
    unset (``None``).  That is the safe outcome — ``prompt_dangerous_approval``
    denies dangerous commands when no callback is registered in an interactive
    context, and the gateway approval queue blocks when its notify callback is
    absent.
    """
    ctx = contextvars.copy_context()
    parent_approval_cb = None
    setters = None
    try:
        get_approval, set_approval = _callback_api()
        parent_approval_cb = get_approval()
        setters = (set_approval,)
    except Exception:
        logger.debug("Could not capture parent approval callback", exc_info=True)

    def _runner(*args, **kwargs):
        def _inner():
            if setters is not None:
                (set_approval,) = setters
                try:
                    if parent_approval_cb is not None:
                        set_approval(parent_approval_cb)
                except Exception:
                    logger.debug(
                        "Failed to install propagated approval callback; "
                        "dangerous-command approval will fail closed",
                        exc_info=True,
                    )
            try:
                return target(*args, **kwargs)
            finally:
                if setters is not None:
                    (set_approval,) = setters
                    try:
                        set_approval(None)
                    except Exception:
                        logger.debug(
                            "Failed to clear propagated approval callback",
                            exc_info=True,
                        )

        return ctx.run(_inner)

    return _runner
