"""A finished background delegation does not keep its dispatching turn's state alive."""

import contextvars
import gc
import weakref

from tools import async_delegation as ad
from tools.process_registry import process_registry

_turn_state = contextvars.ContextVar("_turn_state")


class _TurnState:
    """Stands in for what a dispatching gateway turn has bound (its agent, callbacks, history)."""


def _dispatch_under_turn_state():
    state = _TurnState()
    ref = weakref.ref(state)

    def dispatch():
        _turn_state.set(state)
        return ad.dispatch_async_delegation(goal="retention", context=None, toolsets=None, role="leaf",
                                            model=None, session_key="retention-test",
                                            runner=lambda: {"summary": "done"})

    handle = contextvars.copy_context().run(dispatch)
    return handle, ref


def test_completed_record_releases_dispatch_context():
    ad._reset_for_tests()
    try:
        handle, ref = _dispatch_under_turn_state()
        assert handle["status"] == "dispatched"
        # Join the worker thread: a finished record only proves bookkeeping ran, not that the
        # worker has left the Contexts it copied, and those would still hold the state alive.
        ad._executor.shutdown(wait=True)
        assert ad.active_count() == 0
        while not process_registry.completion_queue.empty():
            process_registry.completion_queue.get_nowait()
        # The completed record stays listed (bounded history) but no longer pins the turn's state.
        assert handle["delegation_id"] in ad._records
        gc.collect()
        assert ref() is None
    finally:
        if ad._executor is not None:
            ad._executor.shutdown(wait=True)
        ad._reset_for_tests()
