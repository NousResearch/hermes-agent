"""Call-local abandonment is distinct from session cancellation and user input."""
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from tools import clarify_gateway as cm
from tools.clarify_tool import clarify_tool
from tools.thread_context import propagate_context_to_thread


@pytest.fixture(autouse=True)
def clean_queue():
    yield
    cm.clear_session("scope-test")


def test_scope_cancellation_wakes_only_owned_prompt():
    owner, other = cm.ClarifyWaitScope(), cm.ClarifyWaitScope()
    entered = threading.Event()
    with cm.bind_wait_scope(other):
        other_entry = cm.register("other", "scope-test", "Other?", None)
    cm.register("unowned", "scope-test", "Unowned?", None)

    def wait():
        entry = cm.register("owned", "scope-test", "Owned?", None)
        original_wait = entry.event.wait

        def observed_wait(timeout=None):
            entered.set()
            return original_wait(timeout)

        entry.event.wait = observed_wait
        return cm.wait_for_response("owned", timeout=0)

    with ThreadPoolExecutor(1) as pool:
        with cm.bind_wait_scope(owner):
            future = pool.submit(propagate_context_to_thread(wait))
        assert entered.wait(3)
        owner.cancel()
        assert future.result(timeout=3) == cm.CANCELLED
    assert not cm.resolve_gateway_clarify("owned", "late")
    assert not other_entry.event.is_set()
    assert cm.resolve_gateway_clarify("other", "answer")
    assert cm.wait_for_response("other", timeout=1) == "answer"
    assert cm.resolve_gateway_clarify("unowned", "still active")


def test_cancelled_scope_cannot_register_again_or_leak_to_next_call():
    scope = cm.ClarifyWaitScope()
    scope.cancel()
    with cm.bind_wait_scope(scope):
        with pytest.raises(cm.ClarifyWaitAbandoned):
            cm.register("late", "scope-test", "Late?", None)
        with pytest.raises(cm.ClarifyWaitAbandoned):
            clarify_tool([{"question": "Late?"}], callback=lambda _: pytest.fail("callback invoked"))
        assert cm.wait_for_response("late", timeout=0) == cm.CANCELLED
    assert not cm.has_pending("scope-test")
    # A pooled worker's subsequent independent call must not inherit abandonment.
    cm.register("next", "scope-test", "Next?", None)
    assert cm.resolve_gateway_clarify("next", "ok")
    assert cm.wait_for_response("next", timeout=1) == "ok"


@pytest.mark.parametrize("callback_raises", [False, True])
def test_late_callback_does_not_resume_plugin_even_when_callback_raises(callback_raises):
    scope = cm.ClarifyWaitScope()

    def callback(_questions):
        scope.cancel()
        if callback_raises:
            raise ValueError("surface disappeared")
        return {"answers": {"q0": "late answer"}, "outcome": "submitted"}

    with cm.bind_wait_scope(scope), pytest.raises(cm.ClarifyWaitAbandoned):
        clarify_tool([{"question": "Continue?"}], callback=callback)


def test_answer_won_before_scope_cancel_is_not_reused_as_plugin_continuation():
    scope = cm.ClarifyWaitScope()
    with cm.bind_wait_scope(scope):
        entry = cm.register("race", "scope-test", "Continue?", None)
        assert cm.resolve_gateway_clarify("race", "yes")
        scope.cancel()
        # Preserve the original selection on the entry, but ownership was abandoned.
        assert entry.response == "yes"
        assert cm.wait_for_response("race", timeout=0) == cm.CANCELLED


def test_cancel_prompt_preserves_other_prompt_and_already_selected_answer():
    cm.register("delivery-failed", "scope-test", "Failed?", None)
    cm.register("answer-won", "scope-test", "Answered?", None)
    assert cm.resolve_gateway_clarify("answer-won", "keep me")
    assert cm.cancel_prompt("delivery-failed")
    assert not cm.cancel_prompt("answer-won")
    assert not cm.resolve_gateway_clarify("delivery-failed", "late")
    assert cm.wait_for_response("answer-won", timeout=1) == "keep me"
