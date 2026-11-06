"""Typed clarify replies must claim unresolved FIFO entries (issue #135339)."""

from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import pytest

from tools import clarify_gateway as cg


@pytest.fixture(autouse=True)
def isolated_clarify_queue():
    with cg._lock:
        cg._entries.clear()
        cg._session_index.clear()
        cg._notify_cbs.clear()
    yield
    with cg._lock:
        cg._entries.clear()
        cg._session_index.clear()
        cg._notify_cbs.clear()


@pytest.mark.parametrize("first_response", ["answer A", cg.CANCELLED, cg.SKIPPED])
def test_completed_head_does_not_hide_the_next_pending_reply(first_response):
    first = cg.register("first", "session", "First?", None)
    second = cg.register("second", "session", "Second?", None)
    other = cg.register("other", "other-session", "Unrelated?", None)

    assert cg.resolve_gateway_clarify(first.clarify_id, first_response)
    # The worker has not removed the completed head from the queue yet.
    assert cg.get_pending_for_session("session") is second
    assert cg.has_pending("session")
    assert cg.attempt_text_response_for_session("session", "answer B") == cg.TEXT_RESOLVED
    assert not cg.has_pending("session")
    assert cg.get_pending_for_session("session", include_choice_prompts=True) is None
    assert cg.wait_for_response(first.clarify_id, 5) == first_response
    assert cg.wait_for_response(second.clarify_id, 5) == "answer B"
    assert cg.get_pending_for_session("other-session") is other
    assert not other.event.is_set()


def test_concurrent_text_replies_claim_distinct_entries_without_worker_cleanup():
    first = cg.register("first", "session", "First?", None)
    second = cg.register("second", "session", "Second?", None)
    start = Barrier(2)

    def reply(text):
        start.wait(timeout=5)
        return cg.attempt_text_response_for_session("session", text)

    with ThreadPoolExecutor(max_workers=2) as pool:
        replies = [pool.submit(reply, text) for text in ("answer A", "answer B")]
        assert [future.result(timeout=5) for future in replies] == [cg.TEXT_RESOLVED] * 2
    assert {first.response, second.response} == {"answer A", "answer B"}
    assert not cg.has_pending("session")
