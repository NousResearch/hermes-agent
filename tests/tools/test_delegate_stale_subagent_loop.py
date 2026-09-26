"""Regression tests for #94858 — parent-agent retry loop against a dead subagent.

Original symptom (from the issue's agent.log):
  - Parent spawns a subagent, the child finishes.
  - Parent keeps calling delegate_task(action='steer'/'stop',
    subagent_id='sa-0-...') against the already-finished child.
  - The tool keeps returning a recoverable-looking
    "No live subagent 'sa-0-...' ..." error.
  - The model interprets the error as something to retry, and the parent
    burns tokens in an unbounded loop until the operator kills the
    gateway.

The fix has two halves (one test each):
  1. **Bounded retry** — after a small per-(parent, target) cap, the
     error text explicitly names the loop and stops inviting retries.
  2. **Always-on, structured non-retryable surface** — every "subagent
     is gone / no longer accepting" error is JSON-tagged with
     ``recoverable=False`` and ``do_not_retry=True`` so a well-behaved
     model stops on the first failure, not the fourth.

Pre-fix behavior:
  - The error string never changes shape no matter how many times the
    parent calls. Tests that drive the stub for N>=4 attempts and
    expect the error text to evolve would FAIL pre-fix and PASS
    post-fix — that is the regression we are pinning.

Post-fix behavior:
  - Attempt #1: tagged non-retryable, plain "subagent is gone" reason.
  - Attempts #2..#limit: same tagged shape (no escalation yet, the
    model is expected to honour the first one).
  - Attempts #limit+1..: loop-detected, names the offending call and
    points at action='list' / spawn / completion-message as the
    real alternatives.
"""

import json
import threading
import weakref

import pytest

from tools.delegate_tool import (
    _dead_subagent_hits,
    _record_dead_subagent_hit,
    _reset_dead_subagent_hits,
    _STALE_SUBAGENT_RETRY_LIMIT,
    _handle_control_action,
    _non_retryable_subagent_error,
    _register_subagent,
    _unregister_subagent,
)


# ---------------------------------------------------------------------------
# Stub fixtures
# ---------------------------------------------------------------------------


class _StubChild:
    """Weakref-able stand-in for a live child AIAgent.

    `accept_steer=False` makes the underlying steer() return False —
    i.e. the child is registered but has closed its steering window.
    That is the second terminal class from the issue log: the model
    thinks the child is alive, but the registry no longer accepts work
    for it. The first terminal class (target missing entirely) is
    exercised by simply NOT registering the child.
    """

    def __init__(self, parent=None, accept_steer: bool = False):
        self.steered: list[str] = []
        self.accept_steer = accept_steer
        self._live_transcript_path = "/tmp/live/loop-94858.log"
        if parent is not None:
            self._delegate_parent_ref = weakref.ref(parent)

    def steer(self, text: str) -> bool:
        if not self.accept_steer:
            return False
        self.steered.append(text)
        return True


class _StubParent:
    """Minimal parent agent with a session_id for the durable spine."""

    def __init__(self, session_id: str = "sess-94858"):
        self.session_id = session_id


def _register(sid: str, child, **extra) -> None:
    record = {
        "subagent_id": sid,
        "parent_id": None,
        "depth": 0,
        "goal": "loop regression",
        "model": "test-model",
        "started_at": 1000.0,
        "status": "running",
        "tool_count": 0,
        "agent": child,
    }
    record.update(extra)
    _register_subagent(record)


@pytest.fixture(autouse=True)
def _isolate_loop_counter():
    """Wipe the module-level loop counter around every test.

    The counter is keyed on (parent_session_id, subagent_id) and lives
    for the lifetime of the process, so cross-test bleed would silently
    pre-bias the assertions (a previous test's misses would push the
    next test straight into the loop-detected branch).
    """
    with _counter_lock():
        _dead_subagent_hits.clear()
    yield
    with _counter_lock():
        _dead_subagent_hits.clear()


def _counter_lock():
    # Same lock the production helpers use; imported here so the fixture
    # and the production code can never disagree about which lock guards
    # the dict.
    from tools.delegate_tool import _dead_subagent_hits_lock

    return _dead_subagent_hits_lock


# ---------------------------------------------------------------------------
# 1. Bounded retry — counter helpers
# ---------------------------------------------------------------------------


def test_counter_increments_per_target():
    """Each (parent, subagent_id) miss is counted independently.

    Pre-fix: there was no counter, so this assertion was structurally
    impossible — pre-fix code would not even have _record_dead_subagent_hit
    to call. Post-fix: same parent targeting two different dead ids
    keeps each counter at 1.
    """
    parent_sid = "sess-counter-1"
    a_count = _record_dead_subagent_hit(parent_sid, "sa-0-aaa")
    b_count = _record_dead_subagent_hit(parent_sid, "sa-0-bbb")
    assert a_count == 1
    assert b_count == 1
    a_count_2 = _record_dead_subagent_hit(parent_sid, "sa-0-aaa")
    assert a_count_2 == 2
    # Different parent sees its own counter
    other_count = _record_dead_subagent_hit("sess-counter-2", "sa-0-aaa")
    assert other_count == 1


def test_counter_resets_when_child_re_registers():
    """A recycled public id (new child run) wipes the miss counter.

    Without this, a parent that retried a dead id earlier in the same
    session would inherit the per-(parent, target) count and trip the
    loop detector on the very first call against the new, healthy
    child. The reset lives in _register_subagent.
    """
    parent = _StubParent("sess-counter-reset")
    _record_dead_subagent_hit(parent.session_id, "sa-0-recycle")
    _record_dead_subagent_hit(parent.session_id, "sa-0-recycle")
    _record_dead_subagent_hit(parent.session_id, "sa-0-recycle")
    assert _record_dead_subagent_hit(parent.session_id, "sa-0-recycle") == 4

    # Now the SAME public id is taken over by a brand-new child run.
    new_child = _StubChild(parent, accept_steer=True)
    _register(
        "sa-0-recycle",
        new_child,
        owner_agent_session_id=parent.session_id,
    )
    try:
        # The counter was wiped at register time. A first failure against
        # the new child starts back at 1.
        assert _record_dead_subagent_hit(parent.session_id, "sa-0-recycle") == 1
    finally:
        _unregister_subagent("sa-0-recycle")


# ---------------------------------------------------------------------------
# 2. Always-on, structured non-retryable surface
# ---------------------------------------------------------------------------


def test_missing_target_error_is_marked_non_retryable():
    """The very FIRST failure against a dead/missing target is tagged
    non-retryable. This is the half of the fix that addresses a
    well-behaved model: a single attempt should be enough.
    """
    parent = _StubParent("sess-surface-1")
    out = _handle_control_action("steer", "sa-0-ghost", "hello", parent)
    payload = json.loads(out)
    assert payload["error"]
    assert payload["recoverable"] is False
    assert payload["do_not_retry"] is True
    assert payload["subagent_id"] == "sa-0-ghost"
    # The hint is hard-line about not retrying — pre-fix this was a
    # soft "use action='list'" suggestion that the model interpreted as
    # "try list, then try steer again".
    assert "do not retry" in payload["hint"].lower()
    # The plain first-attempt reason does NOT yet trip the loop detector.
    assert "loop_detected" not in payload


def test_closed_acceptance_error_is_marked_non_retryable():
    """A child that is registered but has closed its steering window
    produces the same non-retryable surface. Same loop class from the
    model's point of view: retrying will never flip the answer.
    """
    parent = _StubParent("sess-surface-2")
    child = _StubChild(parent, accept_steer=False)  # steer() always returns False
    _register("sa-0-closed", child, owner_agent_session_id=parent.session_id)
    try:
        out = _handle_control_action("steer", "sa-0-closed", "nudge", parent)
        payload = json.loads(out)
        assert payload["recoverable"] is False
        assert payload["do_not_retry"] is True
        assert "no longer accepting" in payload["error"].lower()
        assert child.steered == []  # we never actually wrote to it
    finally:
        _unregister_subagent("sa-0-closed")


def test_stop_unknown_id_is_marked_non_retryable():
    """The stop path also goes through the new surface. A parent that
    keeps issuing stop on a finished child hits the same loop class
    as steer (#94858's log shows both `steer` and `stop` errors in the
    same runaway conversation)."""
    parent = _StubParent("sess-surface-3")
    out = _handle_control_action("stop", "sa-0-finished", None, parent)
    payload = json.loads(out)
    assert payload["recoverable"] is False
    assert payload["do_not_retry"] is True
    assert payload["subagent_id"] == "sa-0-finished"


# ---------------------------------------------------------------------------
# 3. Bounded retry — end-to-end: stub steer that always fails
# ---------------------------------------------------------------------------


def _drive_n_steers(n: int, parent_sid: str, child_sid: str) -> list[dict]:
    """Helper: invoke _handle_control_action n times against the same
    (parent, dead child) and parse each result.

    Pre-fix this returned n identical "No live subagent ..." strings.
    Post-fix the first three are tagged non-retryable; from the fourth
    onward the payload escalates to loop_detected=True.
    """
    parent = _StubParent(parent_sid)
    payloads: list[dict] = []
    for _ in range(n):
        raw = _handle_control_action("steer", child_sid, "nudge", parent)
        payloads.append(json.loads(raw))
    return payloads


def test_retry_against_dead_target_is_bounded_and_obvious():
    """The headline regression for #94858.

    Pre-fix: any number of calls against a dead subagent return the
    same recoverable-looking error and the model has no way to know it
    is in a loop. The error payload never carries loop_detected.

    Post-fix: attempts 1..limit are tagged non-retryable, attempt
    limit+1 and beyond are explicitly flagged as a retry loop and
    point at action='list' / spawn / completion-message.

    This test runs enough attempts to cross the loop threshold and
    asserts the payload's *shape* changed, not just its text. A model
    that pattern-matches on `loop_detected` will stop the instant it
    sees the field flip — no text parsing required.
    """
    parent_sid = "sess-94858"
    child_sid = "sa-0-b392cfae"  # the dead id from the original log
    attempts = _STALE_SUBAGENT_RETRY_LIMIT + 3  # well past the threshold
    payloads = _drive_n_steers(attempts, parent_sid, child_sid)

    assert len(payloads) == attempts

    # First N attempts: non-retryable, but the model is still given the
    # benefit of the doubt — no loop_detected yet, the assumption is it
    # will read the do_not_retry hint and stop.
    for i, p in enumerate(payloads[:_STALE_SUBAGENT_RETRY_LIMIT]):
        assert p["recoverable"] is False, f"attempt {i + 1} not non-retryable"
        assert p["do_not_retry"] is True
        assert "loop_detected" not in p
        assert "do not retry" in p["hint"].lower()

    # Attempt limit+1 and beyond: the loop detector trips. The reason
    # text names the offending call so the model can see exactly which
    # call it is making and stop making it.
    for i, p in enumerate(payloads[_STALE_SUBAGENT_RETRY_LIMIT:], start=_STALE_SUBAGENT_RETRY_LIMIT):
        assert p["recoverable"] is False
        assert p["do_not_retry"] is True
        assert p["loop_detected"] is True, f"attempt {i + 1} should be flagged"
        # Names the exact call the model made.
        assert child_sid in p["error"]
        assert f"action='steer'" in p["error"]
        # Names a real alternative.
        assert "action='list'" in p["hint"]
        # Tells the model the prior failures were the same error.
        assert p["attempts"] == i + 1


def test_loop_detected_payload_uses_stable_keys():
    """The loop-detected payload is structured for programmatic use,
    not just prose. A future prompt-builder or guardrail can match on
    these keys without parsing free text.
    """
    parent_sid = "sess-stable-keys"
    child_sid = "sa-0-stable"
    payloads = _drive_n_steers(_STALE_SUBAGENT_RETRY_LIMIT + 1, parent_sid, child_sid)
    tripped = payloads[-1]
    for key in (
        "error",
        "recoverable",
        "subagent_id",
        "do_not_retry",
        "hint",
        "loop_detected",
        "attempts",
    ):
        assert key in tripped, f"missing key: {key}"


def test_loop_counter_is_per_parent_session():
    """Two parents hammering the same dead id each get their own
    counter, so a long-running operator session with many parallel
    parents doesn't accidentally trip the breaker for the wrong
    conversation. The original log shows the parent session was
    `session_parent_main`; sibling sessions must be unaffected.
    """
    p1 = "sess-parent-a"
    p2 = "sess-parent-b"
    same_dead = "sa-0-shared"

    # Drive p1 well past the loop threshold.
    p1_payloads = _drive_n_steers(_STALE_SUBAGENT_RETRY_LIMIT + 2, p1, same_dead)
    assert p1_payloads[-1]["loop_detected"] is True

    # p2 against the same dead id sees its own counter starting at 1.
    # Below the threshold the helper omits `loop_detected` entirely
    # (the first attempts are not yet escalated to a loop) — assert
    # the key is absent rather than testing a sentinel False.
    p2_payloads = _drive_n_steers(2, p2, same_dead)
    assert "loop_detected" not in p2_payloads[0]
    assert p2_payloads[0]["recoverable"] is False


def test_help_non_retryable_helper_includes_loop_metadata_only_after_threshold():
    """Direct test of the JSON-shape helper. Pre-fix there was no
    non-retryable marker at all; the helper itself is part of the fix.
    """
    plain = _non_retryable_subagent_error(
        subagent_id="sa-0",
        reason="gone",
        hint="do not retry",
        loop_count=_STALE_SUBAGENT_RETRY_LIMIT,  # exactly at threshold
    )
    plain_payload = json.loads(plain)
    # At-or-below threshold: no loop_detected key.
    assert "loop_detected" not in plain_payload
    assert plain_payload["recoverable"] is False
    assert plain_payload["do_not_retry"] is True

    tripped = _non_retryable_subagent_error(
        subagent_id="sa-0",
        reason="loop",
        hint="see list",
        loop_count=_STALE_SUBAGENT_RETRY_LIMIT + 1,  # past threshold
    )
    tripped_payload = json.loads(tripped)
    assert tripped_payload["loop_detected"] is True
    assert tripped_payload["attempts"] == _STALE_SUBAGENT_RETRY_LIMIT + 1


# ---------------------------------------------------------------------------
# 4. Pre-fix behaviour, captured as a delta test
# ---------------------------------------------------------------------------
#
# The text below documents what the *old* tool_error() payload looked
# like for the same call. We keep this delta test here so a future
# refactor that accidentally regresses the structured surface (e.g. by
# replacing the new helper with a bare tool_error(...)) trips a
# readable failure, and so the diff between the two shapes is captured
# once in code rather than only in a PR description.
#
# Pre-fix payload (one attempt against a dead subagent):
#   '{"error": "No live subagent \\'sa-0-...\\' ... Use action=\\'list\\' ..."}'
# Post-fix payload (same one attempt):
#   '{"error": "No live subagent \\'sa-0-...\\' ...",
#     "recoverable": false,
#     "subagent_id": "sa-0-...",
#     "do_not_retry": true,
#     "hint": "Do not retry delegate_task(...) ..."}'


def test_error_payload_is_structured_not_plain():
    """The payload MUST have a structured recoverable flag — a plain
    ``{"error": "..."}`` is the exact regression that lets the loop
    continue. This test pins the contract."""
    parent = _StubParent("sess-structured")
    out = _handle_control_action("steer", "sa-0-b392cfae", "nudge", parent)
    payload = json.loads(out)
    # Required keys for downstream tooling / guardrails.
    for key in ("error", "recoverable", "do_not_retry", "subagent_id", "hint"):
        assert key in payload, f"missing structured key: {key}"
    # And a plain `error` key with a recognizable message is preserved
    # so existing log scrapers and the operator-facing TUI overlay
    # keep working.
    assert "No live subagent" in payload["error"]


# ---------------------------------------------------------------------------
# 5. Circuit-breaker hygiene (review findings, pinned on current main)
# ---------------------------------------------------------------------------


def test_eviction_never_un_trips_a_live_breaker():
    """Flooding the counter with fresh keys must not launder an already-tripped one.

    The original FIFO evicted ``next(iter(dict))`` unconditionally once the dict
    passed 1024 entries, so 1025 concurrent fresh dead-ids could silently reset
    another session's tripped breaker and hand that runaway loop full slack again.
    """
    from tools.delegate_tool import _DEAD_SUBAGENT_HITS_CAP, _dead_subagent_hits

    tripped_key = ("sess-hygiene-tripped", "sa-0-tripped")
    for _ in range(_STALE_SUBAGENT_RETRY_LIMIT + 2):
        _record_dead_subagent_hit(*tripped_key)
    assert _dead_subagent_hits[tripped_key] > _STALE_SUBAGENT_RETRY_LIMIT

    # Well past the cap in fresh (parent, target) pairs, none of them tripped.
    for i in range(_DEAD_SUBAGENT_HITS_CAP + 50):
        _record_dead_subagent_hit(f"sess-hygiene-flood-{i}", f"sa-0-flood-{i}")

    assert _dead_subagent_hits.get(tripped_key) == _STALE_SUBAGENT_RETRY_LIMIT + 2
    assert len(_dead_subagent_hits) <= _DEAD_SUBAGENT_HITS_CAP + 1


def test_reset_runs_without_owner_session_id():
    """A recycled id must reset even when the record carries no owner session id.

    The reset used to be gated on ``owner_agent_session_id`` being present, so a
    record registered without it (a parent with no session_id, or any future
    spawn path that forgets to stamp the field) left the recycled id inheriting
    its predecessor's miss count -- the exact false positive the lifecycle
    pairing exists to prevent.

    The parent here has NO session_id, so both the miss counter and the reset
    key land in the "" bucket; that is the shape the old gate broke.
    """
    sid = "sa-0-recycle-no-owner"
    for _ in range(3):
        _record_dead_subagent_hit("", sid)
    assert _record_dead_subagent_hit("", sid) == 4

    # Register WITHOUT owner_agent_session_id. A parent with no session_id
    # records its misses under the "" key, so the reset must still fire.
    _register_subagent({"subagent_id": sid, "agent": _StubChild(_StubParent()), "goal": "g"})
    try:
        assert _record_dead_subagent_hit("", sid) == 1
    finally:
        _unregister_subagent(sid)

