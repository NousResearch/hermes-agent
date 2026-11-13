"""Coalesced gateway approvals under concurrent leader/follower threads.

Identical concurrent approvals share ONE prompt: the first caller (leader) enqueues and notifies, later
callers (followers) wait on the leader's event and adopt its outcome. These tests force the interleavings
where the leader's prompt goes away without an answer and where two identical requests race to lead.
"""

import threading

from tools import approval as mod
from tools import approval_gateway_wait as wait_mod


SESSION_KEY = "approval-coalesce-race"
APPROVAL = {"command": "rm -rf build", "description": "d", "pattern_key": "dangerous", "pattern_keys": ["dangerous"]}
# Long enough that a follower left waiting on a dead leader is unmistakable; the tests release it on failure.
APPROVAL_TIMEOUT = 60


def _clear():
    mod._gateway_queues.clear()
    mod._gateway_notify_cbs.clear()


def _leader_never_polls(follower_waiting):
    raise AssertionError("the leader's notify failed; it must not wait")


def _run_leader_then_follower(monkeypatch, notify_cb, *, leader_poll=_leader_never_polls):
    """Start a leader, then a follower that coalesces onto it; *leader_poll* decides how the leader's own
    wait ends once the follower is blocked on the leader's event. Returns (results, follower, release) —
    ``release()`` frees a follower still blocked on the leader's (already dequeued) entry."""
    monkeypatch.setattr(wait_mod._ctx, "_fire_approval_hook", lambda name, **kw: None)
    monkeypatch.setattr(wait_mod._ctx, "_get_approval_timeout", lambda: APPROVAL_TIMEOUT)
    real_poll = wait_mod._poll_event
    follower_waiting = threading.Event()
    leader_enqueued = threading.Event()
    results: dict = {}

    def poll(event, session_key, *, interrupt_log, timeout_s=None):
        if "Coalesced" in interrupt_log:
            follower_waiting.set()
            return real_poll(event, session_key, interrupt_log=interrupt_log)
        return leader_poll(follower_waiting)

    monkeypatch.setattr(wait_mod, "_poll_event", poll)

    def leader_notify(data):
        results["leader_entry"] = mod._gateway_queues[SESSION_KEY][0]
        leader_enqueued.set()
        notify_cb(data, follower_waiting)

    def run(name, cb):
        results[name] = wait_mod._await_gateway_decision(SESSION_KEY, cb, dict(APPROVAL))

    leader = threading.Thread(target=run, args=("leader", leader_notify), daemon=True)
    leader.start()
    assert leader_enqueued.wait(10)
    follower = threading.Thread(target=run, args=("follower", lambda data: None), daemon=True)
    follower.start()
    leader.join(10)
    assert not leader.is_alive()

    def release():
        results["leader_entry"].event.set()
        follower.join(10)

    return results, follower, release


def test_follower_adopts_the_leaders_timeout_instead_of_hanging(monkeypatch):
    _clear()

    def leader_times_out(follower_waiting):
        assert follower_waiting.wait(10)
        return "timeout"

    results, follower, release = _run_leader_then_follower(
        monkeypatch, lambda data, follower_waiting: None, leader_poll=leader_times_out)
    follower.join(10)
    alive = follower.is_alive()
    release()
    assert not alive, "follower still blocked after the leader's prompt expired"
    assert results["leader"] == {"resolved": False, "choice": None, "reason": None}
    assert results["follower"] == {"resolved": False, "choice": None, "reason": None, "coalesced": True}
    # No prompt is left pending for a late /approve to be told "ok" about.
    assert mod.resolve_gateway_approval(SESSION_KEY, "once") == 0


def test_follower_wakes_when_the_leaders_notify_fails(monkeypatch):
    _clear()

    def notify_fails_after_follower_joined(data, follower_waiting):
        assert follower_waiting.wait(10)
        raise RuntimeError("platform send failed")

    results, follower, release = _run_leader_then_follower(monkeypatch, notify_fails_after_follower_joined)
    follower.join(10)
    alive = follower.is_alive()
    release()
    assert not alive, "follower still blocked after the leader's prompt could not be delivered"
    assert results["leader"] == {"resolved": False, "choice": None, "notify_failed": True}
    decision = results["follower"]
    assert decision["resolved"] is False and decision["choice"] is None and decision["coalesced"] is True
    # Never delivered, so it is a withdrawal — not a user deny and not a silent timeout.
    assert decision["cancelled"]


def test_identical_requests_racing_to_lead_send_one_prompt(monkeypatch):
    """Two identical approvals that both reach the leader lookup before either is enqueued must still
    coalesce: the lookup and the enqueue are one critical section, so the second sees the first."""
    _clear()
    monkeypatch.setattr(wait_mod._ctx, "_fire_approval_hook", lambda name, **kw: None)
    monkeypatch.setattr(wait_mod._ctx, "_get_approval_timeout", lambda: APPROVAL_TIMEOUT)
    both_built = threading.Barrier(2)
    prompted = threading.Event()
    settled = threading.Event()  # a second prompt went out, or the second caller started following
    notified: list[str] = []
    notified_lock = threading.Lock()

    class _BothBuildBeforeEitherEnqueues(wait_mod._ApprovalEntry):
        __slots__ = ()

        def __init__(self, data):
            both_built.wait(10)
            super().__init__(data)

    real_follow = wait_mod._await_coalesced_leader

    def follow(session_key, leader, payload):
        settled.set()
        return real_follow(session_key, leader, payload)

    def notify(data):
        with notified_lock:
            notified.append(data["request_id"])
            prompted.set()
            if len(notified) == 2:
                settled.set()

    monkeypatch.setattr(wait_mod, "_ApprovalEntry", _BothBuildBeforeEitherEnqueues)
    monkeypatch.setattr(wait_mod, "_await_coalesced_leader", follow)
    results: list[dict] = []

    def run():
        results.append(wait_mod._await_gateway_decision(SESSION_KEY, notify, dict(APPROVAL)))

    threads = [threading.Thread(target=run, daemon=True) for _ in range(2)]
    for thread in threads:
        thread.start()
    assert prompted.wait(10)
    assert settled.wait(10)
    prompts = len(notified)
    queued = len(mod._gateway_queues.get(SESSION_KEY, []))
    # One /approve answers the only prompt (and, on failure, every duplicate so the threads exit).
    mod.resolve_gateway_approval(SESSION_KEY, "session", resolve_all=True)
    for thread in threads:
        thread.join(10)
    assert (prompts, queued) == (1, 1)
    assert sorted((r["resolved"], r["choice"], bool(r.get("coalesced"))) for r in results) == [
        (True, "session", False), (True, "session", True)]


def test_followers_released_by_once_share_one_fresh_prompt(monkeypatch):
    """``once`` covers only the leader's execution, so each follower still needs consent of its own —
    but one prompt at a time: released followers coalesce again behind the first to re-prompt instead
    of each sending a duplicate."""
    from tests.waiting import wait_until

    _clear()
    monkeypatch.setattr(wait_mod._ctx, "_fire_approval_hook", lambda name, **kw: None)
    monkeypatch.setattr(wait_mod._ctx, "_get_approval_timeout", lambda: APPROVAL_TIMEOUT)
    notified: list[str] = []
    follows: list[str] = []
    real_follow = wait_mod._await_coalesced_leader

    def follow(session_key, leader, payload):
        follows.append(leader.data["request_id"])
        return real_follow(session_key, leader, payload)

    monkeypatch.setattr(wait_mod, "_await_coalesced_leader", follow)
    results: list[dict] = []

    def run():
        results.append(wait_mod._await_gateway_decision(SESSION_KEY, notified.append, dict(APPROVAL)))

    threads = [threading.Thread(target=run, daemon=True) for _ in range(3)]
    threads[0].start()
    wait_until(lambda: len(notified) == 1, what="the leader's prompt")
    for thread in threads[1:]:
        thread.start()
    wait_until(lambda: len(follows) == 2, what="both followers waiting on the leader")

    assert mod.resolve_gateway_approval(SESSION_KEY, "once") == 1
    # Settled: either a duplicate prompt went out per follower, or one follower re-led and the other follows it.
    wait_until(lambda: len(notified) == 3 or (len(notified) == 2 and len(follows) == 3),
               what="released followers re-prompted or re-coalesced")
    prompts = len(notified)
    queued = len(mod._gateway_queues.get(SESSION_KEY, []))
    while any(thread.is_alive() for thread in threads):  # answer each remaining execution once
        mod.resolve_gateway_approval(SESSION_KEY, "once", resolve_all=True)
        for thread in threads:
            thread.join(0.05)
    assert (prompts, queued) == (2, 1)
    assert [r["choice"] for r in results] == ["once"] * 3
