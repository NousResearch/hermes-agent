"""A gateway approval choice that lands after the deadline check is an answer, not a timeout (#112548).

``_await_gateway_decision`` polls the entry's event with a deadline; ``resolve_gateway_approval`` may
commit a choice between the poll giving up and the entry leaving the queue. The verdict must be the
choice committed under the approval lock — the client was acked ``ok`` for it — never "timeout".
"""

from tools import approval as mod
from tools import approval_gateway_wait as wait_mod


SESSION_KEY = "approval-late-choice"
APPROVAL = {"command": "rm -rf build", "description": "d", "pattern_key": "dangerous", "pattern_keys": ["dangerous"]}


def _clear():
    mod._gateway_queues.clear()
    mod._gateway_notify_cbs.clear()


def test_choice_landing_after_the_deadline_check_is_resolved(monkeypatch):
    _clear()
    hooks: list[tuple[str, dict]] = []
    monkeypatch.setattr(wait_mod._ctx, "_fire_approval_hook", lambda name, **kw: hooks.append((name, kw)))

    settles = []

    def deadline_passed_then_user_answered(event, session_key, *, interrupt_log, timeout_s=None):
        # The poll loop has already decided "timeout"; the user's /approve lands before the entry leaves the queue.
        request_id = mod._gateway_queues[session_key][0].data["request_id"]
        assert mod.register_gateway_settle(session_key, request_id, settles.append)
        assert mod.resolve_gateway_approval(session_key, "once") == 1
        return "timeout"

    monkeypatch.setattr(wait_mod, "_poll_event", deadline_passed_then_user_answered)

    decision = wait_mod._await_gateway_decision(SESSION_KEY, lambda data: None, APPROVAL)

    assert decision == {"resolved": True, "choice": "once", "reason": None}
    assert hooks[-1][0] == "post_approval_response" and hooks[-1][1]["choice"] == "once"
    assert settles == ["resolved"]
    assert SESSION_KEY not in mod._gateway_queues


def test_plain_timeout_still_reports_unresolved(monkeypatch):
    _clear()
    monkeypatch.setattr(wait_mod._ctx, "_fire_approval_hook", lambda name, **kw: None)
    monkeypatch.setattr(wait_mod, "_poll_event", lambda event, session_key, *, interrupt_log, timeout_s=None: "timeout")

    decision = wait_mod._await_gateway_decision(SESSION_KEY, lambda data: None, APPROVAL)

    assert decision == {"resolved": False, "choice": None, "reason": None}
    # Nothing is left for a late /approve to hit: the client learns nothing was pending.
    assert mod.resolve_gateway_approval(SESSION_KEY, "once") == 0


def test_multiple_observers_and_settle_hook_run_on_expiry(monkeypatch):
    _clear()
    monkeypatch.setattr(wait_mod._ctx, "_fire_approval_hook", lambda name, **kw: None)
    monkeypatch.setattr(wait_mod, "_poll_event", lambda event, session_key, *, interrupt_log, timeout_s=None: "timeout")
    outcomes = []

    def notify(data):
        rid = data["request_id"]
        assert mod.register_gateway_settle(SESSION_KEY, rid, lambda reason: outcomes.append(("notice", reason)))
        assert mod.observe_gateway_settle(SESSION_KEY, rid, lambda reason: outcomes.append(("card", reason)))

    wait_mod._await_gateway_decision(SESSION_KEY, notify, APPROVAL)
    assert outcomes == [("notice", "timeout"), ("card", "timeout")]


def test_prompt_and_wait_use_same_profile_timeout_when_config_changes(monkeypatch):
    _clear()
    current = {"timeout": 3600}
    monkeypatch.setattr(wait_mod._ctx, "_get_approval_timeout", lambda: current["timeout"])
    monkeypatch.setattr(wait_mod._ctx, "_fire_approval_hook", lambda name, **kw: None)
    observed = {}

    def notify(data):
        observed["prompt_timeout"] = data["timeout_s"]
        current["timeout"] = 300

    def poll(event, session_key, *, interrupt_log, timeout_s=None):
        observed["wait_timeout"] = timeout_s
        return "timeout"

    monkeypatch.setattr(wait_mod, "_poll_event", poll)
    decision = wait_mod._await_gateway_decision(SESSION_KEY, notify, APPROVAL)

    assert decision["resolved"] is False
    assert observed == {"prompt_timeout": 3600, "wait_timeout": 3600}


def test_gateway_approval_captures_routed_profiles_timeout(tmp_path, monkeypatch):
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    _clear()
    default = tmp_path / "default"
    routed = tmp_path / "routed"
    for home, timeout in ((default, 300), (routed, 3600)):
        home.mkdir()
        (home / "config.yaml").write_text(f"approvals:\n  timeout: {timeout}\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(default))
    monkeypatch.setattr(wait_mod._ctx, "_fire_approval_hook", lambda name, **kw: None)
    observed = []

    def poll(event, session_key, *, interrupt_log, timeout_s=None):
        observed.append(timeout_s)
        return "timeout"

    monkeypatch.setattr(wait_mod, "_poll_event", poll)
    for home in (default, routed):
        token = set_hermes_home_override(str(home))
        try:
            wait_mod._await_gateway_decision(SESSION_KEY, lambda data: observed.append(data["timeout_s"]), APPROVAL)
        finally:
            reset_hermes_home_override(token)
    assert observed == [300, 300, 3600, 3600]


def test_coalesced_follower_waits_the_leaders_captured_timeout(monkeypatch):
    from tools.approval_gateway_wait import _ApprovalEntry

    _clear()
    monkeypatch.setattr(wait_mod._ctx, "_get_approval_timeout", lambda: 300)  # edited since the leader prompted
    monkeypatch.setattr(wait_mod._ctx, "_fire_approval_hook", lambda name, **kw: None)
    seen = []
    monkeypatch.setattr(wait_mod, "_poll_event",
                        lambda event, session_key, *, interrupt_log, timeout_s=None: seen.append(timeout_s) or "timeout")
    leader = _ApprovalEntry(dict(APPROVAL, request_id="leader", timeout_s=3600))

    wait_mod._await_coalesced_leader(SESSION_KEY, leader, {})

    assert seen == [3600]


def test_leader_publishes_its_timeout_before_the_pre_request_hooks(monkeypatch):
    """A follower can find the leader while its ``pre_approval_request`` hooks run, before the prompt
    is sent; the leader's timeout must already be on the entry by then."""
    _clear()
    monkeypatch.setattr(wait_mod._ctx, "_get_approval_timeout", lambda: 3600)
    published = []

    def hook(name, **kw):
        if name == "pre_approval_request":
            published.append(mod._gateway_queues[SESSION_KEY][0].data.get("timeout_s"))

    monkeypatch.setattr(wait_mod._ctx, "_fire_approval_hook", hook)
    seen = []
    monkeypatch.setattr(wait_mod, "_poll_event",
                        lambda event, session_key, *, interrupt_log, timeout_s=None: seen.append(timeout_s) or "timeout")

    wait_mod._await_gateway_decision(SESSION_KEY, lambda data: None, dict(APPROVAL))

    assert published == [3600]
    assert seen == [3600]


def test_resolve_commits_the_choice_before_releasing_the_approval_lock(monkeypatch):
    """``_drop_entry`` reads ``entry.result`` and leaves the queue in one ``_lock`` section, which only
    closes the race if ``resolve_gateway_approval`` commits ``result``/``event`` INSIDE the section that
    pops the entry. A commit after the lock is released is a window where the waiter pops-and-loses an
    acked choice as a timeout (#112548)."""
    _clear()
    entry = wait_mod._ApprovalEntry(APPROVAL)
    mod._gateway_queues[SESSION_KEY] = [entry]
    seen: dict = {}
    real_lock = mod._lock

    class _Instrumented:
        def __enter__(self):
            return real_lock.__enter__()

        def __exit__(self, *exc):
            seen["result_at_release"], seen["event_at_release"] = entry.result, entry.event.is_set()
            return real_lock.__exit__(*exc)

    monkeypatch.setattr(mod, "_lock", _Instrumented())
    assert mod.resolve_gateway_approval(SESSION_KEY, "once", reason="fine") == 1
    assert seen == {"result_at_release": "once", "event_at_release": True}
    assert entry.reason == "fine"


def test_withdrawn_entry_settles_with_a_wire_reason(monkeypatch):
    """A wait woken with no choice (session teardown) withdraws its open request with a RequestCancelReason,
    not the poll-state token ``"set"``."""
    _clear()
    monkeypatch.setattr(wait_mod._ctx, "_fire_approval_hook", lambda name, **kw: None)
    reasons: list[str] = []

    def torn_down(event, session_key, *, interrupt_log, timeout_s=None):
        mod.register_gateway_settle(session_key, mod._gateway_queues[session_key][0].data["request_id"], reasons.append)
        mod.clear_session(session_key)
        return "set"

    monkeypatch.setattr(wait_mod, "_poll_event", torn_down)
    decision = wait_mod._await_gateway_decision(SESSION_KEY, lambda data: None, APPROVAL)
    assert decision["choice"] is None and decision["cancelled"]
    assert reasons == ["session_closed"]
