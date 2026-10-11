"""Degrading smart approvals when the guardian LLM is unreachable.

The bug this covers: in ``smart`` mode, a guardian call that fails (429, timeout,
provider down, empty body) returns the verdict 'escalate', which means "ask the
human". On a surface where the human cannot see the prompt, the request sits
until ``approvals.timeout`` and then fails closed -- the agent is blocked for
minutes and the user was never asked anything. Reported as "I got an approval
timeout message but I wasn't seeing any active request."

Two properties, and the second is the one that matters:

  1. After N consecutive guardian FAILURES the session degrades to manual, so
     flagged commands are asked directly instead of vanishing into a wait.
  2. A guardian that merely says ESCALATE is a WORKING reviewer and must never
     count toward that -- otherwise a cautious-but-healthy reviewer silently
     disables itself, which is the same class of bug in the other direction.

Any real answer (approve/deny/escalate) clears the run, so recovery is automatic.
"""

import pytest

from tools import approval_context as ctx
from tools import approval_smart as smart


@pytest.fixture(autouse=True)
def _clean_tally():
    """Each test starts with no accumulated failures for the current session."""
    ctx.mark_smart_success()
    yield
    ctx.mark_smart_success()


def _fail(monkeypatch, times=1):
    """Drive the real failure path `times` times, as an unreachable guardian would."""
    def boom(*_a, **_k):
        raise RuntimeError("429 rate limited")
    monkeypatch.setattr(smart, "_smart_approve", smart._smart_approve)  # keep real fn
    monkeypatch.setattr(ctx, "mark_smart_failure", ctx.mark_smart_failure)
    for _ in range(times):
        smart._ctx.mark_smart_failure()


class TestDegradation:
    def test_starts_smart_and_stays_smart_below_threshold(self):
        key = ctx.get_current_session_key()
        assert ctx._effective_approval_mode(key) == "smart"
        smart._ctx.mark_smart_failure()
        assert ctx._effective_approval_mode(key) == "smart", "one failure must not degrade"

    def test_degrades_to_manual_at_threshold(self):
        key = ctx.get_current_session_key()
        threshold = ctx._get_smart_failure_threshold()
        for _ in range(threshold - 1):
            smart._ctx.mark_smart_failure()
        assert ctx._effective_approval_mode(key) == "smart"
        smart._ctx.mark_smart_failure()
        assert ctx._effective_approval_mode(key) == "manual", (
            f"{threshold} consecutive guardian failures must degrade to manual"
        )

    def test_recovers_when_the_guardian_answers_again(self):
        key = ctx.get_current_session_key()
        for _ in range(ctx._get_smart_failure_threshold()):
            smart._ctx.mark_smart_failure()
        assert ctx._effective_approval_mode(key) == "manual"
        ctx.mark_smart_success()
        assert ctx._effective_approval_mode(key) == "smart", (
            "degradation must be self-healing, not sticky"
        )

    def test_threshold_zero_disables_degradation(self, monkeypatch):
        monkeypatch.setattr(ctx, "_get_smart_failure_threshold", lambda: 0)
        key = ctx.get_current_session_key()
        for _ in range(10):
            smart._ctx.mark_smart_failure()
        assert ctx._effective_approval_mode(key) == "smart"

    def test_non_smart_mode_is_untouched(self, monkeypatch):
        monkeypatch.setattr(ctx, "_get_approval_mode", lambda: "manual")
        key = ctx.get_current_session_key()
        for _ in range(10):
            smart._ctx.mark_smart_failure()
        assert ctx._effective_approval_mode(key) == "manual"

    def test_failures_are_per_session(self):
        key = ctx.get_current_session_key()
        for _ in range(ctx._get_smart_failure_threshold()):
            smart._record_smart_failure("some-other-session")
        assert ctx._effective_approval_mode(key) == "smart", (
            "another session's failures must not degrade this one"
        )


class TestOnlyInfrastructureCounts:
    """The distinction the whole design rests on: unreachable != uncertain."""

    def test_reachable_escalate_does_not_count(self, monkeypatch):
        """A guardian that answers ESCALATE is working. It must not degrade anything."""
        class _Choice:
            message = type("M", (), {"content": "ESCALATE"})()
            finish_reason = "stop"

        class _Resp:
            choices = [_Choice()]

        import agent.auxiliary_client as aux
        monkeypatch.setattr(aux, "call_llm", lambda **_k: _Resp(), raising=False)
        monkeypatch.setattr(aux, "_get_task_timeout", lambda _t: 30, raising=False)
        key = ctx.get_current_session_key()
        for _ in range(ctx._get_smart_failure_threshold() + 2):
            assert smart._smart_approve("rm -rf /", "recursive delete") == "escalate"
        assert ctx._effective_approval_mode(key) == "smart", (
            "a cautious but reachable reviewer must never disable itself"
        )

    def test_unreachable_guardian_counts(self, monkeypatch):
        import agent.auxiliary_client as aux

        def _boom(**_k):
            raise RuntimeError("429 rate limited")

        monkeypatch.setattr(aux, "call_llm", _boom, raising=False)
        monkeypatch.setattr(aux, "_get_task_timeout", lambda _t: 30, raising=False)
        key = ctx.get_current_session_key()
        threshold = ctx._get_smart_failure_threshold()
        for _ in range(threshold):
            assert smart._smart_approve("echo hi", "flagged") == "escalate"
        assert ctx._effective_approval_mode(key) == "manual"

    def test_empty_body_counts_as_failure(self, monkeypatch):
        """A 200 with no verdict is an infrastructure failure, not a verdict."""
        class _Choice:
            message = type("M", (), {"content": "   "})()
            finish_reason = "length"

        class _Resp:
            choices = [_Choice()]

        import agent.auxiliary_client as aux
        monkeypatch.setattr(aux, "call_llm", lambda **_k: _Resp(), raising=False)
        monkeypatch.setattr(aux, "_get_task_timeout", lambda _t: 30, raising=False)
        key = ctx.get_current_session_key()
        for _ in range(ctx._get_smart_failure_threshold()):
            smart._smart_approve("echo hi", "flagged")
        assert ctx._effective_approval_mode(key) == "manual"


class TestSurfacing:
    """Degrading silently would be worse than not degrading: the user would read
    'no prompt' as 'it was approved'."""

    def test_no_notice_before_threshold(self):
        assert ctx.smart_approval_failure_notice() == ""

    def test_notice_after_threshold_names_the_cause(self):
        for _ in range(ctx._get_smart_failure_threshold()):
            smart._ctx.mark_smart_failure()
        notice = ctx.smart_approval_failure_notice()
        assert notice, "a degraded session must say so"
        assert "UNREACHABLE" in notice
        assert "MANUAL" in notice
        assert "Nothing here was auto-approved" in notice, (
            "the notice must pre-empt the 'it was approved' reading"
        )
        assert "fallback_providers" in notice, "it must name the fix"

    def test_notice_clears_on_recovery(self):
        for _ in range(ctx._get_smart_failure_threshold()):
            smart._ctx.mark_smart_failure()
        assert ctx.smart_approval_failure_notice() != ""
        ctx.mark_smart_success()
        assert ctx.smart_approval_failure_notice() == ""


class TestNeverBreaksTheGate:
    """These are called on the approval path. A guard that raises when the config is
    broken turns 'reviewer unavailable' into 'approval gate crashes'."""

    def test_config_read_failure_does_not_raise(self, monkeypatch):
        monkeypatch.setattr(ctx, "_get_approval_config",
                            lambda: (_ for _ in ()).throw(RuntimeError("config unreadable")))
        assert ctx._get_smart_failure_threshold() == 3
        ctx.mark_smart_failure()
        ctx.mark_smart_success()
        assert ctx.smart_approval_failure_notice() == ""
        assert ctx._effective_approval_mode(ctx.get_current_session_key()) in {"smart", "manual"}

    def test_garbage_threshold_falls_back_to_default(self, monkeypatch):
        monkeypatch.setattr(ctx, "_get_approval_config", lambda: {"smart_failure_threshold": "lots"})
        assert ctx._get_smart_failure_threshold() == 3

    def test_tally_is_bounded(self):
        for i in range(smart._SMART_FAILURE_TALLY_MAX_SESSIONS + 50):
            smart._record_smart_failure(f"session-{i}")
        assert len(smart._smart_failures) <= smart._SMART_FAILURE_TALLY_MAX_SESSIONS
