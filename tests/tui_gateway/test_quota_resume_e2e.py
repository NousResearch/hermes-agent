"""End-to-end: a provider quota wall arms a backend resume; nothing else does.

Exercises the real chain — classifier verdict -> turn result -> gateway payload ->
armed session state -> poller dispatch — with the actual error payloads captured
from Anthropic and the ChatGPT/Codex backend, not hand-written approximations.

The gateway helpers are rebound onto ``tui_gateway.server``'s namespace at import
(method_ctx.bind_module), so they are patched and called *there*.
"""

from __future__ import annotations

import time
from typing import Any, Optional

import pytest

# Captured verbatim from a real ChatGPT/Codex 429 (request dump, account scrubbed).
CODEX_QUOTA_BODY = {
    "type": "usage_limit_reached",
    "message": "The usage limit has been reached",
    "plan_type": "prolite",
    "resets_at": 1788646607,
    "resets_in_seconds": 510651,
}
# Captured verbatim from Anthropic: an SSE error inside an HTTP 200 stream.
ANTHROPIC_OVERLOAD_BODY = {
    "type": "error",
    "error": {"details": None, "type": "overloaded_error", "message": "Overloaded"},
}


class _FakeError(Exception):
    """Provider SDK error shape that extract_api_error_context understands."""

    def __init__(self, body, message="", status_code=None):
        super().__init__(message or str(body))
        self.body = body
        self.status_code = status_code
        self.response = None


@pytest.fixture()
def server():
    import tui_gateway.server as server_mod

    return server_mod


def _session(**over) -> dict:
    import threading

    session = {"history_lock": threading.RLock(), "running": False}
    session.update(over)
    return session


# ── error context -> plan (real extractor, real payloads) ─────────────────────

def test_captured_codex_quota_payload_yields_a_deadline():
    from agent.agent_runtime_helpers import extract_api_error_context
    from agent.quota_resume import SOURCE_PROVIDER_ERROR, plan_quota_resume

    ctx = extract_api_error_context(_FakeError(CODEX_QUOTA_BODY, status_code=429))
    assert ctx["reset_at"] == 1788646607

    plan = plan_quota_resume(
        failure_reason="rate_limit", error_context=ctx, provider="openai-codex",
        now=1788646607 - 3600, grace_seconds=0.0, allow_usage_api=False,
    )
    assert (plan.eligible, plan.source) == (True, SOURCE_PROVIDER_ERROR)
    assert plan.resume_at == 1788646607


def test_captured_anthropic_overload_payload_yields_no_plan():
    """The exact failure this feature must NOT fire on: capacity, not quota."""
    from agent.agent_runtime_helpers import extract_api_error_context
    from agent.quota_resume import plan_quota_resume

    ctx = extract_api_error_context(_FakeError(ANTHROPIC_OVERLOAD_BODY, status_code=200))
    assert "reset_at" not in ctx

    plan = plan_quota_resume(
        failure_reason="overloaded", error_context=ctx, provider="anthropic",
        allow_usage_api=False,
    )
    assert plan.eligible is False


# ── terminal turn result carries the plan ─────────────────────────────────────

def test_max_retries_result_attaches_quota_resume(monkeypatch):
    """The terminal result is where the deadline stops being thrown away."""
    from agent.error_classifier import FailoverReason
    import agent.turn_recovery as turn_recovery

    reset_at = time.time() + 1800

    class _Classified:
        reason = FailoverReason.rate_limit
        retryable = True
        billing_unverified = False

    plan = turn_recovery._quota_resume_plan(
        agent=type("A", (), {"_credential_pool": None})(),
        classified=_Classified(),
        error_context={"reset_at": reset_at},
        provider="anthropic",
    )
    assert plan is not None
    assert plan["eligible"] is True
    assert plan["provider"] == "anthropic"
    assert plan["resume_at"] == pytest.approx(reset_at + 45.0, abs=1.0)


def test_max_retries_result_omits_plan_for_overload():
    from agent.error_classifier import FailoverReason
    import agent.turn_recovery as turn_recovery

    class _Classified:
        reason = FailoverReason.overloaded
        retryable = True
        billing_unverified = False

    assert turn_recovery._quota_resume_plan(
        agent=type("A", (), {"_credential_pool": None})(),
        classified=_Classified(), error_context={}, provider="anthropic",
    ) is None


# ── arming ────────────────────────────────────────────────────────────────────

def test_arm_stores_due_time_and_prompt(server, monkeypatch):
    monkeypatch.setattr(server, "_quota_resume_config", lambda: (True, 2))
    monkeypatch.setattr(server, "_emit", lambda *a, **k: None)
    session = _session()
    resume_at = time.time() + 600

    armed = server._arm_quota_resume(
        "sid-1", session,
        {"eligible": True, "resume_at": resume_at, "source": "provider_error", "provider": "anthropic"},
        "finish the migration",
    )
    assert armed is True
    assert session["_quota_resume_at"] == resume_at
    assert session["_quota_resume_prompt"] == "finish the migration"


@pytest.mark.parametrize(
    "plan, prompt, reason",
    [
        ({"eligible": False, "resume_at": time.time() + 600}, "p", "ineligible plan"),
        ({"eligible": True}, "p", "no resume_at"),
        ({"eligible": True, "resume_at": time.time() + 600}, "   ", "empty prompt"),
        ({"eligible": True, "resume_at": "later"}, "p", "non-numeric resume_at"),
    ],
)
def test_arm_refuses_unusable_input(server, monkeypatch, plan, prompt, reason):
    monkeypatch.setattr(server, "_quota_resume_config", lambda: (True, 2))
    monkeypatch.setattr(server, "_emit", lambda *a, **k: None)
    session = _session()
    assert server._arm_quota_resume("sid-1", session, plan, prompt) is False, reason
    assert "_quota_resume_at" not in session


def test_arm_respects_disabled_config(server, monkeypatch):
    monkeypatch.setattr(server, "_quota_resume_config", lambda: (False, 2))
    monkeypatch.setattr(server, "_emit", lambda *a, **k: None)
    session = _session()
    assert server._arm_quota_resume(
        "sid-1", session, {"eligible": True, "resume_at": time.time() + 60}, "p") is False


def test_arm_refuses_hosted_room_session(server, monkeypatch):
    """Hosted rooms recover through their own durable lease machine."""
    monkeypatch.setattr(server, "_quota_resume_config", lambda: (True, 2))
    monkeypatch.setattr(server, "_emit", lambda *a, **k: None)
    session = _session(source="bot_room")
    assert server._arm_quota_resume(
        "sid-1", session, {"eligible": True, "resume_at": time.time() + 60}, "p") is False


def test_arm_stops_after_max_attempts(server, monkeypatch):
    monkeypatch.setattr(server, "_quota_resume_config", lambda: (True, 2))
    monkeypatch.setattr(server, "_emit", lambda *a, **k: None)
    session = _session(_quota_resume_attempt=2)
    assert server._arm_quota_resume(
        "sid-1", session, {"eligible": True, "resume_at": time.time() + 60}, "p") is False


# ── firing ────────────────────────────────────────────────────────────────────

def _capture_submit(server, monkeypatch) -> list:
    submitted: list = []
    monkeypatch.setattr(server, "_emit", lambda *a, **k: None)
    monkeypatch.setattr(server, "_ensure_active_session_slot", lambda sid, s: None)
    monkeypatch.setattr(
        server, "_run_prompt_submit",
        lambda rid, sid, session, text, **kw: submitted.append({"text": text, **kw}))
    return submitted


def test_fires_when_due_and_carries_the_original_prompt(server, monkeypatch):
    submitted = _capture_submit(server, monkeypatch)
    session = _session(
        _quota_resume_at=time.time() - 1,
        _quota_resume_prompt="rerun the failing suite",
        _quota_resume_meta={"provider": "anthropic"},
    )
    server._maybe_fire_quota_resume("sid-1", session)

    assert len(submitted) == 1
    assert "rerun the failing suite" in submitted[0]["text"]
    assert submitted[0]["display_kind"] == "auto_continue"
    # Pending state is consumed exactly once.
    assert "_quota_resume_at" not in session
    assert session["_quota_resume_attempt"] == 1


def test_does_not_fire_before_the_deadline(server, monkeypatch):
    submitted = _capture_submit(server, monkeypatch)
    session = _session(_quota_resume_at=time.time() + 600, _quota_resume_prompt="later")
    server._maybe_fire_quota_resume("sid-1", session)
    assert submitted == []
    assert session["_quota_resume_at"]  # still armed


def test_running_session_cancels_the_pending_resume(server, monkeypatch):
    """A live turn means the user is driving; the stale resume is dropped."""
    submitted = _capture_submit(server, monkeypatch)
    session = _session(running=True, _quota_resume_at=time.time() - 1, _quota_resume_prompt="p")
    server._maybe_fire_quota_resume("sid-1", session)
    assert submitted == []
    assert "_quota_resume_at" not in session


def test_finalized_session_never_fires(server, monkeypatch):
    submitted = _capture_submit(server, monkeypatch)
    session = _session(_finalized=True, _quota_resume_at=time.time() - 1, _quota_resume_prompt="p")
    server._maybe_fire_quota_resume("sid-1", session)
    assert submitted == []


def test_foreign_owner_defers_and_releases(server, monkeypatch):
    """A sibling backend on the same HERMES_HOME owns the session — never double-write."""
    submitted = _capture_submit(server, monkeypatch)
    monkeypatch.setattr(server, "_ensure_active_session_slot", lambda sid, s: {"owner": "other"})
    session = _session(_quota_resume_at=time.time() - 1, _quota_resume_prompt="p")
    server._maybe_fire_quota_resume("sid-1", session)
    assert submitted == []
    assert session["running"] is False  # released, not wedged


def test_unarmed_session_is_a_no_op(server, monkeypatch):
    submitted = _capture_submit(server, monkeypatch)
    server._maybe_fire_quota_resume("sid-1", _session())
    assert submitted == []


def test_dispatch_failure_releases_the_turn(server, monkeypatch):
    monkeypatch.setattr(server, "_emit", lambda *a, **k: None)
    monkeypatch.setattr(server, "_ensure_active_session_slot", lambda sid, s: None)

    def _boom(*a, **k):
        raise RuntimeError("submit exploded")

    monkeypatch.setattr(server, "_run_prompt_submit", _boom)
    session = _session(_quota_resume_at=time.time() - 1, _quota_resume_prompt="p")
    server._maybe_fire_quota_resume("sid-1", session)
    assert session["running"] is False  # not left claimed forever


# ── cancellation ──────────────────────────────────────────────────────────────

def test_clear_removes_all_pending_state(server):
    session = _session(
        _quota_resume_at=1.0, _quota_resume_prompt="p", _quota_resume_meta={"provider": "x"})
    server._clear_quota_resume(session)
    for key in ("_quota_resume_at", "_quota_resume_prompt", "_quota_resume_meta"):
        assert key not in session


def test_config_defaults_expose_a_settings_toggle():
    """Desktop Settings renders booleans from DEFAULT_CONFIG; the toggle must exist."""
    from hermes_cli.config_defaults import DEFAULT_CONFIG

    cfg = DEFAULT_CONFIG["desktop"]["quota_resume"]
    assert cfg["enabled"] is True
    assert isinstance(cfg["max_attempts"], int)
