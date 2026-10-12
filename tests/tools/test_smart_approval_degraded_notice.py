"""The degraded-state notice for smart approvals.

When the guardian LLM cannot run, the gate falls back to asking a human — intended
fail-safe behaviour, unchanged here. These tests pin the *announcement*: the reason must be
recorded, surfaced to the prompt (gateway payload / queued card), and never invented for a
genuine ESCALATE verdict, which is a decision rather than a degradation.
"""

from types import SimpleNamespace

import pytest

from tools import approval, approval_smart

RELAY_ERROR = "Synchronous Relay LLM execution cannot run on an event-loop thread"


def _response(text: str):
    msg = SimpleNamespace(content=text)
    choice = SimpleNamespace(message=msg, finish_reason="stop")
    return SimpleNamespace(choices=[choice])


def _patch_guardian(monkeypatch, *, raises=None, answers=None, finish_reason="stop"):
    def fake_call_llm(**kwargs):
        if raises is not None:
            raise raises
        r = _response(answers)
        r.choices[0].finish_reason = finish_reason
        return r

    monkeypatch.setattr("agent.auxiliary_client.call_llm", fake_call_llm, raising=False)
    monkeypatch.setattr("agent.auxiliary_client._get_task_timeout", lambda task: 1, raising=False)


def _spec(**over):
    base = {"pending_keys": ["k"], "noun": "command", "smart_log": "smart: {command}",
            "transport": None, "pending_denied": "", "transport_denied": "",
            "gateway_refused": "refused", "notify_failed": "notify failed",
            "user_approved": "", "pending_keys_flag": True, "redact_cli": False}
    base.update(over)
    return SimpleNamespace(**base)


def test_a_failed_guardian_call_records_a_reason(monkeypatch):
    _patch_guardian(monkeypatch, raises=RuntimeError(RELAY_ERROR))
    verdict = approval_smart._smart_verdict("echo hi", "flagged", "k", ["k"], "sess")
    assert verdict == "escalate"
    reason = approval_smart.unavailable_reason()
    assert reason and "RuntimeError" in reason and "event-loop thread" in reason


def test_an_empty_answer_is_a_degradation_too(monkeypatch):
    _patch_guardian(monkeypatch, answers="")
    assert approval_smart._smart_verdict("echo hi", "flagged", "k", ["k"], "sess") == "escalate"
    assert "empty answer" in (approval_smart.unavailable_reason() or "")


def test_a_genuine_escalate_verdict_is_not_a_degradation(monkeypatch):
    _patch_guardian(monkeypatch, answers="ESCALATE")
    assert approval_smart._smart_verdict("echo hi", "flagged", "k", ["k"], "sess") == "escalate"
    assert approval_smart.unavailable_reason() is None


def test_an_approval_verdict_clears_a_previous_reason(monkeypatch):
    _patch_guardian(monkeypatch, raises=RuntimeError(RELAY_ERROR))
    approval_smart._smart_verdict("echo hi", "flagged", "k", ["k"], "sess")
    assert approval_smart.unavailable_reason()
    monkeypatch.undo()
    _patch_guardian(monkeypatch, answers="APPROVE")
    approval_smart._smart_verdict("echo hi", "flagged", "k", ["k"], "sess")
    assert approval_smart.unavailable_reason() is None


def test_the_gate_reports_the_reason_alongside_the_verdict(monkeypatch):
    _patch_guardian(monkeypatch, raises=RuntimeError(RELAY_ERROR))
    result, smart_denied, degraded = approval._smart_gate(
        _spec(), "echo hi", "flagged", "k", ["k"], "sess", human_present=True)
    assert result is None and smart_denied is False
    assert degraded and "RuntimeError" in degraded


def test_the_gate_reports_no_reason_for_a_real_verdict(monkeypatch):
    _patch_guardian(monkeypatch, answers="ESCALATE")
    result, smart_denied, degraded = approval._smart_gate(
        _spec(), "echo hi", "flagged", "k", ["k"], "sess", human_present=True)
    assert result is None and smart_denied is False
    assert degraded is None


def test_the_queued_card_carries_the_notice(monkeypatch):
    submitted = {}
    monkeypatch.setattr(approval, "submit_pending",
                        lambda session_key, pending: submitted.update(pending or {}))
    out = approval._pending_result(
        _spec(), "sess", command="python3 -c 'print(1)'", description="script execution",
        pattern_key="k", pattern_keys=["k"], body=None, smart_denied=False,
        smart_unavailable="RuntimeError: " + RELAY_ERROR)
    assert out["smart_unavailable"].startswith("RuntimeError")
    assert "why you're being asked" in out["message"]
    assert submitted.get("smart_unavailable")


def test_the_queued_card_is_unchanged_when_the_guardian_ran(monkeypatch):
    submitted = {}
    monkeypatch.setattr(approval, "submit_pending",
                        lambda session_key, pending: submitted.update(pending or {}))
    out = approval._pending_result(
        _spec(), "sess", command="echo hi", description="flagged",
        pattern_key="k", pattern_keys=["k"], body=None, smart_denied=False)
    assert "smart_unavailable" not in out
    assert "why you're being asked" not in out["message"]
    assert "smart_unavailable" not in submitted


def _verdict_from_route(monkeypatch, provider, model, answer="ESCALATE"):
    """A fake guardian that records the route it "used" into route_info, like call_llm does."""
    def fake_call_llm(**kwargs):
        route = kwargs.get("route_info")
        if route is not None:
            route.update({"provider": provider, "model": model})
        return _response(answer)

    monkeypatch.setattr("agent.auxiliary_client.call_llm", fake_call_llm, raising=False)
    monkeypatch.setattr("agent.auxiliary_client._get_task_timeout", lambda task: 1, raising=False)


def test_a_verdict_from_a_fallback_route_is_announced(monkeypatch):
    """The verdict is real, but the model behind it is not the configured one - say whose it was."""
    monkeypatch.setattr(approval_smart, "_expected_route",
                        lambda: ("modelark-coding", "deepseek-v4-flash"))
    _verdict_from_route(monkeypatch, "deepseek", "deepseek-v4-flash")
    assert approval_smart._smart_verdict("echo hi", "flagged", "k", ["k"], "sess") == "escalate"
    reason = approval_smart.unavailable_reason() or ""
    assert "fallback route" in reason and "deepseek" in reason and "modelark-coding" in reason


def test_a_verdict_on_the_expected_route_is_not_announced(monkeypatch):
    monkeypatch.setattr(approval_smart, "_expected_route",
                        lambda: ("modelark-coding", "deepseek-v4-flash"))
    _verdict_from_route(monkeypatch, "modelark-coding", "deepseek-v4-flash")
    assert approval_smart._smart_verdict("echo hi", "flagged", "k", ["k"], "sess") == "escalate"
    assert approval_smart.unavailable_reason() is None


def test_manual_mode_says_no_model_was_asked(monkeypatch):
    """Manual mode never consults a model: the note must say exactly that.

    The note travels the same channel as the guardian-degradation note asserted above; this
    pins the text, which is the part manual mode adds.
    """
    monkeypatch.setattr(approval.approval_context, "_get_approval_mode", lambda: "manual")
    assert approval._no_model_note() == (
        "approvals mode is 'manual' - no model verdict was requested for this command")
