"""Where the degraded-smart-approval notice must appear.

The notice is announcement-only: it never changes a decision, and the fall-back to a human
prompt is deliberate. These tests pin each *surface* that renders a prompt, so the degradation
cannot be silent on one transport while visible on another.
"""

from types import SimpleNamespace

import pytest

from tools import approval, approval_prompt


def test_the_gateway_text_fallback_carries_the_notice():
    from gateway.run import _format_exec_approval_fallback

    with_notice = _format_exec_approval_fallback(
        "rm -rf /tmp/x", "recursive delete", "/", smart_unavailable="RuntimeError: relay refused")
    assert "smart approval could not run" in with_notice
    assert "RuntimeError: relay refused" in with_notice

    without = _format_exec_approval_fallback("rm -rf /tmp/x", "recursive delete", "/")
    assert "smart approval could not run" not in without


def test_the_cli_panel_prints_the_notice(capsys, monkeypatch):
    monkeypatch.setattr(approval_prompt._ctx, "_get_approval_timeout", lambda: 5)
    monkeypatch.setattr(approval_prompt, "_read_choice", lambda prompt, timeout: "deny")
    choice = approval_prompt.prompt_dangerous_approval(
        "echo hi", "flagged", timeout_seconds=5, smart_unavailable="RuntimeError: relay refused")
    out = capsys.readouterr().out
    assert choice == "deny"
    assert "smart approval could not run (RuntimeError: relay refused)" in out


def test_a_panel_callback_that_accepts_the_keyword_receives_it(monkeypatch):
    seen = {}

    def panel(command, description, *, allow_permanent=True, smart_unavailable=None):
        seen["smart_unavailable"] = smart_unavailable
        return "deny"

    monkeypatch.setattr(approval_prompt._ctx, "_get_approval_timeout", lambda: 5)
    assert approval_prompt.prompt_dangerous_approval(
        "echo hi", "flagged", timeout_seconds=5, approval_callback=panel,
        smart_unavailable="RuntimeError: relay refused") == "deny"
    assert seen["smart_unavailable"] == "RuntimeError: relay refused"


def test_a_legacy_callback_is_not_handed_the_new_keyword(monkeypatch):
    seen = {}

    # A genuinely legacy callback: explicit parameters, no **kwargs. (callbacks declared with
    # **kwargs are treated as accepting any keyword — that is the existing contract for `title`
    # too, so it is not the legacy case.)
    def legacy(command, description, *, allow_permanent=True):
        seen["allow_permanent"] = allow_permanent
        return "deny"

    monkeypatch.setattr(approval_prompt._ctx, "_get_approval_timeout", lambda: 5)
    assert approval_prompt.prompt_dangerous_approval(
        "echo hi", "flagged", timeout_seconds=5, approval_callback=legacy,
        smart_unavailable="RuntimeError: relay refused") == "deny"
    assert seen == {"allow_permanent": True}


def test_the_queued_card_keeps_the_notice_out_when_the_guardian_ran(monkeypatch):
    submitted = {}
    monkeypatch.setattr(approval, "submit_pending",
                        lambda session_key, pending: submitted.update(pending or {}))
    spec = SimpleNamespace(pending_keys=["k"], noun="command", smart_log="", transport=None,
                           pending_denied="", transport_denied="", gateway_refused="r",
                           notify_failed="n", user_approved="")
    out = approval._pending_result(spec, "sess", command="echo hi", description="flagged",
                                   pattern_key="k", pattern_keys=["k"], body=None,
                                   smart_denied=False, smart_unavailable=None)
    assert "smart_unavailable" not in out and "smart_unavailable" not in submitted
    assert "smart approval could not run" not in out["message"]
