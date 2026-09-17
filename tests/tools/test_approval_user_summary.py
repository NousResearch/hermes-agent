"""Approval and background-task outcomes carry one plain sentence for the person at the keyboard.

The model-facing ``BLOCKED: ... Do NOT retry`` text is unchanged; surfaces render ``user_summary``
first (UX message audit, findings tools-runtime-15, -26, -31).
"""

from __future__ import annotations

import json


def test_denied_result_carries_a_user_summary_beside_the_model_text(monkeypatch):
    from tools import approval
    monkeypatch.setattr("tools.approval_context._get_approval_timeout", lambda: 300)
    res = approval._denied("BLOCKED: User denied this command. Do NOT retry.", pattern_key="rm",
                           description="delete files", outcome="denied")
    assert res["message"].startswith("BLOCKED")  # model contract untouched
    assert res["user_summary"] == "You denied this command — it did not run."
    timed = approval._denied("BLOCKED: timed out", pattern_key="rm", description="d", outcome="timeout", noun="code")
    assert "5 minutes" in timed["user_summary"] and "did not run" in timed["user_summary"]
    assert "BLOCKED" not in timed["user_summary"] and "NOT" not in timed["user_summary"]


def test_cli_completion_line_prefers_user_summary_over_blocked_text():
    from agent.display import _detect_tool_failure
    result = json.dumps({"output": "", "exit_code": -1, "status": "blocked",
                         "error": "BLOCKED: User denied this command. The user has NOT consented ...",
                         "user_summary": "You denied this command — it did not run."})
    failed, suffix = _detect_tool_failure("terminal", result)
    assert failed and "You denied this command" in suffix and "BLOCKED" not in suffix


def test_timeout_notice_names_wait_and_config_command(monkeypatch):
    from agent.i18n import t
    from tools import approval_context as ctx
    monkeypatch.setattr(ctx, "_get_approval_timeout", lambda: 300)
    text = t("approval.timeout", **ctx.approval_timeout_notice_kwargs())
    assert "5 minutes" in text and "was not run" in text
    assert "hermes config set approvals.timeout 900" in text
    assert "{" not in text  # every placeholder filled


def test_stalled_delegation_text_has_no_issue_numbers_or_worker_jargon():
    from tools.async_delegation import _stalled_error_text
    text = _stalled_error_text({"goal": "Summarise the  weekly report", "_stall_quiet_seconds": 610.0})
    assert text.startswith('Background task "Summarise the weekly report"')
    assert "10 min" in text and "run it again" in text
    for banned in ("#60203", "completion event", "wedged", "Re-dispatch", "delegation"):
        assert banned not in text


def test_prompt_title_reaches_the_plain_prompt_and_only_title_aware_callbacks(monkeypatch, capsys):
    from tools import approval_prompt as ap
    seen = {}

    def legacy_cb(command, description, *, allow_permanent=True):
        seen["legacy"] = (command, description)
        return "once"

    def aware_cb(command, description, *, allow_permanent=True, title=None):
        seen["title"] = title
        return "once"

    assert ap.prompt_dangerous_approval("x", "d", approval_callback=legacy_cb, title="Save to memory?") == "once"
    assert ap.prompt_dangerous_approval("x", "d", approval_callback=aware_cb, title="Save to memory?") == "once"
    assert seen["title"] == "Save to memory?"

    monkeypatch.setattr(ap, "_read_choice", lambda *_a, **_k: "d")
    monkeypatch.setattr("prompt_toolkit.application.current.get_app_or_none", lambda: None)
    ap.prompt_dangerous_approval("x", "d", title="MCP server 'gh' is asking")
    out = capsys.readouterr().out
    assert "MCP server 'gh' is asking" in out and "DANGEROUS COMMAND" not in out


def test_every_surface_renders_the_timeout_window_with_one_formatter(monkeypatch):
    """CLI notice, tool user_summary and gateway card must agree on the wording of the same window."""
    from tools import approval, approval_context as ctx
    from gateway.platforms.base_exec_approval import format_approval_timed_out_notice
    monkeypatch.setattr(ctx, "_get_approval_timeout", lambda: 90)
    window = ctx.format_approval_window(90)
    assert window in ctx.approval_timeout_notice_kwargs()["waited"]
    assert window in approval._user_summary("timeout")
    assert window in format_approval_timed_out_notice(90)
