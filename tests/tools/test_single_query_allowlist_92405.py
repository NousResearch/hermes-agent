from tools import approval
from tools import tirith_security


def _configure_single_query(monkeypatch, *, approved: bool) -> None:
    monkeypatch.setattr(approval, "_YOLO_MODE_FROZEN", False)
    monkeypatch.setattr(approval, "_should_skip_container_guards", lambda *args, **kwargs: False)
    monkeypatch.setattr(approval, "detect_hardline_command", lambda _cmd: (False, None))
    monkeypatch.setattr(approval, "_check_sudo_stdin_guard", lambda _cmd: (False, None))
    monkeypatch.setattr(approval, "_match_user_deny_rule", lambda _cmd: None)
    monkeypatch.setattr(approval, "_get_approval_mode", lambda: "smart")
    monkeypatch.setattr(approval, "is_current_session_yolo_enabled", lambda: False)
    monkeypatch.setattr(approval, "_command_matches_permanent_allowlist", lambda _cmd: False)
    monkeypatch.setattr(approval, "_resolve_cli_approval_callback", lambda callback: callback)
    monkeypatch.setattr(approval, "_is_interactive_cli", lambda: False)
    monkeypatch.setattr(approval, "_is_gateway_approval_context", lambda: False)
    monkeypatch.setattr(approval, "_is_single_query_approval_context", lambda: True)
    monkeypatch.setattr(approval, "_get_single_query_approval_mode", lambda: "deny")
    monkeypatch.setattr(approval, "_is_cron_approval_context", lambda: False)
    monkeypatch.setattr(approval, "get_current_session_key", lambda: "sq-session")
    monkeypatch.setattr(
        approval,
        "detect_dangerous_command",
        lambda _cmd: (True, "script execution via heredoc", "script execution via heredoc"),
    )
    monkeypatch.setattr(
        approval,
        "is_approved",
        lambda session_key, pattern_key: (
            approved
            and session_key == "sq-session"
            and pattern_key == "script execution via heredoc"
        ),
    )
    monkeypatch.setattr(tirith_security, "check_command_security", lambda _cmd: {"action": "allow"})


def test_single_query_deny_honors_pattern_allowlist(monkeypatch):
    _configure_single_query(monkeypatch, approved=True)

    result = approval.check_all_command_guards("python3 <<EOF\nprint(1)\nEOF", "local")

    assert result == {"approved": True, "message": None}


def test_single_query_deny_still_blocks_unapproved_pattern(monkeypatch):
    _configure_single_query(monkeypatch, approved=False)

    result = approval.check_all_command_guards("python3 <<EOF\nprint(1)\nEOF", "local")

    assert result["approved"] is False
    assert result["pattern_key"] == "script execution via heredoc"
    assert "single-query mode" in result["message"]
