"""Task 11 authenticated UI smoke acceptance contract tests."""

from __future__ import annotations


def test_authenticated_ui_smoke_pending_when_auth_is_unavailable():
    from tools.browser_tool import build_authenticated_ui_smoke_contract

    result = build_authenticated_ui_smoke_contract(
        auth_available=False,
        title="Cloudflare Access",
        body_text="Sign in with the configured identity provider to continue.",
        required_body_terms=["Orders", "Site Factory"],
        required_shell_terms=["MAPA CRM", "Dashboard"],
        forbidden_first_viewport_terms=["terminal", "shell command", "deploy log"],
    )

    assert result["status"] == "authenticated_ui_smoke_pending"
    assert result["authenticated_ui_acceptance"] is False
    assert result["body_verified"] is False
    assert result["shared_shell_verified"] is False
    assert result["leftover_technical_terms"] == []


def test_authenticated_ui_smoke_requires_target_body_and_shared_shell_terms():
    from tools.browser_tool import build_authenticated_ui_smoke_contract

    result = build_authenticated_ui_smoke_contract(
        auth_available=True,
        title="MAPA CRM",
        body_text="MAPA CRM Dashboard Site Factory Orders Ready for review.",
        required_body_terms=["Orders", "Site Factory"],
        required_shell_terms=["MAPA CRM", "Dashboard"],
        forbidden_first_viewport_terms=["terminal", "shell command", "deploy log"],
    )

    assert result["status"] == "authenticated_ui_smoke_passed"
    assert result["authenticated_ui_acceptance"] is True
    assert result["body_verified"] is True
    assert result["shared_shell_verified"] is True
    assert result["leftover_technical_terms"] == []


def test_authenticated_ui_smoke_fails_on_leftover_technical_shell_copy():
    from tools.browser_tool import build_authenticated_ui_smoke_contract

    result = build_authenticated_ui_smoke_contract(
        auth_available=True,
        title="MAPA CRM",
        body_text=(
            "MAPA CRM Dashboard Site Factory Orders Ready for review. "
            "terminal shell command deploy log"
        ),
        required_body_terms=["Orders", "Site Factory"],
        required_shell_terms=["MAPA CRM", "Dashboard"],
        forbidden_first_viewport_terms=["terminal", "shell command", "deploy log"],
    )

    assert result["status"] == "authenticated_ui_smoke_failed"
    assert result["authenticated_ui_acceptance"] is False
    assert result["body_verified"] is True
    assert result["shared_shell_verified"] is True
    assert result["leftover_technical_terms"] == [
        "terminal",
        "shell command",
        "deploy log",
    ]
