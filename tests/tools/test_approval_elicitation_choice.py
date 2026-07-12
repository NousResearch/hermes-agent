"""Approval-scope contracts for MCP elicitations."""

from unittest.mock import patch

from tools.approval import request_elicitation_consent, request_elicitation_choice


def test_elicitation_choice_preserves_cli_always_scope():
    with (
        patch("tools.approval.get_current_session_key", return_value="test"),
        patch("tools.approval._is_gateway_approval_context", return_value=False),
        patch("tools.approval.sys.stdin.isatty", return_value=True),
        patch("tools.approval.prompt_dangerous_approval", return_value="always") as prompt,
    ):
        choice = request_elicitation_choice(
            "Allow Computer Use to use Finder?",
            "Computer Use app access",
            allow_permanent=True,
        )

    assert choice == "always"
    prompt.assert_called_once_with(
        "Allow Computer Use to use Finder?",
        "Computer Use app access",
        timeout_seconds=None,
        allow_permanent=True,
    )


def test_elicitation_choice_headless_cli_fails_closed_without_prompting():
    with (
        patch("tools.approval.get_current_session_key", return_value="headless"),
        patch("tools.approval._is_gateway_approval_context", return_value=False),
        patch("tools.approval.sys.stdin.isatty", return_value=False),
        patch(
            "tools.approval.prompt_dangerous_approval",
            side_effect=AssertionError("headless context must not prompt"),
        ),
    ):
        assert request_elicitation_choice("confirm", "description") == "deny"


def test_elicitation_choice_propagates_permanent_flag_to_gateway_payload():
    from tools import approval

    captured = []
    session_key = "gateway-approval-scope"
    approval._gateway_notify_cbs[session_key] = lambda data: None
    try:
        with (
            patch("tools.approval.get_current_session_key", return_value=session_key),
            patch("tools.approval._is_gateway_approval_context", return_value=True),
            patch(
                "tools.approval._await_gateway_decision",
                side_effect=lambda _key, _cb, data, **_kw: (
                    captured.append(data)
                    or {"resolved": True, "choice": "session"}
                ),
            ),
        ):
            choice = request_elicitation_choice(
                "confirm",
                "description",
                allow_permanent=False,
            )
    finally:
        approval._gateway_notify_cbs.pop(session_key, None)

    assert choice == "session"
    assert captured == [
        {
            "command": "confirm",
            "description": "description",
            "pattern_key": "mcp_elicitation",
            "pattern_keys": ["mcp_elicitation"],
            "allow_permanent": False,
        }
    ]


def test_gateway_resolver_rejects_hidden_permanent_choice():
    from tools import approval

    session_key = "gateway-no-permanent"
    entry = approval._ApprovalEntry({"allow_permanent": False})
    approval._gateway_queues[session_key] = [entry]
    try:
        assert approval.resolve_gateway_approval(session_key, "always") == 0
        assert approval.has_blocking_approval(session_key) is True
        assert entry.event.is_set() is False

        assert approval.resolve_gateway_approval(session_key, "session") == 1
        assert entry.result == "session"
        assert entry.event.is_set() is True
    finally:
        approval._gateway_queues.pop(session_key, None)


def test_legacy_elicitation_api_normalizes_scoped_approval_to_accept():
    with patch(
        "tools.approval.request_elicitation_choice", return_value="session"
    ):
        assert request_elicitation_consent("confirm", "description") == "accept"
