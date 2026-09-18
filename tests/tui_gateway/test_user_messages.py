"""Behaviour contracts for the user-facing copy the gateway sends to the TUI / Desktop.

Each test asserts the message says WHAT happened and WHICH command to run next, and that
the lead phrases clients pattern-match on (``session busy``, ``unknown method:``,
``invalid params for``) survive the rewording.
"""

from __future__ import annotations

import pytest

from tui_gateway import user_messages as um


def test_turn_error_text_leads_with_a_plain_title_and_keeps_the_raw_body_on_a_details_line():
    raw = 'Error code: 401 - {"error": {"message": "Incorrect API key provided", "type": "invalid_request_error"}}'
    text = um.turn_error_text(raw, {"layer": "auth", "code": "auth", "retryable": False, "provider": "openai"})
    title, details, hint = text.split("\n")

    assert not title.startswith("Error")
    assert "401" not in title and "{" not in title
    assert "API key" in title and "openai" in title
    assert details.startswith("Details: ") and "Incorrect API key provided" in details
    assert "/model" in hint and "/retry" in hint


def test_turn_error_text_without_a_surface_still_names_the_next_step():
    text = um.turn_error_text("HTTP 400: invalid model id 'kimi-k2.6'")

    assert "kimi-k2.6" in text
    assert "/retry" in text or "/model" in text
    assert not text.startswith("Error:")


@pytest.mark.parametrize("command", ["undo", "compress", "reload-mcp", "rollback restore"])
def test_busy_message_names_the_real_gesture_not_a_missing_slash_command(command):
    text = um.busy_message(command)

    assert text.startswith("session busy")  # clients match this lead phrase (4009)
    assert "/interrupt" not in text
    # Shared gateway: name both gestures, never state the terminal one as the only option.
    assert "Ctrl+C" in text and "Stop button" in text
    assert "Press Ctrl+C" not in text
    assert f"/{command}" in text


def test_agent_init_and_resume_failures_point_at_existing_commands():
    init = um.agent_init_failed_message(RuntimeError("Unknown provider 'openrouterr'"))
    assert not init.startswith("agent init failed")
    assert "openrouterr" in init and "/model" in init and "hermes setup" in init
    assert "/setup" not in init  # ui-tui-only launcher; Desktop has no such command

    resume = um.resume_failed_message(ValueError("corrupt row"))
    assert not resume.startswith("resume failed")
    assert "corrupt row" in resume and "/sessions" in resume and "/new" in resume
    assert "/sessions new" not in resume  # ui-tui-only alias; Desktop ignores the argument


def test_still_starting_copy_is_not_phrased_as_fatal():
    assert "timed out" not in um.AGENT_STILL_STARTING
    assert "try again" in um.AGENT_STILL_STARTING.lower()
