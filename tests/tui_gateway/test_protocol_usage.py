"""Provider quota, rate limits, and fallback behavior on usage RPCs."""

import sys
import threading
import time
import types
from unittest.mock import MagicMock

import pytest


@pytest.fixture
def server():
    stdout = sys.stdout
    from tui_gateway import server
    sys.stdout = stdout
    try:
        yield server
    finally:
        for sid in list(server._sessions):
            server._close_session_by_id(sid, end_reason="test_cleanup")


def test_usage_live_and_session_output_include_provider_limits(server, monkeypatch):
    from agent import account_usage

    class Agent:
        provider = "openai-codex"
        base_url = "https://example.invalid/v1"
        api_key = "secret"
        model = "gpt-test"
        session_input_tokens = 123
        session_output_tokens = 45
        session_total_tokens = 168
        session_api_calls = 2
        session_reasoning_tokens = 0
        session_prompt_tokens = 123
        session_completion_tokens = 45
        context_compressor = None

        def get_rate_limit_state(self):
            return types.SimpleNamespace(has_data=True)

    fetch = MagicMock(return_value=object())
    monkeypatch.setattr(account_usage, "fetch_account_usage", fetch)
    monkeypatch.setattr(
        account_usage,
        "render_account_usage_lines",
        lambda _: ["Account limits", "Session: 93% remaining"],
    )
    monkeypatch.setattr(account_usage, "nous_credits_lines", list)
    monkeypatch.setattr(
        "agent.rate_limit_tracker.format_rate_limit_display",
        lambda _: "Requests: 50% remaining",
    )
    server._sessions["usage-live"] = {
        "session_key": "usage-live",
        "agent": Agent(),
        "history": [],
        "history_lock": threading.Lock(),
    }
    desktop = server.handle_request(
        {
            "id": 1,
            "method": "slash.exec",
            "params": {"session_id": "usage-live", "command": "usage"},
        }
    )
    tui = server.handle_request(
        {
            "id": 2,
            "method": "session.usage",
            "params": {"session_id": "usage-live"},
        }
    )
    assert "Session Token Usage" in desktop["result"]["output"]
    assert "Session: 93% remaining" in desktop["result"]["output"]
    assert "Requests: 50% remaining" in desktop["result"]["output"]
    assert "secret" not in desktop["result"]["output"]
    assert tui["result"]["account_lines"] == ["Account limits", "Session: 93% remaining"]
    assert tui["result"]["rate_limit_lines"] == ["Requests: 50% remaining"]
    assert tui["result"]["total"] == 168
    assert fetch.call_args.args == ("openai-codex",)


def test_usage_no_agent_uses_metadata_and_codex_pool_fallback(server, monkeypatch):
    from agent import account_usage

    seen = []

    def fetch(provider, **kwargs):
        seen.append({"provider": provider, **kwargs})
        return object()

    monkeypatch.setattr(account_usage, "fetch_account_usage", fetch)
    monkeypatch.setattr(
        account_usage,
        "render_account_usage_lines",
        lambda _: ["Weekly: 80% remaining"],
    )
    monkeypatch.setattr(account_usage, "nous_credits_lines", list)
    server._sessions["usage-mirror"] = {
        "session_key": "usage-mirror",
        "agent": None,
        "_metadata_mirror": {
            "provider": "openai-codex",
            "usage": {"calls": 1},
        },
        "history": [],
        "history_lock": threading.Lock(),
    }
    desktop = server.handle_request(
        {
            "id": 1,
            "method": "slash.exec",
            "params": {"session_id": "usage-mirror", "command": "usage"},
        }
    )
    tui = server.handle_request(
        {
            "id": 2,
            "method": "session.usage",
            "params": {"session_id": "usage-mirror"},
        }
    )
    assert "Session Token Usage" in desktop["result"]["output"]
    assert "Weekly: 80% remaining" in desktop["result"]["output"]
    assert tui["result"]["account_lines"] == ["Weekly: 80% remaining"]
    assert seen == [
        {"provider": "openai-codex", "base_url": None, "api_key": None},
        {"provider": "openai-codex", "base_url": None, "api_key": None},
    ]


def test_usage_provider_failure_is_fail_open(server, monkeypatch):
    from agent import account_usage

    monkeypatch.setattr(
        account_usage,
        "fetch_account_usage",
        MagicMock(side_effect=RuntimeError("offline")),
    )
    monkeypatch.setattr(account_usage, "nous_credits_lines", list)
    server._sessions["usage-fail"] = {
        "session_key": "usage-fail",
        "agent": None,
        "_metadata_mirror": {
            "provider": "openai-codex",
            "usage": {"calls": 0},
        },
        "history": [],
        "history_lock": threading.Lock(),
    }
    response = server.handle_request(
        {
            "id": 1,
            "method": "session.usage",
            "params": {"session_id": "usage-fail"},
        }
    )
    assert "error" not in response and "account_lines" not in response["result"]


def test_usage_provider_timeout_is_prompt_and_fail_open(server, monkeypatch):
    from agent import account_usage

    release_fetch = threading.Event()

    def fetch(*args, **kwargs):
        release_fetch.wait()
        return object()

    from tui_gateway import usage_provider

    monkeypatch.setattr(usage_provider, "_ACCOUNT_USAGE_TIMEOUT_SECONDS", 0.01)
    monkeypatch.setattr(account_usage, "fetch_account_usage", fetch)
    monkeypatch.setattr(account_usage, "nous_credits_lines", list)
    server._sessions["usage-timeout"] = {
        "session_key": "usage-timeout",
        "agent": None,
        "_metadata_mirror": {
            "provider": "openai-codex",
            "usage": {"calls": 0},
        },
        "history": [],
        "history_lock": threading.Lock(),
    }

    started = time.monotonic()
    try:
        response = server.handle_request(
            {
                "id": 1,
                "method": "session.usage",
                "params": {"session_id": "usage-timeout"},
            }
        )
        elapsed = time.monotonic() - started
    finally:
        release_fetch.set()

    # The budget includes queueing and profile resolution, so a loaded worker
    # may time out before reaching fetch. Both paths must return fail-open.
    assert elapsed < 2.0
    assert "error" not in response
    assert "account_lines" not in response["result"]


@pytest.mark.parametrize("command", ["usage reset", "usage reset --force", "usage unexpected"])
def test_usage_subcommands_fall_through_to_worker(server, command):
    worker = MagicMock(run=MagicMock(return_value="worker result"), pop_seed=MagicMock(return_value=""))
    server._sessions["usage-worker"] = {
        "session_key": "usage-worker",
        "agent": None,
        "slash_worker": worker,
    }
    response = server.handle_request(
        {
            "id": 1,
            "method": "slash.exec",
            "params": {"session_id": "usage-worker", "command": command},
        }
    )
    assert response["result"] == {"output": "worker result"}
    worker.run.assert_called_once_with(command)


@pytest.mark.parametrize("secondary", [False, True])
def test_usage_credits_read_the_session_profile(server, monkeypatch, tmp_path, secondary):
    from agent import account_usage
    from hermes_constants import get_hermes_home, reset_hermes_home_override, set_hermes_home_override
    from agent.secret_scope import get_secret

    profile = tmp_path / "secondary"
    profile.mkdir()
    (profile / ".env").write_text("USAGE_TEST_SECRET=secondary-only\n")
    monkeypatch.setenv("USAGE_TEST_SECRET", "launch-only")
    monkeypatch.setattr(account_usage, "fetch_account_usage", lambda *a, **kw: None)
    seen = []

    def credits():
        seen.append((get_hermes_home(), get_secret("USAGE_TEST_SECRET")))
        return ["$42 remaining"]

    monkeypatch.setattr(account_usage, "nous_credits_lines", credits)
    session = {"agent": None}
    if secondary:
        session["profile_home"] = str(profile)
    else:
        monkeypatch.setenv("HERMES_HOME", str(profile))
    server._sessions["credits-profile"] = session
    home_token = set_hermes_home_override(str(tmp_path / "foreign"))
    try:
        response = server.handle_request({"id": 1, "method": "session.usage",
                                          "params": {"session_id": "credits-profile"}})
    finally:
        reset_hermes_home_override(home_token)
    assert response["result"]["credits_lines"] == ["$42 remaining"]
    assert seen == [(profile, "secondary-only")]


@pytest.mark.parametrize("command,force", [
    ("usage reset", False),
    ("usage reset --force", True),
    ("usage RESET --FORCE", True),
])
def test_usage_reset_targets_the_displayed_live_account(server, monkeypatch, command, force):
    from agent import account_usage

    redeem = MagicMock(return_value=types.SimpleNamespace(message="Reset the displayed account"))
    monkeypatch.setattr(account_usage, "redeem_codex_reset_credit", redeem)
    worker = MagicMock(run=MagicMock(return_value="Reset a different account"))
    server._sessions["usage-reset-live"] = {
        "session_key": "usage-reset-live", "slash_worker": worker,
        "agent": types.SimpleNamespace(provider="openai-codex", api_key="live-account-token",
                                       base_url="https://live-account.invalid"),
    }
    response = server.handle_request({
        "id": 1, "method": "slash.exec",
        "params": {"session_id": "usage-reset-live", "command": command},
    })
    assert response["result"]["output"] == "Reset the displayed account"
    redeem.assert_called_once_with(base_url="https://live-account.invalid", api_key="live-account-token", force=force)
    worker.run.assert_not_called()


def test_live_usage_reset_rejects_other_providers(server, monkeypatch):
    from agent import account_usage

    redeem = MagicMock()
    monkeypatch.setattr(account_usage, "redeem_codex_reset_credit", redeem)
    server._sessions["usage-reset-other"] = {
        "agent": types.SimpleNamespace(provider="openrouter", api_key="other-token", base_url="https://other.invalid"),
    }
    response = server.handle_request({
        "id": 1, "method": "slash.exec",
        "params": {"session_id": "usage-reset-other", "command": "usage reset --force"},
    })
    assert "openai-codex" in response["result"]["output"].lower()
    redeem.assert_not_called()
