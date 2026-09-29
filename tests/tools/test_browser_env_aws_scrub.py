"""The agent-browser subprocess must not inherit the operator's AWS credential chain."""

import pytest

import tools.browser_tool as bt
import tools.browser_tool_session as bt_session

AWS_VARS = {
    "AWS_ACCESS_KEY_ID": "browser-must-not-inherit",
    "AWS_SECRET_ACCESS_KEY": "browser-must-not-inherit",
    "AWS_SESSION_TOKEN": "browser-must-not-inherit",
    "AWS_PROFILE": "browser-must-not-inherit",
    "AWS_SHARED_CREDENTIALS_FILE": "/tmp/browser-credentials",
    "AWS_WEB_IDENTITY_TOKEN_FILE": "/tmp/browser-token",
    "AWS_CONTAINER_CREDENTIALS_FULL_URI": "http://127.0.0.1/creds",
}


@pytest.fixture
def aws_env(monkeypatch):
    for key, value in AWS_VARS.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setenv("BROWSERBASE_API_KEY", "allowed-browser-key")


def test_build_browser_env_excludes_aws_chain_keeps_backend_keys(aws_env):
    env = bt._build_browser_env()

    assert not set(AWS_VARS) & env.keys()
    assert env["BROWSERBASE_API_KEY"] == "allowed-browser-key"


def test_scrub_covers_shared_sanitizer_leftovers(monkeypatch):
    """Even if the shared sanitizer lets AWS vars through, the browser boundary drops them."""
    import tools.environments.local as local

    monkeypatch.setattr(local, "hermes_subprocess_env", lambda inherit_credentials=False: dict(AWS_VARS, PATH="/usr/bin"))

    env = bt._build_browser_env()

    assert not set(AWS_VARS) & env.keys()
    assert "PATH" in env


def test_command_env_excludes_aws_chain_keeps_backend_keys(aws_env, monkeypatch):
    import hermes_cli.browser_runtime as browser_runtime

    monkeypatch.setattr(browser_runtime, "chromium_executable", lambda: None)

    env = bt_session._agent_browser_command_env("/tmp/hermes-browser-socket")

    assert not set(AWS_VARS) & env.keys()
    assert env["BROWSERBASE_API_KEY"] == "allowed-browser-key"
    assert env["AGENT_BROWSER_SOCKET_DIR"] == "/tmp/hermes-browser-socket"
