import importlib


def _approval(monkeypatch):
    from tools import approval

    # Keep each test isolated from context/env state that previous gateway tests
    # may have bound in the process.
    monkeypatch.delenv("HERMES_GATEWAY_SESSION", raising=False)
    monkeypatch.delenv("HERMES_SESSION_PLATFORM", raising=False)
    monkeypatch.delenv("HERMES_CRON_SESSION", raising=False)
    monkeypatch.delenv("HERMES_EXEC_ASK", raising=False)
    monkeypatch.delenv("HERMES_INTERACTIVE", raising=False)
    return approval


def test_gateway_session_blocks_delayed_gateway_start(monkeypatch):
    approval = _approval(monkeypatch)
    monkeypatch.setenv("HERMES_SESSION_PLATFORM", "discord")

    result = approval.check_all_command_guards(
        "sleep 30; /Users/dobby/.hermes/hermes-agent/venv/bin/hermes gateway start",
        "local",
    )

    assert result["approved"] is False
    assert result["pattern_key"] == "gateway_self_mutation"
    assert "gateway self-protection" in result["message"]
    assert "Do not retry" in result["message"]


def test_gateway_session_allows_gateway_status(monkeypatch):
    approval = _approval(monkeypatch)
    monkeypatch.setenv("HERMES_SESSION_PLATFORM", "discord")

    result = approval.check_all_command_guards("hermes gateway status", "local")

    assert result["approved"] is True


def test_cron_session_blocks_launchd_gateway_mutation(monkeypatch):
    approval = _approval(monkeypatch)
    monkeypatch.setenv("HERMES_CRON_SESSION", "1")

    result = approval.check_all_command_guards(
        "launchctl bootout gui/501/ai.hermes.gateway; launchctl bootstrap gui/501 ~/Library/LaunchAgents/ai.hermes.gateway.plist",
        "local",
    )

    assert result["approved"] is False
    assert result["pattern_key"] == "gateway_self_mutation"


def test_cli_context_does_not_block_gateway_restart_guard(monkeypatch):
    approval = _approval(monkeypatch)

    result = approval.check_all_command_guards("hermes gateway restart", "local")

    # Non-interactive CLI/local behavior remains unchanged; the self-protection
    # guard is scoped only to gateway-hosted runtimes.
    assert result["approved"] is True


def test_gateway_session_blocks_hermes_update(monkeypatch):
    approval = _approval(monkeypatch)
    monkeypatch.setenv("HERMES_SESSION_PLATFORM", "discord")

    result = approval.check_all_command_guards("hermes update", "local")

    assert result["approved"] is False
    assert result["pattern_key"] == "gateway_self_mutation"


def test_execute_code_blocks_subprocess_list_gateway_restart(monkeypatch):
    approval = _approval(monkeypatch)
    monkeypatch.setenv("HERMES_SESSION_PLATFORM", "discord")

    result = approval.check_execute_code_guard(
        "import subprocess\nsubprocess.run(['hermes', 'gateway', 'restart'])\n",
        "local",
    )

    assert result["approved"] is False
    assert result["pattern_key"] == "gateway_self_mutation"


def test_execute_code_blocks_os_system_gateway_restart(monkeypatch):
    approval = _approval(monkeypatch)
    monkeypatch.setenv("HERMES_SESSION_PLATFORM", "discord")

    result = approval.check_execute_code_guard(
        "import os\nos.system('hermes gateway restart')\n",
        "local",
    )

    assert result["approved"] is False
    assert result["pattern_key"] == "gateway_self_mutation"
