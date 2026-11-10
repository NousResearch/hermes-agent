"""Generic background-process credential policy is not widened by caller extras."""
import os
from unittest.mock import MagicMock, patch

from tools.environments.local_env_policy import _HERMES_PROVIDER_ENV_FORCE_PREFIX
from tools.process_registry import ProcessRegistry


def test_spawn_local_strips_blocked_vars_from_background_env(tmp_path, monkeypatch):
    monkeypatch.setattr("hermes_cli.process_identity._update_supervised_pgids", lambda verb, pgids: None)
    registry = ProcessRegistry()
    captured = {}

    def fake_popen(cmd, **kwargs):
        captured["env"] = kwargs["env"]
        proc = MagicMock()
        proc.pid = 4321
        proc.stdout = iter([])
        proc.stdin = MagicMock()
        proc.poll.return_value = None
        return proc

    fake_thread = MagicMock()

    with patch.dict(os.environ, {
        "PATH": "/usr/bin:/bin",
        "HOME": "/home/user",
        "USER": "tester",
        "TELEGRAM_BOT_TOKEN": "bot-secret",
        "FIRECRAWL_API_KEY": "fc-secret",
    }, clear=True), \
        patch("tools.process_registry._find_shell", return_value="/bin/bash"), \
        patch("subprocess.Popen", side_effect=fake_popen), \
        patch("threading.Thread", return_value=fake_thread), \
        patch.object(registry, "_write_checkpoint"):
        registry.spawn_local(
            "echo hello",
            cwd=str(tmp_path),
            env_vars={
                "MY_CUSTOM_VAR": "keep-me",
                "TELEGRAM_BOT_TOKEN": "drop-me",
                f"{_HERMES_PROVIDER_ENV_FORCE_PREFIX}TELEGRAM_BOT_TOKEN": "forced-bot-token",
            },
        )

    env = captured["env"]
    assert env["MY_CUSTOM_VAR"] == "keep-me"
    # Tier-1 messaging credentials are absolute denies at the generic
    # background-process intent. A force prefix cannot manufacture an
    # exceptional credential channel.
    assert "TELEGRAM_BOT_TOKEN" not in env
    assert "FIRECRAWL_API_KEY" not in env
    assert f"{_HERMES_PROVIDER_ENV_FORCE_PREFIX}TELEGRAM_BOT_TOKEN" not in env
    assert env["PYTHONUNBUFFERED"] == "1"
