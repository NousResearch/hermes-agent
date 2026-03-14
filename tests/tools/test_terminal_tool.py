import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from tools.terminal_tool import terminal_tool


def test_background_terminal_uses_environment_cwd_by_default():
    fake_env = MagicMock()
    fake_env.cwd = "/tmp/hermes-project"
    fake_env.env = {}
    fake_session = SimpleNamespace(id="proc_123", pid=12345)

    with patch(
        "tools.terminal_tool._get_env_config",
        return_value={"env_type": "local", "timeout": 180},
    ), patch(
        "tools.terminal_tool.get_or_create_environment",
        return_value=(fake_env, "local"),
    ), patch(
        "tools.terminal_tool._check_dangerous_command",
        return_value={"approved": True},
    ), patch(
        "tools.process_registry.process_registry.spawn_local",
        return_value=fake_session,
    ) as spawn_local:
        raw = terminal_tool(command="pytest -q", background=True)

    result = json.loads(raw)
    assert result["session_id"] == "proc_123"
    assert result["error"] is None
    assert spawn_local.call_args.kwargs["cwd"] == "/tmp/hermes-project"
