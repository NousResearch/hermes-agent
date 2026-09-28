"""Managed Modal hands the sudo password to the command the same way every other backend does:
prepended to the exec's stdin, never piped into one pipeline of the command text."""

import base64
from types import SimpleNamespace

from tools.environments.managed_modal import ManagedModalEnvironment

SYNTHETIC_SUDO_PASSWORD = "synthetic_sudo_password_121932_not_a_real_credential"


def _env_with_gateway(request):
    env = ManagedModalEnvironment.__new__(ManagedModalEnvironment)
    env.cwd = "/workspace"
    env.timeout = 20
    env._sandbox_id = "sandbox-test"
    env._persistent = False
    env._request = request
    return env


def test_managed_modal_sudo_uses_stdin_payload_not_command_text():
    captured = []
    body = {"status": "completed", "output": "ok", "returncode": 0}
    env = _env_with_gateway(lambda *args, **kwargs: (
        captured.append(kwargs["json"]) or SimpleNamespace(status_code=200, json=lambda: body)))
    env._prepare_command = lambda command: (command, SYNTHETIC_SUDO_PASSWORD + "\n")

    result = env.execute("sudo cat", stdin_data="ordinary stdin payload")

    assert result == {"output": "ok", "returncode": 0}
    payload = captured[0]
    assert SYNTHETIC_SUDO_PASSWORD not in payload["command"]
    assert base64.b64encode(SYNTHETIC_SUDO_PASSWORD.encode()).decode() not in payload["command"]
    assert payload["stdinData"] == SYNTHETIC_SUDO_PASSWORD + "\nordinary stdin payload"
