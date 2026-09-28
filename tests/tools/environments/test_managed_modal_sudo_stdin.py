"""Managed Modal hands the sudo password to the command the same way every other backend does:
prepended to the exec's stdin, never piped into one pipeline of the command text."""

import base64
import os
import subprocess
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


_FAKE_SUDO = """#!/bin/sh
# Like `sudo -S -p ''`: read one password line from stdin, then run the command.
[ "$1" = "-S" ] || { echo "sudo: a password is required" >&2; exit 1; }
shift; [ "$1" = "-p" ] && shift 2
IFS= read -r pw || pw=""
[ "$pw" = "$EXPECTED_SUDO_PASSWORD" ] || { echo "sudo: no password was provided" >&2; exit 1; }
exec "$@"
"""


def test_every_sudo_in_a_compound_command_gets_the_password(tmp_path, monkeypatch):
    """The gateway runs the command text in a shell whose stdin is ``stdinData``. Every
    ``sudo -S`` in an ``&&`` list must read its own password line from that stream."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "sudo").write_text(_FAKE_SUDO)
    (bin_dir / "sudo").chmod(0o755)
    monkeypatch.setenv("SUDO_PASSWORD", SYNTHETIC_SUDO_PASSWORD)

    def gateway(method, path, json=None, timeout=None):
        if not path.endswith("/execs"):  # sandbox teardown on garbage collection
            return SimpleNamespace(status_code=200, json=lambda: {})
        shell_env = {**os.environ, "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
                     "EXPECTED_SUDO_PASSWORD": SYNTHETIC_SUDO_PASSWORD}
        proc = subprocess.run(["bash", "-c", json["command"]], input=json.get("stdinData", ""),
                              capture_output=True, text=True, cwd=tmp_path, env=shell_env, timeout=20)
        body = {"status": "completed", "output": proc.stdout + proc.stderr, "returncode": proc.returncode}
        return SimpleNamespace(status_code=200, json=lambda: body)

    result = _env_with_gateway(gateway).execute("cd / && sudo echo one && sudo echo two")

    assert result == {"output": "one\ntwo\n", "returncode": 0}
