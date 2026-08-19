"""The tui_gateway ``shell.exec`` handler must not leak process credentials.

Regression: ``shell.exec`` ran ``subprocess.run(..., shell=True)`` with no
``env=``, so the spawned shell inherited the whole ``os.environ`` — every
provider key, gateway bot token, etc. The sibling ``cli.exec`` handler in the
same module already sanitizes via ``hermes_subprocess_env``; this brings
``shell.exec`` in line so a command run through it cannot read Hermes-managed
credentials from either stripping tier.
"""
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture()
def server():
    with patch.dict(
        "sys.modules",
        {
            "hermes_constants": MagicMock(
                get_hermes_home=MagicMock(return_value="/tmp/hermes_test")
            ),
            "hermes_cli.env_loader": MagicMock(),
            "hermes_cli.banner": MagicMock(),
            "hermes_state": MagicMock(),
        },
    ):
        import importlib

        mod = importlib.import_module("tui_gateway.server")
        yield mod
        mod._sessions.clear()
        mod._pending.clear()
        mod._answers.clear()


def test_shell_exec_scrubs_credentials_end_to_end(server, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-leak-secret-xyz")
    resp = server._methods["shell.exec"](1, {"command": 'echo "leak=$OPENAI_API_KEY"'})
    assert "result" in resp, resp  # command was allowed and ran
    out = resp["result"]["stdout"]
    assert "leak=" in out  # the command executed
    assert "sk-leak-secret-xyz" not in out  # but the secret was scrubbed


def _capture_shell_exec_env(server, command="true"):
    """Run the real shell.exec handler and return the env dict it passed.

    Assertions must target this dict, not the handler's stdout: shell.exec
    truncates stdout to the last 4000 characters, so a secret that leaks near
    the start of a large ``env`` dump would fall outside the window and a
    stdout ``not in`` check would pass while the secret is still exported
    (false pass), and a benign trailing variable can fall outside the window on
    a big-environment host and fail a stdout ``in`` check (false fail). The env
    dict the child actually receives is complete and host-independent.
    """
    captured = {}

    def fake_run(*args, **kwargs):
        captured.setdefault("env", kwargs.get("env"))
        stub = MagicMock()
        stub.stdout = ""
        stub.stderr = ""
        stub.returncode = 0
        return stub

    with patch("tui_gateway.server.subprocess.run", side_effect=fake_run):
        server._methods["shell.exec"](1, {"command": command})
    assert "env" in captured, "shell.exec did not spawn a subprocess"
    # Fail loudly if the handler ever stops passing an explicit env: a None here
    # would otherwise make the caller's `name not in env` checks pass vacuously.
    assert captured["env"] is not None, "shell.exec must pass an explicit env, not inherit"
    return captured["env"]


def test_shell_exec_scrubs_multiple_credential_classes(server, monkeypatch):
    secrets = {
        "OPENAI_API_KEY": "provider-secret-value",
        "GITHUB_TOKEN": "github-secret-value",
        "SLACK_BOT_TOKEN": "messaging-secret-value",
        "MODAL_TOKEN_SECRET": "infrastructure-secret-value",
        "AUXILIARY_REVIEW_API_KEY": "internal-side-model-secret-value",
        "GATEWAY_RELAY_SECRET": "relay-secret-value",
    }
    for name, value in secrets.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setenv("HERMES_SHELL_EXEC_CANARY", "benign-canary-value")

    env = _capture_shell_exec_env(server)

    # Authoritative check on the full child environment (not truncated stdout):
    # every managed secret name and value is absent regardless of env size.
    values = set(env.values())
    for name, value in secrets.items():
        assert name not in env, f"{name} leaked into shell.exec child env"
        assert value not in values, f"{name} value leaked into shell.exec child env"
    # Benign variables survive so ordinary commands still work.
    assert env.get("HERMES_SHELL_EXEC_CANARY") == "benign-canary-value"
    assert "PATH" in env


def test_shell_exec_leak_is_caught_even_when_pushed_past_stdout_truncation(
    server, monkeypatch
):
    # Demonstrate the truncation hazard itself, then the check that closes it.
    # shell.exec returns only the last 4000 characters of stdout, so render an
    # unsanitized ``env`` dump with the secret ahead of >4000 chars of ordinary
    # variables (constructed, since ``env`` prints in no guaranteed order): the
    # secret lands outside the returned window and an old-style stdout
    # ``not in`` assertion passes on a real leak.
    pad = {f"PAD_VAR_{i:03d}": "x" * 40 for i in range(300)}
    for name, value in pad.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-should-be-scrubbed")

    leaked_dump = "\n".join(
        ["OPENAI_API_KEY=sk-should-be-scrubbed"]
        + [f"{name}={value}" for name, value in pad.items()]
    )
    stdout_window = leaked_dump[-4000:]
    assert "sk-should-be-scrubbed" in leaked_dump  # the secret did leak...
    assert "sk-should-be-scrubbed" not in stdout_window  # ...yet stdout looks clean

    # The authoritative check targets the env dict the handler actually passes,
    # where there is no window for a leak to hide in, wherever it would print.
    env = _capture_shell_exec_env(server, command="env")
    assert "OPENAI_API_KEY" not in env
    assert "sk-should-be-scrubbed" not in set(env.values())


def test_shell_exec_passes_sanitized_env(server, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-leak-secret-xyz")
    captured = {}

    def fake_run(*args, **kwargs):
        # patch() replaces run on the shared subprocess module, so keep only
        # the handler's (first) call and return a stub — never re-enter run.
        captured.setdefault("env", kwargs.get("env"))
        stub = MagicMock()
        stub.stdout = ""
        stub.stderr = ""
        stub.returncode = 0
        return stub

    with patch("tui_gateway.server.subprocess.run", side_effect=fake_run):
        server._methods["shell.exec"](1, {"command": "echo hi"})

    assert "env" in captured
    env = captured["env"]
    assert env is not None  # env is explicitly passed, not inherited
    assert "OPENAI_API_KEY" not in env  # Tier-2 secret removed for shell.exec
    assert "PATH" in env  # benign vars preserved so commands still work
