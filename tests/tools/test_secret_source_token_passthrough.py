"""Secret-source tokens remain usable by operator terminal workflows (PR #137459)."""
import subprocess
import sys

import pytest

from agent import secret_scope
from tools.env_passthrough import clear_env_passthrough, register_env_passthrough
from tools.environments.local import build_subprocess_env


@pytest.fixture(autouse=True)
def isolated_passthrough():
    clear_env_passthrough()
    secret_scope.set_multiplex_active(False)
    yield
    clear_env_passthrough()
    secret_scope.set_multiplex_active(False)


def _child_token(name):
    result = subprocess.run(
        [sys.executable, "-c", "import os, sys; print(os.environ.get(sys.argv[1], 'absent'))", name],
        env=build_subprocess_env(), capture_output=True, text=True, check=True, timeout=30,
    )
    return result.stdout.strip()


@pytest.mark.parametrize("name", ["OP_SERVICE_ACCOUNT_TOKEN", "BWS_ACCESS_TOKEN"])
def test_operator_token_reaches_terminal_without_sharing_opt_in(monkeypatch, name):
    monkeypatch.setenv(name, "operator-token")
    register_env_passthrough([name])

    assert _child_token(name) == "operator-token"


@pytest.mark.parametrize("name", ["OP_SERVICE_ACCOUNT_TOKEN", "BWS_ACCESS_TOKEN"])
def test_routed_terminal_does_not_fall_back_to_process_token(monkeypatch, name):
    monkeypatch.setenv(name, "launch-token")
    register_env_passthrough([name])
    secret_scope.set_multiplex_active(True)
    token = secret_scope.set_secret_scope({})
    try:
        assert _child_token(name) == "absent"
    finally:
        secret_scope.reset_secret_scope(token)
