"""Owner HOME policy must survive real child-environment entrypoints."""
import os
import sys

import pytest


@pytest.fixture(autouse=True)
def unpinned_temp_environment(monkeypatch):
    # The canonical runner may own TMPDIR without a Hermes marker. These HOME
    # cases exercise Hermes-default scratch, not the explicit-user-temp override.
    for key in ("TMPDIR", "TMP", "TEMP", "HERMES_SCRATCH_DIR"):
        monkeypatch.delenv(key, raising=False)



@pytest.mark.parametrize("owner_mode", ["real", "profile"])
@pytest.mark.parametrize("entrypoint", ["factory", "acp", "code"])
def test_owner_home_at_child_env_boundary(tmp_path, monkeypatch, owner_mode, entrypoint):
    from hermes_constants import set_hermes_home_override, reset_hermes_home_override
    from tools.terminal_scope import set_terminal_scope, reset_terminal_scope
    from tools.environments.local import build_subprocess_env
    from agent.acp_client import _build_subprocess_env
    from tools.code_execution_env import _build_child_env

    real = tmp_path / "real"
    owner = tmp_path / "owner"
    real.mkdir()
    (owner / "home").mkdir(parents=True)
    monkeypatch.setenv("HOME", str(real))
    monkeypatch.setenv("HERMES_REAL_HOME", str(real))
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "launcher"))
    monkeypatch.setenv("TERMINAL_HOME_MODE", "profile" if owner_mode == "real" else "real")
    before = dict(os.environ)
    home_token = set_hermes_home_override(str(owner))
    scope_token = set_terminal_scope({"TERMINAL_HOME_MODE": owner_mode})
    try:
        if entrypoint == "factory":
            child = build_subprocess_env()
        elif entrypoint == "acp":
            child = _build_subprocess_env()
        else:
            child = _build_child_env(rpc_endpoint="socket", rpc_token="token", tmpdir=str(tmp_path), child_python=sys.executable)
        assert child["HOME"] == str(real if owner_mode == "real" else owner / "home")
        assert child["HERMES_HOME"] == str(owner)
        assert child["HERMES_REAL_HOME"] == str(real)
        assert child["TMPDIR"].startswith(str(owner / "cache" / "scratch"))
    finally:
        reset_terminal_scope(scope_token)
        reset_hermes_home_override(home_token)
    assert dict(os.environ) == before
