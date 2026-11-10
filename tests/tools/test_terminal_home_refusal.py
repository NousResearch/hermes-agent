"""HOME refusal and ContextVar restoration at the real child builders."""
import os
import sys

import pytest


@pytest.mark.parametrize("entrypoint", ["factory", "acp", "code"])
def test_home_refusal_reaches_real_builder(tmp_path, monkeypatch, entrypoint):
    from hermes_constants import set_hermes_home_override, reset_hermes_home_override
    from tools.terminal_scope import (
        TerminalPolicyRefusal, TerminalPolicyUnavailable, get_terminal_scope,
        set_terminal_scope, reset_terminal_scope,
    )
    from tools.environments.local import build_subprocess_env
    from agent.acp_client import _build_subprocess_env
    from tools.code_execution_env import _build_child_env

    owner = tmp_path / "owner"
    (owner / "home").mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "launch"))
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_REAL_HOME", str(tmp_path))
    monkeypatch.setenv("TERMINAL_HOME_MODE", "real")
    before = dict(os.environ)
    outer = {"TERMINAL_HOME_MODE": "profile"}
    outer_token = set_terminal_scope(outer)
    home_token = set_hermes_home_override(str(owner))
    token = set_terminal_scope(TerminalPolicyRefusal("owner policy unreadable"))
    try:
        with pytest.raises(TerminalPolicyUnavailable, match="owner policy unreadable"):
            if entrypoint == "factory":
                build_subprocess_env()
            elif entrypoint == "acp":
                _build_subprocess_env()
            else:
                _build_child_env(rpc_endpoint="socket", rpc_token="token", tmpdir=str(tmp_path), child_python=sys.executable)
    finally:
        reset_terminal_scope(token)
        assert get_terminal_scope() is outer
        reset_terminal_scope(outer_token)
        reset_hermes_home_override(home_token)
    assert dict(os.environ) == before


def test_auto_home_uses_native_container_detection(tmp_path, monkeypatch):
    """Run natively in the container gate; a host lane can run the same assertion.

    Unlike the semantic table tests this does not fake is_container or the OS.
    This checks HOME selection for the detector's native result, not the detector itself.
    """
    from hermes_constants import is_container, get_subprocess_home
    from tools.terminal_scope import set_terminal_scope, reset_terminal_scope
    real = tmp_path / "real"
    owner = tmp_path / "owner"
    real.mkdir()
    (owner / "home").mkdir(parents=True)
    monkeypatch.setenv("HOME", str(real))
    monkeypatch.setenv("HERMES_REAL_HOME", str(real))
    monkeypatch.setenv("HERMES_HOME", str(owner))
    monkeypatch.setenv("TERMINAL_HOME_MODE", "profile" if not is_container() else "real")
    expected = owner / "home" if is_container() else real
    token = set_terminal_scope({"TERMINAL_HOME_MODE": "auto"})
    try:
        assert (get_subprocess_home() or str(real)) == str(expected)
    finally:
        reset_terminal_scope(token)
