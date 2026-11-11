"""A routed profile's ``terminal.home_mode`` governs the HOME of its terminal subprocesses.

One multiplexed gateway serves several profiles; the launch profile's process env must not decide the
subprocess HOME of a turn routed to another profile.
"""

import os
import sys

import pytest

import tools.terminal_tool as tt
from tools.environments.local import LocalEnvironment


@pytest.fixture
def bee(monkeypatch, tmp_path):
    """A second profile whose config asks for the profile home, under a launch env that sets no mode."""
    for name in [k for k in os.environ if k.startswith("TERMINAL_")]:
        monkeypatch.delenv(name)
    monkeypatch.setattr(tt, "_terminal_config_bridge_attempted", True)
    home = tmp_path / ".hermes" / "profiles" / "bee"
    (home / "home").mkdir(parents=True)
    (home / "config.yaml").write_text("terminal:\n  backend: local\n  home_mode: profile\n", encoding="utf-8")
    return home


@pytest.mark.platforms("posix")
def test_routed_profile_home_mode_reaches_the_terminal_subprocess(bee, tmp_path):
    import gateway.run as gw

    with gw._profile_runtime_scope(bee):
        environment = LocalEnvironment(cwd=str(tmp_path), timeout=30)
        environment.init_session()
        try:
            result = environment.execute('printf "%s" "$HOME"', cwd=str(tmp_path), timeout=30)
        finally:
            environment.cleanup()

    assert result["output"].strip() == str(bee / "home")


@pytest.mark.platforms("posix")
def test_routed_profile_home_mode_reaches_the_execute_code_child(bee, tmp_path):
    import gateway.run as gw
    from tools.code_execution_env import _build_child_env

    with gw._profile_runtime_scope(bee):
        child_env = _build_child_env(rpc_endpoint="unused", rpc_token="unused", tmpdir=str(tmp_path),
                                     child_python=sys.executable)

    assert child_env["HOME"] == str(bee / "home")
