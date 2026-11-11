"""``terminal.home_mode`` in config.yaml governs the HOME of terminal subprocesses on every surface.

The CLI and the standalone gateway bridge it with their own key tables; processes that rely on the
shared bridge (``hermes serve`` / Desktop in-process agents, the dashboard, the TUI launcher) must
honour it the same way, or a user who set ``home_mode: profile`` for a per-profile CLI identity gets
the real OS home there.
"""

import os

import pytest

import tools.terminal_tool as tt
from tools.environments.local import LocalEnvironment


@pytest.mark.platforms("posix")
def test_profile_home_mode_from_config_reaches_the_terminal_subprocess(monkeypatch, tmp_path):
    hermes_home = tmp_path / ".hermes"
    (hermes_home / "home").mkdir(parents=True)
    (hermes_home / "config.yaml").write_text(
        "terminal:\n  backend: local\n  home_mode: profile\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    for name in [k for k in os.environ if k.startswith("TERMINAL_")]:
        monkeypatch.delenv(name)
    monkeypatch.setattr(tt, "_terminal_config_bridge_attempted", False)

    # The bridge an in-process Desktop/serve agent runs before its first terminal command.
    tt._ensure_terminal_env_bridged()
    environment = LocalEnvironment(cwd=str(tmp_path), timeout=30)
    environment.init_session()
    try:
        result = environment.execute('printf "%s" "$HOME"', cwd=str(tmp_path), timeout=30)
    finally:
        environment.cleanup()

    assert result["output"].strip() == str(hermes_home / "home")
