"""A user terminal backend under ``plugins.isolation: host``: the plugin's environment object cannot cross
the host boundary (its methods would run outside the Hermes process's interrupt, activity and terminal-scope
state), so creating it refuses with the setting to change before the host starts a sandbox nobody can use
or clean up. Real plugin, real host process, real loader against a temp HERMES_HOME.
"""

from __future__ import annotations

import hermes_yaml as yaml
import pytest

from hermes_cli import plugins as plugins_mod
from hermes_cli.plugins import PluginManager

BACKEND_PLUGIN = '''
from pathlib import Path
from agent.terminal_env_provider import TerminalEnvironmentProvider

MARKER = Path(__file__).with_name("created")

class Env:
    def execute(self, command, **kw):
        return {"output": command, "returncode": 0}
    def cleanup(self):
        pass

class Box(TerminalEnvironmentProvider):
    @property
    def name(self):
        return "hostbox"
    def is_available(self):
        return True
    def create_environment(self, *, cwd, timeout, task_id="default", image=None, container_config=None, **kw):
        MARKER.write_text("sandbox started")
        return Env()

def register(ctx):
    ctx.register_terminal_environment_provider(Box())
'''


def _load(tmp_path, monkeypatch, isolation):
    home = tmp_path / ".hermes"
    plugin_dir = home / "plugins" / "hostbox"
    plugin_dir.mkdir(parents=True)
    (plugin_dir / "plugin.yaml").write_text(yaml.safe_dump({"name": "hostbox", "version": "1.0"}), encoding="utf-8")
    (plugin_dir / "__init__.py").write_text(BACKEND_PLUGIN, encoding="utf-8")
    (home / "config.yaml").write_text(yaml.safe_dump(
        {"plugins": {"enabled": ["hostbox"], "isolation": isolation}, "terminal": {"backend": "hostbox"}}),
        encoding="utf-8")
    empty_bundled = tmp_path / "bundled"
    empty_bundled.mkdir()
    monkeypatch.setenv("HOME", str(tmp_path / "os-home"))
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(plugins_mod, "get_bundled_plugins_dir", lambda: empty_bundled)
    manager = PluginManager()
    manager.discover_and_load()
    assert manager._plugins["hostbox"].error is None
    return manager, plugin_dir / "created"


def test_a_hosted_terminal_backend_refuses_with_the_isolation_setting(tmp_path, monkeypatch):
    from tools.terminal_tool_backends import _create_environment

    manager, marker = _load(tmp_path, monkeypatch, "host")
    try:
        with pytest.raises(ValueError, match=r"plugins\.isolation: in_process"):
            _create_environment("hostbox", image="", cwd="/", timeout=10)
        assert not marker.exists()  # no sandbox started in the host
    finally:
        manager.unload()
        manager._plugin_host().shutdown()


def test_the_same_backend_in_process_runs_commands(tmp_path, monkeypatch):
    from tools.terminal_tool_backends import _create_environment

    manager, marker = _load(tmp_path, monkeypatch, "in_process")
    try:
        env = _create_environment("hostbox", image="", cwd="/", timeout=10)
        assert env.execute("echo hi") == {"output": "echo hi", "returncode": 0}
        assert marker.exists()
    finally:
        manager.unload()
