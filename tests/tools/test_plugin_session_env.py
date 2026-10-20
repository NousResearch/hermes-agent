"""ctx.register_session_env: a session's commands run with the values a plugin resolves for it at spawn."""

import pytest

from gateway.session_context import clear_session_vars, set_session_vars
from hermes_cli import plugins_session_env
from hermes_cli.plugins import PluginManager

_PLUGIN = """
PANES = {}
REFUSED = []

def resolve(session_id):
    if session_id == 'boom':
        raise RuntimeError('unreadable marker')
    return PANES.get(session_id)

def register(ctx):
    ctx.register_session_env(['PANE', 'WINDOW', 'OPENAI_API_KEY'], resolve)
    try:
        ctx.register_session_env('HERMES_SESSION_ID', resolve)
    except ValueError:
        REFUSED.append('HERMES_SESSION_ID')
"""


@pytest.fixture
def plugin(tmp_path, monkeypatch):
    monkeypatch.setattr(plugins_session_env, "_REGISTRATIONS", {})
    home = tmp_path / "home"
    plugin_dir = home / "plugins" / "session_env_probe"
    plugin_dir.mkdir(parents=True)
    for path, text in ((plugin_dir / "plugin.yaml", "name: session_env_probe\nversion: 0.1.0\n"),
                       (plugin_dir / "__init__.py", _PLUGIN),
                       (home / "config.yaml", "plugins:\n  enabled: [session_env_probe]\n")):
        path.write_text(text, encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("PANE", "gateway")
    monkeypatch.setenv("WINDOW", "gateway-win")
    manager = PluginManager()
    manager.discover_and_load()
    return manager._plugins["session_env_probe"].module


def _as_session(session_id, fn):
    tokens = set_session_vars(session_key=f"k-{session_id}", session_id=session_id, source="cli", cron_session="")
    try:
        return fn()
    finally:
        clear_session_vars(tokens)


@pytest.mark.platforms("posix")  # the shared bash snapshot path
def test_each_command_runs_with_its_sessions_current_values(plugin, tmp_path):
    from tools.environments.local import LocalEnvironment

    shell = LocalEnvironment(cwd=str(tmp_path), timeout=30)
    try:
        def run(sid):
            return _as_session(sid, lambda: shell.execute('echo "$PANE/${WINDOW-unset}"')["output"].strip())

        plugin.PANES.update(a={"PANE": "pane-a", "WINDOW": None}, b={"PANE": "pane-b"})
        assert [run("a"), run("b"), run("a")] == ["pane-a/unset", "pane-b/gateway-win", "pane-a/unset"]
        # The conversation moved to another pane, then no pane shows it: the snapshot kept neither value.
        plugin.PANES["a"] = {"PANE": "pane-c"}
        assert run("a") == "pane-c/gateway-win"
        del plugin.PANES["a"]
        assert [run("a"), run("boom")] == ["gateway/gateway-win", "gateway/gateway-win"]
    finally:
        shell.cleanup()


def test_a_plugin_cannot_set_core_or_credential_names(plugin):
    from tools.environments.local import build_subprocess_env

    plugin.PANES["a"] = {"PANE": "pane-a", "OPENAI_API_KEY": "sk-from-plugin"}
    child = _as_session("a", build_subprocess_env)

    assert plugin.REFUSED == ["HERMES_SESSION_ID"]
    assert (child["PANE"], child["HERMES_SESSION_ID"], "OPENAI_API_KEY" in child) == ("pane-a", "a", False)
