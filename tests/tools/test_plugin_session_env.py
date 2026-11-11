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
    ctx.register_session_env(['PANE', 'WINDOW', 'OPENAI_API_KEY', 'ACME_SANDBOX_TOKEN'], resolve)
    try:
        ctx.register_session_env('HERMES_SESSION_ID', resolve)
    except ValueError:
        REFUSED.append('HERMES_SESSION_ID')
"""


@pytest.fixture
def plugin(tmp_path, monkeypatch):
    monkeypatch.setattr(plugins_session_env, "_REGISTRATIONS", {})
    monkeypatch.setattr(plugins_session_env, "_DECLARED", {})
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


def test_a_plugin_cannot_set_core_or_credential_names(plugin, monkeypatch):
    from agent import terminal_env_registry
    from tools.environments.local import build_subprocess_env

    # A terminal backend owns this credential: the scrub removes it, so the session env may not add it back.
    monkeypatch.setattr(terminal_env_registry, "plugin_strip_env_keys", lambda: frozenset({"ACME_SANDBOX_TOKEN"}))
    plugin.PANES["a"] = {"PANE": "pane-a", "OPENAI_API_KEY": "sk-from-plugin", "ACME_SANDBOX_TOKEN": "tok"}
    child = _as_session("a", build_subprocess_env)

    assert plugin.REFUSED == ["HERMES_SESSION_ID"]
    assert (child["PANE"], child["HERMES_SESSION_ID"]) == ("pane-a", "a")
    assert not {"OPENAI_API_KEY", "ACME_SANDBOX_TOKEN"} & child.keys()


@pytest.mark.platforms("windows")
def test_a_lowercase_name_replaces_or_unsets_the_inherited_windows_variable(monkeypatch):
    """Windows names are case-insensitive: ``pane`` acts on the inherited ``PANE``, with no second alias."""
    from hermes_constants import hermes_home_key
    from tools.environments.local import build_subprocess_env

    monkeypatch.setattr(plugins_session_env, "_REGISTRATIONS", {})
    monkeypatch.setattr(plugins_session_env, "_DECLARED", {})
    monkeypatch.setenv("PANE", "gateway")
    monkeypatch.setenv("WINDOW", "gateway-win")
    values = {"a": {"pane": "pane-a", "window": None}}
    plugins_session_env.register_session_env(hermes_home_key(), ["pane", "window"], values.get)
    child = _as_session("a", build_subprocess_env)
    assert [k for k in child if k.upper() in {"PANE", "WINDOW"}] == ["PANE"] and child["PANE"] == "pane-a"


def test_an_unloaded_plugins_names_stay_out_of_the_shell_snapshot(monkeypatch):
    """A snapshot taken while a value was injected may hold it: after the plugin unloads, its names
    stay excluded, so no later command restores another session's value."""
    from hermes_constants import hermes_home_key

    monkeypatch.setattr(plugins_session_env, "_REGISTRATIONS", {})
    monkeypatch.setattr(plugins_session_env, "_DECLARED", {})
    release = plugins_session_env.register_session_env(hermes_home_key(), ["PANE"], lambda sid: {"PANE": sid})
    release()
    assert "PANE" in plugins_session_env.session_env_names()


_HOLDER = """
def register(ctx):
    ctx.register_session_env(['PANE'], lambda sid: {'PANE': 'holder-' + sid})
"""
_TAKER = """
def register(ctx):
    ctx.register_session_env(['WINDOW', 'PANE'], lambda sid: {'PANE': 'taker', 'WINDOW': 'taker'})
"""


def _home(root, plugins, isolation="in_process"):
    for name, body in plugins.items():
        plugin_dir = root / "plugins" / name
        plugin_dir.mkdir(parents=True)
        (plugin_dir / "plugin.yaml").write_text(f"name: {name}\nversion: 0.1.0\n", encoding="utf-8")
        (plugin_dir / "__init__.py").write_text(body, encoding="utf-8")
    (root / "config.yaml").write_text(
        f"plugins:\n  enabled: [{', '.join(plugins)}]\n  isolation: {isolation}\n", encoding="utf-8")
    return root


@pytest.mark.parametrize("isolation", ["in_process", pytest.param("host", marks=pytest.mark.platforms("posix"))])
def test_a_second_plugin_cannot_take_a_declared_name(tmp_path, monkeypatch, isolation):
    """One value per name: the later plugin is refused, naming the holder, instead of a silent last-wins.
    The holder reloads, and its unload frees the name."""
    from hermes_cli import plugins as plugins_mod

    monkeypatch.setattr(plugins_session_env, "_REGISTRATIONS", {})
    monkeypatch.setattr(plugins_session_env, "_DECLARED", {})
    monkeypatch.setattr(plugins_mod, "get_bundled_plugins_dir", lambda: tmp_path / "no-bundled")
    monkeypatch.setenv("HOME", str(tmp_path / "os-home"))
    monkeypatch.setenv("HERMES_HOME", str(_home(tmp_path / "home", {"aaa_holder": _HOLDER, "zzz_taker": _TAKER},
                                                  isolation)))
    plugins_mod._reset_plugin_managers_for_tests()
    manager = plugins_mod.get_plugin_manager()
    try:
        manager.discover_and_load()
        assert manager._plugins["aaa_holder"].error is None
        assert "session env 'PANE' is already registered by 'aaa_holder'" in (manager._plugins["zzz_taker"].error or "")
        assert plugins_session_env.resolve_session_env("s") == {"PANE": "holder-s"}

        manager.discover_and_load(force=True)
        assert manager._plugins["aaa_holder"].error is None
        assert plugins_session_env.resolve_session_env("s") == {"PANE": "holder-s"}

        manager.unload("aaa_holder")
        assert plugins_session_env.resolve_session_env("s") == {}
        plugins_session_env.register_session_env(manager.scope_key, ["PANE"], lambda sid: {"PANE": "next"})
        assert plugins_session_env.resolve_session_env("s") == {"PANE": "next"}
    finally:
        plugins_mod._reset_plugin_managers_for_tests()


_TWICE = """
def register(ctx):
    ctx.register_session_env(['PANE'], lambda sid: {'PANE': 'first'})
    ctx.register_session_env(['PANE', 'WINDOW'], lambda sid: {'PANE': 'second', 'WINDOW': 'second'})
"""


@pytest.mark.parametrize("disposed", [0, 1])
@pytest.mark.parametrize("isolation", ["in_process", pytest.param("host", marks=pytest.mark.platforms("posix"))])
def test_one_plugin_declaring_a_name_twice_keeps_the_later_value(tmp_path, monkeypatch, isolation, disposed):
    """Only another plugin is refused: one plugin's overlapping registrations load, the later one's value
    wins, and disposing either one leaves the other's value."""
    from hermes_cli import plugins as plugins_mod

    monkeypatch.setattr(plugins_session_env, "_REGISTRATIONS", {})
    monkeypatch.setattr(plugins_session_env, "_DECLARED", {})
    monkeypatch.setattr(plugins_mod, "get_bundled_plugins_dir", lambda: tmp_path / "no-bundled")
    monkeypatch.setenv("HOME", str(tmp_path / "os-home"))
    monkeypatch.setenv("HERMES_HOME", str(_home(tmp_path / "home", {"twice": _TWICE}, isolation)))
    plugins_mod._reset_plugin_managers_for_tests()
    manager = plugins_mod.get_plugin_manager()
    try:
        manager.discover_and_load()
        assert manager._plugins["twice"].error is None
        assert plugins_session_env.resolve_session_env("s") == {"PANE": "second", "WINDOW": "second"}
        regs = [r for r in manager._ownership_ledger["twice"] if r.kind == "session_env"]
        regs[disposed].dispose()
        assert plugins_session_env.resolve_session_env("s") == (
            {"PANE": "second", "WINDOW": "second"} if disposed == 0 else {"PANE": "first"})
    finally:
        plugins_mod._reset_plugin_managers_for_tests()


@pytest.mark.platforms("posix")  # the shared bash snapshot path
def test_a_profiles_names_never_reach_another_profiles_shells(tmp_path, monkeypatch):
    """A multiplexed gateway serves two homes: A's plugin declaring PANE leaves B's own PANE export alone
    (it still carries to B's next command), and B may declare the same name for its own sessions."""
    from hermes_constants import hermes_home_key, reset_hermes_home_override, set_hermes_home_override
    from tools.environments.local import LocalEnvironment

    monkeypatch.setattr(plugins_session_env, "_REGISTRATIONS", {})
    monkeypatch.setattr(plugins_session_env, "_DECLARED", {})
    monkeypatch.delenv("PANE", raising=False)
    home_a, home_b = tmp_path / "a", tmp_path / "b"
    home_a.mkdir()
    home_b.mkdir()
    plugins_session_env.register_session_env(hermes_home_key(home_a), ["PANE"], lambda sid: {"PANE": "a-" + sid})

    def as_profile(home, fn):
        token = set_hermes_home_override(home)
        try:
            return fn()
        finally:
            reset_hermes_home_override(token)

    assert as_profile(home_a, plugins_session_env.session_env_names) == {"PANE"}
    assert as_profile(home_b, plugins_session_env.session_env_names) == frozenset()
    shell = as_profile(home_b, lambda: LocalEnvironment(cwd=str(tmp_path), timeout=30))
    try:
        run = lambda command: as_profile(home_b, lambda: _as_session("s", lambda: shell.execute(command)["output"].strip()))
        run("export PANE=b-own")
        assert run('echo "${PANE-unset}"') == "b-own"
    finally:
        shell.cleanup()

    # A different plugin from A's: only the scope check lets B declare PANE too.
    plugins_session_env.register_session_env(hermes_home_key(home_b), ["PANE"], lambda sid: {"PANE": "b-" + sid},
                                             owner="b_plugin")
    assert as_profile(home_a, lambda: plugins_session_env.resolve_session_env("s")) == {"PANE": "a-s"}
    assert as_profile(home_b, lambda: plugins_session_env.resolve_session_env("s")) == {"PANE": "b-s"}
