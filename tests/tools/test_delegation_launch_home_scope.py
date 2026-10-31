"""A multiplexed gateway registers every home's plugins into that home's scope bucket while a
served profile session resolves under the profile's own home: without reading the launch home's
registrations, a delegation backend installed in the launch home (where boot discovery ran) is
invisible to every served profile and ``delegate_task`` fails closed (#145746).

Real discovery through PluginManager against two temp homes (launch + served profile), the real
scope-keyed registration, the real resolution.
"""

from pathlib import Path

import hermes_yaml as yaml
import pytest

from hermes_cli import plugins as plugins_mod
from hermes_cli.plugins import PluginManager
from hermes_constants import reset_hermes_home_override, set_hermes_home_override

REPO_PLUGINS = Path(__file__).resolve().parents[2] / "plugins"

PROBE_PLUGIN = '''
from agent.delegation_provider import DelegationProvider

class ProbeOptions(DelegationProvider):
    @property
    def name(self):
        return "probe-options"

    def description(self, config):
        return "routes every task on the probe backend"

    def route_task(self, task, config):
        return dict(config)

def register(ctx):
    ctx.register_delegation_provider(ProbeOptions())
'''

UNAVAILABLE_PLUGIN = '''
from agent.delegation_provider import DelegationProvider

class ProbeDown(DelegationProvider):
    @property
    def name(self):
        return "probe-options"

    def is_available(self):
        return False

    def description(self, config):
        return "registered here but cannot run"

    def route_task(self, task, config):
        return dict(config)

def register(ctx):
    ctx.register_delegation_provider(ProbeDown())
'''


def _launch_home(tmp_path, monkeypatch, plugin_source=PROBE_PLUGIN):
    """The process's launch home: the user plugin installed and discovered here, HERMES_HOME pointing
    at it (what ``get_process_hermes_home`` reads), boot discovery run."""
    home = tmp_path / "launch-home"
    plugin_dir = home / "plugins" / "probe-options"
    plugin_dir.mkdir(parents=True)
    (plugin_dir / "plugin.yaml").write_text(yaml.safe_dump({"name": "probe-options", "version": "1.0"}), encoding="utf-8")
    (plugin_dir / "__init__.py").write_text(plugin_source, encoding="utf-8")
    (home / "config.yaml").write_text(yaml.safe_dump({"plugins": {"enabled": ["probe-options"]}}), encoding="utf-8")
    _shared_env(tmp_path, monkeypatch, home)
    PluginManager().discover_and_load()
    return home


def _profile_scope(tmp_path, monkeypatch, plugin_source=None):
    """A served profile's home: discovered under its own home override (the gateway's per-adapter
    pass), carrying only the bundled plugins unless the test installs one."""
    home = tmp_path / "profile-home"
    (home / "config.yaml").parent.mkdir(parents=True, exist_ok=True)
    (home / "config.yaml").write_text(
        yaml.safe_dump({"plugins": {"enabled": ["probe-options"] if plugin_source else []}}), encoding="utf-8")
    if plugin_source is not None:
        plugin_dir = home / "plugins" / "probe-options"
        plugin_dir.mkdir(parents=True)
        (plugin_dir / "plugin.yaml").write_text(yaml.safe_dump({"name": "probe-options", "version": "1.0"}), encoding="utf-8")
        (plugin_dir / "__init__.py").write_text(plugin_source, encoding="utf-8")
    token = set_hermes_home_override(home)
    PluginManager().discover_and_load()
    return home, token


def _shared_env(tmp_path, monkeypatch, launch_home):
    # Bundled plugins: only the delegation category (the default strategy is a bundled plugin).
    bundled = tmp_path / "bundled"
    bundled.mkdir(exist_ok=True)
    (bundled / "delegation").symlink_to(REPO_PLUGINS / "delegation", target_is_directory=True)
    monkeypatch.setenv("HOME", str(tmp_path / "os-home"))
    monkeypatch.setenv("HERMES_HOME", str(launch_home))
    monkeypatch.setattr(plugins_mod, "get_bundled_plugins_dir", lambda: bundled)


def test_a_profile_session_resolves_the_backend_the_launch_home_registered(tmp_path, monkeypatch):
    from agent.delegation_registry import backend_status, get_active_provider

    _launch_home(tmp_path, monkeypatch)
    _, token = _profile_scope(tmp_path, monkeypatch)
    try:
        provider, reason = backend_status("probe-options")
        assert provider is not None and reason == "", reason
        assert get_active_provider({"backend": "probe-options"}).name == "probe-options"
    finally:
        reset_hermes_home_override(token)


def test_the_active_home_registration_wins_over_the_launch_home(tmp_path, monkeypatch):
    """A profile that registers its own backend under the same name keeps its own: the fallback only
    fills the gaps."""
    from agent.delegation_registry import _registry, backend_status
    from hermes_constants import hermes_home_key

    _launch_home(tmp_path, monkeypatch)
    profile, token = _profile_scope(tmp_path, monkeypatch, plugin_source=PROBE_PLUGIN)
    try:
        provider, reason = backend_status("probe-options")
        assert provider is _registry.get_provider("probe-options", scope=hermes_home_key(profile))
        assert reason == ""
    finally:
        reset_hermes_home_override(token)


def test_an_unavailable_launch_home_backend_reports_unavailable_not_missing(tmp_path, monkeypatch):
    """The fix's read feeds the failure copy too: a backend registered in the launch home whose plugin
    reports it cannot run is 'unavailable here', never 'not installed'."""
    from agent.delegation_registry import backend_status

    _launch_home(tmp_path, monkeypatch, plugin_source=UNAVAILABLE_PLUGIN)
    _, token = _profile_scope(tmp_path, monkeypatch)
    try:
        provider, reason = backend_status("probe-options")
        assert provider is not None, reason
        assert "unavailable here" in reason and "not installed" not in reason, reason
    finally:
        reset_hermes_home_override(token)


def test_delegation_lists_the_launch_home_backends_under_a_profile(tmp_path, monkeypatch):
    """``/delegation`` under a served profile must offer what boot discovery registered: the launch
    home's registrations are selectable, with ``default`` still first."""
    from agent.delegation_backends import list_backends

    _launch_home(tmp_path, monkeypatch)
    _, token = _profile_scope(tmp_path, monkeypatch)
    try:
        names = [info.name for info in list_backends()]
        assert names[0] == "default" and "probe-options" in names, names
    finally:
        reset_hermes_home_override(token)
