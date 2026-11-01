"""Delegation backends are per profile on a process that serves several (the multiplexed gateway).

Real plugin discovery through the per-home plugin managers against two temp homes; the profile is bound the
way the gateway binds a routed turn (the ``HERMES_HOME`` override contextvar).
"""

import contextlib
from pathlib import Path

import hermes_yaml as yaml

from hermes_cli import plugins as plugins_mod
from hermes_constants import reset_hermes_home_override, set_hermes_home_override

REPO_PLUGINS = Path(__file__).resolve().parents[2] / "plugins"

BACKEND_PLUGIN = '''
from agent.delegation_provider import DelegationProvider

class Backend(DelegationProvider):
    @property
    def name(self):
        return "{name}"

    def route_task(self, task, config):
        return {{"label": "{name}"}}

def register(ctx):
    {body}
'''


def _write_plugin(home: Path, name: str, *, body: str = "ctx.register_delegation_provider(Backend())") -> None:
    plugin_dir = home / "plugins" / name
    plugin_dir.mkdir(parents=True)
    (plugin_dir / "plugin.yaml").write_text(yaml.safe_dump({"name": name, "version": "1.0"}), encoding="utf-8")
    (plugin_dir / "__init__.py").write_text(BACKEND_PLUGIN.format(name=name, body=body), encoding="utf-8")


def _write_config(home: Path, enabled: list[str], backend: str) -> None:
    (home / "config.yaml").write_text(yaml.safe_dump(
        {"plugins": {"enabled": enabled}, "delegation": {"backend": backend}}), encoding="utf-8")


def _two_profiles(tmp_path, monkeypatch):
    root = tmp_path / ".hermes"
    profile_b = root / "profiles" / "b"
    profile_b.mkdir(parents=True)
    _write_plugin(root, "aplug")
    _write_config(root, ["aplug"], "aplug")
    _write_plugin(profile_b, "bplug")
    _write_config(profile_b, ["bplug"], "bplug")
    bundled = tmp_path / "bundled"
    bundled.mkdir()
    (bundled / "delegation").symlink_to(REPO_PLUGINS / "delegation", target_is_directory=True)
    monkeypatch.setenv("HOME", str(tmp_path / "os-home"))
    monkeypatch.setenv("HERMES_HOME", str(root))
    monkeypatch.setattr(plugins_mod, "get_bundled_plugins_dir", lambda: bundled)
    return root, profile_b


@contextlib.contextmanager
def _profile(home: Path):
    token = set_hermes_home_override(str(home))
    try:
        yield
    finally:
        reset_hermes_home_override(token)


def _status(name: str):
    from agent.delegation_registry import backend_status
    provider, reason = backend_status(name)
    return (provider.name if provider is not None else None), reason


def test_each_profile_routes_on_its_own_plugin_backend_a_b_a(tmp_path, monkeypatch):
    _root, profile_b = _two_profiles(tmp_path, monkeypatch)
    plugins_mod.discover_plugins()  # the launch profile discovers first, as a gateway boot does

    assert _status("aplug") == ("aplug", "")
    with _profile(profile_b):
        assert _status("bplug") == ("bplug", "")
        provider, reason = _status("aplug")
        assert provider is None and "'aplug' is not installed" in reason
    assert _status("aplug") == ("aplug", "")
    provider, reason = _status("bplug")
    assert provider is None and "'bplug' is not installed" in reason


def test_a_backend_plugin_enabled_after_discovery_is_named_with_the_load_fix(tmp_path, monkeypatch):
    _root, profile_b = _two_profiles(tmp_path, monkeypatch)
    with _profile(profile_b):
        plugins_mod.discover_plugins()
        _write_plugin(profile_b, "lateplug")
        _write_config(profile_b, ["bplug", "lateplug"], "lateplug")

        provider, reason = _status("lateplug")

    assert provider is None
    assert "This profile enables lateplug, which this process has not loaded" in reason
    assert "If one of them provides 'lateplug', restart Hermes (the gateway, when it serves this session)" in reason
    assert "bplug" not in reason.split("enables", 1)[1]  # a plugin that loaded is not blamed


def test_a_backend_plugin_that_failed_to_load_is_named(tmp_path, monkeypatch):
    _root, profile_b = _two_profiles(tmp_path, monkeypatch)
    _write_plugin(profile_b, "brokenplug", body='raise RuntimeError("boom")')
    _write_config(profile_b, ["bplug", "brokenplug"], "brokenplug")
    with _profile(profile_b):
        plugins_mod.discover_plugins()
        provider, reason = _status("brokenplug")

    assert provider is None
    assert "This profile enables brokenplug, which this process has not loaded" in reason
    assert "`hermes logs` says why" in reason


def test_an_alias_entry_and_a_disabled_entry_are_not_named_unloaded(tmp_path, monkeypatch):
    _root, profile_b = _two_profiles(tmp_path, monkeypatch)
    _write_plugin(profile_b, "langfuse")
    _write_plugin(profile_b, "bothplug")
    (profile_b / "config.yaml").write_text(yaml.safe_dump({"plugins": {
        "enabled": ["bplug", "observability/langfuse", "bothplug"], "disabled": ["bothplug"]}}), encoding="utf-8")
    with _profile(profile_b):
        plugins_mod.discover_plugins()
        assert _status("langfuse") == ("langfuse", "")  # the legacy key selected it, and it loaded
        _provider, reason = _status("ghost")

    assert "langfuse" not in reason and "bothplug" not in reason


def test_an_unrelated_unloaded_plugin_keeps_the_install_advice_for_a_missing_backend(tmp_path, monkeypatch):
    _root, profile_b = _two_profiles(tmp_path, monkeypatch)
    with _profile(profile_b):
        plugins_mod.discover_plugins()
        _write_plugin(profile_b, "lateplug")
        _write_config(profile_b, ["bplug", "lateplug"], "ghost")

        provider, reason = _status("ghost")

    assert provider is None
    assert "Install or enable the plugin that does (`hermes plugins list`)" in reason
    assert "This profile enables lateplug" in reason  # still named, as a possible cause
