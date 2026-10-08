"""Required guards stay profile scoped and legacy RPC opt-in remains explicit."""
from hermes_constants import reset_hermes_home_override, set_hermes_home_override
from hermes_cli import plugins as pmod
from tools.required_plugin_gate import required_tool_block
from tools.registry import registry


def _write_profile(home, enabled):
    home.mkdir()
    (home / "config.yaml").write_text(
        "plugins:\n  enabled: " + str(enabled) + "\n  required:\n"
        "    frozen-guard:\n      tools: [terminal, execute_code]\n"
    )
    plugin = home / "plugins" / "frozen-guard"
    plugin.mkdir(parents=True)
    (plugin / "plugin.yaml").write_text("name: frozen-guard\nversion: 1.0.0\n")
    (plugin / "__init__.py").write_text('''
def register(ctx):
    for name, flag in (("guard_allowed", True), ("guard_denied", False), ("guard_truthy", "yes")):
        ctx.register_tool(name, "frozen_guard", {"name": name, "parameters": {"type": "object"}},
                          lambda args, **kwargs: "{}", allow_code_execution=flag)
''')


def test_real_discovery_required_guard_and_rpc_opt_in_across_profiles(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    _write_profile(a, ["frozen-guard"])
    _write_profile(b, [])
    try:
        for home, loaded in ((a, True), (b, False), (a, True)):
            token = set_hermes_home_override(str(home))
            try:
                pmod.discover_plugins()
                manager = pmod.get_plugin_manager()
                assert ("frozen-guard" in manager.loaded_plugin_names()) is loaded
                directive = pmod._get_pre_tool_call_directive_details("terminal", {})
                assert (directive.action == "block") is not loaded
                assert required_tool_block("read_file") is None
                if loaded:
                    assert registry.get_entry("guard_allowed").allow_code_execution is True
                    assert registry.get_entry("guard_denied").allow_code_execution is False
                    assert registry.get_entry("guard_truthy").allow_code_execution is False
                    assert registry.dispatch("guard_allowed", {}) == "{}"
            finally:
                reset_hermes_home_override(token)
    finally:
        pmod._reset_plugin_managers_for_tests()


def test_required_guard_rejects_failed_deferred_or_unimported_plugins(tmp_path):
    home = tmp_path / "profile"
    _write_profile(home, ["frozen-guard"])
    token = set_hermes_home_override(str(home))
    try:
        pmod.discover_plugins()
        plugin = pmod.get_plugin_manager()._plugins["frozen-guard"]
        for field, value in (("error", "registration failed"), ("deferred", True), ("module", None)):
            previous = getattr(plugin, field)
            setattr(plugin, field, value)
            try:
                assert "not loaded" in required_tool_block("terminal")
            finally:
                setattr(plugin, field, previous)
        (home / "config.yaml").write_text("plugins: {}\n")
        assert required_tool_block("terminal") is None
    finally:
        reset_hermes_home_override(token)
        pmod._reset_plugin_managers_for_tests()
