"""``commands.catalog`` says when its plugin discovery failed (#134889 PLG-7, CLI-13).

The catalog degrades per loader, so a plugin discovery failure still answers successfully, with no
plugin commands. ``plugin_discovery_error`` tells a client that list is unknown rather than empty;
the classic CLI then refuses dispatch instead of letting a same-named skill or bundle run
(tests/hermes_cli/test_gateway_plugin_commands.py).
"""
from importlib import import_module
from types import SimpleNamespace

from tui_gateway.command_discovery import command_catalog


def _catalog(get_plugin_commands):
    stubs = {"hermes_cli.plugins": SimpleNamespace(get_plugin_commands=get_plugin_commands),
             "agent.skill_commands": SimpleNamespace(get_interactive_skill_commands=dict)}
    return command_catalog(load_cfg=dict, module_loader=lambda name: stubs.get(name) or import_module(name))


def test_a_failed_plugin_discovery_is_a_field_not_only_warning_text():
    def broken():
        raise RuntimeError("boom in register()")
    result = _catalog(broken)
    assert result["plugin_discovery_error"] == "boom in register()"
    assert "Plugin commands" not in {c["name"] for c in result["categories"]}


def test_a_working_plugin_discovery_leaves_the_field_empty():
    assert _catalog(dict)["plugin_discovery_error"] == ""
    result = _catalog(lambda: {"probe": {"description": "Probe"}})
    assert result["plugin_discovery_error"] == ""
    assert ["/probe", "Probe"] in next(c["pairs"] for c in result["categories"] if c["name"] == "Plugin commands")


def test_plugin_commands_lists_names_a_display_entry_hides():
    """``/logs`` is an Ink display entry, not a built-in, so a plugin may register it; the categories
    show Ink's row, and ``plugin_commands`` still names the plugin's for routing."""
    result = _catalog(lambda: {"logs": {"description": "Plugin logs"}, "probe": {"description": "Probe"}})
    assert result["plugin_commands"] == {"logs": "Plugin logs", "probe": "Probe"}
    plugin_rows = next(c["pairs"] for c in result["categories"] if c["name"] == "Plugin commands")
    assert ["/logs", "Plugin logs"] not in plugin_rows
