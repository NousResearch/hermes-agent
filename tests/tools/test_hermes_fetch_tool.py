import json

from tools.registry import discover_builtin_tools, registry
from toolsets import resolve_toolset


def test_hermes_fetch_tool_is_discovered_and_in_core_toolset():
    discover_builtin_tools()

    entry = registry.get_entry("hermes_fetch")
    assert entry is not None
    assert entry.toolset == "hermes"
    assert "hermes_fetch" in resolve_toolset("hermes-cli")


def test_hermes_fetch_tool_returns_text_payload():
    discover_builtin_tools()

    payload = json.loads(registry.dispatch("hermes_fetch", {"format": "compact"}))

    assert payload["success"] is True
    assert payload["format"] == "compact"
    assert "Hermes" in payload["text"]
    assert "Version" in payload["text"]


def test_hermes_fetch_tool_returns_structured_payload():
    discover_builtin_tools()

    payload = json.loads(registry.dispatch("hermes_fetch", {"format": "json"}))

    assert payload["success"] is True
    assert payload["format"] == "json"
    assert payload["info"]["version"]
    assert "model" in payload["info"]


def test_hermes_fetch_tool_rejects_unknown_format():
    discover_builtin_tools()

    payload = json.loads(registry.dispatch("hermes_fetch", {"format": "xml"}))

    assert payload["success"] is False
    assert "format must be" in payload["error"]
