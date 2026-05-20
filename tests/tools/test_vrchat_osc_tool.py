from __future__ import annotations

import json

from model_tools import get_tool_definitions
from toolsets import TOOLSETS, resolve_toolset
from tools.registry import discover_builtin_tools, registry
from tools import vrchat_osc_tool as vrchat


class FakeOscClient:
    def __init__(self):
        self.sent: list[tuple[str, object]] = []

    def send_message(self, address: str, value: object) -> None:
        self.sent.append((address, value))


def test_vrchat_toolset_is_opt_in_and_resolves_tools():
    assert "vrchat" in TOOLSETS
    assert set(resolve_toolset("vrchat")) == {
        "vrchat_chatbox",
        "vrchat_typing",
        "vrchat_avatar_param",
        "vrchat_send_osc",
        "vrchat_status",
    }


def test_chatbox_sends_official_osc_payload(monkeypatch):
    fake = FakeOscClient()
    monkeypatch.setattr(vrchat, "_get_client", lambda host, port: fake)

    result = vrchat.vrchat_chatbox("hello vrc", immediate=True, notify=False)

    assert result["success"] is True
    assert fake.sent == [("/chatbox/input", ["hello vrc", True, False])]


def test_chatbox_rejects_empty_and_overlong_text(monkeypatch):
    fake = FakeOscClient()
    monkeypatch.setattr(vrchat, "_get_client", lambda host, port: fake)

    assert vrchat.vrchat_chatbox(" ")["success"] is False
    result = vrchat.vrchat_chatbox("x" * 145)

    assert result["success"] is False
    assert result["length"] == 145
    assert fake.sent == []


def test_avatar_param_rejects_paths_and_sends_parameter(monkeypatch):
    fake = FakeOscClient()
    monkeypatch.setattr(vrchat, "_get_client", lambda host, port: fake)

    bad = vrchat.vrchat_avatar_param("avatar/parameters/Unsafe", 1)
    good = vrchat.vrchat_avatar_param("GestureLeft", 1)

    assert bad["success"] is False
    assert good["success"] is True
    assert fake.sent == [("/avatar/parameters/GestureLeft", 1)]


def test_raw_osc_validates_address_and_args(monkeypatch):
    fake = FakeOscClient()
    monkeypatch.setattr(vrchat, "_get_client", lambda host, port: fake)

    bad_address = vrchat.vrchat_send_osc("avatar/parameters/Test", [1])
    bad_arg = vrchat.vrchat_send_osc("/avatar/parameters/Test", [{"bad": "value"}])
    good = vrchat.vrchat_send_osc("/avatar/parameters/Test", [0.5])

    assert bad_address["success"] is False
    assert bad_arg["success"] is False
    assert good["success"] is True
    assert fake.sent == [("/avatar/parameters/Test", 0.5)]


def test_registry_handler_returns_json_string(monkeypatch):
    fake = FakeOscClient()
    monkeypatch.setattr(vrchat, "_get_client", lambda host, port: fake)

    raw = registry.dispatch("vrchat_typing", {"is_typing": True})
    parsed = json.loads(raw)

    assert parsed["success"] is True
    assert fake.sent == [("/chatbox/typing", True)]


def test_tool_definitions_include_vrchat_when_enabled():
    discover_builtin_tools()

    defs = get_tool_definitions(enabled_toolsets=["vrchat"], quiet_mode=True)
    names = {item["function"]["name"] for item in defs}

    assert {
        "vrchat_chatbox",
        "vrchat_typing",
        "vrchat_avatar_param",
        "vrchat_send_osc",
        "vrchat_status",
    }.issubset(names)


def test_vrchat_tools_are_not_in_hermes_cli_core():
    assert not set(resolve_toolset("vrchat")) & set(resolve_toolset("hermes-cli"))
