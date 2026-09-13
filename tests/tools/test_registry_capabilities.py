"""Shared capability metadata on ToolEntry/registry (fail-closed `unknown` default)."""

import json

import pytest

import tools.file_tools  # noqa: F401  (triggers production registrations)
from tools.registry import TOOL_CAPABILITIES, UNKNOWN_CAPABILITY, ToolRegistry, registry


def _dummy_handler(args, **kwargs):
    return json.dumps({"ok": True})


def _make_schema(name="cap_tool"):
    return {
        "name": name,
        "description": f"A {name}",
        "parameters": {"type": "object", "properties": {}},
    }


@pytest.fixture
def reg():
    return ToolRegistry()


def _register(reg, name="cap_tool", **kwargs):
    reg.register(
        name=name,
        toolset="core",
        schema=_make_schema(name),
        handler=_dummy_handler,
        **kwargs,
    )
    return reg.get_entry(name)


def test_canonical_set_contains_fail_closed_marker(reg):
    """`unknown` is the reserved fail-closed marker of the canonical set."""
    assert UNKNOWN_CAPABILITY in TOOL_CAPABILITIES
    assert all(isinstance(cap, str) for cap in TOOL_CAPABILITIES)


def test_omitted_capabilities_fail_closed_to_unknown(reg):
    """Existing (declaration-less) registrations are `unknown`, never `read`."""
    entry = _register(reg)
    assert entry.capabilities == (UNKNOWN_CAPABILITY,)


def test_explicit_capabilities_normalized_and_deduped(reg):
    entry = _register(
        reg,
        capabilities=["read", "mutate", "read", "secrets", "mutate", "read"],
    )
    assert entry.capabilities == ("read", "mutate", "secrets")


def test_explicit_unknown_is_valid(reg):
    assert _register(reg, capabilities=[UNKNOWN_CAPABILITY]).capabilities == (
        UNKNOWN_CAPABILITY,
    )


@pytest.mark.parametrize("bad", [
    [],                # empty declaration
    ["read", "bogus"],  # non-canonical tag
    ["read", 123],      # non-string tag
    ["READ"],           # case-sensitive: must match canonical set exactly
])
def test_invalid_capabilities_rejected_and_nothing_registered(reg, bad):
    with pytest.raises(ValueError):
        reg.register(
            name="cap_tool",
            toolset="core",
            schema=_make_schema(),
            handler=_dummy_handler,
            capabilities=bad,
        )
    assert reg.get_entry("cap_tool") is None


def test_get_entry_returns_capability_metadata(reg):
    _register(reg, capabilities=["read", "money"])
    entry = reg.get_entry("cap_tool")
    assert entry is not None
    assert entry.capabilities == ("read", "money")


def test_override_and_restore_retain_each_entries_capabilities(reg):
    """re-registration and restore_registration keep the metadata of the winner."""
    _register(reg, capabilities=["read", "mutate"])
    first = reg.get_entry("cap_tool")
    _register(reg, capabilities=["secrets"])
    second = reg.get_entry("cap_tool")
    assert second is not first
    assert second.capabilities == ("secrets",)

    assert reg.restore_registration("cap_tool", current=second, previous=first) is True
    restored = reg.get_entry("cap_tool")
    assert restored is first
    assert restored.capabilities == ("read", "mutate")


def test_core_file_tools_declare_exact_capabilities():
    """The four core file tools classify read/mutate exactly as declared."""
    expected = {
        "read_file": ("read",),
        "search_files": ("read",),
        "write_file": ("mutate",),
        "patch": ("mutate",),
    }
    for name, caps in expected.items():
        entry = registry.get_entry(name)
        assert entry is not None
        assert entry.capabilities == caps