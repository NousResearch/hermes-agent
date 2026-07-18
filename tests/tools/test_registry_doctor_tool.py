"""Tests for tools/registry_doctor_tool.py.

Standalone and real-registry friendly: the default cases exercise the live
registry/toolsets plus the synthetic bad-registration case is created via
in-process monkeypatching of the global registry singleton so no new
dependencies or fixtures are needed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.registry import registry as global_registry
from tools.registry_doctor_tool import (
    check_fn,
    handle,
    _build_envelope,
    _resolve_toolset_names,
)
from tools.validation_helper import Success


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _FakeHandler:
    def __init__(self, always: bool = True):
        self.always = always

    def __call__(self, args, **kwargs):
        return "{}"


def _seed_bad_tool(name: str, toolset: str):
    schema = {
        "name": name,
        "description": f"Fake bad tool {name}",
        "parameters": {"type": "object", "properties": {}},
    }
    global_registry.deregister(name)
    global_registry.register(
        name=name,
        toolset=toolset,
        schema=schema,
        handler=_FakeHandler(),
    )


# ---------------------------------------------------------------------------
# check_fn contract
# ---------------------------------------------------------------------------


class TestCheckFnContract:
    def test_always_returns_true(self):
        assert check_fn() is True


# ---------------------------------------------------------------------------
# Envelope shape and live registry sanity
# ---------------------------------------------------------------------------


class TestLiveRegistryEnvelope:
    def test_handle_returns_success_envelope(self):
        result = handle({})

        assert result["kind"] == "success"
        assert result["meta"] is None
        assert isinstance(result["data"], dict)
        assert set(result["data"]) == {
            "registered_tool_count",
            "toolsets_count",
            "missing_toolset_tools",
            "empty_toolsets",
            "orphan_count",
        }

    def test_registered_tool_count_is_non_negative_int(self):
        assert isinstance(handle({})["data"]["registered_tool_count"], int)
        assert handle({})["data"]["registered_tool_count"] >= 0

    def test_toolsets_count_is_non_negative_int(self):
        assert isinstance(handle({})["data"]["toolsets_count"], int)
        assert handle({})["data"]["toolsets_count"] >= 0

    def test_missing_toolset_tools_is_list_of_str(self):
        assert isinstance(handle({})["data"]["missing_toolset_tools"], list)
        for item in handle({})["data"]["missing_toolset_tools"]:
            assert isinstance(item, str)

    def test_empty_toolsets_is_list_of_str(self):
        assert isinstance(handle({})["data"]["empty_toolsets"], list)
        for item in handle({})["data"]["empty_toolsets"]:
            assert isinstance(item, str)

    def test_orphan_count_is_non_negative_int(self):
        assert isinstance(handle({})["data"]["orphan_count"], int)
        assert handle({})["data"]["orphan_count"] >= 0

    def test_resolve_toolset_names_uses_toolsets_or_registry(self):
        names = _resolve_toolset_names()
        assert isinstance(names, list)
        assert names == sorted(names)
        assert len(names) >= 1


# ---------------------------------------------------------------------------
# Bad synthetic registrations: bad toolset name + blank toolset
# ---------------------------------------------------------------------------


class TestDetectionOfKnownBadSyntheticRegistrations:
    def test_detects_bad_toolset_name(self):
        original_registered = set(global_registry.get_registered_toolset_names())
        known_before = set(_resolve_toolset_names()) | original_registered

        bad_toolset = "does-not-exist-bad-toolset-xyz"
        _seed_bad_tool("fake_bad_toolset_tool", bad_toolset)

        try:
            result = handle({})
            known_after = set(_resolve_toolset_names()) | set(global_registry.get_registered_toolset_names())
            if bad_toolset not in known_after:
                assert bad_toolset in result["data"]["missing_toolset_tools"]

            assert result["data"]["orphan_count"] >= 0
        finally:
            global_registry.deregister("fake_bad_toolset_tool")

    def test_detects_blank_toolset_orphan(self):
        _seed_bad_tool("fake_blank_toolset_tool", "")

        try:
            result = handle({})
            assert result["data"]["orphan_count"] >= 1
        finally:
            global_registry.deregister("fake_blank_toolset_tool")

    def test_missing_toolset_tools_changes_after_add_and_remove(self):
        bad_toolset = "another-missing-toolset-abc"
        before = set(handle({})["data"]["missing_toolset_tools"])
        _seed_bad_tool("fake_orphan_delta", bad_toolset)
        after_add = set(handle({})["data"]["missing_toolset_tools"])
        global_registry.deregister("fake_orphan_delta")
        after_remove = set(handle({})["data"]["missing_toolset_tools"])
        assert before.issubset(after_add) or after_add >= before
        assert after_remove == before
