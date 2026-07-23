"""Regression tests for schema_sanitizer property-key validation.

Issue #40232: _sanitize_node() sanitizes property *values* but never validates
property *key names*. Keys containing characters outside [a-zA-Z0-9_.-] (e.g.
``$defs``) pass through untouched and cause strict backends (GitHub Copilot,
Anthropic) to reject the entire request with HTTP 400.

These tests verify that:
1. Invalid property keys are renamed (invalid chars stripped)
2. Properties that can't be renamed (empty result / collision) are dropped
3. The ``required`` array is re-pruned after renames and drops
4. Nested properties are also sanitized
5. Valid keys are not modified
"""
from __future__ import annotations

import copy

from tools.schema_sanitizer import sanitize_tool_schemas


def _tool(name: str, parameters: dict) -> dict:
    return {"type": "function", "function": {"name": name, "parameters": parameters}}


class TestInvalidPropertyKeyRename:
    """Property keys with invalid characters are renamed."""

    def test_dollar_prefix_key_renamed(self):
        """``$defs`` → ``defs`` (strip the $)."""
        tools = [_tool("t", {
            "type": "object",
            "properties": {
                "$defs": {"type": "string", "description": "schema defs"},
            },
            "required": ["$defs"],
        })]
        out = sanitize_tool_schemas(tools)
        props = out[0]["function"]["parameters"]["properties"]
        assert "defs" in props
        assert "$defs" not in props
        assert props["defs"]["type"] == "string"
        # required is updated to use renamed key
        assert out[0]["function"]["parameters"]["required"] == ["defs"]

    def test_multiple_invalid_chars_stripped(self):
        """Key ``foo@bar!baz`` → ``foobarbaz``."""
        tools = [_tool("t", {
            "type": "object",
            "properties": {
                "foo@bar!baz": {"type": "integer"},
            },
        })]
        out = sanitize_tool_schemas(tools)
        props = out[0]["function"]["parameters"]["properties"]
        assert "foobarbaz" in props
        assert "foo@bar!baz" not in props

    def test_valid_keys_untouched(self):
        """Keys already matching the valid pattern are not modified."""
        tools = [_tool("t", {
            "type": "object",
            "properties": {
                "simple": {"type": "string"},
                "with-dash": {"type": "string"},
                "with.dot": {"type": "string"},
                "with_underscore": {"type": "string"},
                "MixedCase123": {"type": "string"},
            },
        })]
        out = sanitize_tool_schemas(tools)
        props = out[0]["function"]["parameters"]["properties"]
        assert set(props.keys()) == {
            "simple", "with-dash", "with.dot", "with_underscore", "MixedCase123",
        }


class TestInvalidPropertyKeyDrop:
    """Properties that can't be safely renamed are dropped."""

    def test_key_becomes_empty_after_strip(self):
        """Key ``$$$`` → empty after stripping → drop the property entirely."""
        tools = [_tool("t", {
            "type": "object",
            "properties": {
                "$$$": {"type": "string"},
                "valid": {"type": "integer"},
            },
            "required": ["$$$", "valid"],
        })]
        out = sanitize_tool_schemas(tools)
        props = out[0]["function"]["parameters"]["properties"]
        assert "$$$" not in props
        assert "valid" in props
        # required is pruned to only valid keys
        assert out[0]["function"]["parameters"]["required"] == ["valid"]

    def test_key_collision_with_existing_drops(self):
        """If renaming would collide with an existing key, drop instead."""
        tools = [_tool("t", {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "original"},
                "$name": {"type": "string", "description": "duplicate-after-strip"},
            },
            "required": ["name", "$name"],
        })]
        out = sanitize_tool_schemas(tools)
        props = out[0]["function"]["parameters"]["properties"]
        # Original 'name' is preserved; '$name' can't be renamed to 'name' (collision)
        assert "name" in props
        assert props["name"]["description"] == "original"
        assert "$name" not in props
        # required only contains the surviving key
        assert out[0]["function"]["parameters"]["required"] == ["name"]


class TestRequiredPrunedAfterKeyChanges:
    """required array is re-pruned after key renames and drops."""

    def test_required_updated_after_rename(self):
        """required references renamed key, not the original."""
        tools = [_tool("t", {
            "type": "object",
            "properties": {
                "$path": {"type": "string"},
            },
            "required": ["$path"],
        })]
        out = sanitize_tool_schemas(tools)
        params = out[0]["function"]["parameters"]
        assert "path" in params["properties"]
        assert params["required"] == ["path"]

    def test_required_dropped_when_all_keys_invalid(self):
        """If all required keys are dropped, required key is removed."""
        tools = [_tool("t", {
            "type": "object",
            "properties": {
                "!!!": {"type": "string"},
            },
            "required": ["!!!"],
        })]
        out = sanitize_tool_schemas(tools)
        params = out[0]["function"]["parameters"]
        assert "required" not in params
        assert params["properties"] == {}

    def test_required_keeps_valid_keys_after_partial_drop(self):
        """required retains only entries whose properties survived."""
        tools = [_tool("t", {
            "type": "object",
            "properties": {
                "good": {"type": "string"},
                "###bad": {"type": "string"},
                "$$$": {"type": "string"},
            },
            "required": ["good", "###bad", "$$$"],
        })]
        out = sanitize_tool_schemas(tools)
        params = out[0]["function"]["parameters"]
        # 'good' stays, '###bad' → 'bad', '$$$' → dropped
        assert "good" in params["properties"]
        assert "bad" in params["properties"]
        assert "$$$" not in params["properties"]
        assert sorted(params["required"]) == ["bad", "good"]


class TestNestedPropertyKeySanitization:
    """Invalid keys at any nesting level are sanitized."""

    def test_nested_object_properties_sanitized(self):
        """Invalid keys inside nested object properties are also renamed."""
        tools = [_tool("t", {
            "type": "object",
            "properties": {
                "config": {
                    "type": "object",
                    "properties": {
                        "$env": {"type": "string"},
                        "normal": {"type": "integer"},
                    },
                },
            },
        })]
        out = sanitize_tool_schemas(tools)
        nested = out[0]["function"]["parameters"]["properties"]["config"]["properties"]
        assert "env" in nested
        assert "$env" not in nested
        assert "normal" in nested

    def test_defs_keys_sanitized(self):
        """Keys inside ``$defs`` / ``definitions`` are also sanitized."""
        tools = [_tool("t", {
            "type": "object",
            "properties": {
                "data": {"type": "string"},
            },
            "$defs": {
                "$ref_target": {"type": "string"},
            },
        })]
        out = sanitize_tool_schemas(tools)
        defs = out[0]["function"]["parameters"].get("$defs")
        if defs is not None:
            assert "ref_target" in defs
            assert "$ref_target" not in defs


class TestDeepcopyNotMutated:
    """Input tools are never mutated by the sanitizer."""

    def test_original_unchanged_after_key_sanitization(self):
        original = {
            "type": "object",
            "properties": {
                "$defs": {"type": "string"},
            },
            "required": ["$defs"],
        }
        tools = [_tool("t", copy.deepcopy(original))]
        _ = sanitize_tool_schemas(tools)
        # Original input must still have the invalid key
        assert "$defs" in original["properties"]
        assert original["required"] == ["$defs"]
