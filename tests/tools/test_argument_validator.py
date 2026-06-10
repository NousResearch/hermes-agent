"""Tests for tools.argument_validator."""

from unittest.mock import MagicMock

from tools.argument_validator import validate_tool_arguments


class TestMissingRequired:
    def test_missing_required_returns_false(self):
        registry = MagicMock()
        entry = MagicMock()
        entry.schema = {
            "parameters": {
                "type": "object",
                "required": ["path"],
                "properties": {"path": {"type": "string"}},
            }
        }
        registry.get_entry.return_value = entry

        ok, err = validate_tool_arguments("read_file", {}, registry)
        assert ok is False
        assert "Missing required" in err
        assert "path" in err

    def test_optional_missing_returns_true(self):
        registry = MagicMock()
        entry = MagicMock()
        entry.schema = {
            "parameters": {
                "type": "object",
                "properties": {"limit": {"type": "integer"}},
            }
        }
        registry.get_entry.return_value = entry

        ok, err = validate_tool_arguments("read_file", {"path": "/tmp"}, registry)
        assert ok is True
        assert err == ""


class TestPlaceholderDetection:
    def test_placeholder_value_returns_false(self):
        registry = MagicMock()
        entry = MagicMock()
        entry.schema = {"parameters": {"type": "object", "properties": {}}}
        registry.get_entry.return_value = entry

        cases = [
            "your_api_key_here",
            "/path/to/your/",
            "<INSERT>",
            "TODO",
            "PLACEHOLDER",
            "example.com",
        ]
        for value in cases:
            ok, err = validate_tool_arguments("read_file", {"path": value}, registry)
            assert ok is False, f"expected block for {value}"
            assert "placeholder" in err.lower()


class TestPathExistence:
    def test_missing_path_returns_false(self):
        registry = MagicMock()
        entry = MagicMock()
        entry.schema = {
            "parameters": {
                "type": "object",
                "required": ["path"],
                "properties": {"path": {"type": "string"}},
            }
        }
        registry.get_entry.return_value = entry

        ok, err = validate_tool_arguments(
            "read_file", {"path": "/does/not/exist/at/all"}, registry
        )
        assert ok is False
        assert "File not found" in err

    def test_existing_path_returns_true(self, tmp_path):
        registry = MagicMock()
        entry = MagicMock()
        entry.schema = {
            "parameters": {
                "type": "object",
                "required": ["path"],
                "properties": {"path": {"type": "string"}},
            }
        }
        registry.get_entry.return_value = entry

        ok, err = validate_tool_arguments("read_file", {"path": str(tmp_path)}, registry)
        assert ok is True
        assert err == ""


class TestValidArguments:
    def test_known_tool_with_good_args(self):
        registry = MagicMock()
        entry = MagicMock()
        entry.schema = {
            "parameters": {
                "type": "object",
                "required": ["path"],
                "properties": {"path": {"type": "string"}, "limit": {"type": "integer"}},
            }
        }
        registry.get_entry.return_value = entry

        ok, err = validate_tool_arguments("read_file", {"path": "/tmp", "limit": 10}, registry)
        assert ok is True
        assert err == ""

    def test_non_string_values_are_ignored_for_placeholders(self):
        registry = MagicMock()
        entry = MagicMock()
        entry.schema = {"parameters": {"type": "object", "properties": {}}}
        registry.get_entry.return_value = entry

        ok, err = validate_tool_arguments(
            "write_file", {"path": 123, "content": None}, registry
        )
        assert ok is True
        assert err == ""


class TestUnknownTool:
    def test_unknown_tool_skips_required_check(self):
        registry = MagicMock()
        registry.get_entry.return_value = None

        ok, err = validate_tool_arguments("nonexistent_tool", {}, registry)
        assert ok is True
        assert err == ""


class TestTypeAndEnum:
    def test_type_mismatch_returns_false(self):
        registry = MagicMock()
        entry = MagicMock()
        entry.schema = {
            "parameters": {
                "type": "object",
                "properties": {
                    "limit": {"type": "integer"},
                    "offset": {"type": "number"},
                    "verbose": {"type": "boolean"},
                    "paths": {"type": "array"},
                    "config": {"type": "object"},
                },
            }
        }
        registry.get_entry.return_value = entry

        type_cases = [
            ("limit", "10", "integer"),
            ("offset", "5.5", "number"),
            ("verbose", "true", "boolean"),
            ("paths", "not_a_list", "array"),
            ("config", "not_an_object", "object"),
        ]
        for key, value, expected in type_cases:
            ok, err = validate_tool_arguments(
                "dummy_tool", {key: value}, registry
            )
            assert ok is False, f"expected type block for {key}={value!r}"
            assert "type mismatch" in err.lower()
            assert expected in err

    def test_enum_violation_returns_false(self):
        registry = MagicMock()
        entry = MagicMock()
        entry.schema = {
            "parameters": {
                "type": "object",
                "properties": {
                    "mode": {"type": "string", "enum": ["read", "write", "append"]},
                },
            }
        }
        registry.get_entry.return_value = entry

        ok, err = validate_tool_arguments(
            "dummy_tool", {"mode": "delete"}, registry
        )
        assert ok is False
        assert "not one of the allowed values" in err

    def test_correct_type_and_enum_pass(self):
        registry = MagicMock()
        entry = MagicMock()
        entry.schema = {
            "parameters": {
                "type": "object",
                "properties": {
                    "limit": {"type": "integer"},
                    "mode": {"type": "string", "enum": ["read", "write"]},
                    "ratio": {"type": "number"},
                    "enabled": {"type": "boolean"},
                },
            }
        }
        registry.get_entry.return_value = entry

        ok, err = validate_tool_arguments(
            "dummy_tool",
            {"limit": 10, "mode": "read", "ratio": 0.5, "enabled": True},
            registry,
        )
        assert ok is True
        assert err == ""

    def test_unknown_type_is_skipped(self):
        registry = MagicMock()
        entry = MagicMock()
        entry.schema = {
            "parameters": {
                "type": "object",
                "properties": {
                    "payload": {"type": "null"},
                    "data": {"type": "unknown_type"},
                },
            }
        }
        registry.get_entry.return_value = entry

        ok, err = validate_tool_arguments(
            "dummy_tool", {"payload": None, "data": "anything"}, registry
        )
        assert ok is True
        assert err == ""

    def test_missing_enum_field_passes_when_not_provided(self):
        registry = MagicMock()
        entry = MagicMock()
        entry.schema = {
            "parameters": {
                "type": "object",
                "properties": {
                    "mode": {"type": "string", "enum": ["read", "write"]},
                },
            }
        }
        registry.get_entry.return_value = entry

        ok, err = validate_tool_arguments("dummy_tool", {}, registry)
        assert ok is True
        assert err == ""
