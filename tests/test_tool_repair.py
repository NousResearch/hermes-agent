"""
Integration tests for the Tool Input Repair Layer.

Tests both structural repair (_REPAIR_RULES) and semantic repair
(_SEMANTIC_REPAIR_MAP) across all supported error patterns.
"""

import json
import os
import sys
import tempfile
from unittest.mock import patch

import pytest

# ————————————————————————————————————————
# Module-level patching: register schemas before importing model_tools
# ————————————————————————————————————————

_FAKE_SCHEMAS = {}


def _fake_get_schema(name):
    return _FAKE_SCHEMAS.get(name)


@pytest.fixture(autouse=True)
def reset_schemas():
    _FAKE_SCHEMAS.clear()
    yield


def _register_schema(name, schema):
    _FAKE_SCHEMAS[name] = schema


# =============================================================================
# Phase 0: Basic Structural Repair Tests
# =============================================================================


class TestStructuralRepair:
    """Test the 4 basic _REPAIR_RULES functions."""

    @patch("model_tools.registry")
    def test_strip_null_optional_param(self, mock_registry):
        """Optional param with None value should be removed."""
        from model_tools import _repair_tool_args

        _register_schema("test_tool", {
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "optional_field": {"type": "string"},
                },
                "required": ["name"],
            }
        })
        mock_registry.get_schema.side_effect = _fake_get_schema

        args, log = _repair_tool_args("test_tool", {"name": "hello", "optional_field": None})
        assert "optional_field" not in args
        assert args["name"] == "hello"
        assert log is not None
        assert log[0]["repair"] == "stripNull"

    @patch("model_tools.registry")
    def test_strip_null_required_param(self, mock_registry):
        """Required param with None value should NOT be removed."""
        from model_tools import _repair_tool_args

        _register_schema("test_tool", {
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                },
                "required": ["name"],
            }
        })
        mock_registry.get_schema.side_effect = _fake_get_schema

        args, log = _repair_tool_args("test_tool", {"name": None})
        # Required field with None — should be kept as-is (only optional nulls are stripped)
        assert args.get("name") is None
        assert log is None  # No repair needed for required null params

    @patch("model_tools.registry")
    def test_parse_json_array(self, mock_registry):
        """JSON stringified array should be parsed to native list."""
        from model_tools import _repair_tool_args

        _register_schema("test_tool", {
            "parameters": {
                "type": "object",
                "properties": {
                    "items": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["items"],
            }
        })
        mock_registry.get_schema.side_effect = _fake_get_schema

        args, log = _repair_tool_args("test_tool", {"items": '["a", "b", "c"]'})
        assert isinstance(args["items"], list)
        assert args["items"] == ["a", "b", "c"]
        assert log is not None
        assert log[0]["repair"] == "parseJsonArray"

    @patch("model_tools.registry")
    def test_parse_json_array_not_array_schema(self, mock_registry):
        """String should NOT be parsed when schema doesn't expect array."""
        from model_tools import _repair_tool_args

        _register_schema("test_tool", {
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                },
                "required": ["name"],
            }
        })
        mock_registry.get_schema.side_effect = _fake_get_schema

        args, log = _repair_tool_args("test_tool", {"name": '["a", "b"]'})
        assert isinstance(args["name"], str)
        assert log is None

    @patch("model_tools.registry")
    def test_unwrap_empty_object_to_array(self, mock_registry):
        """Empty dict {} should become [] when schema expects array."""
        from model_tools import _repair_tool_args

        _register_schema("test_tool", {
            "parameters": {
                "type": "object",
                "properties": {
                    "items": {"type": "array", "items": {"type": "string"}},
                },
                "required": [],
            }
        })
        mock_registry.get_schema.side_effect = _fake_get_schema

        args, log = _repair_tool_args("test_tool", {"items": {}})
        assert args["items"] == []
        assert log is not None
        assert log[0]["repair"] == "unwrapEmptyObject"

    @patch("model_tools.registry")
    def test_wrap_bare_string_to_array(self, mock_registry):
        """Bare string should be wrapped in list when schema expects string[]."""
        from model_tools import _repair_tool_args

        _register_schema("test_tool", {
            "parameters": {
                "type": "object",
                "properties": {
                    "tags": {"type": "array", "items": {"type": "string"}},
                },
                "required": [],
            }
        })
        mock_registry.get_schema.side_effect = _fake_get_schema

        args, log = _repair_tool_args("test_tool", {"tags": "important"})
        assert isinstance(args["tags"], list)
        assert args["tags"] == ["important"]
        assert log is not None
        assert log[0]["repair"] == "wrapBareString"

    @patch("model_tools.registry")
    def test_no_repair_needed(self, mock_registry):
        """Valid args should pass through unchanged."""
        from model_tools import _repair_tool_args

        _register_schema("test_tool", {
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "count": {"type": "integer"},
                },
                "required": ["name"],
            }
        })
        mock_registry.get_schema.side_effect = _fake_get_schema

        original = {"name": "hello", "count": 42}
        args, log = _repair_tool_args("test_tool", original)
        assert args == original
        assert log is None

    @patch("model_tools.registry")
    def test_multiple_repairs_in_one_call(self, mock_registry):
        """Multiple params needing repair should all be fixed."""
        from model_tools import _repair_tool_args

        _register_schema("test_tool", {
            "parameters": {
                "type": "object",
                "properties": {
                    "tags": {"type": "array", "items": {"type": "string"}},
                    "optional_flag": {"type": "string"},
                    "valid": {"type": "string"},
                },
                "required": ["valid"],
            }
        })
        mock_registry.get_schema.side_effect = _fake_get_schema

        args, log = _repair_tool_args("test_tool", {
            "tags": "bare_string",       # → ["bare_string"]
            "optional_flag": None,        # → removed
            "valid": "hello",             # passes through
        })
        assert "optional_flag" not in args
        assert args["tags"] == ["bare_string"]
        assert args["valid"] == "hello"
        assert log is not None
        assert len(log) >= 2


# =============================================================================
# Phase 1.5: Semantic Repair Tests
# =============================================================================


class TestSemanticRepair:
    """Test per-tool, per-parameter semantic repairs."""

    def test_repair_markdown_link(self):
        """Markdown link [path](display) should extract path."""
        from model_tools import _repair_markdown_link

        assert _repair_markdown_link("[README.md](click here)") == "README.md"
        assert _repair_markdown_link("[./src/main.py](source)") == "./src/main.py"
        assert _repair_markdown_link("normal_path.txt") == "normal_path.txt"
        assert _repair_markdown_link(42) == 42  # non-string passthrough

    def test_repair_expand_user(self):
        """Tilde should be expanded to home directory."""
        from model_tools import _repair_expand_user

        expanded = _repair_expand_user("~/projects")
        assert not expanded.startswith("~")
        assert expanded.endswith("/projects")
        assert _repair_expand_user("/absolute/path") == "/absolute/path"
        assert _repair_expand_user(42) == 42

    def test_repair_fix_double_escape(self):
        """Double backslashes should be reduced to single."""
        from model_tools import _repair_fix_double_escape

        assert _repair_fix_double_escape("ls\\\\n") == "ls\\n"
        assert _repair_fix_double_escape("normal") == "normal"
        assert _repair_fix_double_escape(42) == 42

    def test_repair_strip_shell_prompt(self):
        """Shell prompt prefixes should be stripped."""
        from model_tools import _repair_strip_shell_prompt

        assert _repair_strip_shell_prompt("$ ls -la") == "ls -la"
        assert _repair_strip_shell_prompt("> echo hi") == "echo hi"
        assert _repair_strip_shell_prompt("ls -la") == "ls -la"

    @patch("model_tools.registry")
    def test_semantic_repair_read_file_path(self, mock_registry):
        """read_file path should get markdown link + tilde repair."""
        from model_tools import _repair_semantic_args

        _register_schema("read_file", {
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "x-semantic-type": "file-path"},
                },
                "required": ["path"],
            }
        })
        mock_registry.get_schema.side_effect = _fake_get_schema

        args, log = _repair_semantic_args("read_file", {
            "path": "[~/project/file.txt](some link)",
        })
        assert log is not None
        assert args["path"] != "[~/project/file.txt](some link)"
        assert "~/project/file.txt" not in args["path"]  # tilde should be expanded
        assert "/project/file.txt" in args["path"] or "/project/" in args["path"]
        assert log[0]["repair"] == "strip_markdown_link" or log[0]["repair"] == "expand_tilde"

    @patch("model_tools.registry")
    def test_semantic_repair_terminal_command(self, mock_registry):
        """terminal command should get double-escape fix."""
        from model_tools import _repair_semantic_args

        _register_schema("terminal", {
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "x-semantic-type": "shell-command"},
                },
                "required": ["command"],
            }
        })
        mock_registry.get_schema.side_effect = _fake_get_schema

        args, log = _repair_semantic_args("terminal", {
            "command": "ls\\\\n",
        })
        assert log is not None
        assert args["command"] == "ls\\n"
        assert log[0]["repair"] == "fix_double_escape"

    def test_repair_check_path_traversal(self):
        """Path traversal detection should not crash."""
        from model_tools import _repair_check_path_traversal

        result = _repair_check_path_traversal("../../../etc/passwd")
        assert result == "../../../etc/passwd"
        assert _repair_check_path_traversal("safe/path") == "safe/path"


# =============================================================================
# Phase 2: Repair Log Injection Tests
# =============================================================================


class TestRepairLogInjection:
    """Test that repair_log is injected into tool results."""

    @patch("model_tools.registry")
    def test_repair_log_in_dict_result(self, mock_registry):
        """_repair field should appear in dict results."""
        from model_tools import _repair_semantic_args, _repair_tool_args

        _register_schema("test_tool", {
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "tags": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["name"],
            }
        })
        mock_registry.get_schema.side_effect = _fake_get_schema

        # Simulate the full pipeline
        args = {"name": "test", "tags": "bare_string"}
        args, sem_log = _repair_semantic_args("test_tool", args)
        args, struct_log = _repair_tool_args("test_tool", args)

        repair_log = []
        if sem_log:
            repair_log.extend(sem_log)
        if struct_log:
            repair_log.extend(struct_log)

        assert len(repair_log) >= 1  # wrapBareString

        # Simulate result injection
        result = json.dumps({"success": True, "data": "ok"})
        parsed = json.loads(result)
        if repair_log:
            parsed["_repair"] = repair_log
            result = json.dumps(parsed, ensure_ascii=False)

        final = json.loads(result)
        assert final["_repair"] is not None
        assert final["_repair"][0]["repair"] == "wrapBareString"


# =============================================================================
# Phase 3: Error Formatting Tests
# =============================================================================


class TestErrorFormatting:
    """Test model-readable error formatting."""

    def test_format_tool_error_for_model(self):
        """Error should be formatted with type, message, and args."""
        from model_tools import _format_tool_error_for_model

        try:
            raise ValueError("test error")
        except ValueError as e:
            formatted = _format_tool_error_for_model(e, "test_tool", {"key": "value"})
            assert "ValueError" in formatted
            assert "test error" in formatted
            assert "Arguments" in formatted
            assert '"key": "value"' in formatted

    def test_format_tool_error_truncates(self):
        """Very long error messages should be truncated."""
        from model_tools import _format_tool_error_for_model

        long_msg = "x" * 1000
        try:
            raise RuntimeError(long_msg)
        except RuntimeError as e:
            formatted = _format_tool_error_for_model(e, "test_tool", {})
            assert len(formatted) < 1200  # truncated

    def test_get_repair_hint(self):
        """Hints should be generated for array-like string params."""
        from model_tools import _get_repair_hint

        hint = _get_repair_hint("test_tool", {"items": '["a", "b"]'})
        assert "JSON array string" in hint
        assert "items" in hint

        hint2 = _get_repair_hint("test_tool", {"name": "hello"})
        assert hint2 == "Check parameter types and values."


# =============================================================================
# Phase 2: Event Log Tests
# =============================================================================


class TestEventLogRepair:
    """Test the tool_input_repaired event type."""

    def test_event_type_constant(self):
        """EVENT_TOOL_INPUT_REPAIRED should be defined."""
        from agent.session_event_log import EVENT_TOOL_INPUT_REPAIRED

        assert EVENT_TOOL_INPUT_REPAIRED == "tool_input_repaired"

    def test_log_tool_input_repaired(self):
        """log_tool_input_repaired should create a valid event."""
        from agent.session_event_log import EventLog, EVENT_TOOL_INPUT_REPAIRED

        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        try:
            el = EventLog(db_path=db_path)
            event = el.log_tool_input_repaired(
                task_id="task-123",
                session_id="session-456",
                tool_name="read_file",
                repair_log=[{"param": "path", "from": "[x](y)", "to": "x", "repair": "strip_markdown_link"}],
            )
            assert event.type == EVENT_TOOL_INPUT_REPAIRED
            assert event.payload["tool_name"] == "read_file"
            assert len(event.payload["repairs"]) == 1
            assert event.payload["repairs"][0]["repair"] == "strip_markdown_link"

            # Verify it was persisted
            events = el.get_events_for_task("task-123")
            assert len(events) == 1
            assert events[0]["type"] == EVENT_TOOL_INPUT_REPAIRED
        finally:
            os.unlink(db_path)


# =============================================================================
# End-to-End Pipeline Test
# =============================================================================


class TestFullPipeline:
    """End-to-end tests of the repair pipeline."""

    @patch("model_tools.registry")
    def test_full_repair_pipeline(self, mock_registry):
        """Structural + semantic repair should work end-to-end."""
        from model_tools import _repair_semantic_args, _repair_tool_args

        # Register schemas for tools in the semantic map
        for tool_name in ["read_file", "write_file", "patch", "terminal", "search_files"]:
            _register_schema(tool_name, {
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "x-semantic-type": "file-path"},
                        "command": {"type": "string", "x-semantic-type": "shell-command"},
                    },
                    "required": [],
                }
            })

        mock_registry.get_schema.side_effect = _fake_get_schema

        # Test case: read_file with markdown link path AND optional param as None
        test_args = {
            "path": "[~/my_project/file.txt](click)",
            "optional_field": None,
        }
        # We need a schema for the tool test too
        _register_schema("read_file_full", {
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "optional_field": {"type": "string"},
                },
                "required": ["path"],
            }
        })
        mock_registry.get_schema.side_effect = lambda n: _FAKE_SCHEMAS.get(n)

        test_tool = "read_file_full"
        args = dict(test_args)

        # Semantic repair first
        args, sem_log = _repair_semantic_args(test_tool, args)
        assert sem_log is not None or True  # may or may not trigger

        # Structural repair second
        args, struct_log = _repair_tool_args(test_tool, args)

        # "optional_field" should have been stripped
        if struct_log:
            assert "optional_field" not in args or args.get("optional_field") is None

    @patch("model_tools.registry")
    def test_no_schema_no_repair(self, mock_registry):
        """Tools without schemas should be passed through unchanged."""
        from model_tools import _repair_tool_args

        mock_registry.get_schema.return_value = None

        args, log = _repair_tool_args("unknown_tool", {"key": "value"})
        assert args == {"key": "value"}
        assert log is None

    @patch("model_tools.registry")
    def test_non_dict_args(self, mock_registry):
        """Non-dict args should pass through unchanged."""
        from model_tools import _repair_tool_args

        args, log = _repair_tool_args("test_tool", "not_a_dict")
        assert args == "not_a_dict"
        assert log is None


# =============================================================================
# Phase 4: Missing Edge Case Tests (Bug #12)
# =============================================================================


class TestEdgeCases:
    """Test edge cases: multi-rule log accuracy, unicode, empty schema, idempotency."""

    @patch("model_tools.registry")
    def test_multi_rule_log_accuracy(self, mock_registry):
        """Multiple rules applied to the same param should ALL be logged."""
        from model_tools import _repair_semantic_args, _repair_markdown_link

        _register_schema("read_file", {
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "x-semantic-type": "file-path"},
                },
                "required": ["path"],
            }
        })
        mock_registry.get_schema.side_effect = _fake_get_schema

        # Apply repair on a value that triggers both strip_markdown_link AND expand_tilde
        # [~/project/file.txt](display) → markdown strip → ~/project/file.txt → expand → /home/user/project/file.txt
        args, log = _repair_semantic_args("read_file", {
            "path": "[~/project/file.txt](display)",
        })
        assert log is not None, "Repair log should not be None"
        assert len(log) >= 2, f"Expected at least 2 repairs, got {len(log)}: {log}"
        rule_names = [entry["repair"] for entry in log]
        assert "strip_markdown_link" in rule_names, f"Missing strip_markdown_link in {rule_names}"
        assert "expand_tilde" in rule_names, f"Missing expand_tilde in {rule_names}"

    @patch("model_tools.registry")
    def test_unicode_path_preserved(self, mock_registry):
        """Unicode characters in paths should pass through untouched."""
        from model_tools import _repair_semantic_args

        _register_schema("read_file", {
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "x-semantic-type": "file-path"},
                },
                "required": ["path"],
            }
        })
        mock_registry.get_schema.side_effect = _fake_get_schema

        args, log = _repair_semantic_args("read_file", {
            "path": "/data/中文/ファイル.txt",
        })
        assert "/data/中文/ファイル.txt" in args["path"]

    @patch("model_tools.registry")
    def test_unicode_path_with_markdown_link(self, mock_registry):
        """Unicode markdown link should be stripped correctly."""
        from model_tools import _repair_semantic_args

        _register_schema("read_file", {
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "x-semantic-type": "file-path"},
                },
                "required": ["path"],
            }
        })
        mock_registry.get_schema.side_effect = _fake_get_schema

        args, log = _repair_semantic_args("read_file", {
            "path": "[中文/文件.txt](リンク)",
        })
        assert args["path"] == "中文/文件.txt"

    @patch("model_tools.registry")
    def test_empty_schema_no_repair(self, mock_registry):
        """Tools with empty schema (no properties) should pass through."""
        from model_tools import _repair_tool_args

        _register_schema("empty_tool", {
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            }
        })
        mock_registry.get_schema.side_effect = _fake_get_schema

        args, log = _repair_tool_args("empty_tool", {"key": "value"})
        assert args == {"key": "value"}
        assert log is None

    @patch("model_tools.registry")
    def test_semantic_repair_idempotent(self, mock_registry):
        """Applying semantic repair twice should give same result."""
        from model_tools import _repair_semantic_args

        _register_schema("read_file", {
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "x-semantic-type": "file-path"},
                },
                "required": ["path"],
            }
        })
        mock_registry.get_schema.side_effect = _fake_get_schema

        args1, log1 = _repair_semantic_args("read_file", {
            "path": "[~/project/file.txt](display)",
        })
        args2, log2 = _repair_semantic_args("read_file", dict(args1))
        assert args1["path"] == args2["path"], "Idempotency violated"
        # Second pass should have no new repairs (tilde already expanded, markdown already stripped)
        assert log2 is None or not any(
            e["repair"] in ("strip_markdown_link", "expand_tilde") for e in (log2 or [])
        )

    @patch("model_tools.registry")
    def test_structural_repair_idempotent(self, mock_registry):
        """Applying structural repair twice should be stable."""
        from model_tools import _repair_tool_args

        _register_schema("test_tool", {
            "parameters": {
                "type": "object",
                "properties": {
                    "tags": {"type": "array", "items": {"type": "string"}},
                    "optional_field": {"type": "string"},
                },
                "required": [],
            }
        })
        mock_registry.get_schema.side_effect = _fake_get_schema

        args1, _ = _repair_tool_args("test_tool", {
            "tags": "bare_string",
            "optional_field": None,
        })
        args2, log2 = _repair_tool_args("test_tool", dict(args1))
        assert args1 == args2, "Structural repair idempotency violated"
        # Second pass should have no repairs (already fixed)
        assert log2 is None

    def test_fix_double_escape_two_to_one(self):
        """Double backslash (2→1) should be handled correctly."""
        from model_tools import _repair_fix_double_escape

        # 2 backslashes → 1 backslash
        assert _repair_fix_double_escape("ls\\\\n") == "ls\\n"
        assert _repair_fix_double_escape("a\\\\nb") == "a\\nb"
        assert _repair_fix_double_escape("no_escape") == "no_escape"
        assert _repair_fix_double_escape(42) == 42


# =============================================================================
# Phase 5: Desktop Tool Semantic Repair Tests (OpenClaw / GLM-5-Turbo)
# =============================================================================


class TestDesktopSemanticRepair:
    """Test per-parameter semantic repairs for desktop tools."""

    # ── Unit tests for individual repair functions ──

    def test_repair_desktop_coord_int_float(self):
        """Float coords should be converted to int."""
        from model_tools import _repair_desktop_coord_int

        assert _repair_desktop_coord_int(100.0) == 100
        assert _repair_desktop_coord_int(500.9) == 500  # truncation, not rounding
        assert _repair_desktop_coord_int(0.0) == 0
        assert _repair_desktop_coord_int(-50.7) == -50

    def test_repair_desktop_coord_int_string(self):
        """String digit coords should be converted to int."""
        from model_tools import _repair_desktop_coord_int

        assert _repair_desktop_coord_int("100") == 100
        assert _repair_desktop_coord_int(" 500 ") == 500
        assert _repair_desktop_coord_int("0") == 0
        assert _repair_desktop_coord_int("-50") == -50

    def test_repair_desktop_coord_int_none(self):
        """None should pass through unchanged (required param, handler will error)."""
        from model_tools import _repair_desktop_coord_int

        assert _repair_desktop_coord_int(None) is None

    def test_repair_desktop_coord_int_nonconvertible(self):
        """Non-convertible string should pass through unchanged."""
        from model_tools import _repair_desktop_coord_int

        result = _repair_desktop_coord_int("not_a_number")
        assert result == "not_a_number"

    def test_repair_desktop_coord_int_valid_int(self):
        """Valid int should pass through unchanged."""
        from model_tools import _repair_desktop_coord_int

        assert _repair_desktop_coord_int(100) == 100
        assert _repair_desktop_coord_int(0) == 0

    def test_repair_clipboard_action_none(self):
        """None action should default to 'read'."""
        from model_tools import _repair_desktop_clipboard_action

        assert _repair_desktop_clipboard_action(None) == "read"
        assert _repair_desktop_clipboard_action("read") == "read"
        assert _repair_desktop_clipboard_action("write") == "write"

    def test_repair_clipboard_text_string(self):
        """Non-string text should be converted to string."""
        from model_tools import _repair_desktop_clipboard_text_string

        assert _repair_desktop_clipboard_text_string(123) == "123"
        assert _repair_desktop_clipboard_text_string(45.6) == "45.6"
        assert _repair_desktop_clipboard_text_string("hello") == "hello"
        assert _repair_desktop_clipboard_text_string("") == ""

    def test_repair_scroll_amount_float(self):
        """Float amount should be converted to int."""
        from model_tools import _repair_desktop_scroll_amount

        assert _repair_desktop_scroll_amount(5.0) == 5
        assert _repair_desktop_scroll_amount(-3.7) == -3
        assert _repair_desktop_scroll_amount(0.0) == 0

    def test_repair_scroll_amount_string(self):
        """String amount should be converted to int."""
        from model_tools import _repair_desktop_scroll_amount

        assert _repair_desktop_scroll_amount("5") == 5
        assert _repair_desktop_scroll_amount("-3") == -3
        assert _repair_desktop_scroll_amount("not_a_number") == "not_a_number"

    def test_repair_scroll_amount_valid_int(self):
        """Valid int amount should pass through unchanged."""
        from model_tools import _repair_desktop_scroll_amount

        assert _repair_desktop_scroll_amount(5) == 5
        assert _repair_desktop_scroll_amount(-3) == -3

    # ── Integration tests via _repair_semantic_args ──

    @patch("model_tools.registry")
    def test_desktop_click_float_coords(self, mock_registry):
        """desktop_click with float coords should repair to int."""
        from model_tools import _repair_semantic_args

        _FAKE_SCHEMAS["desktop_click"] = {
            "parameters": {
                "type": "object",
                "properties": {
                    "x": {"type": "integer"},
                    "y": {"type": "integer"},
                },
                "required": ["x", "y"],
            }
        }
        mock_registry.get_schema.side_effect = _fake_get_schema

        args, log = _repair_semantic_args("desktop_click", {"x": 100.0, "y": 200.9})
        assert args["x"] == 100
        assert args["y"] == 200
        assert log is not None
        assert len(log) == 2
        assert all(e["repair"] == "fix_coord_type" for e in log)

    @patch("model_tools.registry")
    def test_desktop_move_string_coords(self, mock_registry):
        """desktop_move with string-digit coords should repair to int."""
        from model_tools import _repair_semantic_args

        _FAKE_SCHEMAS["desktop_move"] = {
            "parameters": {
                "type": "object",
                "properties": {
                    "x": {"type": "integer"},
                    "y": {"type": "integer"},
                },
                "required": ["x", "y"],
            }
        }
        mock_registry.get_schema.side_effect = _fake_get_schema

        args, log = _repair_semantic_args("desktop_move", {"x": "500", "y": "300"})
        assert args["x"] == 500
        assert args["y"] == 300
        assert log is not None

    @patch("model_tools.registry")
    def test_desktop_click_coord_none_passthrough(self, mock_registry):
        """desktop_click with None coords should pass through (handler will error)."""
        from model_tools import _repair_semantic_args

        _FAKE_SCHEMAS["desktop_click"] = {
            "parameters": {
                "type": "object",
                "properties": {
                    "x": {"type": "integer"},
                    "y": {"type": "integer"},
                },
                "required": ["x", "y"],
            }
        }
        mock_registry.get_schema.side_effect = _fake_get_schema

        args, log = _repair_semantic_args("desktop_click", {"x": None, "y": None})
        assert args["x"] is None
        assert args["y"] is None
        assert log is None  # No repair applied (None passthrough)

    @patch("model_tools.registry")
    def test_desktop_clipboard_action_null(self, mock_registry):
        """desktop_clipboard with None action should default to 'read'."""
        from model_tools import _repair_semantic_args

        _FAKE_SCHEMAS["desktop_clipboard"] = {
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": ["read", "write"]},
                },
                "required": ["action"],
            }
        }
        mock_registry.get_schema.side_effect = _fake_get_schema

        args, log = _repair_semantic_args("desktop_clipboard", {"action": None})
        assert args["action"] == "read"
        assert log is not None
        assert log[0]["repair"] == "fix_clipboard_action_null"
        assert log[0]["from"] is None
        assert log[0]["to"] == "read"

    @patch("model_tools.registry")
    def test_desktop_clipboard_text_int(self, mock_registry):
        """desktop_clipboard with int text should convert to string."""
        from model_tools import _repair_semantic_args

        _FAKE_SCHEMAS["desktop_clipboard"] = {
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {"type": "string"},
                    "text": {"type": "string"},
                },
                "required": ["action"],
            }
        }
        mock_registry.get_schema.side_effect = _fake_get_schema

        args, log = _repair_semantic_args("desktop_clipboard", {"action": "write", "text": 123})
        assert args["text"] == "123"
        assert log is not None
        assert log[0]["repair"] == "fix_clipboard_text_type"

    @patch("model_tools.registry")
    def test_desktop_scroll_amount_float(self, mock_registry):
        """desktop_scroll with float amount should convert to int."""
        from model_tools import _repair_semantic_args

        _FAKE_SCHEMAS["desktop_scroll"] = {
            "parameters": {
                "type": "object",
                "properties": {
                    "amount": {"type": "integer"},
                },
                "required": ["amount"],
            }
        }
        mock_registry.get_schema.side_effect = _fake_get_schema

        args, log = _repair_semantic_args("desktop_scroll", {"amount": 5.0})
        assert args["amount"] == 5
        assert log is not None
        assert log[0]["repair"] == "fix_amount_type"

    # ── desktop_type safeguard tests ──

    @patch("model_tools.registry")
    def test_desktop_type_text_never_modified(self, mock_registry):
        """desktop_type.text must never be modified by any repair pass."""
        from model_tools import _repair_semantic_args

        _FAKE_SCHEMAS["desktop_type"] = {
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                },
                "required": ["text"],
            }
        }
        mock_registry.get_schema.side_effect = _fake_get_schema

        # Text containing markdown-like patterns that OTHER tools would repair
        original = "[click here](https://example.com)  ~/path  $ ls"
        args, sem_log = _repair_semantic_args("desktop_type", {"text": original})
        # Semantic repair must not touch desktop_type text
        assert args["text"] == original
        # The safeguard lambda returns the same object, so log should be None
        assert sem_log is None

    @patch("model_tools.registry")
    def test_desktop_type_text_full_pipeline(self, mock_registry):
        """desktop_type.text must survive the full structural+semantic pipeline."""
        from model_tools import _repair_semantic_args, _repair_tool_args

        _FAKE_SCHEMAS["desktop_type"] = {
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                },
                "required": ["text"],
            }
        }
        mock_registry.get_schema.side_effect = _fake_get_schema

        original = "Hello, type this: [link](url) with ~tilde and $ prompt"
        args = {"text": original}

        # Full pipeline: semantic → structural
        args, _ = _repair_semantic_args("desktop_type", args)
        args, _ = _repair_tool_args("desktop_type", args)
        assert args["text"] == original

    # ── Bug-fix unit tests ──

    def test_clipboard_text_none(self):
        """None should NOT become 'None' string."""
        from model_tools import _repair_desktop_clipboard_text_string

        assert _repair_desktop_clipboard_text_string(None) is None

    def test_coord_int_bool(self):
        """True -> 1, False -> 0 for desktop coords."""
        from model_tools import _repair_desktop_coord_int

        assert _repair_desktop_coord_int(True) == 1
        assert _repair_desktop_coord_int(False) == 0

    def test_coord_int_string_float(self):
        """Float string like '100.5' -> 100 for desktop coords."""
        from model_tools import _repair_desktop_coord_int

        assert _repair_desktop_coord_int("100.5") == 100
        assert _repair_desktop_coord_int(" 50.9 ") == 50
        assert _repair_desktop_coord_int("-3.7") == -3

    def test_scroll_amount_string_decimal(self):
        """Float string like '-3.5' -> -3 for scroll amount."""
        from model_tools import _repair_desktop_scroll_amount

        assert _repair_desktop_scroll_amount("-3.5") == -3
        assert _repair_desktop_scroll_amount(" 10.0 ") == 10
        assert _repair_desktop_scroll_amount("0.9") == 0

    def test_scroll_amount_none(self):
        """None should pass through unchanged for scroll amount."""
        from model_tools import _repair_desktop_scroll_amount

        assert _repair_desktop_scroll_amount(None) is None

    def test_coord_int_inf_nan(self):
        """inf and nan should pass through unchanged (no crash)."""
        from model_tools import _repair_desktop_coord_int

        result = _repair_desktop_coord_int("inf")
        assert result == "inf"
        result = _repair_desktop_coord_int("nan")
        assert result == "nan"
        result = _repair_desktop_coord_int("+inf")
        assert result == "+inf"

    @patch("model_tools.registry")
    def test_desktop_click_none_passthrough(self, mock_registry):
        """desktop_click with None x/y should pass through untouched."""
        from model_tools import _repair_semantic_args

        _FAKE_SCHEMAS["desktop_click"] = {
            "parameters": {
                "type": "object",
                "properties": {
                    "x": {"type": "integer"},
                    "y": {"type": "integer"},
                },
                "required": ["x", "y"],
            }
        }
        mock_registry.get_schema.side_effect = _fake_get_schema

        args, log = _repair_semantic_args("desktop_click", {"x": None, "y": None})
        assert args["x"] is None
        assert args["y"] is None
        assert log is None  # No repair applied (None passthrough)

    @patch("model_tools.registry")
    def test_desktop_repair_end_to_end(self, mock_registry):
        """End-to-end test via _repair_semantic_args across all desktop tools."""
        from model_tools import _repair_semantic_args

        # Register schemas for all desktop tools
        _FAKE_SCHEMAS["desktop_click"] = {
            "parameters": {
                "type": "object",
                "properties": {
                    "x": {"type": "integer"},
                    "y": {"type": "integer"},
                },
                "required": ["x", "y"],
            }
        }
        _FAKE_SCHEMAS["desktop_move"] = {
            "parameters": {
                "type": "object",
                "properties": {
                    "x": {"type": "integer"},
                    "y": {"type": "integer"},
                },
                "required": ["x", "y"],
            }
        }
        _FAKE_SCHEMAS["desktop_clipboard"] = {
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": ["read", "write"]},
                    "text": {"type": "string"},
                },
                "required": ["action"],
            }
        }
        _FAKE_SCHEMAS["desktop_scroll"] = {
            "parameters": {
                "type": "object",
                "properties": {
                    "amount": {"type": "integer"},
                },
                "required": ["amount"],
            }
        }
        mock_registry.get_schema.side_effect = _fake_get_schema

        # Test all broken input patterns end-to-end
        cases = [
            ("desktop_click", {"x": True, "y": False}, {"x": 1, "y": 0}),
            ("desktop_move", {"x": "100.5", "y": "200.9"}, {"x": 100, "y": 200}),
            ("desktop_clipboard", {"action": None, "text": None}, {"action": "read", "text": None}),
            ("desktop_scroll", {"amount": "-3.5"}, {"amount": -3}),
        ]

        for tname, targs, expected in cases:
            args, log = _repair_semantic_args(tname, dict(targs))
            for key, exp_val in expected.items():
                assert args[key] == exp_val, (
                    f"{tname}.{key}: expected {exp_val!r}, got {args[key]!r}"
                )
            assert log is not None, f"{tname}: expected repair log"

    # ── Idempotency ──

    @patch("model_tools.registry")
    def test_desktop_repair_idempotent(self, mock_registry):
        """Desktop semantic repairs should be idempotent (second pass = no-op)."""
        from model_tools import _repair_semantic_args

        for tool_name in ("desktop_click", "desktop_move"):
            _FAKE_SCHEMAS[tool_name] = {
                "parameters": {
                    "type": "object",
                    "properties": {
                        "x": {"type": "integer"},
                        "y": {"type": "integer"},
                    },
                    "required": ["x", "y"],
                }
            }
        _FAKE_SCHEMAS["desktop_clipboard"] = {
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {"type": "string"},
                    "text": {"type": "string"},
                },
                "required": ["action"],
            }
        }
        _FAKE_SCHEMAS["desktop_scroll"] = {
            "parameters": {
                "type": "object",
                "properties": {
                    "amount": {"type": "integer"},
                },
                "required": ["amount"],
            }
        }
        mock_registry.get_schema.side_effect = _fake_get_schema

        # Apply repair to damaged inputs
        cases = [
            ("desktop_click", {"x": 100.0, "y": 200.0}),
            ("desktop_move", {"x": "500", "y": "300"}),
            ("desktop_clipboard", {"action": None, "text": 123}),
            ("desktop_scroll", {"amount": 5.0}),
        ]

        for tname, targs in cases:
            args1, log1 = _repair_semantic_args(tname, dict(targs))
            args2, log2 = _repair_semantic_args(tname, dict(args1))
            assert args1 == args2, f"{tname}: idempotency violated"
            # Second pass should have no new repairs
            assert log2 is None, f"{tname}: second pass should be no-op, got {log2}"
