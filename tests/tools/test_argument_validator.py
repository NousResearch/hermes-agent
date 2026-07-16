"""Tests for tools.argument_validator."""

from unittest.mock import MagicMock

from tools.argument_validator import validate_tool_arguments
from model_tools import handle_function_call


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


class TestValidationViaHandleFunctionCall:
    """The shared post-coercion gate lives in model_tools.handle_function_call,
    so the sequential (tool_executor) path must validate through it."""

    def test_sequential_path_blocks_bad_path_via_handle_function_call(self):
        result = handle_function_call(
            "read_file", {"path": "/does/not/exist/anywhere"}, task_id="default"
        )
        assert '"error"' in result
        assert "File not found" in result

    def test_handle_function_call_passes_valid_path(self, tmp_path):
        real_file = tmp_path / "test.txt"
        real_file.write_text("hello")
        result = handle_function_call(
            "read_file", {"path": str(real_file)}, task_id="default"
        )
        assert '"error"' not in result


class TestValidationViaInvokeTool:
    """The concurrent (invoke_tool) path routes registry tools through
    handle_function_call, so it must hit the same shared gate."""

    def test_concurrent_path_blocks_bad_path_via_invoke_tool(self):
        from unittest.mock import MagicMock
        from agent.agent_runtime_helpers import invoke_tool

        agent = MagicMock()
        agent.session_id = "sess-concurrent"
        agent.valid_tool_names = None
        agent.enabled_toolsets = None
        agent.disabled_toolsets = None
        agent._memory_manager = None
        agent._current_turn_id = ""
        agent._current_api_request_id = ""

        result = invoke_tool(
            agent,
            "read_file",
            {"path": "/does/not/exist/concurrent"},
            effective_task_id="task-concurrent",
        )
        assert '"error"' in result
        assert "File not found" in result


class TestCoercionBeforeValidation:
    """String-encoded ints/bools must be coerced before the type/placeholder
    checks run, so "10" and "true" are accepted rather than rejected."""

    def test_string_int_coerced_before_type_check(self, tmp_path):
        from model_tools import coerce_tool_args

        real_file = tmp_path / "test.txt"
        real_file.write_text("hello", encoding="utf-8")

        coerced = coerce_tool_args("read_file", {"path": str(real_file), "limit": "10"})
        assert coerced["limit"] == 10
        assert isinstance(coerced["limit"], int)

        registry = MagicMock()
        entry = MagicMock()
        entry.schema = {
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}, "limit": {"type": "integer"}},
            }
        }
        registry.get_entry.return_value = entry

        ok, err = validate_tool_arguments(
            "read_file", coerced, registry
        )
        assert ok is True, err
        assert err == ""

    def test_string_bool_coerced_before_type_check(self, monkeypatch):
        import tools.registry as _reg_mod
        from model_tools import coerce_tool_args

        schema = {
            "parameters": {
                "type": "object",
                "properties": {"verbose": {"type": "boolean"}, "path": {"type": "string"}},
            }
        }
        monkeypatch.setattr(
            _reg_mod.registry, "get_schema",
            lambda name: schema if name == "bool_tool" else _reg_mod.registry.get_schema.__wrapped__(name),
        )

        coerced = coerce_tool_args("bool_tool", {"verbose": "true", "path": "/x"})
        assert coerced["verbose"] is True
        assert isinstance(coerced["verbose"], bool)

        registry = MagicMock()
        entry = MagicMock()
        entry.schema = schema
        registry.get_entry.return_value = entry

        ok, err = validate_tool_arguments("bool_tool", coerced, registry)
        assert ok is True, err


class TestRemoteBackendSkipsPathExistence:
    """On remote/container backends the path lives inside the sandbox, not on
    the host FS, so os.path.exists() must be skipped when task_id resolves to
    a non-local backend."""

    def test_remote_task_skips_file_not_found(self, monkeypatch):
        import tools.file_tools as file_tools

        monkeypatch.setattr(
            file_tools, "_terminal_env_type_for_task", lambda tid="default": "docker"
        )

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
            "read_file",
            {"path": "/container/only/path"},
            registry,
            task_id="task-remote",
        )
        assert ok is True, err

    def test_local_task_still_checks_existence(self, monkeypatch):
        import tools.file_tools as file_tools

        monkeypatch.setattr(
            file_tools, "_terminal_env_type_for_task", lambda tid="default": "local"
        )

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
            "read_file",
            {"path": "/does/not/exist/local"},
            registry,
            task_id="task-local",
        )
        assert ok is False
        assert "File not found" in err
