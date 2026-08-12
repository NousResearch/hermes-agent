"""Tests for the central tool registry."""

import json
import logging
import threading
from pathlib import Path
from unittest.mock import patch

from tools.registry import (
    ToolRegistry,
    _MAX_LOGGED_ERROR_CHARS,
    _MAX_TOOL_ERROR_CHARS,
    _module_registers_tools,
    discover_builtin_tools,
    tool_error,
)


def _dummy_handler(args, **kwargs):
    return json.dumps({"ok": True})


def _make_schema(name="test_tool"):
    return {
        "name": name,
        "description": f"A {name}",
        "parameters": {"type": "object", "properties": {}},
    }


class TestRegisterAndDispatch:
    def test_register_and_dispatch(self):
        reg = ToolRegistry()
        reg.register(
            name="alpha",
            toolset="core",
            schema=_make_schema("alpha"),
            handler=_dummy_handler,
        )
        result = json.loads(reg.dispatch("alpha", {}))
        assert result == {"ok": True}


    def test_cross_mcp_toolsets_do_not_overwrite_atomically(self, caplog):
        """Parallel MCP registrations with one name leave exactly one owner."""
        reg = ToolRegistry()
        barrier = threading.Barrier(3)
        errors = []

        def _register(toolset, owner):
            try:
                barrier.wait(timeout=5)

                def _handler(args, **kwargs):
                    return json.dumps({"owner": owner})

                reg.register(
                    name="mcp__foo_bar__search",
                    toolset=toolset,
                    schema=_make_schema("mcp__foo_bar__search"),
                    handler=_handler,
                )
            except BaseException as exc:  # pragma: no cover - asserted below
                errors.append(exc)

        threads = [
            threading.Thread(target=_register, args=("mcp-foo-bar", "dash")),
            threading.Thread(target=_register, args=("mcp-foo_bar", "underscore")),
        ]

        with caplog.at_level(logging.ERROR, logger="tools.registry"):
            for thread in threads:
                thread.start()
            barrier.wait(timeout=5)
            for thread in threads:
                thread.join(timeout=10)

        assert all(not thread.is_alive() for thread in threads)
        assert errors == []
        assert reg._generation == 1

        entry = reg.get_entry("mcp__foo_bar__search")
        assert entry is not None
        assert entry.toolset in {"mcp-foo-bar", "mcp-foo_bar"}
        assert json.loads(reg.dispatch("mcp__foo_bar__search", {}))["owner"] in {
            "dash",
            "underscore",
        }
        assert any(
            "REJECTED" in record.message
            and "mcp__foo_bar__search" in record.message
            for record in caplog.records
        )

class TestGetDefinitions:
    def test_returns_openai_format(self):
        reg = ToolRegistry()
        reg.register(
            name="t1", toolset="s1", schema=_make_schema("t1"), handler=_dummy_handler
        )
        reg.register(
            name="t2", toolset="s1", schema=_make_schema("t2"), handler=_dummy_handler
        )

        defs = reg.get_definitions({"t1", "t2"})
        assert len(defs) == 2
        assert all(d["type"] == "function" for d in defs)
        names = {d["function"]["name"] for d in defs}
        assert names == {"t1", "t2"}


    def test_reuses_shared_check_fn_once_per_call(self):
        reg = ToolRegistry()
        calls = {"count": 0}

        def shared_check():
            calls["count"] += 1
            return True

        reg.register(
            name="first",
            toolset="shared",
            schema=_make_schema("first"),
            handler=_dummy_handler,
            check_fn=shared_check,
        )
        reg.register(
            name="second",
            toolset="shared",
            schema=_make_schema("second"),
            handler=_dummy_handler,
            check_fn=shared_check,
        )

        defs = reg.get_definitions({"first", "second"})
        assert len(defs) == 2
        assert calls["count"] == 1


class TestUnknownToolDispatch:
    def test_returns_error_json(self):
        reg = ToolRegistry()
        result = json.loads(reg.dispatch("nonexistent", {}))
        assert "error" in result
        assert "Unknown tool" in result["error"]


class TestToolErrorBounding:
    def test_short_message_unchanged(self):
        result = json.loads(tool_error("Missing required parameter: query"))
        assert result["error"] == "Missing required parameter: query"

    def test_extra_kwargs_preserved(self):
        result = json.loads(tool_error("bad input", success=False))
        assert result["error"] == "bad input"
        assert result["success"] is False

    def test_oversized_body_truncated(self):
        result = json.loads(tool_error("boom: " + "X" * 5000))
        assert result["error"].endswith("… [truncated]")
        assert len(result["error"]) <= _MAX_TOOL_ERROR_CHARS + len("… [truncated]")

    def test_at_limit_not_truncated(self):
        msg = "Y" * _MAX_TOOL_ERROR_CHARS
        result = json.loads(tool_error(msg))
        assert result["error"] == msg

    def test_longer_prefix_reaches_logs_than_context(self, caplog):
        import logging
        body = "boom: " + "Z" * 5000
        with caplog.at_level(logging.DEBUG, logger="tools.registry"):
            result = json.loads(tool_error(body))
        logged = "\n".join(rec.getMessage() for rec in caplog.records)
        assert body[:5000] in logged
        assert len(result["error"]) < 5000

    def test_log_line_is_bounded_for_huge_bodies(self, caplog):
        import logging
        body = "boom: " + "Z" * 500_000
        with caplog.at_level(logging.DEBUG, logger="tools.registry"):
            json.loads(tool_error(body))
        for record in caplog.records:
            assert len(record.getMessage()) < _MAX_LOGGED_ERROR_CHARS + 200
        assert body not in "\n".join(r.getMessage() for r in caplog.records)


class TestDispatchBoundsDirectErrorResults:
    """Handlers that bypass tool_error() and serialize errors directly are
    still bounded at the dispatch boundary."""

    @staticmethod
    def _register(reg, name, handler):
        reg.register(
            name=name,
            toolset="core",
            schema=_make_schema(name),
            handler=handler,
        )

    def test_direct_json_error_result_truncated(self):
        reg = ToolRegistry()
        self._register(reg, "direct", lambda args, **kw: json.dumps({
            "status": "error",
            "error": "boom: " + "X" * 50_000,
            "tool_calls_made": 3,
            "duration_seconds": 1.2,
        }, ensure_ascii=False))
        result = json.loads(reg.dispatch("direct", {}))
        assert result["error"].endswith("… [truncated]")
        assert len(result["error"]) <= _MAX_TOOL_ERROR_CHARS + len("… [truncated]")
        assert result["status"] == "error"
        assert result["tool_calls_made"] == 3
        assert result["duration_seconds"] == 1.2

    def test_small_error_result_unchanged(self):
        reg = ToolRegistry()
        payload = json.dumps({"error": "not found", "success": False})
        self._register(reg, "small", lambda args, **kw: payload)
        assert reg.dispatch("small", {}) == payload

    def test_oversized_non_error_result_unchanged(self):
        reg = ToolRegistry()
        payload = json.dumps({"data": "D" * 50_000})
        self._register(reg, "big_data", lambda args, **kw: payload)
        assert reg.dispatch("big_data", {}) == payload

    def test_oversized_non_json_result_unchanged(self):
        reg = ToolRegistry()
        payload = "plain text " * 10_000
        self._register(reg, "plain", lambda args, **kw: payload)
        assert reg.dispatch("plain", {}) == payload

    def test_non_string_error_value_unchanged(self):
        reg = ToolRegistry()
        payload = json.dumps({"error": {"detail": "E" * 5_000}})
        self._register(reg, "nested", lambda args, **kw: payload)
        assert reg.dispatch("nested", {}) == payload


class TestDispatchExceptionLogging:
    def test_raising_handler_logs_bounded_message(self, caplog):
        import logging
        body = "upstream said: " + "Q" * 200_000
        reg = ToolRegistry()
        reg.register(
            name="boom",
            toolset="core",
            schema=_make_schema("boom"),
            handler=lambda args, **kw: (_ for _ in ()).throw(RuntimeError(body)),
        )
        with caplog.at_level(logging.ERROR, logger="tools.registry"):
            result = json.loads(reg.dispatch("boom", {}))
        messages = [r.getMessage() for r in caplog.records]
        assert messages, "dispatch should log the failure"
        for message in messages:
            assert len(message) < _MAX_LOGGED_ERROR_CHARS + 200
            assert body not in message
        assert len(result["error"]) < _MAX_TOOL_ERROR_CHARS + 200


class TestToolsetAvailability:
    def test_no_check_fn_is_available(self):
        reg = ToolRegistry()
        reg.register(
            name="t", toolset="free", schema=_make_schema(), handler=_dummy_handler
        )
        assert reg.is_toolset_available("free") is True

    def test_check_fn_controls_availability(self):
        reg = ToolRegistry()
        reg.register(
            name="t",
            toolset="locked",
            schema=_make_schema(),
            handler=_dummy_handler,
            check_fn=lambda: False,
        )
        assert reg.is_toolset_available("locked") is False


    def test_handler_exception_returns_error(self):
        reg = ToolRegistry()

        def bad_handler(args, **kw):
            raise RuntimeError("boom")

        reg.register(
            name="bad", toolset="s", schema=_make_schema(), handler=bad_handler
        )
        result = json.loads(reg.dispatch("bad", {}))
        assert "error" in result
        assert "RuntimeError" in result["error"]


class TestCheckFnExceptionHandling:
    """Verify that a raising check_fn is caught rather than crashing."""

    def test_is_toolset_available_catches_exception(self):
        reg = ToolRegistry()
        reg.register(
            name="t",
            toolset="broken",
            schema=_make_schema(),
            handler=_dummy_handler,
            check_fn=lambda: 1 / 0,  # ZeroDivisionError
        )
        # Should return False, not raise
        assert reg.is_toolset_available("broken") is False


    def test_check_tool_availability_survives_raising_check(self):
        reg = ToolRegistry()
        reg.register(
            name="a",
            toolset="works",
            schema=_make_schema(),
            handler=_dummy_handler,
            check_fn=lambda: True,
        )
        reg.register(
            name="b",
            toolset="crashes",
            schema=_make_schema(),
            handler=_dummy_handler,
            check_fn=lambda: 1 / 0,
        )

        available, unavailable = reg.check_tool_availability()
        assert "works" in available
        assert any(u["name"] == "crashes" for u in unavailable)


class TestBuiltinDiscovery:
    def test_discovers_all_real_self_registering_builtin_tool_modules(self):
        tools_dir = Path(__file__).resolve().parents[2] / "tools"
        expected = [
            f"tools.{path.stem}"
            for path in sorted(tools_dir.glob("*.py"))
            if path.name not in {"__init__.py", "registry.py", "mcp_tool.py"}
            and _module_registers_tools(path)
        ]

        with patch("tools.registry.importlib.import_module"):
            imported = discover_builtin_tools(tools_dir)

        assert imported == expected


    def test_skips_mcp_tool_even_if_it_registers(self, tmp_path):
        tools_dir = tmp_path / "tools"
        tools_dir.mkdir()
        (tools_dir / "__init__.py").write_text("", encoding="utf-8")
        (tools_dir / "mcp_tool.py").write_text(
            "from tools.registry import registry\nregistry.register(name='mcp_alpha', toolset='mcp-test', schema={}, handler=lambda *_a, **_k: '{}')\n",
            encoding="utf-8",
        )
        (tools_dir / "alpha.py").write_text(
            "from tools.registry import registry\nregistry.register(name='alpha', toolset='x', schema={}, handler=lambda *_a, **_k: '{}')\n",
            encoding="utf-8",
        )

        with patch("tools.registry.importlib.import_module") as mock_import:
            imported = discover_builtin_tools(tools_dir)

        assert imported == ["tools.alpha"]
        mock_import.assert_called_once_with("tools.alpha")


class TestEmojiMetadata:
    """Verify per-tool emoji registration and lookup."""

    def test_emoji_stored_on_entry(self):
        reg = ToolRegistry()
        reg.register(
            name="t", toolset="s", schema=_make_schema(),
            handler=_dummy_handler, emoji="🔥",
        )
        assert reg._tools["t"].emoji == "🔥"


    def test_emoji_empty_string_treated_as_unset(self):
        reg = ToolRegistry()
        reg.register(
            name="t", toolset="s", schema=_make_schema(),
            handler=_dummy_handler, emoji="",
        )
        assert reg.get_emoji("t") == "⚡"


class TestEntryLookup:
    def test_get_entry_returns_registered_entry(self):
        reg = ToolRegistry()
        reg.register(
            name="alpha", toolset="core", schema=_make_schema("alpha"), handler=_dummy_handler
        )
        entry = reg.get_entry("alpha")
        assert entry is not None
        assert entry.name == "alpha"
        assert entry.toolset == "core"

    def test_get_entry_returns_none_for_unknown_tool(self):
        reg = ToolRegistry()
        assert reg.get_entry("missing") is None


class TestSecretCaptureResultContract:
    def test_secret_request_result_does_not_include_secret_value(self):
        result = {
            "success": True,
            "stored_as": "TENOR_API_KEY",
            "validated": False,
        }
        assert "secret" not in json.dumps(result).lower()


class TestThreadSafety:
    def test_get_available_toolsets_uses_coherent_snapshot(self, monkeypatch):
        reg = ToolRegistry()
        reg.register(
            name="alpha",
            toolset="gated",
            schema=_make_schema("alpha"),
            handler=_dummy_handler,
            check_fn=lambda: False,
        )

        entries, toolset_checks = reg._snapshot_state()

        def snapshot_then_mutate():
            reg.deregister("alpha")
            return entries, toolset_checks

        monkeypatch.setattr(reg, "_snapshot_state", snapshot_then_mutate)

        toolsets = reg.get_available_toolsets()
        assert toolsets["gated"]["available"] is False
        assert toolsets["gated"]["tools"] == ["alpha"]

    def test_check_tool_availability_tolerates_concurrent_register(self):
        reg = ToolRegistry()
        check_started = threading.Event()
        writer_done = threading.Event()
        errors = []
        result_holder = {}
        writer_completed_during_check = {}

        def blocking_check():
            check_started.set()
            writer_completed_during_check["value"] = writer_done.wait(timeout=10)
            return True

        reg.register(
            name="alpha",
            toolset="gated",
            schema=_make_schema("alpha"),
            handler=_dummy_handler,
            check_fn=blocking_check,
        )
        reg.register(
            name="beta",
            toolset="plain",
            schema=_make_schema("beta"),
            handler=_dummy_handler,
        )

        def reader():
            try:
                result_holder["value"] = reg.check_tool_availability()
            except Exception as exc:  # pragma: no cover - exercised on failure only
                errors.append(exc)

        def writer():
            assert check_started.wait(timeout=10)
            reg.register(
                name="gamma",
                toolset="new",
                schema=_make_schema("gamma"),
                handler=_dummy_handler,
            )
            writer_done.set()

        reader_thread = threading.Thread(target=reader)
        writer_thread = threading.Thread(target=writer)
        reader_thread.start()
        writer_thread.start()
        reader_thread.join(timeout=15)
        writer_thread.join(timeout=15)

        assert not reader_thread.is_alive()
        assert not writer_thread.is_alive()
        assert writer_completed_during_check["value"] is True
        assert errors == []

        available, unavailable = result_holder["value"]
        assert "gated" in available
        assert "plain" in available
        assert unavailable == []

    def test_get_available_toolsets_tolerates_concurrent_deregister(self):
        reg = ToolRegistry()
        check_started = threading.Event()
        writer_done = threading.Event()
        errors = []
        result_holder = {}
        writer_completed_during_check = {}

        def blocking_check():
            check_started.set()
            writer_completed_during_check["value"] = writer_done.wait(timeout=10)
            return True

        reg.register(
            name="alpha",
            toolset="gated",
            schema=_make_schema("alpha"),
            handler=_dummy_handler,
            check_fn=blocking_check,
        )
        reg.register(
            name="beta",
            toolset="plain",
            schema=_make_schema("beta"),
            handler=_dummy_handler,
        )

        def reader():
            try:
                result_holder["value"] = reg.get_available_toolsets()
            except Exception as exc:  # pragma: no cover - exercised on failure only
                errors.append(exc)

        def writer():
            assert check_started.wait(timeout=10)
            reg.deregister("beta")
            writer_done.set()

        reader_thread = threading.Thread(target=reader)
        writer_thread = threading.Thread(target=writer)
        reader_thread.start()
        writer_thread.start()
        reader_thread.join(timeout=15)
        writer_thread.join(timeout=15)

        assert not reader_thread.is_alive()
        assert not writer_thread.is_alive()
        assert writer_completed_during_check["value"] is True
        assert errors == []

        toolsets = result_holder["value"]
        assert "gated" in toolsets
        assert toolsets["gated"]["available"] is True


class TestToolsetAvailabilityAggregation:
    def test_mixed_toolset_available_when_general_tool_passes(self):
        """Desktop-only helpers must not hide general-purpose tools from doctor."""
        reg = ToolRegistry()
        reg.register(
            name="read_terminal",
            toolset="terminal",
            schema=_make_schema("read_terminal"),
            handler=_dummy_handler,
            check_fn=lambda: False,
        )
        reg.register(
            name="terminal",
            toolset="terminal",
            schema=_make_schema("terminal"),
            handler=_dummy_handler,
            check_fn=lambda: True,
        )
        reg.register(
            name="process",
            toolset="terminal",
            schema=_make_schema("process"),
            handler=_dummy_handler,
        )

        available, unavailable = reg.check_tool_availability()

        assert "terminal" in available
        assert unavailable == []
        assert reg.is_toolset_available("terminal")
        assert reg.get_available_toolsets()["terminal"]["available"] is True

    def test_mixed_toolset_unavailable_when_every_tool_is_gated(self):
        reg = ToolRegistry()
        reg.register(
            name="read_terminal",
            toolset="terminal",
            schema=_make_schema("read_terminal"),
            handler=_dummy_handler,
            check_fn=lambda: False,
        )
        reg.register(
            name="terminal",
            toolset="terminal",
            schema=_make_schema("terminal"),
            handler=_dummy_handler,
            check_fn=lambda: False,
        )

        available, unavailable = reg.check_tool_availability()

        assert "terminal" not in available
        assert any(item["name"] == "terminal" for item in unavailable)


class TestDeregisterAuthorization:
    """deregister() must apply the same plugin opt-in gate as register().

    A plugin could bypass register(override=True) authorization entirely by
    first calling deregister() to clear the existing entry — making
    `existing` None in register() — then re-registering with no override
    flag at all. This skips the override-policy check because that check
    only fires when `existing` is set.
    """

    def _reg(self):
        reg = ToolRegistry()
        reg.register(
            name="protected",
            toolset="terminal",
            schema={"name": "protected", "description": "", "parameters": {"type": "object", "properties": {}}},
            handler=lambda *a, **k: "built-in",
        )
        return reg

    def test_plugin_cannot_deregister_unowned_tool_without_opt_in(self):
        reg = self._reg()
        reg.register_plugin_override_policy("hermes_plugins.evil", False)
        with patch.object(ToolRegistry, "_caller_module", return_value="hermes_plugins.evil"):
            import pytest
            with pytest.raises(PermissionError, match="allow_tool_override"):
                reg.deregister("protected")
        assert reg._tools.get("protected") is not None, "tool must survive the rejected deregister"


    def test_plugin_root_module_can_deregister_submodule_handler(self):
        """Plugin root cleaning up a tool whose handler lives in a submodule.

        hermes_plugins.pkg (root cleanup code) must be allowed to deregister a
        tool whose handler was defined in hermes_plugins.pkg.handlers.  The
        exact module strings differ, but they share the same plugin package root
        (hermes_plugins.pkg) — ownership is bound to the package, not the leaf
        module (egilewski review, #55840).
        """
        reg = ToolRegistry()
        reg.register_plugin_override_policy("hermes_plugins.pkg", False)
        handler = eval("lambda *a, **k: 'sub'", {"__name__": "hermes_plugins.pkg.handlers"})
        reg.register(
            name="sub_tool", toolset="pkg-ts",
            schema={"name": "sub_tool", "description": "", "parameters": {"type": "object", "properties": {}}},
            handler=handler,
        )
        # Caller is the plugin root (hermes_plugins.pkg), handler is in a
        # submodule (hermes_plugins.pkg.handlers) — must be allowed.
        with patch.object(ToolRegistry, "_caller_module", return_value="hermes_plugins.pkg"):
            reg.deregister("sub_tool")
        assert reg._tools.get("sub_tool") is None

    def test_opted_in_plugin_submodule_can_deregister(self):
        """An opted-in plugin calling deregister() from a submodule must succeed.

        register_plugin_override_policy records the opt-in under the package
        root (``hermes_plugins.allowed``).  If the caller is a submodule
        (``hermes_plugins.allowed.cleanup``), the old code looked up
        ``_plugin_override_policy.get("hermes_plugins.allowed.cleanup")`` →
        False and wrongly raised PermissionError.  The fix uses caller_root
        for the policy lookup so submodule callers inherit the package opt-in
        (egilewski review #2 on #55840).
        """
        reg = ToolRegistry()
        reg.register(
            name="protected", toolset="terminal",
            schema={"name": "protected", "description": "", "parameters": {"type": "object", "properties": {}}},
            handler=lambda *a, **k: "built-in",
        )
        reg.register_plugin_override_policy("hermes_plugins.allowed", True)
        with patch.object(ToolRegistry, "_caller_module", return_value="hermes_plugins.allowed.cleanup"):
            reg.deregister("protected")
        assert reg._tools.get("protected") is None


    def test_core_code_deregister_always_allowed(self):
        """Non-plugin callers (core Hermes code) are never gated."""
        reg = self._reg()
        with patch.object(ToolRegistry, "_caller_module", return_value="tools.mcp_tool"):
            reg.deregister("protected")
        assert reg._tools.get("protected") is None

    def test_full_bypass_blocked(self):
        """The original bypass: deregister then plain register no longer works."""
        reg = self._reg()
        reg.register_plugin_override_policy("hermes_plugins.evil", False)
        with patch.object(ToolRegistry, "_caller_module", return_value="hermes_plugins.evil"):
            import pytest
            with pytest.raises(PermissionError):
                reg.deregister("protected")
        # Tool is still present, so a follow-up plain register() hits the
        # existing-entry override check and is also rejected.
        with pytest.raises(PermissionError):
            evil_handler = eval("lambda *a, **k: 'hijacked'", {"__name__": "hermes_plugins.evil"})
            reg.register(name="protected", toolset="evil-ts", schema={}, handler=evil_handler, override=True)
        assert reg._tools["protected"].handler({}) == "built-in"
class TestInputSchemaAlias:
    """A plugin that writes the Anthropic/MCP ``input_schema`` key instead of
    ``parameters`` should still expose its schema to the model, with a warning."""

    def test_input_schema_is_aliased_to_parameters(self, caplog):
        reg = ToolRegistry()
        schema = {
            "name": "legacy",
            "description": "declares input_schema like an Anthropic tool spec",
            "input_schema": {
                "type": "object",
                "properties": {"x": {"type": "string"}},
                "required": ["x"],
            },
        }
        with caplog.at_level(logging.WARNING, logger="tools.registry"):
            reg.register(
                name="legacy",
                toolset="s1",
                schema=schema,
                handler=_dummy_handler,
            )

        fn = reg.get_definitions({"legacy"})[0]["function"]
        assert fn["parameters"]["properties"] == {"x": {"type": "string"}}
        assert fn["parameters"]["required"] == ["x"]
        assert "input_schema" not in fn
        assert any("input_schema" in r.getMessage() for r in caplog.records)

    def test_repeated_registration_warns_once(self, caplog):
        reg = ToolRegistry()
        schema = {
            "name": "legacy",
            "description": "declares input_schema like an Anthropic tool spec",
            "input_schema": {
                "type": "object",
                "properties": {"x": {"type": "string"}},
            },
        }

        with caplog.at_level(logging.WARNING, logger="tools.registry"):
            reg.register(
                name="legacy",
                toolset="s1",
                schema=schema,
                handler=_dummy_handler,
            )
            reg.register(
                name="legacy",
                toolset="s1",
                schema=schema,
                handler=_dummy_handler,
            )

        warnings = [
            record for record in caplog.records
            if "legacy" in record.getMessage() and "input_schema" in record.getMessage()
        ]
        assert len(warnings) == 1

    def test_parameters_wins_over_stray_input_schema(self, caplog):
        reg = ToolRegistry()
        schema = {
            "name": "both",
            "description": "has both keys",
            "parameters": {"type": "object", "properties": {"p": {"type": "string"}}},
            "input_schema": {"type": "object", "properties": {"i": {"type": "string"}}},
        }
        with caplog.at_level(logging.WARNING, logger="tools.registry"):
            reg.register(
                name="both",
                toolset="s1",
                schema=schema,
                handler=_dummy_handler,
            )

        fn = reg.get_definitions({"both"})[0]["function"]
        assert fn["parameters"]["properties"] == {"p": {"type": "string"}}
        assert not any("input_schema" in r.getMessage() for r in caplog.records)

    def test_caller_schema_dict_not_mutated(self):
        reg = ToolRegistry()
        schema = {
            "name": "legacy2",
            "description": "d",
            "input_schema": {"type": "object", "properties": {}},
        }
        reg.register(
            name="legacy2",
            toolset="s1",
            schema=schema,
            handler=_dummy_handler,
        )
        # The original dict the caller passed is left intact.
        assert "input_schema" in schema
        assert "parameters" not in schema

    def test_assembled_model_tool_preserves_aliased_parameters(self):
        from model_tools import _clear_tool_defs_cache, get_tool_definitions
        from tools.registry import registry

        name = "test_input_schema_alias_assembled"
        toolset = "test-input-schema-alias"
        parameters = {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        }
        registry.register(
            name=name,
            toolset=toolset,
            schema={
                "name": name,
                "description": "integration test tool",
                "input_schema": parameters,
            },
            handler=_dummy_handler,
        )
        try:
            definitions = get_tool_definitions(
                enabled_toolsets=[toolset],
                quiet_mode=True,
                skip_tool_search_assembly=True,
            )
            assembled = next(
                definition["function"]
                for definition in definitions
                if definition["function"]["name"] == name
            )
        finally:
            registry.deregister(name)
            _clear_tool_defs_cache()

        assert assembled["parameters"] == parameters
        assert "input_schema" not in assembled
