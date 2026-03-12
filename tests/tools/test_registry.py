"""Tests for the central tool registry."""

import json

from tools.registry import ToolRegistry


def _dummy_handler(args, **kwargs):
    return json.dumps({"ok": True})


def _make_schema(name="test_tool"):
    return {"name": name, "description": f"A {name}", "parameters": {"type": "object", "properties": {}}}


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
        assert result["success"] is True
        assert result["error"] is None
        assert result["error_type"] is None
        assert result["retryable"] is False
        assert result["data"] == {"ok": True}
        assert result["metrics"]["tool_name"] == "alpha"
        assert isinstance(result["metrics"]["duration_ms"], int)

    def test_dispatch_passes_args(self):
        reg = ToolRegistry()

        def echo_handler(args, **kw):
            return json.dumps(args)

        reg.register(name="echo", toolset="core", schema=_make_schema("echo"), handler=echo_handler)
        result = json.loads(reg.dispatch("echo", {"msg": "hi"}))
        assert result["success"] is True
        assert result["data"] == {"msg": "hi"}


class TestAsyncDispatch:
    def test_async_handler_result_is_normalized(self):
        reg = ToolRegistry()

        async def async_handler(args, **kw):
            return json.dumps({"value": args.get("x", 0)})

        reg.register(
            name="async_echo",
            toolset="core",
            schema=_make_schema("async_echo"),
            handler=async_handler,
            is_async=True,
        )
        result = json.loads(reg.dispatch("async_echo", {"x": 7}))

        assert result["success"] is True
        assert result["data"] == {"value": 7}
        assert result["metrics"]["tool_name"] == "async_echo"
        assert isinstance(result["metrics"]["duration_ms"], int)

    def test_async_handler_exception_returns_error_envelope(self):
        reg = ToolRegistry()

        async def failing_async_handler(args, **kw):
            raise ValueError("async boom")

        reg.register(
            name="async_fail",
            toolset="core",
            schema=_make_schema("async_fail"),
            handler=failing_async_handler,
            is_async=True,
        )
        result = json.loads(reg.dispatch("async_fail", {}))

        assert result["success"] is False
        assert result["error_type"] == "ValueError"
        assert "async boom" in result["error"]
        assert result["metrics"]["tool_name"] == "async_fail"


class TestGetDefinitions:
    def test_returns_openai_format(self):
        reg = ToolRegistry()
        reg.register(name="t1", toolset="s1", schema=_make_schema("t1"), handler=_dummy_handler)
        reg.register(name="t2", toolset="s1", schema=_make_schema("t2"), handler=_dummy_handler)

        defs = reg.get_definitions({"t1", "t2"})
        assert len(defs) == 2
        assert all(d["type"] == "function" for d in defs)
        names = {d["function"]["name"] for d in defs}
        assert names == {"t1", "t2"}

    def test_skips_unavailable_tools(self):
        reg = ToolRegistry()
        reg.register(
            name="available",
            toolset="s",
            schema=_make_schema("available"),
            handler=_dummy_handler,
            check_fn=lambda: True,
        )
        reg.register(
            name="unavailable",
            toolset="s",
            schema=_make_schema("unavailable"),
            handler=_dummy_handler,
            check_fn=lambda: False,
        )
        defs = reg.get_definitions({"available", "unavailable"})
        assert len(defs) == 1
        assert defs[0]["function"]["name"] == "available"


class TestUnknownToolDispatch:
    def test_returns_error_json(self):
        reg = ToolRegistry()
        result = json.loads(reg.dispatch("nonexistent", {}))
        assert result["success"] is False
        assert "Unknown tool" in result["error"]
        assert result["error_type"] == "UnknownToolError"
        assert result["metrics"]["tool_name"] == "nonexistent"


class TestToolsetAvailability:
    def test_no_check_fn_is_available(self):
        reg = ToolRegistry()
        reg.register(name="t", toolset="free", schema=_make_schema(), handler=_dummy_handler)
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

    def test_check_toolset_requirements(self):
        reg = ToolRegistry()
        reg.register(name="a", toolset="ok", schema=_make_schema(), handler=_dummy_handler, check_fn=lambda: True)
        reg.register(name="b", toolset="nope", schema=_make_schema(), handler=_dummy_handler, check_fn=lambda: False)

        reqs = reg.check_toolset_requirements()
        assert reqs["ok"] is True
        assert reqs["nope"] is False

    def test_get_all_tool_names(self):
        reg = ToolRegistry()
        reg.register(name="z_tool", toolset="s", schema=_make_schema(), handler=_dummy_handler)
        reg.register(name="a_tool", toolset="s", schema=_make_schema(), handler=_dummy_handler)
        assert reg.get_all_tool_names() == ["a_tool", "z_tool"]

    def test_handler_exception_returns_error(self):
        reg = ToolRegistry()

        def bad_handler(args, **kw):
            raise RuntimeError("boom")

        reg.register(name="bad", toolset="s", schema=_make_schema(), handler=bad_handler)
        result = json.loads(reg.dispatch("bad", {}))
        assert result["success"] is False
        assert "RuntimeError" in result["error"]
        assert result["error_type"] == "RuntimeError"
        assert result["metrics"]["tool_name"] == "bad"
        assert isinstance(result["metrics"]["duration_ms"], int)

    def test_malformed_envelope_is_sanitized(self):
        reg = ToolRegistry()

        def malformed_handler(args, **kw):
            return json.dumps({
                "success": "false",
                "error": {"message": "bad"},
                "error_type": 404,
                "retryable": "yes",
                "metrics": "bad metrics",
                "path": "/tmp/result.txt",
            })

        reg.register(name="sanitize", toolset="s", schema=_make_schema(), handler=malformed_handler)
        result = json.loads(reg.dispatch("sanitize", {}))

        assert result == {
            "success": False,
            "error": '{"message": "bad"}',
            "error_type": "404",
            "retryable": True,
            "data": {"path": "/tmp/result.txt"},
            "metrics": {
                "tool_name": "sanitize",
                "duration_ms": result["metrics"]["duration_ms"],
            },
        }
        assert isinstance(result["metrics"]["duration_ms"], int)

    def test_single_error_key_payload_is_wrapped_as_plain_data(self):
        reg = ToolRegistry()

        def error_payload_handler(args, **kw):
            return json.dumps({"error": "plain data"})

        reg.register(name="error_payload", toolset="s", schema=_make_schema(), handler=error_payload_handler)
        result = json.loads(reg.dispatch("error_payload", {}))

        assert result["success"] is True
        assert result["error"] is None
        assert result["error_type"] is None
        assert result["retryable"] is False
        assert result["data"] == {"error": "plain data"}
        assert result["metrics"]["tool_name"] == "error_payload"

    def test_scalar_data_envelope_preserves_extras(self):
        reg = ToolRegistry()

        def scalar_data_handler(args, **kw):
            return json.dumps({
                "success": True,
                "data": "saved",
                "path": "/tmp/result.txt",
                "bytes": 42,
            })

        reg.register(name="scalar_data", toolset="s", schema=_make_schema(), handler=scalar_data_handler)
        result = json.loads(reg.dispatch("scalar_data", {}))

        assert result["success"] is True
        assert result["data"] == {
            "value": "saved",
            "path": "/tmp/result.txt",
            "bytes": 42,
        }

    def test_unknown_bool_strings_use_default(self):
        reg = ToolRegistry()

        assert reg._coerce_bool("maybe", default=False) is False
        assert reg._coerce_bool("maybe", default=True) is True


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

    def test_check_toolset_requirements_survives_raising_check(self):
        reg = ToolRegistry()
        reg.register(name="a", toolset="good", schema=_make_schema(), handler=_dummy_handler, check_fn=lambda: True)
        reg.register(name="b", toolset="bad", schema=_make_schema(), handler=_dummy_handler, check_fn=lambda: (_ for _ in ()).throw(ImportError("no module")))

        reqs = reg.check_toolset_requirements()
        assert reqs["good"] is True
        assert reqs["bad"] is False

    def test_get_definitions_skips_raising_check(self):
        reg = ToolRegistry()
        reg.register(
            name="ok_tool",
            toolset="s",
            schema=_make_schema("ok_tool"),
            handler=_dummy_handler,
            check_fn=lambda: True,
        )
        reg.register(
            name="bad_tool",
            toolset="s2",
            schema=_make_schema("bad_tool"),
            handler=_dummy_handler,
            check_fn=lambda: (_ for _ in ()).throw(OSError("network down")),
        )
        defs = reg.get_definitions({"ok_tool", "bad_tool"})
        assert len(defs) == 1
        assert defs[0]["function"]["name"] == "ok_tool"

    def test_check_tool_availability_survives_raising_check(self):
        reg = ToolRegistry()
        reg.register(name="a", toolset="works", schema=_make_schema(), handler=_dummy_handler, check_fn=lambda: True)
        reg.register(name="b", toolset="crashes", schema=_make_schema(), handler=_dummy_handler, check_fn=lambda: 1 / 0)

        available, unavailable = reg.check_tool_availability()
        assert "works" in available
        assert any(u["name"] == "crashes" for u in unavailable)
