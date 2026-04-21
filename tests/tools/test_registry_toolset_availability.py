from tools.registry import ToolRegistry


def _dummy_handler(args, **kwargs):
    return "{}"


def _make_schema(name: str):
    return {
        "name": name,
        "description": name,
        "parameters": {"type": "object", "properties": {}},
    }


def test_toolset_availability_uses_any_available_tool_in_group():
    reg = ToolRegistry()
    reg.register(
        name="browser_cdp",
        toolset="browser",
        schema=_make_schema("browser_cdp"),
        handler=_dummy_handler,
        check_fn=lambda: False,
    )
    reg.register(
        name="browser_navigate",
        toolset="browser",
        schema=_make_schema("browser_navigate"),
        handler=_dummy_handler,
        check_fn=lambda: True,
    )

    available, unavailable = reg.check_tool_availability()

    assert "browser" in available
    assert all(item["name"] != "browser" for item in unavailable)


def test_unavailable_toolset_unions_env_requirements_across_tools():
    reg = ToolRegistry()
    reg.register(
        name="cloud_a",
        toolset="cloud",
        schema=_make_schema("cloud_a"),
        handler=_dummy_handler,
        check_fn=lambda: False,
        requires_env=["API_KEY_A"],
    )
    reg.register(
        name="cloud_b",
        toolset="cloud",
        schema=_make_schema("cloud_b"),
        handler=_dummy_handler,
        check_fn=lambda: False,
        requires_env=["API_KEY_B"],
    )

    _, unavailable = reg.check_tool_availability()
    cloud = next(item for item in unavailable if item["name"] == "cloud")

    assert cloud["env_vars"] == ["API_KEY_A", "API_KEY_B"]
