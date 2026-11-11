"""The tier reported for an assembly (shown by the CLI) follows what the model was given."""

from tools.tool_search import ToolSearchConfig, assemble_tool_defs


def _td(name: str) -> dict:
    return {"type": "function", "function": {
        "name": name, "description": f"{name} does one thing.", "parameters": {"type": "object", "properties": {}}}}


_NAMES = [f"probe_tool_{i}" for i in range(4)]


def _deferrable() -> list[dict]:
    return [_td(name) for name in _NAMES]


def _config(**raw) -> ToolSearchConfig:
    return ToolSearchConfig.from_raw({"defer": _NAMES, **raw})


def test_passthrough_is_tier_zero():
    result = assemble_tool_defs(_deferrable(), config=_config(enabled="off"))
    assert (result.activated, result.tier, result.listing_form) == (False, 0, "none")


def test_bridge_with_a_per_tool_listing_is_tier_one():
    result = assemble_tool_defs(_deferrable(), context_length=1_000_000,
                                config=_config(enabled="on"))
    assert result.activated and result.listing_form in ("full", "names", "mixed")
    assert result.tier == 1


def test_bare_bridge_is_tier_two():
    result = assemble_tool_defs(_deferrable(), context_length=1_000_000,
                                config=_config(enabled="on", listing="off"))
    assert (result.activated, result.tier, result.listing_form) == (True, 2, "none")
