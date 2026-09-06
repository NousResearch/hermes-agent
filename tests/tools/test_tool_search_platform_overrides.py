"""Per-platform tool-classification overrides (defer / exclude / pin)."""
from types import SimpleNamespace

import pytest

from tools.tool_search import (
    PlatformOverride,
    ToolSearchConfig,
    _parse_platforms,
    assemble_tool_defs,
    classify_tools,
    is_deferrable_tool_name,
)


def _td(name):
    return {"type": "function",
            "function": {"name": name, "description": "x",
                         "parameters": {"type": "object", "properties": {}}}}



def _fake_registry(monkey_target=None):
    """A registry stub: every known name resolves to an ordinary plugin toolset.

    Needed because is_deferrable_tool_name() treats a name the registry does
    not know as NOT deferrable, so a test running without toolsets loaded
    would report a false negative for `defer`.
    """
    from types import SimpleNamespace
    def get_entry(n):
        if n.startswith("mcp__"):
            return SimpleNamespace(toolset="mcp-coder")
        return SimpleNamespace(toolset="plugin-x")
    return get_entry

def _cfg(spec):
    return ToolSearchConfig.from_raw({"platforms": {"telegram": spec}})


# ── 1. defer ────────────────────────────────────────────────────────────
def test_defer_makes_a_core_tool_deferrable(monkeypatch):
    from tools import registry as reg_mod
    monkeypatch.setattr(reg_mod.registry, "get_entry", _fake_registry())
    ov = _cfg({"defer": ["session_search"]}).override_for("telegram")
    assert is_deferrable_tool_name("session_search", ov) is True
    # ...and without the override it is still core, as before.
    assert is_deferrable_tool_name("session_search", None) is False


def test_defer_moves_the_tool_between_classify_buckets(monkeypatch):
    from tools import registry as reg_mod
    monkeypatch.setattr(reg_mod.registry, "get_entry", _fake_registry())
    tools = [_td("session_search"), _td("terminal")]
    ov = _cfg({"defer": ["session_search"]}).override_for("telegram")
    visible, deferrable = classify_tools(tools, ov)
    assert [t["function"]["name"] for t in deferrable] == ["session_search"]
    assert [t["function"]["name"] for t in visible] == ["terminal"]


# ── 2. exclude ──────────────────────────────────────────────────────────
def test_exclude_removes_the_tool_from_the_assembled_array():
    cfg = _cfg({"exclude": ["computer_use"]})
    tools = [_td("computer_use"), _td("terminal"), _td("session_search")]
    result = assemble_tool_defs(tools, context_length=65536,
                                config=cfg, platform="telegram")
    names = {t["function"]["name"] for t in result.tool_defs}
    assert "computer_use" not in names
    assert "terminal" in names


def test_exclude_does_not_apply_to_a_different_platform():
    cfg = _cfg({"exclude": ["computer_use"]})
    tools = [_td("computer_use"), _td("terminal")]
    result = assemble_tool_defs(tools, context_length=65536,
                                config=cfg, platform="cli")
    assert "computer_use" in {t["function"]["name"] for t in result.tool_defs}


# ── 3. pin ──────────────────────────────────────────────────────────────
def test_pin_keeps_an_mcp_tool_visible(monkeypatch):
    """MCP tools defer by prefix; pin must beat that rule."""
    from tools import registry as reg_mod
    monkeypatch.setattr(
        reg_mod.registry, "get_entry",
        lambda n: SimpleNamespace(toolset="mcp-coder") if n.startswith("mcp__") else None,
    )
    ov = _cfg({"pin": ["mcp__coder__coder_implement"]}).override_for("telegram")
    assert is_deferrable_tool_name("mcp__coder__coder_implement", ov) is False
    # a sibling MCP tool that was NOT pinned still defers
    assert is_deferrable_tool_name("mcp__coder__coder_port", ov) is True


def test_pin_beats_defer_on_overlap():
    ov = _cfg({"defer": ["session_search"], "pin": ["session_search"]}).override_for("telegram")
    assert is_deferrable_tool_name("session_search", ov) is False


# ── 4. regression: absent config changes nothing ────────────────────────
def test_no_platforms_key_leaves_classification_untouched():
    cfg = ToolSearchConfig.from_raw({})
    assert cfg.platforms == ()
    assert cfg.override_for("telegram") == PlatformOverride()
    for name in ("terminal", "session_search", "clarify", "memory"):
        assert is_deferrable_tool_name(name, cfg.override_for("telegram")) is False


def test_bridge_tools_are_never_deferred_or_excluded():
    ov = _cfg({"defer": ["tool_search"], "exclude": ["tool_call"]}).override_for("telegram")
    assert is_deferrable_tool_name("tool_search", ov) is False


# ── malformed config must not raise ─────────────────────────────────────
@pytest.mark.parametrize("raw", [None, [], "nope", {"p": "notadict"}, {"p": {"defer": 5}}])
def test_malformed_platforms_are_ignored_not_raised(raw):
    assert isinstance(_parse_platforms(raw), tuple)


def test_scalar_string_is_accepted_as_a_single_name():
    ov = _cfg({"defer": "session_search"}).override_for("telegram")
    assert ov.defer == frozenset({"session_search"})

# ── 5. bridge reachability ──────────────────────────────────────────────
def test_a_deferred_tool_is_still_reachable_through_the_bridge(monkeypatch):
    """Regression: defer must not make a tool invisible AND uncallable.

    The bridge dispatch sites (tool_describe / tool_call / scoped_deferrable_
    names) re-check deferrability without knowing which platform assembled the
    array. Before _bridge_override() they used the unmodified core set, so a
    tool moved out of core by config was rejected with "not a deferrable tool"
    — the model could see it in the catalog and never call it.
    """
    from tools import registry as reg_mod
    from tools.tool_search import _bridge_override
    monkeypatch.setattr(reg_mod.registry, "get_entry", _fake_registry())
    monkeypatch.setattr(
        "tools.tool_search.load_config",
        lambda: ToolSearchConfig.from_raw({"platforms": {"telegram": {"defer": ["todo"]}}}),
    )
    assert is_deferrable_tool_name("todo", _bridge_override()) is True
