"""Tests for ToolRegistry.get_catalog() and get_single_definition()."""

import json

from tools.registry import ToolRegistry


def _make_schema(name, desc=""):
    return {
        "name": name,
        "description": desc or f"A {name} tool",
        "parameters": {"type": "object", "properties": {}},
    }


def _dummy_handler(args, **kw):
    return json.dumps({"ok": True})


class TestGetCatalog:
    def test_returns_compact_entries(self):
        reg = ToolRegistry()
        reg.register(name="alpha", toolset="core", schema=_make_schema("alpha", "Do alpha"),
                     handler=_dummy_handler, description="Do alpha")
        reg.register(name="beta", toolset="core", schema=_make_schema("beta", "Do beta"),
                     handler=_dummy_handler, description="Do beta")

        catalog = reg.get_catalog()
        assert len(catalog) == 2
        for entry in catalog:
            assert set(entry.keys()) == {"name", "description", "toolset"}

    def test_sorted_by_name(self):
        reg = ToolRegistry()
        reg.register(name="zeta", toolset="z", schema=_make_schema("zeta"),
                     handler=_dummy_handler, description="Zeta tool")
        reg.register(name="alpha", toolset="a", schema=_make_schema("alpha"),
                     handler=_dummy_handler, description="Alpha tool")
        catalog = reg.get_catalog()
        assert catalog[0]["name"] == "alpha"
        assert catalog[1]["name"] == "zeta"

    def test_filters_by_tool_names(self):
        reg = ToolRegistry()
        reg.register(name="a", toolset="s", schema=_make_schema("a"),
                     handler=_dummy_handler, description="A")
        reg.register(name="b", toolset="s", schema=_make_schema("b"),
                     handler=_dummy_handler, description="B")
        reg.register(name="c", toolset="s", schema=_make_schema("c"),
                     handler=_dummy_handler, description="C")

        catalog = reg.get_catalog(tool_names={"a", "c"})
        names = [e["name"] for e in catalog]
        assert names == ["a", "c"]

    def test_respects_check_fn(self):
        reg = ToolRegistry()
        reg.register(name="ok", toolset="s", schema=_make_schema("ok"),
                     handler=_dummy_handler, check_fn=lambda: True, description="OK")
        reg.register(name="no", toolset="s", schema=_make_schema("no"),
                     handler=_dummy_handler, check_fn=lambda: False, description="NO")

        catalog = reg.get_catalog()
        names = [e["name"] for e in catalog]
        assert "ok" in names
        assert "no" not in names

    def test_check_fn_exception_treated_as_unavailable(self):
        reg = ToolRegistry()

        def bad_check():
            raise RuntimeError("boom")

        reg.register(name="broken", toolset="s", schema=_make_schema("broken"),
                     handler=_dummy_handler, check_fn=bad_check, description="Broken")

        catalog = reg.get_catalog()
        assert len(catalog) == 0

    def test_description_from_schema_fallback(self):
        reg = ToolRegistry()
        reg.register(name="t", toolset="s",
                     schema={"name": "t", "description": "From schema", "parameters": {}},
                     handler=_dummy_handler)

        catalog = reg.get_catalog()
        assert catalog[0]["description"] == "From schema"

    def test_empty_registry(self):
        reg = ToolRegistry()
        assert reg.get_catalog() == []


class TestGetSingleDefinition:
    def test_returns_openai_format(self):
        reg = ToolRegistry()
        reg.register(name="tool1", toolset="s", schema=_make_schema("tool1", "Desc"),
                     handler=_dummy_handler)

        defn = reg.get_single_definition("tool1")
        assert defn is not None
        assert defn["type"] == "function"
        assert defn["function"]["name"] == "tool1"

    def test_returns_none_for_missing(self):
        reg = ToolRegistry()
        assert reg.get_single_definition("ghost") is None

    def test_respects_check_fn(self):
        reg = ToolRegistry()
        reg.register(name="hidden", toolset="s", schema=_make_schema("hidden"),
                     handler=_dummy_handler, check_fn=lambda: False)

        assert reg.get_single_definition("hidden") is None

    def test_returns_for_passing_check(self):
        reg = ToolRegistry()
        reg.register(name="visible", toolset="s", schema=_make_schema("visible"),
                     handler=_dummy_handler, check_fn=lambda: True)

        defn = reg.get_single_definition("visible")
        assert defn is not None
        assert defn["function"]["name"] == "visible"
