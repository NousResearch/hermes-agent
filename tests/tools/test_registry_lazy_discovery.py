"""Deferred built-in tool discovery: ``defer_builtin_tools`` indexes tools/*.py from source without
importing them, and the registry imports a module the first time a lookup needs one of its tools
or toolsets. These pin that lookups see exactly what eager discovery would have registered."""

import ast
import json
import logging
import os
import subprocess
import sys
import textwrap
from pathlib import Path
from unittest.mock import patch

import pytest

from hermes_constants import reset_hermes_home_override, set_hermes_home_override
from tools.registry import ToolModuleIndex, ToolRegistry, _index_tree

REPO_ROOT = Path(__file__).resolve().parents[2]


def _index(source: str):
    return _index_tree(ast.parse(textwrap.dedent(source)))


def _register_line(name: str, toolset: str) -> str:
    return (f"registry.register(name={name!r}, toolset={toolset!r}, "
            f"schema={{'name': {name!r}, 'description': 'd', 'parameters': {{'type': 'object'}}}}, "
            f"handler=lambda args, **kw: '{{}}')\n")


class _DeferredModules:
    """Built-in modules as source: importing one runs its body against the test's registry under
    the module's own ``__name__``, the way a real tool module's body runs."""

    def __init__(self, reg: ToolRegistry, sources: dict[str, str], namespace_extra=None):
        self.reg, self.sources, self.extra = reg, sources, namespace_extra or {}
        self.imported: list[str] = []
        reg._defer_modules({module: _index(source) for module, source in sources.items()})

    def __call__(self, module: str):
        self.imported.append(module)
        exec(textwrap.dedent(self.sources[module]), {"__name__": module, "registry": self.reg, **self.extra})


@pytest.fixture
def deferred():
    patchers = []

    def make(sources, **extra):
        reg = ToolRegistry()
        modules = _DeferredModules(reg, sources, extra)
        patcher = patch("tools.registry.importlib.import_module", side_effect=modules)
        patcher.start()
        patchers.append(patcher)
        return reg, modules

    yield make
    for patcher in patchers:
        patcher.stop()


_TWO_MODULES = {
    "tools.lazy_alpha": _register_line("alpha", "x"),
    "tools.lazy_beta": _register_line("beta", "y"),
}


class TestLookupsImportOnlyWhatTheyNeed:
    def test_nothing_is_imported_until_a_lookup_names_a_tool(self, deferred):
        reg, modules = deferred(_TWO_MODULES)
        assert modules.imported == []
        assert reg.get_entry("alpha").toolset == "x"
        assert modules.imported == ["tools.lazy_alpha"]

    def test_toolset_lookup_imports_only_the_modules_registering_into_it(self, deferred):
        reg, modules = deferred(_TWO_MODULES)
        assert reg.get_tool_names_for_toolset("y") == ["beta"]
        assert modules.imported == ["tools.lazy_beta"]

    def test_schema_request_imports_only_the_requested_tools(self, deferred):
        reg, modules = deferred(_TWO_MODULES)
        assert [d["function"]["name"] for d in reg.get_definitions({"beta"})] == ["beta"]
        assert modules.imported == ["tools.lazy_beta"]

    def test_whole_registry_read_imports_every_module_in_discovery_order(self, deferred):
        reg, modules = deferred(_TWO_MODULES)
        assert reg.get_all_tool_names() == ["alpha", "beta"]
        assert modules.imported == ["tools.lazy_alpha", "tools.lazy_beta"]

    def test_a_name_no_module_registers_imports_nothing(self, deferred):
        reg, modules = deferred(_TWO_MODULES)
        assert reg.get_entry("nope") is None
        assert modules.imported == []

    def test_each_module_is_imported_once(self, deferred):
        reg, modules = deferred(_TWO_MODULES)
        reg.get_entry("alpha")
        reg.get_entry("alpha")
        reg.get_all_entries()
        assert modules.imported == ["tools.lazy_alpha", "tools.lazy_beta"]

    def test_mcp_prefix_lookup_skips_modules_that_register_no_mcp_toolset(self, deferred):
        reg, modules = deferred(_TWO_MODULES)
        reg.register(name="mcp__srv__t", toolset="mcp-srv", schema={"name": "mcp__srv__t"},
                     handler=lambda args, **kw: "{}")
        assert [e.name for e in reg.get_entries_for_toolset_prefix("mcp-")] == ["mcp__srv__t"]
        assert modules.imported == []


class TestOpaqueModules:
    """A module whose registrations cannot be read statically is imported whenever the index
    cannot rule it out, so it is never missed."""

    SOURCES = {
        "tools.lazy_alpha": _register_line("alpha", "x"),
        "tools.lazy_mystery": "for _name in _names():\n    " + _register_line("mystery", "x").replace(
            "name='mystery'", "name=_name"),
    }

    def test_indexed_name_does_not_import_the_opaque_module(self, deferred):
        reg, modules = deferred(self.SOURCES, _names=lambda: ["mystery"])
        reg.get_entry("alpha")
        assert modules.imported == ["tools.lazy_alpha"]

    def test_unindexed_name_lookup_imports_it(self, deferred):
        reg, modules = deferred(self.SOURCES, _names=lambda: ["mystery"])
        assert reg.get_entry("mystery") is not None
        assert modules.imported == ["tools.lazy_mystery"]

    def test_toolset_lookup_imports_it(self, deferred):
        reg, modules = deferred(self.SOURCES, _names=lambda: ["mystery"])
        assert reg.get_tool_names_for_toolset("x") == ["alpha", "mystery"]
        assert modules.imported == ["tools.lazy_alpha", "tools.lazy_mystery"]


class TestMutationsMeetDeferredBuiltins:
    def test_a_shadowing_registration_meets_the_deferred_builtin(self, deferred):
        """Eager discovery had the built-in registered first, so a same-name tool from another
        toolset is rejected; deferral must not let it win by registering into an empty slot."""
        reg, modules = deferred(_TWO_MODULES)
        reg.register(name="alpha", toolset="other", schema={"name": "alpha"}, handler=lambda args, **kw: "{}")
        assert modules.imported == ["tools.lazy_alpha"]
        assert reg.get_entry("alpha").toolset == "x"

    def test_a_deregistered_deferred_tool_stays_gone(self, deferred):
        reg, _ = deferred(_TWO_MODULES)
        reg.deregister("alpha")
        assert reg.get_entry("alpha") is None
        assert "alpha" not in reg.get_all_tool_names()


class TestImportContext:
    def test_deferred_import_binds_the_launch_profile(self, deferred):
        """A served profile's turn that first needs a tool must not leak its home into the module's
        import-time values (schema paths, module state shared by every profile)."""
        from hermes_constants import get_hermes_home_override
        seen: list = []
        reg, _ = deferred({"tools.lazy_alpha": "SEEN.append(home())\n" + _register_line("alpha", "x")},
                          SEEN=seen, home=get_hermes_home_override)
        token = set_hermes_home_override("/tmp/served-profile")
        try:
            reg.get_entry("alpha")
            assert get_hermes_home_override() == "/tmp/served-profile"
        finally:
            reset_hermes_home_override(token)
        assert seen == [None]

    def test_failed_import_is_logged_and_not_retried_by_lookups(self, deferred, caplog):
        reg, modules = deferred({"tools.lazy_alpha": "raise ImportError('optional dep missing')\n"
                                                     + _register_line("alpha", "x")})
        with caplog.at_level(logging.WARNING, logger="tools.registry"):
            assert reg.get_entry("alpha") is None
            assert reg.get_entry("alpha") is None
        assert modules.imported == ["tools.lazy_alpha"]
        assert any("tools.lazy_alpha" in rec.getMessage() for rec in caplog.records)


class TestStaticIndex:
    def test_literal_registrations(self):
        idx = _index(_register_line("a", "x") + _register_line("b", "y"))
        assert idx == ToolModuleIndex(names=frozenset({"a", "b"}), toolsets=frozenset({"x", "y"}))

    def test_positional_name_and_toolset(self):
        idx = _index("registry.register('a', 'x', {}, handler)\n")
        assert idx == ToolModuleIndex(names=frozenset({"a"}), toolsets=frozenset({"x"}))

    @pytest.mark.parametrize("source", [
        # Name/toolset straight from a literal row.
        """
        _TOOLS = (("a", 1), ("b", 2))
        for _name, _n in _TOOLS:
            registry.register(name=_name, toolset="x", schema={}, handler=h)
        """,
        # The row holds a module-level schema constant; the name is read from it.
        """
        A = {"name": "a", "description": "d"}
        B = {"name": "b"}
        for _schema, _h in ((A, h1), (B, h2)):
            registry.register(name=_schema["name"], toolset="x", schema=_schema, handler=_h)
        """,
        # Inline schema dicts in a named table.
        """
        _TOOLS = (({"name": "a"}, h1), ({"name": "b"}, h2))
        for _schema, _h in _TOOLS:
            registry.register(name=_schema["name"], toolset="x", schema=_schema, handler=_h)
        """,
    ])
    def test_table_driven_loops(self, source):
        assert _index(source) == ToolModuleIndex(names=frozenset({"a", "b"}), toolsets=frozenset({"x"}))

    def test_toolset_from_the_loop_row(self):
        idx = _index("""
        for _name, _actions in (("discord", A), ("discord_admin", B)):
            registry.register(name=_name, toolset=_name, schema={}, handler=h)
        """)
        assert idx.toolsets == frozenset({"discord", "discord_admin"})

    @pytest.mark.parametrize("source", [
        # Computed at runtime.
        "registry.register(name=make_name(), toolset='x', schema={}, handler=h)\n",
        # A constant rebound later in the module.
        "NAME = 'a'\nNAME = 'b'\nregistry.register(name=NAME, toolset='x', schema={}, handler=h)\n",
        # A schema mutated in place.
        "S = {'name': 'a'}\nS['name'] = 'b'\nregistry.register(name=S['name'], toolset='x', schema=S, handler=h)\n",
        # A further registration outside the module-level statements the index reads.
        _register_line("a", "x") + "if FLAG:\n    " + _register_line("b", "x"),
        _register_line("a", "x") + "def later():\n    " + _register_line("b", "x"),
        # Keyword splat.
        "registry.register(**SPEC)\n",
    ])
    def test_unresolvable_registrations_make_the_index_opaque(self, source):
        assert _index(source) == ToolModuleIndex(names=None, toolsets=None)

    def test_module_registering_nothing_at_import_is_not_a_tool_module(self):
        assert _index("def later():\n    " + _register_line("a", "x")) is None


_REALITY_PROBE = """
import json, sys
from tools import registry as registry_mod
recorded = {}
real_register = registry_mod.ToolRegistry.register

def recording_register(self, name, toolset, *args, **kwargs):
    caller = sys._getframe(1).f_globals.get("__name__", "")
    recorded.setdefault(caller, []).append([name, toolset])
    return real_register(self, name, toolset, *args, **kwargs)

registry_mod.ToolRegistry.register = recording_register
index = registry_mod._scan_builtin_tool_modules(None)
imported = registry_mod.discover_builtin_tools()
print(json.dumps({
    "index": {m: [sorted(i.names) if i.names is not None else None,
                  sorted(i.toolsets) if i.toolsets is not None else None] for m, i in index.items()},
    "imported": imported,
    "recorded": recorded,
}))
"""


def test_static_index_matches_what_every_builtin_module_registers(tmp_path):
    """The index decides which module a lookup imports, so it must be exact: each built-in module
    registers exactly the names and toolsets the index read from its source, and every
    registration made while importing the built-ins comes from an indexed module."""
    env = {**os.environ, "HERMES_HOME": str(tmp_path / "home"), "PYTHONPATH": str(REPO_ROOT)}
    env.pop("HERMES_PROFILE", None)
    probe = subprocess.run([sys.executable, "-c", _REALITY_PROBE], cwd=REPO_ROOT, env=env,
                           stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=120, check=False)
    assert probe.returncode == 0, probe.stderr[-4000:]
    result = json.loads(probe.stdout.strip().splitlines()[-1])
    index, recorded = result["index"], result["recorded"]

    opaque = sorted(m for m, (names, _) in index.items() if names is None)
    assert not opaque, (
        f"{opaque} register tools in a way the discovery index cannot read, so every toolset lookup "
        "imports them. Register at module level with literal names/toolsets (or a module-level `for` "
        "over a literal table), as tools/AGENTS.md shows.")
    assert set(result["imported"]) == set(index)
    for module, (names, toolsets) in index.items():
        actual = recorded.get(module, [])
        assert sorted({n for n, _ in actual}) == names, module
        assert sorted({t for _, t in actual}) == toolsets, module
    unindexed = sorted(m for m in recorded if m.startswith("tools.") and m not in index)
    assert not unindexed, f"{unindexed} registered tools while the built-ins imported but are not indexed"
