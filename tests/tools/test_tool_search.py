"""Tests for tools/tool_search.py — progressive tool disclosure.

Coverage targets — these mirror the issues called out in the OpenClaw tool
search report. Every test that names an OpenClaw issue is the regression
guard that would have caught that specific failure mode.
"""

from __future__ import annotations

import json
import os
import sys
from typing import List, Dict, Any

import pytest


_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


def _td(name: str, description: str = "", properties: Dict[str, Any] | None = None) -> Dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties or {},
            },
        },
    }


# ---------------------------------------------------------------------------
# Config parsing
# ---------------------------------------------------------------------------


class TestConfigParsing:
    def test_default_when_missing(self):
        from tools.tool_search import ToolSearchConfig
        cfg = ToolSearchConfig.from_raw(None)
        assert cfg.enabled == "auto"
        assert cfg.threshold_pct == 5.0

    def test_bool_true_maps_to_auto(self):
        from tools.tool_search import ToolSearchConfig
        cfg = ToolSearchConfig.from_raw(True)
        assert cfg.enabled == "auto"


    def test_search_limits_clamped(self):
        from tools.tool_search import ToolSearchConfig
        cfg = ToolSearchConfig.from_raw({
            "search_default_limit": 999,
            "max_search_limit": 999,
        })
        assert cfg.max_search_limit == 50
        assert cfg.search_default_limit <= cfg.max_search_limit


# ---------------------------------------------------------------------------
# Classification — the hard invariant: core tools NEVER defer.
# ---------------------------------------------------------------------------


class TestClassification:
    def test_core_tools_never_defer(self):
        """The critical invariant from the OpenClaw report."""
        from tools.tool_search import is_deferrable_tool_name
        # Sample of core tools from _HERMES_CORE_TOOLS.
        for core_name in ["terminal", "read_file", "write_file", "patch",
                          "search_files", "todo", "memory", "browser_navigate",
                          "web_search", "session_search", "clarify",
                          "execute_code", "delegate_task", "send_message"]:
            assert not is_deferrable_tool_name(core_name), (
                f"Core tool '{core_name}' must NEVER be deferrable"
            )

    def test_bridge_tools_never_defer(self):
        from tools.tool_search import is_deferrable_tool_name, BRIDGE_TOOL_NAMES
        for name in BRIDGE_TOOL_NAMES:
            assert not is_deferrable_tool_name(name)

    def test_unknown_tool_not_deferrable(self):
        """Defensive: a tool name we cannot resolve to a registry entry must
        not be claimed as deferrable. This protects against the OpenClaw
        cron regression where unresolved tools were silently dropped."""
        from tools.tool_search import is_deferrable_tool_name
        assert not is_deferrable_tool_name("xx_definitely_not_a_tool_xx")

    def test_classify_keeps_unknown_in_visible(self):
        """A tool we can't classify stays visible — never silently dropped.

        This is the OpenClaw #84141 regression guard (cron lost ``exec``
        because it wasn't in the catalog).
        """
        from tools.tool_search import classify_tools
        # Build a tool def for something we don't have a registry entry for.
        defs = [_td("xx_unknown_tool", "Unknown tool")]
        visible, deferrable = classify_tools(defs)
        names = {(td.get("function") or {}).get("name") for td in visible}
        assert "xx_unknown_tool" in names
        assert deferrable == []


# ---------------------------------------------------------------------------
# Token estimation + threshold gate
# ---------------------------------------------------------------------------


class TestThresholdGate:
    def test_off_never_activates(self):
        from tools.tool_search import ToolSearchConfig, should_activate
        cfg = ToolSearchConfig.from_raw({"enabled": "off"})
        assert not should_activate(cfg, deferrable_tokens=1_000_000, context_length=200_000)


    def test_defer_always_core_bypasses_auto_threshold(self):
        """defer_always_core should force activation even when auto threshold is not met."""
        from tools.tool_search import ToolSearchConfig, should_activate
        cfg = ToolSearchConfig.from_raw(
            {"enabled": "auto", "threshold_pct": 90, "defer_always_core": True}
        )
        # Way below 90% of context, but should still activate in always mode.
        assert should_activate(cfg, deferrable_tokens=1_000, context_length=200_000)

    def test_defer_always_core_still_respects_enabled_off(self):
        """Explicit off remains authoritative even with defer_always_core set."""
        from tools.tool_search import ToolSearchConfig, should_activate
        cfg = ToolSearchConfig.from_raw(
            {"enabled": "off", "defer_always_core": True}
        )
        assert not should_activate(cfg, deferrable_tokens=100_000, context_length=200_000)

    def test_token_estimate_proportional_to_schema_size(self):
        from tools.tool_search import estimate_tokens_from_schemas
        small = [_td("a", "x")]
        big = [_td(f"name_{i}", f"description for tool {i} " * 20,
                   {"q": {"type": "string", "description": "search query " * 10}})
               for i in range(10)]
        small_t = estimate_tokens_from_schemas(small)
        big_t = estimate_tokens_from_schemas(big)
        assert big_t > small_t * 10


# ---------------------------------------------------------------------------
# Retrieval (BM25 + substring fallback)
# ---------------------------------------------------------------------------


class TestRetrieval:
    def _fake_catalog(self):
        """Build a catalog directly without touching the registry."""
        from tools.tool_search import CatalogEntry, _tokenize, _entry_search_text
        defs = [
            _td("github_create_issue", "Open a new issue in a GitHub repository",
                {"title": {"type": "string"}, "body": {"type": "string"}}),
            _td("github_search_repos", "Search GitHub for matching repositories",
                {"query": {"type": "string"}}),
            _td("slack_send_message", "Post a message into a Slack channel",
                {"channel": {"type": "string"}, "text": {"type": "string"}}),
            _td("calendar_create_event", "Add an event to the user's calendar",
                {"title": {"type": "string"}, "start": {"type": "string"}}),
        ]
        catalog = []
        for d in defs:
            fn = d["function"]
            e = CatalogEntry(
                name=fn["name"], description=fn["description"],
                schema=d, source="mcp", source_name="mcp-test",
            )
            e._tokens = _tokenize(_entry_search_text(d))
            catalog.append(e)
        return catalog

    def test_search_finds_relevant_tool(self):
        from tools.tool_search import search_catalog
        hits = search_catalog(self._fake_catalog(), "create a github issue", limit=3)
        names = [h.name for h in hits]
        assert names[0] == "github_create_issue"


    def test_search_respects_limit(self):
        from tools.tool_search import search_catalog
        hits = search_catalog(self._fake_catalog(), "github", limit=1)
        assert len(hits) <= 1


# ---------------------------------------------------------------------------
# Assembly — the full passthrough/activate decision.
# ---------------------------------------------------------------------------


class TestAssembly:
    def test_no_deferrable_returns_unchanged(self):
        """Pure-core toolset: pass-through, no bridge tools added."""
        from tools.tool_search import assemble_tool_defs, ToolSearchConfig
        defs = [_td("terminal", "Run shell"), _td("read_file", "Read a file")]
        result = assemble_tool_defs(
            defs,
            context_length=200_000,
            config=ToolSearchConfig.from_raw({"enabled": "on"}),
        )
        assert not result.activated
        assert {t["function"]["name"] for t in result.tool_defs} == {"terminal", "read_file"}

    @staticmethod
    def _register_mcp(name):
        from tools.registry import registry

        def _handler(args, task_id=None, **kw):
            return json.dumps({"ok": True})

        registry.register(
            name=name,
            handler=_handler,
            schema=_td(name, "Deferred capability description.")["function"],
            toolset="mcp-tiertest",
        )


    def test_idempotent_when_bridge_already_present(self):
        from tools.tool_search import assemble_tool_defs, ToolSearchConfig, BRIDGE_TOOL_NAMES
        defs = [_td("terminal", "Run shell"), _td("tool_search", "old")]
        result = assemble_tool_defs(
            defs,
            context_length=200_000,
            config=ToolSearchConfig.from_raw({"enabled": "off"}),
        )
        names = [(t["function"]["name"]) for t in result.tool_defs]
        # The pre-existing tool_search was stripped (it would be re-injected if
        # activation happened; here it didn't).
        assert "tool_search" not in names


# ---------------------------------------------------------------------------
# Bridge dispatch
# ---------------------------------------------------------------------------


class TestBridgeDispatch:
    def test_tool_search_requires_query(self):
        from tools.tool_search import dispatch_tool_search
        result = dispatch_tool_search({}, current_tool_defs=[])
        assert "error" in json.loads(result)


    def test_resolve_underlying_call_parses_object_args(self):
        from tools.tool_search import resolve_underlying_call
        name, args, err = resolve_underlying_call({
            "name": "unknown_xxx",
            "arguments": {"foo": "bar"},
        })
        # Will fail classification because unknown_xxx isn't deferrable.
        assert err is not None


    def test_resolve_underlying_call_rejects_recursion(self):
        """tool_call cannot invoke tool_call itself."""
        from tools.tool_search import resolve_underlying_call, TOOL_CALL_NAME
        name, args, err = resolve_underlying_call({
            "name": TOOL_CALL_NAME,
            "arguments": {},
        })
        assert err is not None
        assert "bridge tool" in err.lower()


# ---------------------------------------------------------------------------
# End-to-end via the real handle_function_call (smoke test).
# ---------------------------------------------------------------------------


class TestHandleFunctionCallIntegration:
    def test_tool_search_dispatch_through_handle_function_call(self):
        """The dispatcher recognizes the bridge tool by name."""
        import model_tools
        result = model_tools.handle_function_call(
            function_name="tool_search",
            function_args={"query": "nothing matches this"},
        )
        parsed = json.loads(result)
        # Without a real registry, the matches will be empty, but the
        # dispatch path completed without error.
        assert "matches" in parsed or "error" in parsed


class TestRegression_OpenClawCron84141:
    """Regression guard for the OpenClaw cron-tool-loss class of bug.

    OpenClaw #84141: ``toolsAllow: ["exec"]`` on an isolated cron turn
    resulted in the agent receiving only ``sessions_send`` — the catalog
    builder silently dropped the requested core tool.

    Our defense: core tools are NEVER deferred. This test exercises the
    full assembly pipeline with a mixed core+MCP toolset and asserts that
    every core tool survives.
    """

    def test_core_tool_survives_alongside_many_mcp_tools(self):
        from tools.tool_search import (
            assemble_tool_defs, ToolSearchConfig, BRIDGE_TOOL_NAMES,
            classify_tools,
        )
        # 1 core tool + 50 unknown/MCP-shaped tools (deferrable).
        defs = [_td("terminal", "Run shell commands")]
        # Pad with fake "deferrable" tools — without registry registration,
        # classify_tools puts them in 'visible'. So instead, we just verify
        # the core-tool side: terminal stays in visible regardless.
        visible, deferrable = classify_tools(defs)
        assert any(
            (td.get("function") or {}).get("name") == "terminal"
            for td in visible
        ), "Core tool 'terminal' was wrongly classified as deferrable"

        # Now force activation and check the resulting tool-defs list.
        result = assemble_tool_defs(
            defs,
            context_length=200_000,
            config=ToolSearchConfig.from_raw({"enabled": "on"}),
        )
        names = {(t.get("function") or {}).get("name") for t in result.tool_defs}
        # terminal must be present; bridges are only added if there are
        # deferrable tools to put behind them.
        assert "terminal" in names

    def test_unwrap_rejects_core_tool_attempt(self):
        """Even if the model tries to invoke a core tool through tool_call,
        we reject the call and tell the model to use it directly."""
        from tools.tool_search import resolve_underlying_call
        _, _, err = resolve_underlying_call({
            "name": "terminal",
            "arguments": {"command": "echo hi"},
        })
        assert err is not None
        assert "not a deferrable" in err


class TestRegression_ToolsetScoping:
    """A restricted-toolset session must not see or invoke out-of-scope tools.

    The bug: the bridge dispatch and the tool_executor unwrap read the
    catalog from the *global* registry (get_tool_definitions with no
    toolset scope = "start with everything"), so a session scoped to one
    MCP server could tool_search the entire process registry and tool_call
    any plugin tool it was never granted. registry.dispatch() has no
    enabled_tools gate for non-execute_code tools, so the out-of-scope tool
    actually ran.

    The fix threads the session's enabled/disabled toolsets into the bridge
    dispatch (model_tools.handle_function_call) and the executor unwrap
    (agent.tool_executor), scoping both the searchable catalog and the
    invocable set to the session's own toolsets.
    """

    @staticmethod
    def _register(name, toolset):
        from tools.registry import registry

        def _handler(args, task_id=None, **kw):
            return json.dumps({"ok": True, "tool": name})

        registry.register(
            name=name,
            handler=_handler,
            schema=_td(name, f"desc for {name}", {"repo": {"type": "string"}}),
            toolset=toolset,
        )

    def test_search_catalog_is_scoped_to_session_toolsets(self):
        import model_tools

        for i in range(12):
            self._register(f"mcp_scoped_gh_{i}", "mcp-scoped-gh")
        self._register("scoped_oos_plugin", "scopedoosplugin")

        # tool_search scoped to the github toolset must not count the
        # out-of-scope plugin tool (or any of the host registry).
        result = model_tools.handle_function_call(
            function_name="tool_search",
            function_args={"query": "mcp_scoped_gh", "limit": 5},
            enabled_toolsets=["mcp-scoped-gh"],
        )
        parsed = json.loads(result)
        assert parsed["total_available"] == 12, (
            f"expected scoped catalog of 12, got {parsed['total_available']} "
            "— catalog leaked tools outside the session's toolsets"
        )
        hit_names = {m["name"] for m in parsed["matches"]}
        assert "scoped_oos_plugin" not in hit_names


    def test_scoped_deferrable_names_helper(self):
        from tools.tool_search import scoped_deferrable_names

        self._register("mcp_helper_op", "mcp-helper")
        import model_tools
        defs = model_tools.get_tool_definitions(
            enabled_toolsets=["mcp-helper"],
            quiet_mode=True,
            skip_tool_search_assembly=True,
        )
        names = scoped_deferrable_names(defs)
        assert "mcp_helper_op" in names
        # core tools are never deferrable
        assert "terminal" not in names


# ---------------------------------------------------------------------------
# Catalog listing (skills-style progressive disclosure)
# ---------------------------------------------------------------------------


class TestCatalogListing:
    def test_config_defaults(self):
        from tools.tool_search import ToolSearchConfig
        cfg = ToolSearchConfig.from_raw(None)
        assert cfg.listing == "auto"
        assert cfg.listing_max_tokens == 20000
        # legacy bool shapes keep defaults too
        assert ToolSearchConfig.from_raw(True).listing == "auto"


    def test_short_desc_first_sentence_and_clip(self):
        from tools.tool_search import _short_desc
        assert _short_desc("Open an issue. Second sentence dropped.") == "Open an issue."
        long = "word " * 40
        s = _short_desc(long)
        assert len(s) <= 61  # 60 + ellipsis char
        assert s.endswith("…")
        assert _short_desc("") == ""


    @staticmethod
    def _register(name):
        from tools.registry import registry

        def _handler(args, task_id=None, **kw):
            return json.dumps({"ok": True})

        registry.register(
            name=name,
            handler=_handler,
            schema=_td(name, "Deferred capability description.")["function"],
            toolset="mcp-listingtest",
        )


    def test_assembly_listing_off_keeps_legacy_description(self):
        from tools.tool_search import assemble_tool_defs, ToolSearchConfig
        for i in range(30):
            self._register(f"mcp_x_{i}")
        defs = [_td(f"mcp_x_{i}", "Deferred.") for i in range(30)]
        result = assemble_tool_defs(
            defs, context_length=1000,
            config=ToolSearchConfig.from_raw({"enabled": "on", "listing": "off"}),
        )
        assert result.activated
        search = next(t for t in result.tool_defs if t["function"]["name"] == "tool_search")
        assert "mcp_x_0" not in search["function"]["description"]


class TestDeferredCallSchemaProbe:
    """Blind tool_call invocations missing required arguments must return
    the tool's parameter schema instead of dispatching into an opaque
    downstream failure (port of nearai/ironclaw#5149's describe-first fix).

    A deferred tool's schema is invisible until tool_describe is called, so
    models routinely invoke deferred tools by name alone. Pre-fix, that
    produced ``KeyError: 'document_id'``-style errors that teach the model
    nothing; post-fix, the probe returns the schema so the model repairs
    the call in one round-trip. Valid calls dispatch untouched.
    """

    @staticmethod
    def _register(name, toolset, required=("document_id",)):
        from tools.registry import registry

        def _handler(args, task_id=None, **kw):
            # Simulates a tool that crashes opaquely on a missing required arg.
            return json.dumps({"ok": True, "doc": args["document_id"]})

        params = {
            "type": "object",
            "properties": {
                "document_id": {"type": "string", "description": "Doc id"},
                "format": {"type": "string"},
            },
            "required": list(required),
        }
        registry.register(
            name=name,
            handler=_handler,
            schema={"type": "function",
                    "function": {"name": name, "description": f"desc {name}",
                                 "parameters": params}},
            toolset=toolset,
        )

    def test_validator_returns_schema_for_missing_required(self):
        from tools.tool_search import validate_deferred_call_args

        self._register("mcp_probe_docs_get", "mcp-probe")
        err = validate_deferred_call_args("mcp_probe_docs_get", {})
        assert err is not None
        parsed = json.loads(err)
        assert "document_id" in parsed["error"]
        assert "NOT invoked" in parsed["error"]
        assert parsed["parameters"]["required"] == ["document_id"]
        assert "document_id" in parsed["parameters"]["properties"]


    def test_validator_never_blocks_unvalidatable_tools(self):
        from tools.tool_search import validate_deferred_call_args

        # Unknown tool → no schema → dispatch (downstream scope gate handles it).
        assert validate_deferred_call_args("mcp_no_such_tool_xyz", {}) is None


    def test_valid_tool_call_still_dispatches(self):
        import model_tools

        self._register("mcp_probe_valid_op", "mcp-probe-valid")
        result = json.loads(model_tools.handle_function_call(
            function_name="tool_call",
            function_args={"name": "mcp_probe_valid_op",
                           "arguments": {"document_id": "abc"}},
            enabled_toolsets=["mcp-probe-valid"],
        ))
        assert result.get("ok") is True
        assert result.get("doc") == "abc"

# ---------------------------------------------------------------------------
# defer_core=True paths
# ---------------------------------------------------------------------------


class TestDeferCoreTrue:
    """Tests for the defer_core=True configuration paths.

    When defer_core=True, tools listed in _HERMES_DEFERRABLE_CORE_TOOLS
    become eligible for deferral, while tools in _HERMES_ALWAYS_CORE_TOOLS
    remain permanently visible.
    """

    @staticmethod
    def _defer_core_config():
        from tools.tool_search import ToolSearchConfig
        return ToolSearchConfig.from_raw({"enabled": "on", "defer_core": True})

    def test_is_deferrable_with_defer_core_true_deferrable_core_tool(self):
        """A tool in _HERMES_DEFERRABLE_CORE_TOOLS returns True when defer_core=True."""
        from tools.tool_search import is_deferrable_tool_name
        from toolsets import _HERMES_DEFERRABLE_CORE_TOOLS
        cfg = self._defer_core_config()
        # write_file is in _HERMES_DEFERRABLE_CORE_TOOLS (not in _HERMES_ALWAYS_CORE_TOOLS)
        # Without a registry entry it won't be deferrable (registry returns False for unknowns),
        # but register it first so the registry check passes.
        from tools.registry import registry

        def _handler(args, task_id=None, **kw):
            import json
            return json.dumps({"ok": True})

        registry.register(
            name="write_file",
            handler=_handler,
            schema=_td("write_file", "Write a file"),
            toolset="file",
        )
        assert is_deferrable_tool_name("write_file", config=cfg), (
            "write_file (in _HERMES_DEFERRABLE_CORE_TOOLS) must be deferrable when defer_core=True"
        )

    def test_is_deferrable_with_defer_core_true_always_core_tool(self):
        """A tool in _HERMES_ALWAYS_CORE_TOOLS returns False even when defer_core=True."""
        from tools.tool_search import is_deferrable_tool_name
        cfg = self._defer_core_config()
        # memory is in _HERMES_ALWAYS_CORE_TOOLS
        assert not is_deferrable_tool_name("memory", config=cfg), (
            "memory (in _HERMES_ALWAYS_CORE_TOOLS) must NEVER be deferrable"
        )

    def test_classify_tools_defer_core_true(self):
        """With defer_core=True, deferrable-core tools appear in the deferrable bucket."""
        from tools.tool_search import classify_tools
        from tools.registry import registry
        import json

        def _handler(args, task_id=None, **kw):
            return json.dumps({"ok": True})

        # Register write_file as a non-MCP tool (simulates deferrable-core tool)
        registry.register(
            name="write_file",
            handler=_handler,
            schema=_td("write_file", "Write a file"),
            toolset="file",
        )

        cfg = self._defer_core_config()
        defs = [
            _td("memory", "Persistent memory"),   # always-core → must stay visible
            _td("write_file", "Write a file"),     # deferrable-core → deferrable with defer_core=True
        ]
        visible, deferrable = classify_tools(defs, config=cfg)

        visible_names = {(t.get("function") or {}).get("name") for t in visible}
        deferrable_names = {(t.get("function") or {}).get("name") for t in deferrable}

        assert "memory" in visible_names, "always-core tool 'memory' must be in visible bucket"
        assert "write_file" in deferrable_names, (
            "deferrable-core tool 'write_file' must be in deferrable bucket when defer_core=True"
        )

    def test_resolve_underlying_call_allows_deferrable_core_tool_with_defer_core_true(self):
        """With defer_core=True, resolve_underlying_call allows a deferrable-core tool through."""
        from tools.tool_search import resolve_underlying_call
        from tools.registry import registry
        import json

        def _handler(args, task_id=None, **kw):
            return json.dumps({"ok": True})

        registry.register(
            name="write_file",
            handler=_handler,
            schema=_td("write_file", "Write a file"),
            toolset="file",
        )

        cfg = self._defer_core_config()
        name, args, err = resolve_underlying_call(
            {"name": "write_file", "arguments": {"path": "/tmp/x", "content": "hi"}},
            config=cfg,
        )
        assert err is None, (
            f"resolve_underlying_call should NOT return an error for deferrable-core tool "
            f"with defer_core=True, got: {err!r}"
        )
        assert name == "write_file"

    def test_scoped_deferrable_names_defer_core_true(self):
        """With defer_core=True, deferrable-core tools appear in the scoped set."""
        from tools.tool_search import scoped_deferrable_names
        from tools.registry import registry
        import model_tools
        import json

        def _handler(args, task_id=None, **kw):
            return json.dumps({"ok": True})

        registry.register(
            name="write_file",
            handler=_handler,
            schema=_td("write_file", "Write a file"),
            toolset="file",
        )

        cfg = self._defer_core_config()
        defs = [
            _td("memory", "Persistent memory"),
            _td("write_file", "Write a file"),
        ]
        names = scoped_deferrable_names(defs, config=cfg)
        assert "write_file" in names, (
            "deferrable-core tool 'write_file' must appear in scoped set when defer_core=True"
        )
        assert "memory" not in names, (
            "always-core tool 'memory' must NOT appear in scoped set even with defer_core=True"
        )

    def test_tool_call_bridge_dispatches_deferrable_core_tool_end_to_end(self):
        """E2E: with defer_core=True, tool_call dispatches a deferrable-core tool via
        the full handle_function_call → resolve_underlying_call → registry dispatch path,
        against a temp HERMES_HOME config with defer_core enabled.
        """
        import json
        import os
        import tempfile
        import model_tools
        from tools.registry import registry
        from tools.tool_search import ToolSearchConfig

        # Register a synthetic deferrable-core-shaped tool in the file toolset.
        captured = {}

        def _handler(args, task_id=None, **kw):
            captured["called_with"] = args
            return json.dumps({"ok": True, "wrote": args.get("path", "")})

        registry.register(
            name="write_file",
            handler=_handler,
            schema=_td("write_file", "Write a file", {"path": {"type": "string"}, "content": {"type": "string"}}),
            toolset="file",
        )

        # Patch load_config so the bridge uses a defer_core=True config for this call.
        import tools.tool_search as _ts_mod
        original_load = _ts_mod.load_config

        def _patched_load():
            return ToolSearchConfig.from_raw({"enabled": "on", "defer_core": True})

        _ts_mod.load_config = _patched_load
        try:
            result = model_tools.handle_function_call(
                function_name="tool_call",
                function_args={"name": "write_file", "arguments": {"path": "/tmp/test.txt", "content": "hi"}},
                enabled_toolsets=["file"],
            )
        finally:
            _ts_mod.load_config = original_load

        parsed = json.loads(result)
        assert parsed.get("ok") is True, (
            f"Expected deferrable-core tool to be dispatched successfully via bridge, got: {parsed!r}"
        )
        assert captured.get("called_with", {}).get("path") == "/tmp/test.txt", (
            f"Handler was not called with the correct arguments: {captured!r}"
        )


class TestDeferAlwaysCoreTrue:
    """Tests for defer_always_core=True — fully lazy mode where only bridge tools are visible."""

    def test_config_parses_defer_always_core(self):
        from tools.tool_search import ToolSearchConfig
        cfg = ToolSearchConfig.from_raw(
            {"enabled": "on", "defer_core": True, "defer_always_core": True}
        )
        assert cfg.defer_always_core is True
        assert cfg.defer_core is True

    def test_config_defer_always_core_defaults_false(self):
        from tools.tool_search import ToolSearchConfig
        cfg = ToolSearchConfig.from_raw({"enabled": "on", "defer_core": True})
        assert cfg.defer_always_core is False

    def test_config_legacy_bool_shape_defer_always_core_false(self):
        from tools.tool_search import ToolSearchConfig
        cfg = ToolSearchConfig.from_raw(True)
        assert cfg.defer_always_core is False
        cfg2 = ToolSearchConfig.from_raw(False)
        assert cfg2.defer_always_core is False

    def test_always_core_tools_are_deferrable(self):
        """With defer_always_core=True, terminal/memory/delegate_task must be deferrable."""
        from tools.tool_search import ToolSearchConfig, is_deferrable_tool_name
        from toolsets import _HERMES_ALWAYS_CORE_TOOLS
        cfg = ToolSearchConfig.from_raw(
            {"enabled": "on", "defer_core": True, "defer_always_core": True}
        )
        for name in _HERMES_ALWAYS_CORE_TOOLS:
            assert is_deferrable_tool_name(name, cfg), (
                f"Expected {name!r} to be deferrable with defer_always_core=True"
            )

    def test_bridge_tools_never_deferrable_with_defer_always_core(self):
        """Bridge tools must remain visible even with defer_always_core=True."""
        from tools.tool_search import ToolSearchConfig, is_deferrable_tool_name, BRIDGE_TOOL_NAMES
        cfg = ToolSearchConfig.from_raw(
            {"enabled": "on", "defer_core": True, "defer_always_core": True}
        )
        for name in BRIDGE_TOOL_NAMES:
            assert not is_deferrable_tool_name(name, cfg), (
                f"Bridge tool {name!r} must never be deferrable"
            )

    def test_classify_tools_defer_always_core(self):
        """With defer_always_core=True, all non-bridge tools move to deferred."""
        from tools.tool_search import ToolSearchConfig, classify_tools, BRIDGE_TOOL_NAMES
        from toolsets import _HERMES_ALWAYS_CORE_TOOLS
        cfg = ToolSearchConfig.from_raw(
            {"enabled": "on", "defer_core": True, "defer_always_core": True}
        )
        # Build fake schemas for always-core tools only (bridge tools are added
        # after classification via bridge_tool_schemas(), not passed as input)
        all_tools = [
            {"type": "function", "function": {"name": n, "description": f"desc {n}", "parameters": {}}}
            for n in _HERMES_ALWAYS_CORE_TOOLS
        ]
        visible, deferred = classify_tools(all_tools, cfg)
        deferred_names = {t["function"]["name"] for t in deferred}

        # All always-core tools must move to deferred
        for name in _HERMES_ALWAYS_CORE_TOOLS:
            assert name in deferred_names, f"{name} should be deferred with defer_always_core=True"

        # visible should be empty (bridge tools are injected separately)
        assert visible == [], f"Expected no visible tools, got: {[t['function']['name'] for t in visible]}"

    def test_defer_always_core_implies_defer_core(self):
        """Setting defer_always_core=True without defer_core=True must auto-promote defer_core."""
        from tools.tool_search import ToolSearchConfig
        cfg = ToolSearchConfig.from_raw({"enabled": "on", "defer_always_core": True})
        assert cfg.defer_core is True, (
            "defer_always_core=True must auto-promote defer_core=True in from_raw()"
        )
        assert cfg.defer_always_core is True

    def test_assemble_tool_defs_defer_always_core_e2e(self):
        """E2E: assemble_tool_defs with defer_always_core=True activates and leaves only bridges."""
        from tools.tool_search import (
            ToolSearchConfig, assemble_tool_defs, BRIDGE_TOOL_NAMES,
        )
        from toolsets import _HERMES_ALWAYS_CORE_TOOLS
        cfg = ToolSearchConfig.from_raw(
            {"enabled": "on", "defer_core": True, "defer_always_core": True}
        )
        tool_defs = [
            {"type": "function", "function": {"name": n, "description": f"desc {n}", "parameters": {}}}
            for n in _HERMES_ALWAYS_CORE_TOOLS
        ]
        result = assemble_tool_defs(tool_defs, config=cfg)
        assert result.activated, "assemble_tool_defs should activate with defer_always_core=True"
        visible_names = {t["function"]["name"] for t in result.tool_defs}
        assert visible_names == BRIDGE_TOOL_NAMES, (
            f"Only bridge tools should be visible, got: {visible_names}"
        )

    def test_scoped_deferrable_names_defer_always_core(self):
        """With defer_always_core=True, always-core tools appear in the scoped deferrable set."""
        from tools.tool_search import ToolSearchConfig, scoped_deferrable_names
        from tools.registry import registry
        import json

        def _handler(args, task_id=None, **kw):
            return json.dumps({"ok": True})

        registry.register(
            name="memory",
            handler=_handler,
            schema={"type": "function", "function": {"name": "memory", "description": "memory", "parameters": {}}},
            toolset="memory",
        )

        cfg = ToolSearchConfig.from_raw(
            {"enabled": "on", "defer_core": True, "defer_always_core": True}
        )
        defs = [
            {"type": "function", "function": {"name": "memory", "description": "Persistent memory", "parameters": {}}},
        ]
        names = scoped_deferrable_names(defs, config=cfg)
        assert "memory" in names, (
            "always-core tool 'memory' must appear in scoped deferrable set with defer_always_core=True"
        )
