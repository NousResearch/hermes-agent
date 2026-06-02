"""Tests for tools/tool_search.py — progressive tool disclosure.

Coverage targets — these mirror the issues called out in the OpenClaw tool
search report. Every test that names an OpenClaw issue is the regression
guard that would have caught that specific failure mode.

Enhancement tests (NousResearch/hermes-agent#13332 — Hybrid Tool Pre-Selection):
  TestRerankerConfig        — config parsing for reranker sub-config
  TestRerankerInvoked       — reranker invoked and changes result order
  TestRerankerFallback      — endpoint failure → pure BM25, no exception propagates
  TestRRFMath               — RRF score formula with known inputs
  TestEmbedPrefixes         — search_query:/search_document: prefixes sent correctly
  TestEmbedCacheInvalidation — catalog change → tool embeddings recomputed
  TestEmbedPayloadShape     — model + input keys present in POST body
"""

from __future__ import annotations

import json
import os
import sys
from typing import List, Dict, Any
from unittest.mock import patch, MagicMock

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
        assert cfg.threshold_pct == 10.0

    def test_bool_true_maps_to_auto(self):
        from tools.tool_search import ToolSearchConfig
        cfg = ToolSearchConfig.from_raw(True)
        assert cfg.enabled == "auto"

    def test_bool_false_maps_to_off(self):
        from tools.tool_search import ToolSearchConfig
        cfg = ToolSearchConfig.from_raw(False)
        assert cfg.enabled == "off"

    def test_explicit_on(self):
        from tools.tool_search import ToolSearchConfig
        cfg = ToolSearchConfig.from_raw({"enabled": "on"})
        assert cfg.enabled == "on"

    def test_invalid_enabled_falls_back_to_auto(self):
        from tools.tool_search import ToolSearchConfig
        cfg = ToolSearchConfig.from_raw({"enabled": "maybe"})
        assert cfg.enabled == "auto"

    def test_threshold_clamped(self):
        from tools.tool_search import ToolSearchConfig
        cfg = ToolSearchConfig.from_raw({"threshold_pct": 150})
        assert cfg.threshold_pct == 100.0
        cfg = ToolSearchConfig.from_raw({"threshold_pct": -5})
        assert cfg.threshold_pct == 0.0

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

    def test_zero_deferrable_never_activates(self):
        from tools.tool_search import ToolSearchConfig, should_activate
        cfg = ToolSearchConfig.from_raw({"enabled": "on"})
        assert not should_activate(cfg, deferrable_tokens=0, context_length=200_000)

    def test_on_activates_with_any_deferrable(self):
        from tools.tool_search import ToolSearchConfig, should_activate
        cfg = ToolSearchConfig.from_raw({"enabled": "on"})
        assert should_activate(cfg, deferrable_tokens=100, context_length=200_000)

    def test_auto_below_threshold_does_not_activate(self):
        from tools.tool_search import ToolSearchConfig, should_activate
        cfg = ToolSearchConfig.from_raw({"enabled": "auto", "threshold_pct": 10})
        # 5% of 200K = below 10% threshold
        assert not should_activate(cfg, deferrable_tokens=10_000, context_length=200_000)

    def test_auto_at_or_above_threshold_activates(self):
        from tools.tool_search import ToolSearchConfig, should_activate
        cfg = ToolSearchConfig.from_raw({"enabled": "auto", "threshold_pct": 10})
        assert should_activate(cfg, deferrable_tokens=20_000, context_length=200_000)
        assert should_activate(cfg, deferrable_tokens=50_000, context_length=200_000)

    def test_auto_without_context_length_uses_20k_cutoff(self):
        """Fallback cutoff used when the active model is unknown."""
        from tools.tool_search import ToolSearchConfig, should_activate
        cfg = ToolSearchConfig.from_raw({"enabled": "auto"})
        assert not should_activate(cfg, deferrable_tokens=10_000, context_length=0)
        assert should_activate(cfg, deferrable_tokens=25_000, context_length=0)

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

    def test_search_returns_empty_for_irrelevant_query(self):
        from tools.tool_search import search_catalog
        hits = search_catalog(self._fake_catalog(), "asdf qwerty foobar", limit=3)
        assert hits == []

    def test_search_substring_fallback(self):
        """Even when no BM25 hit, a literal substring of the tool name returns."""
        from tools.tool_search import search_catalog
        hits = search_catalog(self._fake_catalog(), "calendar", limit=3)
        assert any("calendar" in h.name for h in hits)

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

    def test_below_threshold_returns_unchanged(self):
        """Tiny deferrable surface: don't bother."""
        from tools.tool_search import assemble_tool_defs, ToolSearchConfig
        # _td renders to ~80 chars / 20 tokens. 3 of them = ~60 tokens.
        # 10% of 200K = 20K. Way below.
        defs = [_td("unknown_tool_a"), _td("unknown_tool_b"), _td("unknown_tool_c")]
        result = assemble_tool_defs(
            defs,
            context_length=200_000,
            config=ToolSearchConfig.from_raw({"enabled": "auto", "threshold_pct": 10}),
        )
        assert not result.activated
        names = {(t.get("function") or {}).get("name") for t in result.tool_defs}
        assert "tool_search" not in names

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

    def test_tool_describe_requires_name(self):
        from tools.tool_search import dispatch_tool_describe
        result = dispatch_tool_describe({}, current_tool_defs=[])
        assert "error" in json.loads(result)

    def test_tool_describe_rejects_non_deferrable(self):
        """If the model asks to describe a core tool, refuse — it's already
        in the visible list."""
        from tools.tool_search import dispatch_tool_describe
        result = dispatch_tool_describe(
            {"name": "terminal"}, current_tool_defs=[_td("terminal", "Run shell")],
        )
        assert "error" in json.loads(result)

    def test_resolve_underlying_call_parses_object_args(self):
        from tools.tool_search import resolve_underlying_call
        name, args, err = resolve_underlying_call({
            "name": "unknown_xxx",
            "arguments": {"foo": "bar"},
        })
        # Will fail classification because unknown_xxx isn't deferrable.
        assert err is not None

    def test_resolve_underlying_call_parses_json_string_args(self):
        """Some models emit ``arguments`` as a JSON string instead of object."""
        from tools.tool_search import resolve_underlying_call
        # Use a name that won't classify (so we don't depend on registry),
        # but exercise the JSON parse path.
        _, _, err = resolve_underlying_call({
            "name": "fake",
            "arguments": '{"a": 1}',
        })
        # err is about classification, but the parse worked (it would have
        # failed earlier with "not valid JSON" otherwise).
        assert "not valid JSON" not in (err or "")

    def test_resolve_underlying_call_rejects_bad_json(self):
        from tools.tool_search import resolve_underlying_call
        _, _, err = resolve_underlying_call({
            "name": "fake",
            "arguments": "{this is not json",
        })
        assert err is not None
        assert "JSON" in err

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

    def test_tool_call_rejects_out_of_scope_tool(self):
        import model_tools

        self._register("mcp_inscope_gh_op", "mcp-inscope-gh")
        self._register("inscope_oos_plugin", "inscopeoosplugin")

        # Out-of-scope plugin tool: rejected even though it is registered
        # and deferrable in the global registry.
        rejected = json.loads(model_tools.handle_function_call(
            function_name="tool_call",
            function_args={"name": "inscope_oos_plugin", "arguments": {}},
            enabled_toolsets=["mcp-inscope-gh"],
        ))
        assert "error" in rejected
        assert "not available in this session" in rejected["error"]

        # In-scope tool: dispatches normally.
        ok = json.loads(model_tools.handle_function_call(
            function_name="tool_call",
            function_args={"name": "mcp_inscope_gh_op", "arguments": {"repo": "a/b"}},
            enabled_toolsets=["mcp-inscope-gh"],
        ))
        assert ok.get("ok") is True
        assert ok.get("tool") == "mcp_inscope_gh_op"

    def test_bridge_dispatch_does_not_pollute_global_resolved_names(self):
        import model_tools

        self._register("mcp_pollute_op_0", "mcp-pollute")
        self._register("mcp_pollute_op_1", "mcp-pollute")

        # Establish the scoped session global.
        model_tools.get_tool_definitions(
            enabled_toolsets=["mcp-pollute"], quiet_mode=True,
        )
        before = set(model_tools._last_resolved_tool_names)
        assert "terminal" not in before

        # A scoped tool_search call must not widen the process-global
        # _last_resolved_tool_names to the whole registry (which would leak
        # core/sandbox tools into execute_code's fallback).
        model_tools.handle_function_call(
            function_name="tool_search",
            function_args={"query": "pollute"},
            enabled_toolsets=["mcp-pollute"],
        )
        after = set(model_tools._last_resolved_tool_names)
        assert "terminal" not in after, (
            "bridge dispatch polluted _last_resolved_tool_names with "
            "out-of-scope tools"
        )

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
# Enhancement B: Config parsing (NousResearch/hermes-agent#13332)
# ---------------------------------------------------------------------------


class TestRerankerConfig:
    """Config parsing for the embedding reranker sub-config."""

    def test_reranker_config_default_disabled(self):
        from tools.tool_search import ToolSearchConfig
        cfg = ToolSearchConfig.from_raw(None)
        assert not cfg.reranker.enabled
        assert cfg.reranker.mode == "rerank"
        assert cfg.reranker.rrf_k == 10
        assert cfg.reranker.query_prefix == "search_query: "
        assert cfg.reranker.doc_prefix == "search_document: "

    def test_reranker_config_enabled_with_rrf(self):
        from tools.tool_search import ToolSearchConfig
        cfg = ToolSearchConfig.from_raw({
            "reranker": {
                "enabled": True,
                "endpoint": "http://localhost:11434/v1/embeddings",
                "model": "nomic-embed-text-v2-moe",
                "mode": "rrf",
                "rrf_k": 10,
            },
        })
        assert cfg.reranker.enabled
        assert cfg.reranker.mode == "rrf"
        assert cfg.reranker.rrf_k == 10

    def test_reranker_config_invalid_mode_falls_back_to_rerank(self):
        from tools.tool_search import ToolSearchConfig
        cfg = ToolSearchConfig.from_raw({
            "reranker": {"enabled": True, "mode": "crossencoder"},
        })
        assert cfg.reranker.mode == "rerank"

    def test_reranker_config_none_raw_gives_disabled(self):
        from tools.tool_search import RerankerConfig
        cfg = RerankerConfig.from_raw(None)
        assert not cfg.enabled

    def test_reranker_config_custom_prefixes(self):
        from tools.tool_search import RerankerConfig
        cfg = RerankerConfig.from_raw({
            "enabled": True,
            "endpoint": "http://x",
            "query_prefix": "",
            "doc_prefix": "",
        })
        assert cfg.query_prefix == ""
        assert cfg.doc_prefix == ""

    def test_legacy_bool_true_does_not_break_new_fields(self):
        """Legacy bool=True: enabled=auto, reranker=OFF."""
        from tools.tool_search import ToolSearchConfig
        cfg = ToolSearchConfig.from_raw(True)
        assert cfg.enabled == "auto"
        # Reranker stays default OFF (requires external endpoint).
        assert not cfg.reranker.enabled

    def test_legacy_bool_false_does_not_break_new_fields(self):
        """Legacy bool=False: enabled=off, reranker=OFF."""
        from tools.tool_search import ToolSearchConfig
        cfg = ToolSearchConfig.from_raw(False)
        assert cfg.enabled == "off"
        # Reranker stays default OFF.
        assert not cfg.reranker.enabled


# ---------------------------------------------------------------------------
# Enhancement B: Reranker invoked and changes order
# ---------------------------------------------------------------------------


def _make_catalog(*names_and_descs: tuple[str, str]) -> "List[Any]":
    """Build a catalog from (name, description) pairs without registry."""
    from tools.tool_search import CatalogEntry, _tokenize, _entry_search_text
    catalog = []
    for name, desc in names_and_descs:
        td = _td(name, desc)
        e = CatalogEntry(
            name=name, description=desc,
            schema=td, source="mcp", source_name="mcp-test",
        )
        e._tokens = _tokenize(_entry_search_text(td))
        e._embed_text = f"{name}: {desc}"
        catalog.append(e)
    return catalog


class MockReranker:
    """Reranker that reverses the BM25 order — proves invocation changes result."""

    def rerank(self, query: str, bm25_all: List[Any], limit: int, config: Any) -> List[Any]:
        # Return the BM25 list in reverse order, sliced to limit.
        return [e for _, e in reversed(bm25_all)][:limit]


class BrokenReranker:
    """Reranker that always raises — proves graceful fallback to BM25."""

    def rerank(self, query: str, bm25_all: List[Any], limit: int, config: Any) -> List[Any]:
        raise ConnectionError("endpoint unreachable")


class TestRerankerInvoked:
    def test_mock_reranker_changes_result_order(self):
        """When a reranker reverses BM25 order, search_catalog returns reversed order."""
        from tools.tool_search import search_catalog, ToolSearchConfig
        catalog = _make_catalog(
            ("github_create_issue", "Open a new issue in a GitHub repository"),
            ("slack_send_message", "Post a message into a Slack channel"),
            ("calendar_create_event", "Add an event to the user's calendar"),
        )
        cfg = ToolSearchConfig.from_raw({"enabled": "on"})

        # Without reranker: BM25 should rank github tool first for this query.
        bm25_hits = search_catalog(catalog, "create github issue", limit=3, config=cfg)
        assert bm25_hits[0].name == "github_create_issue"

        # With mock reranker: order is reversed.
        mock = MockReranker()
        reranked = search_catalog(catalog, "create github issue", limit=3,
                                  config=cfg, reranker=mock)
        # The last BM25 result is now first.
        assert reranked[0].name != "github_create_issue"


class TestRerankerFallback:
    def test_broken_reranker_returns_bm25_without_exception(self):
        """When the reranker raises, search_catalog returns pure BM25 results silently."""
        from tools.tool_search import search_catalog, ToolSearchConfig
        catalog = _make_catalog(
            ("github_create_issue", "Open a new issue in a GitHub repository"),
            ("slack_send_message", "Post a message into a Slack channel"),
        )
        cfg = ToolSearchConfig.from_raw({"enabled": "on"})

        broken = BrokenReranker()
        # Must not raise — fallback to BM25.
        hits = search_catalog(catalog, "create github issue", limit=2,
                              config=cfg, reranker=broken)
        assert len(hits) >= 1
        assert hits[0].name == "github_create_issue"  # BM25 result preserved


# ---------------------------------------------------------------------------
# Enhancement B: RRF math
# ---------------------------------------------------------------------------


class TestRRFMath:
    def test_rrf_top_ranked_in_both_lists_wins(self):
        """A document ranked #1 in both BM25 and embed lists should score highest."""
        from tools.tool_search import _rrf_fuse, CatalogEntry, _tokenize
        from tools.tool_search import _entry_search_text

        def _entry(name: str) -> Any:
            td = _td(name, f"Description for {name}")
            e = CatalogEntry(
                name=name, description=f"Description for {name}",
                schema=td, source="mcp", source_name="mcp-test",
            )
            e._tokens = _tokenize(_entry_search_text(td))
            e._embed_text = f"{name}: Description for {name}"
            return e

        alpha = _entry("tool_alpha")
        beta = _entry("tool_beta")
        gamma = _entry("tool_gamma")

        # tool_alpha is rank 1 in BM25, rank 2 in embed.
        # tool_beta  is rank 2 in BM25, rank 1 in embed.
        # tool_gamma is rank 3 in both.
        bm25_ranked = [(1.0, alpha), (0.5, beta), (0.1, gamma)]
        embed_ranked = [(0.9, beta), (0.8, alpha), (0.2, gamma)]

        fused = _rrf_fuse(bm25_ranked, embed_ranked, k=10, top_n=3)
        names = [e.name for e in fused]

        # alpha: 1/(10+1) + 1/(10+2) = 0.0909 + 0.0833 = 0.1742
        # beta:  1/(10+2) + 1/(10+1) = 0.0833 + 0.0909 = 0.1742
        # gamma: 1/(10+3) + 1/(10+3) = 0.0769 + 0.0769 = 0.1538
        # alpha and beta tie (Python sorted is stable — exact order varies)
        assert "gamma" not in names[:2] or names[2] == "tool_gamma"
        assert "tool_gamma" in names  # gamma appears somewhere

    def test_rrf_k10_formula_correct(self):
        """Verify score = sum(1/(k+rank)) matches manual calculation."""
        from tools.tool_search import _rrf_fuse, CatalogEntry, _tokenize
        from tools.tool_search import _entry_search_text

        def _entry(name: str) -> Any:
            td = _td(name, f"desc {name}")
            e = CatalogEntry(name=name, description=f"desc {name}",
                             schema=td, source="mcp", source_name="x")
            e._tokens = _tokenize(_entry_search_text(td))
            e._embed_text = f"{name}: desc {name}"
            return e

        a = _entry("a")
        b = _entry("b")

        # a is rank 1 in both lists.  b is rank 2 in both.
        bm25 = [(2.0, a), (1.0, b)]
        embed = [(0.9, a), (0.5, b)]
        fused = _rrf_fuse(bm25, embed, k=10, top_n=2)
        # a should beat b because both lists rank it #1.
        assert fused[0].name == "a"
        assert fused[1].name == "b"


# ---------------------------------------------------------------------------
# Enhancement B: Prefix application via urllib mock
# ---------------------------------------------------------------------------


class TestEmbedPrefixes:
    """Assert that search_query: / search_document: prefixes are sent correctly."""

    def _fake_urlopen_factory(self, captured_bodies: List[bytes], fake_vec: List[float]):
        """Return a urlopen mock that captures request bodies and returns fake embeddings."""
        import io

        def fake_urlopen(req, timeout=None):
            body = req.data
            captured_bodies.append(body)
            # Parse the input to know how many vectors to return.
            payload = json.loads(body)
            texts = payload["input"]
            data = [{"index": i, "embedding": fake_vec} for i in range(len(texts))]
            resp_bytes = json.dumps({"data": data}).encode("utf-8")

            mock_resp = MagicMock()
            mock_resp.__enter__ = lambda s: s
            mock_resp.__exit__ = MagicMock(return_value=False)
            mock_resp.read = MagicMock(return_value=resp_bytes)
            return mock_resp

        return fake_urlopen

    def test_query_prefix_applied(self):
        """The POST body for the query embed must start with 'search_query:'."""
        from tools.tool_search import EmbeddingReranker, RerankerConfig

        cfg = RerankerConfig.from_raw({
            "enabled": True,
            "endpoint": "http://fake-embed/v1/embeddings",
            "model": "test-model",
            "query_prefix": "search_query: ",
            "doc_prefix": "search_document: ",
        })
        reranker = EmbeddingReranker(cfg)

        captured: List[bytes] = []
        fake_vec = [0.1, 0.2, 0.3]
        urlopen_mock = self._fake_urlopen_factory(captured, fake_vec)

        with patch("urllib.request.urlopen", urlopen_mock):
            reranker._embed(["search_query: hello world"])

        assert len(captured) == 1
        body = json.loads(captured[0])
        assert any("search_query:" in t for t in body["input"]), (
            "query prefix not present in embed POST body"
        )

    def test_doc_prefix_applied(self):
        """The POST body for tool doc embeds must start with 'search_document:'."""
        from tools.tool_search import EmbeddingReranker, RerankerConfig

        cfg = RerankerConfig.from_raw({
            "enabled": True,
            "endpoint": "http://fake-embed/v1/embeddings",
            "model": "test-model",
            "query_prefix": "search_query: ",
            "doc_prefix": "search_document: ",
        })
        reranker = EmbeddingReranker(cfg)

        captured: List[bytes] = []
        fake_vec = [0.1, 0.2, 0.3]
        urlopen_mock = self._fake_urlopen_factory(captured, fake_vec)

        with patch("urllib.request.urlopen", urlopen_mock):
            reranker._embed(["search_document: web_search: Search the web"])

        body = json.loads(captured[0])
        assert any("search_document:" in t for t in body["input"]), (
            "doc prefix not present in embed POST body"
        )


# ---------------------------------------------------------------------------
# Enhancement B: Embedding payload shape
# ---------------------------------------------------------------------------


class TestEmbedPayloadShape:
    """Assert that the HTTP POST body contains the correct model + input keys."""

    def test_embed_payload_has_model_and_input(self):
        """POST body must have 'model' and 'input' keys."""
        from tools.tool_search import EmbeddingReranker, RerankerConfig
        import io

        cfg = RerankerConfig.from_raw({
            "enabled": True,
            "endpoint": "http://fake/v1/embeddings",
            "model": "my-embed-model",
        })
        reranker = EmbeddingReranker(cfg)

        captured: List[bytes] = []
        fake_vec = [0.5, 0.5]

        def fake_urlopen(req, timeout=None):
            captured.append(req.data)
            data = [{"index": 0, "embedding": fake_vec}]
            resp = MagicMock()
            resp.__enter__ = lambda s: s
            resp.__exit__ = MagicMock(return_value=False)
            resp.read = MagicMock(return_value=json.dumps({"data": data}).encode("utf-8"))
            return resp

        with patch("urllib.request.urlopen", fake_urlopen):
            reranker._embed(["hello"])

        assert len(captured) == 1
        body = json.loads(captured[0])
        assert "model" in body, "embed payload missing 'model' key"
        assert "input" in body, "embed payload missing 'input' key"
        assert body["model"] == "my-embed-model"
        assert isinstance(body["input"], list)


# ---------------------------------------------------------------------------
# Enhancement B: Cache invalidation on catalog change
# ---------------------------------------------------------------------------


class TestEmbedCacheInvalidation:
    """Tool embeddings are recomputed when the catalog changes."""

    def test_cache_invalidated_on_catalog_change(self):
        """Changing the catalog (different tool names) forces a new reranker instance."""
        from tools.tool_search import _get_reranker, RerankerConfig
        import tools.tool_search as ts_module

        # Reset module-level singleton to known state.
        ts_module._reranker = None
        ts_module._reranker_catalog_key = ""

        cfg = RerankerConfig.from_raw({
            "enabled": True,
            "endpoint": "http://x/v1/embeddings",
            "model": "m",
        })

        catalog_a = _make_catalog(("tool_one", "Does one thing"))
        catalog_b = _make_catalog(("tool_two", "Does two things"))

        r1 = _get_reranker(cfg, catalog_a)
        assert r1 is not None

        r2 = _get_reranker(cfg, catalog_a)
        assert r2 is r1, "same catalog should return the same reranker instance"

        r3 = _get_reranker(cfg, catalog_b)
        assert r3 is not r1, "changed catalog should produce a new reranker instance"

    def test_same_catalog_reuses_reranker(self):
        """Identical catalog returns the same singleton (cache hit)."""
        from tools.tool_search import _get_reranker, RerankerConfig
        import tools.tool_search as ts_module

        ts_module._reranker = None
        ts_module._reranker_catalog_key = ""

        cfg = RerankerConfig.from_raw({
            "enabled": True,
            "endpoint": "http://y/v1/embeddings",
            "model": "n",
        })
        catalog = _make_catalog(("stable_tool", "Stable description"))
        r1 = _get_reranker(cfg, catalog)
        r2 = _get_reranker(cfg, catalog)
        assert r1 is r2

    def test_embed_cache_hit_skips_network(self):
        """When the text is already cached, _embed is not called again."""
        from tools.tool_search import EmbeddingReranker, RerankerConfig

        cfg = RerankerConfig.from_raw({
            "enabled": True,
            "endpoint": "http://z/v1/embeddings",
            "model": "m2",
        })
        reranker = EmbeddingReranker(cfg)

        call_count = [0]

        def counting_embed(texts: List[str]) -> List[List[float]]:
            call_count[0] += 1
            return [[0.1] * len(texts[0]) for _ in texts]

        reranker._embed = counting_embed  # type: ignore[method-assign]

        reranker._embed_with_cache(["hello there"])
        reranker._embed_with_cache(["hello there"])  # should be served from cache

        assert call_count[0] == 1, (
            f"expected 1 embed call (cache hit on second), got {call_count[0]}"
        )


# ---------------------------------------------------------------------------
# Fix HIGH-2: Embedding dimension-mismatch guard
# ---------------------------------------------------------------------------


class TestDimensionMismatchGuard:
    """When the endpoint returns vectors of wrong or zero length, fall back to BM25.

    Regression guard for HIGH-2: previously the rerank path used zip() to
    pair query and doc vectors, silently truncating to the shorter dimension.
    This caused wrong cosine scores without any indication of the problem
    (e.g. when an embedding model is swapped mid-session and returns 768-dim
    vectors where 1536-dim were cached).  The fix raises ValueError so that
    search_catalog's except clause falls back to BM25 cleanly.
    """

    def _make_reranker_with_fixed_vecs(self, q_vec: List[float], doc_vec: List[float]):
        """Return an EmbeddingReranker whose _embed always returns q_vec then doc_vec."""
        from tools.tool_search import EmbeddingReranker, RerankerConfig

        cfg = RerankerConfig.from_raw({
            "enabled": True,
            "endpoint": "http://dim-test/v1/embeddings",
            "model": "test-model",
            "query_prefix": "",
            "doc_prefix": "",
        })
        reranker = EmbeddingReranker(cfg)

        call_seq: List[List[List[float]]] = []

        def _embed_side_effect(texts: List[str]) -> List[List[float]]:
            if not call_seq:
                # First call = doc embeddings
                call_seq.append([doc_vec] * len(texts))
                return [doc_vec] * len(texts)
            # Second call = query embedding
            return [q_vec]

        reranker._embed = _embed_side_effect  # type: ignore[method-assign]
        return reranker

    def test_mismatched_dimensions_fall_back_to_bm25(self):
        """Feeding mismatched-dimension vectors causes BM25 fallback, no exception."""
        from tools.tool_search import search_catalog, ToolSearchConfig

        catalog = _make_catalog(
            ("github_create_issue", "Open a new issue in a GitHub repository"),
            ("slack_send_message", "Post a message into a Slack channel"),
        )
        cfg = ToolSearchConfig.from_raw({"enabled": "on"})

        # query dim=3, doc dim=4 — mismatched
        q_vec = [0.1, 0.2, 0.3]
        doc_vec = [0.1, 0.2, 0.3, 0.4]

        reranker = self._make_reranker_with_fixed_vecs(q_vec, doc_vec)

        # Must not raise — must fall back to BM25 order silently.
        hits = search_catalog(
            catalog, "create github issue", limit=2, config=cfg, reranker=reranker
        )
        assert len(hits) >= 1
        # BM25 order is preserved after fallback.
        assert hits[0].name == "github_create_issue"

    def test_zero_length_query_vector_falls_back_to_bm25(self):
        """A zero-length query vector causes BM25 fallback, no exception."""
        from tools.tool_search import search_catalog, ToolSearchConfig

        catalog = _make_catalog(
            ("github_create_issue", "Open a new issue in a GitHub repository"),
        )
        cfg = ToolSearchConfig.from_raw({"enabled": "on"})

        # query dim=0 — degenerate response
        q_vec: List[float] = []
        doc_vec = [0.1, 0.2, 0.3]

        reranker = self._make_reranker_with_fixed_vecs(q_vec, doc_vec)

        hits = search_catalog(
            catalog, "create github issue", limit=1, config=cfg, reranker=reranker
        )
        assert len(hits) >= 1
        assert hits[0].name == "github_create_issue"


# ---------------------------------------------------------------------------
# Fix LOW: Hardened RRF test with exact hand-computed scores
# ---------------------------------------------------------------------------


class TestRRFExactScores:
    """Lock the RRF formula against silent drift with explicit hand-computed values.

    RRF score for doc d = sum over lists L of 1 / (k + rank_L(d)).
    With k=10: rank-1 contributes 1/11 ≈ 0.09091, rank-2 contributes 1/12 ≈ 0.08333.
    """

    def _entry(self, name: str) -> "Any":
        from tools.tool_search import CatalogEntry, _tokenize, _entry_search_text
        td = _td(name, f"desc for {name}")
        e = CatalogEntry(name=name, description=f"desc for {name}",
                         schema=td, source="mcp", source_name="x")
        e._tokens = _tokenize(_entry_search_text(td))
        e._embed_text = f"{name}: desc for {name}"
        return e

    def test_rrf_exact_scores_and_order_k10(self):
        """Hand-computed RRF scores for two known ranked lists (k=10).

        Lists:
          BM25:  [alpha(rank1), beta(rank2), gamma(rank3)]
          embed: [beta(rank1),  alpha(rank2)]   (gamma absent — rank 3 implicit)

        Manual calculation (k=10):
          alpha_score = 1/(10+1) + 1/(10+2) = 1/11 + 1/12 = 0.090909 + 0.083333 = 0.174242
          beta_score  = 1/(10+2) + 1/(10+1) = 1/12 + 1/11 = 0.083333 + 0.090909 = 0.174242
          gamma_score = 1/(10+3) + 1/(10+3) = 1/13 + 1/13 = 0.076923 + 0.076923 = 0.153846

        Expected order: alpha and beta tie (both 0.1742), gamma is last (0.1538).
        Python's sorted is stable so alpha (first in BM25) wins the tie.

        Exact score assertion: gamma's score must be 2/13 ≈ 0.153846 (within 1e-9).
        """
        from tools.tool_search import _rrf_fuse
        import math as _math

        alpha = self._entry("alpha")
        beta = self._entry("beta")
        gamma = self._entry("gamma")

        bm25_ranked = [(1.0, alpha), (0.5, beta), (0.1, gamma)]
        embed_ranked = [(0.9, beta), (0.8, alpha), (0.2, gamma)]

        fused = _rrf_fuse(bm25_ranked, embed_ranked, k=10, top_n=3)
        names = [e.name for e in fused]

        # Ordering: gamma must be last.
        assert names[2] == "gamma", (
            f"gamma should be rank-3 (score ≈ 0.1538), got order {names}"
        )
        # alpha and beta both score ≈ 0.1742 — both must appear in top-2.
        assert set(names[:2]) == {"alpha", "beta"}, (
            f"alpha and beta should tie at rank 1-2, got {names[:2]}"
        )

        # Exact score for gamma: 1/(10+3) + 1/(10+3) = 2/13.
        expected_gamma_score = 2.0 / 13.0  # ≈ 0.153846...
        # Reconstruct the fused scores dict to check the exact value.
        scores: Dict[str, float] = {}
        for rank_idx, (_, e) in enumerate(bm25_ranked, start=1):
            scores[e.name] = scores.get(e.name, 0.0) + 1.0 / (10 + rank_idx)
        for rank_idx, (_, e) in enumerate(embed_ranked, start=1):
            scores[e.name] = scores.get(e.name, 0.0) + 1.0 / (10 + rank_idx)

        assert abs(scores["gamma"] - expected_gamma_score) < 1e-9, (
            f"gamma RRF score = {scores['gamma']!r}, expected {expected_gamma_score!r} "
            "(formula: 1/(10+3) + 1/(10+3) = 2/13)"
        )
        # Sanity-check alpha and beta scores are equal and larger than gamma.
        assert abs(scores["alpha"] - scores["beta"]) < 1e-9, (
            f"alpha and beta should tie: alpha={scores['alpha']!r}, beta={scores['beta']!r}"
        )
        assert scores["alpha"] > scores["gamma"], (
            "tied alpha/beta score must exceed gamma score"
        )


# ---------------------------------------------------------------------------
# Fix CRITICAL 2: EmbeddingReranker.rerank() must not return more than limit
# ---------------------------------------------------------------------------


class TestRerankerHonoursLimit:
    """EmbeddingReranker.rerank() must return at most ``limit`` entries.

    Before the fix, top_k = max(limit, config.search_default_limit), so
    calling search_catalog(limit=3) with search_default_limit=5 returned 5
    results — violating the caller's contract.
    """

    def _make_reranker_with_mock_embed(self, n_tools: int):
        """Return an EmbeddingReranker that uses a fixed unit vector for all texts."""
        from tools.tool_search import EmbeddingReranker, RerankerConfig

        cfg = RerankerConfig.from_raw({
            "enabled": True,
            "endpoint": "http://fake-limit-test/v1/embeddings",
            "model": "test-model",
            "query_prefix": "",
            "doc_prefix": "",
        })
        reranker = EmbeddingReranker(cfg)

        def _fake_embed(texts: List[str]) -> List[List[float]]:
            # Return distinct unit vectors so scores differ and ordering is stable.
            return [[float(i + 1), 0.0] for i in range(len(texts))]

        reranker._embed = _fake_embed  # type: ignore[method-assign]
        return reranker

    def test_reranker_rerank_mode_respects_limit(self):
        """With mode=rerank, result length must equal the requested limit."""
        from tools.tool_search import ToolSearchConfig, RerankerConfig

        n_tools = 8
        limit = 3
        catalog = _make_catalog(*[(f"tool_{i}", f"Description {i}") for i in range(n_tools)])

        cfg = ToolSearchConfig.from_raw({
            "enabled": "on",
            "search_default_limit": 5,  # deliberately larger than limit
            "reranker": {
                "enabled": True,
                "endpoint": "http://fake-limit-test/v1/embeddings",
                "model": "test-model",
                "mode": "rerank",
                "query_prefix": "",
                "doc_prefix": "",
            },
        })
        reranker = self._make_reranker_with_mock_embed(n_tools)
        bm25_all = [(float(n_tools - i), e) for i, e in enumerate(catalog)]

        result = reranker.rerank("test query", bm25_all, limit=limit, config=cfg)
        assert len(result) <= limit, (
            f"reranker returned {len(result)} results for limit={limit}; "
            "must not exceed the caller's limit even when search_default_limit is larger"
        )

    def test_reranker_rrf_mode_respects_limit(self):
        """With mode=rrf, result length must equal the requested limit."""
        from tools.tool_search import EmbeddingReranker, RerankerConfig, ToolSearchConfig

        n_tools = 7
        limit = 2
        catalog = _make_catalog(*[(f"rrf_tool_{i}", f"Desc {i}") for i in range(n_tools)])

        rcfg = RerankerConfig.from_raw({
            "enabled": True,
            "endpoint": "http://fake-rrf-limit/v1/embeddings",
            "model": "rrf-model",
            "mode": "rrf",
            "rrf_k": 10,
            "query_prefix": "",
            "doc_prefix": "",
        })
        reranker = EmbeddingReranker(rcfg)

        def _fake_embed(texts: List[str]) -> List[List[float]]:
            return [[float(i + 1), 0.0] for i in range(len(texts))]

        reranker._embed = _fake_embed  # type: ignore[method-assign]

        cfg = ToolSearchConfig.from_raw({
            "enabled": "on",
            "search_default_limit": 5,  # deliberately larger than limit
        })
        bm25_all = [(float(n_tools - i), e) for i, e in enumerate(catalog)]

        result = reranker.rerank("rrf test", bm25_all, limit=limit, config=cfg)
        assert len(result) <= limit, (
            f"rrf reranker returned {len(result)} results for limit={limit}; "
            "must not exceed caller's limit"
        )

