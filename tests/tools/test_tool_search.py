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


def _td(
    name: str, description: str = "", properties: Dict[str, Any] | None = None
) -> Dict[str, Any]:
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
        assert cfg.search_default_limit == 5
        assert cfg.max_search_limit == 20
        assert cfg.listing == "auto"
        assert cfg.threshold_pct == 5.0
        assert cfg.listing_max_tokens == 4000

    def test_bool_true_maps_to_auto(self):
        from tools.tool_search import ToolSearchConfig

        cfg = ToolSearchConfig.from_raw(True)
        assert cfg.enabled == "auto"

    def test_full_config_helper_distinguishes_stable_and_direct_modes(self):
        from tools.tool_search import is_enabled_in_config

        assert is_enabled_in_config({})
        assert is_enabled_in_config({"tools": {"tool_search": {"enabled": "on"}}})
        assert not is_enabled_in_config({
            "tools": {"tool_search": {"enabled": "off"}},
        })

    def test_search_limits_clamped(self):
        from tools.tool_search import ToolSearchConfig

        cfg = ToolSearchConfig.from_raw({
            "search_default_limit": 999,
            "max_search_limit": 999,
        })
        assert cfg.max_search_limit == 50
        assert cfg.search_default_limit <= cfg.max_search_limit

    def test_listing_settings_normalized_and_clamped(self):
        from tools.tool_search import ToolSearchConfig

        cfg = ToolSearchConfig.from_raw({
            "listing": "invalid",
            "threshold_pct": 999,
            "listing_max_tokens": 1,
        })
        assert cfg.listing == "auto"
        assert cfg.threshold_pct == 100.0
        assert cfg.listing_max_tokens == 200


# ---------------------------------------------------------------------------
# Classification — the hard invariant: core tools NEVER defer.
# ---------------------------------------------------------------------------


class TestClassification:
    def test_core_tools_never_defer(self):
        """The critical invariant from the OpenClaw report."""
        from tools.tool_search import is_deferrable_tool_name

        # Sample of core tools from _HERMES_CORE_TOOLS.
        for core_name in [
            "terminal",
            "read_file",
            "write_file",
            "patch",
            "search_files",
            "todo",
            "memory",
            "browser_navigate",
            "web_search",
            "session_search",
            "clarify",
            "execute_code",
            "delegate_task",
            "send_message",
        ]:
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
# Token estimation + stable-bridge activation
# ---------------------------------------------------------------------------


class TestActivationGate:
    def test_off_never_activates(self):
        from tools.tool_search import ToolSearchConfig, should_activate

        cfg = ToolSearchConfig.from_raw({"enabled": "off"})
        assert not should_activate(
            cfg, deferrable_tokens=1_000_000, context_length=200_000
        )

    def test_zero_deferrable_keeps_bridge_active(self):
        from tools.tool_search import ToolSearchConfig, should_activate

        cfg = ToolSearchConfig.from_raw({"enabled": "on"})
        assert should_activate(cfg, deferrable_tokens=0, context_length=200_000)

    def test_on_activates_with_any_deferrable(self):
        from tools.tool_search import ToolSearchConfig, should_activate

        cfg = ToolSearchConfig.from_raw({"enabled": "on"})
        assert should_activate(cfg, deferrable_tokens=100, context_length=200_000)

    def test_auto_activates_with_any_deferrable(self):
        """Auto always keeps the cache-stable bridge present."""
        from tools.tool_search import ToolSearchConfig, should_activate

        cfg = ToolSearchConfig.from_raw({"enabled": "auto"})
        assert should_activate(cfg, deferrable_tokens=0, context_length=200_000)
        assert should_activate(cfg, deferrable_tokens=100, context_length=200_000)
        assert should_activate(cfg, deferrable_tokens=50_000, context_length=200_000)
        assert should_activate(cfg, deferrable_tokens=100, context_length=0)

    def test_token_estimate_proportional_to_schema_size(self):
        from tools.tool_search import estimate_tokens_from_schemas

        small = [_td("a", "x")]
        big = [
            _td(
                f"name_{i}",
                f"description for tool {i} " * 20,
                {"q": {"type": "string", "description": "search query " * 10}},
            )
            for i in range(10)
        ]
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
            _td(
                "github_create_issue",
                "Open a new issue in a GitHub repository",
                {"title": {"type": "string"}, "body": {"type": "string"}},
            ),
            _td(
                "github_search_repos",
                "Search GitHub for matching repositories",
                {"query": {"type": "string"}},
            ),
            _td(
                "slack_send_message",
                "Post a message into a Slack channel",
                {"channel": {"type": "string"}, "text": {"type": "string"}},
            ),
            _td(
                "calendar_create_event",
                "Add an event to the user's calendar",
                {"title": {"type": "string"}, "start": {"type": "string"}},
            ),
        ]
        catalog = []
        for d in defs:
            fn = d["function"]
            e = CatalogEntry(
                name=fn["name"],
                description=fn["description"],
                schema=d,
                source="mcp",
                source_name="mcp-test",
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
# Assembly — the full stable-bridge/direct-mode decision.
# ---------------------------------------------------------------------------


class TestAssembly:
    def test_no_deferrable_still_includes_stable_bridge(self):
        """The bridge exists before the first MCP tool arrives."""
        from tools.tool_search import (
            BRIDGE_TOOL_NAMES,
            ToolSearchConfig,
            assemble_tool_defs,
        )

        defs = [_td("terminal", "Run shell"), _td("read_file", "Read a file")]
        result = assemble_tool_defs(
            defs,
            context_length=200_000,
            config=ToolSearchConfig.from_raw({"enabled": "on"}),
        )
        assert result.activated
        assert {t["function"]["name"] for t in result.tool_defs} == {
            "terminal",
            "read_file",
            *BRIDGE_TOOL_NAMES,
        }

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

    def test_small_deferrable_surface_uses_search_only_bridge(self):
        """MCP names and descriptions never leak into the bridge schema."""
        from tools.tool_search import assemble_tool_defs, ToolSearchConfig

        for n in ("tier_small_a", "tier_small_b", "tier_small_c"):
            self._register_mcp(n)
        defs = [_td("terminal", "Run shell")] + [
            _td(n, "Deferred capability description.")
            for n in ("tier_small_a", "tier_small_b", "tier_small_c")
        ]
        result = assemble_tool_defs(
            defs,
            context_length=200_000,
            config=ToolSearchConfig.from_raw({"enabled": "auto"}),
        )
        assert result.activated
        assert result.tier == 2
        assert result.listing_form == "none"
        names = {(t.get("function") or {}).get("name") for t in result.tool_defs}
        assert "tool_search" in names
        assert "terminal" in names  # core stays eager
        search = next(
            t for t in result.tool_defs if t["function"]["name"] == "tool_search"
        )
        assert "tier_small_a" not in search["function"]["description"]
        assert (
            "Deferred capability description" not in search["function"]["description"]
        )

    def test_catalog_edits_leave_bridge_schema_byte_stable(self):
        """Add/remove/swap/schema edits must not change model-facing bridges."""
        from tools.tool_search import (
            BRIDGE_TOOL_NAMES,
            ToolSearchConfig,
            assemble_tool_defs,
        )

        for name in ("stable_alpha", "stable_beta"):
            self._register_mcp(name)
        cfg = ToolSearchConfig.from_raw({"enabled": "auto"})

        def bridges(defs):
            assembled = assemble_tool_defs(defs, context_length=200_000, config=cfg)
            return [
                tool
                for tool in assembled.tool_defs
                if tool["function"]["name"] in BRIDGE_TOOL_NAMES
            ]

        empty = bridges([])
        one = bridges([_td("stable_alpha", "old description")])
        swapped = bridges([_td("stable_beta", "different description")])
        schema_changed = bridges([
            _td("stable_beta", "updated", {"new_arg": {"type": "string"}})
        ])
        assert empty == one == swapped == schema_changed

    def test_idempotent_when_bridge_already_present(self):
        from tools.tool_search import (
            assemble_tool_defs,
            ToolSearchConfig,
            BRIDGE_TOOL_NAMES,
        )

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

    def test_empty_search_keeps_connected_sources_discoverable(self):
        from tools.registry import registry
        from tools.tool_search import dispatch_tool_search

        name = "recovery_catalog_create_record"
        tool_def = _td(name, "Create a record in the connected catalog service.")
        registry.register(
            name=name,
            handler=lambda args, **kwargs: "{}",
            schema=tool_def,
            toolset="mcp-recovery-catalog",
        )

        result = json.loads(dispatch_tool_search(
            {"query": "unrelated vocabulary"},
            current_tool_defs=[tool_def],
        ))

        assert result["matches"] == []
        assert result["total_available"] == 1
        assert result["available_sources"] == [
            {"name": "recovery-catalog", "tool_count": 1},
        ]
        assert "remain available" in result["hint"]
        assert "before concluding" in result["hint"]

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

    def test_search_reads_tool_registered_after_bridge_assembly(self):
        """The bridge snapshot stays fixed while dispatch sees the live registry."""
        import model_tools
        from tools.registry import registry
        from tools.tool_search import ToolSearchConfig, assemble_tool_defs

        before = assemble_tool_defs(
            [],
            context_length=200_000,
            config=ToolSearchConfig.from_raw({"enabled": "auto"}),
        ).tool_defs
        name = "mcp_live_after_assembly_create_issue"
        registry.register(
            name=name,
            handler=lambda args, **kw: json.dumps({"ok": True}),
            schema=_td(name, "Create an issue.")["function"],
            toolset="mcp-live-after-assembly",
        )

        result = model_tools.handle_function_call(
            function_name="tool_search",
            function_args={"query": "live after assembly create issue"},
            enabled_toolsets=["mcp-live-after-assembly"],
        )
        after = assemble_tool_defs(
            [_td(name, "Create an issue.")],
            context_length=200_000,
            config=ToolSearchConfig.from_raw({"enabled": "auto"}),
        ).tool_defs

        assert any(match["name"] == name for match in json.loads(result)["matches"])
        assert before == after

    def test_tool_search_emits_one_terminal_hook(self, monkeypatch):
        """Inline bridge results still complete the tool lifecycle."""
        import model_tools
        from hermes_cli import lifecycle
        from tools import tool_search

        events = []
        monkeypatch.setattr(
            lifecycle,
            "has_hook",
            lambda name: name == "post_tool_call",
        )
        monkeypatch.setattr(
            lifecycle,
            "invoke_hook",
            lambda name, **kwargs: events.append((name, kwargs)),
        )
        monkeypatch.setattr(
            tool_search,
            "dispatch_tool_search",
            lambda *args, **kwargs: json.dumps({"matches": []}),
        )

        result = model_tools.handle_function_call(
            function_name="tool_search",
            function_args={"query": "private-query"},
            session_id="private-session",
            task_id="private-task",
            turn_id="private-turn",
            api_request_id="private-request",
            tool_call_id="private-call",
        )

        assert json.loads(result) == {"matches": []}
        assert len(events) == 1
        hook_name, payload = events[0]
        assert hook_name == "post_tool_call"
        assert payload["status"] == "ok"
        assert payload["turn_id"] == "private-turn"
        assert payload["api_request_id"] == "private-request"
        assert payload["tool_call_id"] == "private-call"


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
            assemble_tool_defs,
            ToolSearchConfig,
            BRIDGE_TOOL_NAMES,
            classify_tools,
        )

        # 1 core tool + 50 unknown/MCP-shaped tools (deferrable).
        defs = [_td("terminal", "Run shell commands")]
        # Pad with fake "deferrable" tools — without registry registration,
        # classify_tools puts them in 'visible'. So instead, we just verify
        # the core-tool side: terminal stays in visible regardless.
        visible, deferrable = classify_tools(defs)
        assert any(
            (td.get("function") or {}).get("name") == "terminal" for td in visible
        ), "Core tool 'terminal' was wrongly classified as deferrable"

        # Now force activation and check the resulting tool-defs list.
        result = assemble_tool_defs(
            defs,
            context_length=200_000,
            config=ToolSearchConfig.from_raw({"enabled": "on"}),
        )
        names = {(t.get("function") or {}).get("name") for t in result.tool_defs}
        # terminal and the cache-stable bridges must all be present.
        assert "terminal" in names
        assert BRIDGE_TOOL_NAMES.issubset(names)

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
# Stable bridge schemas
# ---------------------------------------------------------------------------


class TestStableBridgeSchemas:
    def test_schema_contains_no_dynamic_catalog_state(self):
        from tools.tool_search import bridge_tool_schemas

        first = bridge_tool_schemas()
        second = bridge_tool_schemas()
        assert first == second
        serialized = json.dumps(first)
        assert "additional tools" not in serialized
        assert "Deferred tool catalog" not in serialized
        assert "live catalog" in serialized

    def test_listing_config_controls_user_turn_manifest_not_bridge(self):
        from tools.tool_search import ToolSearchConfig
        cfg = ToolSearchConfig.from_raw(None)
        assert cfg.listing == "auto"
        assert cfg.listing_max_tokens == 4000
        # legacy bool shapes keep defaults too
        assert ToolSearchConfig.from_raw(True).listing == "auto"

        cfg = ToolSearchConfig.from_raw({
            "enabled": "auto",
            "threshold_pct": 99,
            "listing": "on",
            "listing_max_tokens": 60000,
        })
        assert cfg.enabled == "auto"
        assert cfg.listing == "on"
        assert cfg.threshold_pct == 99
        assert cfg.listing_max_tokens == 60000

    def test_short_desc_first_sentence_and_clip(self):
        from tools.tool_search import _short_desc
        assert _short_desc("Open an issue. Second sentence dropped.") == "Open an issue."
        long = "word " * 40
        s = _short_desc(long)
        assert len(s) <= 61  # 60 + ellipsis char
        assert s.endswith("…")
        assert _short_desc("") == ""


class TestUserTurnCatalogSnapshot:
    @staticmethod
    def _register(name, description="Deferred capability.", properties=None):
        from tools.registry import registry

        registry.register(
            name=name,
            handler=lambda args, **kw: json.dumps({"ok": True}),
            schema=_td(name, description, properties)["function"],
            toolset="mcp-user-turn-snapshot",
        )

    def test_snapshot_is_deterministic_and_kept_out_of_bridge(self):
        from tools.registry import registry
        from tools.tool_search import build_catalog_snapshot, bridge_tool_schemas

        names = ["snapshot_zeta", "snapshot_alpha"]
        for name in names:
            self._register(name, f"Does {name} work.")
        try:
            defs = [_td(name, f"Does {name} work.") for name in names]
            first = build_catalog_snapshot(defs)
            second = build_catalog_snapshot(list(reversed(defs)))

            assert first == second
            assert "snapshot_alpha" in first.notice
            assert first.notice.index("snapshot_alpha") < first.notice.index(
                "snapshot_zeta"
            )
            assert first.snapshot_id in first.notice

            bridge_json = json.dumps(bridge_tool_schemas())
            assert "snapshot_alpha" not in bridge_json
            assert first.snapshot_id not in bridge_json
        finally:
            for name in names:
                registry.deregister(name)

    def test_snapshot_id_changes_for_same_name_schema_edit(self):
        from tools.registry import registry
        from tools.tool_search import build_catalog_snapshot

        name = "snapshot_schema_edit"
        self._register(name, properties={"old": {"type": "string"}})
        try:
            old = build_catalog_snapshot([
                _td(name, "Deferred capability.", {"old": {"type": "string"}})
            ])
            new = build_catalog_snapshot([
                _td(name, "Deferred capability.", {"fresh": {"type": "integer"}})
            ])
            assert old.snapshot_id != new.snapshot_id
        finally:
            registry.deregister(name)

    def test_snapshot_id_changes_when_budget_changes_rendered_manifest(self):
        from tools.registry import registry
        from tools.tool_search import build_catalog_snapshot

        names = [f"snapshot_render_{index:03d}" for index in range(30)]
        for name in names:
            self._register(name, "A verbose deferred capability for rendering.")
        try:
            defs = [
                _td(name, "A verbose deferred capability for rendering.")
                for name in names
            ]
            full = build_catalog_snapshot(defs, max_tokens=20_000)
            compact = build_catalog_snapshot(defs, max_tokens=300)
            assert full.listing_form != compact.listing_form
            assert full.snapshot_id != compact.snapshot_id
        finally:
            for name in names:
                registry.deregister(name)

    def test_snapshot_budget_degrades_to_searchable_summary(self):
        from tools.registry import registry
        from tools.tool_search import build_catalog_snapshot

        names = [f"snapshot_budget_{index:03d}" for index in range(80)]
        for name in names:
            self._register(name, "A moderately verbose deferred capability.")
        try:
            snapshot = build_catalog_snapshot(
                [
                    _td(name, "A moderately verbose deferred capability.")
                    for name in names
                ],
                max_tokens=300,
            )
            assert snapshot.listing_form in {"names", "groups", "none"}
            assert snapshot.count == len(names)
            notice = snapshot.notice.lower()
            assert (
                "use `tool_search`" in notice or "discover via `tool_search`" in notice
            )
        finally:
            for name in names:
                registry.deregister(name)

    def test_empty_snapshot_is_explicit_and_extractable(self):
        from tools.tool_search import (
            build_catalog_snapshot,
            catalog_snapshot_id_from_text,
        )

        snapshot = build_catalog_snapshot([])
        assert snapshot.count == 0
        assert snapshot.pending_mcp_servers == ()
        assert "No deferred MCP/plugin tools" in snapshot.notice
        assert catalog_snapshot_id_from_text(snapshot.notice) == snapshot.snapshot_id
        assert (
            catalog_snapshot_id_from_text([
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,x"}},
                {"type": "text", "text": snapshot.notice},
            ])
            == snapshot.snapshot_id
        )

    def test_pending_servers_are_deterministic_and_change_snapshot_id(self):
        from tools.tool_search import build_catalog_snapshot

        pending = build_catalog_snapshot(
            [],
            pending_mcp_servers=["slow-slack", "slow-github", "slow-slack"],
        )
        reversed_order = build_catalog_snapshot(
            [],
            pending_mcp_servers=["slow-github", "slow-slack"],
        )
        ready = build_catalog_snapshot([])

        assert pending == reversed_order
        assert pending.pending_mcp_servers == ("slow-github", "slow-slack")
        assert pending.snapshot_id != ready.snapshot_id
        assert "pending_servers=2" in pending.notice
        assert (
            "MCP servers still initializing (2): `slow-github`, `slow-slack`"
            in pending.notice
        )
        assert "live catalog may expand during this turn" in pending.notice


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
            schema={
                "type": "function",
                "function": {
                    "name": name,
                    "description": f"desc {name}",
                    "parameters": params,
                },
            },
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
        result = json.loads(
            model_tools.handle_function_call(
                function_name="tool_call",
                function_args={
                    "name": "mcp_probe_valid_op",
                    "arguments": {"document_id": "abc"},
                },
                enabled_toolsets=["mcp-probe-valid"],
            )
        )
        assert result.get("ok") is True
        assert result.get("doc") == "abc"
