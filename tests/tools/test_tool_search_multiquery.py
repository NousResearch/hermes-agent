"""Multi-query ``tool_search``, batched ``tool_describe``, and stemming.

Covers the upgrade that replaced the single ``query`` string with
``queries: [str, ...]`` (grouped, split-shape response), the single
``name`` with ``names: [str, ...]`` (map response with ``not_found``),
and added Snowball stemming to the shared tokenizer.
"""

import json

import pytest


def _td(name, desc, props=None, required=None):
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": desc,
            "parameters": {
                "type": "object",
                "properties": props or {},
                "required": required or [],
            },
        },
    }


def _register(name, toolset, desc="Deferred capability.", props=None, required=None):
    from tools.registry import registry

    registry.register(
        name=name,
        handler=lambda args, **kw: json.dumps({"ok": True}),
        schema=_td(name, desc, props, required),
        toolset=toolset,
    )
    return _td(name, desc, props, required)


@pytest.fixture
def issue_defs():
    """A small deferred catalog registered under an MCP toolset."""
    return [
        _register("mq_linear_create_issue", "mcp-mq-linear",
                  "Create a new issue in a team.",
                  {"title": {"type": "string"}, "team": {"type": "string"}},
                  ["title", "team"]),
        _register("mq_linear_list_issues", "mcp-mq-linear",
                  "List issues in the workspace.",
                  {"query": {"type": "string"}}),
        _register("mq_slack_post_message", "mcp-mq-slack",
                  "Post a message to a channel.",
                  {"channel": {"type": "string"}, "text": {"type": "string"}},
                  ["channel", "text"]),
    ]


# ---------------------------------------------------------------------------
# Stemming
# ---------------------------------------------------------------------------


class TestStemming:
    def test_tokenize_stems_index_and_query_identically(self):
        from tools.tool_search import _tokenize
        # Same stem on both sides is the whole contract.
        assert _tokenize("issues") == _tokenize("issue")
        assert _tokenize("creating messages") == _tokenize("create message")

    def test_plural_query_finds_singular_tool_name(self, issue_defs):
        """The measured miss on the old tokenizer: 'issues' skipped create_issue."""
        from tools.tool_search import build_catalog, search_catalog

        catalog = build_catalog(issue_defs)
        names = [h.name for h in search_catalog(catalog, "issues", limit=5)]
        assert "mq_linear_create_issue" in names
        assert "mq_linear_list_issues" in names

    def test_substring_fallback_still_uses_raw_name(self, issue_defs):
        """Fallback matches the unstemmed tool name, unchanged by stemming."""
        from tools.tool_search import build_catalog, search_catalog

        catalog = build_catalog(issue_defs)
        names = [h.name for h in search_catalog(catalog, "post_mess", limit=5)]
        assert names == ["mq_slack_post_message"]


# ---------------------------------------------------------------------------
# Exact-name ranking and shared corpus statistics
# ---------------------------------------------------------------------------


class TestCatalogRanking:
    def test_exact_name_beats_shorter_siblings(self):
        from tools.tool_search import build_catalog, search_catalog

        exact = _td(
            "github_create_issue",
            "Create a new issue with a title, body, assignees, labels, "
            "milestone, project metadata, and linked context for a repository.",
        )
        catalog = build_catalog([
            exact,
            _td("github_create_issue_comment", "Comment."),
            _td("github_create_issue_label", "Label."),
        ])

        assert search_catalog(catalog, "github_create_issue", limit=1) == [catalog[0]]

    def test_exact_short_name_beats_prefixed_names(self):
        from tools.tool_search import build_catalog, search_catalog

        catalog = build_catalog([
            _td("list", "List one item."),
            _td("list_x", "List x."),
            _td("list_all_the_open_items", "List every open item."),
        ])

        assert search_catalog(catalog, "list", limit=1) == [catalog[0]]

    def test_precomputed_corpus_stats_preserve_results(self, issue_defs):
        from tools.tool_search import _corpus_stats, build_catalog, search_catalog

        catalog = build_catalog(issue_defs)
        expected = search_catalog(catalog, "create issues", limit=3)
        actual = search_catalog(
            catalog,
            "create issues",
            limit=3,
            corpus_stats=_corpus_stats(catalog),
        )

        assert actual == expected


# ---------------------------------------------------------------------------
# Multi-query dispatch_tool_search
# ---------------------------------------------------------------------------


class TestMultiQuerySearch:
    def test_grouped_names_plus_shared_tool_map(self, issue_defs):
        from tools.tool_search import ToolSearchConfig, dispatch_tool_search

        result = json.loads(dispatch_tool_search(
            {"queries": ["create linear issue", "post slack message"]},
            current_tool_defs=issue_defs,
            config=ToolSearchConfig.from_raw({}),
        ))

        assert result["queries"] == ["create linear issue", "post slack message"]
        assert result["total_available"] == 3
        # Groups carry NAMES only, in query order.
        assert [g["query"] for g in result["results"]] == result["queries"]
        for group in result["results"]:
            for name in group["matches"]:
                assert isinstance(name, str)
        assert "mq_linear_create_issue" in result["results"][0]["matches"]
        assert "mq_slack_post_message" in result["results"][1]["matches"]
        # The shared map holds each matched tool exactly once, and nothing else.
        matched = {n for g in result["results"] for n in g["matches"]}
        assert set(result["tools"]) == matched
        record = result["tools"]["mq_linear_create_issue"]
        assert record["source"] == "mcp"
        assert record["source_name"] == "mcp-mq-linear"
        assert record["description"].startswith("Create a new issue")
        assert record["required"] == ["title", "team"]
        # All queries matched → no fallback block.
        assert "available_sources" not in result
        assert "hint" not in result

    def test_limit_applies_per_query(self, issue_defs):
        from tools.tool_search import ToolSearchConfig, dispatch_tool_search

        result = json.loads(dispatch_tool_search(
            {"queries": ["issues", "message"], "limit": 1},
            current_tool_defs=issue_defs,
            config=ToolSearchConfig.from_raw({}),
        ))
        for group in result["results"]:
            assert len(group["matches"]) <= 1

    def test_partial_miss_adds_fallback_to_empty_group(self, issue_defs):
        from tools.tool_search import ToolSearchConfig, dispatch_tool_search

        result = json.loads(dispatch_tool_search(
            {"queries": ["issues", "zzzz nonsense qqqq"]},
            current_tool_defs=issue_defs,
            config=ToolSearchConfig.from_raw({}),
        ))
        assert result["results"][1]["matches"] == []
        assert "available_sources" not in result["results"][0]
        assert "hint" not in result["results"][0]
        missed = result["results"][1]
        assert "This query returned no lexical matches" in missed["hint"]
        source_names = {s["name"] for s in missed["available_sources"]}
        assert {"mq-linear", "mq-slack"} <= source_names
        assert "available_sources" not in result
        assert "hint" not in result

    def test_bare_string_query_coerced_to_single_query(self, issue_defs):
        from tools.tool_search import ToolSearchConfig, dispatch_tool_search

        result = json.loads(dispatch_tool_search(
            {"queries": "post slack message"},
            current_tool_defs=issue_defs,
            config=ToolSearchConfig.from_raw({}),
        ))
        assert result["queries"] == ["post slack message"]

    def test_max_queries_config_respected(self, issue_defs):
        from tools.tool_search import ToolSearchConfig, dispatch_tool_search

        cfg = ToolSearchConfig.from_raw({"max_queries": 2})
        ok = json.loads(dispatch_tool_search(
            {"queries": ["a b", "c d"]}, current_tool_defs=issue_defs, config=cfg))
        assert "error" not in ok
        over = json.loads(dispatch_tool_search(
            {"queries": ["a", "b", "c"]}, current_tool_defs=issue_defs, config=cfg))
        assert "too many queries" in over["error"]


# ---------------------------------------------------------------------------
# Batched dispatch_tool_describe
# ---------------------------------------------------------------------------


class TestBatchedDescribe:
    def test_map_response_with_not_found(self, issue_defs):
        from tools.tool_search import ToolSearchConfig, dispatch_tool_describe

        # Deferrable in the global registry, but NOT in this session's defs —
        # the stale/out-of-scope case that lands in not_found.
        _register("mq_out_of_scope_op", "mcp-mq-elsewhere")

        result = json.loads(dispatch_tool_describe(
            {"names": ["mq_linear_create_issue", "mq_slack_post_message",
                       "mq_out_of_scope_op", "mcp__bogus__missing"]},
            current_tool_defs=issue_defs,
            config=ToolSearchConfig.from_raw({}),
        ))
        assert set(result["tools"]) == {"mq_linear_create_issue",
                                        "mq_slack_post_message"}
        schema = result["tools"]["mq_linear_create_issue"]
        assert schema["description"] == "Create a new issue in a team."
        assert schema["parameters"]["required"] == ["title", "team"]
        # Deferrable-but-absent and unknown names collect in not_found; found
        # ones still resolve.
        assert result["not_found"] == ["mq_out_of_scope_op", "mcp__bogus__missing"]
        assert "tool_search" in result["hint"]
        assert "errors" not in result

    def test_real_schemas_and_unknown_name_are_classified_independently(self):
        from tools.tool_search import ToolSearchConfig, dispatch_tool_describe

        tool_defs = [
            _register("mcp__linear__get_issue", "mcp-linear"),
            _register("mcp__granola__list_meeting_folders", "mcp-granola"),
        ]
        result = json.loads(dispatch_tool_describe(
            {
                "names": [
                    "mcp__linear__get_issue",
                    "mcp__granola__list_meeting_folders",
                    "mcp__linear__does_not_exist_zzz",
                ]
            },
            current_tool_defs=tool_defs,
            config=ToolSearchConfig.from_raw({}),
        ))

        assert set(result["tools"]) == {
            "mcp__linear__get_issue",
            "mcp__granola__list_meeting_folders",
        }
        assert result["not_found"] == ["mcp__linear__does_not_exist_zzz"]
        assert "errors" not in result

    def test_unregistered_core_name_is_not_found(self, issue_defs):
        from tools.tool_search import ToolSearchConfig, dispatch_tool_describe

        result = json.loads(dispatch_tool_describe(
            {"names": ["terminal", "mq_linear_create_issue"]},
            current_tool_defs=issue_defs,
            config=ToolSearchConfig.from_raw({}),
        ))
        assert "mq_linear_create_issue" in result["tools"]
        assert "terminal" in result["not_found"]
        assert "errors" not in result

    def test_registered_direct_surface_name_keeps_exact_error(self):
        from tools.tool_search import ToolSearchConfig, dispatch_tool_describe

        name = "mq_desktop_direct_action"
        tool_def = _register(name, "desktop_ui")
        result = json.loads(dispatch_tool_describe(
            {"names": [name]},
            current_tool_defs=[tool_def],
            config=ToolSearchConfig.from_raw({}),
        ))

        assert result["errors"][name] == (
            f"'{name}' is not a deferrable tool. If you see it in the tools list "
            "already, call it directly; otherwise check the spelling against tool_search."
        )
        assert name not in result.get("not_found", [])

    def test_registry_lookup_failure_is_not_found(self, monkeypatch):
        from tools.registry import registry
        from tools.tool_search import ToolSearchConfig, dispatch_tool_describe

        def fail_lookup(name):
            raise RuntimeError("registry unavailable")

        monkeypatch.setattr(registry, "get_entry", fail_lookup)
        result = json.loads(dispatch_tool_describe(
            {"names": ["mq_unknown_during_lookup"]},
            current_tool_defs=[],
            config=ToolSearchConfig.from_raw({}),
        ))

        assert result["not_found"] == ["mq_unknown_during_lookup"]
        assert "errors" not in result

    def test_duplicates_deduped_silently(self, issue_defs):
        from tools.tool_search import ToolSearchConfig, dispatch_tool_describe

        result = json.loads(dispatch_tool_describe(
            {"names": ["mq_linear_create_issue", "mq_linear_create_issue"]},
            current_tool_defs=issue_defs,
            config=ToolSearchConfig.from_raw({}),
        ))
        assert list(result["tools"]) == ["mq_linear_create_issue"]
        assert "not_found" not in result

    def test_empty_and_overcap_names_error(self, issue_defs):
        from tools.tool_search import ToolSearchConfig, dispatch_tool_describe

        cfg = ToolSearchConfig.from_raw({})
        assert "error" in json.loads(dispatch_tool_describe(
            {}, current_tool_defs=issue_defs, config=cfg))
        assert "error" in json.loads(dispatch_tool_describe(
            {"names": []}, current_tool_defs=issue_defs, config=cfg))
        over = ["n%d" % i for i in range(cfg.max_describe_names + 1)]
        parsed = json.loads(dispatch_tool_describe(
            {"names": over}, current_tool_defs=issue_defs, config=cfg))
        assert "too many names" in parsed["error"]

    def test_bare_string_name_coerced(self, issue_defs):
        from tools.tool_search import ToolSearchConfig, dispatch_tool_describe

        result = json.loads(dispatch_tool_describe(
            {"names": "mq_linear_create_issue"},
            current_tool_defs=issue_defs,
            config=ToolSearchConfig.from_raw({}),
        ))
        assert "mq_linear_create_issue" in result["tools"]


# ---------------------------------------------------------------------------
# Config + bridge schema
# ---------------------------------------------------------------------------


class TestConfigAndSchema:
    def test_new_caps_default_and_parse(self):
        from tools.tool_search import ToolSearchConfig

        cfg = ToolSearchConfig.from_raw(None)
        assert cfg.max_queries >= 1
        assert cfg.max_describe_names >= 1
        # Operator-tunable with no upper clamp; floored at 1.
        big = ToolSearchConfig.from_raw({"max_queries": 500,
                                         "max_describe_names": 500})
        assert big.max_queries == 500
        assert big.max_describe_names == 500
        floored = ToolSearchConfig.from_raw({"max_queries": 0,
                                             "max_describe_names": -3})
        assert floored.max_queries == 1
        assert floored.max_describe_names == 1

    def test_limit_default_within_cap(self):
        from tools.tool_search import ToolSearchConfig

        cfg = ToolSearchConfig.from_raw({})
        assert 1 <= cfg.search_default_limit <= cfg.max_search_limit <= 50

    def test_bridge_schema_declares_array_inputs(self):
        from tools.tool_search import bridge_tool_schemas

        schemas = {s["function"]["name"]: s["function"] for s in bridge_tool_schemas(3)}
        search_params = schemas["tool_search"]["parameters"]
        assert search_params["required"] == ["queries"]
        assert search_params["properties"]["queries"]["type"] == "array"
        describe_params = schemas["tool_describe"]["parameters"]
        assert describe_params["required"] == ["names"]
        assert describe_params["properties"]["names"]["type"] == "array"
