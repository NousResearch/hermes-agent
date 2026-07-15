import json
from unittest.mock import MagicMock, patch

import httpx

from tools.agentwiki_routing import AgentWikiDecision, AgentWikiResult, maybe_agentwiki_route_search
from tools.web_tools import web_search_tool


def _web_config(**overrides):
    cfg = {
        "agentwikis": {
            "enabled": True,
            "shadow_mode": False,
            "registry": {
                "index_json_url": "https://agentwikis.test/index.json",
                "llms_txt_url": "https://agentwikis.test/llms.txt",
                "refresh_ttl_seconds": 0,
                "conditional_get": True,
            },
            "retrieval": {
                "timeout_ms": 2000,
                "max_pages_per_query": 6,
                "retry_html_accept_markdown": True,
            },
            "routing": {
                "min_select_score": 70,
                "abstain_on_tie": True,
                "freshness_trigger_terms": ["latest", "current", "recent", "today", "this week", "new"],
            },
            "domains": {
                "hermes": {"wiki_slug": "hermes", "aliases": ["hermes", "hermes-agent"]},
                "github": {"wiki_slug": "github", "aliases": ["github", "github actions", "pull request"]},
                "vercel": {"wiki_slug": "vercel", "aliases": ["vercel", "next.js deploy", "vercel.json"]},
                "claude_code": {"wiki_slug": "claude-code", "aliases": ["claude code", "claude hooks", "claude mcp"]},
            },
        }
    }
    agentwikis = cfg["agentwikis"]
    for key, value in overrides.items():
        agentwikis[key] = value
    return cfg


def _index_payload():
    return {
        "name": "Agent Wikis",
        "wikis": [
            {
                "slug": "hermes",
                "title": "Hermes",
                "description": "Hermes docs",
                "tags": ["agents", "configuration", "workflows"],
                "scope": {
                    "covers": "Hermes Agent setup, configuration, CLI commands, features, skills, providers, deployment, integrations, and troubleshooting.",
                    "notCovered": "Real-time community discussion, releases after the date below, and third-party pricing — use web search for those.",
                    "currentAs": "2026-07-01 (v0.18.0)",
                },
                "raw_base": "/raw/hermes/",
                "html_base": "/wiki/hermes/",
            },
            {
                "slug": "github",
                "title": "GitHub",
                "description": "GitHub docs",
                "tags": ["github", "actions"],
                "scope": {
                    "covers": "GitHub repos, pull requests, Actions, CodeQL, Dependabot, and collaboration workflows.",
                    "notCovered": "Enterprise administration, billing, and pricing are out of scope.",
                    "currentAs": "2026-06-19",
                },
                "raw_base": "/raw/github/",
                "html_base": "/wiki/github/",
            },
            {
                "slug": "vercel",
                "title": "Vercel",
                "description": "Vercel docs",
                "tags": ["vercel", "deployments"],
                "scope": {
                    "covers": "Vercel deployments, preview deployments, routing, build output, serverless and edge runtime basics.",
                    "notCovered": "Pricing/plans and AI SDK / AI Gateway specifics are out of scope.",
                    "currentAs": "2026-06-19",
                },
                "raw_base": "/raw/vercel/",
                "html_base": "/wiki/vercel/",
            },
            {
                "slug": "claude-code",
                "title": "Claude Code",
                "description": "Claude Code docs",
                "tags": ["claude-code", "coding-agent"],
                "scope": {
                    "covers": "Claude Code installation, configuration, permissions, hooks, MCP, subagents, and troubleshooting.",
                    "notCovered": "Non-Claude-Code API/platform pricing and general Anthropic API questions are out of scope.",
                    "currentAs": "2026-07-01 (2.1.198)",
                },
                "raw_base": "/raw/claude-code/",
                "html_base": "/wiki/claude-code/",
            },
        ],
    }


def _client(routes):
    def handler(request):
        url = str(request.url)
        if url not in routes:
            raise AssertionError(f"Unexpected URL {url}")
        spec = routes[url]
        status = spec.get("status", 200)
        headers = spec.get("headers", {})
        body = spec.get("body", "")
        if isinstance(body, (dict, list)):
            return httpx.Response(status, headers=headers, json=body, request=request)
        return httpx.Response(status, headers=headers, text=body, request=request)

    return httpx.Client(transport=httpx.MockTransport(handler))


def test_agentwiki_route_returns_raw_markdown_backed_results():
    client = _client(
        {
            "https://agentwikis.test/index.json": {
                "body": _index_payload(),
                "headers": {"etag": '"v1"', "last-modified": "Tue, 01 Jul 2026 00:00:00 GMT"},
            },
            "https://agentwikis.test/wiki/hermes/llms.txt": {
                "body": "# Hermes\n- [Hermes Agent Knowledge Base](/raw/hermes/README.md)\n- [Cron & Scheduling](/raw/hermes/wiki/concepts/cron-scheduling.md)\n"
            },
            "https://agentwikis.test/raw/hermes/README.md": {
                "body": "# Hermes Agent Knowledge Base\nHermes Agent docs root."
            },
            "https://agentwikis.test/raw/hermes/wiki/concepts/cron-scheduling.md": {
                "body": "---\ntitle: \"Cron & Scheduling\"\nupdated: 2026-07-01\n---\n\n# Cron & Scheduling\nHermes cron jobs run on a durable scheduler and support recurring prompts."
            },
            "https://agentwikis.test/raw/hermes/wiki/index.md": {
                "body": "# Hermes Index\nCron and scheduling pages are listed here."
            },
        }
    )

    result = maybe_agentwiki_route_search(
        "How do I configure Hermes cron jobs?",
        _web_config(),
        limit=2,
        client=client,
    )

    assert result.response_data is not None
    assert result.decision.selected_source == "agentwikis"
    assert result.decision.reason == "success"
    urls = [item["url"] for item in result.response_data["data"]["web"]]
    assert "https://agentwikis.test/raw/hermes/wiki/concepts/cron-scheduling.md" in urls
    assert result.response_data["source_routing"]["selected_wiki_slug"] == "hermes"


def test_agentwiki_route_rejects_out_of_scope_examples():
    client = _client({"https://agentwikis.test/index.json": {"body": _index_payload()}})

    github = maybe_agentwiki_route_search(
        "How do I manage GitHub Enterprise billing?",
        _web_config(),
        client=client,
    )
    assert github.response_data is None
    assert github.decision.reason == "out_of_scope"
    assert github.decision.selected_wiki_slug == "github"

    vercel = maybe_agentwiki_route_search(
        "How do I use the Vercel AI SDK?",
        _web_config(),
        client=client,
    )
    assert vercel.response_data is None
    assert vercel.decision.reason == "out_of_scope"
    assert vercel.decision.selected_wiki_slug == "vercel"

    claude = maybe_agentwiki_route_search(
        "What does the Claude API cost?",
        _web_config(),
        client=client,
    )
    assert claude.response_data is None
    assert claude.decision.reason == "out_of_scope"
    assert claude.decision.selected_wiki_slug == "claude-code"


def test_agentwiki_route_rejects_stale_or_time_sensitive_queries():
    client = _client({"https://agentwikis.test/index.json": {"body": _index_payload()}})

    later_date = maybe_agentwiki_route_search(
        "What changed in Hermes after 2026-07-10?",
        _web_config(),
        client=client,
    )
    assert later_date.response_data is None
    assert later_date.decision.reason == "stale_for_query"

    latest = maybe_agentwiki_route_search(
        "What is the latest Claude Code release?",
        _web_config(),
        client=client,
    )
    assert latest.response_data is None
    assert latest.decision.reason == "stale_for_query"

    pricing = maybe_agentwiki_route_search(
        "What is Vercel pricing?",
        _web_config(),
        client=client,
    )
    assert pricing.response_data is None
    assert pricing.decision.reason == "out_of_scope"


def test_agentwiki_registry_falls_back_to_llms_txt_when_index_json_unavailable():
    client = _client(
        {
            "https://agentwikis.test/index.json": {"status": 503, "body": "down"},
            "https://agentwikis.test/llms.txt": {
                "body": "# Agent Wikis\n\n## Hermes\n> Covers: Hermes Agent setup and configuration.\n> Not covered: pricing.\n> Current as of: 2026-07-01\n- [Hermes Agent Knowledge Base](/raw/hermes/README.md)\n"
            },
            "https://agentwikis.test/wiki/hermes/llms.txt": {
                "body": "# Hermes\n- [Hermes Agent Knowledge Base](/raw/hermes/README.md)\n"
            },
            "https://agentwikis.test/raw/hermes/README.md": {
                "body": "# Hermes Agent Knowledge Base\nHermes setup docs."
            },
            "https://agentwikis.test/raw/hermes/wiki/index.md": {
                "body": "# Hermes Index\nHermes setup docs index."
            },
        }
    )

    result = maybe_agentwiki_route_search("Hermes setup", _web_config(), client=client)

    assert result.response_data is not None
    assert result.decision.selected_source == "agentwikis"


def test_web_search_tool_uses_agentwiki_result_before_provider_search(monkeypatch):
    agentwiki_payload = {
        "success": True,
        "data": {
            "web": [
                {
                    "title": "Cron & Scheduling",
                    "url": "https://agentwikis.test/raw/hermes/wiki/concepts/cron-scheduling.md",
                    "description": "Agentwiki (hermes): cron docs",
                    "position": 1,
                }
            ]
        },
        "source_routing": {"selected_source": "agentwikis"},
    }
    monkeypatch.setattr(
        "tools.web_tools.maybe_agentwiki_route_search",
        lambda query, web_config, limit=5: AgentWikiResult(
            response_data=agentwiki_payload,
            decision=AgentWikiDecision(selected_source="agentwikis", reason="success", selected_wiki_slug="hermes"),
        ),
    )
    monkeypatch.setattr("tools.web_tools._load_web_config", lambda: _web_config())
    monkeypatch.setattr("tools.interrupt.is_interrupted", lambda: False)

    with patch.object(__import__("tools.web_tools", fromlist=["_debug"])._debug, "log_call"), patch.object(
        __import__("tools.web_tools", fromlist=["_debug"])._debug, "save"
    ), patch("tools.web_tools._ensure_web_plugins_loaded") as ensure_loaded:
        payload = json.loads(web_search_tool("Hermes cron jobs", limit=1))

    ensure_loaded.assert_not_called()
    assert payload["source_routing"]["selected_source"] == "agentwikis"
    assert payload["data"]["web"][0]["url"].endswith("cron-scheduling.md")


def test_web_search_tool_falls_back_to_provider_when_agentwiki_abstains(monkeypatch):
    monkeypatch.setattr(
        "tools.web_tools.maybe_agentwiki_route_search",
        lambda query, web_config, limit=5: AgentWikiResult(
            response_data=None,
            decision=AgentWikiDecision(selected_source="web_search", reason="out_of_scope", selected_wiki_slug="github", fallback_occurred=True),
        ),
    )
    monkeypatch.setattr("tools.web_tools._load_web_config", lambda: _web_config())
    monkeypatch.setattr("tools.interrupt.is_interrupted", lambda: False)

    provider = MagicMock()
    provider.supports_search.return_value = True
    provider.name = "fake-search"
    provider.search.return_value = {
        "success": True,
        "data": {"web": [{"title": "live", "url": "https://example.com", "description": "live", "position": 1}]},
    }

    with patch("tools.web_tools._ensure_web_plugins_loaded"), patch(
        "tools.web_tools._get_search_backend", return_value="fake-search"
    ), patch(
        "agent.web_search_registry.get_provider", return_value=provider
    ), patch(
        "agent.web_search_registry.get_active_search_provider", return_value=provider
    ), patch(
        "agent.web_search_registry._disabled_web_plugin_for", return_value=None
    ), patch.object(__import__("tools.web_tools", fromlist=["_debug"])._debug, "log_call"), patch.object(
        __import__("tools.web_tools", fromlist=["_debug"])._debug, "save"
    ):
        payload = json.loads(web_search_tool("GitHub Enterprise billing", limit=1))

    provider.search.assert_called_once()
    assert payload["source_routing"]["selected_source"] == "web_search"
    assert payload["source_routing"]["selection_reason"] == "out_of_scope"
