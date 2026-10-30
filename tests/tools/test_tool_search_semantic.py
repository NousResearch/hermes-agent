"""Tests for semantic expansion, multilingual (CJK) tokenization,
soft-gate admission, and compact parameter signatures in tool_search.
"""

from __future__ import annotations

import json
import pytest

from tools.tool_search import (
    CatalogEntry,
    _shared_tool_record,
    _compact_parameters,
    build_catalog,
    search_catalog,
)
from tools.tool_search_catalog import (
    _is_cjk,
    _cjk_ngrams,
    _tokenize,
    _expand_semantic_tokens,
    SEMANTIC_ACTION_SYNONYMS,
)


def _td(name: str, description: str = "", properties: dict | None = None, required: list[str] | None = None) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties or {},
                "required": required or [],
            },
        },
    }


class TestCJKTokenization:
    def test_is_cjk_detection(self):
        assert _is_cjk("中")
        assert _is_cjk("文")
        assert not _is_cjk("a")
        assert not _is_cjk("1")
        assert not _is_cjk("@")

    def test_cjk_ngrams_extraction(self):
        text = "查询工单"
        ngrams = _cjk_ngrams(text)
        # Should include unigrams
        assert "查" in ngrams
        assert "询" in ngrams
        assert "工" in ngrams
        assert "单" in ngrams
        # Should include recognized semantic action terms
        assert "查询" in ngrams

    def test_mixed_cjk_and_ascii_tokenize(self):
        text = "搜索 GitHub 仓库里的 issues"
        tokens = _tokenize(text)
        # English tokens (stemmed)
        assert "github" in tokens
        assert "issu" in tokens
        # CJK tokens
        assert "搜" in tokens
        assert "索" in tokens
        assert "搜索" in tokens
        assert "仓" in tokens
        assert "库" in tokens

    def test_empty_or_symbol_only_tokenize(self):
        assert _tokenize("") == []
        assert _tokenize("!@#$%^&*()") == []


class TestSemanticSynonyms:
    def test_semantic_synonym_clusters_exist(self):
        assert "fetch" in SEMANTIC_ACTION_SYNONYMS
        assert "scrape" in SEMANTIC_ACTION_SYNONYMS["fetch"]
        assert "extract" in SEMANTIC_ACTION_SYNONYMS["fetch"]

        assert "queri" in SEMANTIC_ACTION_SYNONYMS
        assert "search" in SEMANTIC_ACTION_SYNONYMS["queri"]

    def test_expand_semantic_tokens(self):
        tokens = ["fetch", "page"]
        expanded = _expand_semantic_tokens(tokens)
        assert "fetch" in expanded
        assert "page" in expanded
        assert "extract" in expanded
        assert "scrape" in expanded

    def test_synonym_recall_for_tools(self):
        defs = [
            _td("web_extract", "Extract and scrape markdown content from a webpage URL",
                {"url": {"type": "string"}}, required=["url"]),
            _td("slack_send_message", "Post a message into a Slack channel",
                {"channel": {"type": "string"}, "text": {"type": "string"}}, required=["channel"]),
        ]
        catalog = build_catalog(defs)

        # Query using 'fetch' which is a synonym of 'extract'/'scrape'
        hits = search_catalog(catalog, "fetch webpage", limit=5)
        assert len(hits) > 0
        assert hits[0].name == "web_extract"

    def test_synonym_recall_for_query_verb(self):
        defs = [
            _td("github_list_issues", "Enumerate and display repository issues",
                {"repo": {"type": "string"}}, required=["repo"]),
            _td("linear_create_issue", "Open a new issue in Linear tracker",
                {"title": {"type": "string"}}, required=["title"]),
        ]
        catalog = build_catalog(defs)

        # Query using 'list' or 'query' should match github_list_issues
        hits = search_catalog(catalog, "list issues", limit=3)
        assert any(h.name == "github_list_issues" for h in hits)


class TestMultilingualSearch:
    def test_cjk_search_query_matches_chinese_tool(self):
        defs = [
            _td("order_query_tool", "查询电商平台用户订单与物流信息",
                {"order_id": {"type": "string"}}, required=["order_id"]),
            _td("email_sender", "发送通知邮件",
                {"to": {"type": "string"}}, required=["to"]),
        ]
        catalog = build_catalog(defs)

        hits = search_catalog(catalog, "查询订单", limit=3)
        assert len(hits) > 0
        assert hits[0].name == "order_query_tool"

    def test_mixed_cjk_query_matches_english_tool_by_entity(self):
        defs = [
            _td("github_issues", "Inspect and search GitHub issue tracker",
                {"repo": {"type": "string"}}, required=["repo"]),
        ]
        catalog = build_catalog(defs)

        hits = search_catalog(catalog, "搜索 GitHub", limit=3)
        assert len(hits) > 0
        assert hits[0].name == "github_issues"


class TestCompactParameters:
    def test_compact_parameters_normal(self):
        schema = {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "Target webpage URL"},
                "timeout": {"type": "integer", "description": "Request timeout in seconds"},
                "headers": {"type": "object"},
            },
            "required": ["url"],
        }
        compact = _compact_parameters(schema)
        assert compact["url"] == "string (required)"
        assert compact["timeout"] == "integer"
        assert compact["headers"] == "object"

    def test_compact_parameters_empty_or_malformed(self):
        assert _compact_parameters({}) == {}
        assert _compact_parameters(None) == {}
        assert _compact_parameters({"properties": "not a dict"}) == {}

    def test_shared_tool_record_includes_compact_parameters(self):
        entry = CatalogEntry(
            name="demo_tool",
            description="A demonstration tool",
            schema=_td("demo_tool", "A demonstration tool",
                       {"path": {"type": "string"}, "force": {"type": "boolean"}},
                       required=["path"]),
            source="mcp",
            source_name="mcp-demo",
        )
        record = _shared_tool_record(entry)
        assert "compact_parameters" in record
        assert record["compact_parameters"]["path"] == "string (required)"
        assert record["compact_parameters"]["force"] == "boolean"
        assert record["required"] == ["path"]
