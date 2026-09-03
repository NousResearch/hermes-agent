"""Tests for the name-coverage bonus in tools/tool_search.py.

BM25 on a flat REST-style catalog rewards token co-occurrence in long tool
names: ``"accounts"`` ranks ``put_accounts_by_account_id`` above the 2-token
``get_accounts``, and ``"list accounts"`` never surfaces ``get_accounts``
at all. The bonus multiplies BM25 by ``1 + w * F1(query tokens, name
tokens)`` so a short name fully explained by the query outranks a long
name that merely contains those tokens.

Invariants pinned here: the bonus never turns a zero score positive (empty
results and the substring fallback are unchanged), exact-name pins still
win, and it composes with the multi-query dispatcher.
"""

from __future__ import annotations

import os
import sys
from typing import Any, Dict

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
            "parameters": {"type": "object", "properties": properties or {}},
        },
    }


# A slice of the real Cloudflare MCP catalog (names + descriptions verbatim
# in spirit) that reproduces the measured failure with plain BM25.
_CF_SLICE = [
    _td("get_accounts", "GET /accounts List Accounts. List all accounts you have ownership or verified access to."),
    _td("post_accounts", "POST /accounts Create Account. Create an account (only available for tenant admins)."),
    _td("get_accounts_by_account_id", "GET /accounts/{account_id} Account Details. Get information about a specific account."),
    _td("put_accounts_by_account_id", "PUT /accounts/{account_id} Update Account. Update an existing account."),
    _td("get_accounts_builds_account_limits", "GET /accounts/{account_id}/builds/account/limits Get account limits. Get the account limits."),
    _td("get_accounts_workers_accountsettings", "GET /accounts/{account_id}/workers/account-settings Fetch Account Settings."),
    _td("get_accounts_rules_lists", "GET /accounts/{account_id}/rules/lists Get lists. Fetches all lists in the account."),
    _td("get_accounts_rules_lists_by_list_id", "GET /accounts/{account_id}/rules/lists/{list_id} Get a list. Fetches the details of a list."),
    _td("get_accounts_rules_lists_items", "GET /accounts/{account_id}/rules/lists/{list_id}/items Get list items. Fetches all the items in the list."),
    _td("get_accounts_rules_lists_items_by_item_id", "GET /accounts/{account_id}/rules/lists/{list_id}/items/{item_id} Get a list item."),
    _td("get_zones", "GET /zones List Zones. Lists, searches, sorts, and filters your zones."),
    _td("get_zones_environments", "GET /zones/{zone_id}/environments List environments. Lists all environments for a zone."),
    _td("delete_zones_by_zone_id", "DELETE /zones/{zone_id} Delete Zone. Deletes an existing zone."),
    _td("get_zones_dns_records", "GET /zones/{zone_id}/dns_records List DNS Records. List, search, sort, and filter a zone's DNS records."),
    _td("delete_zones_dns_records_by_dns_record_id", "DELETE /zones/{zone_id}/dns_records/{dns_record_id} Delete DNS Record."),
]


@pytest.fixture
def cf_catalog():
    from tools.tool_search import build_catalog
    return build_catalog(_CF_SLICE)


def _names(hits):
    return [h.name for h in hits]


class TestNameCoverageFactor:
    def test_no_overlap_is_neutral(self):
        from tools.tool_search import _name_coverage_factor
        assert _name_coverage_factor({"zone"}, frozenset({"get", "account"})) == 1.0
        assert _name_coverage_factor(set(), frozenset({"get"})) == 1.0
        assert _name_coverage_factor({"get"}, frozenset()) == 1.0

    def test_full_cover_beats_partial_cover(self):
        from tools.tool_search import _name_coverage_factor
        q = {"get", "account"}
        short = _name_coverage_factor(q, frozenset({"get", "account"}))
        long = _name_coverage_factor(q, frozenset({"get", "account", "rule", "list", "by", "id"}))
        assert short > long > 1.0

    def test_factor_is_bounded(self):
        from tools import tool_search
        from tools.tool_search import _name_coverage_factor
        f = _name_coverage_factor({"a", "b"}, frozenset({"a", "b"}))
        assert f == pytest.approx(1.0 + tool_search._NAME_COVERAGE_WEIGHT)


class TestShortNameRanking:
    def test_bare_noun_surfaces_shortest_matching_names(self, cf_catalog):
        """'accounts' -> the two 2-token names first; the verb is a tie."""
        from tools.tool_search import search_catalog
        assert set(_names(search_catalog(cf_catalog, "accounts", limit=2))) == \
            {"get_accounts", "post_accounts"}
        assert _names(search_catalog(cf_catalog, "zones", limit=1)) == ["get_zones"]

    def test_list_accounts_reaches_top5(self, cf_catalog):
        """The measured failure: BM25 alone returns only rules_lists_* names."""
        from tools.tool_search import search_catalog
        top5 = _names(search_catalog(cf_catalog, "list accounts", limit=5))
        assert "get_accounts" in top5

    def test_verb_noun_query_prefers_matching_short_name(self, cf_catalog):
        from tools.tool_search import search_catalog
        assert _names(search_catalog(cf_catalog, "get accounts", limit=1)) == ["get_accounts"]
        assert _names(search_catalog(cf_catalog, "dns records", limit=1)) == ["get_zones_dns_records"]

    def test_specific_query_still_finds_specific_tool(self, cf_catalog):
        """A query naming the long tool's distinguishing tokens still wins."""
        from tools.tool_search import search_catalog
        assert _names(search_catalog(cf_catalog, "get list items", limit=1)) == \
            ["get_accounts_rules_lists_items"]
        assert _names(search_catalog(cf_catalog, "delete dns record", limit=1)) == \
            ["delete_zones_dns_records_by_dns_record_id"]

    def test_bonus_disabled_reproduces_plain_bm25(self, cf_catalog, monkeypatch):
        from tools import tool_search
        from tools.tool_search import search_catalog
        with_bonus = _names(search_catalog(cf_catalog, "accounts", limit=2))
        monkeypatch.setattr(tool_search, "_NAME_COVERAGE_WEIGHT", 0.0)
        without = _names(search_catalog(cf_catalog, "accounts", limit=2))
        assert "get_accounts" in with_bonus
        assert "get_accounts" not in without  # the pre-bonus behavior


class TestSemanticsPreserved:
    def test_zero_score_stays_zero_and_fallback_unchanged(self, cf_catalog):
        from tools.tool_search import search_catalog
        assert search_catalog(cf_catalog, "zzzz", limit=5) == []
        # "zon" is a substring of get_zones but never a token: substring
        # fallback path, which the bonus must not touch.
        names = _names(search_catalog(cf_catalog, "zon", limit=50))
        assert names and all("zon" in n for n in names)

    def test_exact_name_still_pinned_first(self, cf_catalog):
        from tools.tool_search import search_catalog
        hits = search_catalog(cf_catalog, "get_accounts_rules_lists_by_list_id", limit=1)
        assert _names(hits) == ["get_accounts_rules_lists_by_list_id"]

    def test_mcp_prefix_is_not_a_name_token(self):
        from tools.tool_search import build_catalog
        [entry] = build_catalog([_td("mcp__cloudflare__get_accounts", "List accounts.")])
        assert "mcp" not in entry._name_tokens
        assert {"get", "account"} <= set(entry._name_tokens)

    def test_dispatch_multi_query_uses_bonus(self):
        from tools.registry import registry
        from tools.tool_search import ToolSearchConfig, dispatch_tool_search
        import json

        for td in _CF_SLICE:
            registry.register(
                name=td["function"]["name"],
                handler=lambda args, **kwargs: "{}",
                schema=td,
                toolset="mcp-namebonus-test",
            )
        out = json.loads(dispatch_tool_search(
            {"queries": ["get accounts", "zones"], "limit": 1},
            current_tool_defs=_CF_SLICE,
            config=ToolSearchConfig.from_raw({"enabled": "on"}),
        ))
        assert out["results"][0]["matches"] == ["get_accounts"]
        assert out["results"][1]["matches"] == ["get_zones"]
