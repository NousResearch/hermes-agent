"""Hosted-connector discovery in ``dispatch_tool_search`` (#138601).

The gateway leg ranks relevance for the whole use_case; the local rarest-token
gate used to re-filter those entries as one merged corpus, so a domain word no
catalog document carries ("zones", "dns") became the gate token and zeroed the
merge — connected apps were invisible with no hint. The gateway's own shortlist
is now searched (and kept, in gateway order, when the lexical pass over it is
empty), and connected connectors appear in ``available_sources``.
"""

import json

from tools.connectors.gateway.bridge import ConnectorLeg
from tools.tool_search import dispatch_tool_search

_CF_SLUGS = ["CLOUDFLARE_MCP_SEARCH", "CLOUDFLARE_MCP_EXECUTE", "CLOUDFLARE_MCP_DOCS"]
_CF_SCHEMAS = {
    "CLOUDFLARE_MCP_SEARCH": {
        "connector": "cloudflare_mcp", "tool": "CLOUDFLARE_MCP_SEARCH",
        "description": "Search Cloudflare resources",
        "input_schema": {"type": "object", "properties": {}},
    },
    "CLOUDFLARE_MCP_EXECUTE": {
        "connector": "cloudflare_mcp", "tool": "CLOUDFLARE_MCP_EXECUTE",
        "description": "Execute JavaScript code that can read, create, update, "
                       "or delete resources through the Cloudflare API",
        "input_schema": {"type": "object", "properties": {}},
    },
    "CLOUDFLARE_MCP_DOCS": {
        "connector": "cloudflare_mcp", "tool": "CLOUDFLARE_MCP_DOCS",
        "description": "Read Cloudflare documentation",
        "input_schema": {"type": "object", "properties": {}},
    },
}


def _cloudflare_leg(queries):
    """Gateway answers every use_case with the connector's ranked shortlist."""
    return ConnectorLeg(payload={
        "results": [{"use_case": q["use_case"], "tools": list(_CF_SLUGS)} for q in queries],
        "schemas": _CF_SCHEMAS,
    })


def _selective_cloudflare_leg(queries):
    """Gateway only matches use_cases that name the app; other groups come back empty."""
    results = [
        {"use_case": q["use_case"], "tools": list(_CF_SLUGS) if "cloudflare" in q["use_case"] else []}
        for q in queries
    ]
    return ConnectorLeg(payload={"results": results, "schemas": _CF_SCHEMAS})


def _session_defs():
    """A session that manages connections and has one local deferrable tool."""
    return [
        {"type": "function", "function": {"name": "manage_connections", "parameters": {}}},
        {"type": "function", "function": {
            "name": "mcp__github__create_issue",
            "description": "Create a GitHub issue",
            "parameters": {"type": "object", "properties": {}, "required": ["title"]},
        }},
    ]


def test_catalog_absent_domain_words_do_not_hide_connected_connector():
    """The reported miss: 'cloudflare zones dns workers' returned no matches although the
    connector was connected — 'zones'/'dns'/'workers' appear in no document, so the
    rarest-token gate rejected the gateway's entries. Gateway order now surfaces them."""
    out = json.loads(dispatch_tool_search(
        {"queries": ["cloudflare zones dns workers"], "limit": 5},
        current_tool_defs=_session_defs(),
        connector_search=_cloudflare_leg,
    ))
    [group] = out["results"]
    assert group["matches"] == [
        "connectors__cloudflare_mcp__SEARCH",
        "connectors__cloudflare_mcp__EXECUTE",
        "connectors__cloudflare_mcp__DOCS",
    ]
    assert out["tools"]["connectors__cloudflare_mcp__EXECUTE"]["source_name"] == "cloudflare_mcp"


def test_vendor_word_query_still_matches_lexically():
    """A plain vendor-word query keeps matching through the BM25 pass, not only the
    gateway-order fallback."""
    out = json.loads(dispatch_tool_search(
        {"queries": ["cloudflare"]},
        current_tool_defs=_session_defs(),
        connector_search=_cloudflare_leg,
    ))
    assert set(out["results"][0]["matches"]) == {
        "connectors__cloudflare_mcp__SEARCH",
        "connectors__cloudflare_mcp__EXECUTE",
        "connectors__cloudflare_mcp__DOCS",
    }


def test_missed_query_lists_connected_connector_among_available_sources():
    """When a query group still finds nothing, the empty-group hint must name the
    connected connector — otherwise the model reports the capability absent."""
    out = json.loads(dispatch_tool_search(
        {"queries": ["totally unrelated xyz", "cloudflare zones"]},
        current_tool_defs=_session_defs(),
        connector_search=_selective_cloudflare_leg,
    ))
    missed, _hit = out["results"]
    assert missed["matches"] == []
    sources = {row["name"]: row["tool_count"] for row in missed["available_sources"]}
    assert sources["cloudflare_mcp"] == 3
    assert missed["hint"]
    assert _hit["matches"]


def test_source_name_is_normalized_lowercase_from_vendor_cased_schema():
    """Vendor-cased gateway schemas must not leak mixed-case source names into the
    tools map / available_sources summary; the composed name is lowercase already."""
    def cased_leg(queries):
        return ConnectorLeg(payload={
            "results": [{"use_case": q["use_case"], "tools": ["CUSTOM_X_READ"]} for q in queries],
            "schemas": {"CUSTOM_X_READ": {
                "connector": "CUSTOM_X", "tool": "CUSTOM_X_READ", "description": "d",
                "input_schema": {"type": "object", "properties": {}},
            }},
        })

    out = json.loads(dispatch_tool_search(
        {"queries": ["custom_x read"]},
        current_tool_defs=_session_defs(),
        connector_search=cased_leg,
    ))
    [group] = out["results"]
    assert group["matches"] == ["connectors__custom_x__READ"]
    assert out["tools"]["connectors__custom_x__READ"]["source_name"] == "custom_x"
