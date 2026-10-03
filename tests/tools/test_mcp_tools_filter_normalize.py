"""Regression tests for issue #122373: bare string/list mcp_servers tools filter."""

from tools.mcp_schema_cache import config_fingerprint, normalize_tools_filter


def test_config_fingerprint_accepts_comma_string():
    fp = config_fingerprint({"command": "x", "tools": "search_x,query_docs"})
    assert isinstance(fp, str) and fp


def test_normalize_comma_string_is_include_list():
    assert normalize_tools_filter({"tools": "search_x,query_docs"}) == {
        "include": ["search_x", "query_docs"]}


def test_normalize_whitespace_string_is_include_list():
    assert normalize_tools_filter({"tools": "search_x query_docs"}) == {
        "include": ["search_x", "query_docs"]}


def test_normalize_list_is_include_list():
    assert normalize_tools_filter({"tools": ["a", "b"]}) == {"include": ["a", "b"]}


def test_normalize_dict_passthrough():
    d = {"include": ["a"], "exclude": ["b"]}
    assert normalize_tools_filter({"tools": d}) is d


def test_normalize_missing_or_empty():
    assert normalize_tools_filter({}) == {}
    assert normalize_tools_filter({"tools": ""}) == {}
    assert normalize_tools_filter({"tools": None}) == {}


def test_fingerprint_stable_across_string_and_list():
    a = config_fingerprint({"command": "x", "tools": "search_x,query_docs"})
    b = config_fingerprint({"command": "x", "tools": ["search_x", "query_docs"]})
    assert a == b
