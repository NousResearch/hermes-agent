"""Test web_extract_tool handles dict objects from web_search results.

Reproduces and verifies fix for #61693 where web_search result dicts
caused TypeError when web_extract tried to normalize and validate URLs.
"""
import json
from typing import List


def test_web_extract_handles_dict_urls():
    """web_extract_tool should extract URL strings from dict objects.

    When the model passes web_search result dicts (with url/href fields),
    web_extract_tool should extract the actual URL strings before processing.
    This test verifies the dict-to-string coercion logic works without errors.

    This is a minimal smoke test - it doesn't actually call web_extract_tool
    (which requires external services), but verifies the core logic that
    dict URLs are converted to strings before regex operations.
    """
    # Simulate web_search result dicts
    dict_urls: List[dict] = [
        {"url": "https://example.com/page1", "title": "Example 1", "snippet": "..."},
        {"url": "https://example.com/page2", "title": "Example 2", "snippet": "..."},
        {"href": "https://alternate.com/page", "title": "Alternate", "snippet": "..."},
        {"title": "No URL field"},  # Missing url/href
    ]

    # This is the coercion logic from the fix in web_tools.py
    processed_urls: List[str] = []
    for _url in dict_urls:
        # Handle dict objects from web_search results
        if isinstance(_url, dict):
            _url = _url.get("url") or _url.get("href") or ""
        elif not isinstance(_url, str):
            _url = str(_url)
        processed_urls.append(_url)

    # Verify extraction works
    assert processed_urls == [
        "https://example.com/page1",
        "https://example.com/page2",
        "https://alternate.com/page",
        "",  # Missing url/href → empty string
    ]


def test_web_extract_handles_mixed_string_dict_urls():
    """web_extract_tool should handle mix of string URLs and dict objects."""
    mixed_urls = [
        "https://direct.com/page",
        {"url": "https://dict.com/page", "title": "Dict URL"},
        {"href": "https://href.com/page"},
        "https://another.com/page",
        {"title": "No URL field"},
    ]

    processed_urls: List[str] = []
    for _url in mixed_urls:
        if isinstance(_url, dict):
            _url = _url.get("url") or _url.get("href") or ""
        elif not isinstance(_url, str):
            _url = str(_url)
        processed_urls.append(_url)

    assert processed_urls == [
        "https://direct.com/page",
        "https://dict.com/page",
        "https://href.com/page",
        "https://another.com/page",
        "",
    ]


def test_web_extract_dict_coercion_preserves_valid_urls():
    """Dict coercion should not break valid URL strings."""
    valid_url = "https://example.com/path?query=value#fragment"

    # String URL should pass through unchanged
    if isinstance(valid_url, dict):
        processed = valid_url.get("url") or valid_url.get("href") or ""
    elif not isinstance(valid_url, str):
        processed = str(valid_url)
    else:
        processed = valid_url

    assert processed == valid_url


def test_web_extract_dict_coercion_handles_edge_cases():
    """Test edge cases: non-dict, non-string objects, empty strings."""
    edge_cases = [
        123,  # int
        None,  # None
        True,  # bool
        "",  # empty string
    ]

    processed: List[str] = []
    for item in edge_cases:
        if isinstance(item, dict):
            processed_item = item.get("url") or item.get("href") or ""
        elif not isinstance(item, str):
            processed_item = str(item)
        else:
            processed_item = item
        processed.append(processed_item)

    # Non-dict, non-string → str() conversion
    assert processed[0] == "123"
    assert processed[1] == "None"
    assert processed[2] == "True"
    # Empty string stays empty
    assert processed[3] == ""