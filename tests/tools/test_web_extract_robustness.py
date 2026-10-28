"""Tests for web_extract truncate-store robustness (findings from #54843 review).

Covers two robustness gaps left unaddressed when #54843 merged:
  1. _store_full_text bounded by MAX_STORED_TEXT_CHARS (no unbounded disk write).
  2. _truncate_with_footer emits a CONCRETE read_file offset for the omitted
     middle (was a literal `offset=<line>` placeholder the model had to guess).
"""
from __future__ import annotations

import re

import tools.web_tools as wt
from tools import web_tools_truncate


def test_store_full_text_is_bounded(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    # Force the cache dir under the temp home.
    from hermes_constants import get_hermes_dir
    huge = "x\n" * (web_tools_truncate.MAX_STORED_TEXT_CHARS)  # > MAX_STORED_TEXT_CHARS chars
    assert len(huge) > web_tools_truncate.MAX_STORED_TEXT_CHARS
    path = web_tools_truncate._store_full_text("https://example.com/big", huge)
    assert path is not None
    stored = open(path, encoding="utf-8").read()
    # Stored copy capped (+ short marker), not the full unbounded blob.
    assert len(stored) <= web_tools_truncate.MAX_STORED_TEXT_CHARS + 200
    assert "stored copy truncated" in stored


def test_small_page_not_truncated_no_footer(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    content = "short page\nwith a few lines\n"
    model_text, truncated = web_tools_truncate._truncate_with_footer(
        content, "https://example.com/s", char_limit=15000
    )
    assert not truncated
    assert model_text == content
    assert "[TRUNCATED]" not in model_text


def test_policy_blocked_url_is_refused_not_fetched(tmp_path, monkeypatch):
    """#127696: the per-URL website policy is a gate on the fetch path.

    A policy-blocked URL must come back as a per-URL policy error and must never
    enter the vendor batch — mixed with an allowed URL, only the allowed one is
    dispatched.
    """
    import asyncio
    from unittest.mock import AsyncMock, patch

    from tools.web_tools_extract import _extract_safe_urls

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text(
        "security:\n"
        "  website_blocklist:\n"
        "    enabled: true\n"
        "    domains:\n"
        "      - blocked.example\n"
    )

    class _Provider:
        name = "test"

    allowed_page = {"url": "https://ok.example/y", "title": "ok", "content": "page", "error": None}
    with patch("tools.web_tools_extract._dispatch_extract", new_callable=AsyncMock) as dispatch:
        dispatch.return_value = [allowed_page]
        results = asyncio.run(
            _extract_safe_urls(_Provider(), ["https://blocked.example/x", "https://ok.example/y"], None)
        )
    assert results[0]["error"] == (
        "Blocked by website policy: 'blocked.example' matched rule 'blocked.example' from config"
    )
    assert results[0]["content"] == ""
    assert results[1] == allowed_page
    # Fail-closed: the blocked URL never reached the vendor batch.
    assert dispatch.await_count == 1
    assert "https://blocked.example/x" not in dispatch.call_args.args[1]


def test_policy_blocked_url_alone_never_dispatches_or_reads_cache(tmp_path, monkeypatch):
    """#127696: a fully-blocked batch makes no vendor call and no cache read."""
    import asyncio
    from unittest.mock import patch

    from tools.web_tools_extract import _extract_safe_urls

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text(
        "security:\n"
        "  website_blocklist:\n"
        "    enabled: true\n"
        "    domains:\n"
        "      - blocked.example\n"
    )

    class _Provider:
        name = "test"

    with (
        patch("tools.web_tools_extract._dispatch_extract") as dispatch,
        patch("tools.web_result_cache.extract_cache_get") as cache_get,
    ):
        results = asyncio.run(_extract_safe_urls(_Provider(), ["https://blocked.example/x"], None))
    assert results[0]["error"].startswith("Blocked by website policy")
    assert dispatch.await_count == 0
    assert cache_get.call_count == 0
