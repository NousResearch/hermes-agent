"""Tests for web_extract truncate-store robustness (findings from #54843 review).

Covers two robustness gaps left unaddressed when #54843 merged:
  1. _store_full_text bounded by MAX_STORED_TEXT_CHARS (no unbounded disk write).
  2. _truncate_with_footer emits a CONCRETE read_file offset for the omitted
     middle (was a literal `offset=<line>` placeholder the model had to guess).
"""
from __future__ import annotations

import re

import tools.web_tools as wt


def test_store_full_text_is_bounded(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    # Force the cache dir under the temp home.
    from hermes_constants import get_hermes_dir  # noqa: F401
    huge = "x\n" * (wt.MAX_STORED_TEXT_CHARS)  # > MAX_STORED_TEXT_CHARS chars
    assert len(huge) > wt.MAX_STORED_TEXT_CHARS
    path = wt._store_full_text("https://example.com/big", huge)
    assert path is not None
    stored = open(path, encoding="utf-8").read()
    # Stored copy capped (+ short marker), not the full unbounded blob.
    assert len(stored) <= wt.MAX_STORED_TEXT_CHARS + 200
    assert "stored copy truncated" in stored


def test_truncate_footer_gives_concrete_offset(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    # Build content well over the limit with many lines so head has a known count.
    content = "\n".join(f"line {i}" for i in range(5000))
    model_text, truncated = wt._truncate_with_footer(
        content, "https://example.com/page", char_limit=4000
    )
    assert truncated
    # Footer must contain a real integer offset, NOT the <line> placeholder.
    assert "offset=<line>" not in model_text
    m = re.search(r"offset=(\d+) limit=\d+", model_text)
    assert m, f"no concrete offset in footer: {model_text[-400:]}"
    offset = int(m.group(1))
    # Offset should point past the head we showed (head is ~75% of 4000 chars).
    assert offset > 1


def test_small_page_not_truncated_no_footer(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    content = "short page\nwith a few lines\n"
    model_text, truncated = wt._truncate_with_footer(
        content, "https://example.com/s", char_limit=15000
    )
    assert not truncated
    assert model_text == content
    assert "[TRUNCATED]" not in model_text
