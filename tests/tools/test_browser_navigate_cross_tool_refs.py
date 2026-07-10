"""Regression tests for the browser_navigate cross-tool-reference stripping
in model_tools.py's get_tool_definitions().

The static browser_navigate schema (tools/browser_tool.py) hardcodes two
sentences that name other tools by name: "prefer web_search or web_extract"
and "prefer curl via the terminal tool or web_extract". AGENTS.md's Known
Pitfalls section forbids shipping schema descriptions that mention tools
from other toolsets by name, because a model reading the description will
attempt to call them even when they are unavailable (missing API key,
disabled toolset), causing hallucinated tool calls.

model_tools.py post-processes the description to strip references to
whichever of {web_search, web_extract, terminal} are not actually available
this session. This file locks in that behavior for every availability
combination so a future edit to either sentence can't silently reintroduce
a dangling cross-reference (as happened in the abandoned PR #43352, which
fixed only the first sentence and was closed for "prefer web_search or
web_extract" still remaining).
"""

from __future__ import annotations

import pytest

import model_tools
from tools.registry import invalidate_check_fn_cache, registry

_BROWSER_TOOL_NAMES = [
    "browser_navigate", "browser_snapshot", "browser_click",
    "browser_type", "browser_scroll", "browser_back",
    "browser_press", "browser_get_images", "browser_vision",
    "browser_console", "browser_cdp", "browser_dialog",
]
_OTHER_TOOL_NAMES = ["web_search", "web_extract", "terminal", "process"]


def _force_available(monkeypatch, names):
    """Force check_fn() -> True for every registered tool in *names* that
    exists, so schema availability in this test doesn't depend on whether
    the local environment has API keys / a browser backend installed."""
    for name in names:
        entry = registry.get_entry(name)
        if entry is not None:
            monkeypatch.setattr(entry, "check_fn", lambda: True)


def _browser_navigate_description(enabled_toolsets, disabled_toolsets=None):
    model_tools._tool_defs_cache.clear()
    defs = model_tools.get_tool_definitions(
        enabled_toolsets=enabled_toolsets,
        disabled_toolsets=disabled_toolsets,
        quiet_mode=True,
        skip_tool_search_assembly=True,
    )
    for tool_def in defs:
        fn = tool_def.get("function", {})
        if fn.get("name") == "browser_navigate":
            return fn.get("description", "")
    raise AssertionError("browser_navigate not present in tool definitions")


@pytest.fixture(autouse=True)
def _clean_registry_state(monkeypatch):
    _force_available(monkeypatch, _BROWSER_TOOL_NAMES + _OTHER_TOOL_NAMES)
    invalidate_check_fn_cache()
    model_tools._tool_defs_cache.clear()
    yield
    model_tools._tool_defs_cache.clear()
    invalidate_check_fn_cache()


class TestBrowserNavigateCrossToolReferences:
    def test_no_web_tools_no_terminal_strips_both_sentences(self):
        desc = _browser_navigate_description(
            enabled_toolsets=["browser"], disabled_toolsets=["web"],
        )
        assert "web_search" not in desc
        assert "web_extract" not in desc
        assert "terminal" not in desc
        # The rest of the description must still read as a coherent sentence.
        assert "Use browser tools when you need to interact with a page" in desc

    def test_terminal_available_no_web_tools_keeps_only_terminal_mention(self):
        desc = _browser_navigate_description(
            enabled_toolsets=["browser", "terminal"], disabled_toolsets=["web"],
        )
        assert "web_search" not in desc
        assert "web_extract" not in desc
        assert "prefer curl via the terminal tool;" in desc

    def test_web_tools_available_no_terminal_keeps_only_web_extract_mention(self):
        desc = _browser_navigate_description(
            enabled_toolsets=["browser", "web"],
        )
        assert "terminal" not in desc
        assert "prefer web_search or web_extract (faster, cheaper)" in desc
        assert "prefer web_extract;" in desc

    def test_everything_available_preserves_original_description(self):
        desc = _browser_navigate_description(
            enabled_toolsets=["browser", "web", "terminal"],
        )
        assert "prefer web_search or web_extract (faster, cheaper)" in desc
        assert "prefer curl via the terminal tool or web_extract;" in desc
