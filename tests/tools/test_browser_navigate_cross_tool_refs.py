"""Regression tests for the browser_navigate cross-tool-reference stripping
in model_tools.py's get_tool_definitions().

The static browser_navigate schema (tools/browser_tool.py) hardcodes two
sentences that name other tools by name: "prefer web_search or web_extract"
and "prefer curl via the terminal tool or web_extract". Shipping a schema
description that names an unavailable tool causes the model to attempt
calls to it, since the model has no other signal that the tool doesn't
actually exist in this session's tool list.

model_tools.py post-processes the description to strip references to
whichever of {web_search, web_extract, terminal} are not actually available
this session. This file locks in that behavior for every availability
combination -- including the "browser" toolset's own default shape, which
bundles web_search but not web_extract or terminal (toolsets.py) -- so a
future edit to either sentence can't silently reintroduce a dangling
cross-reference. Two prior fixes were rejected for exactly this failure
mode:

  * PR #43352 fixed only the first sentence via a regex that broke on the
    literal periods inside ".md, .txt, .json, ..." and
    "raw.githubusercontent.com" in the second sentence, and was self-closed
    for still leaving "prefer web_search or web_extract" in place.
  * An earlier revision of this fix checked
    ``{"web_search", "web_extract"} & available_tool_names`` (a set
    intersection) to decide whether to touch the first sentence at all --
    which is truthy whenever *either* tool is present. Because the
    ``browser`` toolset bundles web_search but not web_extract, enabling
    only ``browser`` left "web_extract" named in the schema despite it
    being genuinely unavailable. This file's
    ``test_browser_only_default_config_never_names_unavailable_web_extract``
    and ``test_browser_plus_terminal_without_web_toolset_never_names_unavailable_web_extract``
    cases pin that exact scenario.
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

# Every cross-tool name that could plausibly appear in the browser_navigate
# description. Used by the invariant check below.
_CROSS_TOOL_NAMES = ("web_search", "web_extract", "terminal")


def _force_available(monkeypatch, names):
    """Force check_fn() -> True for every registered tool in *names* that
    exists, so schema availability in this test doesn't depend on whether
    the local environment has API keys / a browser backend installed."""
    for name in names:
        entry = registry.get_entry(name)
        if entry is not None:
            monkeypatch.setattr(entry, "check_fn", lambda: True)


def _get_definitions(enabled_toolsets, disabled_toolsets=None):
    model_tools._tool_defs_cache.clear()
    return model_tools.get_tool_definitions(
        enabled_toolsets=enabled_toolsets,
        disabled_toolsets=disabled_toolsets,
        quiet_mode=True,
        skip_tool_search_assembly=True,
    )


def _browser_navigate_description(enabled_toolsets, disabled_toolsets=None):
    defs = _get_definitions(enabled_toolsets, disabled_toolsets)
    for tool_def in defs:
        fn = tool_def.get("function", {})
        if fn.get("name") == "browser_navigate":
            return fn.get("description", "")
    raise AssertionError("browser_navigate not present in tool definitions")


def _assert_no_unavailable_tool_named(enabled_toolsets, disabled_toolsets=None):
    """Invariant: every cross-tool name mentioned in the browser_navigate
    description must correspond to a tool that is actually available."""
    defs = _get_definitions(enabled_toolsets, disabled_toolsets)
    available = {d["function"]["name"] for d in defs}
    desc = next(
        d["function"]["description"] for d in defs
        if d["function"]["name"] == "browser_navigate"
    )
    for tool_name in _CROSS_TOOL_NAMES:
        if tool_name in desc:
            assert tool_name in available, (
                f"browser_navigate description names {tool_name!r}, but it "
                f"is not in the available tool set {sorted(available)!r}"
            )
    return desc


@pytest.fixture(autouse=True)
def _clean_registry_state(monkeypatch):
    _force_available(monkeypatch, _BROWSER_TOOL_NAMES + _OTHER_TOOL_NAMES)
    invalidate_check_fn_cache()
    model_tools._tool_defs_cache.clear()
    yield
    model_tools._tool_defs_cache.clear()
    invalidate_check_fn_cache()


class TestBrowserNavigateCrossToolReferences:
    def test_browser_only_default_config_never_names_unavailable_web_extract(self):
        """The 'browser' toolset bundles web_search but NOT web_extract or
        terminal (toolsets.py). This is the realistic default shape for a
        browser-only session and must not leave web_extract named."""
        desc = _assert_no_unavailable_tool_named(enabled_toolsets=["browser"])
        assert "web_search" in desc
        assert "prefer web_search (faster, cheaper)" in desc

    def test_browser_plus_terminal_without_web_toolset_never_names_unavailable_web_extract(self):
        """browser + terminal, with the 'web' toolset never enabled: terminal
        is available but web_extract still is not (only web_search is,
        bundled via the browser toolset)."""
        desc = _assert_no_unavailable_tool_named(
            enabled_toolsets=["browser", "terminal"],
        )
        assert "web_search" in desc
        assert "prefer web_search (faster, cheaper)" in desc
        assert "prefer curl via the terminal tool;" in desc

    def test_web_toolset_disabled_strips_both_sentences(self):
        desc = _assert_no_unavailable_tool_named(
            enabled_toolsets=["browser"], disabled_toolsets=["web"],
        )
        assert "web_search" not in desc
        assert "web_extract" not in desc
        assert "terminal" not in desc
        # The rest of the description must still read as a coherent sentence.
        assert "Use browser tools when you need to interact with a page" in desc

    def test_terminal_available_web_toolset_disabled_keeps_only_terminal_mention(self):
        desc = _assert_no_unavailable_tool_named(
            enabled_toolsets=["browser", "terminal"], disabled_toolsets=["web"],
        )
        assert "web_search" not in desc
        assert "web_extract" not in desc
        assert "prefer curl via the terminal tool;" in desc

    def test_web_toolset_available_no_terminal_keeps_only_web_extract_mention(self):
        desc = _assert_no_unavailable_tool_named(
            enabled_toolsets=["browser", "web"],
        )
        assert "terminal" not in desc
        assert "prefer web_search or web_extract (faster, cheaper)" in desc
        assert "prefer web_extract;" in desc

    def test_everything_available_preserves_original_description(self):
        desc = _assert_no_unavailable_tool_named(
            enabled_toolsets=["browser", "web", "terminal"],
        )
        assert "prefer web_search or web_extract (faster, cheaper)" in desc
        assert "prefer curl via the terminal tool or web_extract;" in desc
