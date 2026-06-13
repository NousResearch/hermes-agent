"""Regression test for #45701.

web_search's default result count must be configurable via
``web.search_default_limit`` instead of a hardcoded 5, while an explicit
``limit`` argument still wins.
"""

import tools.web_tools as wt
import agent.web_search_registry as wsr


def _mock_provider(captured):
    class _P:
        name = "mock"

        def supports_search(self):
            return True

        def search(self, query, limit):
            captured["limit"] = limit
            return {"success": True, "data": {"web": []}}

    return _P()


def _patch_search(monkeypatch, captured, *, config_limit=None):
    monkeypatch.setattr(wt, "_ensure_web_plugins_loaded", lambda: None)
    monkeypatch.setattr(wt, "_get_search_backend", lambda: "")
    monkeypatch.setattr(wsr, "get_active_search_provider", lambda: _mock_provider(captured))
    cfg = {} if config_limit is None else {"search_default_limit": config_limit}
    monkeypatch.setattr(wt, "_load_web_config", lambda: cfg)


def test_omitted_limit_uses_configured_default(monkeypatch):
    """The fix: when the caller omits limit, the configured default is used.
    Fails on main, where the limit is a hardcoded 5."""
    captured = {}
    _patch_search(monkeypatch, captured, config_limit=20)
    wt.web_search_tool("hello")
    assert captured["limit"] == 20


def test_explicit_limit_overrides_config(monkeypatch):
    captured = {}
    _patch_search(monkeypatch, captured, config_limit=20)
    wt.web_search_tool("hello", limit=7)
    assert captured["limit"] == 7


def test_default_falls_back_to_5_when_unset(monkeypatch):
    captured = {}
    _patch_search(monkeypatch, captured, config_limit=None)
    wt.web_search_tool("hello")
    assert captured["limit"] == 5


def test_helper_clamps_out_of_range(monkeypatch):
    monkeypatch.setattr(wt, "_load_web_config", lambda: {"search_default_limit": 500})
    assert wt._get_web_search_default_limit() == 100
    monkeypatch.setattr(wt, "_load_web_config", lambda: {"search_default_limit": 0})
    assert wt._get_web_search_default_limit() == 1
