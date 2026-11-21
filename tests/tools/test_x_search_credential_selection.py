"""Explicit X Search credentials must not silently switch billing paths."""

import json
from types import SimpleNamespace

import pytest


@pytest.fixture
def credentials(monkeypatch):
    from agent import credential_pool
    from tools import xai_http

    monkeypatch.setattr(xai_http, "_resolve_explicit_xai_api_key", lambda: "test-paid")
    monkeypatch.setattr(xai_http, "_xai_base_url_override", lambda: "")
    pool = SimpleNamespace(select=lambda: SimpleNamespace(access_token="test-oauth"))
    monkeypatch.setattr(credential_pool, "load_pool", lambda _: pool)
    return pool


def test_explicit_oauth_overrides_shared_api_key(credentials, monkeypatch):
    from tools import x_search_tool as search

    monkeypatch.setattr(search, "_load_x_search_config", lambda: {"credential_source": "oauth"})
    assert search._resolve_xai_bearer() == ("test-oauth", "https://api.x.ai/v1", "xai-oauth")


@pytest.mark.parametrize("failure", ["missing", "error", "blank"])
def test_explicit_oauth_never_falls_back(credentials, monkeypatch, failure):
    from tools import x_search_tool as search

    def select():
        if failure == "error":
            raise RuntimeError("refresh failed")
        return None if failure == "missing" else SimpleNamespace(access_token="")

    credentials.select = select
    monkeypatch.setattr(search, "_load_x_search_config", lambda: {"credential_source": "oauth"})
    monkeypatch.setattr("requests.post", lambda *a, **kw: pytest.fail("must not send a paid request"))
    result = json.loads(search.x_search_tool("test"))
    assert "No API-key fallback" in result["error"]
    assert search.check_x_search_requirements() is False


@pytest.mark.parametrize("source", ["api_key", "invalid", None])
def test_explicit_selection_never_switches_to_other_source(credentials, monkeypatch, source):
    from tools import xai_http
    from tools import x_search_tool as search

    monkeypatch.setattr(xai_http, "_resolve_explicit_xai_api_key", lambda: "")
    monkeypatch.setattr(search, "_load_x_search_config", lambda: {"credential_source": source})
    with pytest.raises(RuntimeError):
        search._resolve_xai_bearer()


@pytest.mark.parametrize("config", [{}, {"credential_source": "auto"}, {"credential_source": "api_key"}])
def test_legacy_auto_and_explicit_key_prefer_key(credentials, monkeypatch, config):
    from tools import x_search_tool as search

    monkeypatch.setattr(search, "_load_x_search_config", lambda: config)
    assert search._resolve_xai_bearer()[::2] == ("test-paid", "xai")


def test_unreadable_selection_fails_closed(credentials, monkeypatch):
    from tools import x_search_tool as search

    def unreadable():
        raise OSError("unreadable config")

    monkeypatch.setattr("hermes_cli.config.load_config", unreadable)
    monkeypatch.setattr("requests.post", lambda *a, **kw: pytest.fail("must not make a request"))
    result = json.loads(search.x_search_tool("test"))
    assert "Cannot read X Search credential selection" in result["error"]
