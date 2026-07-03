"""Tests for the dot payment tool module (no live Dot API required)."""

import json
from unittest.mock import patch

import pytest

from tools import dot_tool as t


@pytest.fixture
def dot_env(monkeypatch):
    monkeypatch.setenv("DOT_API_KEY", "dot-key")
    monkeypatch.setenv("DOT_API_BASE_URL", "https://api.dot.example.com")
    monkeypatch.delenv("DOT_API_KEY_HEADER", raising=False)


class TestAvailability:
    def test_unavailable_without_key(self, monkeypatch):
        monkeypatch.delenv("DOT_API_KEY", raising=False)
        monkeypatch.setenv("DOT_API_BASE_URL", "https://api.dot.example.com")
        assert t.check_dot_requirements() is False

    def test_unavailable_without_base_url(self, monkeypatch):
        monkeypatch.setenv("DOT_API_KEY", "x")
        monkeypatch.delenv("DOT_API_BASE_URL", raising=False)
        assert t.check_dot_requirements() is False

    def test_available_with_both(self, dot_env):
        assert t.check_dot_requirements() is True


class TestAuthHeaders:
    def test_bearer_by_default(self, dot_env):
        assert t._auth_headers() == {"Authorization": "Bearer dot-key"}

    def test_custom_header(self, dot_env, monkeypatch):
        monkeypatch.setenv("DOT_API_KEY_HEADER", "x-api-key")
        assert t._auth_headers() == {"x-api-key": "dot-key"}


class TestAsItems:
    def test_bare_list(self):
        assert t._as_items([{"a": 1}]) == [{"a": 1}]

    def test_wrapped_list(self):
        assert t._as_items({"data": [{"a": 1}]}) == [{"a": 1}]
        assert t._as_items({"payouts": [{"b": 2}]}) == [{"b": 2}]

    def test_single_object_wrapped(self):
        assert t._as_items({"status": "paid"}) == [{"status": "paid"}]

    def test_garbage_is_empty(self):
        assert t._as_items(None) == []
        assert t._as_items("nope") == []


class TestHandler:
    def test_noop_when_not_configured(self, monkeypatch):
        monkeypatch.delenv("DOT_API_KEY", raising=False)
        monkeypatch.delenv("DOT_API_BASE_URL", raising=False)
        out = json.loads(t.dot_tool({"action": "get_payment_history", "user_id": "u1"}))
        assert "error" in out and out["error"]

    def test_unknown_action(self, dot_env):
        out = json.loads(t.dot_tool({"action": "bogus", "user_id": "u1"}))
        assert "error" in out and out["error"]

    def test_missing_user_id(self, dot_env):
        out = json.loads(t.dot_tool({"action": "get_payment_status", "user_id": ""}))
        assert "error" in out and out["error"]

    def test_history_success(self, dot_env):
        payload = [{"id": "p1", "status": "paid"}, {"id": "p2", "status": "pending"}]
        with patch.object(t, "_dot_get", return_value=(payload, None)) as g:
            out = json.loads(t.dot_tool({"action": "get_payment_history", "user_id": "u1"}))
        assert out["_type"] == "dot_payment_history"
        assert out["items"] == payload
        assert out["error"] is None
        g.assert_called_once()

    def test_status_success_scopes_to_gig(self, dot_env):
        payload = {"data": [{"id": "p1", "status": "paid", "campaign_id": "g9"}]}
        with patch.object(t, "_dot_get", return_value=(payload, None)) as g:
            out = json.loads(
                t.dot_tool(
                    {"action": "get_payment_status", "user_id": "u1", "gig_id": "g9"}
                )
            )
        assert out["_type"] == "dot_payment_status"
        assert out["gig_id"] == "g9"
        assert out["items"] == [{"id": "p1", "status": "paid", "campaign_id": "g9"}]
        # gig id was forwarded to Dot as a query param
        _, params = g.call_args.args
        assert params.get(t._DOT_GIG_PARAM) == "g9"

    def test_campaign_id_alias(self, dot_env):
        with patch.object(t, "_dot_get", return_value=([], None)) as g:
            t.dot_tool({"action": "get_payment_status", "user_id": "u1", "campaign_id": "g5"})
        _, params = g.call_args.args
        assert params.get(t._DOT_GIG_PARAM) == "g5"

    def test_dot_error_degrades_gracefully(self, dot_env):
        with patch.object(t, "_dot_get", return_value=(None, "HTTP 502")):
            out = json.loads(t.dot_tool({"action": "get_payment_history", "user_id": "u1"}))
        assert "error" in out and "502" in out["error"]

    def test_handler_never_raises_on_internal_error(self, dot_env):
        with patch.object(t, "_get_payment_history", side_effect=RuntimeError("boom")):
            out = json.loads(t.dot_tool({"action": "get_payment_history", "user_id": "u1"}))
        assert "error" in out and out["error"]

    def test_limit_capped(self, dot_env):
        with patch.object(t, "_dot_get", return_value=([], None)) as g:
            t.dot_tool({"action": "get_payment_history", "user_id": "u1", "limit": 999})
        _, params = g.call_args.args
        assert params["limit"] == t._HARD_LIMIT


class TestDotGet:
    def test_missing_base_url_returns_error(self, monkeypatch):
        monkeypatch.setenv("DOT_API_KEY", "k")
        monkeypatch.delenv("DOT_API_BASE_URL", raising=False)
        data, err = t._dot_get("/payouts", {"user_id": "u1"})
        assert data is None
        assert err

    def test_http_error_is_captured(self, dot_env):
        import urllib.error

        with patch("urllib.request.urlopen", side_effect=urllib.error.HTTPError(
            "u", 404, "nf", {}, None
        )):
            data, err = t._dot_get("/payouts", {"user_id": "u1"})
        assert data is None
        assert err == "HTTP 404"
