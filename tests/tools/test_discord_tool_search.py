"""Guild message search in the Discord REST tool."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import tools.discord_tool as dt


class TestSearchMessages:
    def _run(self, monkeypatch, payload, status=200, capture=None):
        def fake_req(method, path, token, params=None, body=None, timeout=15):
            if capture is not None:
                capture.append((method, path, params))
            return payload

        monkeypatch.setattr(dt, "_discord_request", fake_req)
        return json.loads(dt._search_messages("tok", guild_id="1", query="deploy"))

    def test_flattens_nested_hit_groups(self, monkeypatch):
        payload = {
            "messages": [
                [{"id": "10", "content": "deploy done", "channel_id": "99",
                  "author": {"id": "5", "username": "scott"}}],
                [{"id": "11", "content": "deploy failed", "channel_id": "98",
                  "author": {"id": "6", "username": "remi"}}],
            ],
            "total_results": 2,
            "doing_deep_historical_index": False,
        }
        out = self._run(monkeypatch, payload)
        assert out["count"] == 2
        assert [m["id"] for m in out["messages"]] == ["10", "11"]
        assert out["messages"][0]["channel_id"] == "99"
        assert out["total_results"] == 2

    def test_index_not_ready_returns_structured_signal(self, monkeypatch):
        """202 has no `messages` key; the agent must be told to retry, not see zero hits."""
        payload = {"message": "indexing", "code": 110000,
                   "documents_indexed": 42, "retry_after": 5}
        out = self._run(monkeypatch, payload)
        assert out["index_not_ready"] is True
        assert out["retry_after_seconds"] == 5
        assert "messages" not in out

    def test_requires_at_least_one_filter(self, monkeypatch):
        monkeypatch.setattr(dt, "_discord_request",
                            lambda *a, **k: pytest.fail("must not call the API"))
        out = json.loads(dt._search_messages("tok", guild_id="1"))
        assert "error" in out
        assert "at least one filter" in out["error"]

    def test_clamps_limit_and_offset_to_server_schema(self, monkeypatch):
        cap = []
        monkeypatch.setattr(
            dt, "_discord_request",
            lambda m, p, t, params=None, body=None, timeout=15: cap.append(params) or {"messages": []})
        dt._search_messages("tok", guild_id="1", query="x", limit=500, offset=99999)
        assert cap[0]["limit"] == "25"
        assert cap[0]["offset"] == str(dt._SEARCH_MAX_OFFSET)

    def test_truncates_overlong_content_query(self, monkeypatch):
        cap = []
        monkeypatch.setattr(
            dt, "_discord_request",
            lambda m, p, t, params=None, body=None, timeout=15: cap.append(params) or {"messages": []})
        dt._search_messages("tok", guild_id="1", query="z" * 5000)
        assert len(cap[0]["content"]) == dt._SEARCH_MAX_CONTENT_LENGTH

    def test_filter_only_search_without_query_is_allowed(self, monkeypatch):
        cap = []
        monkeypatch.setattr(
            dt, "_discord_request",
            lambda m, p, t, params=None, body=None, timeout=15: cap.append(params) or {"messages": []})
        dt._search_messages("tok", guild_id="1", has="image")
        assert cap[0]["has"] == "image"
        assert "content" not in cap[0]

    def test_hits_the_guild_search_route(self, monkeypatch):
        cap = []
        self._run(monkeypatch, {"messages": []}, capture=cap)
        assert cap[0][0] == "GET"
        assert cap[0][1] == "/guilds/1/messages/search"


class TestRegistration:
    def test_search_messages_is_in_the_core_toolset(self):
        assert "search_messages" in dt._CORE_ACTION_NAMES
        assert "search_messages" in dt._CORE_ACTIONS
        assert "search_messages" not in dt._ADMIN_ACTIONS

    def test_only_guild_id_is_required(self):
        assert dt._REQUIRED_PARAMS["search_messages"] == ["guild_id"]

    def test_new_params_are_plumbed_through_handler_defaults(self):
        for key in ("author_id", "has", "offset"):
            assert key in dt._HANDLER_DEFAULTS

    def test_schema_exposes_search_params(self):
        schema = dt._build_schema(["search_messages"], {}, tool_name="discord")
        props = schema["parameters"]["properties"]
        for key in ("author_id", "has", "offset", "query"):
            assert key in props

    def test_content_intent_note_covers_search(self):
        caps = {"detected": True, "has_message_content": False}
        schema = dt._build_schema(["search_messages"], caps, tool_name="discord")
        assert "MESSAGE_CONTENT" in schema["description"]

    def test_403_hint_directs_to_fetch_messages_fallback(self):
        msg = dt._enrich_403("search_messages", "{}")
        assert "fetch_messages" in msg
