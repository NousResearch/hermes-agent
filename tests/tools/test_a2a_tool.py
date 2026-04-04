"""Tests for the A2A (Agent-to-Agent) client tools.

All tests use mocks — no real A2A servers or network calls are made.
Covers: a2a_discover, a2a_call (non-streaming + SSE), a2a_local_scan,
        agent card caching, named agent resolution.
"""

import json
import time
from unittest.mock import MagicMock, patch

import httpx
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _agent_card(name="test-agent", streaming=True, skills=None):
    return {
        "name": name,
        "description": f"{name} description",
        "url": "http://localhost:9000",
        "version": "1.0.0",
        "capabilities": {"streaming": streaming},
        "skills": [{"id": s, "name": s} for s in (skills or ["coding"])],
        "metadata": {"model": "test-model"},
    }


def _mock_http_response(status_code=200, json_data=None, text=""):
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = json_data or {}
    resp.text = text
    if status_code >= 400:
        resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            f"HTTP {status_code}", request=MagicMock(), response=resp
        )
    else:
        resp.raise_for_status.return_value = None
    return resp


def _jsonrpc_result(text="hello", state="completed"):
    return {
        "jsonrpc": "2.0",
        "id": "t1",
        "result": {
            "id": "t1",
            "status": {"state": state},
            "artifacts": [{"parts": [{"type": "text", "text": text}]}],
        },
    }


# ---------------------------------------------------------------------------
# a2a_discover
# ---------------------------------------------------------------------------

class TestA2ADiscover:
    def setup_method(self):
        from tools.a2a_tool import _CARD_CACHE
        _CARD_CACHE.clear()

    def test_returns_summary(self):
        card = _agent_card()
        mock_resp = _mock_http_response(json_data=card)
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.get.return_value = mock_resp

        with patch("tools.a2a_tool.httpx.Client", return_value=mock_client):
            from tools.a2a_tool import a2a_discover
            result = json.loads(a2a_discover("http://localhost:9000"))

        assert result["name"] == "test-agent"
        assert result["streaming"] is True
        assert "coding" in result["skills"]
        assert result["endpoint"] == "http://localhost:9000"

    def test_http_error_returns_error_json(self):
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_resp = MagicMock()
        mock_resp.status_code = 404
        mock_resp.text = "Not Found"
        mock_resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            "404", request=MagicMock(), response=mock_resp
        )
        mock_client.get.return_value = mock_resp

        with patch("tools.a2a_tool.httpx.Client", return_value=mock_client):
            from tools.a2a_tool import a2a_discover
            result = json.loads(a2a_discover("http://localhost:9001"))

        assert "error" in result

    def test_connection_error_returns_error_json(self):
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.get.side_effect = httpx.ConnectError("refused")

        with patch("tools.a2a_tool.httpx.Client", return_value=mock_client):
            from tools.a2a_tool import a2a_discover
            result = json.loads(a2a_discover("http://localhost:9002"))

        assert "error" in result

    def test_caches_card_on_first_fetch(self):
        card = _agent_card()
        mock_resp = _mock_http_response(json_data=card)
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.get.return_value = mock_resp

        with patch("tools.a2a_tool.httpx.Client", return_value=mock_client):
            from tools.a2a_tool import a2a_discover, _get_cached_card
            a2a_discover("http://localhost:9003")
            cached = _get_cached_card("http://localhost:9003")

        assert cached is not None
        assert cached["name"] == "test-agent"

    def test_cache_hit_skips_network(self):
        card = _agent_card(name="cached-agent")
        from tools.a2a_tool import _set_cached_card, a2a_discover
        _set_cached_card("http://localhost:9004", card)

        with patch("tools.a2a_tool.httpx.Client") as mock_cls:
            result = json.loads(a2a_discover("http://localhost:9004"))
            mock_cls.assert_not_called()

        assert result["name"] == "cached-agent"

    def test_expired_cache_refetches(self):
        card = _agent_card(name="fresh-agent")
        mock_resp = _mock_http_response(json_data=card)
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.get.return_value = mock_resp

        from tools.a2a_tool import _CARD_CACHE
        _CARD_CACHE["http://localhost:9005"] = {
            "card": _agent_card(name="stale-agent"),
            "expires": time.monotonic() - 1,
        }

        with patch("tools.a2a_tool.httpx.Client", return_value=mock_client):
            from tools.a2a_tool import a2a_discover
            result = json.loads(a2a_discover("http://localhost:9005"))

        assert result["name"] == "fresh-agent"


# ---------------------------------------------------------------------------
# a2a_call — non-streaming
# ---------------------------------------------------------------------------

class TestA2ACallNonStreaming:
    def setup_method(self):
        from tools.a2a_tool import _CARD_CACHE
        _CARD_CACHE.clear()

    def _mock_client(self, card=None, post_response=None):
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.get.return_value = _mock_http_response(json_data=card or _agent_card(streaming=False))
        mock_client.post.return_value = _mock_http_response(json_data=post_response or _jsonrpc_result("4"))
        return mock_client

    def test_returns_response_text(self):
        with patch("tools.a2a_tool.httpx.Client", return_value=self._mock_client()):
            from tools.a2a_tool import a2a_call
            result = a2a_call("http://localhost:9000", "what is 2+2?", stream=False)
        assert result == "4"

    def test_jsonrpc_error_returns_error_json(self):
        error_resp = {"jsonrpc": "2.0", "id": "t1", "error": {"code": -32000, "message": "Agent error"}}
        client = self._mock_client(post_response=error_resp)
        with patch("tools.a2a_tool.httpx.Client", return_value=client):
            from tools.a2a_tool import a2a_call
            result = json.loads(a2a_call("http://localhost:9000", "hi", stream=False))
        assert "error" in result

    def test_empty_artifacts_returns_error(self):
        empty_resp = {"jsonrpc": "2.0", "id": "t1", "result": {"id": "t1", "status": {"state": "completed"}, "artifacts": []}}
        client = self._mock_client(post_response=empty_resp)
        with patch("tools.a2a_tool.httpx.Client", return_value=client):
            from tools.a2a_tool import a2a_call
            result = json.loads(a2a_call("http://localhost:9000", "hi", stream=False))
        assert "error" in result

    def test_bearer_token_sent_in_header(self):
        client = self._mock_client()
        with patch("tools.a2a_tool.httpx.Client", return_value=client):
            from tools.a2a_tool import a2a_call
            a2a_call("http://localhost:9000", "hi", bearer_token="secret", stream=False)
        _, kwargs = client.post.call_args
        assert kwargs["headers"]["Authorization"] == "Bearer secret"

    def test_session_id_included_in_payload(self):
        client = self._mock_client()
        with patch("tools.a2a_tool.httpx.Client", return_value=client):
            from tools.a2a_tool import a2a_call
            a2a_call("http://localhost:9000", "hi", session_id="sess-42", stream=False)
        _, kwargs = client.post.call_args
        assert kwargs["json"]["params"]["sessionId"] == "sess-42"


# ---------------------------------------------------------------------------
# a2a_call — SSE streaming
# ---------------------------------------------------------------------------

class TestA2ACallStreaming:
    def setup_method(self):
        from tools.a2a_tool import _CARD_CACHE
        _CARD_CACHE.clear()

    def _sse_response(self, events):
        """Build a mock streaming response from a list of result dicts."""
        lines = []
        for event in events:
            lines.append(f"data: {json.dumps(event)}")
            lines.append("")
        sse_text = "\n".join(lines)

        mock_resp = MagicMock()
        mock_resp.raise_for_status.return_value = None
        mock_resp.iter_lines.return_value = iter(sse_text.splitlines())
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        return mock_resp

    def test_streaming_returns_final_text(self):
        events = [
            {"result": {"status": {"state": "working"}, "artifacts": [{"parts": [{"type": "text", "text": "partial"}]}]}},
            {"result": {"status": {"state": "completed"}, "artifacts": [{"parts": [{"type": "text", "text": "final answer"}]}]}},
        ]
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.stream.return_value = self._sse_response(events)

        with patch("tools.a2a_tool.httpx.Client", return_value=mock_client):
            from tools.a2a_tool import a2a_call
            result = a2a_call("http://localhost:9000", "hi", stream=True)

        assert result == "final answer"

    def test_streaming_uses_sendsubscribe_method(self):
        events = [
            {"result": {"status": {"state": "completed"}, "artifacts": [{"parts": [{"type": "text", "text": "ok"}]}]}},
        ]
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.stream.return_value = self._sse_response(events)

        with patch("tools.a2a_tool.httpx.Client", return_value=mock_client):
            from tools.a2a_tool import a2a_call
            a2a_call("http://localhost:9000", "hi", stream=True)

        _, kwargs = mock_client.stream.call_args
        assert kwargs["json"]["method"] == "tasks/sendSubscribe"

    def test_auto_detect_streaming_from_card(self):
        card = _agent_card(streaming=True)
        events = [
            {"result": {"status": {"state": "completed"}, "artifacts": [{"parts": [{"type": "text", "text": "streamed"}]}]}},
        ]
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.get.return_value = _mock_http_response(json_data=card)
        mock_client.stream.return_value = self._sse_response(events)

        with patch("tools.a2a_tool.httpx.Client", return_value=mock_client):
            from tools.a2a_tool import a2a_call
            result = a2a_call("http://localhost:9006", "hi")

        assert result == "streamed"
        mock_client.stream.assert_called_once()

    def test_auto_detect_non_streaming_from_card(self):
        card = _agent_card(streaming=False)
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.get.return_value = _mock_http_response(json_data=card)
        mock_client.post.return_value = _mock_http_response(json_data=_jsonrpc_result("non-streamed"))

        with patch("tools.a2a_tool.httpx.Client", return_value=mock_client):
            from tools.a2a_tool import a2a_call
            result = a2a_call("http://localhost:9007", "hi")

        assert result == "non-streamed"
        mock_client.stream.assert_not_called()


# ---------------------------------------------------------------------------
# Named agent resolution
# ---------------------------------------------------------------------------

class TestNamedAgentResolution:
    def setup_method(self):
        from tools.a2a_tool import _CARD_CACHE
        _CARD_CACHE.clear()

    def test_resolves_url_from_config(self):
        agents = {"myagent": {"url": "http://192.168.1.100:9000"}}
        card = _agent_card(streaming=False)
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.get.return_value = _mock_http_response(json_data=card)
        mock_client.post.return_value = _mock_http_response(json_data=_jsonrpc_result("result"))

        with patch("tools.a2a_tool._load_a2a_agents", return_value=agents), \
             patch("tools.a2a_tool.httpx.Client", return_value=mock_client):
            from tools.a2a_tool import _tool_a2a_call
            result = _tool_a2a_call({"agent_name": "myagent", "message": "hello"})

        assert result == "result"

    def test_unknown_agent_returns_error(self):
        with patch("tools.a2a_tool._load_a2a_agents", return_value={}):
            from tools.a2a_tool import _tool_a2a_call
            result = json.loads(_tool_a2a_call({"agent_name": "ghost", "message": "hi"}))
        assert "error" in result
        assert "ghost" in result["error"]

    def test_no_url_and_no_agent_name_returns_error(self):
        from tools.a2a_tool import _tool_a2a_call
        result = json.loads(_tool_a2a_call({"message": "hi"}))
        assert "error" in result

    def test_missing_message_returns_error(self):
        from tools.a2a_tool import _tool_a2a_call
        result = json.loads(_tool_a2a_call({"url": "http://localhost:9000"}))
        assert "error" in result

    def test_config_bearer_token_forwarded(self):
        agents = {"secure": {"url": "http://localhost:9000", "bearer_token": "tok123"}}
        card = _agent_card(streaming=False)
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.get.return_value = _mock_http_response(json_data=card)
        mock_client.post.return_value = _mock_http_response(json_data=_jsonrpc_result("ok"))

        with patch("tools.a2a_tool._load_a2a_agents", return_value=agents), \
             patch("tools.a2a_tool.httpx.Client", return_value=mock_client):
            from tools.a2a_tool import _tool_a2a_call
            _tool_a2a_call({"agent_name": "secure", "message": "hi"})

        _, kwargs = mock_client.post.call_args
        assert kwargs["headers"]["Authorization"] == "Bearer tok123"


# ---------------------------------------------------------------------------
# a2a_local_scan
# ---------------------------------------------------------------------------

class TestA2ALocalScan:
    def _make_client(self, port_responses):
        """port_responses: dict of port -> (status_code, card) or Exception."""
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)

        def _get(url, **_):
            port = int(url.split(":")[2].split("/")[0])
            val = port_responses.get(port)
            if isinstance(val, Exception):
                raise val
            status, card = val
            return _mock_http_response(status_code=status, json_data=card)

        mock_client.get.side_effect = _get
        return mock_client

    def test_finds_running_agent(self):
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)

        def _get(url, **_):
            port = int(url.split(":")[2].split("/")[0])
            if port == 9000:
                return _mock_http_response(200, json_data=_agent_card(name="local-agent"))
            raise httpx.ConnectError("refused")

        mock_client.get.side_effect = _get

        with patch("tools.a2a_tool.httpx.Client", return_value=mock_client):
            from tools.a2a_tool import a2a_local_scan
            result = json.loads(a2a_local_scan())

        assert result["found"] == 1
        assert result["agents"][0]["name"] == "local-agent"
        assert result["agents"][0]["endpoint"] == "http://localhost:9000"

    def test_no_agents_found(self):
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.get.side_effect = httpx.ConnectError("refused")

        with patch("tools.a2a_tool.httpx.Client", return_value=mock_client):
            from tools.a2a_tool import a2a_local_scan
            result = json.loads(a2a_local_scan())

        assert result["found"] == 0
        assert result["agents"] == []

    def test_non_200_port_skipped(self):
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.get.return_value = _mock_http_response(status_code=404)

        with patch("tools.a2a_tool.httpx.Client", return_value=mock_client):
            from tools.a2a_tool import a2a_local_scan
            result = json.loads(a2a_local_scan())

        assert result["found"] == 0

    def test_multiple_agents_found(self):

        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)

        def _get(url, **_):
            port = int(url.split(":")[2].split("/")[0])
            if port in (9000, 9002):
                return _mock_http_response(200, json_data=_agent_card(name=f"agent-{port}"))
            raise httpx.ConnectError("refused")

        mock_client.get.side_effect = _get

        with patch("tools.a2a_tool.httpx.Client", return_value=mock_client):
            from tools.a2a_tool import a2a_local_scan
            result = json.loads(a2a_local_scan())

        assert result["found"] == 2
        names = {a["name"] for a in result["agents"]}
        assert names == {"agent-9000", "agent-9002"}

    def test_range_too_large_returns_error(self):
        from tools.a2a_tool import _tool_a2a_local_scan
        result = json.loads(_tool_a2a_local_scan({"port_start": 9000, "port_end": 9200}))
        assert "error" in result

    def test_inverted_range_returns_error(self):
        from tools.a2a_tool import _tool_a2a_local_scan
        result = json.loads(_tool_a2a_local_scan({"port_start": 9010, "port_end": 9000}))
        assert "error" in result
