import json

import requests

from tools import browser_use_tool


class _Response:
    def __init__(self, status_code=200, payload=None, text=""):
        self.status_code = status_code
        self._payload = payload if payload is not None else {}
        self.text = text

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}")


class _Session:
    def __init__(self, responses, call_log):
        self._responses = responses
        self._call_log = call_log
        self.trust_env = True
        self.proxies = None

    def request(self, method, url, **kwargs):
        self._call_log.append(
            {
                "method": method,
                "url": url,
                "kwargs": kwargs,
                "trust_env": self.trust_env,
                "proxies": self.proxies,
            }
        )
        result = self._responses.pop(0)
        if isinstance(result, Exception):
            raise result
        return result

    def close(self):
        return None


def test_candidate_endpoints_follow_requested_order(monkeypatch):
    monkeypatch.setenv("BROWSER_USE_AGENT_URL", "http://env-endpoint:5056")

    assert browser_use_tool._candidate_endpoints("http://explicit-endpoint:5000/v1/query") == [
        "http://explicit-endpoint:5000",
        "http://env-endpoint:5056",
        "http://browser-use-agent:5000",
        "http://127.0.0.1:5056",
        "http://host.docker.internal:5056",
    ]


def test_check_browser_use_requirements_probes_health(monkeypatch):
    call_log = []
    responses = [_Response(payload={"ok": True})]
    monkeypatch.setenv("BROWSER_USE_AGENT_URL", "http://127.0.0.1:5056")
    monkeypatch.setattr(
        browser_use_tool.requests,
        "Session",
        lambda: _Session(responses, call_log),
    )

    assert browser_use_tool.check_browser_use_requirements() is True
    assert call_log[0]["url"] == "http://127.0.0.1:5056/health"
    assert call_log[0]["trust_env"] is False


def test_browser_use_posts_query_with_clamped_steps(monkeypatch):
    call_log = []
    responses = [_Response(payload={"result": "Example Domain"})]
    monkeypatch.setattr(
        browser_use_tool.requests,
        "Session",
        lambda: _Session(responses, call_log),
    )

    result = json.loads(
        browser_use_tool.browser_use_tool(
            {
                "task": "Open example.com and return page title",
                "max_steps": 999,
                "headless": False,
                "endpoint": "http://127.0.0.1:5056",
            }
        )
    )

    assert result == {
        "success": True,
        "endpoint": "http://127.0.0.1:5056",
        "result": "Example Domain",
    }
    assert call_log[0]["url"] == "http://127.0.0.1:5056/v1/query"
    assert call_log[0]["kwargs"]["json"] == {
        "task": "Open example.com and return page title",
        "max_steps": 50,
        "headless": False,
    }
    assert call_log[0]["trust_env"] is False


def test_browser_use_surfaces_http_error_cleanly(monkeypatch):
    call_log = []
    responses = [_Response(status_code=500, payload={"detail": "browser-use query failed: missing API key"})]
    monkeypatch.setattr(
        browser_use_tool.requests,
        "Session",
        lambda: _Session(responses, call_log),
    )

    result = json.loads(
        browser_use_tool.browser_use_tool(
            {
                "task": "Open example.com and return page title",
                "endpoint": "http://browser-use-agent:5000",
            }
        )
    )

    assert result["endpoint"] == "http://browser-use-agent:5000"
    assert "HTTP 500" in result["error"]
    assert "missing API key" in result["error"]
