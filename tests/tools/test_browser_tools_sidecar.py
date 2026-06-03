import json
import os
from io import BytesIO

from tools import browser_tools_sidecar as sidecar


class _FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self, *_args, **_kwargs):
        return json.dumps(self.payload).encode("utf-8")


def test_check_requirements_requires_env(monkeypatch):
    monkeypatch.delenv("BROWSER_TOOLS_URL", raising=False)
    assert sidecar.check_browser_tools_sidecar_requirements() is False

    monkeypatch.setenv("BROWSER_TOOLS_URL", "http://browser-tools:8790")
    assert sidecar.check_browser_tools_sidecar_requirements() is True


def test_fetch_posts_to_sidecar(monkeypatch):
    captured = {}

    def fake_urlopen(req, timeout):
        captured["url"] = req.full_url
        captured["timeout"] = timeout
        captured["body"] = json.loads(req.data.decode("utf-8"))
        return _FakeResponse({"success": True, "title": "Example Domain"})

    monkeypatch.setenv("BROWSER_TOOLS_URL", "http://browser-tools:8790/")
    monkeypatch.setattr(sidecar.urllib.request, "urlopen", fake_urlopen)

    result = json.loads(sidecar.browser_tools_fetch("https://example.com", text_limit=999999))

    assert result["success"] is True
    assert result["title"] == "Example Domain"
    assert captured["url"] == "http://browser-tools:8790/fetch"
    assert captured["body"]["url"] == "https://example.com"
    assert captured["body"]["text_limit"] == sidecar.MAX_TEXT_LIMIT


def test_extract_posts_selectors_to_sidecar(monkeypatch):
    captured = {}

    def fake_urlopen(req, timeout):
        captured["url"] = req.full_url
        captured["body"] = json.loads(req.data.decode("utf-8"))
        return _FakeResponse({"success": True, "fields": {"quotes": ["hello"]}})

    monkeypatch.setenv("BROWSER_TOOLS_URL", "http://browser-tools:8790")
    monkeypatch.setattr(sidecar.urllib.request, "urlopen", fake_urlopen)

    result = json.loads(sidecar.browser_tools_extract(
        "https://quotes.toscrape.com/",
        selectors={"quotes": ".quote .text::text"},
    ))

    assert result["success"] is True
    assert result["fields"]["quotes"] == ["hello"]
    assert captured["url"] == "http://browser-tools:8790/extract"
    assert captured["body"]["selectors"] == {"quotes": ".quote .text::text"}


def test_missing_sidecar_env_returns_helpful_error(monkeypatch):
    monkeypatch.delenv("BROWSER_TOOLS_URL", raising=False)

    result = json.loads(sidecar.browser_tools_fetch("https://example.com"))

    assert result["success"] is False
    assert "BROWSER_TOOLS_URL" in result["error"]
