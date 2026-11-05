"""``web.firecrawl_max_age_hours`` lets web_extract ask Firecrawl for a fresh copy.

Firecrawl answers a scrape from its own cached copy of the page while that copy is younger than
``maxAge`` (the v4 SDK sends 4 h when the caller names none; the REST API defaults to 2 days).
Hermes' ``web.cache_*`` settings never reach that copy, so this key is the only way to get a live
fetch. Unset keeps Firecrawl's default; 0 always fetches live; a bad value fails loud.
"""
from __future__ import annotations

import asyncio
import json

import pytest

from plugins.web import keyless_mcp
from plugins.web.firecrawl import provider as firecrawl_provider

URL = "https://news.example/story"


def _rest_payload() -> dict:
    return {"success": True, "data": {"markdown": "# story", "metadata": {"title": "Story", "url": URL, "sourceURL": URL, "statusCode": 200}}}


class _SdkClient:
    """Records the keyword arguments of each ``scrape`` call (stand-in for the v4 SDK client)."""

    def __init__(self):
        self.calls: list[dict] = []

    def scrape(self, **kwargs):
        self.calls.append(kwargs)
        return _rest_payload()


@pytest.fixture
def sdk_scrape(monkeypatch):
    def _run(web_config: dict) -> tuple[dict, list[dict]]:
        client = _SdkClient()
        monkeypatch.setattr("tools.web_tools._load_web_config", lambda: web_config)
        monkeypatch.setattr(firecrawl_provider, "_get_firecrawl_client", lambda _capability=None: client)
        monkeypatch.setattr(firecrawl_provider, "check_website_access", lambda url: None)
        monkeypatch.setattr(firecrawl_provider, "is_safe_url", lambda url: True)
        entry = asyncio.run(firecrawl_provider._scrape_one(URL, ["markdown"], "markdown"))
        return entry, client.calls

    return _run


def test_unset_keeps_firecrawls_default_window(sdk_scrape):
    entry, calls = sdk_scrape({})
    assert entry["content"] == "# story"
    assert "max_age" not in calls[0]


@pytest.mark.parametrize(("hours", "max_age_ms"), [(0, 0), (6, 21_600_000), (0.5, 1_800_000), ("24", 86_400_000)])
def test_hours_become_firecrawl_max_age_in_ms(sdk_scrape, hours, max_age_ms):
    entry, calls = sdk_scrape({"firecrawl_max_age_hours": hours})
    assert entry["content"] == "# story"
    assert calls[0]["max_age"] == max_age_ms


@pytest.mark.parametrize("bad", [-1, "soon", True, float("inf")])
def test_unusable_value_names_the_key_and_skips_the_scrape(sdk_scrape, bad):
    entry, calls = sdk_scrape({"firecrawl_max_age_hours": bad})
    assert calls == []
    assert entry["content"] == ""
    assert "web.firecrawl_max_age_hours" in entry["error"]


def test_hermes_config_set_knows_the_key():
    """The Firecrawl plugin declares the key, so ``hermes config set`` saves it without the
    unknown-key notice (and a typo still gets one)."""
    from hermes_cli.config import _validate_config_key

    assert _validate_config_key("web.firecrawl_max_age_hours") == (True, None)
    assert _validate_config_key("web.firecrawl_max_age_hourz")[0] is False


def _capture_rest_posts(monkeypatch) -> list[dict]:
    sent: list[dict] = []

    class _Response:
        def raise_for_status(self):
            return None

        def json(self):
            return _rest_payload()

    def _fake_post(url, *, json, headers, timeout):
        sent.append({"url": url, "json": json})
        return _Response()

    monkeypatch.setattr(firecrawl_provider.httpx, "post", _fake_post)
    return sent


def test_keyless_rest_client_sends_max_age_as_maxAge(monkeypatch):
    sent = _capture_rest_posts(monkeypatch)
    firecrawl_provider._KeylessFirecrawlClient().scrape(url=URL, formats=["markdown"], max_age=0)
    assert sent[0]["json"] == {"url": URL, "formats": ["markdown"], "maxAge": 0}


def test_keyless_ring_scrape_honours_the_setting(monkeypatch):
    sent = _capture_rest_posts(monkeypatch)
    monkeypatch.setattr("tools.web_tools._load_web_config", lambda: {"firecrawl_max_age_hours": 0})
    [entry] = keyless_mcp.firecrawl_extract_keyless([URL])
    assert entry["content"] == "# story"
    assert sent[0]["json"]["maxAge"] == 0


def test_keyless_ring_reports_an_unusable_value_without_scraping(monkeypatch):
    sent = _capture_rest_posts(monkeypatch)
    monkeypatch.setattr("tools.web_tools._load_web_config", lambda: {"firecrawl_max_age_hours": "soon"})
    [entry] = keyless_mcp.firecrawl_extract_keyless([URL])
    assert sent == []
    assert entry["error"].startswith("web.firecrawl_max_age_hours must be")


def test_web_extract_tool_asks_firecrawl_for_a_live_copy(tmp_path, monkeypatch, public_dns):
    """Real config file, real extract dispatch (no key: Firecrawl's free tier through the keyless ring,
    pinned by the explicit selection): the setting reaches the request Firecrawl receives."""
    from hermes_cli.config import atomic_config_write
    from tests.tools.conftest import register_all_web_providers
    from tools import web_tools

    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.delenv("FIRECRAWL_API_KEY", raising=False)
    monkeypatch.delenv("FIRECRAWL_API_URL", raising=False)
    atomic_config_write(home / "config.yaml", {"web": {"extract_backend": "firecrawl", "firecrawl_max_age_hours": 0, "cache_enabled": False}})
    register_all_web_providers()
    sent = _capture_rest_posts(monkeypatch)

    result = json.loads(asyncio.run(web_tools.web_extract_tool([URL])))

    assert result["results"][0]["content"] == "# story"
    assert [post["json"]["maxAge"] for post in sent] == [0]
