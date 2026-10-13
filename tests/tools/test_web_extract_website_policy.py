"""Regression for #127696: ``security.website_blocklist`` is enforced on every web_extract backend.

The shared extract path consulted the policy only to decide whether to use the cache; blocked URLs
were still dispatched, so only providers that re-check the policy themselves (Firecrawl) honoured
it. The policy is now applied once, before any provider sees the URL.
"""

import asyncio
import json
from unittest.mock import patch

import tools.web_tools as wt

BLOCKED = "https://blocked.example/page"
ALLOWED = "https://allowed.example/page"


class _AsyncTrue:
    async def __call__(self, *a, **k):
        return True


def _policy(url, *a, **k):
    if "blocked.example" in url:
        return {"host": "blocked.example", "rule": "blocked.example", "source": "config",
                "message": "Blocked by website policy: blocked.example"}
    return None


def test_blocked_urls_never_reach_a_provider_without_its_own_policy_check(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    dispatched = []

    class PolicyUnawareProvider:
        name = "fake"
        display_name = "Fake"

        def supports_extract(self):
            return True

        async def extract(self, urls, **kwargs):
            dispatched.extend(urls)
            return [{"url": u, "title": "T", "content": "page body", "raw_content": "page body", "metadata": {}}
                    for u in urls]

    with patch("tools.web_tools._ensure_web_plugins_loaded"), \
         patch("tools.web_tools._get_extract_backend", return_value="fake"), \
         patch("tools.web_tools.async_is_safe_url", new=_AsyncTrue()), \
         patch("tools.website_policy.check_website_access", side_effect=_policy), \
         patch("agent.web_search_registry.get_provider", return_value=PolicyUnawareProvider()):
        result = json.loads(asyncio.new_event_loop().run_until_complete(
            wt.web_extract_tool([BLOCKED, ALLOWED])))

    assert dispatched == [ALLOWED]
    by_url = {r["url"]: r for r in result["results"]}
    assert "website policy" in json.dumps(by_url[BLOCKED]).lower()
    assert by_url[ALLOWED].get("content")
    assert [r["url"] for r in result["results"]] == [BLOCKED, ALLOWED]
