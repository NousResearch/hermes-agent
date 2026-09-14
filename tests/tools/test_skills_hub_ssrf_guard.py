#!/usr/bin/env python3
"""SSRF guard regression: Browse.sh + LobeHub adapters must route through
_guardd_http_get (is_safe_url + per-hop redirect re-check), never plain httpx.

Bug: BrowseShSource.fetch did _get_text(md_url, follow_redirects=True) with
md_url from /api/skills/{slug} (attacker-influenced) — a catalog entry pointing
skillMdUrl at http://169.254.169.254/... was fetched. Same for LobeHub index/agent.
Fix: GuardedFetchMixin + _fetch_json, both classes inherit it.
Refs: tools/skills_hub_sources.py (BrowseShSource, LobeHubSource),
      tools/skills_hub_models.py (GuardedFetchMixin),
      tools/skills_hub.py (_guarded_http_get), tools/url_safety.py (is_safe_url).
"""
import unittest
from unittest.mock import patch

from tools.skills_hub_sources import BrowseShSource, LobeHubSource


class _Resp:
    def __init__(self, status_code=200, json_data=None, text=""):
        self.status_code = status_code
        self._json_data = json_data
        self.text = text

    def json(self):
        return self._json_data


EVIL_ITEM = {
    "slug": "evil.com/pwn-abc123", "name": "pwn", "description": "x",
    "hostname": "evil.com", "category": "t", "tags": [],
    "sourceUrl": "", "recommendedMethod": "", "proxies": False, "installCount": 0,
}
METADATA = "http://169.254.169.254/latest/meta-data/iam/security-credentials/"


class TestBrowseShSSRFGuard(unittest.TestCase):
    def test_uses_guarded_mixin(self):
        from tools.skills_hub_models import GuardedFetchMixin
        self.assertTrue(issubclass(BrowseShSource, GuardedFetchMixin))
        self.assertTrue(issubclass(LobeHubSource, GuardedFetchMixin))

    def test_metadata_skill_md_url_blocked(self):
        src = BrowseShSource()
        with patch.object(BrowseShSource, "_catalog_item", return_value=EVIL_ITEM):
            with patch("tools.skills_hub._guarded_http_get") as g:
                # detail API ok, content fetch blocked (metadata) -> None
                def side_effect(url, timeout=20):
                    if "api/skills/evil" in url:
                        return _Resp(200, {"skillMdUrl": METADATA})
                    return None  # blocked metadata URL
                g.side_effect = side_effect
                bundle = src.fetch("browse-sh/evil.com/pwn-abc123")
                self.assertIsNone(bundle)
                fetched_urls = [c.args[0] for c in g.call_args_list]
                self.assertIn(METADATA, fetched_urls)

    def test_no_unguarded_get_in_browsesh_path(self):
        src = BrowseShSource()
        with patch.object(BrowseShSource, "_catalog_item", return_value=EVIL_ITEM):
            def guarded(url, timeout=20):
                if "api/skills/evil" in url:
                    return _Resp(200, {"skillMdUrl": METADATA})
                return None  # metadata content blocked
            with patch("tools.skills_hub._guarded_http_get", side_effect=guarded):
                with patch("tools.skills_hub_sources._get_text") as u1:
                    with patch("tools.skills_hub_sources._get_json") as u2:
                        u1.side_effect = AssertionError("unguarded _get_text used")
                        u2.side_effect = AssertionError("unguarded _get_json used")
                        # content blocked -> None, but must not touch unguarded fns
                        bundle = src.fetch("browse-sh/evil.com/pwn-abc123")
                        self.assertIsNone(bundle)


if __name__ == "__main__":
    unittest.main()
