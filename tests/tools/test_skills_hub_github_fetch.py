"""GitHubSource file downloads: which host serves a skill's files."""

from unittest.mock import MagicMock

import httpx
import pytest

from tools.skills_hub_github import GitHubAuth, GitHubSource


@pytest.mark.parametrize("raw_status", [200, 404])
def test_pinned_file_comes_from_raw_and_spends_no_api_call(monkeypatch, raw_status):
    """Every file of a skill folder was one Contents API call: an anonymous user (60/h) could
    never install a skill with ~58+ files (remotion-best-practices: 143 calls, rate_limited).
    A pinned file now comes from raw.githubusercontent.com; a raw miss (private repo) still
    falls back to the API."""
    import tools.skills_hub as hub
    hosts = []

    def fake_get(url, **_kw):
        hosts.append(url.split("/")[2])
        ok = "raw.githubusercontent.com" not in url or raw_status == 200
        return httpx.Response(200 if ok else raw_status, content=b"body" if ok else b"")

    monkeypatch.setattr(hub, "_skills_hub_http_get", fake_get)
    auth = MagicMock(spec=GitHubAuth)
    auth.get_headers.return_value = {}
    assert GitHubSource(auth)._fetch_file_bytes("o/r", "s/a b.md", ref="c" * 40) == b"body"
    assert hosts == (["raw.githubusercontent.com"] if raw_status == 200
                     else ["raw.githubusercontent.com", "api.github.com"])


def test_unreachable_raw_host_falls_back_to_the_api_and_is_tried_once(monkeypatch):
    """A DNS-poisoned or NXDOMAIN raw.githubusercontent.com (api.github.com fine) raises
    SSRFConnectionBlocked, a ValueError the raw attempt did not catch: every install failed before
    the API fallback. A dropped-packet firewall costs a 15 s timeout per file, so stop after one."""
    import tools.skills_hub as hub
    from tools.url_safety import SSRFConnectionBlocked
    hosts = []

    def fake_get(url, **_kw):
        hosts.append(url.split("/")[2])
        if "raw.githubusercontent.com" in url:
            raise SSRFConnectionBlocked("DNS resolution failed for: raw.githubusercontent.com")
        return httpx.Response(200, content=b"body")

    monkeypatch.setattr(hub, "_skills_hub_http_get", fake_get)
    auth = MagicMock(spec=GitHubAuth)
    auth.get_headers.return_value = {}
    source = GitHubSource(auth)
    assert [source._fetch_file_bytes("o/r", f"s/{i}.md", ref="c" * 40) for i in range(3)] == [b"body"] * 3
    assert hosts == ["raw.githubusercontent.com"] + ["api.github.com"] * 3
