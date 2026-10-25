"""Skills Hub GETs re-check every redirect hop, inside and outside a resolution session (#63919)."""

from unittest.mock import MagicMock

import httpx
import pytest

import tools.skills_hub as hub
from tools.skills_hub_github import GitHubAuth, GitHubSource
from tools.skills_hub_models import _get_text
from tools.skills_hub_sources import BrowseShSource

METADATA = "http://169.254.169.254/latest/meta-data/"


@pytest.fixture
def transport(monkeypatch):
    """Route the hub's SSRF-safe client through a mock transport; plain ``httpx.get`` is off limits."""
    routes: dict[str, httpx.Response] = {}
    seen: list[httpx.Request] = []

    def handle(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return routes.get(str(request.url).split("?")[0], httpx.Response(404))

    def client(**kwargs):
        assert kwargs["follow_redirects"] is False
        return httpx.Client(transport=httpx.MockTransport(handle), **kwargs)

    def plain_get(*_a, **_kw):
        raise AssertionError("Skills Hub fetch bypassed the SSRF-safe client")

    monkeypatch.setattr(hub, "create_ssrf_safe_client", client)
    monkeypatch.setattr(hub.httpx, "get", plain_get)
    return routes, seen


def test_browse_sh_skill_md_redirect_to_metadata_is_not_followed(transport):
    """A catalog-supplied skillMdUrl that 302s into link-local space stops at the first hop,
    with the real URL policy, both inside and outside a resolution session."""
    routes, seen = transport
    md_url = "https://93.184.216.34/skill.md"
    routes[md_url] = httpx.Response(302, headers={"location": METADATA})
    routes[METADATA] = httpx.Response(200, text="iam-credentials")
    src = BrowseShSource()
    item = {"slug": "demo/skill", "name": "demo"}
    src._fetch_catalog = lambda: [item]
    src._resolve_skill_md_url = lambda _slug, _item: md_url

    with hub.skills_hub_http_session():
        assert src.fetch("browse-sh/demo/skill") is None
    assert src.fetch("browse-sh/demo/skill") is None

    assert [str(r.url) for r in seen] == [md_url, md_url]


def test_redirect_into_a_policy_blocked_host_is_not_followed(transport, monkeypatch):
    """The connect-time SSRF check knows nothing of the website blocklist; each hop must."""
    routes, seen = transport
    monkeypatch.setattr(hub, "is_safe_url", lambda _url: True)
    monkeypatch.setattr(hub, "check_website_access", lambda url: (
        {"host": "blocked.example", "rule": "blocked.example"} if "blocked.example" in url else None))
    routes["https://browse.sh/api/skills/demo"] = httpx.Response(
        302, headers={"location": "https://blocked.example/skill.json"})
    routes["https://blocked.example/skill.json"] = httpx.Response(
        200, json={"skillMdUrl": "https://blocked.example/SKILL.md"})

    with hub.skills_hub_http_session():
        assert BrowseShSource()._resolve_skill_md_url("demo", {}) is None
    assert _get_text("https://browse.sh/api/skills/demo") is None

    assert {r.url.host for r in seen} == {"browse.sh"}


def test_github_token_stays_on_origin_and_params_on_first_hop(transport, monkeypatch):
    routes, seen = transport
    monkeypatch.setattr(hub, "is_safe_url", lambda _url: True)
    monkeypatch.setattr(hub, "check_website_access", lambda _url: None)
    api = "https://api.github.com/repos/o/r/contents/SKILL.md"
    moved = "https://api.github.com/repositories/1/contents/SKILL.md"
    cdn = "https://cdn.example/SKILL.md"
    routes[api] = httpx.Response(301, headers={"location": moved})
    routes[moved] = httpx.Response(302, headers={"location": cdn})
    routes[cdn] = httpx.Response(200, content=b"# skill")
    auth = MagicMock(spec=GitHubAuth)
    auth.get_headers.return_value = {"Authorization": "token ghp_secret", "Accept": "application/vnd.github.v3.raw"}

    resp = GitHubSource(auth)._github_get(api, params={"ref": "main"})

    assert resp is not None and resp.content == b"# skill"
    assert [(str(r.url), r.headers.get("authorization")) for r in seen] == [
        (f"{api}?ref=main", "token ghp_secret"),
        (moved, "token ghp_secret"),
        (cdn, None),
    ]
    assert seen[-1].headers["accept"] == "application/vnd.github.v3.raw"


def test_blocked_github_hop_is_not_retried(transport, monkeypatch):
    routes, seen = transport
    monkeypatch.setattr(hub, "is_safe_url", lambda url: "169.254" not in url)
    monkeypatch.setattr(hub, "check_website_access", lambda _url: None)
    sleeps = []
    monkeypatch.setattr("tools.skills_hub_github.time.sleep", sleeps.append)
    api = "https://api.github.com/repos/o/r"
    routes[api] = httpx.Response(302, headers={"location": METADATA})

    assert GitHubSource(MagicMock(spec=GitHubAuth))._github_get(api, headers={}) is None
    assert (len(seen), sleeps) == (1, [])
