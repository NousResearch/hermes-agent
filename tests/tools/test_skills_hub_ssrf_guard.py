"""Skills Hub outbound fetches must re-validate redirect targets (SSRF)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import httpx
import pytest


class _FakeResponse:
    def __init__(self, status_code=200, headers=None, text="", content=b"", json_data=None):
        self.status_code = status_code
        self.headers = headers or {}
        self.text = text
        self.content = content
        self._json_data = json_data

    def json(self):
        if self._json_data is None:
            raise ValueError("no json")
        return self._json_data


def test_guarded_http_get_blocks_redirect_to_metadata_ip():
    """Public first hop that 302s to 169.254.169.254 must not be followed."""
    from tools.skills_hub import _guarded_http_get

    public = "https://cdn.example/skill.md"
    metadata = "http://169.254.169.254/latest/meta-data/"
    calls = []

    def fake_get(url, **kwargs):
        calls.append(url)
        assert kwargs.get("follow_redirects") is False
        if url == public:
            return _FakeResponse(
                status_code=302,
                headers={"location": metadata},
            )
        raise AssertionError(f"should not request blocked hop: {url}")

    with patch("tools.skills_hub.httpx.get", side_effect=fake_get):
        with patch("tools.skills_hub.is_safe_url", side_effect=lambda u: u == public):
            with patch("tools.skills_hub.check_website_access", return_value=None):
                resp = _guarded_http_get(public, timeout=5)

    assert resp is None
    assert calls == [public]


def test_guarded_http_get_allows_safe_redirect_chain():
    from tools.skills_hub import _guarded_http_get

    first = "https://browse.sh/api/skills/x"
    second = "https://cdn.example/skill.md"
    body = "# hello"

    def fake_get(url, **kwargs):
        assert kwargs.get("follow_redirects") is False
        if url == first:
            return _FakeResponse(status_code=302, headers={"location": second})
        if url == second:
            return _FakeResponse(status_code=200, text=body)
        raise AssertionError(url)

    with patch("tools.skills_hub.httpx.get", side_effect=fake_get):
        with patch("tools.skills_hub.is_safe_url", return_value=True):
            with patch("tools.skills_hub.check_website_access", return_value=None):
                resp = _guarded_http_get(first, timeout=5)

    assert resp is not None
    assert resp.status_code == 200
    assert resp.text == body


def test_guarded_http_get_params_only_on_first_hop():
    from tools.skills_hub import _guarded_http_get

    first = "https://clawhub.example/download"
    second = "https://cdn.example/file.zip"
    seen_params = []

    def fake_get(url, **kwargs):
        seen_params.append(kwargs.get("params"))
        if url == first:
            return _FakeResponse(status_code=302, headers={"location": second})
        return _FakeResponse(status_code=200, content=b"PK")

    with patch("tools.skills_hub.httpx.get", side_effect=fake_get):
        with patch("tools.skills_hub.is_safe_url", return_value=True):
            with patch("tools.skills_hub.check_website_access", return_value=None):
                resp = _guarded_http_get(
                    first,
                    params={"slug": "x", "version": "1"},
                    timeout=5,
                )

    assert resp is not None
    assert seen_params[0] == {"slug": "x", "version": "1"}
    assert seen_params[1] is None


def test_guarded_http_get_strips_authorization_on_cross_origin_redirect():
    """GitHubAuth Authorization must not ride a safe cross-origin Location."""
    from tools.skills_hub import _guarded_http_get

    first = "https://api.github.com/repos/org/repo/contents/SKILL.md"
    second = "https://cdn.example/skills/SKILL.md"
    seen_headers = []

    def fake_get(url, **kwargs):
        seen_headers.append(kwargs.get("headers"))
        if url == first:
            return _FakeResponse(status_code=302, headers={"location": second})
        return _FakeResponse(status_code=200, text="# skill\n")

    with patch("tools.skills_hub.httpx.get", side_effect=fake_get):
        with patch("tools.skills_hub.is_safe_url", return_value=True):
            with patch("tools.skills_hub.check_website_access", return_value=None):
                resp = _guarded_http_get(
                    first,
                    headers={
                        "Accept": "application/vnd.github.v3+json",
                        "Authorization": "token ghp_secret",
                    },
                    timeout=5,
                )

    assert resp is not None
    assert resp.status_code == 200
    assert seen_headers[0]["Authorization"] == "token ghp_secret"
    assert "Authorization" not in (seen_headers[1] or {})
    assert seen_headers[1]["Accept"] == "application/vnd.github.v3+json"


def test_guarded_http_get_keeps_authorization_on_same_origin_redirect():
    from tools.skills_hub import _guarded_http_get

    first = "https://api.github.com/repos/org/repo"
    second = "https://api.github.com/repos/org/repo/"
    seen_headers = []

    def fake_get(url, **kwargs):
        seen_headers.append(kwargs.get("headers"))
        if url == first:
            return _FakeResponse(status_code=301, headers={"location": second})
        return _FakeResponse(status_code=200, json_data={"ok": True})

    with patch("tools.skills_hub.httpx.get", side_effect=fake_get):
        with patch("tools.skills_hub.is_safe_url", return_value=True):
            with patch("tools.skills_hub.check_website_access", return_value=None):
                resp = _guarded_http_get(
                    first,
                    headers={"Authorization": "token ghp_secret"},
                    timeout=5,
                )

    assert resp is not None
    assert seen_headers[0]["Authorization"] == "token ghp_secret"
    assert seen_headers[1]["Authorization"] == "token ghp_secret"


def test_guarded_http_get_reraises_decoding_error():
    """DecodingError must escape the helper so index Accept-Encoding retry works."""
    from tools.skills_hub import _guarded_http_get

    def fake_get(url, **kwargs):
        raise httpx.DecodingError("brotli: decoder process called with data")

    with patch("tools.skills_hub.httpx.get", side_effect=fake_get):
        with patch("tools.skills_hub.is_safe_url", return_value=True):
            with patch("tools.skills_hub.check_website_access", return_value=None):
                with pytest.raises(httpx.DecodingError):
                    _guarded_http_get("https://example.com/index.json", timeout=5)


def test_browse_sh_fetch_uses_guarded_get_for_skill_md_url():
    """Catalog-controlled skillMdUrl must go through _guarded_http_get."""
    from tools.skills_hub import BrowseShSource

    src = BrowseShSource()
    item = {
        "slug": "demo/skill",
        "name": "demo",
        "title": "Demo",
        "description": "d",
        "tags": [],
        "hostname": "example.com",
        "sourceUrl": "https://github.com/example/repo",
    }
    md_url = "https://cdn.example/skills/demo.md"

    with patch.object(src, "_fetch_catalog", return_value=[item]):
        with patch.object(src, "_resolve_skill_md_url", return_value=md_url):
            with patch(
                "tools.skills_hub._guarded_http_get",
                return_value=_FakeResponse(status_code=200, text="# skill\n"),
            ) as guarded:
                with patch("tools.skills_hub.httpx.get") as raw_get:
                    bundle = src.fetch("browse-sh/demo/skill")

    assert bundle is not None
    assert bundle.files["SKILL.md"] == "# skill\n"
    guarded.assert_called()
    assert guarded.call_args[0][0] == md_url
    raw_get.assert_not_called()


def test_browse_sh_fetch_returns_none_when_guard_blocks():
    from tools.skills_hub import BrowseShSource

    src = BrowseShSource()
    item = {
        "slug": "demo/skill",
        "name": "demo",
        "title": "Demo",
        "description": "d",
        "tags": [],
    }

    with patch.object(src, "_fetch_catalog", return_value=[item]):
        with patch.object(
            src,
            "_resolve_skill_md_url",
            return_value="https://evil.example/x",
        ):
            with patch("tools.skills_hub._guarded_http_get", return_value=None):
                assert src.fetch("browse-sh/demo/skill") is None


def test_clawhub_zip_download_uses_guarded_get():
    from tools.skills_hub import ClawHubSource

    src = ClawHubSource()
    with patch(
        "tools.skills_hub._guarded_http_get",
        return_value=None,
    ) as guarded:
        with patch("tools.skills_hub.httpx.get") as raw_get:
            files = src._download_zip("some-skill", "1.0.0")

    assert files == {}
    guarded.assert_called()
    raw_get.assert_not_called()
