"""Regression: ClawHub ZIP download must use SSRF-guarded HTTP."""

from unittest.mock import patch

import httpx

from tools.skills_hub import ClawHubSource, _guarded_http_get


def test_download_zip_uses_guarded_http_get():
    src = ClawHubSource()
    with patch("tools.skills_hub._guarded_http_get", return_value=None) as mock_get:
        files = src._download_zip("demo-skill", "1.0.0")
    assert files == {}
    assert mock_get.call_count == 1
    called_url = mock_get.call_args.args[0]
    assert "/download" in called_url
    assert "slug=demo-skill" in called_url
    assert "version=1.0.0" in called_url


def test_download_zip_returns_empty_when_ssrf_blocked():
    src = ClawHubSource()
    with patch("tools.skills_hub._guarded_http_get", return_value=None):
        assert src._download_zip("evil", "9.9.9") == {}


def test_guarded_http_get_blocks_private_redirect_before_safe_http_get():
    public_url = "https://clawhub.example/download?slug=demo-skill&version=1.0.0"
    private_url = "http://127.0.0.1:8080/admin"
    redirect = httpx.Response(
        302,
        headers={"location": private_url},
        request=httpx.Request("GET", public_url),
    )

    with (
        patch("tools.skills_hub.is_safe_url", side_effect=lambda url: url == public_url),
        patch("tools.skills_hub.check_website_access", return_value=None),
        patch("tools.skills_hub._ssrf_safe_http_get", return_value=redirect) as mock_safe_get,
    ):
        assert _guarded_http_get(public_url) is None

    mock_safe_get.assert_called_once_with(
        public_url, timeout=20, headers=None, params=None
    )
