"""Regression: ClawHub ZIP download must use SSRF-guarded HTTP."""

from unittest.mock import patch

from tools.skills_hub import ClawHubSource


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
