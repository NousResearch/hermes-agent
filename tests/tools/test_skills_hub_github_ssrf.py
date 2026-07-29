"""SSRF + auth-strip coverage for GitHub Skills Hub API fetches.

Salvages the incomplete open #63920 surface that still leaves
``GitHubSource._get_repo_tree`` / ``_github_get`` on raw ``httpx.get``
with ``follow_redirects=True`` on ``upstream/main``.
"""

from unittest.mock import MagicMock, patch

import httpx
import pytest


def _resp(status: int, *, location: str | None = None, json_data=None):
    r = MagicMock(spec=httpx.Response)
    r.status_code = status
    r.headers = {"location": location} if location else {}
    r.json.return_value = json_data or {}
    return r


class TestGuardedHttpGetAuthStrip:
    def test_strips_authorization_on_cross_origin_redirect(self):
        from tools.skills_hub import _guarded_http_get

        seen = []

        def fake_ssrf_get(url, *, timeout=20, headers=None, params=None):
            seen.append({"url": url, "headers": dict(headers or {})})
            if len(seen) == 1:
                return _resp(302, location="https://cdn.example.com/blob")
            return _resp(200, json_data={"ok": True})

        with patch("tools.skills_hub.is_safe_url", return_value=True), patch(
            "tools.skills_hub.check_website_access", return_value=None
        ), patch("tools.skills_hub._ssrf_safe_http_get", side_effect=fake_ssrf_get):
            resp = _guarded_http_get(
                "https://api.github.com/repos/org/repo",
                headers={
                    "Accept": "application/vnd.github.v3+json",
                    "Authorization": "token ghp_secret",
                },
                timeout=15,
            )

        assert resp is not None
        assert resp.status_code == 200
        assert seen[0]["headers"].get("Authorization") == "token ghp_secret"
        assert "Authorization" not in seen[1]["headers"]
        assert seen[1]["headers"]["Accept"] == "application/vnd.github.v3+json"

    def test_blocks_private_redirect_target(self):
        from tools.skills_hub import _guarded_http_get

        def fake_ssrf_get(url, *, timeout=20, headers=None, params=None):
            return _resp(302, location="http://169.254.169.254/latest/meta-data")

        def safe(url: str) -> bool:
            return "169.254" not in url

        with patch("tools.skills_hub.is_safe_url", side_effect=safe), patch(
            "tools.skills_hub.check_website_access", return_value=None
        ), patch("tools.skills_hub._ssrf_safe_http_get", side_effect=fake_ssrf_get):
            resp = _guarded_http_get(
                "https://api.github.com/repos/org/repo",
                headers={"Authorization": "token ghp_secret"},
            )

        assert resp is None


class TestGetRepoTreeUsesGuardedFetch:
    def test_get_repo_tree_does_not_call_raw_httpx_get(self):
        from tools.skills_hub import GitHubAuth, GitHubSource

        src = GitHubSource(auth=GitHubAuth())
        calls = {"n": 0}

        def fake_guarded(url, **kwargs):
            calls["n"] += 1
            if "git/trees" in url:
                return _resp(
                    200,
                    json_data={
                        "sha": "abc",
                        "truncated": False,
                        "tree": [{"path": "SKILL.md", "type": "blob"}],
                    },
                )
            return _resp(200, json_data={"default_branch": "main"})

        with patch("tools.skills_hub._guarded_http_get", side_effect=fake_guarded), patch(
            "tools.skills_hub.httpx.get"
        ) as raw_get:
            result = src._get_repo_tree("org/repo")

        assert result is not None
        assert calls["n"] == 2
        raw_get.assert_not_called()
