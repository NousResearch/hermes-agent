"""Coverage tests for GitHubSource download / cache / auth helpers.

This file targets uncovered branches in ``tools/skills_hub.py``:

* ``GitHubAuth`` token resolution (``_resolve_token`` / ``_try_gh_cli`` /
  ``_try_github_app``) — subprocess + secret-scope are mocked; no real gh CLI
  or token is ever touched.
* GitHubSource download helpers (``_download_directory``,
  ``_download_directory_via_tree``, ``_download_directory_recursive``,
  ``_find_skill_in_repo_tree``) and file helpers (``_fetch_file_content``,
  ``_fetch_file_bytes``).
* The SSRF-guarded redirect chain (``_guarded_http_get`` /
  ``_ssrf_safe_http_get``) and ``_github_get`` retry/backoff.
* Groupings / cache / metadata helpers (``_get_skillsh_groupings``,
  ``_parse_skillsh_groupings``, ``_read_cache``, ``_write_cache``,
  ``_meta_to_dict``).

Everything is deterministic and offline.
"""

import json
import subprocess
import time

from unittest.mock import MagicMock, call, patch

import httpx

from tools.skills_hub import (
    GitHubAuth,
    GitHubSource,
    SkillMeta,
    _guarded_http_get,
    _ssrf_safe_http_get,
)


def _source(auth=None):
    auth = auth or MagicMock(spec=GitHubAuth)
    auth.get_headers.return_value = {}
    return GitHubSource(auth=auth)


# ---------------------------------------------------------------------------
# GitHubAuth — token resolution (PAT / gh CLI / GitHub App)
# ---------------------------------------------------------------------------


class TestGitHubAuthResolveToken:
    def test_cached_pat_token_short_circuits(self):
        auth = GitHubAuth()
        auth._cached_token = "pat-token"
        auth._cached_method = "pat"

        with patch("agent.secret_scope.get_secret") as get_secret:
            assert auth._resolve_token() == "pat-token"
        get_secret.assert_not_called()

    def test_cached_app_token_returned_while_valid(self):
        auth = GitHubAuth()
        auth._cached_token = "app-token"
        auth._cached_method = "github-app"
        auth._app_token_expiry = time.time() + 100

        with patch("agent.secret_scope.get_secret") as get_secret:
            assert auth._resolve_token() == "app-token"
        get_secret.assert_not_called()

    def test_expired_app_token_re_resolves(self):
        auth = GitHubAuth()
        auth._cached_token = "app-token"
        auth._cached_method = "github-app"
        auth._app_token_expiry = time.time() - 100  # already expired

        with patch("agent.secret_scope.get_secret", return_value=None), \
             patch("tools.skills_hub.subprocess.run", side_effect=FileNotFoundError), \
             patch("tools.skills_hub.httpx.post") as post:
            post.return_value = MagicMock(status_code=500)
            assert auth._resolve_token() is None
        assert auth._cached_method == "anonymous"

    def test_pat_from_github_token_env(self):
        auth = GitHubAuth()

        def _get_secret(name):
            return "env-token" if name in ("GITHUB_TOKEN", "GH_TOKEN") else None

        with patch("agent.secret_scope.get_secret", side_effect=_get_secret):
            assert auth._resolve_token() == "env-token"
        assert auth._cached_method == "pat"

    def test_pat_from_gh_token_env(self):
        auth = GitHubAuth()

        def _get_secret(name):
            return "gh-env-token" if name == "GH_TOKEN" else None

        with patch("agent.secret_scope.get_secret", side_effect=_get_secret):
            assert auth._resolve_token() == "gh-env-token"
        assert auth._cached_method == "pat"

    def test_all_methods_fail_becomes_anonymous(self):
        auth = GitHubAuth()

        def _get_secret(name):
            return None

        with patch("agent.secret_scope.get_secret", side_effect=_get_secret), \
             patch("tools.skills_hub.subprocess.run", side_effect=FileNotFoundError), \
             patch("tools.skills_hub.httpx.post") as post:
            post.return_value = MagicMock(status_code=500)
            assert auth._resolve_token() is None
        assert auth._cached_method == "anonymous"

    def test_resolve_token_uses_gh_cli_and_sets_method(self):
        auth = GitHubAuth()
        with patch("agent.secret_scope.get_secret", return_value=None), \
             patch("tools.skills_hub.subprocess.run", return_value=MagicMock(returncode=0, stdout="gh-token\n")):
            assert auth._resolve_token() == "gh-token"
        assert auth._cached_method == "gh-cli"
        assert auth.is_authenticated() is True
        assert auth.auth_method() == "gh-cli"

    def test_resolve_token_uses_github_app_when_env_and_cli_fail(self, tmp_path):
        key_file = tmp_path / "key.pem"
        key_file.write_text("x", encoding="utf-8")
        auth = GitHubAuth()

        def _get_secret(name):
            return {
                "GITHUB_TOKEN": None,
                "GH_TOKEN": None,
                "GITHUB_APP_ID": "1",
                "GITHUB_APP_PRIVATE_KEY_PATH": str(key_file),
                "GITHUB_APP_INSTALLATION_ID": "9",
            }.get(name)

        fake_jwt = MagicMock()
        fake_jwt.encode.return_value = "signed.jwt"
        resp = MagicMock(status_code=201, json=lambda: {"token": "app-token"})

        with patch("agent.secret_scope.get_secret", side_effect=_get_secret), \
             patch("tools.skills_hub.subprocess.run", side_effect=FileNotFoundError), \
             patch.dict("sys.modules", {"jwt": fake_jwt}), \
             patch("tools.skills_hub.httpx.post", return_value=resp):
            assert auth._resolve_token() == "app-token"

        assert auth._cached_method == "github-app"
        assert auth.is_authenticated() is True
        assert auth._app_token_expiry > time.time()


class TestGitHubAuthTryGhCli:
    def test_returns_token_on_success(self):
        auth = GitHubAuth()
        result = MagicMock(returncode=0, stdout="gh-token\n")
        with patch("tools.skills_hub.subprocess.run", return_value=result) as run:
            assert auth._try_gh_cli() == "gh-token"
        run.assert_called_once()
        assert run.call_args.args[0] == ["gh", "auth", "token"]

    def test_returns_none_on_nonzero_returncode(self):
        auth = GitHubAuth()
        result = MagicMock(returncode=1, stdout="")
        with patch("tools.skills_hub.subprocess.run", return_value=result):
            assert auth._try_gh_cli() is None

    def test_returns_none_on_filenotfound(self):
        auth = GitHubAuth()
        with patch("tools.skills_hub.subprocess.run", side_effect=FileNotFoundError):
            assert auth._try_gh_cli() is None

    def test_returns_none_on_timeout(self):
        auth = GitHubAuth()
        with patch("tools.skills_hub.subprocess.run", side_effect=subprocess.TimeoutExpired("gh auth token", 5)):
            assert auth._try_gh_cli() is None

    def test_returns_none_on_blank_stdout(self):
        auth = GitHubAuth()
        result = MagicMock(returncode=0, stdout="   \n")
        with patch("tools.skills_hub.subprocess.run", return_value=result):
            assert auth._try_gh_cli() is None


class TestGitHubAuthTryGithubApp:
    def test_returns_none_when_credentials_missing(self):
        auth = GitHubAuth()
        with patch("agent.secret_scope.get_secret", return_value=None):
            assert auth._try_github_app() is None

    def test_returns_none_on_pyjwt_missing(self):
        auth = GitHubAuth()

        def _get_secret(name):
            return {"GITHUB_APP_ID": "1", "GITHUB_APP_PRIVATE_KEY_PATH": "/tmp/k", "GITHUB_APP_INSTALLATION_ID": "2"}.get(name)

        with patch("agent.secret_scope.get_secret", side_effect=_get_secret), \
             patch.dict("sys.modules", {"jwt": None}):
            assert auth._try_github_app() is None

    def test_returns_none_when_key_file_missing(self, tmp_path):
        auth = GitHubAuth()

        def _get_secret(name):
            return {
                "GITHUB_APP_ID": "1",
                "GITHUB_APP_PRIVATE_KEY_PATH": str(tmp_path / "nope.pem"),
                "GITHUB_APP_INSTALLATION_ID": "2",
            }.get(name)

        fake_jwt = MagicMock()
        with patch("agent.secret_scope.get_secret", side_effect=_get_secret), \
             patch.dict("sys.modules", {"jwt": fake_jwt}):
            assert auth._try_github_app() is None
        fake_jwt.encode.assert_not_called()

    def test_returns_token_on_success(self, tmp_path):
        key_file = tmp_path / "key.pem"
        key_file.write_text("-----BEGIN RSA PRIVATE KEY-----\n", encoding="utf-8")
        auth = GitHubAuth()

        def _get_secret(name):
            return {
                "GITHUB_APP_ID": "123",
                "GITHUB_APP_PRIVATE_KEY_PATH": str(key_file),
                "GITHUB_APP_INSTALLATION_ID": "456",
            }.get(name)

        fake_jwt = MagicMock()
        fake_jwt.encode.return_value = "signed.jwt"
        resp = MagicMock(status_code=201, json=lambda: {"token": "app-token"})

        with patch("agent.secret_scope.get_secret", side_effect=_get_secret), \
             patch.dict("sys.modules", {"jwt": fake_jwt}), \
             patch("tools.skills_hub.httpx.post", return_value=resp) as post:
            assert auth._try_github_app() == "app-token"

        assert "api.github.com/app/installations/456/access_tokens" in post.call_args.args[0]
        headers = post.call_args.kwargs["headers"]
        assert headers["Authorization"] == "Bearer signed.jwt"

    def test_returns_none_on_non_201(self, tmp_path):
        key_file = tmp_path / "key.pem"
        key_file.write_text("x", encoding="utf-8")
        auth = GitHubAuth()

        def _get_secret(name):
            return {
                "GITHUB_APP_ID": "1",
                "GITHUB_APP_PRIVATE_KEY_PATH": str(key_file),
                "GITHUB_APP_INSTALLATION_ID": "2",
            }.get(name)

        fake_jwt = MagicMock()
        fake_jwt.encode.return_value = "signed.jwt"
        resp = MagicMock(status_code=400, json=lambda: {})
        with patch("agent.secret_scope.get_secret", side_effect=_get_secret), \
             patch.dict("sys.modules", {"jwt": fake_jwt}), \
             patch("tools.skills_hub.httpx.post", return_value=resp):
            assert auth._try_github_app() is None

    def test_returns_none_on_httpx_error(self, tmp_path):
        key_file = tmp_path / "key.pem"
        key_file.write_text("x", encoding="utf-8")
        auth = GitHubAuth()

        def _get_secret(name):
            return {
                "GITHUB_APP_ID": "1",
                "GITHUB_APP_PRIVATE_KEY_PATH": str(key_file),
                "GITHUB_APP_INSTALLATION_ID": "2",
            }.get(name)

        fake_jwt = MagicMock()
        fake_jwt.encode.return_value = "signed.jwt"
        with patch("agent.secret_scope.get_secret", side_effect=_get_secret), \
             patch.dict("sys.modules", {"jwt": fake_jwt}), \
             patch("tools.skills_hub.httpx.post", side_effect=httpx.HTTPError("boom")):
            assert auth._try_github_app() is None


class TestGitHubAuthPublicApi:
    def test_get_headers_without_token(self):
        auth = GitHubAuth()
        with patch("agent.secret_scope.get_secret", return_value=None), \
             patch("tools.skills_hub.subprocess.run", side_effect=FileNotFoundError), \
             patch("tools.skills_hub.httpx.post") as post:
            post.return_value = MagicMock(status_code=500)
            headers = auth.get_headers()
        assert headers == {"Accept": "application/vnd.github.v3+json"}
        assert "Authorization" not in headers

    def test_get_headers_with_token(self):
        auth = GitHubAuth()
        with patch("agent.secret_scope.get_secret", side_effect=lambda name: "tok" if name == "GITHUB_TOKEN" else None):
            headers = auth.get_headers()
        assert headers == {
            "Accept": "application/vnd.github.v3+json",
            "Authorization": "token tok",
        }

    def test_is_authenticated_true(self):
        auth = GitHubAuth()
        with patch("agent.secret_scope.get_secret", side_effect=lambda name: "tok" if name == "GITHUB_TOKEN" else None):
            assert auth.is_authenticated() is True

    def test_is_authenticated_false(self):
        auth = GitHubAuth()
        with patch("agent.secret_scope.get_secret", return_value=None), \
             patch("tools.skills_hub.subprocess.run", side_effect=FileNotFoundError), \
             patch("tools.skills_hub.httpx.post") as post:
            post.return_value = MagicMock(status_code=500)
            assert auth.is_authenticated() is False

    def test_auth_method_reports_pat(self):
        auth = GitHubAuth()
        with patch("agent.secret_scope.get_secret", side_effect=lambda name: "tok" if name == "GITHUB_TOKEN" else None):
            assert auth.auth_method() == "pat"

    def test_auth_method_reports_anonymous(self):
        auth = GitHubAuth()
        with patch("agent.secret_scope.get_secret", return_value=None), \
             patch("tools.skills_hub.subprocess.run", side_effect=FileNotFoundError), \
             patch("tools.skills_hub.httpx.post") as post:
            post.return_value = MagicMock(status_code=500)
            assert auth.auth_method() == "anonymous"


# ---------------------------------------------------------------------------
# GitHubSource — download helpers
# ---------------------------------------------------------------------------


class TestDownloadDirectoryViaTreeMissedBranches:
    def test_returns_empty_dict_when_path_absent(self):
        src = _source()
        src._get_repo_tree = MagicMock(return_value=("main", [{"type": "blob", "path": "other/file.txt"}]))
        src._fetch_file_content = MagicMock()

        assert src._download_directory_via_tree("owner/repo", "skills/missing") == {}
        src._fetch_file_content.assert_not_called()

    def test_returns_none_when_tree_unavailable(self):
        src = _source()
        src._get_repo_tree = MagicMock(return_value=None)
        src._fetch_file_content = MagicMock()

        assert src._download_directory_via_tree("owner/repo", "skills/skill") is None

    def test_skips_blob_when_fetch_fails(self):
        src = _source()
        src._get_repo_tree = MagicMock(return_value=(
            "main",
            [
                {"type": "blob", "path": "skills/skill/SKILL.md"},
                {"type": "blob", "path": "skills/skill/broken.md"},
            ],
        ))
        src._fetch_file_content = MagicMock(side_effect=lambda repo, path: None if path.endswith("broken.md") else "content")

        result = src._download_directory_via_tree("owner/repo", "skills/skill")

        assert result == {"SKILL.md": "content"}
        assert src._fetch_file_content.call_args_list == [
            call("owner/repo", "skills/skill/SKILL.md"),
            call("owner/repo", "skills/skill/broken.md"),
        ]

    def test_returns_none_when_only_non_blob_entries(self):
        src = _source()
        src._get_repo_tree = MagicMock(return_value=("main", [{"type": "tree", "path": "skills/skill/scripts"}]))
        src._fetch_file_content = MagicMock()

        assert src._download_directory_via_tree("owner/repo", "skills/skill") is None

    def test_skips_non_matching_paths_in_tree(self):
        src = _source()
        src._get_repo_tree = MagicMock(return_value=(
            "main",
            [
                {"type": "blob", "path": "skills/skill/SKILL.md"},
                {"type": "blob", "path": "other/skip.md"},
            ],
        ))
        src._fetch_file_content = MagicMock(return_value="content")

        result = src._download_directory_via_tree("owner/repo", "skills/skill")

        assert result == {"SKILL.md": "content"}
        # the non-matching blob's path is skipped (continue) and never fetched
        src._fetch_file_content.assert_called_once_with("owner/repo", "skills/skill/SKILL.md")

    def test_download_directory_falls_back_on_none(self):
        src = _source()
        src._download_directory_via_tree = MagicMock(return_value=None)
        src._download_directory_recursive = MagicMock(return_value={"SKILL.md": "# ok"})

        assert src._download_directory("owner/repo", "skills/skill") == {"SKILL.md": "# ok"}
        src._download_directory_recursive.assert_called_once_with("owner/repo", "skills/skill")

    def test_download_directory_returns_empty_without_fallback(self):
        src = _source()
        src._download_directory_via_tree = MagicMock(return_value={})
        src._download_directory_recursive = MagicMock(return_value={"SKILL.md": "# ok"})

        assert src._download_directory("owner/repo", "skills/skill") == {}
        src._download_directory_recursive.assert_not_called()


class TestDownloadDirectoryRecursiveMissedBranches:
    def test_returns_empty_when_github_get_none(self):
        src = _source()
        src._github_get = MagicMock(return_value=None)
        assert src._download_directory_recursive("owner/repo", "skill") == {}

    def test_returns_empty_on_non_200(self):
        src = _source()
        src._github_get = MagicMock(return_value=MagicMock(status_code=404))
        assert src._download_directory_recursive("owner/repo", "skill") == {}

    def test_returns_empty_when_entries_not_list(self):
        src = _source()
        resp = MagicMock(status_code=200)
        resp.json.return_value = {"not": "a list"}
        src._github_get = MagicMock(return_value=resp)
        assert src._download_directory_recursive("owner/repo", "skill") == {}

    def test_recurses_into_subdirectories(self):
        src = _source()
        root_resp = MagicMock(status_code=200)
        root_resp.json.return_value = [
            {"name": "SKILL.md", "type": "file", "path": "skill/SKILL.md"},
            {"name": "broken.md", "type": "file", "path": "skill/broken.md"},
            {"name": "scripts", "type": "dir", "path": "skill/scripts"},
            {"name": "empty", "type": "dir", "path": "skill/empty"},
        ]
        sub_resp = MagicMock(status_code=200)
        sub_resp.json.return_value = [
            {"name": "run.py", "type": "file", "path": "skill/scripts/run.py"},
        ]
        empty_resp = MagicMock(status_code=200)
        empty_resp.json.return_value = []

        src._github_get = MagicMock(side_effect=[root_resp, sub_resp, empty_resp])
        src._fetch_file_content = MagicMock(
            side_effect=lambda repo, path: None if "broken" in path else "content"
        )

        result = src._download_directory_recursive("owner/repo", "skill")

        assert result == {"SKILL.md": "content", "scripts/run.py": "content"}
        # broken.md skipped via the fetch-returns-None branch; the empty
        # subdirectory is fetched (3 calls) but contributes no files.
        assert src._github_get.call_count == 3


class TestFindSkillInRepoTreeMissedBranches:
    def test_finds_top_level_skill_dir(self):
        src = _source()
        src._get_repo_tree = MagicMock(return_value=(
            "main",
            [{"type": "blob", "path": "my-skill/SKILL.md"}, {"type": "blob", "path": "README.md"}],
        ))
        assert src._find_skill_in_repo_tree("owner/repo", "my-skill") == "owner/repo/my-skill"

    def test_returns_none_when_no_match(self):
        src = _source()
        src._get_repo_tree = MagicMock(return_value=(
            "main",
            [{"type": "blob", "path": "other-skill/SKILL.md"}, {"type": "blob", "path": "README.md"}],
        ))
        assert src._find_skill_in_repo_tree("owner/repo", "missing") is None


# ---------------------------------------------------------------------------
# GitHubSource — file fetch helpers
# ---------------------------------------------------------------------------


class TestFetchFileContentMissedBranches:
    def test_decodes_utf8(self):
        src = _source()
        src._fetch_file_bytes = MagicMock(return_value=b"hello world")
        assert src._fetch_file_content("owner/repo", "SKILL.md") == "hello world"

    def test_returns_none_when_bytes_none(self):
        src = _source()
        src._fetch_file_bytes = MagicMock(return_value=None)
        assert src._fetch_file_content("owner/repo", "SKILL.md") is None

    def test_returns_none_on_unicode_decode_error(self):
        src = _source()
        src._fetch_file_bytes = MagicMock(return_value=b"\xff\xfe\x00")
        assert src._fetch_file_content("owner/repo", "SKILL.md") is None


class TestFetchFileBytesMissedBranches:
    def test_returns_content_on_200_with_raw_accept(self):
        src = _source()
        resp = MagicMock(status_code=200, content=b"raw-bytes")
        src._github_get = MagicMock(return_value=resp)

        assert src._fetch_file_bytes("owner/repo", "skill/SKILL.md") == b"raw-bytes"
        assert src._github_get.call_args.kwargs["headers"]["Accept"] == "application/vnd.github.v3.raw"

    def test_passes_ref_param(self):
        src = _source()
        resp = MagicMock(status_code=200, content=b"raw-bytes")
        src._github_get = MagicMock(return_value=resp)

        src._fetch_file_bytes("owner/repo", "skill/SKILL.md", ref="abc123")
        assert src._github_get.call_args.kwargs["params"] == {"ref": "abc123"}

    def test_returns_none_on_non_200(self):
        src = _source()
        src._github_get = MagicMock(return_value=MagicMock(status_code=404))
        assert src._fetch_file_bytes("owner/repo", "skill/SKILL.md") is None


# ---------------------------------------------------------------------------
# _github_get — retry / backoff branches
# ---------------------------------------------------------------------------


class TestGitHubGetRetryBranches:
    def test_returns_200_response(self):
        src = _source()
        resp = MagicMock(status_code=200)
        src.auth.get_headers.return_value = {"Accept": "application/vnd.github.v3+json"}
        with patch("tools.skills_hub.httpx.get", return_value=resp) as get:
            assert src._github_get("https://api.github.com/repos/x") is resp
        assert get.call_args.kwargs["follow_redirects"] is True

    def test_returns_none_when_all_attempts_raise(self):
        src = _source()
        src.auth.get_headers.return_value = {}
        with patch("tools.skills_hub.httpx.get", side_effect=httpx.ConnectError("boom")), \
             patch("tools.skills_hub.time.sleep"):
            assert src._github_get("https://api.github.com/repos/x", max_retries=3) is None

    def test_retries_on_5xx(self):
        src = _source()
        src.auth.get_headers.return_value = {}
        resp5xx = MagicMock(status_code=500)
        resp200 = MagicMock(status_code=200)
        with patch("tools.skills_hub.httpx.get", side_effect=[resp5xx, resp200]) as get, \
             patch("tools.skills_hub.time.sleep") as _sleep:
            assert src._github_get("https://api.github.com/repos/x", max_retries=3) is resp200
        assert get.call_count == 2
        _sleep.assert_called_once_with(1.0)

    def test_retries_on_429_with_retry_after(self):
        src = _source()
        src.auth.get_headers.return_value = {}
        resp429 = MagicMock(status_code=429)
        resp429.headers = {"X-RateLimit-Remaining": "0", "Retry-After": "2"}
        resp200 = MagicMock(status_code=200)
        with patch("tools.skills_hub.httpx.get", side_effect=[resp429, resp200]) as get, \
             patch("tools.skills_hub.time.sleep") as _sleep:
            assert src._github_get("https://api.github.com/repos/x", max_retries=3) is resp200
        assert get.call_count == 2
        _sleep.assert_called_once_with(2.0)

    def test_retries_on_429_using_reset_header(self):
        src = _source()
        src.auth.get_headers.return_value = {}
        reset = str(int(time.time()) + 60)
        resp429 = MagicMock(status_code=429)
        resp429.headers = {"X-RateLimit-Remaining": "0", "X-RateLimit-Reset": reset}
        resp200 = MagicMock(status_code=200)
        with patch("tools.skills_hub.httpx.get", side_effect=[resp429, resp200]) as get, \
             patch("tools.skills_hub.time.sleep") as _sleep:
            assert src._github_get("https://api.github.com/repos/x", max_retries=3) is resp200
        assert get.call_count == 2
        # waits the Reset-derived delta (~60s, capped by the <=60 check)
        _sleep.assert_called_once()
        assert 0 < _sleep.call_args.args[0] <= 60

    def test_returns_last_resp_when_no_attempts(self):
        src = _source()
        src.auth.get_headers.return_value = {}
        with patch("tools.skills_hub.httpx.get") as get:
            assert src._github_get("https://api.github.com/repos/x", max_retries=0) is None
        get.assert_not_called()

    def test_rate_limit_exhausted_without_retries_flags(self):
        src = _source()
        src.auth.get_headers.return_value = {}
        src._check_rate_limit_response = MagicMock()
        resp403 = MagicMock(status_code=403)
        resp403.headers = {"X-RateLimit-Remaining": "0"}
        with patch("tools.skills_hub.httpx.get", return_value=resp403):
            assert src._github_get("https://api.github.com/repos/x", max_retries=3) is resp403
        src._check_rate_limit_response.assert_called_once_with(resp403)


# ---------------------------------------------------------------------------
# SSRF-guarded redirect chain
# ---------------------------------------------------------------------------


class TestGuardedHttpGet:
    def test_returns_response_on_success(self):
        resp = MagicMock(status_code=200)
        with patch("tools.skills_hub.is_safe_url", return_value=True), \
             patch("tools.skills_hub.check_website_access", return_value=None), \
             patch("tools.skills_hub._ssrf_safe_http_get", return_value=resp) as get:
            assert _guarded_http_get("https://example.com/SKILL.md") is resp
        get.assert_called_once_with("https://example.com/SKILL.md", timeout=20)

    def test_returns_none_when_url_unsafe(self):
        with patch("tools.skills_hub.is_safe_url", return_value=False), \
             patch("tools.skills_hub.check_website_access", return_value=None), \
             patch("tools.skills_hub._ssrf_safe_http_get") as get:
            assert _guarded_http_get("http://127.0.0.1/SKILL.md") is None
        get.assert_not_called()

    def test_returns_none_when_access_blocked(self):
        with patch("tools.skills_hub.is_safe_url", return_value=True), \
             patch("tools.skills_hub.check_website_access", return_value={"host": "x", "rule": "blocked"}), \
             patch("tools.skills_hub._ssrf_safe_http_get") as get:
            assert _guarded_http_get("https://example.com/SKILL.md") is None
        get.assert_not_called()

    def test_returns_none_on_transport_error(self):
        with patch("tools.skills_hub.is_safe_url", return_value=True), \
             patch("tools.skills_hub.check_website_access", return_value=None), \
             patch("tools.skills_hub._ssrf_safe_http_get", side_effect=httpx.ConnectError("down")):
            assert _guarded_http_get("https://example.com/SKILL.md") is None

    def test_follows_redirect_to_next_url(self):
        redirect = MagicMock(status_code=302)
        redirect.headers = {"location": "/next/SKILL.md"}
        ok = MagicMock(status_code=200)
        with patch("tools.skills_hub.is_safe_url", return_value=True), \
             patch("tools.skills_hub.check_website_access", return_value=None), \
             patch("tools.skills_hub._ssrf_safe_http_get", side_effect=[redirect, ok]) as get:
            assert _guarded_http_get("https://example.com/SKILL.md") is ok
        assert get.call_count == 2
        assert get.call_args_list[1].args[0] == "https://example.com/next/SKILL.md"

    def test_returns_none_when_redirect_has_no_location(self):
        redirect = MagicMock(status_code=301)
        redirect.headers = {}
        with patch("tools.skills_hub.is_safe_url", return_value=True), \
             patch("tools.skills_hub.check_website_access", return_value=None), \
             patch("tools.skills_hub._ssrf_safe_http_get", return_value=redirect):
            assert _guarded_http_get("https://example.com/SKILL.md") is None

    def test_returns_none_when_redirect_loop_exceeds_limit(self):
        redirect = MagicMock(status_code=302)
        redirect.headers = {"location": "/same"}
        with patch("tools.skills_hub.is_safe_url", return_value=True), \
             patch("tools.skills_hub.check_website_access", return_value=None), \
             patch("tools.skills_hub._ssrf_safe_http_get", return_value=redirect) as get:
            assert _guarded_http_get("https://example.com/SKILL.md") is None
        assert get.call_count == 6  # _MAX_SKILL_FETCH_REDIRECTS(5) + 1


class TestSsrfSafeHttpGet:
    def test_returns_client_get_response(self):
        resp = MagicMock(status_code=200)
        client = MagicMock()
        client.get.return_value = resp
        cm = MagicMock()
        cm.__enter__.return_value = client

        with patch("tools.url_safety.create_ssrf_safe_client", return_value=cm) as create:
            assert _ssrf_safe_http_get("https://example.com/SKILL.md") is resp
        create.assert_called_once_with(timeout=20, follow_redirects=False)
        client.get.assert_called_once_with("https://example.com/SKILL.md")


# ---------------------------------------------------------------------------
# Groupings / cache / metadata helpers
# ---------------------------------------------------------------------------


class TestGetSkillshGroupings:
    def test_returns_cached_map(self):
        src = _source()
        src._skillsh_groupings["repo"] = {"s": "T"}
        src._fetch_file_content = MagicMock()
        assert src._get_skillsh_groupings("repo") == {"s": "T"}
        src._fetch_file_content.assert_not_called()

    def test_returns_cached_none(self):
        src = _source()
        src._skillsh_groupings["repo"] = None
        src._fetch_file_content = MagicMock()
        assert src._get_skillsh_groupings("repo") is None
        src._fetch_file_content.assert_not_called()

    def test_fetches_and_parses_when_uncached(self):
        src = _source()
        content = json.dumps({"groupings": [{"title": "Inference", "skills": ["dynamo"]}]})
        src._fetch_file_content = MagicMock(return_value=content)
        result = src._get_skillsh_groupings("repo")
        assert result == {"dynamo": "Inference"}
        assert src._skillsh_groupings["repo"] == {"dynamo": "Inference"}
        src._fetch_file_content.assert_called_once_with("repo", "skills.sh.json")

    def test_caches_none_when_content_missing(self):
        src = _source()
        src._fetch_file_content = MagicMock(return_value=None)
        assert src._get_skillsh_groupings("repo") is None
        assert src._skillsh_groupings["repo"] is None


class TestParseSkillshGroupingsMissedBranches:
    def test_returns_none_on_invalid_json(self):
        assert GitHubSource._parse_skillsh_groupings("{not json") is None

    def test_returns_none_on_non_string(self):
        assert GitHubSource._parse_skillsh_groupings(123) is None

    def test_returns_none_when_not_dict(self):
        assert GitHubSource._parse_skillsh_groupings("[1, 2]") is None

    def test_returns_none_when_groupings_not_list(self):
        assert GitHubSource._parse_skillsh_groupings(json.dumps({"groupings": "nope"})) is None

    def test_skips_non_dict_group(self):
        content = json.dumps({"groupings": [["not", "a", "dict"]]})
        assert GitHubSource._parse_skillsh_groupings(content) == {}

    def test_skips_group_with_bad_fields(self):
        content = json.dumps({"groupings": [{"title": 5, "skills": ["a"]}, {"title": "ok", "skills": "notlist"}]})
        assert GitHubSource._parse_skillsh_groupings(content) == {}

    def test_skips_non_string_members(self):
        content = json.dumps({"groupings": [{"title": "T", "skills": ["good", 7, ""]}]})
        assert GitHubSource._parse_skillsh_groupings(content) == {"good": "T"}

    def test_first_grouping_wins_on_duplicate(self):
        content = json.dumps({
            "groupings": [
                {"title": "First", "skills": ["dup"]},
                {"title": "Second", "skills": ["dup"]},
            ]
        })
        assert GitHubSource._parse_skillsh_groupings(content) == {"dup": "First"}


class TestReadCache:
    def test_returns_none_when_file_missing(self, tmp_path):
        src = _source()
        with patch("tools.skills_hub._index_cache_dir", return_value=tmp_path):
            assert src._read_cache("nope") is None

    def test_returns_data_when_fresh(self, tmp_path):
        (tmp_path / "key.json").write_text(json.dumps([1, 2, 3]), encoding="utf-8")
        src = _source()
        with patch("tools.skills_hub._index_cache_dir", return_value=tmp_path):
            assert src._read_cache("key") == [1, 2, 3]

    def test_returns_none_when_expired(self, tmp_path):
        cache_file = tmp_path / "key.json"
        cache_file.write_text(json.dumps([1]), encoding="utf-8")
        past = time.time() - 10_000
        import os
        os.utime(cache_file, (past, past))
        src = _source()
        with patch("tools.skills_hub._index_cache_dir", return_value=tmp_path):
            assert src._read_cache("key") is None

    def test_returns_none_on_invalid_json(self, tmp_path):
        (tmp_path / "key.json").write_text("not json", encoding="utf-8")
        src = _source()
        with patch("tools.skills_hub._index_cache_dir", return_value=tmp_path):
            assert src._read_cache("key") is None

    def test_returns_none_on_oserror_reading_directory(self, tmp_path):
        (tmp_path / "key.json").mkdir()  # directory where a file is expected
        src = _source()
        with patch("tools.skills_hub._index_cache_dir", return_value=tmp_path):
            assert src._read_cache("key") is None


class TestWriteCache:
    def test_writes_json_to_cache_file(self, tmp_path):
        src = _source()
        with patch("tools.skills_hub._index_cache_dir", return_value=tmp_path):
            src._write_cache("key", [1, 2])
        assert (tmp_path / "key.json").read_text(encoding="utf-8") == json.dumps([1, 2])

    def test_swallows_oserror(self, tmp_path):
        (tmp_path / "key.json").mkdir()  # write_text on a directory raises OSError
        src = _source()
        with patch("tools.skills_hub._index_cache_dir", return_value=tmp_path):
            src._write_cache("key", [1])  # must not raise


class TestMetaToDict:
    def test_serializes_all_fields(self):
        meta = SkillMeta(
            name="n",
            description="d",
            source="github",
            identifier="owner/repo/skill",
            trust_level="community",
            repo="owner/repo",
            path="skill",
            tags=["a", "b"],
            extra={"category": "X"},
        )
        assert GitHubSource._meta_to_dict(meta) == {
            "name": "n",
            "description": "d",
            "source": "github",
            "identifier": "owner/repo/skill",
            "trust_level": "community",
            "repo": "owner/repo",
            "path": "skill",
            "tags": ["a", "b"],
            "extra": {"category": "X"},
        }
