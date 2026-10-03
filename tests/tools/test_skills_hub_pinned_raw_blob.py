"""Pinned skill-tree blobs use the raw endpoint without weakening retry behavior."""
from unittest.mock import MagicMock

import httpx

from tools.skills_hub_github import GitHubAuth, GitHubSource, _report_tree_fetch_progress


def test_pinned_blob_uses_raw_endpoint(monkeypatch):
    source = GitHubSource(auth=MagicMock(spec=GitHubAuth))
    seen = []
    monkeypatch.setattr("tools.skills_hub_github.httpx.get", lambda url, **kwargs: seen.append(url) or MagicMock(status_code=200, content=b"ok"))
    assert source._fetch_file_bytes("owner/repo", "skill/file.txt", ref="deadbeef") == b"ok"
    assert seen == ["https://raw.githubusercontent.com/owner/repo/deadbeef/skill/file.txt"]


def test_unpinned_blob_stays_on_contents_api(monkeypatch):
    source = GitHubSource(auth=MagicMock(spec=GitHubAuth))
    seen = []
    monkeypatch.setattr("tools.skills_hub_github.httpx.get", lambda url, **kwargs: seen.append(url) or MagicMock(status_code=200, content=b"ok"))
    assert source._fetch_file_bytes("owner/repo", "skill/file.txt") == b"ok"
    assert len(seen) == 1 and "/contents/skill/file.txt" in seen[0]


def test_pinned_blob_retries_transient_error(monkeypatch):
    source = GitHubSource(auth=MagicMock(spec=GitHubAuth))
    source.auth.get_headers.return_value = {}
    attempts = []
    def get(url, **kwargs):
        attempts.append(url)
        if len(attempts) < 3:
            raise httpx.ConnectError("transient")
        return MagicMock(status_code=200, content=b"ok")
    monkeypatch.setattr("tools.skills_hub_github.httpx.get", get)
    monkeypatch.setattr("tools.skills_hub_github.time.sleep", lambda *_args: None)
    assert source._fetch_file_bytes("owner/repo", "skill/file.txt", ref="deadbeef") == b"ok"
    assert len(attempts) == 3


def test_tree_progress_is_throttled_without_tty(monkeypatch, capsys):
    monkeypatch.setattr("tools.skills_hub_github._stderr_is_tty", lambda: False)
    _report_tree_fetch_progress(24, 30)
    _report_tree_fetch_progress(25, 30)
    _report_tree_fetch_progress(30, 30)
    assert capsys.readouterr().err.splitlines() == ["  fetched 25/30", "  fetched 30/30"]


def test_tree_progress_rewrites_when_stderr_is_tty(monkeypatch, capsys):
    monkeypatch.setattr("tools.skills_hub_github._stderr_is_tty", lambda: True)
    _report_tree_fetch_progress(1, 2)
    _report_tree_fetch_progress(2, 2)
    assert capsys.readouterr().err == "\r  fetched 1/2\r  fetched 2/2\n"
