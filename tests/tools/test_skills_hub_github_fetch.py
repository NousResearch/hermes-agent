"""GitHubSource file downloads: which host serves a skill's files."""

from unittest.mock import MagicMock, patch

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


def test_fetch_keeps_a_skill_that_links_a_support_directory():
    """A SKILL.md pointing at ``references/clues/`` (a directory) is prose, not a symlink
    escape: the bundle installs with the directory's blobs, and only a linked SYMLINK
    still rejects the whole bundle."""
    md = (
        "---\nname: demo\ndescription: demo\n---\n\n"
        "Land-cover rules live in `references/clues/` and `references/clues/global.md`.\n"
    )
    tree_entries = [
        {"path": "skills/demo/references/clues", "type": "tree", "mode": "040000"},
        {"path": "skills/demo/references/clues/global.md", "type": "blob", "mode": "100644"},
    ]
    source = GitHubSource(auth=MagicMock(spec=GitHubAuth))
    with patch.object(source, "_fetch_file_content", return_value=md), \
         patch.object(source, "_get_repo_tree", return_value=("main", tree_entries)), \
         patch.object(source, "_fetch_file_bytes", return_value=b"# clues"):
        bundle = source.fetch("owner/repo/skills/demo")
    assert bundle is not None
    assert bundle.files["references/clues/global.md"] == b"# clues"

    tree_entries[0] = {"path": "skills/demo/references/clues", "type": "blob", "mode": "120000"}
    with patch.object(source, "_fetch_file_content", return_value=md), \
         patch.object(source, "_get_repo_tree", return_value=("main", tree_entries)), \
         patch.object(source, "_fetch_file_bytes", return_value=b"# clues"):
        assert source.fetch("owner/repo/skills/demo") is None


def test_tree_less_fetch_rejects_a_linked_directory_but_keeps_json_files(monkeypatch):
    """Without a repo tree each linked path is fetched through the Contents API. A directory answers
    with an ``application/json`` listing, which must reject the fetch; a real ``.json`` support file
    answers raw and is kept, whatever its content looks like."""
    import tools.skills_hub as hub

    md = "---\nname: demo\ndescription: demo\n---\n\nSee `references/clues/` and `references/data.json`.\n"
    raw_ct = {"content-type": "application/vnd.github.v3.raw; charset=utf-8"}
    listing = (200, b'[{"name": "a.md", "type": "file"}]', {"content-type": "application/json; charset=utf-8"})
    data_json = (200, b'[{"type": "object"}]', raw_ct)

    def serve(routes):
        def fake_get(self, url, **_kw):
            status, body, headers = routes[url.rsplit("/contents/", 1)[1]]
            return httpx.Response(status, content=body, headers=headers)
        return fake_get

    monkeypatch.setattr(hub, "_skills_hub_http_get", lambda *a, **k: httpx.Response(404))
    source = GitHubSource(auth=MagicMock(spec=GitHubAuth))
    monkeypatch.setattr(source, "_get_repo_tree", lambda _repo: None)
    monkeypatch.setattr(source, "_fetch_file_content", lambda *a, **k: md)
    monkeypatch.setattr(GitHubSource, "_github_get", serve({
        "skills/demo/references/clues": listing, "skills/demo/references/data.json": data_json}))
    assert source.fetch("owner/repo/skills/demo") is None

    # The same source (update checks reuse one) must not carry that verdict to another repo.
    monkeypatch.setattr(GitHubSource, "_github_get", serve({
        "skills/demo/references/clues": (200, b"# a file here", raw_ct),
        "skills/demo/references/data.json": data_json}))
    assert source.fetch("other/repo/skills/demo") is not None

    md_json_only = md.replace("`references/clues/` and ", "")
    source = GitHubSource(auth=MagicMock(spec=GitHubAuth))
    monkeypatch.setattr(source, "_get_repo_tree", lambda _repo: None)
    monkeypatch.setattr(source, "_fetch_file_content", lambda *a, **k: md_json_only)
    bundle = source.fetch("owner/repo/skills/demo")
    assert bundle is not None and bundle.files["references/data.json"] == b'[{"type": "object"}]'
