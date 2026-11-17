"""Hub update check: a GitHub skill whose own directory is unchanged skips the download even when
its multi-skill repo moved (subtree-sha comparison in ``GitHubSource.unchanged_since``)."""

from unittest.mock import MagicMock

import pytest

from tools.skills_hub_github import GitHubSource
from tools.skills_hub_install import check_for_skill_updates


def _multi_skill_repo(trees: dict, calls: dict) -> GitHubSource:
    """Real ``GitHubSource`` over a stubbed multi-skill repo. ``trees`` maps a root tree sha to the
    ``demo-skill`` subtree sha it holds (None = no such directory); the FIRST key is the default
    branch now. Tree-API and file GETs are counted in ``calls``."""
    src = GitHubSource(auth=MagicMock())

    def _entries(skill_sha):
        if skill_sha is None:
            return [{"path": "other-skill", "type": "tree", "sha": "o" * 40}]
        return [{"path": "demo-skill", "type": "tree", "sha": skill_sha},
                {"path": "other-skill", "type": "tree", "sha": "o" * 40},
                *({"path": f"demo-skill/{p}", "type": "blob", "sha": f"sha-{p}", "size": 3}
                  for p in ("SKILL.md", "scripts/run.sh", "references/notes.md"))]

    head = next(iter(trees))
    api = {"/repos/owner/repo": {"default_branch": "main"},
           "/repos/owner/repo/git/trees/main": {"sha": head, "tree": _entries(trees[head])}}
    api.update({f"/repos/owner/repo/git/trees/{root}": {"sha": root, "tree": _entries(sub)}
                for root, sub in trees.items()})

    def _json(url, **kw):
        path = url.split("api.github.com", 1)[1]
        if "/git/trees/" in path:
            calls.setdefault("trees", []).append(path.rsplit("/", 1)[1])
        return api.get(path)
    src._github_json = _json

    def _file(repo, path, **kw):
        calls["files"] = calls.get("files", 0) + 1
        return "---\nname: demo-skill\n---\nSee scripts/run.sh and references/notes.md\n"
    src._fetch_file_content = _file
    src._fetch_file_bytes = lambda repo, path, **kw: _file(repo, path, **kw).encode()
    return src


@pytest.mark.parametrize("trees, identifier, expected_status, expected_gets", [
    # a sibling skill changed: root moved, demo-skill's subtree did not -> no download
    ({"b" * 40: "s" * 40, "a" * 40: "s" * 40}, "owner/repo/demo-skill", "up_to_date", 0),
    # demo-skill itself changed -> full fetch decides
    ({"b" * 40: "t" * 40, "a" * 40: "s" * 40}, "owner/repo/demo-skill", "update_available", 3),
    # recorded tree no longer retrievable from the API -> full fetch
    ({"b" * 40: "s" * 40}, "owner/repo/demo-skill", "update_available", 3),
    # skill directory absent from the recorded tree -> full fetch
    ({"b" * 40: "s" * 40, "a" * 40: None}, "owner/repo/demo-skill", "update_available", 3),
])
def test_unchanged_skill_subtree_skips_download_when_repo_moved(
        tmp_path, monkeypatch, trees, identifier, expected_status, expected_gets):
    """``source_revision`` is the whole repo's tree sha, so a commit to ANY other path of a
    multi-skill repo moved it and every installed skill from that repo was re-downloaded on every
    update check (all 278 GitHub skills on one install, ~10 min per check). The GitHub source now
    compares the skill directory's own subtree sha at the recorded and the current tree."""
    import tools.skills_hub as hub
    (tmp_path / "skills" / "demo-skill").mkdir(parents=True)
    monkeypatch.setattr(hub, "SKILLS_DIR", tmp_path / "skills")
    lock = MagicMock()
    lock.list_installed.return_value = [{
        "name": "demo-skill", "source": "github", "identifier": identifier,
        "content_hash": "installed-hash", "install_path": "demo-skill",
        "metadata": {"source_revision": "a" * 40},
    }]
    calls: dict = {}
    source = _multi_skill_repo(trees, calls)

    results = check_for_skill_updates(lock=lock, sources=[source])

    assert [r["status"] for r in results] == [expected_status]
    assert calls.get("files", 0) == expected_gets
    if expected_status == "up_to_date":
        assert results[0]["current_hash"] == results[0]["latest_hash"] == "installed-hash"
        assert "bundle" not in results[0]


def test_recorded_tree_lookup_is_shared_across_skills_of_one_repo(tmp_path, monkeypatch):
    """Many skills installed from one repo at one revision cost ONE extra tree lookup, not one each."""
    import tools.skills_hub as hub
    monkeypatch.setattr(hub, "SKILLS_DIR", tmp_path / "skills")
    entries = []
    for name in ("demo-skill", "demo-skill-2", "demo-skill-3"):
        (tmp_path / "skills" / name).mkdir(parents=True)
        entries.append({"name": name, "source": "github", "identifier": "owner/repo/demo-skill",
                        "content_hash": "h", "install_path": name,
                        "metadata": {"source_revision": "a" * 40}})
    lock = MagicMock()
    lock.list_installed.return_value = entries
    calls: dict = {}
    source = _multi_skill_repo({"b" * 40: "s" * 40, "a" * 40: "s" * 40}, calls)

    results = check_for_skill_updates(lock=lock, sources=[source])

    assert [r["status"] for r in results] == ["up_to_date"] * 3
    assert calls.get("files", 0) == 0
    assert calls["trees"] == ["main", "a" * 40]


def test_unchanged_since_edge_cases():
    """Unchanged subtree answers True; a repo-root skill (its directory IS the root tree, which
    moved), an unsplittable identifier, and an empty recorded revision all answer False."""
    source = _multi_skill_repo({"b" * 40: "s" * 40, "a" * 40: "s" * 40}, {})
    assert source.unchanged_since("owner/repo/demo-skill", "a" * 40) is True
    assert source.unchanged_since("owner/repo", "a" * 40) is False  # no skill path: not splittable
    assert source.unchanged_since("owner/repo/", "a" * 40) is False
    assert source.unchanged_since("owner/repo/demo-skill", "") is False


def test_truncated_recorded_tree_is_not_trusted():
    """A truncated listing may omit the skill directory or report it wrongly: fall back to the fetch."""
    calls: dict = {}
    source = _multi_skill_repo({"b" * 40: "s" * 40, "a" * 40: "s" * 40}, calls)
    real = source._github_json
    source._github_json = lambda url, **kw: (
        {**real(url, **kw), "truncated": True} if url.endswith("a" * 40) else real(url, **kw))
    assert source.unchanged_since("owner/repo/demo-skill", "a" * 40) is False
