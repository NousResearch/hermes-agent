"""Exact-commit bundle contracts; HTTP fixtures are not a private-hub E2E."""
import hashlib
from unittest.mock import MagicMock

import httpx
import pytest

from tools.skills_hub_github import GitHubAuth, GitHubSource
from tools.skills_hub_skillssh import SkillsShSource

COMMIT = "a" * 40
TREE = "b" * 40
IDENTIFIER = "sample/bundle/candidates/demo"
FILES = {"SKILL.md": b"---\nname: demo\ndescription: Example.\n---\nRead `references/guide.md`.\n",
         "references/guide.md": b"Example guide\n", "LICENSE": b"Example license\n"}


def blob_hash(data):
    return hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()


@pytest.fixture
def remote(monkeypatch):
    import tools.skills_hub as hub
    calls = []
    entries = [{"path": "candidates/demo/" + rel, "type": "blob", "mode": "100644",
                "sha": blob_hash(body)} for rel, body in FILES.items()]
    state = {"commit": COMMIT, "tree": TREE, "entries": entries, "truncated": False,
             "missing": None, "corrupt": None, "status": 200}

    def get(url, **kwargs):
        calls.append((url, kwargs))
        if "/commits/" in url:
            return httpx.Response(state["status"], json={"sha": state["commit"],
                                  "commit": {"tree": {"sha": state["tree"]}}})
        if "/git/trees/" in url:
            return httpx.Response(state["status"], json={"sha": state["tree"], "tree": state["entries"],
                                                       "truncated": state["truncated"]})
        if "raw.githubusercontent.com" in url or "/contents/" in url:
            rel = url.split("candidates/demo/", 1)[-1]
            if rel == state["missing"]:
                return httpx.Response(404)
            data = b"altered" if rel == state["corrupt"] else FILES.get(rel)
            return httpx.Response(200 if data else 404, content=data or b"")
        raise AssertionError("Unexpected floating or registry request")

    monkeypatch.setattr(hub, "_skills_hub_http_get", get)
    auth = MagicMock(spec=GitHubAuth)
    auth.get_headers.return_value = {}
    return GitHubSource(auth), state, calls


@pytest.mark.parametrize("revision", ["main", "abc123", "A" * 40, " " + COMMIT, COMMIT + "\n", "", "g" * 40])
@pytest.mark.parametrize("action", ["fetch", "inspect"])
def test_invalid_pin_rejected_before_http(remote, revision, action):
    source, _, calls = remote
    with pytest.raises(ValueError):
        getattr(source, action)(IDENTIFIER, revision=revision)
    assert calls == []


def test_commit_tree_blob_inventory_and_candidate_identity(remote):
    source, _, calls = remote
    bundle = source.fetch(IDENTIFIER, revision=COMMIT)
    assert bundle.files == FILES
    assert bundle.identifier == IDENTIFIER
    assert bundle.metadata["requested_revision"] == bundle.metadata["resolved_commit"] == COMMIT
    assert bundle.metadata["source_revision"] == bundle.metadata["source_tree"] == TREE
    assert bundle.metadata["source_blob_hashes"] == {rel: blob_hash(body) for rel, body in FILES.items()}
    file_calls = [url for url, _ in calls if "raw.githubusercontent.com" in url]
    assert all("/" + COMMIT + "/" in url for url in file_calls)
    assert source.inspect(IDENTIFIER, revision=COMMIT).extra["source_tree"] == TREE


@pytest.mark.parametrize("fault", ["truncated", "missing", "corrupt", "wrong_commit", "http", "link", "symlink", "submodule", "collision", "unsafe"])
def test_pinned_bundle_fails_closed(remote, fault):
    source, state, _ = remote
    changes = {
        "truncated": {"truncated": True}, "missing": {"missing": "LICENSE"},
        "corrupt": {"corrupt": "LICENSE"}, "wrong_commit": {"commit": "c" * 40},
        "http": {"status": 404},
        "link": {"entries": [entry for entry in state["entries"] if not entry["path"].endswith("guide.md")]},
        "symlink": {"entries": [{**state["entries"][0], "mode": "120000"}]},
        "submodule": {"entries": [{**state["entries"][0], "type": "commit", "mode": "160000"}]},
        "collision": {"entries": state["entries"] + [{**state["entries"][0], "path": "candidates/demo/skill.md"}]},
        "unsafe": {"entries": state["entries"] + [{**state["entries"][0], "path": "candidates/demo/../escape"}]},
    }
    state.update(changes[fault])
    assert source.fetch(IDENTIFIER, revision=COMMIT) is None
    assert source.inspect(IDENTIFIER, revision=COMMIT) is None


@pytest.mark.parametrize("entries", [
    [("A/x", "blob"), ("a/y", "blob")],
    [("A", "blob"), ("a/y", "blob")],
    [("A", "tree"), ("a/y", "blob")],
    [("A", "tree"), ("A", "tree")],
    [("../unsafe", "tree")],
    [("references", "blob"), ("references/guide.md", "blob")],
])
def test_component_conflicts_rejected_before_blob_download(remote, entries, monkeypatch):
    source, state, _calls = remote
    state["entries"] += [
        {"path": "candidates/demo/" + path, "type": kind,
         "mode": "040000" if kind == "tree" else "100644", "sha": "c" * 40}
        for path, kind in entries]
    monkeypatch.setattr(source, "_fetch_file_bytes", lambda *_a, **_kw: pytest.fail("blob before preflight"))
    assert source.fetch(IDENTIFIER, revision=COMMIT) is None


@pytest.mark.parametrize("explicit", [False, True])
def test_legitimate_explicit_and_implicit_directory_components(remote, explicit):
    source, state, _calls = remote
    if explicit:
        state["entries"].append({"path": "candidates/demo/references", "type": "tree",
                                 "mode": "040000", "sha": "c" * 40})
    assert source.fetch(IDENTIFIER, revision=COMMIT).files == FILES


@pytest.mark.parametrize("path", ["candidates/demo.", "candidates/CON", "A./demo", "A /demo",
                                  "candidates/NUL.txt", "candidates/COM¹", "a?b/demo"])
def test_portable_source_components_rejected_before_http(remote, path):
    source, _state, calls = remote
    assert source.fetch("sample/bundle/" + path, revision=COMMIT) is None
    assert calls == []


@pytest.mark.parametrize("path", [
    "guide.md.", "A./x", "A /x", "guide.md ", "CON", "con.md", "A/NUL.txt",
    "PRN", "AUX.log", "COM1", "COM9.txt", "LPT1", "LPT9.log", "COM¹.txt", "LPT²",
    "COM³", "CONIN$", "CONOUT$.txt", "a<b", "a>b", 'a"b', "a|b", "a?b", "a*b",
    "A/b\x00c", "A/b\x01c", "A/b\x1fc", "A/b\x7fc", "A/b\x85c",
])
@pytest.mark.parametrize("kind", ["blob", "tree"])
def test_portable_components_rejected_before_blob_download(remote, monkeypatch, path, kind):
    source, state, _calls = remote
    state["entries"].append({"path": "candidates/demo/" + path, "type": kind,
                             "mode": "040000" if kind == "tree" else "100644", "sha": "c" * 40})
    monkeypatch.setattr(source, "_fetch_file_bytes", lambda *_a, **_kw: pytest.fail("blob before preflight"))
    assert source.fetch(IDENTIFIER, revision=COMMIT) is None


@pytest.mark.parametrize("entries", [
    [("é.md", "blob"), ("e\u0301.md", "blob")],
    [("É/x", "blob"), ("e\u0301/y", "blob")],
    [("é", "tree"), ("e\u0301/y", "blob")],
    [("é", "blob"), ("e\u0301/y", "blob")],
    [("é", "tree"), ("e\u0301", "tree")],
])
def test_canonical_unicode_components_rejected_before_blob_download(remote, monkeypatch, entries):
    source, state, _calls = remote
    state["entries"] += [
        {"path": "candidates/demo/" + path, "type": kind,
         "mode": "040000" if kind == "tree" else "100644", "sha": "c" * 40}
        for path, kind in entries]
    monkeypatch.setattr(source, "_fetch_file_bytes", lambda *_a, **_kw: pytest.fail("blob before preflight"))
    assert source.fetch(IDENTIFIER, revision=COMMIT) is None


@pytest.mark.parametrize("directory", ["é", "e\u0301", "資料", "COM10", "NULish", "LPT0"])
def test_portable_unicode_spelling_and_legitimate_directories_preserved(remote, monkeypatch, directory):
    source, state, _calls = remote
    files = {"SKILL.md": FILES["SKILL.md"], "references/guide.md": FILES["references/guide.md"],
             directory + "/guide.md": b"Unicode guide\n"}
    state["entries"] = [
        {"path": "candidates/demo/" + rel, "type": "blob", "mode": "100644", "sha": blob_hash(body)}
        for rel, body in files.items()]
    state["entries"].append({"path": "candidates/demo/" + directory, "type": "tree",
                             "mode": "040000", "sha": "c" * 40})
    monkeypatch.setattr(source, "_fetch_file_bytes", lambda _repo, path, **_kw: files[path.split("candidates/demo/", 1)[1]])
    bundle = source.fetch(IDENTIFIER, revision=COMMIT)
    assert bundle.identifier == IDENTIFIER and bundle.files == files
    assert set(bundle.metadata["source_blob_hashes"]) == set(files)


def test_production_imports_belong_to_active_checkout():
    from pathlib import Path
    import tools.skills_hub_install as install
    import hermes_cli.skills_hub as cli
    import tools.skills_hub_skillssh as skillssh
    import tools.skills_hub_github as github
    for module in (install, cli, skillssh, github):
        assert module.__file__ is not None
        assert Path(module.__file__).resolve().is_relative_to(Path.cwd().resolve())


def test_commit_cache_segregates_repo_and_pin(remote):
    source, state, calls = remote
    assert source.fetch(IDENTIFIER, revision=COMMIT)
    state["commit"] = "c" * 40
    # Cached first commit remains immutable even as the default branch advances.
    assert source.fetch(IDENTIFIER, revision=COMMIT)
    assert source.fetch(IDENTIFIER, revision=state["commit"])
    assert source.fetch(IDENTIFIER.replace("sample/bundle", "other/bundle"), revision=state["commit"])
    assert len([url for url, _ in calls if "/commits/" in url]) == 3


def test_skillssh_pin_never_discovers_or_rewrites_candidates(remote, monkeypatch):
    github, state, _ = remote
    source = SkillsShSource(github.auth)
    source.github = github
    monkeypatch.setattr(source, "_fetch_detail_page", lambda *_: pytest.fail("registry fallback"))
    identifier = "skills-sh/" + IDENTIFIER
    assert source.fetch(identifier, revision=COMMIT).identifier == identifier
    assert source.inspect(identifier, revision=COMMIT).identifier == identifier
    state["missing"] = "SKILL.md"
    assert source.fetch(identifier, revision=COMMIT) is None
