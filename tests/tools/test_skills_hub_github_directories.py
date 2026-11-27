"""GitHub directory references install without relaxing non-regular/path guards (#107256)."""

import json
from io import StringIO
from pathlib import PurePosixPath

import httpx
import pytest
from rich.console import Console

from tools import skills_hub
from tools.skills_hub_github import GitHubAuth, GitHubSource


@pytest.fixture
def github_bundle(monkeypatch):
    """Replace only GitHub's HTTP/auth boundary; keep resolution and installation real."""
    repo = "owner/demo-bundle"
    revision = "a" * 40
    monkeypatch.setattr(GitHubAuth, "_resolve_token", lambda self: None)

    def configure(skill_path, reference, extra_entries=()):
        prefix = f"{skill_path}/" if skill_path else ""
        skill_md = (
            "---\nname: demo-bundle\ndescription: A bundled asset example.\n---\n"
            f"Everything is bundled under `{reference}`.\n"
            "See [guide](references/guide.md).\n"
        )
        files = {
            "SKILL.md": skill_md.encode(),
            "assets/brand-kit/nested/logo.svg": b"<svg>example</svg>\n",
            "references/guide.md": b"Use the bundled assets.\n",
            "README.md": b"Bundle license and usage.\n",
        }
        directories = {
            str(parent) for name in files for parent in PurePosixPath(name).parents
            if str(parent) != "."
        }
        entries = [
            {"path": prefix + name, "type": "tree", "mode": "040000"}
            for name in sorted(directories)
        ] + [
            {"path": prefix + name, "type": "blob", "mode": "100644"} for name in files
        ]
        entries.extend({**entry, "path": prefix + entry["path"]} for entry in extra_entries)
        fetched = []

        def get(url, **kwargs):
            if url == f"https://api.github.com/repos/{repo}":
                return httpx.Response(200, json={"default_branch": "main"})
            if url == f"https://api.github.com/repos/{repo}/git/trees/main":
                assert kwargs["params"] == {"recursive": "1"}
                return httpx.Response(200, json={"sha": revision, "tree": entries})
            raw_prefix = f"https://raw.githubusercontent.com/{repo}/{revision}/"
            assert url.startswith(raw_prefix), url
            path = url.removeprefix(raw_prefix)
            assert path.startswith(prefix), path
            relative = path.removeprefix(prefix)
            fetched.append(relative)
            return httpx.Response(200, content=files[relative])

        monkeypatch.setattr(skills_hub, "_skills_hub_http_get", get)
        monkeypatch.setattr(
            "tools.skills_hub_search.create_source_router",
            lambda auth=None: [GitHubSource(auth or GitHubAuth())],
        )
        return f"{repo}/{skill_path}", files, fetched, revision

    return configure


@pytest.mark.parametrize("skill_path", ["", "skills/demo-bundle"], ids=["root", "subdir"])
@pytest.mark.parametrize("trailing_slash", ["", "/"], ids=["bare", "slash"])
@pytest.mark.parametrize("directory", ["assets/brand-kit", "assets/brand-kit/nested"])
def test_github_directory_references_install_complete_scanned_bundle(
    github_bundle, monkeypatch, tmp_path, caplog, skill_path, trailing_slash, directory,
):
    from hermes_cli.skills_hub import do_install

    home = tmp_path / "home"
    monkeypatch.setenv("HERMES_HOME", str(home))
    identifier, files, fetched, revision = github_bundle(
        skill_path, directory + trailing_slash,
        [{"path": "assets/brand-kit/ignored-link", "type": "blob", "mode": "120000"},
         {"path": "assets/brand-kit/ignored-submodule", "type": "commit", "mode": "160000"}],
    )
    output = StringIO()

    assert do_install(
        identifier, category="creative", skip_confirm=True, source_id="github",
        console=Console(file=output, force_terminal=False),
    ) is True, output.getvalue()

    installed = home / "skills" / "creative" / "demo-bundle"
    assert {p.relative_to(installed).as_posix(): p.read_bytes()
            for p in installed.rglob("*") if p.is_file()} == files
    assert set(fetched) == set(files)
    assert "Referenced skill support file is missing" not in caplog.text
    assert "Rejected non-regular" not in caplog.text
    entry = json.loads((home / "skills" / ".hub" / "lock.json").read_text())["installed"]["demo-bundle"]
    assert set(entry["files"]) == set(files)
    assert entry["metadata"]["source_revision"] == revision
    assert entry["scan_verdict"] == "safe"
    assert entry["scan_provenance"]["fresh"] is True
    assert revision in entry["scan_provenance"]["source_url"]
    assert "Quarantined to" in output.getvalue()


@pytest.mark.parametrize("entry", [
    {"path": "assets/unsafe", "type": "blob", "mode": "120000"},
    {"path": "assets/unsafe", "type": "commit", "mode": "160000"},
    {"path": "assets/unsafe", "type": "tree", "mode": "120000"},
    {"path": "assets/unsafe", "type": "unknown", "mode": "040000"},
    {"path": "assets/../outside.md", "type": "blob", "mode": "100644"},
], ids=["symlink", "gitlink", "non-directory-tree", "unsupported", "traversal"])
@pytest.mark.parametrize("skill_path", ["", "skills/demo-bundle"], ids=["root", "subdir"])
def test_github_directory_reference_keeps_unsafe_entry_guards(
    github_bundle, skill_path, entry,
):
    identifier, _files, fetched, _revision = github_bundle(skill_path, entry["path"], [entry])

    assert GitHubSource(GitHubAuth()).fetch(identifier) is None
    assert entry["path"] not in fetched
