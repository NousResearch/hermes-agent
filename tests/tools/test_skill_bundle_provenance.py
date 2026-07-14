"""Multi-file third-party skill bundles and scanner provenance (#60598)."""

import json
import subprocess
import threading
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from io import StringIO
from pathlib import Path

import pytest
from rich.console import Console

from tools.skills_guard import SCANNER_VERSION, scan_skill_cached
from tools.skills_hub import GitHubAuth, GitHubSource, HubLockFile, SkillBundle, UrlSource


SKILL_MD = """---
name: demo-bundle
description: A multi-file test skill.
---
# Demo
Read [the guide](references/guide.md#usage), use `templates/report.md?raw=1`, and run
`scripts/run.py`. See `examples/endpoint-inventory.md`. The repository also
contains assets/logo.png.
"""


class _QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self, *_args):
        pass


@pytest.fixture
def served_repo(tmp_path):
    repo = tmp_path / "upstream"
    repo.mkdir()
    (repo / "SKILL.md").write_text(SKILL_MD)
    for rel, content in {
        "references/guide.md": "safe guide\n",
        "templates/report.md": "report\n",
        "scripts/run.py": "print('ok')\n",
        "assets/logo.png": b"\x89PNG\r\n\x1a\n\x00\xff",
        "examples/endpoint-inventory.md": "example\n",
        "examples/not-installed.md": "must not be copied\n",
        "README.md": "must not be copied\n",
    }.items():
        path = repo / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(content, bytes):
            path.write_bytes(content)
        else:
            path.write_text(content)
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(
        ["git", "-c", "user.name=Test", "-c", "user.email=test@example.com", "commit", "-qm", "fixture"],
        cwd=repo,
        check=True,
    )

    server = ThreadingHTTPServer(
        ("127.0.0.1", 0), partial(_QuietHandler, directory=str(repo))
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield repo, f"http://127.0.0.1:{server.server_port}/SKILL.md"
    finally:
        server.shutdown()
        thread.join()


def test_url_source_fetches_only_referenced_allowed_support_directories(served_repo, monkeypatch):
    _repo, url = served_repo
    monkeypatch.setattr("tools.skills_hub.is_safe_url", lambda _url: True)
    monkeypatch.setattr("tools.skills_hub.check_website_access", lambda _url: None)

    bundle = UrlSource().fetch(url)

    assert bundle is not None
    assert set(bundle.files) == {
        "SKILL.md",
        "references/guide.md",
        "templates/report.md",
        "scripts/run.py",
        "assets/logo.png",
        "examples/endpoint-inventory.md",
    }
    assert bundle.files["assets/logo.png"] == b"\x89PNG\r\n\x1a\n\x00\xff"
    assert "examples/not-installed.md" not in bundle.files
    assert bundle.metadata["source_url"] == url


def test_url_source_rejects_traversal_reference(monkeypatch):
    source = UrlSource()
    skill = "---\nname: bad\ndescription: bad\n---\n[bad](references/../../secret.txt)\n"
    monkeypatch.setattr(source, "_fetch_text", lambda _url: skill)

    assert source.fetch("https://example.com/bad/SKILL.md") is None


def test_github_source_rejects_symlink_in_referenced_directory(monkeypatch):
    source = GitHubSource(GitHubAuth())
    monkeypatch.setattr(source, "_fetch_file_content", lambda _repo, path: SKILL_MD if path.endswith("SKILL.md") else "x")
    source._tree_cache["owner/repo"] = (
        "main",
        [
            {"path": "skill/SKILL.md", "type": "blob", "mode": "100644"},
            {"path": "skill/references/guide.md", "type": "blob", "mode": "120000"},
        ],
    )

    assert source.fetch("owner/repo/skill") is None


def test_github_source_fetches_only_exact_references_and_records_tree_revision(monkeypatch):
    source = GitHubSource(GitHubAuth())
    skill = "---\nname: demo\ndescription: demo\n---\n[guide](references/guide.md)\n"
    fetched = []
    monkeypatch.setattr(
        source,
        "_fetch_file_content",
        lambda _repo, path: skill if path.endswith("SKILL.md") else None,
    )

    def _fetch_bytes(_repo, path):
        fetched.append(path)
        return b"guide"

    monkeypatch.setattr(source, "_fetch_file_bytes", _fetch_bytes, raising=False)
    source._tree_cache["owner/repo"] = (
        "develop",
        [
            {"path": "skill/SKILL.md", "type": "blob", "mode": "100644"},
            {"path": "skill/references/guide.md", "type": "blob", "mode": "100644"},
            {"path": "skill/references/unreferenced.md", "type": "blob", "mode": "100644"},
        ],
    )
    source._tree_revisions = {"owner/repo": "deadbeef"}

    bundle = source.fetch("owner/repo/skill")

    assert bundle is not None
    assert fetched == ["skill/references/guide.md"]
    assert bundle.files["references/guide.md"] == b"guide"
    assert bundle.metadata["source_url"] == "https://github.com/owner/repo/tree/deadbeef/skill"
    assert bundle.metadata["source_revision"] == "deadbeef"


def test_scan_cache_records_full_provenance_and_hash_change_forces_rescan(tmp_path):
    skill = tmp_path / "skill"
    skill.mkdir()
    (skill / "SKILL.md").write_text("---\nname: skill\ndescription: test\n---\n# safe\n")
    cache = tmp_path / "scan-cache"

    first, first_provenance = scan_skill_cached(
        skill, source="owner/repo/skill", source_url="https://github.com/owner/repo", cache_dir=cache
    )
    second, second_provenance = scan_skill_cached(
        skill, source="owner/repo/skill", source_url="https://github.com/owner/repo", cache_dir=cache
    )
    (skill / "SKILL.md").write_text("---\nname: skill\ndescription: changed\n---\n# safe\n")
    third, third_provenance = scan_skill_cached(
        skill, source="owner/repo/skill", source_url="https://github.com/owner/repo", cache_dir=cache
    )

    assert first.verdict == second.verdict == third.verdict == "safe"
    assert first_provenance["fresh"] is True
    assert second_provenance["fresh"] is False
    assert third_provenance["fresh"] is True
    assert first_provenance["bundle_hash"].startswith("sha256:")
    assert len(first_provenance["bundle_hash"].split(":", 1)[1]) == 64
    assert third_provenance["bundle_hash"] != first_provenance["bundle_hash"]
    assert first_provenance["scanner_version"] == SCANNER_VERSION
    assert first_provenance["source_url"] == "https://github.com/owner/repo"
    assert isinstance(first_provenance["findings"], list)
    assert isinstance(first_provenance["rules"], list)
    assert first_provenance["scanned_at"]


def test_scan_cache_never_reuses_provenance_across_sources(tmp_path):
    skill = tmp_path / "skill"
    skill.mkdir()
    (skill / "SKILL.md").write_text("---\nname: skill\ndescription: test\n---\n")
    cache = tmp_path / "scan-cache"

    _first, first = scan_skill_cached(
        skill, source="community", source_url="https://one.example/SKILL.md", cache_dir=cache
    )
    _second, second = scan_skill_cached(
        skill, source="community", source_url="https://two.example/SKILL.md", cache_dir=cache
    )

    assert first["fresh"] is True
    assert second["fresh"] is True
    assert second["source_url"] == "https://two.example/SKILL.md"


def test_lock_file_persists_scan_provenance(tmp_path):
    lock = HubLockFile(tmp_path / "lock.json")
    provenance = {
        "source_url": "https://example.com/SKILL.md",
        "bundle_hash": "sha256:" + "a" * 64,
        "scanner_version": SCANNER_VERSION,
        "findings": [],
        "rules": [],
        "scanned_at": "2026-07-09T00:00:00+00:00",
        "fresh": True,
    }
    lock.record_install(
        name="demo", source="url", identifier="https://example.com/SKILL.md",
        trust_level="community", scan_verdict="safe", skill_hash="sha256:legacy",
        install_path="demo", files=["SKILL.md"], scan_provenance=provenance,
    )

    assert lock.get_installed("demo")["scan_provenance"] == provenance


def test_real_temp_repo_and_home_install_e2e(served_repo, monkeypatch, tmp_path):
    from hermes_cli.skills_hub import do_install
    import tools.skills_hub as hub

    _repo, url = served_repo
    home = tmp_path / "home"
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr("tools.skills_hub.is_safe_url", lambda _url: True)
    monkeypatch.setattr("tools.skills_hub.check_website_access", lambda _url: None)
    monkeypatch.setattr(hub, "create_source_router", lambda auth=None: [UrlSource()])

    sink = StringIO()
    do_install(url, console=Console(file=sink, force_terminal=False), skip_confirm=True)

    installed = home / "skills" / "demo-bundle"
    assert (installed / "references" / "guide.md").read_text() == "safe guide\n"
    assert (installed / "templates" / "report.md").is_file()
    assert (installed / "scripts" / "run.py").is_file()
    assert (installed / "examples" / "endpoint-inventory.md").is_file()
    assert not (installed / "examples" / "not-installed.md").exists()
    assert (installed / "assets" / "logo.png").read_bytes() == b"\x89PNG\r\n\x1a\n\x00\xff"
    entry = json.loads((home / "skills" / ".hub" / "lock.json").read_text())["installed"]["demo-bundle"]
    assert entry["scan_provenance"]["source_url"] == url
    assert entry["scan_provenance"]["fresh"] is True
    assert "Scan provenance: fresh" in sink.getvalue()


def test_bundled_optional_source_still_includes_support_files(tmp_path, monkeypatch):
    from tools.skills_hub import OptionalSkillSource

    root = tmp_path / "optional-skills"
    skill = root / "category" / "official-demo"
    (skill / "references").mkdir(parents=True)
    (skill / "SKILL.md").write_text("---\nname: official-demo\ndescription: demo\n---\n")
    (skill / "references" / "all.md").write_text("all")
    source = OptionalSkillSource()
    source._optional_dir = root

    bundle = source.fetch("official/category/official-demo")
    assert bundle is not None
    assert set(bundle.files) == {"SKILL.md", "references/all.md"}
