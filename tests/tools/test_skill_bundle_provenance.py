"""Multi-file third-party skill bundles and scanner provenance (#60598)."""

import json
import subprocess
import sys
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
def served_repo(tmp_path, monkeypatch):
    # The fixture intentionally serves over loopback. Keep exercising the real
    # HTTP transport while opting this test server into private-address access.
    monkeypatch.setattr("tools.url_safety._global_allow_private_urls", lambda: True)

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


def _make_skills_redirect(link: Path, target: Path) -> bool:
    """Make *link* a directory redirect (Windows junction or POSIX symlink)
    pointing at *target*. Junctions need no admin rights, unlike symlinks."""
    if sys.platform == "win32":
        try:
            subprocess.run(
                ["cmd", "/c", "mklink", "/J", str(link), str(target)],
                check=True,
                capture_output=True,
            )
            return True
        except (subprocess.CalledProcessError, OSError):
            return False
    try:
        link.symlink_to(target, target_is_directory=True)
        return True
    except OSError:
        return False


def test_install_with_junctioned_skills_dir(served_repo, monkeypatch, tmp_path):
    """#86971: install must not mix resolved and unresolved paths when the
    skills directory is a junction/symlink redirect.

    install_dir is resolved by _resolve_lock_install_path (following the
    redirect), so relative_to() must receive the resolved skills root or it
    raises ValueError after the files have already been moved, leaving a lock
    entry without a content_hash (which then poisons 'hermes skills check').
    """
    from hermes_cli.skills_hub import do_install
    import tools.skills_hub as hub

    _repo, url = served_repo
    home = tmp_path / "home"
    home.mkdir()
    real_skills = tmp_path / "real-skills"
    real_skills.mkdir()
    skills_link = home / "skills"
    if not _make_skills_redirect(skills_link, real_skills):
        pytest.skip("Cannot create a junction/symlink in this environment")

    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr("tools.skills_hub.is_safe_url", lambda _url: True)
    monkeypatch.setattr("tools.skills_hub.check_website_access", lambda _url: None)
    monkeypatch.setattr(hub, "create_source_router", lambda auth=None: [UrlSource()])

    sink = StringIO()
    do_install(url, console=Console(file=sink, force_terminal=False), skip_confirm=True)

    # Files landed in the real target, reached through the junction.
    installed = real_skills / "demo-bundle"
    assert (installed / "SKILL.md").is_file()
    assert (installed / "references" / "guide.md").read_text() == "safe guide\n"
    # Lock entry got a valid relative install_path AND the content hash — the
    # record_install call the pre-fix ValueError used to skip.
    entry = json.loads((home / "skills" / ".hub" / "lock.json").read_text())["installed"]["demo-bundle"]
    assert entry["install_path"] == "demo-bundle"
    assert entry["content_hash"].startswith("sha256:")
    # The post-install "Installed:" line (relative_to on the display path)
    # renders instead of raising.
    assert "Installed:" in sink.getvalue()


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
