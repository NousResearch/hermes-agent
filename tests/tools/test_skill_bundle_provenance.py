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
`scripts/run.py`, `references/foo%23bar.md`, and `references/my%20guide.md`. See
`examples/endpoint-inventory.md`. The repository also
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
        "references/foo#bar.md": "encoded delimiter\n",
        "references/my guide.md": "encoded space\n",
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
        "references/foo#bar.md",
        "references/my guide.md",
        "templates/report.md",
        "scripts/run.py",
        "assets/logo.png",
        "examples/endpoint-inventory.md",
    }
    assert bundle.files["assets/logo.png"] == b"\x89PNG\r\n\x1a\n\x00\xff"
    assert bundle.files["references/foo#bar.md"] == b"encoded delimiter\n"
    assert bundle.files["references/my guide.md"] == b"encoded space\n"
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


def test_github_source_fetch_downloads_full_skill_directory(monkeypatch):
    """Support files a skill keeps outside SKILL.md-linked paths still install.

    Regression for skills using non-canonical support dirs (impeccable keeps
    everything under `reference/` singular and `scripts/` linked only from
    reference files): the old link-driven fetch shipped SKILL.md alone.
    """
    source = GitHubSource(GitHubAuth())
    skill_md = (
        "---\nname: full-dir\ndescription: d\n---\n"
        "See [audit](reference/audit.md) and run `node scripts/pin.mjs`.\n"
    )
    fetched: list = []
    monkeypatch.setattr(source, "_fetch_file_content", lambda _repo, path: skill_md)
    monkeypatch.setattr(
        source, "_fetch_file_bytes",
        lambda _repo, path: fetched.append(path) or b"content-of-" + path.encode(),
    )
    source._tree_cache["owner/repo"] = (
        "main",
        [
            {"path": "skill/SKILL.md", "type": "blob", "mode": "100644"},
            {"path": "skill/reference/audit.md", "type": "blob", "mode": "100644"},
            {"path": "skill/reference/deep/native.md", "type": "blob", "mode": "100644"},
            {"path": "skill/scripts/pin.mjs", "type": "blob", "mode": "100644"},
            {"path": "skill/LICENSE", "type": "blob", "mode": "100644"},
            # skipped: symlink, hidden file, pyc, out-of-prefix
            {"path": "skill/reference/link.md", "type": "blob", "mode": "120000"},
            {"path": "skill/.hidden", "type": "blob", "mode": "100644"},
            {"path": "skill/scripts/x.pyc", "type": "blob", "mode": "100644"},
            {"path": "other/README.md", "type": "blob", "mode": "100644"},
        ],
    )

    bundle = source.fetch("owner/repo/skill")

    assert bundle is not None
    assert set(bundle.files) == {
        "SKILL.md",
        "reference/audit.md",
        "reference/deep/native.md",
        "scripts/pin.mjs",
        "LICENSE",
    }


def test_github_source_fetch_still_requires_linked_references(monkeypatch):
    """A SKILL.md-linked references/ path missing from the tree rejects the bundle."""
    source = GitHubSource(GitHubAuth())
    skill_md = (
        "---\nname: dangling\ndescription: d\n---\n"
        "Read [the guide](references/guide.md).\n"
    )
    monkeypatch.setattr(source, "_fetch_file_content", lambda _repo, path: skill_md)
    monkeypatch.setattr(source, "_fetch_file_bytes", lambda _repo, path: b"x")
    source._tree_cache["owner/repo"] = (
        "main",
        [{"path": "skill/SKILL.md", "type": "blob", "mode": "100644"}],
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
    assert (installed / "references" / "foo#bar.md").read_text() == "encoded delimiter\n"
    assert (installed / "references" / "my guide.md").read_text() == "encoded space\n"
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



SKILL_MD_MISSING_REF = """---
name: partial-bundle
description: References a support file that is unreachable.
---
# Partial
Read [the guide](references/present.md#usage) and the appendix at
`references/absent.md`, then run `scripts/run.py`.
"""


@pytest.fixture
def served_repo_missing_support(tmp_path, monkeypatch):
    """Serve a skill whose SKILL.md references a support file that is not
    present on the server, so fetching it returns 404."""
    # Serve over loopback while opting this test server into private-address
    # access, so the SSRF-safe HTTP client is exercised for real (see
    # ``served_repo``).
    monkeypatch.setattr("tools.url_safety._global_allow_private_urls", lambda: True)

    repo = tmp_path / "upstream-missing"
    repo.mkdir()
    (repo / "SKILL.md").write_text(SKILL_MD_MISSING_REF)
    # references/absent.md is deliberately NOT created, so the server 404s it.
    for rel, content in {
        "references/present.md": "present guide\n",
        "scripts/run.py": "print('ok')\n",
    }.items():
        path = repo / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)

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


def test_install_skips_unreachable_support_file_e2e(served_repo_missing_support, monkeypatch, tmp_path):
    """A referenced support file that 404s is skipped rather than aborting the
    whole URL install: the bundle still installs end-to-end through quarantine,
    scan, install, and lock provenance, with only the reachable files landing
    on disk and recorded in the lock file (#66760)."""
    from hermes_cli.skills_hub import do_install
    import tools.skills_hub as hub

    _repo, url = served_repo_missing_support
    home = tmp_path / "home"
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr("tools.skills_hub.is_safe_url", lambda _url: True)
    monkeypatch.setattr("tools.skills_hub.check_website_access", lambda _url: None)
    monkeypatch.setattr(hub, "create_source_router", lambda auth=None: [UrlSource()])

    sink = StringIO()
    do_install(url, console=Console(file=sink, force_terminal=False), skip_confirm=True)

    installed = home / "skills" / "partial-bundle"
    assert (installed / "SKILL.md").is_file()
    assert (installed / "references" / "present.md").read_text() == "present guide\n"
    assert (installed / "scripts" / "run.py").is_file()
    # The unreachable reference neither blocked the install nor was written.
    assert not (installed / "references" / "absent.md").exists()

    entry = json.loads((home / "skills" / ".hub" / "lock.json").read_text())["installed"]["partial-bundle"]
    assert entry["scan_provenance"]["source_url"] == url
    assert "references/present.md" in entry["files"]
    assert "references/absent.md" not in entry["files"]






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


UPSTREAM_STUB_MD = """---
name: upstream-demo
description: Upstream-maintained catalog entry.
metadata:
  hermes:
    upstream:
      repo: acme/design-skill
      path: .hermes/skills/design
---
# Stub
"""


def test_optional_source_upstream_stub_fetches_from_external_repo(tmp_path, monkeypatch):
    """A catalog stub with metadata.hermes.upstream installs the upstream repo's
    content (relabelled official/trusted), not the stub itself."""
    from tools.skills_hub import OptionalSkillSource, SkillBundle

    root = tmp_path / "optional-skills"
    skill = root / "creative" / "upstream-demo"
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text(UPSTREAM_STUB_MD)

    source = OptionalSkillSource()
    source._optional_dir = root

    fetched_ids = []

    class _FakeGitHub:
        def fetch(self, identifier):
            fetched_ids.append(identifier)
            return SkillBundle(
                name="design",
                files={"SKILL.md": "---\nname: design\ndescription: real\n---\nreal body",
                       "reference/audit.md": "audit"},
                source="github",
                identifier=identifier,
                trust_level="community",
                metadata={"source_url": "https://github.com/acme/design-skill"},
            )

    monkeypatch.setattr(source, "_get_github", lambda: _FakeGitHub())

    bundle = source.fetch("official/creative/upstream-demo")

    assert fetched_ids == ["acme/design-skill/.hermes/skills/design"]
    assert bundle is not None
    assert bundle.source == "official"
    assert bundle.identifier == "official/creative/upstream-demo"
    assert bundle.trust_level == "trusted"
    assert bundle.files["SKILL.md"].startswith("---\nname: design")
    assert bundle.metadata["upstream_repo"] == "acme/design-skill"


def test_optional_source_upstream_pointer_rejects_malformed(tmp_path):
    from tools.skills_hub import OptionalSkillSource

    source = OptionalSkillSource()
    bad = [
        "---\nname: x\nmetadata:\n  hermes:\n    upstream:\n      repo: acme\n      path: skills/x\n---\n",          # repo not owner/name
        "---\nname: x\nmetadata:\n  hermes:\n    upstream:\n      repo: a/b/c\n      path: skills/x\n---\n",        # repo too deep
        "---\nname: x\nmetadata:\n  hermes:\n    upstream:\n      repo: acme/skill\n      path: ../../etc\n---\n",  # traversal
        "---\nname: x\nmetadata:\n  hermes:\n    upstream:\n      repo: acme/skill\n---\n",                          # missing path
        "---\nname: x\n---\n",                                                                                       # no pointer
    ]
    for content in bad:
        assert source._upstream_pointer_from_content(content) is None


def test_unified_search_trust_rank_survives_limit_cut():
    """Official/builtin results must survive the limit truncation even when a
    high-volume community source floods the merged list first."""
    from unittest.mock import patch as _patch
    from tools.skills_hub import unified_search, SkillMeta

    community = [
        SkillMeta(name=f"s{i}", description="", source="skills.sh",
                  identifier=f"skills-sh/x/s{i}", trust_level="community")
        for i in range(20)
    ]
    official = [SkillMeta(name="s-official", description="", source="official",
                          identifier="official/cat/s-official", trust_level="builtin")]

    with _patch("tools.skills_hub.parallel_search_sources",
                return_value=(community + official, {}, [])):
        results = unified_search("s", [], source_filter="all", limit=10)

    assert results[0].identifier == "official/cat/s-official"
