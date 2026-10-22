"""Behaviour tests for learned-skill provenance (``tools.skill_sources``).

``hermes skills check`` already answers "has this hub-installed skill moved?" from the
hub lock file. These tests cover the other half: a skill that declares where it came
from in its own frontmatter, checked against the live source. The network tests use a
loopback HTTP server started inside the test, and the repository tests use a local bare
repo, so nothing here reaches the internet.
"""

import io
import subprocess
from typing import Optional
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from threading import Thread

import pytest
from rich.console import Console

from tools import skill_sources

SKILL_TEMPLATE = """\
---
name: {name}
description: Example skill for a sources test.
version: 0.1.0
author: Hermes
license: MIT
metadata:
  hermes:
    tags: [Example]
{block}---

# Example

## When to Use

- A test needs a skill that records where it came from.
"""


def write_skill(name: str, sources_block: str = "", *, root: Optional[Path] = None) -> Path:
    """Create ``<skills dir>/<name>/SKILL.md`` with *sources_block* inserted verbatim."""
    skills_root = root or skill_sources.get_skills_dir()
    skill_dir = skills_root / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(
        SKILL_TEMPLATE.format(name=name, block=sources_block), encoding="utf-8")
    return skill_dir


def url_block(url: str, *, retrieved: str = "2026-10-22", name: str = "url") -> str:
    return f"    sources:\n      - {name}: {url}\n        retrieved: {retrieved}\n"


@pytest.fixture(autouse=True)
def _private_urls_reset(monkeypatch):
    """Each test starts from the secure default and a clean private-URL cache."""
    from tools import url_safety

    monkeypatch.delenv("HERMES_ALLOW_PRIVATE_URLS", raising=False)
    url_safety._reset_allow_private_cache()
    yield
    monkeypatch.delenv("HERMES_ALLOW_PRIVATE_URLS", raising=False)
    url_safety._reset_allow_private_cache()


def allow_private_urls(monkeypatch):
    """Opt into loopback fetches the way a user with a local mirror would."""
    from tools import url_safety

    monkeypatch.setenv("HERMES_ALLOW_PRIVATE_URLS", "true")
    url_safety._reset_allow_private_cache()


def reset_website_policy_cache() -> None:
    """Drop the process-wide website-policy cache so a temp config is actually read."""
    from tools import website_policy

    website_policy._cached_policy = None
    website_policy._cached_policy_path = None
    website_policy._cached_policy_time = 0.0


@pytest.fixture
def site():
    """A real loopback HTTP server whose served body the test can swap."""

    class Handler(BaseHTTPRequestHandler):
        body = b"<html><body><h1>alpha</h1></body></html>"

        def do_GET(self):  # the BaseHTTPRequestHandler API
            payload = self.body
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, format, *args):  # base-class signature
            pass  # keep the test output clean

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/docs", Handler, server
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def git(*args, cwd=None):
    completed = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, check=True)
    return completed.stdout.strip()


# ── read_sources ────────────────────────────────────────────────────────────


class TestReadSources:
    def test_reads_the_three_source_shapes(self, tmp_path):
        skill_dir = write_skill("mixed-sources", """\
    sources:
      - url: https://example.com/docs/guide
        retrieved: 2026-10-22
      - repo: https://github.com/owner/project
        revision: 0123456789abcdef0123456789abcdef01234567
        retrieved: 2026-10-22
      - path: ~/projects/app/docs/setup.md
""", root=tmp_path)
        entries = skill_sources.read_sources(skill_dir)
        assert [entry["kind"] for entry in entries] == ["url", "repo", "path"]
        assert entries[0]["retrieved"] == "2026-10-22"
        assert entries[1]["revision"] == "0123456789abcdef0123456789abcdef01234567"
        assert entries[2]["value"] == "~/projects/app/docs/setup.md"

    def test_an_absent_block_reads_as_empty(self, tmp_path):
        assert skill_sources.read_sources(write_skill("no-sources", "", root=tmp_path)) == []

    def test_a_missing_skill_directory_does_not_raise(self, tmp_path):
        assert skill_sources.read_sources(tmp_path / "does-not-exist") == []

    @pytest.mark.parametrize("block", [
        "    sources: not-a-list\n",
        "    sources:\n      - just-a-string\n",
        "    sources:\n      - url: https://example.com\n        path: /tmp/x\n",  # two kinds
        "    sources:\n      - url: https://example.com\n        revision: deadbeef\n",  # revision without repo
        "    sources:\n      - url: ftp://example.com/guide\n",  # not http(s)
        "    sources:\n      - repo: --upload-pack-touch-pwn\n",  # option injection shape
        "    sources: []\n",
        "    sources:\n      - url: ''\n",
    ])
    def test_an_unusable_block_reads_as_empty_without_raising(self, tmp_path, block):
        assert skill_sources.read_sources(write_skill("broken-sources", block, root=tmp_path)) == []

    def test_one_bad_entry_does_not_discard_the_good_ones(self, tmp_path):
        skill_dir = write_skill("partly-broken", """\
    sources:
      - just-a-string
      - url: https://example.com/docs
        retrieved: 2026-10-22
      - path: relative/path.md
""", root=tmp_path)
        entries = skill_sources.read_sources(skill_dir)
        assert [(entry["kind"], entry["value"]) for entry in entries] == [
            ("url", "https://example.com/docs"), ("path", "relative/path.md")]


# ── scope ───────────────────────────────────────────────────────────────────


class TestScope:
    def test_bundled_and_hub_installed_skills_are_skipped(self, monkeypatch, tmp_path):
        write_skill("learned-skill", url_block("https://example.com/docs"), root=tmp_path)
        write_skill("bundled-skill", url_block("https://example.com/docs"), root=tmp_path)
        write_skill("hub-skill", url_block("https://example.com/docs"), root=tmp_path)

        from tools import skill_usage

        monkeypatch.setattr(skill_usage, "is_bundled", lambda name: name == "bundled-skill")
        monkeypatch.setattr(skill_usage, "is_hub_installed", lambda name: name == "hub-skill")
        monkeypatch.setattr(skill_sources, "get_skills_dir", lambda: tmp_path)

        names = [name for name, _dir, _entries in skill_sources.iter_skill_sources()]
        assert names == ["learned-skill"]

    def test_protected_skills_are_skipped(self, monkeypatch, tmp_path):
        write_skill("protected-skill", url_block("https://example.com/docs"), root=tmp_path)
        from tools import skill_usage

        monkeypatch.setattr(skill_usage, "is_protected_builtin", lambda name: name == "protected-skill")
        monkeypatch.setattr(skill_sources, "get_skills_dir", lambda: tmp_path)

        assert list(skill_sources.iter_skill_sources()) == []


# ── URL sources ─────────────────────────────────────────────────────────────


class TestUrlSources:
    def test_full_lifecycle_baseline_unchanged_changed_unreachable(self, monkeypatch, site):
        url, handler, server = site
        allow_private_urls(monkeypatch)
        write_skill("docs-skill", url_block(url))

        first = skill_sources.check_sources()
        assert [(row["status"]) for row in first] == ["baseline_recorded"]

        second = skill_sources.check_sources()
        assert [row["status"] for row in second] == ["unchanged"]

        handler.body = b"<html><body><h1>beta</h1></body></html>"
        third = skill_sources.check_sources()
        assert [row["status"] for row in third] == ["changed"]

        # Sticky: the alert must survive the run that discovered it, or the user is told
        # "changed" once, misses it, and is then told "unchanged" about a stale skill.
        fourth = skill_sources.check_sources()
        assert [row["status"] for row in fourth] == ["changed"]

        server.shutdown()
        server.server_close()
        fifth = skill_sources.check_sources()
        assert [row["status"] for row in fifth] == ["unreachable"]

    def test_a_changed_url_stays_changed_until_the_skill_is_refreshed(self, monkeypatch, site):
        url, handler, _server = site
        allow_private_urls(monkeypatch)
        write_skill("docs-skill", url_block(url, retrieved="2026-10-22"))
        skill_sources.check_sources()

        handler.body = b"<html><body><h1>beta</h1></body></html>"
        assert [row["status"] for row in skill_sources.check_sources()] == ["changed"]
        assert [row["status"] for row in skill_sources.check_sources()] == ["changed"]

        # The user re-runs /learn: the frontmatter's read date moves on.
        write_skill("docs-skill", url_block(url, retrieved="2026-10-23"))
        assert [row["status"] for row in skill_sources.check_sources()] == ["baseline_recorded"]

        # ...and the refreshed baseline is the new content, so it reads as unchanged next.
        assert [row["status"] for row in skill_sources.check_sources()] == ["unchanged"]

    def test_a_changed_url_does_not_move_the_recorded_baseline(self, monkeypatch, site):
        url, handler, _server = site
        allow_private_urls(monkeypatch)
        write_skill("docs-skill", url_block(url))
        skill_sources.check_sources()
        baseline = skill_sources.load_state()["sources"]["docs-skill"][f"url:{url}"]["hash"]

        handler.body = b"<html><body><h1>beta</h1></body></html>"
        skill_sources.check_sources()
        skill_sources.check_sources()

        stored = skill_sources.load_state()["sources"]["docs-skill"][f"url:{url}"]
        assert stored["hash"] == baseline, "the pre-change hash must survive a changed verdict"
        assert stored["retrieved"] == "2026-10-22"

    def test_markup_noise_alone_is_not_a_change(self, monkeypatch, site):
        """Cache-busting ids and re-indented markup must not read as a content change."""
        url, handler, _server = site
        allow_private_urls(monkeypatch)
        write_skill("docs-skill", url_block(url))
        skill_sources.check_sources()

        handler.body = (b"<html>\n  <body>\n    <script>var build='9f2a';</script>\n"
                        b"    <h1>alpha</h1>\n  </body>\n</html>")
        assert [row["status"] for row in skill_sources.check_sources()] == ["unchanged"]

    def test_a_private_url_is_skipped_when_private_urls_are_not_allowed(self, site):
        url, _handler, _server = site
        write_skill("docs-skill", url_block(url))

        rows = skill_sources.check_sources()
        assert [row["status"] for row in rows] == ["skipped"]
        assert "URL safety" in rows[0]["detail"]

    def test_baselines_are_stored_in_one_profile_sidecar_not_in_the_skill(self, monkeypatch, site):
        url, _handler, _server = site
        allow_private_urls(monkeypatch)
        skill_dir = write_skill("docs-skill", url_block(url))
        skill_md_before = (skill_dir / "SKILL.md").read_text(encoding="utf-8")

        skill_sources.check_sources()

        state = skill_sources.load_state()
        assert f"url:{url}" in state["sources"]["docs-skill"]
        assert skill_sources.state_path().name == ".source_state.json"
        assert (skill_dir / "SKILL.md").read_text(encoding="utf-8") == skill_md_before


# ── path sources ────────────────────────────────────────────────────────────


class TestPathSources:
    def test_lifecycle_create_check_edit_check(self, tmp_path):
        target = tmp_path / "notes.md"
        target.write_text("first version\n", encoding="utf-8")
        write_skill("notes-skill", f"    sources:\n      - path: {target}\n")

        assert [row["status"] for row in skill_sources.check_sources()] == ["baseline_recorded"]
        assert [row["status"] for row in skill_sources.check_sources()] == ["unchanged"]

        target.write_text("second version\n", encoding="utf-8")
        assert [row["status"] for row in skill_sources.check_sources()] == ["changed"]

        # Sticky across runs, cleared only by a refresh.
        assert [row["status"] for row in skill_sources.check_sources()] == ["changed"]
        write_skill("notes-skill", f"    sources:\n      - path: {target}\n        retrieved: 2026-10-23\n")
        assert [row["status"] for row in skill_sources.check_sources()] == ["baseline_recorded"]

        target.unlink()
        assert [row["status"] for row in skill_sources.check_sources()] == ["unreachable"]

    def test_a_relative_path_is_skipped_rather_than_guessed(self):
        write_skill("notes-skill", "    sources:\n      - path: notes.md\n")
        rows = skill_sources.check_sources()
        assert [row["status"] for row in rows] == ["skipped"]


# ── repo sources ────────────────────────────────────────────────────────────


@pytest.fixture
def bare_repo(tmp_path):
    """A local bare repo used as the remote, so the check needs no network."""
    remote = tmp_path / "remote.git"
    work = tmp_path / "work"
    git("init", "--bare", "-q", "-b", "main", str(remote))
    git("init", "-q", "-b", "main", str(work))
    git("config", "user.email", "test@example.com", cwd=work)
    git("config", "user.name", "Test", cwd=work)
    (work / "README.md").write_text("one\n", encoding="utf-8")
    git("add", "README.md", cwd=work)
    git("commit", "-q", "-m", "one", cwd=work)
    git("remote", "add", "origin", str(remote), cwd=work)
    git("push", "-q", "origin", "HEAD:refs/heads/main", cwd=work)
    return remote, work


class TestRepoSources:
    def test_a_recorded_revision_compares_against_the_remote_head(self, bare_repo):
        remote, work = bare_repo
        revision = git("rev-parse", "HEAD", cwd=work)
        write_skill("repo-skill", f"    sources:\n      - repo: {remote}\n        revision: {revision}\n")

        assert [row["status"] for row in skill_sources.check_sources()] == ["unchanged"]

        (work / "README.md").write_text("two\n", encoding="utf-8")
        git("commit", "-q", "-am", "two", cwd=work)
        git("push", "-q", "origin", "HEAD:refs/heads/main", cwd=work)

        rows = skill_sources.check_sources()
        assert [row["status"] for row in rows] == ["changed"]
        assert "new commits since the recorded revision" in rows[0]["detail"]

    def test_a_repo_without_a_revision_is_skipped(self, bare_repo):
        remote, _work = bare_repo
        write_skill("repo-skill", f"    sources:\n      - repo: {remote}\n")
        assert [row["status"] for row in skill_sources.check_sources()] == ["skipped"]

    def test_an_unreachable_remote_reports_unreachable(self, tmp_path):
        write_skill("repo-skill", f"    sources:\n      - repo: {tmp_path / 'nope.git'}\n        revision: {'a' * 40}\n")
        assert [row["status"] for row in skill_sources.check_sources()] == ["unreachable"]

    def test_an_ssh_remote_is_skipped_rather_than_prompting(self):
        write_skill("repo-skill", "    sources:\n      - repo: ssh://git@example.com/owner/project.git\n"
                                  f"        revision: {'a' * 40}\n")
        rows = skill_sources.check_sources()
        assert [row["status"] for row in rows] == ["skipped"]
        assert "credentials" in rows[0]["detail"]


class TestRepoEgressPolicy:
    """An http(s) `repo:` must not be an egress path around URL safety and site policy."""

    def _write_repo_skill(self, repo_url):
        write_skill("repo-skill", f"    sources:\n      - repo: {repo_url}\n        revision: {'a' * 40}\n")

    def test_a_blocklisted_repo_url_is_skipped_and_git_is_never_run(self, monkeypatch, tmp_path):
        config = skill_sources.get_skills_dir().parent / "config.yaml"
        config.write_text("security:\n  website_blocklist:\n    enabled: true\n"
                          "    domains: [blocked.example]\n", encoding="utf-8")
        reset_website_policy_cache()
        called = []
        monkeypatch.setattr(skill_sources, "_ls_remote_head", lambda target: called.append(target) or "b" * 40)
        self._write_repo_skill("https://blocked.example/owner/project.git")

        rows = skill_sources.check_sources()

        assert [row["status"] for row in rows] == ["skipped"]
        assert "website policy" in rows[0]["detail"]
        assert called == [], "git was invoked for a blocklisted repository"

    def test_an_internal_repo_url_is_skipped_and_git_is_never_run(self, monkeypatch):
        called = []
        monkeypatch.setattr(skill_sources, "_ls_remote_head", lambda target: called.append(target) or "b" * 40)
        # Link-local metadata address: the classic SSRF target.
        self._write_repo_skill("http://169.254.169.254/latest/meta-data.git")

        rows = skill_sources.check_sources()

        assert [row["status"] for row in rows] == ["skipped"]
        assert "URL safety" in rows[0]["detail"]
        assert called == []

    def test_an_ext_transport_value_is_rejected_outright(self, tmp_path):
        self._write_repo_skill("ext::sh -c touch% /tmp/pwned")

        assert skill_sources.read_sources(skill_sources.get_skills_dir() / "repo-skill") == []

    def test_git_transports_are_pinned_on_the_command_line(self, monkeypatch):
        seen = {}

        def fake_run(argv, **kwargs):
            seen["argv"] = argv
            class Done:
                returncode = 0
                stdout = f"{'b' * 40}\tHEAD\n"
            return Done()

        monkeypatch.setattr(skill_sources.subprocess, "run", fake_run)
        skill_sources._ls_remote_head("/some/local/repo.git")

        argv = seen["argv"]
        assert argv[:1] == ["git"]
        assert "protocol.allow=never" in argv
        for allowed in ("protocol.https.allow=always", "protocol.http.allow=always",
                        "protocol.file.allow=always"):
            assert allowed in argv


class TestRedirectPolicy:
    def test_a_redirect_to_a_blocklisted_host_is_not_followed(self, monkeypatch, tmp_path):
        config = skill_sources.get_skills_dir().parent / "config.yaml"
        config.write_text("security:\n  website_blocklist:\n    enabled: true\n"
                          "    domains: [blocked.example]\n", encoding="utf-8")
        reset_website_policy_cache()
        allow_private_urls(monkeypatch)

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):  # the BaseHTTPRequestHandler API
                self.send_response(302)
                self.send_header("Location", "http://blocked.example/elsewhere")
                self.send_header("Content-Length", "0")
                self.end_headers()

            def log_message(self, format, *args):
                pass

        server = HTTPServer(("127.0.0.1", 0), Handler)
        thread = Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            write_skill("docs-skill", url_block(f"http://127.0.0.1:{server.server_port}/docs"))
            rows = skill_sources.check_sources()
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)

        assert [row["status"] for row in rows] == ["skipped"]
        assert "website policy" in rows[0]["detail"]


# ── CLI surface ─────────────────────────────────────────────────────────────


def run_do_check(monkeypatch, **patched):
    """Run ``do_check`` with the hub check stubbed and return the printed output."""
    for target, value in patched.items():
        monkeypatch.setattr(target, value)
    from hermes_cli import skills_hub

    buffer = io.StringIO()
    skills_hub.do_check(console=Console(file=buffer, width=200))
    return buffer.getvalue()


class TestCheckCommand:
    def test_hub_only_output_is_unchanged_without_recorded_sources(self, monkeypatch):
        output = run_do_check(monkeypatch, **{"tools.skills_hub_install.check_for_skill_updates":
                                              lambda **kwargs: []})
        assert "No hub-installed skills to check." in output
        assert "Learned skill sources" not in output

    def test_recorded_sources_add_a_table_without_removing_the_hub_one(self, monkeypatch, tmp_path):
        target = tmp_path / "notes.md"
        target.write_text("hello\n", encoding="utf-8")
        write_skill("notes-skill", f"    sources:\n      - path: {target}\n")
        output = run_do_check(monkeypatch, **{
            "tools.skills_hub_install.check_for_skill_updates": lambda **kwargs: [
                {"name": "hub-skill", "source": "official", "status": "up_to_date"}]})
        assert "Skill Updates" in output and "hub-skill" in output
        assert "Learned skill sources" in output and "notes-skill" in output
