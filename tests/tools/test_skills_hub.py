"""Tests for tools/skills_hub.py — source adapters, lock file, taps, dedup logic."""

import json
import time
from typing import List, Optional
from unittest.mock import patch, MagicMock

import httpx
import pytest

from tools.skills_hub import (
    GitHubAuth,
    GitHubSource,
    LobeHubSource,
    SkillsShSource,
    UrlSource,
    WellKnownSkillSource,
    OptionalSkillSource,
    SkillSource,
    SkillBundle,
    SkillMeta,
    HubLockFile,
    TapsManager,
    bundle_content_hash,
    check_for_skill_updates,
    create_source_router,
    parallel_search_sources,
    unified_search,
    append_audit_log,
    quarantine_bundle,
    _referenced_support_paths,
)


# ---------------------------------------------------------------------------
# GitHubSource._parse_frontmatter_quick
# ---------------------------------------------------------------------------


class TestParseFrontmatterQuick:
    def test_valid_frontmatter_including_nested_yaml(self):
        content = "---\nname: test-skill\ndescription: A test.\n---\n\n# Body\n"
        fm = GitHubSource._parse_frontmatter_quick(content)
        assert fm["name"] == "test-skill"
        assert fm["description"] == "A test."

        nested = "---\nname: test\nmetadata:\n  hermes:\n    tags: [a, b]\n---\n\nBody.\n"
        assert GitHubSource._parse_frontmatter_quick(nested)["metadata"]["hermes"]["tags"] == ["a", "b"]

    def test_degenerate_frontmatter_returns_empty(self):
        for content in (
            "# Just a heading\nSome body text.\n",     # no frontmatter at all
            "---\nname: test\nno closing here\n",      # unterminated block
            "",                                         # empty document
            "---\n: : : invalid{{\n---\n\nBody.\n",     # unparseable YAML
            "---\n- just a list\n- of items\n---\n\nBody.\n",  # non-dict YAML
        ):
            assert GitHubSource._parse_frontmatter_quick(content) == {}, repr(content)


# ---------------------------------------------------------------------------
# GitHubSource skills.sh.json grouping sidecar (category support)
# ---------------------------------------------------------------------------


class TestSkillsShGroupings:
    """Parsing + stamping of the skills.sh.json grouping sidecar.

    A tap can ship a repo-root ``skills.sh.json`` declaring category
    groupings; we flatten it to {skill_name: title} and stamp the title onto
    each SkillMeta's ``extra["category"]``. This is the generic cross-ecosystem
    mechanism behind NVIDIA-style categorization — not NVIDIA-specific.
    """

    def test_parse_basic_groupings(self):
        content = json.dumps({
            "$schema": "https://skills.sh/schemas/skills.sh.schema.json",
            "groupings": [
                {"title": "Inference AI", "skills": ["dynamo-router", "dynamo-recipe"]},
                {"title": "Decision Optimization", "skills": ["cuopt-developer"]},
            ],
        })
        mapping = GitHubSource._parse_skillsh_groupings(content)
        assert mapping == {
            "dynamo-router": "Inference AI",
            "dynamo-recipe": "Inference AI",
            "cuopt-developer": "Decision Optimization",
        }


    def test_list_skills_stamps_category_from_sidecar(self):
        auth = MagicMock()
        src = GitHubSource(auth=auth)

        meta = SkillMeta(
            name="cuopt-developer", description="d", source="github",
            identifier="NVIDIA/skills/skills/cuopt-developer", trust_level="trusted",
        )
        contents = [{"type": "dir", "name": "cuopt-developer"}]
        groupings = {"cuopt-developer": "Decision Optimization"}

        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = contents

        with patch.object(src, "_read_cache", return_value=None), \
             patch.object(src, "_write_cache"), \
             patch.object(src, "_get_skillsh_groupings", return_value=groupings), \
             patch.object(src, "inspect", return_value=meta), \
             patch("tools.skills_hub.httpx.get", return_value=resp):
            skills = src._list_skills_in_repo("NVIDIA/skills", "skills/")

        assert len(skills) == 1
        assert skills[0].extra["category"] == "Decision Optimization"

# ---------------------------------------------------------------------------
# GitHubSource.trust_level_for
# ---------------------------------------------------------------------------


class TestTrustLevelFor:
    def _source(self):
        auth = MagicMock(spec=GitHubAuth)
        return GitHubSource(auth=auth)

    def test_trusted_repo(self):
        src = self._source()
        # TRUSTED_REPOS is imported from skills_guard, test with known trusted repo
        from tools.skills_guard import TRUSTED_REPOS
        if TRUSTED_REPOS:
            repo = next(iter(TRUSTED_REPOS))
            assert src.trust_level_for(f"{repo}/some-skill") == "trusted"


    def test_browseable_trusted_repos_have_taps(self):
        # General invariant covering all current and future trusted repos
        # that publish under a single `skills/`-style path. openai/skills
        # is the deliberate exception — it has two taps (`.curated/` and
        # `.system/`) — so we just assert membership not path equality.
        from tools.skills_guard import TRUSTED_REPOS

        tap_repos = {tap["repo"] for tap in GitHubSource.DEFAULT_TAPS}
        for repo in TRUSTED_REPOS:
            assert repo in tap_repos, (
                f"Trusted repo {repo!r} is in TRUSTED_REPOS but missing "
                "from GitHubSource.DEFAULT_TAPS — its skills will not be "
                "browsable via `hermes skills browse`."
            )


class TestGitHubSourceFileFetch:
    def test_quotes_decoded_support_path_before_contents_api_fetch(self):
        src = GitHubSource(auth=MagicMock(spec=GitHubAuth))
        src._github_get = MagicMock(return_value=None)

        src._fetch_file_bytes("owner/repo", "skill/references/foo#bar.md")

        assert src._github_get.call_args.args[0] == (
            "https://api.github.com/repos/owner/repo/contents/"
            "skill/references/foo%23bar.md"
        )


# ---------------------------------------------------------------------------
# SkillsShSource
# ---------------------------------------------------------------------------


class TestSkillsShSource:
    def _source(self):
        auth = MagicMock(spec=GitHubAuth)
        return SkillsShSource(auth=auth)

    @patch("tools.skills_hub._write_index_cache")
    @patch("tools.skills_hub._read_index_cache", return_value=None)
    @patch("tools.skills_hub.httpx.get")
    def test_search_maps_skills_sh_results_to_prefixed_identifiers(self, mock_get, _mock_read_cache, _mock_write_cache):
        mock_get.return_value = MagicMock(
            status_code=200,
            json=lambda: {
                "skills": [
                    {
                        "id": "vercel-labs/agent-skills/vercel-react-best-practices",
                        "skillId": "vercel-react-best-practices",
                        "name": "vercel-react-best-practices",
                        "installs": 207679,
                        "source": "vercel-labs/agent-skills",
                    }
                ]
            },
        )

        results = self._source().search("react", limit=5)

        assert len(results) == 1
        assert results[0].source == "skills.sh"
        assert results[0].identifier == "skills-sh/vercel-labs/agent-skills/vercel-react-best-practices"
        assert "skills.sh" in results[0].description
        assert results[0].repo == "vercel-labs/agent-skills"
        assert results[0].path == "vercel-react-best-practices"
        assert results[0].extra["installs"] == 207679


    @patch("tools.skills_hub._write_index_cache")
    @patch("tools.skills_hub._read_index_cache", return_value=None)
    @patch("tools.skills_hub.httpx.get")
    @patch.object(GitHubSource, "fetch")
    def test_fetch_falls_back_to_tree_search_for_deeply_nested_skills(
        self, mock_fetch, mock_get, _mock_read_cache, _mock_write_cache,
    ):
        """Skills in deeply nested dirs (e.g. cli-tool/components/skills/dev/my-skill/)
        are found via the GitHub Trees API when candidate paths and shallow scan fail."""
        tree_entries = [
            {"path": "README.md", "type": "blob"},
            {"path": "cli-tool/components/skills/development/my-skill/SKILL.md", "type": "blob"},
            {"path": "cli-tool/components/skills/development/other-skill/SKILL.md", "type": "blob"},
        ]

        def _httpx_get_side_effect(url, **kwargs):
            resp = MagicMock()
            if "/api/search" in url:
                resp.status_code = 404
                return resp
            if url.endswith("/contents/"):
                # Root listing for shallow scan — return empty so it falls through
                resp.status_code = 200
                resp.json = lambda: []
                return resp
            if "/contents/" in url:
                # All contents API calls fail (candidate paths miss)
                resp.status_code = 404
                return resp
            if url.endswith("owner/repo"):
                # Repo info → default branch
                resp.status_code = 200
                resp.json = lambda: {"default_branch": "main"}
                return resp
            if "/git/trees/main" in url:
                resp.status_code = 200
                resp.json = lambda: {"tree": tree_entries}
                return resp
            # skills.sh detail page
            resp.status_code = 200
            resp.text = "<h1>my-skill</h1>"
            return resp

        mock_get.side_effect = _httpx_get_side_effect

        resolved_bundle = SkillBundle(
            name="my-skill",
            files={"SKILL.md": "# My Skill"},
            source="github",
            identifier="owner/repo/cli-tool/components/skills/development/my-skill",
            trust_level="community",
        )
        mock_fetch.side_effect = lambda ident: resolved_bundle if "cli-tool/components" in ident else None

        bundle = self._source().fetch("skills-sh/owner/repo/my-skill")

        assert bundle is not None
        assert bundle.source == "skills.sh"
        assert bundle.files["SKILL.md"] == "# My Skill"
        # Verify the tree-resolved identifier was used for the final GitHub fetch
        mock_fetch.assert_any_call("owner/repo/cli-tool/components/skills/development/my-skill")

class TestFindSkillInRepoTree:
    """Tests for GitHubSource._find_skill_in_repo_tree."""

    def _source(self):
        auth = MagicMock(spec=GitHubAuth)
        auth.get_headers.return_value = {"Accept": "application/vnd.github.v3+json"}
        return GitHubSource(auth=auth)

    @patch("tools.skills_hub.httpx.get")
    def test_finds_deeply_nested_skill(self, mock_get):
        tree_entries = [
            {"path": "README.md", "type": "blob"},
            {"path": "cli-tool/components/skills/development/senior-backend/SKILL.md", "type": "blob"},
            {"path": "cli-tool/components/skills/development/other/SKILL.md", "type": "blob"},
        ]

        def _side_effect(url, **kwargs):
            resp = MagicMock()
            if url.endswith("/davila7/claude-code-templates"):
                resp.status_code = 200
                resp.json = lambda: {"default_branch": "main"}
            elif "/git/trees/main" in url:
                resp.status_code = 200
                resp.json = lambda: {"tree": tree_entries}
            else:
                resp.status_code = 404
            return resp

        mock_get.side_effect = _side_effect

        result = self._source()._find_skill_in_repo_tree("davila7/claude-code-templates", "senior-backend")
        assert result == "davila7/claude-code-templates/cli-tool/components/skills/development/senior-backend"

    @patch("tools.skills_hub.httpx.get")
    def test_returns_none_when_repo_api_fails(self, mock_get):
        mock_get.return_value = MagicMock(status_code=404)
        result = self._source()._find_skill_in_repo_tree("owner/repo", "my-skill")
        assert result is None


class TestWellKnownSkillSource:
    @pytest.fixture(autouse=True)
    def _allow_public_skill_fetches(self, monkeypatch):
        monkeypatch.setattr("tools.skills_hub.is_safe_url", lambda _url: True)
        monkeypatch.setattr("tools.skills_hub.check_website_access", lambda _url: None)

    def _source(self):
        return WellKnownSkillSource()

    @patch("tools.skills_hub._write_index_cache")
    @patch("tools.skills_hub._read_index_cache", return_value=None)
    @patch("tools.skills_hub._ssrf_safe_http_get")
    def test_search_reads_index_from_well_known_url(self, mock_get, _mock_read_cache, _mock_write_cache):
        mock_get.return_value = MagicMock(
            status_code=200,
            json=lambda: {
                "skills": [
                    {"name": "git-workflow", "description": "Git rules", "files": ["SKILL.md"]},
                    {"name": "code-review", "description": "Review code", "files": ["SKILL.md", "references/checklist.md"]},
                ]
            },
        )

        results = self._source().search("https://example.com/.well-known/skills/index.json", limit=10)

        assert [r.identifier for r in results] == [
            "well-known:https://example.com/.well-known/skills/git-workflow",
            "well-known:https://example.com/.well-known/skills/code-review",
        ]
        assert all(r.source == "well-known" for r in results)


    @patch("tools.skills_hub._write_index_cache")
    @patch("tools.skills_hub._read_index_cache", return_value=None)
    @patch("tools.skills_hub.httpx.get")
    def test_fetch_rejects_unsafe_file_paths_from_well_known_endpoint(self, mock_get, _mock_read_cache, _mock_write_cache):
        def fake_get(url, *args, **kwargs):
            if url.endswith("/index.json"):
                return MagicMock(status_code=200, json=lambda: {
                    "skills": [{
                        "name": "code-review",
                        "description": "Review code",
                        "files": ["SKILL.md", "../../../escape.txt"],
                    }]
                })
            if url.endswith("/code-review/SKILL.md"):
                return MagicMock(status_code=200, text="# Code Review\n")
            raise AssertionError(url)

        mock_get.side_effect = fake_get

        bundle = self._source().fetch("well-known:https://example.com/.well-known/skills/code-review")

        assert bundle is None


class TestUrlSource:
    @pytest.fixture(autouse=True)
    def _allow_public_skill_fetches(self, monkeypatch):
        monkeypatch.setattr("tools.skills_hub.is_safe_url", lambda _url: True)
        monkeypatch.setattr("tools.skills_hub.check_website_access", lambda _url: None)

    def _source(self):
        return UrlSource()

    @pytest.mark.parametrize(
        "glob_reference",
        [
            "references/*.md",
            "references/file?",
            "references/file?.md",
            "references/agent.v?.md",
            "references/data.file?.md",
            "references/backup.2026?.md",
            "references/file?x.md",
            "references/agent.v?x.md",
            "references/file?[ab].md",
            "references/file??.md",
            "references/[ab].md",
            "references/%2A.md",
            "references/file%3F.md",
            "references/%5Bab%5D.md",
        ],
    )
    def test_support_path_extraction_ignores_glob_shaped_prose(
        self, glob_reference
    ):
        skill_md = f"""
See [the web reference](references/web.md) for details.
Complex cases are documented under `{glob_reference}`.
"""

        assert _referenced_support_paths(skill_md) == {"references/web.md"}

    @pytest.mark.parametrize(
        ("reference", "expected"),
        [
            ("templates/report.md?raw=1", "templates/report.md"),
            ("references/LICENSE?download", "references/LICENSE"),
            ("references/LICENSE?raw", "references/LICENSE"),
            ("references/guide.md?view", "references/guide.md"),
            ("references/guide.md?inline", "references/guide.md"),
            ("references/guide.md?plain", "references/guide.md"),
            ("references/guide.md?preview-mode", "references/guide.md"),
            ("references/guide.md?view&inline&theme=dark", "references/guide.md"),
            ("references/LICENSE?plain&download=1", "references/LICENSE"),
            ("references/LICENSE?.well-known=1", "references/LICENSE"),
            ("references/guide.md#usage", "references/guide.md"),
            ("references/my%20guide.md", "references/my guide.md"),
            ("references/foo%23bar.md", "references/foo#bar.md"),
        ],
    )
    def test_support_path_extraction_preserves_concrete_url_suffixes(
        self, reference, expected
    ):
        assert _referenced_support_paths(f"Use `{reference}`.") == {expected}

    # ── _matches ────────────────────────────────────────────────────────
    def test_matches_bare_md_url(self):
        assert self._source()._matches("https://example.com/path/SKILL.md") is True

    def test_rejects_well_known_url(self):
        # Leave these for WellKnownSkillSource.
        assert self._source()._matches(
            "https://example.com/.well-known/skills/git-workflow/SKILL.md"
        ) is False
        assert self._source()._matches(
            "https://example.com/.well-known/skills/index.json"
        ) is False

    # ── inspect ─────────────────────────────────────────────────────────

    @patch("tools.skills_hub._ssrf_safe_http_get")
    @patch("tools.skills_hub.check_website_access", return_value=None)
    @patch("tools.skills_hub.is_safe_url", return_value=False)
    def test_inspect_blocks_private_url(self, _mock_safe, _mock_policy, mock_get):
        assert self._source().inspect("http://127.0.0.1/SKILL.md") is None
        mock_get.assert_not_called()

    # ── fetch ───────────────────────────────────────────────────────────


    @patch("tools.skills_hub._ssrf_safe_http_get")
    def test_fetch_skips_missing_support_file_instead_of_aborting(self, mock_get):
        # One referenced support file 404s; the other is reachable. The
        # install should still succeed with the file that could be fetched,
        # rather than aborting the whole fetch (regression for a bug where
        # any single missing companion file failed the entire URL install).
        skill_md = (
            "---\n"
            "name: my-skill\n"
            "description: Has support files.\n"
            "---\n\n"
            "See `references/good.md` and `references/missing.md`.\n"
        )

        def _side_effect(url, **kwargs):
            if url.endswith("SKILL.md"):
                return MagicMock(status_code=200, text=skill_md)
            if url.endswith("references/good.md"):
                return MagicMock(status_code=200, content=b"good content")
            if url.endswith("references/missing.md"):
                return MagicMock(status_code=404)
            raise AssertionError(f"unexpected URL: {url}")

        mock_get.side_effect = _side_effect

        bundle = self._source().fetch("https://example.com/my-skill/SKILL.md")
        assert bundle is not None
        assert bundle.name == "my-skill"
        assert bundle.files["SKILL.md"] == skill_md
        assert bundle.files["references/good.md"] == b"good content"
        assert "references/missing.md" not in bundle.files

    @patch("tools.skills_hub._ssrf_safe_http_get")
    @patch("tools.skills_hub.check_website_access", return_value=None)
    @patch("tools.skills_hub.is_safe_url", side_effect=[True, False])
    def test_fetch_blocks_redirect_to_private_url(self, _mock_safe, _mock_policy, mock_get):
        redirect = MagicMock(status_code=302)
        redirect.headers = {"location": "http://127.0.0.1/private/SKILL.md"}
        mock_get.return_value = redirect

        assert self._source().fetch("https://example.com/SKILL.md") is None
        assert mock_get.call_count == 1

    @patch("tools.skills_hub._ssrf_safe_http_get")
    @patch("tools.skills_hub.check_website_access", return_value=None)
    @patch("tools.skills_hub.is_safe_url", return_value=False)
    def test_fetch_blocks_private_url(self, _mock_safe, _mock_policy, mock_get):
        assert self._source().fetch("http://127.0.0.1/SKILL.md") is None
        mock_get.assert_not_called()


    def test_is_valid_skill_name_rejects_sentinel_and_garbage(self):
        invalid = [
            "",
            "SKILL", "skill", "README", "readme", "INDEX", "index",
            "unnamed-skill",
            "../evil", "a/b", "has space", "has.dot",
            "-leading-dash", "1-leading-digit",
            None, 123, ["list"],
        ]
        for name in invalid:
            assert not UrlSource._is_valid_skill_name(name), f"should reject {name!r}"


class TestCheckForSkillUpdates:
    def test_bundle_content_hash_matches_installed_content_hash(self, tmp_path):
        from tools.skills_guard import content_hash

        bundle = SkillBundle(
            name="demo-skill",
            files={
                "SKILL.md": "same content",
                "references/checklist.md": "- [ ] security\n",
            },
            source="github",
            identifier="owner/repo/demo-skill",
            trust_level="community",
        )
        skill_dir = tmp_path / "demo-skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text("same content")
        (skill_dir / "references").mkdir()
        (skill_dir / "references" / "checklist.md").write_text("- [ ] security\n")

        assert bundle_content_hash(bundle) == content_hash(skill_dir)


    def test_reports_update_when_remote_hash_differs(self):
        lock = MagicMock()
        lock.list_installed.return_value = [{
            "name": "demo-skill",
            "source": "github",
            "identifier": "owner/repo/demo-skill",
            "content_hash": "oldhash",
            "install_path": "demo-skill",
        }]

        source = MagicMock()
        source.source_id.return_value = "github"
        source.fetch.return_value = SkillBundle(
            name="demo-skill",
            files={"SKILL.md": "new content"},
            source="github",
            identifier="owner/repo/demo-skill",
            trust_level="community",
        )

        results = check_for_skill_updates(lock=lock, sources=[source])

        assert len(results) == 1
        assert results[0]["name"] == "demo-skill"
        assert results[0]["status"] == "update_available"

class TestCreateSourceRouter:

    def test_url_source_runs_before_github_source(self):
        # UrlSource must win over GitHubSource when both could claim a URL.
        sources = create_source_router(auth=MagicMock(spec=GitHubAuth))
        url_idx = next(i for i, src in enumerate(sources) if isinstance(src, UrlSource))
        gh_idx = next(i for i, src in enumerate(sources) if isinstance(src, GitHubSource))
        assert url_idx < gh_idx


# ---------------------------------------------------------------------------
# HubLockFile
# ---------------------------------------------------------------------------


class TestHubLockFile:

    def test_load_corrupt_json(self, tmp_path):
        lock_file = tmp_path / "lock.json"
        lock_file.write_text("not json{{{")
        lock = HubLockFile(path=lock_file)
        data = lock.load()
        assert data == {"version": 1, "installed": {}}


    def test_list_installed(self, tmp_path):
        lock = HubLockFile(path=tmp_path / "lock.json")
        lock.record_install(
            name="s1", source="github", identifier="x",
            trust_level="trusted", scan_verdict="pass",
            skill_hash="h1", install_path="s1", files=["SKILL.md"],
        )
        lock.record_install(
            name="s2", source="clawhub", identifier="y",
            trust_level="community", scan_verdict="pass",
            skill_hash="h2", install_path="s2", files=["SKILL.md"],
        )
        installed = lock.list_installed()
        assert len(installed) == 2
        names = {e["name"] for e in installed}
        assert names == {"s1", "s2"}


# ---------------------------------------------------------------------------
# TapsManager
# ---------------------------------------------------------------------------


class TestTapsManager:

    def test_load_corrupt_json(self, tmp_path):
        taps_file = tmp_path / "taps.json"
        taps_file.write_text("bad json")
        mgr = TapsManager(path=taps_file)
        assert mgr.load() == []


    def test_remove_existing_tap(self, tmp_path):
        mgr = TapsManager(path=tmp_path / "taps.json")
        mgr.add("owner/repo")
        assert mgr.remove("owner/repo") is True
        assert mgr.load() == []

# ---------------------------------------------------------------------------
# LobeHubSource._convert_to_skill_md
# ---------------------------------------------------------------------------


class TestConvertToSkillMd:
    def test_basic_conversion(self):
        agent_data = {
            "identifier": "test-agent",
            "meta": {
                "title": "Test Agent",
                "description": "A test agent.",
                "tags": ["testing", "demo"],
            },
            "config": {
                "systemRole": "You are a helpful test agent.",
            },
        }
        result = LobeHubSource._convert_to_skill_md(agent_data)
        assert "---" in result
        assert "name: test-agent" in result
        assert "description: A test agent." in result
        assert "tags: [testing, demo]" in result
        assert "# Test Agent" in result
        assert "You are a helpful test agent." in result

# ---------------------------------------------------------------------------
# unified_search — dedup logic
# ---------------------------------------------------------------------------


class TestUnifiedSearchDedup:
    def _make_source(self, source_id, results):
        """Create a mock SkillSource that returns fixed results."""
        src = MagicMock()
        src.source_id.return_value = source_id
        src.search.return_value = results
        return src

    def test_dedup_keeps_first_seen(self):
        # Same identifier from two sources — only the first (community) is kept when equal trust.
        s1 = SkillMeta(name="skill", description="from A", source="a",
                        identifier="shared/skill", trust_level="community")
        s2 = SkillMeta(name="skill", description="from B", source="b",
                        identifier="shared/skill", trust_level="community")
        src_a = self._make_source("a", [s1])
        src_b = self._make_source("b", [s2])
        results = unified_search("skill", [src_a, src_b])
        assert len(results) == 1
        assert results[0].description == "from A"

    def test_dedup_prefers_builtin_over_trusted(self):
        """Regression: builtin must not be overwritten by trusted."""
        builtin = SkillMeta(name="skill", description="builtin", source="a",
                             identifier="shared/skill", trust_level="builtin")
        trusted = SkillMeta(name="skill", description="trusted", source="b",
                             identifier="shared/skill", trust_level="trusted")
        src_a = self._make_source("a", [builtin])
        src_b = self._make_source("b", [trusted])
        results = unified_search("skill", [src_a, src_b])
        assert len(results) == 1
        assert results[0].trust_level == "builtin"


    def test_source_error_handled(self):
        failing = MagicMock()
        failing.source_id.return_value = "fail"
        failing.search.side_effect = RuntimeError("boom")
        ok = self._make_source("ok", [
            SkillMeta(name="s1", description="d", source="ok",
                       identifier="x", trust_level="community")
        ])
        results = unified_search("query", [failing, ok])
        assert len(results) == 1


# ---------------------------------------------------------------------------
# GitHub tap provider labeling + index search/filter
# ---------------------------------------------------------------------------


class TestGithubProviderLabeling:

    def test_inspect_stamps_provider_in_extra(self):
        gs = GitHubSource(auth=GitHubAuth())
        skill_md = (
            "---\nname: accelerated-computing-cudf\n"
            "description: NVIDIA cuDF GPU DataFrames.\n---\n# body\n"
        )
        gs._fetch_file_content = lambda repo, path: skill_md
        meta = gs.inspect("NVIDIA/skills/skills/accelerated-computing-cudf")
        assert meta is not None
        # source stays "github" (no churn to dedup/floor/skip logic) ...
        assert meta.source == "github"
        # ... but the per-tap provider label rides along in extra
        assert meta.extra.get("provider") == "NVIDIA"

def _make_index_source(skills):
    """Build a HermesIndexSource pre-loaded with a fixed skill list."""
    from tools.skills_hub import HermesIndexSource
    src = HermesIndexSource(auth=GitHubAuth())
    src._index = {"skills": skills}
    src._loaded = True
    return src


class TestHermesIndexSearch:
    def test_search_matches_identifier_and_provider(self):
        # NVIDIA skill whose name/description does NOT contain "nvidia" — only
        # the identifier and the provider label do. The old substring-only
        # search over name/description/tags would miss it entirely.
        skills = [
            {
                "name": "accelerated-computing-cudf",
                "description": "GPU DataFrames.",
                "source": "github",
                "identifier": "NVIDIA/skills/skills/accelerated-computing-cudf",
                "tags": [],
                "extra": {"provider": "NVIDIA"},
            },
            {
                "name": "unrelated",
                "description": "nothing here",
                "source": "clawhub",
                "identifier": "clawhub/unrelated",
                "tags": [],
            },
        ]
        src = _make_index_source(skills)
        hits = src.search("nvidia", limit=25)
        ids = [h.identifier for h in hits]
        assert "NVIDIA/skills/skills/accelerated-computing-cudf" in ids
        assert "clawhub/unrelated" not in ids

class TestProviderFilter:
    def test_filter_results_by_provider_narrows_exactly(self):
        from tools.skills_hub import _filter_results_by_provider
        results = [
            SkillMeta(name="a", description="", source="github", identifier="NVIDIA/skills/a",
                      trust_level="trusted", extra={"provider": "NVIDIA"}),
            SkillMeta(name="b", description="", source="github", identifier="openai/skills/b",
                      trust_level="trusted", extra={"provider": "OpenAI"}),
            SkillMeta(name="c", description="", source="official", identifier="official/c",
                      trust_level="builtin"),
        ]
        nv = _filter_results_by_provider(results, "nvidia")
        assert [r.identifier for r in nv] == ["NVIDIA/skills/a"]
        oai = _filter_results_by_provider(results, "openai")
        assert [r.identifier for r in oai] == ["openai/skills/b"]

    def test_unified_search_provider_filter_keeps_index_source(self):
        # A provider filter must NOT be treated as a real source id (which would
        # exclude every source and return nothing). It selects sources like
        # "all", then narrows the merged results by provider.
        nv = SkillMeta(name="cuda", description="gpu", source="github",
                       identifier="NVIDIA/skills/cuda", trust_level="trusted",
                       extra={"provider": "NVIDIA"})
        other = SkillMeta(name="cuda-clone", description="gpu", source="clawhub",
                          identifier="clawhub/cuda-clone", trust_level="community")
        src = MagicMock()
        src.source_id.return_value = "hermes-index"
        src.is_available = True
        src.search.return_value = [nv, other]
        results = unified_search("cuda", [src], source_filter="nvidia", limit=25)
        assert [r.identifier for r in results] == ["NVIDIA/skills/cuda"]


# ---------------------------------------------------------------------------
# append_audit_log
# ---------------------------------------------------------------------------


class TestAppendAuditLog:
    def test_creates_log_entry(self, tmp_path):
        log_file = tmp_path / "audit.log"
        with patch("tools.skills_hub.AUDIT_LOG", log_file):
            append_audit_log("INSTALL", "test-skill", "github", "trusted", "pass")
        content = log_file.read_text()
        assert "INSTALL" in content
        assert "test-skill" in content
        assert "github:trusted" in content
        assert "pass" in content

# ---------------------------------------------------------------------------
# Official skills / binary assets
# ---------------------------------------------------------------------------


class TestOptionalSkillSourceMetadata:
    def test_scan_all_emits_repo_root_relative_metadata(self, tmp_path):
        optional_root = tmp_path / "optional-skills"
        skill_dir = optional_root / "finance" / "3-statement-model"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(
            "---\n"
            "name: 3-statement-model\n"
            "description: test\n"
            "metadata:\n"
            "  hermes:\n"
            "    editorial_name: Three-Statement Model\n"
            "    editorial_description: Build a linked financial model.\n"
            "---\n\nBody\n",
            encoding="utf-8",
        )

        src = OptionalSkillSource()
        src._optional_dir = optional_root

        meta = src.inspect("official/finance/3-statement-model")

        assert meta is not None
        assert meta.repo == "NousResearch/hermes-agent"
        assert meta.path == "optional-skills/finance/3-statement-model"
        assert meta.editorial_name == "Three-Statement Model"
        assert meta.editorial_description == "Build a linked financial model."

    def test_scan_all_accepts_install_prefix_but_rejects_nested_support_skills(self, tmp_path):
        optional_root = tmp_path / "venv" / "lib" / "site-packages" / "optional-skills"
        real = optional_root / "research" / "real-skill"
        nested = real / "references" / "archived-skill"
        nested.mkdir(parents=True)
        (real / "SKILL.md").write_text(
            "---\nname: real-skill\ndescription: real\n---\n", encoding="utf-8"
        )
        (nested / "SKILL.md").write_text(
            "---\nname: archived-skill\ndescription: nested\n---\n", encoding="utf-8"
        )

        src = OptionalSkillSource()
        src._optional_dir = optional_root

        assert [meta.name for meta in src._scan_all()] == ["real-skill"]
        assert src._find_skill_dir("archived-skill") is None


class TestOptionalSkillSourceBinaryAssets:
    def test_fetch_preserves_binary_assets(self, tmp_path):
        optional_root = tmp_path / "optional-skills"
        skill_dir = optional_root / "mlops" / "models" / "neutts"
        (skill_dir / "assets" / "neutts-cli" / "samples").mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(
            "---\nname: neutts\ndescription: test\n---\n\nBody\n",
            encoding="utf-8",
        )
        wav_bytes = b"RIFF\x00\x01fakewav"
        (skill_dir / "assets" / "neutts-cli" / "samples" / "jo.wav").write_bytes(
            wav_bytes
        )
        (skill_dir / "assets" / "neutts-cli" / "samples" / "jo.txt").write_text(
            "hello\n", encoding="utf-8"
        )
        pycache_dir = skill_dir / "assets" / "neutts-cli" / "src" / "neutts_cli" / "__pycache__"
        pycache_dir.mkdir(parents=True)
        (pycache_dir / "cli.cpython-312.pyc").write_bytes(b"junk")

        src = OptionalSkillSource()
        src._optional_dir = optional_root

        bundle = src.fetch("official/mlops/models/neutts")

        assert bundle is not None
        assert bundle.files["assets/neutts-cli/samples/jo.wav"] == wav_bytes
        assert bundle.files["assets/neutts-cli/samples/jo.txt"] == b"hello\n"
        assert "assets/neutts-cli/src/neutts_cli/__pycache__/cli.cpython-312.pyc" not in bundle.files

    def test_fetch_rejects_sibling_directory_traversal(self, tmp_path):
        optional_root = tmp_path / "optional-skills"
        sibling_skill_dir = tmp_path / "optional-skills-escape" / "pwned"
        optional_root.mkdir()
        sibling_skill_dir.mkdir(parents=True)
        (sibling_skill_dir / "SKILL.md").write_text(
            "---\nname: pwned\ndescription: traversal\n---\n\nBody\n",
            encoding="utf-8",
        )

        src = OptionalSkillSource()
        src._optional_dir = optional_root

        bundle = src.fetch("official/../optional-skills-escape/pwned")

        assert bundle is None


class TestOptionalSkillSourceLiveRepoFallback:
    """Skills merged to main after the local install was cut must still be
    searchable and installable without `hermes update` (live-repo fallback)."""

    def _make_source(self, tmp_path, remote_dirs):
        optional_root = tmp_path / "optional-skills"
        optional_root.mkdir(exist_ok=True)
        src = OptionalSkillSource()
        src._optional_dir = optional_root
        src._remote_dirs = dict.fromkeys(remote_dirs, True)
        return src

    @staticmethod
    def _fake_github_with_tree(remote_dirs, extra_files=()):
        """MagicMock GitHubSource whose repo tree contains each skill dir's
        SKILL.md plus any extra files, served byte-exact by _fetch_file_bytes."""
        entries = []
        contents = {}
        for rel_dir in remote_dirs:
            p = f"optional-skills/{rel_dir}/SKILL.md"
            entries.append({"type": "blob", "path": p, "mode": "100644"})
            contents[p] = b"---\nname: " + rel_dir.rsplit("/", 1)[-1].encode() + b"\n---\nBody"
        for rel_path, data in extra_files:
            entries.append({"type": "blob", "path": rel_path, "mode": "100644"})
            contents[rel_path] = data
        fake = MagicMock()
        fake._get_repo_tree.return_value = ("main", entries)
        fake._fetch_file_bytes.side_effect = lambda repo, path: contents.get(path)
        return fake

    def test_fetch_falls_back_to_live_repo_when_missing_locally(self, tmp_path):
        src = self._make_source(tmp_path, ["software-development/ast-grep"])
        src._github = self._fake_github_with_tree(
            ["software-development/ast-grep"],
            extra_files=[
                ("optional-skills/software-development/ast-grep/install.sh", b"#!/bin/sh\n"),
                ("optional-skills/software-development/ast-grep/LICENSE", b"MIT"),
            ],
        )

        bundle = src.fetch("official/software-development/ast-grep")

        assert bundle is not None
        # Provenance is rewritten to official/builtin
        assert bundle.source == "official"
        assert bundle.identifier == "official/software-development/ast-grep"
        assert bundle.trust_level == "builtin"
        # FULL directory arrives — including root-level files GitHubSource.fetch drops
        assert bundle.files["install.sh"] == b"#!/bin/sh\n"
        assert bundle.files["LICENSE"] == b"MIT"

    def test_fetch_bare_name_resolves_via_remote_tree(self, tmp_path):
        src = self._make_source(tmp_path, ["software-development/ast-grep"])
        src._github = self._fake_github_with_tree(["software-development/ast-grep"])

        bundle = src.fetch("official/ast-grep")

        assert bundle is not None
        assert bundle.identifier == "official/software-development/ast-grep"

    def test_fetch_ambiguous_bare_name_refuses(self, tmp_path):
        src = self._make_source(
            tmp_path, ["security/scanner", "devops/scanner"]
        )
        fake_github = MagicMock()
        src._github = fake_github

        assert src.fetch("official/scanner") is None
        fake_github.fetch.assert_not_called()

    def test_local_checkout_wins_over_remote(self, tmp_path):
        src = self._make_source(tmp_path, ["research/local-skill"])
        skill_dir = src._optional_dir / "research" / "local-skill"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(
            "---\nname: local-skill\ndescription: local\n---\nBody",
            encoding="utf-8",
        )
        fake_github = MagicMock()
        src._github = fake_github

        bundle = src.fetch("official/research/local-skill")

        assert bundle is not None
        fake_github.fetch.assert_not_called()

    def test_fallback_rejects_traversal_rel(self, tmp_path):
        src = self._make_source(tmp_path, ["security/whatever"])
        fake_github = MagicMock()
        src._github = fake_github

        assert src._fetch_from_live_repo("../../etc/passwd") is None
        fake_github.fetch.assert_not_called()

    def test_search_surfaces_remote_only_skills(self, tmp_path):
        src = self._make_source(tmp_path, ["software-development/ast-grep"])

        results = src.search("ast-grep")

        assert any(
            r.identifier == "official/software-development/ast-grep"
            and r.trust_level == "builtin"
            for r in results
        )

    def test_inspect_surfaces_remote_only_skill(self, tmp_path):
        src = self._make_source(tmp_path, ["software-development/ast-grep"])

        meta = src.inspect("official/software-development/ast-grep")

        assert meta is not None
        assert meta.repo == "NousResearch/hermes-agent"
        assert meta.path == "optional-skills/software-development/ast-grep"

    def test_offline_degrades_to_local_only(self, tmp_path):
        src = self._make_source(tmp_path, [])
        fake_github = MagicMock()
        src._github = fake_github

        assert src.fetch("official/never-heard-of-it") is None
        assert src.search("never-heard-of-it") == []


class TestQuarantineBundleBinaryAssets:
    def test_quarantine_bundle_writes_binary_files(self, tmp_path):
        import tools.skills_hub as hub

        hub_dir = tmp_path / "skills" / ".hub"
        with patch.object(hub, "SKILLS_DIR", tmp_path / "skills"), \
             patch.object(hub, "HUB_DIR", hub_dir), \
             patch.object(hub, "LOCK_FILE", hub_dir / "lock.json"), \
             patch.object(hub, "QUARANTINE_DIR", hub_dir / "quarantine"), \
             patch.object(hub, "AUDIT_LOG", hub_dir / "audit.log"), \
             patch.object(hub, "TAPS_FILE", hub_dir / "taps.json"), \
             patch.object(hub, "INDEX_CACHE_DIR", hub_dir / "index-cache"):
            bundle = SkillBundle(
                name="neutts",
                files={
                    "SKILL.md": "---\nname: neutts\n---\n",
                    "assets/neutts-cli/samples/jo.wav": b"RIFF\x00\x01fakewav",
                },
                source="official",
                identifier="official/mlops/models/neutts",
                trust_level="builtin",
            )

            q_path = quarantine_bundle(bundle)

        assert (q_path / "SKILL.md").read_text(encoding="utf-8").startswith("---")
        assert (q_path / "assets" / "neutts-cli" / "samples" / "jo.wav").read_bytes() == b"RIFF\x00\x01fakewav"

    def test_quarantine_bundle_rejects_traversal_file_paths(self, tmp_path):
        import tools.skills_hub as hub

        hub_dir = tmp_path / "skills" / ".hub"
        with patch.object(hub, "SKILLS_DIR", tmp_path / "skills"), \
             patch.object(hub, "HUB_DIR", hub_dir), \
             patch.object(hub, "LOCK_FILE", hub_dir / "lock.json"), \
             patch.object(hub, "QUARANTINE_DIR", hub_dir / "quarantine"), \
             patch.object(hub, "AUDIT_LOG", hub_dir / "audit.log"), \
             patch.object(hub, "TAPS_FILE", hub_dir / "taps.json"), \
             patch.object(hub, "INDEX_CACHE_DIR", hub_dir / "index-cache"):
            bundle = SkillBundle(
                name="demo",
                files={
                    "SKILL.md": "---\nname: demo\n---\n",
                    "../../../escape.txt": "owned",
                },
                source="well-known",
                identifier="well-known:https://example.com/.well-known/skills/demo",
                trust_level="community",
            )

            with pytest.raises(ValueError, match="Unsafe bundle file path"):
                quarantine_bundle(bundle)

        assert not (tmp_path / "skills" / "escape.txt").exists()

    def test_quarantine_bundle_rejects_absolute_file_paths(self, tmp_path):
        import tools.skills_hub as hub

        hub_dir = tmp_path / "skills" / ".hub"
        absolute_target = tmp_path / "outside.txt"
        with patch.object(hub, "SKILLS_DIR", tmp_path / "skills"), \
             patch.object(hub, "HUB_DIR", hub_dir), \
             patch.object(hub, "LOCK_FILE", hub_dir / "lock.json"), \
             patch.object(hub, "QUARANTINE_DIR", hub_dir / "quarantine"), \
             patch.object(hub, "AUDIT_LOG", hub_dir / "audit.log"), \
             patch.object(hub, "TAPS_FILE", hub_dir / "taps.json"), \
             patch.object(hub, "INDEX_CACHE_DIR", hub_dir / "index-cache"):
            bundle = SkillBundle(
                name="demo",
                files={
                    "SKILL.md": "---\nname: demo\n---\n",
                    str(absolute_target): "owned",
                },
                source="well-known",
                identifier="well-known:https://example.com/.well-known/skills/demo",
                trust_level="community",
            )

            with pytest.raises(ValueError, match="Unsafe bundle file path"):
                quarantine_bundle(bundle)

        assert not absolute_target.exists()

    def test_quarantine_bundle_rejects_ads_colon_file_paths(self, tmp_path):
        """F-02: a bundle member with a colon in a component (``file.py:payload``)
        is an NTFS Alternate Data Stream marker — the visible file passes
        ``rglob``-based review while hidden, scanner-invisible bytes are written
        into it. Reject it before it reaches disk, on any OS."""
        import tools.skills_hub as hub

        hub_dir = tmp_path / "skills" / ".hub"
        with patch.object(hub, "SKILLS_DIR", tmp_path / "skills"), \
             patch.object(hub, "HUB_DIR", hub_dir), \
             patch.object(hub, "LOCK_FILE", hub_dir / "lock.json"), \
             patch.object(hub, "QUARANTINE_DIR", hub_dir / "quarantine"), \
             patch.object(hub, "AUDIT_LOG", hub_dir / "audit.log"), \
             patch.object(hub, "TAPS_FILE", hub_dir / "taps.json"), \
             patch.object(hub, "INDEX_CACHE_DIR", hub_dir / "index-cache"):
            bundle = SkillBundle(
                name="demo",
                files={
                    "SKILL.md": "---\nname: demo\n---\n",
                    "scripts/helper.py:payload": "print(24680)",
                },
                source="well-known",
                identifier="well-known:https://example.com/.well-known/skills/demo",
                trust_level="community",
            )

            with pytest.raises(ValueError, match="Unsafe bundle file path"):
                quarantine_bundle(bundle)

        assert not (tmp_path / "skills" / "scripts").exists()

    def test_normalize_bundle_path_rejects_colon_anywhere(self):
        """The colon guard covers the whole class, not just ``helper.py:payload``:
        a colon in any component (leading drive letter, mid-path, or bare) is
        rejected, while ordinary portable paths still normalize."""
        from tools.skills_hub import _normalize_bundle_path

        rejected = (
            "scripts/helper.py:payload",   # trailing-component ADS marker
            "scripts/a:b.py",              # mid-component colon
            "a:b/scripts/helper.py",       # leading-component colon
            "scripts/helper.py:",          # empty stream name
            "C:",                          # bare Windows drive letter
            "C:/Windows/System32",         # drive-qualified absolute-ish path
        )
        for bad in rejected:
            with pytest.raises(ValueError, match="Unsafe bundle file path"):
                _normalize_bundle_path(bad, field_name="bundle file path", allow_nested=True)

        # Legitimate portable paths are unaffected.
        assert _normalize_bundle_path(
            "scripts/helper.py", field_name="bundle file path", allow_nested=True
        ) == "scripts/helper.py"
        assert _normalize_bundle_path(
            "assets/data/sample.wav", field_name="bundle file path", allow_nested=True
        ) == "assets/data/sample.wav"


# ---------------------------------------------------------------------------
# GitHubSource._download_directory — tree API + fallback (#2940)
# ---------------------------------------------------------------------------


class TestDownloadDirectoryViaTree:
    """Tests for the Git Trees API path in _download_directory."""

    def _source(self):
        auth = MagicMock(spec=GitHubAuth)
        auth.get_headers.return_value = {}
        return GitHubSource(auth=auth)

    @patch.object(GitHubSource, "_fetch_file_content")
    @patch("tools.skills_hub.httpx.get")
    def test_tree_api_downloads_subdirectories(self, mock_get, mock_fetch):
        """Tree API returns files from nested subdirectories."""
        repo_resp = MagicMock(status_code=200, json=lambda: {"default_branch": "main"})
        tree_resp = MagicMock(status_code=200, json=lambda: {
            "truncated": False,
            "tree": [
                {"type": "blob", "path": "skills/my-skill/SKILL.md"},
                {"type": "blob", "path": "skills/my-skill/scripts/run.py"},
                {"type": "blob", "path": "skills/my-skill/references/api.md"},
                {"type": "tree", "path": "skills/my-skill/scripts"},
                {"type": "blob", "path": "other/file.txt"},
            ],
        })
        mock_get.side_effect = [repo_resp, tree_resp]
        mock_fetch.side_effect = lambda repo, path: f"content-of-{path}"

        src = self._source()
        files = src._download_directory("owner/repo", "skills/my-skill")

        assert "SKILL.md" in files
        assert "scripts/run.py" in files
        assert "references/api.md" in files
        assert "other/file.txt" not in files  # outside target path
        assert len(files) == 3

    @patch.object(GitHubSource, "_download_directory_recursive", return_value={"SKILL.md": "# ok"})
    @patch("tools.skills_hub.httpx.get")
    def test_falls_back_on_truncated_tree(self, mock_get, mock_fallback):
        """When tree is truncated, fall back to recursive Contents API."""
        repo_resp = MagicMock(status_code=200, json=lambda: {"default_branch": "main"})
        tree_resp = MagicMock(status_code=200, json=lambda: {"truncated": True, "tree": []})
        mock_get.side_effect = [repo_resp, tree_resp]

        src = self._source()
        files = src._download_directory("owner/repo", "skills/my-skill")

        assert files == {"SKILL.md": "# ok"}
        mock_fallback.assert_called_once_with("owner/repo", "skills/my-skill")

class TestDownloadDirectoryRecursive:
    """Tests for the Contents API fallback path."""

    def _source(self):
        auth = MagicMock(spec=GitHubAuth)
        auth.get_headers.return_value = {}
        return GitHubSource(auth=auth)

    @patch.object(GitHubSource, "_fetch_file_content")
    @patch("tools.skills_hub.httpx.get")
    def test_recursive_downloads_subdirectories(self, mock_get, mock_fetch):
        """Contents API recursion includes subdirectories."""
        root_resp = MagicMock(status_code=200, json=lambda: [
            {"name": "SKILL.md", "type": "file", "path": "skill/SKILL.md"},
            {"name": "scripts", "type": "dir", "path": "skill/scripts"},
        ])
        sub_resp = MagicMock(status_code=200, json=lambda: [
            {"name": "run.py", "type": "file", "path": "skill/scripts/run.py"},
        ])
        mock_get.side_effect = [root_resp, sub_resp]
        mock_fetch.side_effect = lambda repo, path: f"content-of-{path}"

        src = self._source()
        files = src._download_directory_recursive("owner/repo", "skill")

        assert "SKILL.md" in files
        assert "scripts/run.py" in files

# ---------------------------------------------------------------------------
# Install-path safety (lock-file → uninstall rmtree boundary)
# ---------------------------------------------------------------------------


class TestInstallPathSafety:
    """Guard the lock-file → ``uninstall_skill`` rmtree path.

    The destructive boundary is ``shutil.rmtree(SKILLS_DIR / install_path)``.
    Lock-file ``install_path`` values that are absolute, contain ``..``,
    point at the skills root itself, or are redirected via a symlink/junction
    inside ``skills/`` must be rejected before they reach rmtree.
    """

    @pytest.fixture
    def isolated_skills_dir(self, tmp_path):
        import tools.skills_hub as hub

        skills_dir = tmp_path / "skills"
        skills_dir.mkdir()
        # SKILLS_DIR is a PEP 562 dynamic attribute (resolved per-access via
        # module __getattr__). monkeypatch.setattr must NOT be used here: it
        # captures __getattr__'s live-resolved Path as the "original" value
        # and re-installs it as a REAL module attribute on teardown, which
        # permanently shadows dynamic resolution — every later test in the
        # process then sees this test's tmp dir as the skills root. Set the
        # override attribute directly and delete it on teardown so the
        # module returns to dynamic resolution.
        setattr(hub, "SKILLS_DIR", skills_dir)
        try:
            yield skills_dir
        finally:
            try:
                delattr(hub, "SKILLS_DIR")
            except AttributeError:
                pass

    @pytest.fixture
    def patch_lock_file(self, monkeypatch):
        """Redirect HubLockFile's default path to a test-controlled file.

        HubLockFile.__init__ captures LOCK_FILE as a default arg at class
        definition time, so monkeypatching the module-level LOCK_FILE doesn't
        affect later HubLockFile() calls. Patch __defaults__ instead.
        """
        def _apply(lock_path):
            monkeypatch.setattr(HubLockFile.__init__, "__defaults__", (lock_path,))
        return _apply

    def test_record_install_rejects_unsafe_paths(self, tmp_path):
        """record_install must reject malformed install_path values at write time."""
        lock = HubLockFile(path=tmp_path / "lock.json")
        for bad_install_path in (
            "",
            ".",
            "..",
            "../../etc/passwd",
            "/etc/passwd",
            "skills/../../tmp",
            "C:/Windows/System32",
        ):
            with pytest.raises(ValueError, match="Unsafe"):
                lock.record_install(
                    name="evil",
                    source="github",
                    identifier="x",
                    trust_level="trusted",
                    scan_verdict="pass",
                    skill_hash="h1",
                    install_path=bad_install_path,
                    files=["SKILL.md"],
                )


    def test_uninstall_rejects_poisoned_absolute_path(self, tmp_path, isolated_skills_dir, patch_lock_file):
        """Hand-edited lock.json with absolute install_path must not delete anything."""
        from tools.skills_hub import uninstall_skill

        lock_path = tmp_path / "lock.json"
        target = tmp_path / "victim"
        target.mkdir()
        (target / "file.txt").write_text("important")

        # Bypass record_install's validator to simulate a poisoned lock file.
        lock_path.write_text(json.dumps({
            "installed": {
                "evil": {
                    "source": "github",
                    "identifier": "x",
                    "trust_level": "trusted",
                    "scan_verdict": "pass",
                    "content_hash": "h",
                    "install_path": str(target),
                    "files": [],
                    "metadata": {},
                    "installed_at": "now",
                    "updated_at": "now",
                }
            }
        }))

        patch_lock_file(lock_path)
        ok, msg = uninstall_skill("evil")
        assert ok is False
        assert "Unsafe" in msg or "Refusing" in msg
        assert target.exists()
        assert (target / "file.txt").read_text() == "important"

    def test_uninstall_rejects_traversal(self, tmp_path, isolated_skills_dir, patch_lock_file):
        from tools.skills_hub import uninstall_skill

        lock_path = tmp_path / "lock.json"
        sibling = tmp_path / "sibling"
        sibling.mkdir()
        (sibling / "data").write_text("nope")

        lock_path.write_text(json.dumps({
            "installed": {
                "evil": {
                    "source": "github", "identifier": "x",
                    "trust_level": "trusted", "scan_verdict": "pass",
                    "content_hash": "h",
                    "install_path": "../sibling",
                    "files": [], "metadata": {},
                    "installed_at": "now", "updated_at": "now",
                }
            }
        }))

        patch_lock_file(lock_path)
        ok, msg = uninstall_skill("evil")
        assert ok is False
        assert sibling.exists()
        assert (sibling / "data").read_text() == "nope"

    def test_uninstall_rejects_empty_install_path(self, tmp_path, isolated_skills_dir, patch_lock_file):
        """Empty install_path resolves to SKILLS_DIR itself — must be refused."""
        from tools.skills_hub import uninstall_skill

        # Put a sibling skill alongside to prove rmtree doesn't fire.
        (isolated_skills_dir / "bystander").mkdir()
        (isolated_skills_dir / "bystander" / "SKILL.md").write_text("safe")

        lock_path = tmp_path / "lock.json"
        lock_path.write_text(json.dumps({
            "installed": {
                "evil": {
                    "source": "github", "identifier": "x",
                    "trust_level": "trusted", "scan_verdict": "pass",
                    "content_hash": "h",
                    "install_path": "",
                    "files": [], "metadata": {},
                    "installed_at": "now", "updated_at": "now",
                }
            }
        }))

        patch_lock_file(lock_path)
        ok, msg = uninstall_skill("evil")
        assert ok is False
        assert (isolated_skills_dir / "bystander" / "SKILL.md").read_text() == "safe"


    def test_install_from_quarantine_rejects_symlinks(self, tmp_path):
        """Skill install must not follow symlinks that leak file contents
        from outside the quarantine directory."""
        import tools.skills_hub as hub
        from tools.skills_guard import ScanResult

        skills_dir = tmp_path / "skills"
        quarantine_root = skills_dir / ".hub" / "quarantine"
        quarantine_root.mkdir(parents=True)

        q_dir = quarantine_root / "pending"
        q_dir.mkdir()
        (q_dir / "SKILL.md").write_text("---\nname: bad-skill\n---\n")

        secret = tmp_path / "secret.txt"
        secret.write_text("data exfiltration payload\n")

        leak = q_dir / "leak.txt"
        try:
            leak.symlink_to(secret)
        except (OSError, NotImplementedError):
            pytest.skip("symlink creation unsupported on this platform")

        bundle = hub.SkillBundle(
            name="bad-skill",
            files={"SKILL.md": "---\nname: bad-skill\n---\n"},
            source="community",
            identifier="x",
            trust_level="community",
        )
        scan_result = ScanResult(
            skill_name="bad-skill",
            source="community",
            trust_level="community",
            verdict="safe",
        )

        with patch.object(hub, "SKILLS_DIR", skills_dir), \
             patch.object(hub, "QUARANTINE_DIR", quarantine_root), \
             patch("tools.skill_usage.record_installed") as record_installed:
            with pytest.raises(ValueError, match="symlink"):
                hub.install_from_quarantine(
                    q_dir, "bad-skill", "", bundle, scan_result,
                )

        record_installed.assert_not_called()
        assert not (skills_dir / "bad-skill" / "leak.txt").exists()
        assert secret.read_text() == "data exfiltration payload\n"

    def test_install_from_quarantine_rejects_category_bucket_overwrite(self, tmp_path):
        """Installing a skill whose name matches an existing category directory
        that contains other skills must NOT silently wipe that entire directory.

        Regression test for GitHub issue #75983: ``hermes skills install … --name
        research`` deleted the whole ``skills/research/`` category bucket,
        destroying 16 unrelated skills.
        """
        import tools.skills_hub as hub
        from tools.skills_guard import ScanResult

        skills_dir = tmp_path / "skills"
        skills_dir.mkdir()

        # Simulate a user-created category bucket with multiple skills.
        category = skills_dir / "research"
        category.mkdir()
        for skill in ("alpha", "bravo", "charlie"):
            (category / skill).mkdir()
            (category / skill / "SKILL.md").write_text(f"name: {skill}")

        quarantine_root = skills_dir / ".hub" / "quarantine"
        quarantine_root.mkdir(parents=True)

        q_dir = quarantine_root / "pending"
        q_dir.mkdir()
        (q_dir / "SKILL.md").write_text("---\nname: research\n---\n")

        bundle = hub.SkillBundle(
            name="research",
            files={"SKILL.md": "---\nname: research\n---\n"},
            source="community",
            identifier="x",
            trust_level="community",
        )
        scan_result = ScanResult(
            skill_name="research",
            source="community",
            trust_level="community",
            verdict="safe",
        )

        with patch.object(hub, "SKILLS_DIR", skills_dir), \
             patch.object(hub, "QUARANTINE_DIR", quarantine_root):
            with pytest.raises(ValueError, match="Refusing to overwrite category directory"):
                hub.install_from_quarantine(
                    q_dir, "research", "", bundle, scan_result,
                )

        # Verify the category bucket and its skills are intact.
        assert (category / "alpha" / "SKILL.md").exists()
        assert (category / "bravo" / "SKILL.md").exists()
        assert (category / "charlie" / "SKILL.md").exists()

    def test_install_from_quarantine_allows_existing_skill_overwrite(self, tmp_path):
        """Installing over an existing skill directory (containing SKILL.md) is
        still allowed — that scenario is already guarded by the lock-file check
        in do_install()."""
        import tools.skills_hub as hub
        from tools.skills_guard import ScanResult

        skills_dir = tmp_path / "skills"
        skills_dir.mkdir()

        # Existing skill directory with SKILL.md.
        existing = skills_dir / "my-skill"
        existing.mkdir()
        (existing / "SKILL.md").write_text("old content")
        (existing / "refs").mkdir()
        (existing / "refs" / "guide.md").write_text("old guide")

        quarantine_root = skills_dir / ".hub" / "quarantine"
        quarantine_root.mkdir(parents=True)

        q_dir = quarantine_root / "pending"
        q_dir.mkdir()
        (q_dir / "SKILL.md").write_text("---\nname: my-skill\n---\nnew")
        (q_dir / "refs").mkdir()
        (q_dir / "refs" / "guide.md").write_text("new guide")

        bundle = hub.SkillBundle(
            name="my-skill",
            files={"SKILL.md": "---\nname: my-skill\n---\nnew"},
            source="community",
            identifier="x",
            trust_level="community",
        )
        scan_result = ScanResult(
            skill_name="my-skill",
            source="community",
            trust_level="community",
            verdict="safe",
        )

        with patch.object(hub, "SKILLS_DIR", skills_dir), \
             patch.object(hub, "QUARANTINE_DIR", quarantine_root):
            installed = hub.install_from_quarantine(
                q_dir, "my-skill", "", bundle, scan_result,
            )

        # The old directory was replaced by the new one.
        assert installed.exists()
        assert (installed / "SKILL.md").read_text().strip() == "---\nname: my-skill\n---\nnew"

    def test_install_from_quarantine_allows_empty_category_dir(self, tmp_path):
        """Installing into an existing but empty category directory is allowed
        (no sibling skills to destroy)."""
        import tools.skills_hub as hub
        from tools.skills_guard import ScanResult

        skills_dir = tmp_path / "skills"
        skills_dir.mkdir()

        # Empty category directory (no skills inside).
        category = skills_dir / "research"
        category.mkdir()

        quarantine_root = skills_dir / ".hub" / "quarantine"
        quarantine_root.mkdir(parents=True)

        q_dir = quarantine_root / "pending"
        q_dir.mkdir()
        (q_dir / "SKILL.md").write_text("---\nname: research\n---\n")

        bundle = hub.SkillBundle(
            name="research",
            files={"SKILL.md": "---\nname: research\n---\n"},
            source="community",
            identifier="x",
            trust_level="community",
        )
        scan_result = ScanResult(
            skill_name="research",
            source="community",
            trust_level="community",
            verdict="safe",
        )

        with patch.object(hub, "SKILLS_DIR", skills_dir), \
             patch.object(hub, "QUARANTINE_DIR", quarantine_root):
            installed = hub.install_from_quarantine(
                q_dir, "research", "", bundle, scan_result,
            )

        assert installed.exists()
        assert (installed / "SKILL.md").exists()

    def test_install_from_quarantine_rejects_nested_only_category(self, tmp_path):
        """A category whose skills live only in sub-categories (depth >= 2,
        e.g. ``mlops/training/<skill>``) must also be protected — the guard
        must not be fooled by the absence of direct-child skills (#75983)."""
        import tools.skills_hub as hub
        from tools.skills_guard import ScanResult

        skills_dir = tmp_path / "skills"
        quarantine_root = skills_dir / ".hub" / "quarantine"
        quarantine_root.mkdir(parents=True)

        # Nested-only category: no skill directly under mlops/, skills at depth 2.
        for sub, name in (("training", "trl"), ("inference", "vllm")):
            skill = skills_dir / "mlops" / sub / name
            skill.mkdir(parents=True)
            (skill / "SKILL.md").write_text(f"---\nname: {name}\n---\n")

        q_dir = quarantine_root / "pending"
        q_dir.mkdir()
        (q_dir / "SKILL.md").write_text("---\nname: mlops\n---\n")

        bundle = hub.SkillBundle(
            name="mlops",
            files={"SKILL.md": "---\nname: mlops\n---\n"},
            source="community",
            identifier="x",
            trust_level="community",
        )
        scan_result = ScanResult(
            skill_name="mlops",
            source="community",
            trust_level="community",
            verdict="safe",
        )

        with patch.object(hub, "SKILLS_DIR", skills_dir), \
             patch.object(hub, "QUARANTINE_DIR", quarantine_root):
            with pytest.raises(ValueError, match="category directory"):
                hub.install_from_quarantine(
                    q_dir, "mlops", "", bundle, scan_result,
                )

        # Every nested skill must survive.
        assert (skills_dir / "mlops" / "training" / "trl" / "SKILL.md").is_file()
        assert (skills_dir / "mlops" / "inference" / "vllm" / "SKILL.md").is_file()

    def test_install_from_quarantine_rejects_category_inside_skill(self, tmp_path):
        """Installing with --category naming an existing *skill* directory must
        be refused: it would create a hybrid skill-plus-category dir whose later
        update/uninstall rmtree would destroy the nested skill (#75983 sibling)."""
        import tools.skills_hub as hub
        from tools.skills_guard import ScanResult

        skills_dir = tmp_path / "skills"
        quarantine_root = skills_dir / ".hub" / "quarantine"
        quarantine_root.mkdir(parents=True)

        # Existing normal skill directory.
        outer = skills_dir / "devops"
        outer.mkdir(parents=True)
        (outer / "SKILL.md").write_text("---\nname: devops\n---\n")

        q_dir = quarantine_root / "pending"
        q_dir.mkdir()
        (q_dir / "SKILL.md").write_text("---\nname: docker\n---\n")

        bundle = hub.SkillBundle(
            name="docker",
            files={"SKILL.md": "---\nname: docker\n---\n"},
            source="community",
            identifier="x",
            trust_level="community",
        )
        scan_result = ScanResult(
            skill_name="docker",
            source="community",
            trust_level="community",
            verdict="safe",
        )

        with patch.object(hub, "SKILLS_DIR", skills_dir), \
             patch.object(hub, "QUARANTINE_DIR", quarantine_root):
            with pytest.raises(ValueError, match="existing skill directory"):
                hub.install_from_quarantine(
                    q_dir, "docker", "devops", bundle, scan_result,
                )

        # The outer skill is untouched and no hybrid child was created.
        assert (outer / "SKILL.md").is_file()
        assert not (outer / "docker").exists()

    def test_install_from_quarantine_rejects_regular_file_collision(self, tmp_path):
        """A stray regular file at the install path must produce the caller's
        ValueError contract, not an uncaught NotADirectoryError from iterdir/rmtree."""
        import tools.skills_hub as hub
        from tools.skills_guard import ScanResult

        skills_dir = tmp_path / "skills"
        quarantine_root = skills_dir / ".hub" / "quarantine"
        quarantine_root.mkdir(parents=True)

        (skills_dir / "notes").write_text("not a directory")

        q_dir = quarantine_root / "pending"
        q_dir.mkdir()
        (q_dir / "SKILL.md").write_text("---\nname: notes\n---\n")

        bundle = hub.SkillBundle(
            name="notes",
            files={"SKILL.md": "---\nname: notes\n---\n"},
            source="community",
            identifier="x",
            trust_level="community",
        )
        scan_result = ScanResult(
            skill_name="notes",
            source="community",
            trust_level="community",
            verdict="safe",
        )

        with patch.object(hub, "SKILLS_DIR", skills_dir), \
             patch.object(hub, "QUARANTINE_DIR", quarantine_root):
            with pytest.raises(ValueError, match="not a directory"):
                hub.install_from_quarantine(
                    q_dir, "notes", "", bundle, scan_result,
                )

        assert (skills_dir / "notes").read_text() == "not a directory"

    def test_install_from_quarantine_records_successful_install(self, tmp_path):
        import tools.skills_hub as hub
        from tools.skills_guard import ScanResult

        skills_dir = tmp_path / "skills"
        quarantine_root = skills_dir / ".hub" / "quarantine"
        q_dir = quarantine_root / "pending"
        q_dir.mkdir(parents=True)
        skill_md = "---\nname: good-skill\n---\n\n# Good skill\n"
        (q_dir / "SKILL.md").write_text(skill_md, encoding="utf-8")
        bundle = hub.SkillBundle(
            name="good-skill",
            files={"SKILL.md": skill_md},
            source="community",
            identifier="good/source",
            trust_level="community",
        )
        scan_result = ScanResult(
            skill_name="good-skill",
            source="community",
            trust_level="community",
            verdict="safe",
        )

        with patch.object(hub, "SKILLS_DIR", skills_dir), \
             patch.object(hub, "QUARANTINE_DIR", quarantine_root), \
             patch("tools.skill_usage.record_installed") as record_installed:
            installed = hub.install_from_quarantine(
                q_dir,
                "good-skill",
                "",
                bundle,
                scan_result,
            )

        assert installed == skills_dir / "good-skill"
        assert installed.is_dir()
        record_installed.assert_called_once_with("good-skill")


# ---------------------------------------------------------------------------
# parallel_search_sources — overall_timeout must be honoured even when a
# source blocks for far longer than the budget (regression: the executor used
# `with ... as pool`, whose __exit__ calls shutdown(wait=True) and blocked the
# caller on the slow worker, making overall_timeout a no-op).
# ---------------------------------------------------------------------------


class _FakeSource(SkillSource):
    def __init__(self, sid: str, sleep: float = 0.0, results=None):
        self._sid = sid
        self._sleep = sleep
        self._results = results or []

    def source_id(self) -> str:
        return self._sid

    def search(self, query: str, limit: int = 10) -> List[SkillMeta]:
        if self._sleep:
            time.sleep(self._sleep)
        return list(self._results)

    def fetch(self, identifier: str) -> Optional[SkillBundle]:
        return None

    def inspect(self, identifier: str) -> Optional[SkillMeta]:
        return None


class TestParallelSearchSourcesTimeout:
    def _meta(self, sid: str) -> SkillMeta:
        return SkillMeta(
            name=f"{sid}-skill",
            description="x",
            source=sid,
            identifier=f"{sid}/x",
            trust_level="community",
        )

    def test_slow_source_does_not_block_caller(self):
        """A source sleeping well past overall_timeout must not stall the
        return. Before the fix the executor's `with` block waited on the slow
        worker (~5s); now the call returns promptly and reports the source as
        timed out."""
        fast = _FakeSource("fast", sleep=0.0, results=[self._meta("fast")])
        slow = _FakeSource("slow", sleep=0.3, results=[self._meta("slow")])

        start = time.monotonic()
        all_results, source_counts, timed_out_ids = parallel_search_sources(
            [fast, slow], query="q", overall_timeout=0.05,
        )
        elapsed = time.monotonic() - start

        # Must return long before the slow source's sleep finishes.
        assert elapsed < 1.0, f"call blocked for {elapsed:.2f}s (timeout not honoured)"
        assert "slow" in timed_out_ids
        # Fast source still delivered its result and is not flagged timed out.
        assert source_counts.get("fast") == 1
        assert "fast" not in timed_out_ids
        assert any(r.source == "fast" for r in all_results)

    def test_all_fast_sources_complete_without_timeout(self):
        """Happy path: when every source finishes within budget, none are
        flagged and all results are collected."""
        a = _FakeSource("a", results=[self._meta("a")])
        b = _FakeSource("b", results=[self._meta("b")])

        all_results, source_counts, timed_out_ids = parallel_search_sources(
            [a, b], query="q", overall_timeout=5.0,
        )

        assert timed_out_ids == []
        assert source_counts.get("a") == 1
        assert source_counts.get("b") == 1
        assert len(all_results) == 2


# ---------------------------------------------------------------------------
# _load_hermes_index — centralized index fetch (Browse-hub landing / search)
# ---------------------------------------------------------------------------


class TestLoadHermesIndex:
    """Regression coverage for the Skills-Hub index fetch.

    The centralized index is a large body served with Content-Encoding: br.
    httpx's streaming Brotli decoder (brotlicffi 1.2.0.1, pinned for Discord
    attachment decoding) raises DecodingError on payloads this size, which
    used to cascade into a silently-empty Skills Hub. The fetch must therefore
    (a) not ask for Brotli, and (b) survive a DecodingError by retrying
    uncompressed instead of blanking the hub.
    """

    @staticmethod
    def _isolate_cache(monkeypatch, tmp_path):
        """Point the on-disk cache at an empty tmp dir so no real cache leaks in."""
        import tools.skills_hub as hub

        cache_file = tmp_path / "hermes-index.json"
        monkeypatch.setattr(hub, "_hermes_index_cache_file", lambda: cache_file)
        return cache_file

    def test_fetch_does_not_request_brotli(self, monkeypatch, tmp_path):
        """The index fetch must not negotiate Brotli (the broken decoder path)."""
        import tools.skills_hub as hub

        self._isolate_cache(monkeypatch, tmp_path)

        captured = {}

        def fake_get(url, *args, **kwargs):
            captured["headers"] = kwargs.get("headers", {})
            resp = MagicMock()
            resp.status_code = 200
            resp.json.return_value = {"skills": [{"name": "x"}]}
            return resp

        monkeypatch.setattr(hub.httpx, "get", fake_get)

        data = hub._load_hermes_index()
        assert data == {"skills": [{"name": "x"}]}

        accept = captured["headers"].get("Accept-Encoding", "")
        assert "br" not in [tok.strip() for tok in accept.split(",")], (
            f"index fetch must not request Brotli, got Accept-Encoding={accept!r}"
        )

    def test_persistent_decoding_error_falls_back_to_stale_cache(
        self, monkeypatch, tmp_path
    ):
        """If every attempt fails to decode, serve the stale cache rather than None."""
        import tools.skills_hub as hub

        cache_file = self._isolate_cache(monkeypatch, tmp_path)
        cache_file.write_text(json.dumps({"skills": [{"name": "stale"}]}))
        # Force the cache to look expired so the network path runs.
        old = time.time() - (hub.HERMES_INDEX_TTL + 100)
        import os

        os.utime(cache_file, (old, old))

        def fake_get(url, *args, **kwargs):
            raise httpx.DecodingError("brotli boom")

        monkeypatch.setattr(hub.httpx, "get", fake_get)

        data = hub._load_hermes_index()
        assert data == {"skills": [{"name": "stale"}]}


# ---------------------------------------------------------------------------
# Referenced-path extraction & missing support files (regression: a prose
# glob or a repo-only dev tool referenced in SKILL.md must not abort install)
# ---------------------------------------------------------------------------


class TestReferencedSupportPaths:
    def test_ignores_globs_placeholders_and_truncated_tokens(self):
        md = (
            "Load `references/type-*.md` before drawing, and "
            "`references/type-<name>.md` for the matching type.\n"
            "Real files: [flowchart](references/type-flowchart.md) and "
            "`references/type-architecture.md`.\n"
        )
        assert _referenced_support_paths(md) == {
            "references/type-flowchart.md",
            "references/type-architecture.md",
        }

    def test_keeps_real_backtick_references(self):
        md = "Run `python3 scripts/self_check.py <file>` and see `references/guide.md`.\n"
        assert _referenced_support_paths(md) == {
            "scripts/self_check.py",
            "references/guide.md",
        }


class TestGitHubSourceFetchMissingReferencedFile:
    def _source(self):
        auth = MagicMock(spec=GitHubAuth)
        return GitHubSource(auth=auth)

    def test_fetch_skips_missing_referenced_file_and_keeps_the_rest(self):
        md = (
            "---\nname: demo\ndescription: demo\n---\n\n"
            "See [guide](references/guide.md) and `references/missing.md`.\n"
        )
        tree_entries = [
            {"path": "skills/demo/references/guide.md", "type": "blob", "mode": "100644"},
        ]
        source = self._source()
        with patch.object(source, "_fetch_file_content", return_value=md), \
             patch.object(source, "_get_repo_tree", return_value=("main", tree_entries)), \
             patch.object(source, "_fetch_file_bytes", return_value=b"# guide"):
            bundle = source.fetch("owner/repo/skills/demo")

        assert bundle is not None
        assert bundle.name == "demo"
        # The present referenced file is bundled…
        assert bundle.files["references/guide.md"] == b"# guide"
        # …and the missing one is warned about and skipped, not fatal.
        assert "references/missing.md" not in bundle.files


class TestUrlSourceFetchMissingReferencedFile:
    def test_fetch_skips_missing_referenced_file(self):
        md = "---\nname: demo\ndescription: demo\n---\n\nSee `references/missing.md`.\n"
        source = UrlSource()
        with patch.object(source, "_fetch_text", return_value=md), \
             patch.object(source, "_fetch_bytes", return_value=None):
            bundle = source.fetch("https://example.com/skills/demo/SKILL.md")

        assert bundle is not None
        assert bundle.name == "demo"
        assert "references/missing.md" not in bundle.files
