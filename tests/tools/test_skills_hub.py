"""Tests for tools/skills_hub.py — source adapters, lock file, taps, dedup logic."""

import json
from pathlib import Path
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
    SkillMeta,
    SkillBundle,
    HubLockFile,
    TapsManager,
    bundle_content_hash,
    check_for_skill_updates,
    create_source_router,
    unified_search,
    append_audit_log,
    _skill_meta_to_dict,
    quarantine_bundle,
)


# ---------------------------------------------------------------------------
# GitHubSource._parse_frontmatter_quick
# ---------------------------------------------------------------------------


class TestParseFrontmatterQuick:
    def test_valid_frontmatter(self):
        content = "---\nname: test-skill\ndescription: A test.\n---\n\n# Body\n"
        fm = GitHubSource._parse_frontmatter_quick(content)
        assert fm["name"] == "test-skill"
        assert fm["description"] == "A test."

    def test_no_frontmatter(self):
        content = "# Just a heading\nSome body text.\n"
        fm = GitHubSource._parse_frontmatter_quick(content)
        assert fm == {}

    def test_no_closing_delimiter(self):
        content = "---\nname: test\ndescription: desc\nno closing here\n"
        fm = GitHubSource._parse_frontmatter_quick(content)
        assert fm == {}

    def test_empty_content(self):
        fm = GitHubSource._parse_frontmatter_quick("")
        assert fm == {}

    def test_nested_yaml(self):
        content = "---\nname: test\nmetadata:\n  hermes:\n    tags: [a, b]\n---\n\nBody.\n"
        fm = GitHubSource._parse_frontmatter_quick(content)
        assert fm["metadata"]["hermes"]["tags"] == ["a", "b"]

    def test_invalid_yaml_returns_empty(self):
        content = "---\n: : : invalid{{\n---\n\nBody.\n"
        fm = GitHubSource._parse_frontmatter_quick(content)
        assert fm == {}

    def test_non_dict_yaml_returns_empty(self):
        content = "---\n- just a list\n- of items\n---\n\nBody.\n"
        fm = GitHubSource._parse_frontmatter_quick(content)
        assert fm == {}


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

    def test_community_repo(self):
        src = self._source()
        assert src.trust_level_for("random-user/random-repo/skill") == "community"

    def test_short_identifier(self):
        src = self._source()
        assert src.trust_level_for("no-slash") == "community"

    def test_two_part_identifier(self):
        src = self._source()
        result = src.trust_level_for("owner/repo")
        # No path part — still resolves repo correctly
        assert result in ("trusted", "community")


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
    def test_empty_search_uses_featured_homepage_links(self, mock_get, _mock_read_cache, _mock_write_cache):
        mock_get.return_value = MagicMock(
            status_code=200,
            text='''
                <a href="/vercel-labs/agent-skills/vercel-react-best-practices">React</a>
                <a href="/anthropics/skills/pdf">PDF</a>
                <a href="/vercel-labs/agent-skills/vercel-react-best-practices">React again</a>
            ''',
        )

        results = self._source().search("", limit=10)

        assert [r.identifier for r in results] == [
            "skills-sh/vercel-labs/agent-skills/vercel-react-best-practices",
            "skills-sh/anthropics/skills/pdf",
        ]
        assert all(r.source == "skills.sh" for r in results)

    @patch.object(GitHubSource, "fetch")
    def test_fetch_delegates_to_github_source_and_relabels_bundle(self, mock_fetch):
        mock_fetch.return_value = SkillBundle(
            name="vercel-react-best-practices",
            files={"SKILL.md": "# Test"},
            source="github",
            identifier="vercel-labs/agent-skills/vercel-react-best-practices",
            trust_level="community",
        )

        bundle = self._source().fetch("skills-sh/vercel-labs/agent-skills/vercel-react-best-practices")

        assert bundle is not None
        assert bundle.source == "skills.sh"
        assert bundle.identifier == "skills-sh/vercel-labs/agent-skills/vercel-react-best-practices"
        mock_fetch.assert_called_once_with("vercel-labs/agent-skills/vercel-react-best-practices")

    @patch.object(GitHubSource, "fetch")
    def test_fetch_accepts_common_skills_sh_prefix_typo(self, mock_fetch):
        expected_identifier = "anthropics/skills/frontend-design"
        mock_fetch.side_effect = lambda identifier: SkillBundle(
            name="frontend-design",
            files={"SKILL.md": "# Frontend Design"},
            source="github",
            identifier=expected_identifier,
            trust_level="trusted",
        ) if identifier == expected_identifier else None

        bundle = self._source().fetch("skils-sh/anthropics/skills/frontend-design")

        assert bundle is not None
        assert bundle.source == "skills.sh"
        assert bundle.identifier == "skills-sh/anthropics/skills/frontend-design"
        assert mock_fetch.call_args_list[0] == ((expected_identifier,), {})

    @patch("tools.skills_hub._write_index_cache")
    @patch("tools.skills_hub._read_index_cache", return_value=None)
    @patch("tools.skills_hub.httpx.get")
    @patch.object(GitHubSource, "inspect")
    def test_inspect_delegates_to_github_source_and_relabels_meta(self, mock_inspect, mock_get, _mock_read_cache, _mock_write_cache):
        mock_inspect.return_value = SkillMeta(
            name="vercel-react-best-practices",
            description="React rules",
            source="github",
            identifier="vercel-labs/agent-skills/vercel-react-best-practices",
            trust_level="community",
            repo="vercel-labs/agent-skills",
            path="vercel-react-best-practices",
        )
        mock_get.return_value = MagicMock(
            status_code=200,
            text='''
                <h1>vercel-react-best-practices</h1>
                <code>$ npx skills add https://github.com/vercel-labs/agent-skills --skill vercel-react-best-practices</code>
                <div class="prose"><h1>Vercel React Best Practices</h1><p>React rules.</p></div>
                <a href="/vercel-labs/agent-skills/vercel-react-best-practices/security/socket">Socket</a> Pass
                <a href="/vercel-labs/agent-skills/vercel-react-best-practices/security/snyk">Snyk</a> Pass
            ''',
        )

        meta = self._source().inspect("skills-sh/vercel-labs/agent-skills/vercel-react-best-practices")

        assert meta is not None
        assert meta.source == "skills.sh"
        assert meta.identifier == "skills-sh/vercel-labs/agent-skills/vercel-react-best-practices"
        assert meta.extra["install_command"].endswith("--skill vercel-react-best-practices")
        assert meta.extra["security_audits"]["socket"] == "Pass"
        mock_inspect.assert_called_once_with("vercel-labs/agent-skills/vercel-react-best-practices")

    @patch.object(GitHubSource, "inspect")
    def test_inspect_accepts_common_skills_sh_prefix_typo(self, mock_inspect):
        expected_identifier = "anthropics/skills/frontend-design"
        mock_inspect.side_effect = lambda identifier: SkillMeta(
            name="frontend-design",
            description="Distinctive frontend interfaces.",
            source="github",
            identifier=expected_identifier,
            trust_level="trusted",
            repo="anthropics/skills",
            path="frontend-design",
        ) if identifier == expected_identifier else None

        meta = self._source().inspect("skils-sh/anthropics/skills/frontend-design")

        assert meta is not None
        assert meta.source == "skills.sh"
        assert meta.identifier == "skills-sh/anthropics/skills/frontend-design"
        assert mock_inspect.call_args_list[0] == ((expected_identifier,), {})

    @patch.object(GitHubSource, "_list_skills_in_repo")
    @patch.object(GitHubSource, "inspect")
    def test_inspect_falls_back_to_repo_skill_catalog_when_slug_differs(self, mock_inspect, mock_list_skills):
        resolved = SkillMeta(
            name="vercel-react-best-practices",
            description="React rules",
            source="github",
            identifier="vercel-labs/agent-skills/skills/react-best-practices",
            trust_level="community",
            repo="vercel-labs/agent-skills",
            path="skills/react-best-practices",
        )
        mock_inspect.side_effect = lambda identifier: resolved if identifier == resolved.identifier else None
        mock_list_skills.return_value = [resolved]

        meta = self._source().inspect("skills-sh/vercel-labs/agent-skills/vercel-react-best-practices")

        assert meta is not None
        assert meta.identifier == "skills-sh/vercel-labs/agent-skills/vercel-react-best-practices"
        assert mock_list_skills.called

    @patch("tools.skills_hub._write_index_cache")
    @patch("tools.skills_hub._read_index_cache", return_value=None)
    @patch("tools.skills_hub.httpx.get")
    @patch.object(GitHubSource, "_list_skills_in_repo")
    @patch.object(GitHubSource, "inspect")
    def test_inspect_uses_detail_page_to_resolve_alias_skill(self, mock_inspect, mock_list_skills, mock_get, _mock_read_cache, _mock_write_cache):
        resolved = SkillMeta(
            name="react",
            description="React renderer",
            source="github",
            identifier="vercel-labs/json-render/skills/react",
            trust_level="community",
            repo="vercel-labs/json-render",
            path="skills/react",
        )
        mock_inspect.side_effect = lambda identifier: resolved if identifier == resolved.identifier else None
        mock_list_skills.return_value = [resolved]
        mock_get.return_value = MagicMock(
            status_code=200,
            text='''
                <h1>json-render-react</h1>
                <code>$ npx skills add https://github.com/vercel-labs/json-render --skill json-render-react</code>
                <div class="prose"><h1>@json-render/react</h1><p>React renderer.</p></div>
            ''',
        )

        meta = self._source().inspect("skills-sh/vercel-labs/json-render/json-render-react")

        assert meta is not None
        assert meta.identifier == "skills-sh/vercel-labs/json-render/json-render-react"
        assert meta.path == "skills/react"
        assert mock_get.called

    @patch("tools.skills_hub._write_index_cache")
    @patch("tools.skills_hub._read_index_cache", return_value=None)
    @patch("tools.skills_hub.httpx.get")
    @patch.object(GitHubSource, "_list_skills_in_repo")
    @patch.object(GitHubSource, "fetch")
    def test_fetch_uses_detail_page_to_resolve_alias_skill(self, mock_fetch, mock_list_skills, mock_get, _mock_read_cache, _mock_write_cache):
        resolved_meta = SkillMeta(
            name="react",
            description="React renderer",
            source="github",
            identifier="vercel-labs/json-render/skills/react",
            trust_level="community",
            repo="vercel-labs/json-render",
            path="skills/react",
        )
        resolved_bundle = SkillBundle(
            name="react",
            files={"SKILL.md": "# react"},
            source="github",
            identifier="vercel-labs/json-render/skills/react",
            trust_level="community",
        )
        mock_fetch.side_effect = lambda identifier: resolved_bundle if identifier == resolved_bundle.identifier else None
        mock_list_skills.return_value = [resolved_meta]
        mock_get.return_value = MagicMock(
            status_code=200,
            text='''
                <h1>json-render-react</h1>
                <code>$ npx skills add https://github.com/vercel-labs/json-render --skill json-render-react</code>
                <div class="prose"><h1>@json-render/react</h1><p>React renderer.</p></div>
            ''',
        )

        bundle = self._source().fetch("skills-sh/vercel-labs/json-render/json-render-react")

        assert bundle is not None
        assert bundle.identifier == "skills-sh/vercel-labs/json-render/json-render-react"
        assert bundle.files["SKILL.md"] == "# react"
        assert mock_get.called

    @patch("tools.skills_hub._write_index_cache")
    @patch("tools.skills_hub._read_index_cache", return_value=None)
    @patch.object(SkillsShSource, "_discover_identifier")
    @patch.object(SkillsShSource, "_fetch_detail_page")
    @patch.object(GitHubSource, "fetch")
    def test_fetch_downloads_only_the_resolved_identifier(
        self,
        mock_fetch,
        mock_detail,
        mock_discover,
        _mock_read_cache,
        _mock_write_cache,
    ):
        resolved_identifier = "owner/repo/product-team/product-designer"
        mock_detail.return_value = {"repo": "owner/repo", "install_skill": "product-designer"}
        mock_discover.return_value = resolved_identifier
        resolved_bundle = SkillBundle(
            name="product-designer",
            files={"SKILL.md": "# Product Designer"},
            source="github",
            identifier=resolved_identifier,
            trust_level="community",
        )
        mock_fetch.side_effect = lambda identifier: resolved_bundle if identifier == resolved_identifier else None

        bundle = self._source().fetch("skills-sh/owner/repo/product-designer")

        assert bundle is not None
        assert bundle.identifier == "skills-sh/owner/repo/product-designer"
        # All candidate identifiers are tried before falling back to discovery
        assert mock_fetch.call_args_list[-1] == ((resolved_identifier,), {})
        assert mock_fetch.call_args_list[0] == (("owner/repo/product-designer",), {})

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

    @patch.object(GitHubSource, "_find_skill_in_repo_tree")
    @patch.object(GitHubSource, "_list_skills_in_repo")
    @patch("tools.skills_hub.httpx.get")
    def test_discover_identifier_uses_tree_search_before_root_scan(
        self,
        mock_get,
        mock_list_skills,
        mock_find_in_tree,
    ):
        root_url = "https://api.github.com/repos/owner/repo/contents/"
        mock_list_skills.return_value = []
        mock_find_in_tree.return_value = "owner/repo/product-team/product-designer"

        def _httpx_get_side_effect(url, **kwargs):
            resp = MagicMock()
            if url == root_url:
                resp.status_code = 200
                resp.json = lambda: []
                return resp
            resp.status_code = 404
            return resp

        mock_get.side_effect = _httpx_get_side_effect

        result = self._source()._discover_identifier("owner/repo/product-designer")

        assert result == "owner/repo/product-team/product-designer"
        requested_urls = [call.args[0] for call in mock_get.call_args_list]
        assert root_url not in requested_urls


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
    def test_finds_root_level_skill(self, mock_get):
        tree_entries = [
            {"path": "my-skill/SKILL.md", "type": "blob"},
        ]

        def _side_effect(url, **kwargs):
            resp = MagicMock()
            if "/contents" not in url and "/git/" not in url:
                resp.status_code = 200
                resp.json = lambda: {"default_branch": "main"}
            elif "/git/trees/main" in url:
                resp.status_code = 200
                resp.json = lambda: {"tree": tree_entries}
            else:
                resp.status_code = 404
            return resp

        mock_get.side_effect = _side_effect

        result = self._source()._find_skill_in_repo_tree("owner/repo", "my-skill")
        assert result == "owner/repo/my-skill"

    @patch("tools.skills_hub.httpx.get")
    def test_returns_none_when_skill_not_found(self, mock_get):
        tree_entries = [
            {"path": "other-skill/SKILL.md", "type": "blob"},
        ]

        def _side_effect(url, **kwargs):
            resp = MagicMock()
            if "/contents" not in url and "/git/" not in url:
                resp.status_code = 200
                resp.json = lambda: {"default_branch": "main"}
            elif "/git/trees/main" in url:
                resp.status_code = 200
                resp.json = lambda: {"tree": tree_entries}
            else:
                resp.status_code = 404
            return resp

        mock_get.side_effect = _side_effect

        result = self._source()._find_skill_in_repo_tree("owner/repo", "nonexistent")
        assert result is None

    @patch("tools.skills_hub.httpx.get")
    def test_returns_none_when_repo_api_fails(self, mock_get):
        mock_get.return_value = MagicMock(status_code=404)
        result = self._source()._find_skill_in_repo_tree("owner/repo", "my-skill")
        assert result is None


class TestWellKnownSkillSource:
    def _source(self):
        return WellKnownSkillSource()

    @patch("tools.skills_hub._write_index_cache")
    @patch("tools.skills_hub._read_index_cache", return_value=None)
    @patch("tools.skills_hub.httpx.get")
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
    def test_search_accepts_domain_root_and_resolves_index(self, mock_get, _mock_read_cache, _mock_write_cache):
        mock_get.return_value = MagicMock(
            status_code=200,
            json=lambda: {"skills": [{"name": "git-workflow", "description": "Git rules", "files": ["SKILL.md"]}]},
        )

        results = self._source().search("https://example.com", limit=10)

        assert len(results) == 1
        called_url = mock_get.call_args.args[0]
        assert called_url == "https://example.com/.well-known/skills/index.json"

    @patch("tools.skills_hub._write_index_cache")
    @patch("tools.skills_hub._read_index_cache", return_value=None)
    @patch("tools.skills_hub.httpx.get")
    def test_inspect_fetches_skill_md_from_well_known_endpoint(self, mock_get, _mock_read_cache, _mock_write_cache):
        def fake_get(url, *args, **kwargs):
            if url.endswith("/index.json"):
                return MagicMock(status_code=200, json=lambda: {
                    "skills": [{"name": "git-workflow", "description": "Git rules", "files": ["SKILL.md"]}]
                })
            if url.endswith("/git-workflow/SKILL.md"):
                return MagicMock(status_code=200, text="---\nname: git-workflow\ndescription: Git rules\n---\n\n# Git Workflow\n")
            raise AssertionError(url)

        mock_get.side_effect = fake_get

        meta = self._source().inspect("well-known:https://example.com/.well-known/skills/git-workflow")

        assert meta is not None
        assert meta.name == "git-workflow"
        assert meta.source == "well-known"
        assert meta.extra["base_url"] == "https://example.com/.well-known/skills"

    @patch("tools.skills_hub._write_index_cache")
    @patch("tools.skills_hub._read_index_cache", return_value=None)
    @patch("tools.skills_hub.httpx.get")
    def test_fetch_downloads_skill_files_from_well_known_endpoint(self, mock_get, _mock_read_cache, _mock_write_cache):
        def fake_get(url, *args, **kwargs):
            if url.endswith("/index.json"):
                return MagicMock(status_code=200, json=lambda: {
                    "skills": [{
                        "name": "code-review",
                        "description": "Review code",
                        "files": ["SKILL.md", "references/checklist.md"],
                    }]
                })
            if url.endswith("/code-review/SKILL.md"):
                return MagicMock(status_code=200, text="# Code Review\n")
            if url.endswith("/code-review/references/checklist.md"):
                return MagicMock(status_code=200, text="- [ ] security\n")
            raise AssertionError(url)

        mock_get.side_effect = fake_get

        bundle = self._source().fetch("well-known:https://example.com/.well-known/skills/code-review")

        assert bundle is not None
        assert bundle.source == "well-known"
        assert bundle.files["SKILL.md"] == "# Code Review\n"
        assert bundle.files["references/checklist.md"] == "- [ ] security\n"

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
    def _source(self):
        return UrlSource()

    # ── _matches ────────────────────────────────────────────────────────
    def test_matches_bare_md_url(self):
        assert self._source()._matches("https://example.com/path/SKILL.md") is True

    def test_matches_http_scheme(self):
        assert self._source()._matches("http://example.com/SKILL.md") is True

    def test_rejects_non_md_url(self):
        assert self._source()._matches("https://example.com/path/") is False
        assert self._source()._matches("https://example.com/skills.json") is False

    def test_rejects_well_known_url(self):
        # Leave these for WellKnownSkillSource.
        assert self._source()._matches(
            "https://example.com/.well-known/skills/git-workflow/SKILL.md"
        ) is False
        assert self._source()._matches(
            "https://example.com/.well-known/skills/index.json"
        ) is False

    def test_rejects_wrapped_identifiers(self):
        assert self._source()._matches("github:owner/repo/skill") is False
        assert self._source()._matches("well-known:https://example.com/x") is False
        assert self._source()._matches("official/security/1password") is False

    def test_rejects_non_string(self):
        assert self._source()._matches(None) is False  # type: ignore[arg-type]
        assert self._source()._matches(123) is False   # type: ignore[arg-type]

    def test_search_returns_empty(self):
        # Direct-URL source is not searchable.
        assert self._source().search("anything") == []

    # ── inspect ─────────────────────────────────────────────────────────
    @patch("tools.skills_hub.httpx.get")
    def test_inspect_reads_frontmatter_from_url(self, mock_get):
        mock_get.return_value = MagicMock(
            status_code=200,
            text=(
                "---\n"
                "name: sharethis-chat\n"
                "description: Share agent conversations.\n"
                "metadata:\n"
                "  hermes:\n"
                "    tags: [sharing, chat]\n"
                "---\n\n# Body\n"
            ),
        )
        meta = self._source().inspect("https://sharethis.chat/SKILL.md")
        assert meta is not None
        assert meta.name == "sharethis-chat"
        assert meta.description == "Share agent conversations."
        assert meta.source == "url"
        assert meta.identifier == "https://sharethis.chat/SKILL.md"
        assert meta.trust_level == "community"
        assert meta.tags == ["sharing", "chat"]
        assert meta.extra["awaiting_name"] is False

    @patch("tools.skills_hub.httpx.get")
    def test_inspect_returns_none_when_url_not_md(self, mock_get):
        # _matches filters first — no HTTP call.
        meta = self._source().inspect("https://example.com/not-a-skill")
        assert meta is None
        mock_get.assert_not_called()

    @patch("tools.skills_hub.httpx.get")
    def test_inspect_returns_none_on_404(self, mock_get):
        mock_get.return_value = MagicMock(status_code=404)
        assert self._source().inspect("https://example.com/SKILL.md") is None

    @patch("tools.skills_hub.httpx.get")
    def test_inspect_returns_none_on_http_error(self, mock_get):
        mock_get.side_effect = httpx.HTTPError("boom")
        assert self._source().inspect("https://example.com/SKILL.md") is None

    @patch("tools.skills_hub.httpx.get")
    def test_inspect_flags_awaiting_name_when_unresolvable(self, mock_get):
        # No frontmatter name + a URL path that can't produce a valid slug
        # (``SKILL`` isn't a valid skill name).
        mock_get.return_value = MagicMock(
            status_code=200,
            text="---\ndescription: unnamed.\n---\n",
        )
        meta = self._source().inspect("https://example.com/SKILL.md")
        assert meta is not None
        assert meta.name == ""
        assert meta.extra["awaiting_name"] is True

    # ── fetch ───────────────────────────────────────────────────────────
    @patch("tools.skills_hub.httpx.get")
    def test_fetch_builds_single_file_bundle(self, mock_get):
        skill_md = (
            "---\n"
            "name: sharethis-chat\n"
            "description: Share.\n"
            "---\n\n# Body\n"
        )
        mock_get.return_value = MagicMock(status_code=200, text=skill_md)

        bundle = self._source().fetch("https://sharethis.chat/SKILL.md")

        assert bundle is not None
        assert bundle.name == "sharethis-chat"
        assert bundle.source == "url"
        assert bundle.identifier == "https://sharethis.chat/SKILL.md"
        assert bundle.trust_level == "community"
        assert bundle.files == {"SKILL.md": skill_md}
        assert bundle.metadata["url"] == "https://sharethis.chat/SKILL.md"
        assert bundle.metadata["awaiting_name"] is False

    @patch("tools.skills_hub.httpx.get")
    def test_fetch_falls_back_to_url_directory_name(self, mock_get):
        # Frontmatter has no ``name:`` — we slug from the URL directory.
        mock_get.return_value = MagicMock(
            status_code=200,
            text="---\ndescription: No name.\n---\n\n# Body\n",
        )
        bundle = self._source().fetch("https://example.com/my-skill/SKILL.md")
        assert bundle is not None
        assert bundle.name == "my-skill"
        assert bundle.metadata["awaiting_name"] is False

    @patch("tools.skills_hub.httpx.get")
    def test_fetch_falls_back_to_filename_when_no_parent_dir(self, mock_get):
        mock_get.return_value = MagicMock(
            status_code=200,
            text="---\ndescription: Bare file.\n---\n",
        )
        bundle = self._source().fetch("https://example.com/my-skill.md")
        assert bundle is not None
        assert bundle.name == "my-skill"
        assert bundle.metadata["awaiting_name"] is False

    @patch("tools.skills_hub.httpx.get")
    def test_fetch_awaiting_name_when_unresolvable(self, mock_get):
        # Bare ``SKILL.md`` at the domain root with no frontmatter name.
        mock_get.return_value = MagicMock(
            status_code=200,
            text="---\ndescription: Bare.\n---\n\n# Body\n",
        )
        bundle = self._source().fetch("https://example.com/SKILL.md")
        assert bundle is not None
        assert bundle.name == ""
        assert bundle.metadata["awaiting_name"] is True
        # File content still present — CLI will reuse it after picking a name.
        assert bundle.files["SKILL.md"].startswith("---\n")

    @patch("tools.skills_hub.httpx.get")
    def test_fetch_awaiting_name_rejects_sentinel_slug(self, mock_get):
        # Frontmatter has no name AND the URL filename slug is ``README`` —
        # our valid-name check rejects it, so we flag awaiting_name.
        mock_get.return_value = MagicMock(
            status_code=200,
            text="---\ndescription: no name.\n---\n",
        )
        bundle = self._source().fetch("https://example.com/README.md")
        assert bundle is not None
        assert bundle.name == ""
        assert bundle.metadata["awaiting_name"] is True

    @patch("tools.skills_hub.httpx.get")
    def test_fetch_ignores_unsafe_frontmatter_name_and_falls_through_to_slug(self, mock_get):
        # Traversal / unsafe names are rejected by ``_is_valid_skill_name``;
        # resolver falls through to URL slug (``my-skill`` here) and succeeds.
        mock_get.return_value = MagicMock(
            status_code=200,
            text="---\nname: ../evil\ndescription: Bad.\n---\n",
        )
        bundle = self._source().fetch("https://example.com/my-skill/SKILL.md")
        assert bundle is not None
        assert bundle.name == "my-skill"

    @patch("tools.skills_hub.httpx.get")
    def test_fetch_returns_none_on_404(self, mock_get):
        mock_get.return_value = MagicMock(status_code=404)
        assert self._source().fetch("https://example.com/SKILL.md") is None

    @patch("tools.skills_hub.httpx.get")
    def test_fetch_skips_non_matching_identifier(self, mock_get):
        assert self._source().fetch("owner/repo/skill") is None
        mock_get.assert_not_called()

    # ── _is_valid_skill_name ────────────────────────────────────────────
    def test_is_valid_skill_name_accepts_identifiers(self):
        valid = ["my-skill", "my_skill", "sharethis-chat", "a", "skill-1", "s1"]
        for name in valid:
            assert UrlSource._is_valid_skill_name(name), f"should accept {name!r}"

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

    def test_reports_up_to_date_when_hash_matches(self):
        bundle = SkillBundle(
            name="demo-skill",
            files={"SKILL.md": "same content"},
            source="github",
            identifier="owner/repo/demo-skill",
            trust_level="community",
        )
        lock = MagicMock()
        lock.list_installed.return_value = [{
            "name": "demo-skill",
            "source": "github",
            "identifier": "owner/repo/demo-skill",
            "content_hash": bundle_content_hash(bundle),
            "install_path": "demo-skill",
        }]
        source = MagicMock()
        source.source_id.return_value = "github"
        source.fetch.return_value = bundle

        results = check_for_skill_updates(lock=lock, sources=[source])

        assert results[0]["status"] == "up_to_date"


class TestCreateSourceRouter:
    def test_includes_skills_sh_source(self):
        sources = create_source_router(auth=MagicMock(spec=GitHubAuth))
        assert any(isinstance(src, SkillsShSource) for src in sources)

    def test_includes_well_known_source(self):
        sources = create_source_router(auth=MagicMock(spec=GitHubAuth))
        assert any(isinstance(src, WellKnownSkillSource) for src in sources)

    def test_includes_url_source(self):
        sources = create_source_router(auth=MagicMock(spec=GitHubAuth))
        assert any(isinstance(src, UrlSource) for src in sources)

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
    def test_load_missing_file(self, tmp_path):
        lock = HubLockFile(path=tmp_path / "lock.json")
        data = lock.load()
        assert data == {"version": 1, "installed": {}}

    def test_load_valid_file(self, tmp_path):
        lock_file = tmp_path / "lock.json"
        lock_file.write_text(json.dumps({
            "version": 1,
            "installed": {"my-skill": {"source": "github"}}
        }))
        lock = HubLockFile(path=lock_file)
        data = lock.load()
        assert "my-skill" in data["installed"]

    def test_load_corrupt_json(self, tmp_path):
        lock_file = tmp_path / "lock.json"
        lock_file.write_text("not json{{{")
        lock = HubLockFile(path=lock_file)
        data = lock.load()
        assert data == {"version": 1, "installed": {}}

    def test_save_creates_parent_dir(self, tmp_path):
        lock_file = tmp_path / "subdir" / "lock.json"
        lock = HubLockFile(path=lock_file)
        lock.save({"version": 1, "installed": {}})
        assert lock_file.exists()

    def test_record_install(self, tmp_path):
        lock = HubLockFile(path=tmp_path / "lock.json")
        lock.record_install(
            name="test-skill",
            source="github",
            identifier="owner/repo/test-skill",
            trust_level="trusted",
            scan_verdict="pass",
            skill_hash="abc123",
            install_path="test-skill",
            files=["SKILL.md", "references/api.md"],
        )
        data = lock.load()
        assert "test-skill" in data["installed"]
        entry = data["installed"]["test-skill"]
        assert entry["source"] == "github"
        assert entry["trust_level"] == "trusted"
        assert entry["content_hash"] == "abc123"
        assert "installed_at" in entry

    def test_record_uninstall(self, tmp_path):
        lock = HubLockFile(path=tmp_path / "lock.json")
        lock.record_install(
            name="test-skill", source="github", identifier="x",
            trust_level="community", scan_verdict="pass",
            skill_hash="h", install_path="test-skill", files=["SKILL.md"],
        )
        lock.record_uninstall("test-skill")
        data = lock.load()
        assert "test-skill" not in data["installed"]

    def test_record_uninstall_nonexistent(self, tmp_path):
        lock = HubLockFile(path=tmp_path / "lock.json")
        lock.save({"version": 1, "installed": {}})
        # Should not raise
        lock.record_uninstall("nonexistent")

    def test_get_installed(self, tmp_path):
        lock = HubLockFile(path=tmp_path / "lock.json")
        lock.record_install(
            name="skill-a", source="github", identifier="x",
            trust_level="trusted", scan_verdict="pass",
            skill_hash="h", install_path="skill-a", files=["SKILL.md"],
        )
        assert lock.get_installed("skill-a") is not None
        assert lock.get_installed("nonexistent") is None

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
    def test_load_missing_file(self, tmp_path):
        mgr = TapsManager(path=tmp_path / "taps.json")
        assert mgr.load() == []

    def test_load_valid_file(self, tmp_path):
        taps_file = tmp_path / "taps.json"
        taps_file.write_text(json.dumps({"taps": [{"repo": "owner/repo", "path": "skills/"}]}))
        mgr = TapsManager(path=taps_file)
        taps = mgr.load()
        assert len(taps) == 1
        assert taps[0]["repo"] == "owner/repo"

    def test_load_corrupt_json(self, tmp_path):
        taps_file = tmp_path / "taps.json"
        taps_file.write_text("bad json")
        mgr = TapsManager(path=taps_file)
        assert mgr.load() == []

    def test_add_new_tap(self, tmp_path):
        mgr = TapsManager(path=tmp_path / "taps.json")
        assert mgr.add("owner/repo", "skills/") is True
        taps = mgr.load()
        assert len(taps) == 1
        assert taps[0]["repo"] == "owner/repo"

    def test_add_duplicate_tap(self, tmp_path):
        mgr = TapsManager(path=tmp_path / "taps.json")
        mgr.add("owner/repo")
        assert mgr.add("owner/repo") is False
        assert len(mgr.load()) == 1

    def test_remove_existing_tap(self, tmp_path):
        mgr = TapsManager(path=tmp_path / "taps.json")
        mgr.add("owner/repo")
        assert mgr.remove("owner/repo") is True
        assert mgr.load() == []

    def test_remove_nonexistent_tap(self, tmp_path):
        mgr = TapsManager(path=tmp_path / "taps.json")
        assert mgr.remove("nonexistent") is False

    def test_list_taps(self, tmp_path):
        mgr = TapsManager(path=tmp_path / "taps.json")
        mgr.add("repo-a/skills")
        mgr.add("repo-b/tools")
        taps = mgr.list_taps()
        assert len(taps) == 2


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

    def test_missing_system_role(self):
        agent_data = {
            "identifier": "no-role",
            "meta": {"title": "No Role", "description": "Desc."},
        }
        result = LobeHubSource._convert_to_skill_md(agent_data)
        assert "(No system role defined)" in result

    def test_missing_meta(self):
        agent_data = {"identifier": "bare-agent"}
        result = LobeHubSource._convert_to_skill_md(agent_data)
        assert "name: bare-agent" in result


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
        s1 = SkillMeta(name="skill", description="from A", source="a",
                        identifier="a/skill", trust_level="community")
        s2 = SkillMeta(name="skill", description="from B", source="b",
                        identifier="b/skill", trust_level="community")
        src_a = self._make_source("a", [s1])
        src_b = self._make_source("b", [s2])
        results = unified_search("skill", [src_a, src_b])
        assert len(results) == 1
        assert results[0].description == "from A"

    def test_dedup_prefers_trusted_over_community(self):
        community = SkillMeta(name="skill", description="community", source="a",
                               identifier="a/skill", trust_level="community")
        trusted = SkillMeta(name="skill", description="trusted", source="b",
                             identifier="b/skill", trust_level="trusted")
        src_a = self._make_source("a", [community])
        src_b = self._make_source("b", [trusted])
        results = unified_search("skill", [src_a, src_b])
        assert len(results) == 1
        assert results[0].trust_level == "trusted"

    def test_dedup_prefers_builtin_over_trusted(self):
        """Regression: builtin must not be overwritten by trusted."""
        builtin = SkillMeta(name="skill", description="builtin", source="a",
                             identifier="a/skill", trust_level="builtin")
        trusted = SkillMeta(name="skill", description="trusted", source="b",
                             identifier="b/skill", trust_level="trusted")
        src_a = self._make_source("a", [builtin])
        src_b = self._make_source("b", [trusted])
        results = unified_search("skill", [src_a, src_b])
        assert len(results) == 1
        assert results[0].trust_level == "builtin"

    def test_dedup_trusted_not_overwritten_by_community(self):
        trusted = SkillMeta(name="skill", description="trusted", source="a",
                             identifier="a/skill", trust_level="trusted")
        community = SkillMeta(name="skill", description="community", source="b",
                               identifier="b/skill", trust_level="community")
        src_a = self._make_source("a", [trusted])
        src_b = self._make_source("b", [community])
        results = unified_search("skill", [src_a, src_b])
        assert results[0].trust_level == "trusted"

    def test_source_filter(self):
        s1 = SkillMeta(name="s1", description="d", source="a",
                        identifier="x", trust_level="community")
        s2 = SkillMeta(name="s2", description="d", source="b",
                        identifier="y", trust_level="community")
        src_a = self._make_source("a", [s1])
        src_b = self._make_source("b", [s2])
        results = unified_search("query", [src_a, src_b], source_filter="a")
        assert len(results) == 1
        assert results[0].name == "s1"

    def test_limit_respected(self):
        skills = [
            SkillMeta(name=f"s{i}", description="d", source="a",
                       identifier=f"a/s{i}", trust_level="community")
            for i in range(20)
        ]
        src = self._make_source("a", skills)
        results = unified_search("query", [src], limit=5)
        assert len(results) == 5

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

    def test_appends_multiple_entries(self, tmp_path):
        log_file = tmp_path / "audit.log"
        with patch("tools.skills_hub.AUDIT_LOG", log_file):
            append_audit_log("INSTALL", "s1", "github", "trusted", "pass")
            append_audit_log("UNINSTALL", "s1", "github", "trusted", "n/a")
        lines = log_file.read_text().strip().split("\n")
        assert len(lines) == 2

    def test_extra_field_included(self, tmp_path):
        log_file = tmp_path / "audit.log"
        with patch("tools.skills_hub.AUDIT_LOG", log_file):
            append_audit_log("INSTALL", "s1", "github", "trusted", "pass", extra="hash123")
        content = log_file.read_text()
        assert "hash123" in content


# ---------------------------------------------------------------------------
# _skill_meta_to_dict
# ---------------------------------------------------------------------------


class TestSkillMetaToDict:
    def test_roundtrip(self):
        meta = SkillMeta(
            name="test", description="desc", source="github",
            identifier="owner/repo/test", trust_level="trusted",
            repo="owner/repo", path="skills/test", tags=["a", "b"],
        )
        d = _skill_meta_to_dict(meta)
        assert d["name"] == "test"
        assert d["tags"] == ["a", "b"]
        # Can reconstruct from dict
        restored = SkillMeta(**d)
        assert restored.name == meta.name
        assert restored.trust_level == meta.trust_level


# ---------------------------------------------------------------------------
# Official skills / binary assets
# ---------------------------------------------------------------------------


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

    @patch.object(GitHubSource, "_download_directory_recursive", return_value={"SKILL.md": "# ok"})
    @patch("tools.skills_hub.httpx.get")
    def test_falls_back_on_repo_api_failure(self, mock_get, mock_fallback):
        """When the repo endpoint returns non-200, fall back to Contents API."""
        mock_get.return_value = MagicMock(status_code=404)

        src = self._source()
        files = src._download_directory("owner/repo", "skills/my-skill")

        assert files == {"SKILL.md": "# ok"}
        mock_fallback.assert_called_once()

    @patch.object(GitHubSource, "_fetch_file_content")
    @patch("tools.skills_hub.httpx.get")
    def test_tree_api_skips_failed_file_fetches(self, mock_get, mock_fetch):
        """Files that fail to fetch are skipped, not fatal."""
        repo_resp = MagicMock(status_code=200, json=lambda: {"default_branch": "main"})
        tree_resp = MagicMock(status_code=200, json=lambda: {
            "truncated": False,
            "tree": [
                {"type": "blob", "path": "skills/my-skill/SKILL.md"},
                {"type": "blob", "path": "skills/my-skill/scripts/run.py"},
            ],
        })
        mock_get.side_effect = [repo_resp, tree_resp]
        mock_fetch.side_effect = lambda repo, path: (
            "# Skill" if path.endswith("SKILL.md") else None
        )

        src = self._source()
        files = src._download_directory("owner/repo", "skills/my-skill")

        assert "SKILL.md" in files
        assert "scripts/run.py" not in files

    @patch.object(GitHubSource, "_download_directory_recursive", return_value={})
    @patch("tools.skills_hub.httpx.get")
    def test_falls_back_on_network_error(self, mock_get, mock_fallback):
        """Network errors in tree API trigger fallback."""
        mock_get.side_effect = httpx.ConnectError("connection refused")

        src = self._source()
        src._download_directory("owner/repo", "skills/my-skill")

        mock_fallback.assert_called_once()


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

    @patch.object(GitHubSource, "_fetch_file_content")
    @patch("tools.skills_hub.httpx.get")
    def test_recursive_handles_subdir_failure(self, mock_get, mock_fetch):
        """Subdirectory 403/rate-limit returns empty but doesn't crash."""
        root_resp = MagicMock(status_code=200, json=lambda: [
            {"name": "SKILL.md", "type": "file", "path": "skill/SKILL.md"},
            {"name": "scripts", "type": "dir", "path": "skill/scripts"},
        ])
        sub_resp = MagicMock(status_code=403)
        mock_get.side_effect = [root_resp, sub_resp]
        mock_fetch.return_value = "content"

        src = self._source()
        files = src._download_directory_recursive("owner/repo", "skill")

        assert "SKILL.md" in files
        assert "scripts/run.py" not in files  # lost due to rate limit
