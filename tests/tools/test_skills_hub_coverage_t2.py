"""Coverage for SkillsShSource sitemap catalog walk + discovery/token helpers.

Executes tools.skills_hub._sitemap_catalog / _featured_skills (the sitemap
index fetch with the brotli/gzip Accept-Encoding workaround, the per-skill
sitemap walk, cache read/write, and the fallback to the featured scrape), plus
_discover_identifier, _resolve_github_meta, _finalize_inspect_meta,
_matches_skill_tokens and _token_variants.

All tests are deterministic and offline: httpx.get is patched at the module
level, and _read_index_cache/_write_index_cache are patched the same way the
ClawHub sitemap-walk tests do it.
"""

import unittest
from unittest.mock import patch, MagicMock

import httpx

from tools.skills_hub import SkillsShSource, GitHubAuth, SkillMeta


class _MockResponse:
    def __init__(self, status_code=200, json_data=None, text="", headers=None):
        self.status_code = status_code
        self._json_data = json_data
        self.text = text
        self.headers = headers or {}

    def json(self):
        return self._json_data


# Sitemap index: points at two per-skill sitemaps plus one "other" loc that
# must NOT be collected (it does not contain "sitemap-skills").
_INDEX_XML = (
    "<urlset>"
    "<url><loc>https://www.skills.sh/sitemap-skills-1.xml</loc></url>"
    "<url><loc>https://www.skills.sh/sitemap-skills-2.xml</loc></url>"
    "<url><loc>https://www.skills.sh/sitemap-other.xml</loc></url>"
    "</urlset>"
)

# Per-skill sitemap #1: canonical skills, one duplicate, and one deeper path
# that the _SITEMAP_SKILL_RE must reject (extra "/" segment).
_SKILL_MAP_1 = (
    "<urlset>"
    "<url><loc>https://skills.sh/owner1/repo1/skill-a</loc></url>"
    "<url><loc>https://skills.sh/owner1/repo1/skill-b</loc></url>"
    "<url><loc>https://skills.sh/owner1/repo1/skill-a</loc></url>"
    "<url><loc>https://skills.sh/owner1/repo1/sub/skill-a</loc></url>"
    "</urlset>"
)

_SKILL_MAP_2 = (
    "<urlset>"
    "<url><loc>https://skills.sh/owner2/repo2/skill-c</loc></url>"
    "</urlset>"
)

# A sitemap whose locs all fail the per-skill regex -> the walk yields nothing.
_EMPTY_MAP = (
    "<urlset>"
    "<url><loc>https://skills.sh/owner/repo/sub/deep-skill</loc></url>"
    "<url><loc>https://www.skills.sh/other/page</loc></url>"
    "</urlset>"
)

_FEATURED_HTML = (
    '<a href="/owner5/repo5/skill-d">Skill D</a>'
    '<a href="/owner5/repo5/skill-d">duplicate</a>'
    '<a href="/owner6/repo6/skill-e">Skill E</a>'
    '<a href="/agents/x/y">ignored</a>'
    '<a href="/_next/x/y">ignored</a>'
    '<a href="/api/x/y">ignored</a>'
)

_FEATURED_FALLBACK = SkillMeta(
    name="featured-fallback",
    description="fallback",
    source="skills.sh",
    identifier="skills-sh/owner/repo/featured-fallback",
    trust_level="community",
    repo="owner/repo",
    path="featured-fallback",
)

_CACHE_ITEM_A = {
    "name": "skill-a",
    "description": "desc-a",
    "source": "skills.sh",
    "identifier": "skills-sh/owner/repo/skill-a",
    "trust_level": "community",
    "repo": "owner/repo",
    "path": "skill-a",
    "tags": [],
    "extra": {},
}
_CACHE_ITEM_B = {
    "name": "skill-b",
    "description": "desc-b",
    "source": "skills.sh",
    "identifier": "skills-sh/owner/repo/skill-b",
    "trust_level": "community",
    "repo": "owner/repo",
    "path": "skill-b",
    "tags": [],
    "extra": {},
}


class TestSkillsShSitemapCatalog(unittest.TestCase):
    """SkillsShSource._sitemap_catalog + _featured_skills walk behavior."""

    def _src(self):
        return SkillsShSource(auth=MagicMock(spec=GitHubAuth))

    @staticmethod
    def _index_only_side_effect():
        """A side_effect that serves ONLY the sitemap index URL."""
        def side_effect(url, *args, **kwargs):
            if url == SkillsShSource.SITEMAP_INDEX_URL:
                return _MockResponse(status_code=200, text=_INDEX_XML)
            return _MockResponse(status_code=404)
        return side_effect

    @staticmethod
    def _full_walk_side_effect(index_xml=_INDEX_XML, maps=None, featured_html=""):
        """A side_effect that behaves like the real skills.sh server."""
        maps = maps or {}

        def side_effect(url, *args, **kwargs):
            if url == SkillsShSource.SITEMAP_INDEX_URL:
                return _MockResponse(status_code=200, text=index_xml)
            if "sitemap-skills" in url:
                return _MockResponse(status_code=200, text=maps.get(url, _EMPTY_MAP))
            if url == SkillsShSource.BASE_URL:
                return _MockResponse(status_code=200, text=featured_html)
            return _MockResponse(status_code=404)
        return side_effect

    # ---- cache read block -------------------------------------------------

    @patch("tools.skills_hub._write_index_cache")
    @patch("tools.skills_hub._read_index_cache")
    @patch("tools.skills_hub.httpx.get")
    def test_sitemap_catalog_cache_hit_returns_skill_metas(
        self, mock_get, mock_read, mock_write
    ):
        """A valid cache read yields SkillMeta objects without any network."""
        mock_read.return_value = [_CACHE_ITEM_A, _CACHE_ITEM_B]

        results = self._src()._sitemap_catalog(limit=10)

        self.assertEqual(len(results), 2)
        self.assertTrue(all(isinstance(m, SkillMeta) for m in results))
        self.assertEqual(results[0].identifier, "skills-sh/owner/repo/skill-a")
        self.assertEqual(results[1].name, "skill-b")
        mock_get.assert_not_called()
        mock_write.assert_not_called()

    @patch("tools.skills_hub._write_index_cache")
    @patch("tools.skills_hub._read_index_cache")
    @patch("tools.skills_hub.httpx.get")
    def test_sitemap_catalog_cache_hit_zero_limit_returns_all(
        self, mock_get, mock_read, mock_write
    ):
        """limit <= 0 means "no truncation" on the cached list."""
        mock_read.return_value = [_CACHE_ITEM_A, _CACHE_ITEM_B]

        results = self._src()._sitemap_catalog(limit=0)

        self.assertEqual(len(results), 2)
        mock_get.assert_not_called()
        mock_write.assert_not_called()

    # ---- sitemap index fetch / Accept-Encoding workaround -----------------

    @patch("tools.skills_hub._write_index_cache")
    @patch("tools.skills_hub._read_index_cache", return_value=None)
    @patch("tools.skills_hub.httpx.get")
    def test_sitemap_catalog_walk_caches_on_natural_termination(
        self, mock_get, mock_read, mock_write
    ):
        """Happy path: index + per-skill maps dedupe, emit metas, and cache."""
        maps = {
            "https://www.skills.sh/sitemap-skills-1.xml": _SKILL_MAP_1,
            "https://www.skills.sh/sitemap-skills-2.xml": _SKILL_MAP_2,
        }
        mock_get.side_effect = self._full_walk_side_effect(maps=maps)

        results = self._src()._sitemap_catalog(limit=10)

        self.assertEqual(len(results), 3)
        identifiers = {m.identifier for m in results}
        self.assertEqual(
            identifiers,
            {
                "skills-sh/owner1/repo1/skill-a",
                "skills-sh/owner1/repo1/skill-b",
                "skills-sh/owner2/repo2/skill-c",
            },
        )
        # Duplicate canonical and the deeper non-matching path survived dedupe.
        self.assertEqual(len(results), len(identifiers))
        self.assertEqual(results[0].source, "skills.sh")
        self.assertEqual(results[0].extra["detail_url"], "https://skills.sh/owner1/repo1/skill-a")
        self.assertEqual(results[0].extra["repo_url"], "https://github.com/owner1/repo1")

        # The brotli/gzip Accept-Encoding workaround header is plumbed through.
        mock_get.assert_any_call(
            SkillsShSource.SITEMAP_INDEX_URL,
            timeout=20,
            follow_redirects=True,
            headers={"Accept-Encoding": "gzip"},
        )
        mock_write.assert_called_once()
        self.assertEqual(mock_write.call_args.args[0], "skills_sh_sitemap_v1")
        self.assertEqual(len(mock_write.call_args.args[1]), 3)

    @patch("tools.skills_hub._write_index_cache")
    @patch("tools.skills_hub._read_index_cache", return_value=None)
    @patch("tools.skills_hub.httpx.get")
    def test_sitemap_catalog_limit_slices_results(
        self, mock_get, mock_read, mock_write
    ):
        """A positive limit truncates the walked catalog."""
        maps = {
            "https://www.skills.sh/sitemap-skills-1.xml": _SKILL_MAP_1,
            "https://www.skills.sh/sitemap-skills-2.xml": _SKILL_MAP_2,
        }
        mock_get.side_effect = self._full_walk_side_effect(maps=maps)

        results = self._src()._sitemap_catalog(limit=1)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].identifier, "skills-sh/owner1/repo1/skill-a")

    @patch.object(SkillsShSource, "_featured_skills", return_value=[_FEATURED_FALLBACK])
    @patch("tools.skills_hub._write_index_cache")
    @patch("tools.skills_hub._read_index_cache", return_value=None)
    @patch("tools.skills_hub.httpx.get")
    def test_sitemap_catalog_non_200_index_falls_back_to_featured(
        self, mock_get, mock_read, mock_write, mock_featured
    ):
        """A non-200 sitemap index falls back to the featured scrape."""
        mock_get.return_value = _MockResponse(status_code=404)

        results = self._src()._sitemap_catalog(limit=5)

        self.assertEqual(results, [_FEATURED_FALLBACK])
        mock_featured.assert_called_once_with(5)
        mock_write.assert_not_called()

    @patch.object(SkillsShSource, "_featured_skills", return_value=[_FEATURED_FALLBACK])
    @patch("tools.skills_hub._write_index_cache")
    @patch("tools.skills_hub._read_index_cache", return_value=None)
    @patch("tools.skills_hub.httpx.get")
    def test_sitemap_catalog_httpx_error_on_index_falls_back_to_featured(
        self, mock_get, mock_read, mock_write, mock_featured
    ):
        """An httpx.HTTPError on the index fetch falls back to featured."""
        mock_get.side_effect = httpx.HTTPError("boom")

        results = self._src()._sitemap_catalog(limit=5)

        self.assertEqual(results, [_FEATURED_FALLBACK])
        mock_featured.assert_called_once_with(5)
        mock_write.assert_not_called()

    @patch.object(SkillsShSource, "_featured_skills", return_value=[_FEATURED_FALLBACK])
    @patch("tools.skills_hub._write_index_cache")
    @patch("tools.skills_hub._read_index_cache", return_value=None)
    @patch("tools.skills_hub.httpx.get")
    def test_sitemap_catalog_no_skill_sitemaps_falls_back_to_featured(
        self, mock_get, mock_read, mock_write, mock_featured
    ):
        """An index without any sitemap-skills loc yields no URLs -> fallback."""
        mock_get.side_effect = self._full_walk_side_effect(
            index_xml="<urlset><url><loc>https://www.skills.sh/nonsense.xml</loc></url></urlset>",
            maps={},
        )

        results = self._src()._sitemap_catalog(limit=5)

        self.assertEqual(results, [_FEATURED_FALLBACK])
        mock_featured.assert_called_once_with(5)
        mock_write.assert_not_called()

    @patch.object(SkillsShSource, "_featured_skills", return_value=[_FEATURED_FALLBACK])
    @patch("tools.skills_hub._write_index_cache")
    @patch("tools.skills_hub._read_index_cache", return_value=None)
    @patch("tools.skills_hub.httpx.get")
    def test_sitemap_catalog_skill_map_errors_continue_and_fallback(
        self, mock_get, mock_read, mock_write, mock_featured
    ):
        """Per-skill map failures (HTTPError + non-200) `continue`, not abort."""
        def side_effect(url, *args, **kwargs):
            if url == SkillsShSource.SITEMAP_INDEX_URL:
                return _MockResponse(
                    status_code=200,
                    text="<urlset>"
                    "<url><loc>https://www.skills.sh/sitemap-skills-1.xml</loc></url>"
                    "<url><loc>https://www.skills.sh/sitemap-skills-2.xml</loc></url>"
                    "</urlset>",
                )
            if "sitemap-skills-1" in url:
                raise httpx.HTTPError("map 1 exploded")
            if "sitemap-skills-2" in url:
                return _MockResponse(status_code=403)
            return _MockResponse(status_code=404)

        mock_get.side_effect = side_effect

        results = self._src()._sitemap_catalog(limit=5)

        self.assertEqual(results, [_FEATURED_FALLBACK])
        mock_featured.assert_called_once_with(5)
        mock_write.assert_not_called()

    @patch.object(SkillsShSource, "_featured_skills", return_value=[_FEATURED_FALLBACK])
    @patch("tools.skills_hub._write_index_cache")
    @patch("tools.skills_hub._read_index_cache", return_value=None)
    @patch("tools.skills_hub.httpx.get")
    def test_sitemap_catalog_empty_walk_falls_back_and_does_not_cache(
        self, mock_get, mock_read, mock_write, mock_featured
    ):
        """A walk that yields no skill URLs must NOT poison the sitemap cache."""
        maps = {"https://www.skills.sh/sitemap-skills-1.xml": _EMPTY_MAP}
        mock_get.side_effect = self._full_walk_side_effect(maps=maps)

        results = self._src()._sitemap_catalog(limit=5)

        self.assertEqual(results, [_FEATURED_FALLBACK])
        mock_featured.assert_called_once_with(5)
        # No sitemap cache write from a fruitless walk.
        mock_write.assert_not_called()

    @patch("tools.skills_hub._write_index_cache")
    @patch("tools.skills_hub._read_index_cache", return_value=None)
    @patch("tools.skills_hub.httpx.get")
    def test_search_empty_query_routes_to_sitemap_catalog(
        self, mock_get, mock_read, mock_write
    ):
        """search("", limit=N) is the bulk dump path -> walks the sitemap."""
        maps = {
            "https://www.skills.sh/sitemap-skills-1.xml": _SKILL_MAP_1,
            "https://www.skills.sh/sitemap-skills-2.xml": _SKILL_MAP_2,
        }
        mock_get.side_effect = self._full_walk_side_effect(maps=maps)

        results = self._src().search("", limit=10)

        self.assertEqual(len(results), 3)
        mock_write.assert_called_once()

    # ---- _featured_skills --------------------------------------------------

    @patch("tools.skills_hub._write_index_cache")
    @patch("tools.skills_hub._read_index_cache")
    @patch("tools.skills_hub.httpx.get")
    def test_featured_skills_cache_hit_returns_metas(
        self, mock_get, mock_read, mock_write
    ):
        mock_read.return_value = [_CACHE_ITEM_A]

        results = self._src()._featured_skills(limit=10)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].identifier, "skills-sh/owner/repo/skill-a")
        mock_get.assert_not_called()
        mock_write.assert_not_called()

    @patch("tools.skills_hub._write_index_cache")
    @patch("tools.skills_hub._read_index_cache", return_value=None)
    @patch("tools.skills_hub.httpx.get")
    def test_featured_skills_parses_and_dedupes_and_caches(
        self, mock_get, mock_read, mock_write
    ):
        mock_get.return_value = _MockResponse(status_code=200, text=_FEATURED_HTML)

        results = self._src()._featured_skills(limit=10)

        self.assertEqual(len(results), 2)
        identifiers = {m.identifier for m in results}
        self.assertEqual(
            identifiers,
            {
                "skills-sh/owner5/repo5/skill-d",
                "skills-sh/owner6/repo6/skill-e",
            },
        )
        mock_write.assert_called_once()
        self.assertEqual(mock_write.call_args.args[0], "skills_sh_featured")
        self.assertEqual(len(mock_write.call_args.args[1]), 2)

    @patch("tools.skills_hub._write_index_cache")
    @patch("tools.skills_hub._read_index_cache", return_value=None)
    @patch("tools.skills_hub.httpx.get")
    def test_featured_skills_limit_breaks(
        self, mock_get, mock_read, mock_write
    ):
        html = '<a href="/owner5/repo5/skill-d">D</a><a href="/owner6/repo6/skill-e">E</a>'
        mock_get.return_value = _MockResponse(status_code=200, text=html)

        results = self._src()._featured_skills(limit=1)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].identifier, "skills-sh/owner5/repo5/skill-d")

    @patch("tools.skills_hub._write_index_cache")
    @patch("tools.skills_hub._read_index_cache", return_value=None)
    @patch("tools.skills_hub.httpx.get")
    def test_featured_skills_non_200_returns_empty(
        self, mock_get, mock_read, mock_write
    ):
        mock_get.return_value = _MockResponse(status_code=500)

        results = self._src()._featured_skills(limit=5)

        self.assertEqual(results, [])
        mock_write.assert_not_called()

    @patch("tools.skills_hub._write_index_cache")
    @patch("tools.skills_hub._read_index_cache", return_value=None)
    @patch("tools.skills_hub.httpx.get")
    def test_featured_skills_httpx_error_returns_empty(
        self, mock_get, mock_read, mock_write
    ):
        mock_get.side_effect = httpx.HTTPError("boom")

        results = self._src()._featured_skills(limit=5)

        self.assertEqual(results, [])
        mock_write.assert_not_called()


def _token_based_source():
    """A SkillsShSource whose internal GitHubSource is a controllable mock."""
    src = SkillsShSource(auth=MagicMock(spec=GitHubAuth))
    src.github = MagicMock()
    src.github.trust_level_for.return_value = "community"
    src.github._list_skills_in_repo.return_value = []
    src.github._find_skill_in_repo_tree.return_value = None
    src.github.inspect.return_value = None
    src.github.auth.get_headers.return_value = {}
    return src


class TestSkillsShDiscovery(unittest.TestCase):
    """_discover_identifier / _resolve_github_meta / _finalize_inspect_meta."""

    def test_discover_identifier_short_identifier_returns_none(self):
        src = _token_based_source()
        self.assertIsNone(src._discover_identifier("owner/repo"))
        self.assertIsNone(src._discover_identifier("just-one"))

    @patch("tools.skills_hub.httpx.get")
    def test_discover_identifier_base_paths_match_returns_identifier(self, mock_get):
        src = _token_based_source()
        meta = SkillMeta(
            name="my-skill",
            description="d",
            source="github",
            identifier="owner/repo/skills/foo/my-skill",
            trust_level="community",
            path="skills/foo/my-skill",
        )
        src.github._list_skills_in_repo.return_value = [meta]

        result = src._discover_identifier("owner/repo/foo/my-skill")

        self.assertEqual(result, "owner/repo/skills/foo/my-skill")

    @patch("tools.skills_hub.httpx.get")
    def test_discover_identifier_uses_repo_from_detail_and_token_extras(self, mock_get):
        src = _token_based_source()
        meta = SkillMeta(
            name="my-skill",
            description="d",
            source="github",
            identifier="custom/repo/my-skill",
            trust_level="community",
            path="my-skill",
        )
        calls = []

        def list_side(repo, base_path):
            calls.append((repo, base_path))
            if repo == "custom/repo" and base_path == "skills/":
                return [meta]
            return []

        src.github._list_skills_in_repo.side_effect = list_side

        result = src._discover_identifier(
            "owner/repo/foo/my-skill",
            detail={
                "repo": "custom/repo",
                "install_skill": "my-skill",
                "page_title": "My Skill Page",
                "body_title": "Skill Body",
            },
        )

        self.assertEqual(result, "custom/repo/my-skill")
        # detail.repo overrode the owner/repo default on the first base path.
        self.assertIn(("custom/repo", "skills/"), calls)

    @patch("tools.skills_hub.httpx.get")
    def test_discover_identifier_base_path_exception_continues_to_tree(self, mock_get):
        src = _token_based_source()
        src.github._list_skills_in_repo.side_effect = RuntimeError("boom")
        src.github._find_skill_in_repo_tree.return_value = "owner/repo/tree/path"

        result = src._discover_identifier("owner/repo/foo")

        self.assertEqual(result, "owner/repo/tree/path")

    @patch("tools.skills_hub.httpx.get")
    def test_discover_identifier_tree_result_wins(self, mock_get):
        src = _token_based_source()
        src.github._find_skill_in_repo_tree.return_value = "owner/repo/deep/path"

        result = src._discover_identifier("owner/repo/foo/my-skill")

        self.assertEqual(result, "owner/repo/deep/path")

    @patch("tools.skills_hub.httpx.get")
    def test_discover_identifier_root_listing_direct_inspect_and_skips(self, mock_get):
        src = _token_based_source()
        meta = SkillMeta(
            name="my-skill",
            description="d",
            source="github",
            identifier="owner/repo/packages/my-skill",
            trust_level="community",
            path="packages/my-skill",
        )
        src.github.inspect.side_effect = (
            lambda ident: meta if ident == "owner/repo/packages/my-skill" else None
        )
        # Non-dir + dot/underscore + reserved dirs come first so their skip
        # branches actually run before the matching `packages` dir.
        entries = [
            {"type": "file", "name": "README.md"},
            {"type": "dir", "name": ".hidden"},
            {"type": "dir", "name": "skills"},
            {"type": "dir", "name": "_private"},
            {"type": "dir", "name": "packages"},
        ]
        mock_get.return_value = _MockResponse(status_code=200, json_data=entries)

        result = src._discover_identifier("owner/repo/foo/my-skill")

        self.assertEqual(result, "owner/repo/packages/my-skill")
        # Only the candidate "packages" dir was tried — files, dot/underscore
        # dirs and the reserved skills dirs were skipped.
        self.assertEqual(
            [c.args[0] for c in src.github.inspect.call_args_list],
            ["owner/repo/packages/my-skill"],
        )

    @patch("tools.skills_hub.httpx.get")
    def test_discover_identifier_root_listing_list_skills_match(self, mock_get):
        src = _token_based_source()
        matching = SkillMeta(
            name="my-skill",
            description="d",
            source="github",
            identifier="owner/repo/packages/foo/my-skill",
            trust_level="community",
            path="packages/foo/my-skill",
        )

        def list_side(repo, base_path):
            if base_path == "packages/":
                return [matching]
            return []

        src.github._list_skills_in_repo.side_effect = list_side
        src.github._find_skill_in_repo_tree.return_value = None
        src.github.inspect.return_value = None
        mock_get.return_value = _MockResponse(
            status_code=200, json_data=[{"type": "dir", "name": "packages"}]
        )

        result = src._discover_identifier("owner/repo/foo/my-skill")

        self.assertEqual(result, "owner/repo/packages/foo/my-skill")

    @patch("tools.skills_hub.httpx.get")
    def test_discover_identifier_root_listing_dir_listing_exception_continues(self, mock_get):
        """A dir whose skill listing raises is skipped, not fatal."""
        src = _token_based_source()

        def list_side(repo, base_path):
            if base_path == "packages/":
                raise RuntimeError("boom")
            return []

        src.github._list_skills_in_repo.side_effect = list_side
        src.github._find_skill_in_repo_tree.return_value = None
        src.github.inspect.return_value = None
        mock_get.return_value = _MockResponse(
            status_code=200, json_data=[{"type": "dir", "name": "packages"}]
        )

        result = src._discover_identifier("owner/repo/foo/my-skill")

        self.assertIsNone(result)

    @patch("tools.skills_hub.httpx.get")
    def test_discover_identifier_root_listing_nothing_found_returns_none(self, mock_get):
        src = _token_based_source()
        src.github._find_skill_in_repo_tree.return_value = None
        src.github.inspect.return_value = None
        src.github._list_skills_in_repo.return_value = []
        mock_get.return_value = _MockResponse(
            status_code=200, json_data=[{"type": "dir", "name": "packages"}]
        )

        result = src._discover_identifier("owner/repo/foo/my-skill")

        self.assertIsNone(result)

    @patch("tools.skills_hub.httpx.get")
    def test_discover_identifier_root_listing_exception_returns_none(self, mock_get):
        src = _token_based_source()
        mock_get.side_effect = RuntimeError("api down")

        result = src._discover_identifier("owner/repo/foo/my-skill")

        self.assertIsNone(result)

    # ---- _resolve_github_meta ----------------------------------------------

    @patch("tools.skills_hub.httpx.get")
    def test_resolve_github_meta_candidate_inspect_hit(self, mock_get):
        src = _token_based_source()
        meta = SkillMeta(
            name="foo",
            description="d",
            source="github",
            identifier="owner/repo/foo",
            trust_level="community",
        )
        src.github.inspect.side_effect = (
            lambda ident: meta if ident == "owner/repo/foo" else None
        )

        result = src._resolve_github_meta("owner/repo/foo")

        self.assertIs(result, meta)

    @patch("tools.skills_hub.httpx.get")
    def test_resolve_github_meta_discovers_then_inspects(self, mock_get):
        src = _token_based_source()
        meta = SkillMeta(
            name="foo",
            description="d",
            source="github",
            identifier="owner/repo/deep/path",
            trust_level="community",
        )
        src.github.inspect.side_effect = (
            lambda ident: meta if ident == "owner/repo/deep/path" else None
        )
        src.github._find_skill_in_repo_tree.return_value = "owner/repo/deep/path"

        result = src._resolve_github_meta("owner/repo/foo")

        self.assertIs(result, meta)

    @patch("tools.skills_hub.httpx.get")
    def test_resolve_github_meta_none(self, mock_get):
        src = _token_based_source()
        src.github._find_skill_in_repo_tree.return_value = None
        src.github.inspect.return_value = None
        src.github._list_skills_in_repo.return_value = []
        mock_get.return_value = _MockResponse(status_code=200, json_data=[])

        result = src._resolve_github_meta("owner/repo/foo")

        self.assertIsNone(result)

    # ---- _finalize_inspect_meta ---------------------------------------------

    def _finalize_meta(self):
        return SkillMeta(
            name="orig",
            description="orig desc",
            source="github",
            identifier="github/owner/repo/foo",
            trust_level="trusted",
            repo="owner/repo",
            path="foo",
            extra={"keep": "me"},
        )

    def test_finalize_inspect_meta_sets_fields_and_merges_extra(self):
        src = _token_based_source()
        meta = self._finalize_meta()
        detail = {"repo": "owner/repo", "install_command": "npx skills add owner/repo"}

        result = src._finalize_inspect_meta(meta, "owner/repo/foo", detail)

        self.assertEqual(result.source, "skills.sh")
        self.assertEqual(result.identifier, "skills-sh/owner/repo/foo")
        self.assertEqual(result.trust_level, "community")
        self.assertEqual(result.description, "orig desc")
        self.assertEqual(result.extra["keep"], "me")
        self.assertEqual(result.extra["detail_url"], "https://skills.sh/owner/repo/foo")
        self.assertEqual(result.extra["repo_url"], "https://github.com/owner/repo")
        self.assertEqual(result.extra["install_command"], "npx skills add owner/repo")

    def test_finalize_inspect_meta_description_from_body_summary(self):
        src = _token_based_source()
        meta = self._finalize_meta()
        result = src._finalize_inspect_meta(meta, "owner/repo/foo", {"body_summary": "Great skill"})
        self.assertEqual(result.description, "Great skill")

    def test_finalize_inspect_meta_weekly_installs_description(self):
        src = _token_based_source()
        meta = self._finalize_meta()
        result = src._finalize_inspect_meta(
            meta, "owner/repo/foo", {"weekly_installs": "1,500"}
        )
        self.assertEqual(result.description, "orig desc · 1,500 weekly installs on skills.sh")
        self.assertEqual(result.extra["weekly_installs"], "1,500")

    def test_finalize_inspect_meta_no_detail(self):
        src = _token_based_source()
        meta = self._finalize_meta()
        result = src._finalize_inspect_meta(meta, "owner/repo/foo", None)
        self.assertEqual(result.source, "skills.sh")
        self.assertEqual(result.extra["detail_url"], "https://skills.sh/owner/repo/foo")
        self.assertEqual(result.description, "orig desc")


class TestSkillsShTokenMatching(unittest.TestCase):
    """_matches_skill_tokens / _token_variants."""

    def test_token_variants_none_and_empty(self):
        self.assertEqual(SkillsShSource._token_variants(None), set())
        self.assertEqual(SkillsShSource._token_variants(""), set())
        self.assertEqual(SkillsShSource._token_variants("  ///  "), set())

    def test_token_variants_html_and_normalization(self):
        variants = SkillsShSource._token_variants("repo/my_skill")
        self.assertEqual(
            variants,
            {
                "repo/my_skill",
                "repo/my-skill",
                "repo-my_skill",
                "my_skill",
                "my-skill",
            },
        )
        self.assertIn("my-skill", SkillsShSource._token_variants("<b>repo/my_skill</b>"))
        self.assertIn("repo-my-skill", SkillsShSource._token_variants("repo/my-skill"))

    def test_token_variants_underscore_and_slash_fold(self):
        variants = SkillsShSource._token_variants("owner/repo/my-skill")
        self.assertEqual(
            variants,
            {"owner/repo/my-skill", "owner-repo-my-skill", "my-skill"},
        )

    def test_matches_skill_tokens_true_and_false(self):
        meta = SkillMeta(
            name="my-skill",
            description="d",
            source="github",
            identifier="owner/repo/skills/foo/my-skill",
            trust_level="community",
            path="skills/foo/my-skill",
        )
        self.assertTrue(SkillsShSource._matches_skill_tokens(meta, ["my-skill"]))
        # space variant folds to the dashed token
        self.assertTrue(SkillsShSource._matches_skill_tokens(meta, ["my skill"]))
        self.assertFalse(SkillsShSource._matches_skill_tokens(meta, ["unrelated"]))

    def test_matches_skill_tokens_none_identifier(self):
        meta = SkillMeta(
            name="my-skill",
            description="d",
            source="github",
            identifier=None,
            trust_level="community",
        )
        self.assertTrue(SkillsShSource._matches_skill_tokens(meta, ["my-skill"]))
        self.assertFalse(SkillsShSource._matches_skill_tokens(meta, ["other"]))


if __name__ == "__main__":
    unittest.main()
