"""Coverage tests for WellKnownSkillSource, UrlSource, and OptionalSkillSource edge branches.

These complement tests/tools/test_skills_hub.py (which already covers the
happy/primary paths) by exercising the untested branches:

* WellKnownSkillSource  — _query_to_index_url, _parse_identifier, _parse_index
  (cache hit / bad HTTP / decode error / non-list), _index_entry not-found,
  _fetch_text non-200, inspect() end-to-end, fetch() failure branches.
* UrlSource — _matches rejection branches, _fetch_bytes/_fetch_text non-200,
  the awaiting_name (name=None) bundle contract, inspect() tag parsing, and
  _resolve_skill_name's frontmatter / URL-slug / nothing-usable branches.
* OptionalSkillSource — _list_remote_skill_dirs cache & offline paths,
  _upstream_pointer(_from_content) validation, _fetch_from_upstream,
  live-repo content-fetch / missing-SKILL.md failures, upstream-pointer
  routing through local fetch, and _parse_frontmatter edge cases.

All tests are deterministic and offline: HTTP is patched, no network.
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from tools.skills_hub import (
    OptionalSkillSource,
    UrlSource,
    WellKnownSkillSource,
)


# ---------------------------------------------------------------------------
# WellKnownSkillSource edge branches
# ---------------------------------------------------------------------------


class TestWellKnownSkillSourceEdge:
    @pytest.fixture(autouse=True)
    def _neutralize_http_guards(self, monkeypatch):
        # Keep any path that reaches the real guard from blocking example.com.
        monkeypatch.setattr("tools.skills_hub.is_safe_url", lambda _url: True)
        monkeypatch.setattr("tools.skills_hub.check_website_access", lambda _url: None)

    def _source(self):
        return WellKnownSkillSource()

    # -- _query_to_index_url -------------------------------------------------

    @pytest.mark.parametrize(
        "query, expected",
        [
            # Non-HTTP (bare name / slug) — no index URL.
            ("git-workflow", None),
            ("", None),
            # Exact index.json URL passed through.
            (
                "https://example.com/.well-known/skills/index.json",
                "https://example.com/.well-known/skills/index.json",
            ),
            # URL already containing the well-known base path.
            (
                "https://example.com/.well-known/skills/git-workflow",
                "https://example.com/.well-known/skills/index.json",
            ),
            # Bare origin — append the default base path.
            (
                "https://example.com",
                "https://example.com/.well-known/skills/index.json",
            ),
            (
                "https://example.com/",
                "https://example.com/.well-known/skills/index.json",
            ),
        ],
    )
    def test_query_to_index_url(self, query, expected):
        assert self._source()._query_to_index_url(query) == expected

    # -- search -------------------------------------------------------------

    @patch("tools.skills_hub._write_index_cache")
    @patch("tools.skills_hub._read_index_cache", return_value=None)
    def test_search_returns_empty_when_index_url_none(self, _read, _write):
        # Non-HTTP query produces no index URL -> early return [].
        assert self._source().search("git-workflow") == []
        _read.assert_not_called()

    @patch("tools.skills_hub._write_index_cache")
    @patch("tools.skills_hub._read_index_cache", return_value=None)
    @patch("tools.skills_hub._guarded_http_get")
    def test_search_returns_empty_when_index_fetch_fails(self, mock_get, _read, _write):
        mock_get.return_value = MagicMock(status_code=404)
        assert self._source().search("https://example.com") == []

    @patch("tools.skills_hub._write_index_cache")
    @patch("tools.skills_hub._read_index_cache", return_value=None)
    @patch("tools.skills_hub._guarded_http_get")
    def test_search_skips_entry_with_non_string_name(self, mock_get, _read, _write):
        mock_get.return_value = MagicMock(
            status_code=200,
            json=lambda: {
                "skills": [
                    {"name": 123, "description": "numeric name"},
                    {"name": "good-skill", "description": "ok"},
                ]
            },
        )
        results = self._source().search("https://example.com")
        assert [r.name for r in results] == ["good-skill"]

    @patch("tools.skills_hub._write_index_cache")
    @patch("tools.skills_hub._read_index_cache", return_value=None)
    @patch("tools.skills_hub._guarded_http_get")
    def test_search_non_list_files_defaults_to_skill_md(self, mock_get, _read, _write):
        mock_get.return_value = MagicMock(
            status_code=200,
            json=lambda: {
                "skills": [{"name": "git-workflow", "files": "not-a-list"}]
            },
        )
        results = self._source().search("https://example.com")
        assert results[0].extra["files"] == ["SKILL.md"]

    # -- inspect ------------------------------------------------------------

    @patch("tools.skills_hub._write_index_cache")
    @patch("tools.skills_hub._read_index_cache", return_value=None)
    @patch("tools.skills_hub._guarded_http_get")
    def test_inspect_reads_frontmatter_and_returns_meta(self, mock_get, _read, _write):
        def _side_effect(url, *args, **kwargs):
            if url.endswith("/index.json"):
                return MagicMock(
                    status_code=200,
                    json=lambda: {
                        "skills": [
                            {"name": "my-skill", "description": "entry-desc", "files": ["SKILL.md"]}
                        ]
                    },
                )
            if url.endswith("/my-skill/SKILL.md"):
                return MagicMock(
                    status_code=200,
                    text="---\nname: actual-name\ndescription: fm-desc\n---\nBody\n",
                )
            raise AssertionError(f"unexpected URL: {url}")

        mock_get.side_effect = _side_effect
        meta = self._source().inspect("well-known:https://example.com/.well-known/skills/my-skill")

        assert meta is not None
        assert meta.name == "actual-name"
        assert meta.description == "fm-desc"
        assert meta.identifier == "well-known:https://example.com/.well-known/skills/my-skill"
        assert meta.extra["endpoint"] == "https://example.com/.well-known/skills/my-skill"

    def test_inspect_returns_none_for_unparseable_identifier(self):
        assert self._source().inspect("https://example.com/SKILL.md") is None

    @patch("tools.skills_hub._write_index_cache")
    @patch("tools.skills_hub._read_index_cache", return_value=None)
    @patch("tools.skills_hub._guarded_http_get")
    def test_inspect_returns_none_when_entry_missing(self, mock_get, _read, _write):
        mock_get.return_value = MagicMock(
            status_code=200, json=lambda: {"skills": [{"name": "other-skill"}]}
        )
        assert self._source().inspect("well-known:https://example.com/.well-known/skills/missing") is None

    @patch("tools.skills_hub._write_index_cache")
    @patch("tools.skills_hub._read_index_cache", return_value=None)
    @patch("tools.skills_hub._guarded_http_get")
    def test_inspect_returns_none_when_skill_md_missing(self, mock_get, _read, _write):
        def _side_effect(url, *args, **kwargs):
            if url.endswith("/index.json"):
                return MagicMock(
                    status_code=200, json=lambda: {"skills": [{"name": "my-skill"}]}
                )
            return MagicMock(status_code=404)  # SKILL.md 404

        mock_get.side_effect = _side_effect
        assert self._source().inspect("well-known:https://example.com/.well-known/skills/my-skill") is None

    # -- fetch --------------------------------------------------------------

    @patch("tools.skills_hub._write_index_cache")
    @patch("tools.skills_hub._read_index_cache", return_value=None)
    @patch("tools.skills_hub._guarded_http_get")
    def test_fetch_rejects_unsafe_skill_name_in_fragment(self, mock_get, _read, _write):
        # fragment "a/b" fails _validate_skill_name -> warning + None.
        assert self._source().fetch("well-known:https://example.com/.well-known/skills/index.json#a/b") is None
        mock_get.assert_not_called()  # never reaches network

    @patch("tools.skills_hub._write_index_cache")
    @patch("tools.skills_hub._read_index_cache", return_value=None)
    @patch("tools.skills_hub._guarded_http_get")
    def test_fetch_returns_none_when_entry_missing(self, mock_get, _read, _write):
        mock_get.return_value = MagicMock(
            status_code=200, json=lambda: {"skills": [{"name": "other-skill"}]}
        )
        assert self._source().fetch("well-known:https://example.com/.well-known/skills/missing") is None

    @patch("tools.skills_hub._write_index_cache")
    @patch("tools.skills_hub._read_index_cache", return_value=None)
    @patch("tools.skills_hub._guarded_http_get")
    def test_fetch_full_bundle_with_default_files(self, mock_get, _read, _write):
        body = "---\nname: my-skill\ndescription: d\n---\nBody\n"
        def _side_effect(url, *args, **kwargs):
            if url.endswith("/index.json"):
                return MagicMock(
                    status_code=200, json=lambda: {"skills": [{"name": "my-skill"}]}
                )
            if url.endswith("/my-skill/SKILL.md"):
                return MagicMock(status_code=200, text=body)
            raise AssertionError(f"unexpected URL: {url}")

        mock_get.side_effect = _side_effect
        bundle = self._source().fetch("well-known:https://example.com/.well-known/skills/my-skill")
        assert bundle is not None
        assert bundle.name == "my-skill"
        assert bundle.files["SKILL.md"] == body
        assert bundle.source == "well-known"

    @patch("tools.skills_hub._write_index_cache")
    @patch("tools.skills_hub._read_index_cache", return_value=None)
    @patch("tools.skills_hub._guarded_http_get")
    def test_fetch_returns_none_when_skill_md_text_missing(self, mock_get, _read, _write):
        def _side_effect(url, *args, **kwargs):
            if url.endswith("/index.json"):
                return MagicMock(
                    status_code=200, json=lambda: {"skills": [{"name": "my-skill"}]}
                )
            return MagicMock(status_code=500)  # SKILL.md unreachable

        mock_get.side_effect = _side_effect
        assert self._source().fetch("well-known:https://example.com/.well-known/skills/my-skill") is None

    @patch("tools.skills_hub._write_index_cache")
    @patch("tools.skills_hub._read_index_cache", return_value=None)
    @patch("tools.skills_hub._guarded_http_get")
    def test_fetch_returns_none_when_skill_md_not_in_files(self, mock_get, _read, _write):
        # Entry advertises only a reference file; SKILL.md never lands -> None.
        def _side_effect(url, *args, **kwargs):
            if url.endswith("/index.json"):
                return MagicMock(
                    status_code=200,
                    json=lambda: {"skills": [{"name": "my-skill", "files": ["references/x.md"]}]},
                )
            if url.endswith("/my-skill/references/x.md"):
                return MagicMock(status_code=200, text="ref")
            raise AssertionError(f"unexpected URL: {url}")

        mock_get.side_effect = _side_effect
        assert self._source().fetch("well-known:https://example.com/.well-known/skills/my-skill") is None

    # -- _parse_index / _index_entry / _fetch_text --------------------------

    @patch("tools.skills_hub._guarded_http_get")
    def test_parse_index_uses_cache_without_http(self, mock_get):
        cached = {"index_url": "u", "base_url": "https://example.com/.well-known/skills", "skills": [{"name": "s"}]}
        with patch("tools.skills_hub._read_index_cache", return_value=cached):
            parsed = self._source()._parse_index("https://example.com/.well-known/skills/index.json")
        assert parsed == cached
        mock_get.assert_not_called()

    @patch("tools.skills_hub._write_index_cache")
    @patch("tools.skills_hub._read_index_cache", return_value=None)
    @patch("tools.skills_hub._guarded_http_get")
    def test_parse_index_returns_none_on_non_200(self, mock_get, _read, _write):
        mock_get.return_value = MagicMock(status_code=403)
        assert self._source()._parse_index("https://example.com/.well-known/skills/index.json") is None

    @patch("tools.skills_hub._write_index_cache")
    @patch("tools.skills_hub._read_index_cache", return_value=None)
    @patch("tools.skills_hub._guarded_http_get")
    def test_parse_index_returns_none_on_json_decode_error(self, mock_get, _read, _write):
        resp = MagicMock(status_code=200)
        resp.json.side_effect = json.JSONDecodeError("boom", "doc", 0)
        mock_get.return_value = resp
        assert self._source()._parse_index("https://example.com/.well-known/skills/index.json") is None

    @patch("tools.skills_hub._write_index_cache")
    @patch("tools.skills_hub._read_index_cache", return_value=None)
    @patch("tools.skills_hub._guarded_http_get")
    def test_parse_index_returns_none_when_skills_not_a_list(self, mock_get, _read, _write):
        mock_get.return_value = MagicMock(status_code=200, json=lambda: {"skills": "not-a-list"})
        assert self._source()._parse_index("https://example.com/.well-known/skills/index.json") is None

    @patch("tools.skills_hub._write_index_cache")
    @patch("tools.skills_hub._read_index_cache", return_value=None)
    @patch("tools.skills_hub._guarded_http_get")
    def test_parse_index_writes_cache_on_success(self, mock_get, read_cache, write_cache):
        mock_get.return_value = MagicMock(
            status_code=200, json=lambda: {"skills": [{"name": "x"}]}
        )
        idx = "https://example.com/.well-known/skills/index.json"
        parsed = self._source()._parse_index(idx)
        assert parsed["index_url"] == idx
        assert parsed["base_url"] == "https://example.com/.well-known/skills"
        assert parsed["skills"] == [{"name": "x"}]
        write_cache.assert_called_once()

    @patch("tools.skills_hub._write_index_cache")
    @patch("tools.skills_hub._read_index_cache", return_value=None)
    @patch("tools.skills_hub._guarded_http_get")
    def test_index_entry_returns_none_when_not_found(self, mock_get, _read, _write):
        mock_get.return_value = MagicMock(
            status_code=200, json=lambda: {"skills": [{"name": "other"}]}
        )
        assert self._source()._index_entry("https://example.com/.well-known/skills/index.json", "missing") is None

    @patch("tools.skills_hub._guarded_http_get")
    def test_fetch_text_returns_none_on_non_200(self, mock_get):
        mock_get.return_value = MagicMock(status_code=404)
        assert self._source()._fetch_text("https://example.com/a/SKILL.md") is None

    def test_wrap_identifier(self):
        assert self._source()._wrap_identifier("https://example.com/.well-known/skills", "git-workflow") == (
            "well-known:https://example.com/.well-known/skills/git-workflow"
        )

    # -- _parse_identifier --------------------------------------------------

    @pytest.mark.parametrize(
        "identifier, expected",
        [
            # Non-HTTP raw -> None.
            ("well-known:git-workflow", None),
            # index.json without fragment -> None.
            ("https://example.com/.well-known/skills/index.json", None),
            # index.json with fragment -> parsed.
            (
                "well-known:https://example.com/.well-known/skills/index.json#my-skill",
                {
                    "index_url": "https://example.com/.well-known/skills/index.json",
                    "base_url": "https://example.com/.well-known/skills",
                    "skill_name": "my-skill",
                    "skill_url": "https://example.com/.well-known/skills/my-skill",
                },
            ),
            # Bare path without the well-known base -> None.
            ("https://example.com/SKILL.md", None),
            # Bare path within base -> parsed.
            (
                "https://example.com/.well-known/skills/my-skill",
                {
                    "index_url": "https://example.com/.well-known/skills/index.json",
                    "base_url": "https://example.com/.well-known/skills",
                    "skill_name": "my-skill",
                    "skill_url": "https://example.com/.well-known/skills/my-skill",
                },
            ),
            # Full SKILL.md URL -> base stripped, must contain base path.
            (
                "https://example.com/.well-known/skills/my-skill/SKILL.md",
                {
                    "index_url": "https://example.com/.well-known/skills/index.json",
                    "base_url": "https://example.com/.well-known/skills",
                    "skill_name": "my-skill",
                    "skill_url": "https://example.com/.well-known/skills/my-skill",
                },
            ),
        ],
    )
    def test_parse_identifier(self, identifier, expected):
        assert self._source()._parse_identifier(identifier) == expected


# ---------------------------------------------------------------------------
# UrlSource edge branches
# ---------------------------------------------------------------------------


class TestUrlSourceEdge:
    @pytest.fixture(autouse=True)
    def _neutralize_http_guards(self, monkeypatch):
        monkeypatch.setattr("tools.skills_hub.is_safe_url", lambda _url: True)
        monkeypatch.setattr("tools.skills_hub.check_website_access", lambda _url: None)

    def _source(self):
        return UrlSource()

    # -- _matches -----------------------------------------------------------

    def test_matches_rejects_non_string(self):
        assert self._source()._matches(None) is False
        assert self._source()._matches(123) is False

    def test_matches_rejects_non_http(self):
        assert self._source()._matches("ftp://example.com/SKILL.md") is False
        assert self._source()._matches("git-workflow") is False

    def test_matches_rejects_url_without_md_path(self):
        assert self._source()._matches("https://example.com/index.html") is False
        assert self._source()._matches("https://example.com/") is False

    # -- _fetch_text / _fetch_bytes -----------------------------------------

    @patch("tools.skills_hub._guarded_http_get")
    def test_fetch_text_non_200_returns_none(self, mock_get):
        mock_get.return_value = MagicMock(status_code=404)
        assert self._source()._fetch_text("https://example.com/a/SKILL.md") is None

    @patch("tools.skills_hub._guarded_http_get")
    def test_fetch_bytes_non_200_returns_none(self, mock_get):
        mock_get.return_value = MagicMock(status_code=500)
        assert self._source()._fetch_bytes("https://example.com/a/x.md") is None

    @patch("tools.skills_hub._guarded_http_get")
    def test_fetch_bytes_returns_content(self, mock_get):
        mock_get.return_value = MagicMock(status_code=200, content=b"bin")
        assert self._source()._fetch_bytes("https://example.com/a/x.md") == b"bin"

    # -- inspect ------------------------------------------------------------

    def test_inspect_returns_none_for_non_matching_identifier(self):
        assert self._source().inspect("https://example.com/index.html") is None
        assert self._source().inspect("not-a-url") is None

    @patch("tools.skills_hub._guarded_http_get")
    def test_inspect_reads_tags_from_hermes_metadata(self, mock_get):
        skill_md = (
            "---\n"
            "name: my-skill\n"
            "description: Has tags.\n"
            "metadata:\n"
            "  hermes:\n"
            "    tags: [cli, git]\n"
            "---\n\n"
            "Body\n"
        )
        mock_get.return_value = MagicMock(status_code=200, text=skill_md)
        meta = self._source().inspect("https://example.com/my-skill/SKILL.md")
        assert meta is not None
        assert meta.name == "my-skill"
        assert meta.tags == ["cli", "git"]
        assert meta.extra["awaiting_name"] is False

    # -- fetch --------------------------------------------------------------

    @patch("tools.skills_hub._guarded_http_get")
    def test_fetch_returns_none_when_not_matching(self, mock_get):
        assert self._source().fetch("https://example.com/index.html") is None
        mock_get.assert_not_called()

    @patch("tools.skills_hub._guarded_http_get")
    def test_fetch_no_usable_name_yields_awaiting_name_bundle(self, mock_get):
        # No frontmatter name; URL slug "SKILL" is a sentinel -> name None.
        mock_get.return_value = MagicMock(status_code=200, text="No frontmatter body.\n")
        bundle = self._source().fetch("https://example.com/SKILL.md")
        assert bundle is not None
        assert bundle.name == ""
        assert bundle.metadata["awaiting_name"] is True
        assert bundle.source == "url"

    # -- _resolve_skill_name ------------------------------------------------

    def test_resolve_skill_name_prefers_valid_frontmatter(self):
        assert UrlSource._resolve_skill_name({"name": "  my-skill  "}, "https://example.com/SKILL.md") == "my-skill"

    def test_resolve_skill_name_uses_url_slug_from_skill_md_parent(self):
        # Invalid frontmatter name + valid parent slug.
        assert UrlSource._resolve_skill_name({"name": "bad name"}, "https://example.com/git-workflow/SKILL.md") == "git-workflow"

    def test_resolve_skill_name_uses_url_slug_from_bare_md(self):
        assert UrlSource._resolve_skill_name({}, "https://example.com/my-skill.md") == "my-skill"

    def test_resolve_skill_name_returns_none_when_nothing_usable(self):
        assert UrlSource._resolve_skill_name({}, "https://example.com/SKILL.md") is None
        assert UrlSource._resolve_skill_name({}, "https://example.com/") is None

    def test_resolve_skill_name_ignores_non_string_frontmatter_name(self):
        # Non-string fm name falls through to URL slug.
        assert UrlSource._resolve_skill_name({"name": 123}, "https://example.com/git-workflow/SKILL.md") == "git-workflow"


# ---------------------------------------------------------------------------
# OptionalSkillSource edge branches
# ---------------------------------------------------------------------------


class TestOptionalSkillSourceEdge:
    def _make_source(self, tmp_path, remote_dirs=None):
        optional_root = tmp_path / "optional-skills"
        optional_root.mkdir(exist_ok=True)
        src = OptionalSkillSource()
        src._optional_dir = optional_root
        if remote_dirs is not None:
            src._remote_dirs = dict.fromkeys(remote_dirs, True)
        return src

    @staticmethod
    def _fake_github(tree_entries=(), contents=None):
        fake = MagicMock()
        fake._get_repo_tree.return_value = ("main", list(tree_entries))
        if contents is not None:
            fake._fetch_file_bytes.side_effect = lambda repo, path: contents.get(path)
        return fake

    # -- _list_remote_skill_dirs -------------------------------------------

    def test_list_remote_skill_dirs_uses_memory_cache(self, tmp_path):
        src = self._make_source(tmp_path, ["a/b"])
        src._github = MagicMock()
        assert src._list_remote_skill_dirs() == {"a/b": True}
        src._get_github()._get_repo_tree.assert_not_called()

    def test_list_remote_skill_dirs_index_cache_hit(self, tmp_path):
        src = self._make_source(tmp_path)  # _remote_dirs is None
        src._github = MagicMock()
        with patch("tools.skills_hub._read_index_cache", return_value={"cat/skill": True}):
            dirs = src._list_remote_skill_dirs()
        assert dirs == {"cat/skill": True}
        src._get_github()._get_repo_tree.assert_not_called()

    def test_list_remote_skill_dirs_offline_returns_empty(self, tmp_path):
        src = self._make_source(tmp_path)
        src._github = self._fake_github()  # tree -> None by default? set explicitly
        src._github._get_repo_tree.return_value = None
        with patch("tools.skills_hub._read_index_cache", return_value=None):
            assert src._list_remote_skill_dirs() == {}

    def test_list_remote_skill_dirs_writes_cache_from_tree(self, tmp_path):
        src = self._make_source(tmp_path)
        entries = [
            {"type": "blob", "path": "optional-skills/cat/skill/SKILL.md", "mode": "100644"},
        ]
        src._github = self._fake_github(entries)
        with patch("tools.skills_hub._read_index_cache", return_value=None), \
             patch("tools.skills_hub._write_index_cache") as write_cache:
            dirs = src._list_remote_skill_dirs()
        assert dirs == {"cat/skill": True}
        write_cache.assert_called_once()

    def test_list_remote_skill_dirs_skips_non_blob_and_non_skill_md(self, tmp_path):
        src = self._make_source(tmp_path)
        entries = [
            {"type": "tree", "path": "optional-skills/cat", "mode": "040000"},  # dir
            {"type": "blob", "path": "optional-skills/cat/skill/README.md", "mode": "100644"},
        ]
        src._github = self._fake_github(entries)
        with patch("tools.skills_hub._read_index_cache", return_value=None):
            assert src._list_remote_skill_dirs() == {}

    # -- upstream pointers --------------------------------------------------

    def test_upstream_pointer_returns_none_on_missing_skill_md(self, tmp_path):
        src = self._make_source(tmp_path)
        empty_dir = src._optional_dir / "empty"
        empty_dir.mkdir()
        assert src._upstream_pointer(empty_dir) is None

    def test_upstream_pointer_from_content_bytes_decode_error(self):
        src = OptionalSkillSource()
        # Invalid UTF-8 bytes -> None.
        assert src._upstream_pointer_from_content(b"\xff\xfe\x00") is None

    @pytest.mark.parametrize(
        "content, expected",
        [
            # No metadata dict.
            ("---\nname: x\n---\nBody", None),
            # metadata exists but no hermes.
            ("---\nname: x\nmetadata: {}\n---\nBody", None),
            # hermes exists but no upstream.
            ("---\nname: x\nmetadata:\n  hermes: {}\n---\nBody", None),
            # upstream not a dict.
            ("---\nname: x\nmetadata:\n  hermes:\n    upstream: nope\n---\nBody", None),
            # repo missing a slash.
            (
                "---\nmetadata:\n  hermes:\n    upstream:\n      repo: owner\n      path: a/b\n---\nBody",
                None,
            ),
            # path missing.
            (
                "---\nmetadata:\n  hermes:\n    upstream:\n      repo: owner/repo\n      path: ''\n---\nBody",
                None,
            ),
            # path traversal.
            (
                "---\nmetadata:\n  hermes:\n    upstream:\n      repo: owner/repo\n      path: a/../b\n---\nBody",
                None,
            ),
            # valid.
            (
                "---\nmetadata:\n  hermes:\n    upstream:\n      repo: pbakaus/impeccable\n      path: .hermes/skills/impeccable\n---\nBody",
                {"repo": "pbakaus/impeccable", "path": ".hermes/skills/impeccable"},
            ),
        ],
    )
    def test_upstream_pointer_from_content(self, content, expected):
        src = OptionalSkillSource()
        assert src._upstream_pointer_from_content(content) == expected

    # -- _fetch_from_upstream ----------------------------------------------

    def test_fetch_from_upstream_relabels_bundle(self, tmp_path):
        src = self._make_source(tmp_path)
        inner = MagicMock()
        inner.name = "impeccable"
        inner.files = {"SKILL.md": "body"}
        inner.metadata = {}
        inner.source = "github"
        src._github = MagicMock()
        src._github.fetch.return_value = inner

        bundle = src._fetch_from_upstream(
            {"repo": "pbakaus/impeccable", "path": "hermes/skills/impeccable"}, "cat/impeccable"
        )
        assert bundle is not None
        assert bundle.source == "official"
        assert bundle.identifier == "official/cat/impeccable"
        assert bundle.trust_level == "trusted"
        assert bundle.metadata["upstream_repo"] == "pbakaus/impeccable"

    def test_fetch_from_upstream_returns_none_when_inner_fetch_fails(self, tmp_path):
        src = self._make_source(tmp_path)
        src._github = MagicMock()
        src._github.fetch.return_value = None
        assert src._fetch_from_upstream({"repo": "o/r", "path": "p"}, "cat/skill") is None

    # -- fetch routing (local upstream stub -> _fetch_from_upstream) --------

    def test_fetch_routes_upstream_stub_to_external_repo(self, tmp_path):
        src = self._make_source(tmp_path)
        skill_dir = src._optional_dir / "cat" / "impeccable"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(
            "---\nname: impeccable\nmetadata:\n  hermes:\n    upstream:\n      repo: pbakaus/impeccable\n"
            "      path: .hermes/skills/impeccable\n---\nBody\n",
            encoding="utf-8",
        )
        inner = MagicMock()
        inner.name = "impeccable"
        inner.files = {"SKILL.md": "body"}
        inner.metadata = {}
        src._github = MagicMock()
        src._github.fetch.return_value = inner

        bundle = src.fetch("official/cat/impeccable")
        assert bundle is not None
        assert bundle.identifier == "official/cat/impeccable"
        assert bundle.trust_level == "trusted"
        src._get_github().fetch.assert_called_once()

    # -- live-repo fallback ------------------------------------------------

    @staticmethod
    def _entry(rel_dir, extra=()):
        entries = [{"type": "blob", "path": f"optional-skills/{rel_dir}/SKILL.md", "mode": "100644"}]
        for p in extra:
            entries.append({"type": "blob", "path": p, "mode": "100644"})
        return entries

    def test_fetch_from_live_repo_none_when_content_fetch_fails(self, tmp_path):
        src = self._make_source(tmp_path, ["cat/skill"])
        entries = self._entry("cat/skill", ["optional-skills/cat/skill/LICENSE"])
        src._github = self._fake_github(entries, contents={})  # fetch_file_bytes -> None
        assert src._fetch_from_live_repo("cat/skill") is None

    def test_fetch_from_live_repo_none_when_skill_md_absent(self, tmp_path):
        src = self._make_source(tmp_path, ["cat/skill"])
        # Tree contains only a non-SKILL.md blob under the dir.
        entries = [{"type": "blob", "path": "optional-skills/cat/skill/LICENSE", "mode": "100644"}]
        contents = {"optional-skills/cat/skill/LICENSE": b"MIT"}
        src._github = self._fake_github(entries, contents)
        assert src._fetch_from_live_repo("cat/skill") is None

    def test_fetch_from_live_repo_routes_upstream_pointer(self, tmp_path):
        src = self._make_source(tmp_path, ["cat/impeccable"])
        content = (
            b"---\nname: impeccable\nmetadata:\n  hermes:\n    upstream:\n"
            b"      repo: pbakaus/impeccable\n      path: .hermes/skills/impeccable\n---\nBody\n"
        )
        entries = self._entry("cat/impeccable")
        contents = {"optional-skills/cat/impeccable/SKILL.md": content}
        src._github = self._fake_github(entries, contents)
        inner = MagicMock()
        inner.name = "impeccable"
        inner.files = {"SKILL.md": "body"}
        inner.metadata = {}
        src._github.fetch.return_value = inner

        bundle = src._fetch_from_live_repo("cat/impeccable")
        assert bundle is not None
        assert bundle.identifier == "official/cat/impeccable"
        assert bundle.trust_level == "trusted"

    def test_fetch_from_live_repo_empty_rel_returns_none(self, tmp_path):
        src = self._make_source(tmp_path, ["cat/skill"])
        src._github = MagicMock()
        assert src._fetch_from_live_repo("") is None
        assert src._fetch_from_live_repo("///") is None

    def test_fetch_from_live_repo_normalizes_dot_segments(self, tmp_path):
        # "cat/./skill" -> parts filters "." -> "cat/skill" matches remote dir.
        src = self._make_source(tmp_path, ["cat/skill"])
        entries = self._entry("cat/skill")
        contents = {"optional-skills/cat/skill/SKILL.md": b"---\nname: skill\n---\nBody"}
        src._github = self._fake_github(entries, contents)
        bundle = src._fetch_from_live_repo("cat/./skill")
        assert bundle is not None
        assert bundle.identifier == "official/cat/skill"

    # -- _scan_all / _parse_frontmatter ---------------------------------------

    def test_scan_all_skips_skill_with_unreadable_frontmatter(self, tmp_path):
        src = self._make_source(tmp_path)
        skill_dir = src._optional_dir / "research" / "bad-skill"
        skill_dir.mkdir(parents=True)
        # Invalid UTF-8 -> UnicodeDecodeError -> skipped by _scan_all.
        (skill_dir / "SKILL.md").write_bytes(b"\xff\xfe\x00\x01")
        assert src._scan_all() == []

    def test_parse_frontmatter_tolerates_bom(self):
        src = OptionalSkillSource()
        assert src._parse_frontmatter("\ufeff---\nname: x\n---\nBody") == {"name": "x"}

    def test_parse_frontmatter_empty_for_no_frontmatter(self):
        src = OptionalSkillSource()
        assert src._parse_frontmatter("Just a body text.") == {}

    def test_parse_frontmatter_empty_for_open_frontmatter(self):
        src = OptionalSkillSource()
        assert src._parse_frontmatter("---\nname: x\n") == {}

    def test_parse_frontmatter_empty_for_yaml_error(self):
        src = OptionalSkillSource()
        assert src._parse_frontmatter("---\nname: [\n---\n") == {}
