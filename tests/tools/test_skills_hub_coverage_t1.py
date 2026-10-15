"""Coverage tests for SkillsShSource detail-page parsing + identifier helpers.

Focus ranges (tools/skills_hub.py):
  _meta_from_search_item      (2036-2070)
  _fetch_detail_page          (2072-2088)
  _parse_detail_page          (2090-2125)
  _extract_repo_slug          (2269-2277)
  _extract_first_match        (2280-2287)
  _detail_to_metadata         (2289-2302)
  _extract_weekly_installs    (2304-2309)
  _extract_security_audits    (2311-2322)
  _strip_html                 (2324-2326)
  _normalize_identifier       (2328-2339)
  _candidate_identifiers      (2341-2362)
  _wrap_identifier            (2364-2366)

Deterministic and offline: every network touch is patched (httpx.get,
_read_index_cache / _write_index_cache), guard branches are exercised, and no
hard dependency is placed on the external skills.sh HTML or git remote.
"""

import re
from unittest.mock import MagicMock, patch

import httpx

from tools.skills_hub import GitHubAuth, SkillsShSource


class _MockResponse:
    """Minimal httpx.Response stand-in (status_code + text)."""

    def __init__(self, status_code=200, text=""):
        self.status_code = status_code
        self.text = text


def _detail_html(*, install=True, page_h1=True, prose=True, weekly=True,
                 security=True):
    """Build a skills.sh-style detail HTML page that drives every parse regex.

    The ``weekly`` block matches ``_WEEKLY_INSTALLS_RE`` exactly: the pattern
    expects the literal text ``children\\":\\"<count>\\"`` (two literal
    backslashes before each quote), which is what a JS-embedded JSON dump looks
    like in the real page.
    """
    parts = ["<html><body>"]
    if page_h1:
        parts.append("<h1>Page Title</h1>")
    if prose:
        parts.append(
            '<div class="prose"><h1>Body Heading</h1><p>Summary text.</p></div>'
        )
    if install:
        parts.append(
            "npx skills add https://github.com/Owner/RepoA --skill my-skill"
        )
    if weekly:
        # Standard JS-embedded JSON escapes: children\":\"1.2k\"
        parts.append('Weekly Installs ... children\\\":\\\"1.2k\\\"')
    if security:
        parts.append(
            "/security/agent-trust-hub Pass "
            "/security/socket Warn /security/snyk Fail"
        )
    parts.append("</body></html>")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# _meta_from_search_item
# ---------------------------------------------------------------------------


class TestMetaFromSearchItem:
    def _source(self):
        src = SkillsShSource(auth=MagicMock(spec=GitHubAuth))
        src.github.trust_level_for = MagicMock(return_value="trusted")
        return src

    def test_non_dict_item_returns_none(self):
        assert self._source()._meta_from_search_item("not-a-dict") is None
        assert self._source()._meta_from_search_item(None) is None

    def test_empty_item_returns_none(self):
        assert self._source()._meta_from_search_item({}) is None

    def test_valid_item_maps_to_skill_meta(self):
        meta = self._source()._meta_from_search_item(
            {
                "id": "vercel-labs/agent-skills/vercel-react-best-practices",
                "source": "vercel-labs/agent-skills",
                "skillId": "vercel-react-best-practices",
                "name": "Vercel React Best Practices",
                "installs": 123,
            }
        )
        assert isinstance(meta.name, str)
        assert meta.name == "Vercel React Best Practices"
        assert meta.repo == "vercel-labs/agent-skills"
        assert meta.path == "vercel-react-best-practices"
        assert meta.source == "skills.sh"
        assert meta.identifier == (
            "skills-sh/vercel-labs/agent-skills/vercel-react-best-practices"
        )
        assert meta.trust_level == "trusted"
        assert "123 installs" in meta.description
        assert "vercel-labs/agent-skills" in meta.description
        assert meta.extra["installs"] == 123
        assert meta.extra["detail_url"] == (
            "https://skills.sh/vercel-labs/agent-skills/vercel-react-best-practices"
        )
        assert meta.extra["repo_url"] == (
            "https://github.com/vercel-labs/agent-skills"
        )

    def test_invalid_canonical_rebuilds_from_repo_and_skill_id(self):
        # id has fewer than 2 slashes, so the canonical is rebuilt.
        meta = self._source()._meta_from_search_item(
            {
                "id": "vercel-labs/agent-skills",
                "source": "vercel-labs/agent-skills",
                "skillId": "my-skill",
                "name": "My Skill",
                "installs": 7,
            }
        )
        assert meta is not None
        assert meta.identifier == "skills-sh/vercel-labs/agent-skills/my-skill"
        assert meta.repo == "vercel-labs/agent-skills"

    def test_rebuild_fails_when_source_or_skill_id_missing(self):
        assert self._source()._meta_from_search_item(
            {"id": "owner/repo", "source": "owner/repo"}
        ) is None
        assert self._source()._meta_from_search_item(
            {"id": "owner/repo", "skillId": "skill"}
        ) is None

    def test_rebuilt_canonical_with_under_three_parts_returns_none(self):
        # rebuilt canonical is "x/y" -> only two slash-parts.
        assert self._source()._meta_from_search_item(
            {"id": "owner/repo", "source": "x", "skillId": "y"}
        ) is None

    def test_non_int_installs_produces_no_label(self):
        meta = self._source()._meta_from_search_item(
            {
                "id": "owner/repo/skill",
                "source": "owner/repo",
                "skillId": "skill",
                "installs": "many",
            }
        )
        assert meta.extra["installs"] == "many"
        assert "· " not in meta.description
        assert meta.description == "Indexed by skills.sh from owner/repo"

    def test_missing_name_falls_back_to_skill_token(self):
        meta = self._source()._meta_from_search_item(
            {"id": "owner/repo/my-skill", "source": "owner/repo", "skillId": "my-skill"}
        )
        assert meta.name == "my-skill"


# ---------------------------------------------------------------------------
# _fetch_detail_page
# ---------------------------------------------------------------------------


class TestFetchDetailPage:
    def _source(self):
        return SkillsShSource(auth=MagicMock(spec=GitHubAuth))

    @patch("tools.skills_hub.httpx.get")
    @patch("tools.skills_hub._write_index_cache")
    @patch("tools.skills_hub._read_index_cache", return_value={"cached": True})
    def test_cache_hit_skips_network(self, _mock_read, mock_write, mock_get):
        detail = self._source()._fetch_detail_page("owner/repo/skill")
        assert detail == {"cached": True}
        mock_get.assert_not_called()
        mock_write.assert_not_called()

    @patch("tools.skills_hub.httpx.get")
    @patch("tools.skills_hub._write_index_cache")
    @patch("tools.skills_hub._read_index_cache", return_value=None)
    def test_cache_miss_network_ok_writes_cache(self, _mock_read, mock_write, mock_get):
        mock_get.return_value = _MockResponse(status_code=200, text=_detail_html())
        detail = self._source()._fetch_detail_page("Owner/RepoA/my-skill")
        assert detail is not None
        assert detail["repo"] == "Owner/RepoA"
        mock_write.assert_called_once()
        args = mock_write.call_args[0]
        self_write_key, written = args
        assert isinstance(self_write_key, str) and self_write_key.startswith("skills_sh_detail_")
        assert written == detail

    @patch("tools.skills_hub.httpx.get")
    @patch("tools.skills_hub._write_index_cache")
    @patch("tools.skills_hub._read_index_cache", return_value=None)
    def test_non_200_returns_none_without_write(self, _mock_read, mock_write, mock_get):
        mock_get.return_value = _MockResponse(status_code=404, text=_detail_html())
        assert self._source()._fetch_detail_page("Owner/RepoA/my-skill") is None
        mock_write.assert_not_called()

    @patch("tools.skills_hub.httpx.get", side_effect=httpx.HTTPError("boom"))
    @patch("tools.skills_hub._write_index_cache")
    @patch("tools.skills_hub._read_index_cache", return_value=None)
    def test_http_error_returns_none(self, _mock_read, mock_write, mock_get):
        assert self._source()._fetch_detail_page("owner/repo/skill") is None
        mock_write.assert_not_called()

    @patch("tools.skills_hub.httpx.get")
    @patch("tools.skills_hub._write_index_cache")
    @patch("tools.skills_hub._read_index_cache", return_value=None)
    def test_unparseable_identifier_returns_none_without_write(
        self, _mock_read, mock_write, mock_get
    ):
        mock_get.return_value = _MockResponse(status_code=200, text=_detail_html())
        # "owner/repo" has only two slash-parts -> _parse_detail_page returns None.
        assert self._source()._fetch_detail_page("owner/repo") is None
        mock_write.assert_not_called()


# ---------------------------------------------------------------------------
# _parse_detail_page
# ---------------------------------------------------------------------------


class TestParseDetailPage:
    def _source(self):
        return SkillsShSource(auth=MagicMock(spec=GitHubAuth))

    def test_identifier_with_under_three_parts_returns_none(self):
        assert self._source()._parse_detail_page("owner/repo", "<html/>") is None

    def test_full_page_parses_every_field(self):
        detail = self._source()._parse_detail_page(
            "Owner/RepoA/my-skill", _detail_html()
        )
        assert detail["repo"] == "Owner/RepoA"
        assert detail["install_skill"] == "my-skill"
        assert detail["page_title"] == "Page Title"
        assert detail["body_title"] == "Body Heading"
        assert detail["body_summary"] == "Summary text."
        assert detail["weekly_installs"] == "1.2k"
        assert detail["install_command"] == (
            "npx skills add https://github.com/Owner/RepoA --skill my-skill"
        )
        assert detail["repo_url"] == "https://github.com/Owner/RepoA"
        assert detail["detail_url"] == "https://skills.sh/Owner/RepoA/my-skill"
        assert detail["security_audits"] == {
            "agent-trust-hub": "Pass",
            "socket": "Warn",
            "snyk": "Fail",
        }

    def test_page_without_any_matches_returns_defaults(self):
        detail = self._source()._parse_detail_page("Owner/RepoA/my-skill", "<html></html>")
        assert detail["repo"] == "Owner/RepoA"
        assert detail["install_skill"] == "my-skill"
        assert detail["page_title"] is None
        assert detail["body_title"] is None
        assert detail["body_summary"] is None
        assert detail["weekly_installs"] is None
        assert detail["install_command"] is None
        assert detail["security_audits"] == {}

    def test_install_command_absent_keeps_default_repo_and_skill(self):
        detail = self._source()._parse_detail_page(
            "Owner/RepoA/my-skill", _detail_html(install=False, page_h1=False,
                                                prose=False, weekly=False, security=False)
        )
        assert detail["install_command"] is None
        assert detail["repo"] == "Owner/RepoA"


# ---------------------------------------------------------------------------
# _extract_repo_slug
# ---------------------------------------------------------------------------


class TestExtractRepoSlug:
    def test_strips_github_url_prefix(self):
        assert SkillsShSource._extract_repo_slug("https://github.com/owner/repo") == "owner/repo"

    def test_passthrough_bare_repo(self):
        assert SkillsShSource._extract_repo_slug("owner/repo") == "owner/repo"

    def test_deep_path_returns_first_two_segments(self):
        assert SkillsShSource._extract_repo_slug("owner/repo/extra") == "owner/repo"

    def test_trailing_slash_and_extra_segments_stripped(self):
        assert (
            SkillsShSource._extract_repo_slug("https://github.com/owner/repo/extra/")
            == "owner/repo"
        )

    def test_single_segment_returns_none(self):
        assert SkillsShSource._extract_repo_slug("owner") is None
        assert SkillsShSource._extract_repo_slug("") is None
        assert SkillsShSource._extract_repo_slug("https://github.com/") is None

    def test_surrounding_whitespace_stripped(self):
        assert SkillsShSource._extract_repo_slug("  https://github.com/owner/repo  ") == "owner/repo"


# ---------------------------------------------------------------------------
# _extract_first_match
# ---------------------------------------------------------------------------


class TestExtractFirstMatch:
    def test_returns_first_captured_group(self):
        pat = re.compile(r"<h1[^>]*>(?P<title>.*?)</h1>", re.IGNORECASE | re.DOTALL)
        assert SkillsShSource._extract_first_match(pat, "<h1>Hello</h1>") == "Hello"

    def test_no_match_returns_none(self):
        pat = re.compile(r"(?P<h>zzz)")
        assert SkillsShSource._extract_first_match(pat, "abc") is None

    def test_empty_group_returns_none(self):
        pat = re.compile(r"(?P<h>a*)")
        assert SkillsShSource._extract_first_match(pat, "b") is None

    def test_html_in_group_is_stripped(self):
        pat = re.compile(r"(?P<h><b>bold</b>)")
        assert SkillsShSource._extract_first_match(pat, "<b>bold</b>") == "bold"


# ---------------------------------------------------------------------------
# _detail_to_metadata
# ---------------------------------------------------------------------------


class TestDetailToMetadata:
    def _source(self):
        return SkillsShSource(auth=MagicMock(spec=GitHubAuth))

    def test_none_detail_builds_urls_from_canonical(self):
        metadata = self._source()._detail_to_metadata("owner/repo/skill", None)
        assert metadata["detail_url"] == "https://skills.sh/owner/repo/skill"
        assert metadata["repo_url"] == "https://github.com/owner/repo"

    def test_short_canonical_omits_repo_url(self):
        metadata = self._source()._detail_to_metadata("solo", None)
        assert metadata["detail_url"] == "https://skills.sh/solo"
        assert "repo_url" not in metadata

    def test_detail_truthy_values_copied(self):
        detail = {
            "weekly_installs": "1.2k",
            "install_command": "npx skills add https://github.com/o/r --skill s",
            "repo_url": "https://github.com/owner/repo",
            "detail_url": "https://skills.sh/owner/repo/skill",
            "security_audits": {"socket": "Warn"},
        }
        metadata = self._source()._detail_to_metadata("owner/repo/skill", detail)
        assert metadata["weekly_installs"] == "1.2k"
        assert metadata["install_command"] == detail["install_command"]
        assert metadata["repo_url"] == "https://github.com/owner/repo"
        assert metadata["detail_url"] == "https://skills.sh/owner/repo/skill"
        assert metadata["security_audits"] == {"socket": "Warn"}

    def test_empty_detail_values_not_copied(self):
        metadata = self._source()._detail_to_metadata(
            "owner/repo/skill",
            {"weekly_installs": "", "security_audits": {}, "install_command": None},
        )
        assert "weekly_installs" not in metadata
        assert "security_audits" not in metadata
        assert "install_command" not in metadata


# ---------------------------------------------------------------------------
# _extract_weekly_installs / _extract_security_audits / _strip_html
# ---------------------------------------------------------------------------


class TestExtractWeeklyInstalls:
    def test_matches_count(self):
        assert SkillsShSource._extract_weekly_installs(_detail_html()) == "1.2k"

    def test_no_match_returns_none(self):
        assert SkillsShSource._extract_weekly_installs("<html></html>") is None


class TestExtractSecurityAudits:
    def test_extracts_all_audits_with_status(self):
        html = _detail_html()
        audits = SkillsShSource._extract_security_audits(html, "Owner/RepoA/my-skill")
        assert audits == {
            "agent-trust-hub": "Pass",
            "socket": "Warn",
            "snyk": "Fail",
        }

    def test_no_security_links_returns_empty(self):
        assert SkillsShSource._extract_security_audits("<html></html>", "o/r/s") == {}

    def test_link_without_status_word_not_recorded(self):
        html = "<html>/security/socket</html>"
        assert SkillsShSource._extract_security_audits(html, "o/r/s") == {}

    def test_status_window_is_bounded_to_500_chars(self):
        # a status word far beyond 500 chars must not be picked up
        html = "/security/socket " + ("x" * 600) + " Pass"
        assert SkillsShSource._extract_security_audits(html, "o/r/s") == {}


class TestStripHtml:
    def test_strips_tags(self):
        assert SkillsShSource._strip_html("<b>bold</b><i>italic</i>") == "bolditalic"
        assert SkillsShSource._strip_html("no tags here") == "no tags here"


# ---------------------------------------------------------------------------
# _normalize_identifier / _candidate_identifiers / _wrap_identifier
# ---------------------------------------------------------------------------


class TestNormalizeIdentifier:
    def test_strips_all_prefix_aliases(self):
        assert SkillsShSource._normalize_identifier("skills-sh/o/r/s") == "o/r/s"
        assert SkillsShSource._normalize_identifier("skills.sh/o/r/s") == "o/r/s"
        assert SkillsShSource._normalize_identifier("skils-sh/o/r/s") == "o/r/s"
        assert SkillsShSource._normalize_identifier("skils.sh/o/r/s") == "o/r/s"

    def test_no_prefix_left_unchanged(self):
        assert SkillsShSource._normalize_identifier("owner/repo/skill") == "owner/repo/skill"


class TestCandidateIdentifiers:
    def test_short_identifier_returns_itself(self):
        assert SkillsShSource._candidate_identifiers("owner/repo") == ["owner/repo"]

    def test_builds_four_candidate_paths(self):
        candidates = SkillsShSource._candidate_identifiers("owner/repo/skill")
        assert candidates == [
            "owner/repo/skill",
            "owner/repo/skills/skill",
            "owner/repo/.agents/skills/skill",
            "owner/repo/.claude/skills/skill",
        ]

    def test_leading_slash_in_skill_path_removed(self):
        candidates = SkillsShSource._candidate_identifiers("owner/repo//skill")
        assert candidates[0] == "owner/repo/skill"
        assert len(candidates) == len(set(candidates))

    def test_results_are_dataclass_deduplicated(self):
        candidates = SkillsShSource._candidate_identifiers("owner/repo/skill")
        assert len(candidates) == len(set(candidates))


class TestWrapIdentifier:
    def test_prepends_skills_sh_prefix(self):
        assert SkillsShSource._wrap_identifier("owner/repo/skill") == "skills-sh/owner/repo/skill"
