"""WellKnownSkillSource: reading skills from a domain's well-known Agent Skills index."""

from unittest.mock import MagicMock, patch

import pytest

from tools.skills_hub_sources import WellKnownSkillSource


@pytest.fixture
def allow_public_skill_fetches(monkeypatch):
    monkeypatch.setattr("tools.skills_hub.is_safe_url", lambda _url: True)
    monkeypatch.setattr("tools.skills_hub.check_website_access", lambda _url: None)


@pytest.mark.usefixtures("allow_public_skill_fetches")
class TestWellKnownSkillSource:
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
    @patch("tools.skills_hub._ssrf_safe_http_get")
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

    @patch("tools.skills_hub._write_index_cache")
    @patch("tools.skills_hub._read_index_cache", return_value=None)
    @patch("tools.skills_hub._ssrf_safe_http_get")
    def test_identifier_recorded_for_index_outside_base_path_fetches_again(self, mock_get, _mock_read_cache, _mock_write_cache):
        """Regression for #159541: the identifier an install records for an index at another path
        (``/.well-known/agent-skills/``) must fetch again, or check/update report it unavailable."""
        def fake_get(url, *args, **kwargs):
            if url == "https://example.com/.well-known/agent-skills/index.json":
                return MagicMock(status_code=200, json=lambda: {"skills": [{"name": "demo", "description": "d"}]})
            if url == "https://example.com/.well-known/agent-skills/demo/SKILL.md":
                return MagicMock(status_code=200, text="---\nname: demo\n---\n# Demo\n")
            raise AssertionError(url)

        mock_get.side_effect = fake_get
        installed = self._source().fetch("well-known:https://example.com/.well-known/agent-skills/index.json#demo")

        refetched = self._source().fetch(installed.identifier)

        assert refetched is not None and refetched.files == installed.files
