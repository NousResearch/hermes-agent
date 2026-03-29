"""Behaviour contracts for Agent Skills well-known discovery."""

from unittest.mock import MagicMock, patch

import pytest

from tools.skills_hub_sources import WellKnownSkillSource


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
    @patch("tools.skills_hub._ssrf_safe_http_get")
    def test_search_domain_root_prefers_v2_path(self, mock_get, _mock_read_cache, _mock_write_cache):
        mock_get.return_value = MagicMock(
            status_code=200,
            json=lambda: {
                "$schema": "https://schemas.agentskills.io/discovery/0.2.0/schema.json",
                "skills": [{
                    "name": "agent",
                    "type": "skill-md",
                    "description": "Agent skill",
                    "url": "/published/agent-v2.md",
                }],
            },
        )

        results = self._source().search("https://example.com", limit=10)

        assert [result.identifier for result in results] == [
            "well-known:https://example.com/.well-known/agent-skills/agent"
        ]
        assert mock_get.call_args_list[0].args[0] == (
            "https://example.com/.well-known/agent-skills/index.json"
        )
        assert mock_get.call_count == 1

    @patch("tools.skills_hub._write_index_cache")
    @patch("tools.skills_hub._read_index_cache", return_value=None)
    @patch("tools.skills_hub._ssrf_safe_http_get")
    def test_search_domain_root_falls_back_to_v1(self, mock_get, _mock_read_cache, _mock_write_cache):
        def fake_get(url, **_kwargs):
            if "/.well-known/agent-skills/" in url:
                return MagicMock(status_code=404)
            return MagicMock(
                status_code=200,
                json=lambda: {
                    "skills": [{
                        "name": "git-workflow",
                        "description": "Git rules",
                        "files": ["SKILL.md"],
                    }]
                },
            )

        mock_get.side_effect = fake_get

        results = self._source().search("https://example.com", limit=10)

        assert [result.identifier for result in results] == [
            "well-known:https://example.com/.well-known/skills/git-workflow"
        ]
        assert [call.args[0] for call in mock_get.call_args_list] == [
            "https://example.com/.well-known/agent-skills/index.json",
            "https://example.com/.well-known/skills/index.json",
        ]

    @patch("tools.skills_hub._write_index_cache")
    @patch("tools.skills_hub._read_index_cache", return_value=None)
    @patch("tools.skills_hub._ssrf_safe_http_get")
    def test_search_domain_root_does_not_hide_v2_server_errors(self, mock_get, _mock_read_cache, _mock_write_cache):
        mock_get.return_value = MagicMock(status_code=500)

        results = self._source().search("https://example.com", limit=10)

        assert results == []
        assert mock_get.call_count == 1

    @patch("tools.skills_hub._write_index_cache")
    @patch("tools.skills_hub._read_index_cache", return_value=None)
    @patch("tools.skills_hub._ssrf_safe_http_get")
    def test_inspect_v2_uses_advertised_nonlegacy_url(self, mock_get, _mock_read_cache, _mock_write_cache):
        def fake_get(url, **_kwargs):
            if url.endswith("/.well-known/agent-skills/index.json"):
                return MagicMock(
                    status_code=200,
                    json=lambda: {
                        "skills": [{
                            "name": "agent",
                            "type": "skill-md",
                            "description": "Index description",
                            "url": "/published/skills/agent-v2.md",
                        }]
                    },
                )
            if url == "https://example.com/published/skills/agent-v2.md":
                return MagicMock(
                    status_code=200,
                    text="---\nname: agent\ndescription: Agent skill\n---\n\n# Agent\n",
                )
            raise AssertionError(f"unexpected URL: {url}")

        mock_get.side_effect = fake_get

        meta = self._source().inspect(
            "well-known:https://example.com/.well-known/agent-skills/agent"
        )

        assert meta is not None
        assert meta.name == "agent"
        assert meta.description == "Agent skill"
        assert meta.extra["endpoint"] == "https://example.com/published/skills/agent-v2.md"
        assert mock_get.call_args_list[-1].args[0] == "https://example.com/published/skills/agent-v2.md"

    @patch("tools.skills_hub._write_index_cache")
    @patch("tools.skills_hub._read_index_cache", return_value=None)
    @patch("tools.skills_hub._ssrf_safe_http_get")
    def test_fetch_v2_uses_advertised_nonlegacy_filename(self, mock_get, _mock_read_cache, _mock_write_cache):
        skill_file_url = "https://cdn.example.com/releases/agent-2026-11.md"

        def fake_get(url, **_kwargs):
            if url.endswith("/.well-known/agent-skills/index.json"):
                return MagicMock(
                    status_code=200,
                    json=lambda: {
                        "skills": [{
                            "name": "agent",
                            "type": "skill-md",
                            "description": "Agent skill",
                            "url": skill_file_url,
                        }]
                    },
                )
            if url == skill_file_url:
                return MagicMock(status_code=200, text="# Agent\n")
            raise AssertionError(f"unexpected URL: {url}")

        mock_get.side_effect = fake_get

        bundle = self._source().fetch(
            "well-known:https://example.com/.well-known/agent-skills/agent"
        )

        assert bundle is not None
        assert bundle.files == {"SKILL.md": "# Agent\n"}
        assert bundle.metadata["endpoint"] == skill_file_url
        assert mock_get.call_args_list[-1].args[0] == skill_file_url

    @patch("tools.skills_hub._write_index_cache")
    @patch("tools.skills_hub._read_index_cache", return_value=None)
    @patch("tools.skills_hub._ssrf_safe_http_get")
    def test_search_v2_skips_unknown_entry_types(self, mock_get, _mock_read_cache, _mock_write_cache):
        mock_get.return_value = MagicMock(
            status_code=200,
            json=lambda: {
                "skills": [
                    {"name": "future", "type": "future-format", "url": "/future.bin"},
                    {"name": "agent", "type": "skill-md", "url": "/agent.md"},
                ]
            },
        )

        results = self._source().search(
            "https://example.com/.well-known/agent-skills/index.json", limit=10
        )

        assert [result.name for result in results] == ["agent"]

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
