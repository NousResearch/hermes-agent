"""Behaviour contracts for Agent Skills well-known discovery."""

import hashlib
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import httpx
import pytest

from tools import skills_hub
from tools.skills_hub_sources import WellKnownSkillSource


V2_SCHEMA = "https://schemas.agentskills.io/discovery/0.2.0/schema.json"


def _digest(content: bytes) -> str:
    return f"sha256:{hashlib.sha256(content).hexdigest()}"


class TestWellKnownSkillSource:
    @pytest.fixture(autouse=True)
    def _allow_public_skill_fetches(self, monkeypatch):
        monkeypatch.setattr("tools.skills_hub.is_safe_url", lambda _url: True)
        monkeypatch.setattr("tools.skills_hub.check_website_access", lambda _url: None)

    def _source(self):
        return WellKnownSkillSource()

    @patch("tools.skills_hub._write_index_cache")
    @patch("tools.skills_hub._read_index_cache", return_value=None)
    @patch("tools.skills_hub._guarded_http_get")
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
    @patch("tools.skills_hub._guarded_http_get")
    def test_search_domain_root_prefers_v2_path(self, mock_get, _mock_read_cache, _mock_write_cache):
        mock_get.return_value = MagicMock(
            status_code=200,
            json=lambda: {
                "$schema": V2_SCHEMA,
                "skills": [{
                    "name": "agent",
                    "type": "skill-md",
                    "description": "Agent skill",
                    "url": "/published/agent-v2.md",
                    "digest": "sha256:" + "0" * 64,
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
    @patch("tools.skills_hub._guarded_http_get")
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
    @patch("tools.skills_hub._guarded_http_get")
    def test_search_domain_root_does_not_hide_v2_server_errors(self, mock_get, _mock_read_cache, _mock_write_cache):
        mock_get.return_value = MagicMock(status_code=500)

        results = self._source().search("https://example.com", limit=10)

        assert results == []
        assert mock_get.call_count == 1

    @patch("tools.skills_hub._write_index_cache")
    @patch("tools.skills_hub._read_index_cache", return_value=None)
    @patch("tools.skills_hub._guarded_http_get")
    def test_inspect_v2_uses_advertised_nonlegacy_url(self, mock_get, _mock_read_cache, _mock_write_cache):
        skill_md = b"---\nname: agent\ndescription: Agent skill\n---\n\n# Agent\n"

        def fake_get(url, **_kwargs):
            if url.endswith("/.well-known/agent-skills/index.json"):
                return MagicMock(
                    status_code=200,
                    json=lambda: {
                        "$schema": V2_SCHEMA,
                        "skills": [{
                            "name": "agent",
                            "type": "skill-md",
                            "description": "Index description",
                            "url": "/published/skills/agent-v2.md",
                            "digest": _digest(skill_md),
                        }]
                    },
                )
            if url == "https://example.com/published/skills/agent-v2.md":
                return MagicMock(status_code=200, content=skill_md)
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
    @patch("tools.skills_hub._guarded_http_get")
    def test_fetch_v2_uses_advertised_nonlegacy_filename(self, mock_get, _mock_read_cache, _mock_write_cache):
        skill_file_url = "https://cdn.example.com/releases/agent-2026-11.md"
        skill_md = b"# Agent\n"

        def fake_get(url, **_kwargs):
            if url.endswith("/.well-known/agent-skills/index.json"):
                return MagicMock(
                    status_code=200,
                    json=lambda: {
                        "$schema": V2_SCHEMA,
                        "skills": [{
                            "name": "agent",
                            "type": "skill-md",
                            "description": "Agent skill",
                            "url": skill_file_url,
                            "digest": _digest(skill_md),
                        }]
                    },
                )
            if url == skill_file_url:
                return MagicMock(status_code=200, content=skill_md)
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
    @patch("tools.skills_hub._guarded_http_get")
    def test_search_v2_skips_unknown_entry_types(self, mock_get, _mock_read_cache, _mock_write_cache):
        mock_get.return_value = MagicMock(
            status_code=200,
            json=lambda: {
                "$schema": V2_SCHEMA,
                "skills": [
                    {"name": "future", "type": "future-format", "url": "/future.bin",
                     "digest": "sha256:" + "0" * 64},
                    {"name": "agent", "type": "skill-md", "url": "/agent.md",
                     "digest": "sha256:" + "0" * 64},
                ]
            },
        )

        results = self._source().search(
            "https://example.com/.well-known/agent-skills/index.json", limit=10
        )

        assert [result.name for result in results] == ["agent"]

    @patch("tools.skills_hub._write_index_cache")
    @patch("tools.skills_hub._read_index_cache", return_value=None)
    @patch("tools.skills_hub._guarded_http_get")
    def test_search_v2_rejects_unknown_schema(self, mock_get, _mock_read_cache, _mock_write_cache):
        mock_get.return_value = MagicMock(
            status_code=200,
            json=lambda: {
                "$schema": "https://schemas.agentskills.io/discovery/9.9.9/schema.json",
                "skills": [{
                    "name": "agent", "type": "skill-md", "url": "/agent.md",
                    "digest": "sha256:" + "0" * 64,
                }],
            },
        )

        assert self._source().search(
            "https://example.com/.well-known/agent-skills/index.json", limit=10
        ) == []

    @patch("tools.skills_hub._write_index_cache")
    @patch("tools.skills_hub._read_index_cache", return_value=None)
    @patch("tools.skills_hub._guarded_http_get")
    def test_search_v2_skips_entry_without_digest(self, mock_get, _mock_read_cache, _mock_write_cache):
        mock_get.return_value = MagicMock(
            status_code=200,
            json=lambda: {
                "$schema": V2_SCHEMA,
                "skills": [{"name": "agent", "type": "skill-md", "url": "/agent.md"}],
            },
        )

        assert self._source().search(
            "https://example.com/.well-known/agent-skills/index.json", limit=10
        ) == []

    @patch("tools.skills_hub._write_index_cache")
    @patch("tools.skills_hub._read_index_cache", return_value=None)
    @patch("tools.skills_hub._guarded_http_get")
    def test_fetch_v2_rejects_digest_mismatch(self, mock_get, _mock_read_cache, _mock_write_cache):
        def fake_get(url, **_kwargs):
            if url.endswith("/index.json"):
                return MagicMock(
                    status_code=200,
                    json=lambda: {
                        "$schema": V2_SCHEMA,
                        "skills": [{
                            "name": "agent", "type": "skill-md", "url": "/agent.md",
                            "digest": "sha256:" + "0" * 64,
                        }],
                    },
                )
            return MagicMock(status_code=200, content=b"# Changed after publication\n")

        mock_get.side_effect = fake_get

        assert self._source().fetch(
            "well-known:https://example.com/.well-known/agent-skills/agent"
        ) is None

    def test_v2_index_redirect_to_private_target_is_blocked(self, monkeypatch):
        from tools import url_safety

        requests = []

        def respond(request):
            requests.append(request)
            return httpx.Response(302, headers={"location": "http://127.0.0.1/private/index.json"})

        monkeypatch.setattr(
            url_safety, "create_ssrf_safe_client",
            lambda **kwargs: httpx.Client(transport=httpx.MockTransport(respond), **kwargs),
        )
        monkeypatch.setattr(skills_hub, "is_safe_url", lambda url: "127.0.0.1" not in url)

        assert self._source().search("https://example.com", limit=10) == []
        assert len(requests) == 1

    @patch("tools.skills_hub._write_index_cache")
    @patch("tools.skills_hub._read_index_cache", return_value=None)
    @patch("tools.skills_hub._guarded_http_get")
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


@pytest.mark.parametrize("declared", [None, "1", "invalid", "-1"])
def test_bounded_http_get_enforces_actual_body_limit(monkeypatch, declared):
    data = b"0123456789"

    @contextmanager
    def stream(*_args, **_kwargs):
        headers = {} if declared is None else {"content-length": declared}
        response = httpx.Response(200, headers=headers, stream=httpx.ByteStream(data))
        try:
            yield response
        finally:
            response.close()

    monkeypatch.setattr(skills_hub, "_guarded_http_stream", stream)

    assert skills_hub._guarded_http_get("https://example.com/SKILL.md", max_bytes=9) is None


def test_bounded_http_get_accepts_body_at_exact_limit(monkeypatch):
    data = b"0123456789"

    @contextmanager
    def stream(*_args, **_kwargs):
        response = httpx.Response(200, stream=httpx.ByteStream(data))
        try:
            yield response
        finally:
            response.close()

    monkeypatch.setattr(skills_hub, "_guarded_http_stream", stream)

    response = skills_hub._guarded_http_get("https://example.com/SKILL.md", max_bytes=len(data))
    assert response is not None and response.content == data


def test_bounded_http_get_rejects_declared_oversize_without_reading(monkeypatch):
    response = MagicMock(status_code=200, headers={"content-length": "11"})
    response.iter_bytes.side_effect = AssertionError("oversized body must not be read")

    @contextmanager
    def stream(*_args, **_kwargs):
        yield response

    monkeypatch.setattr(skills_hub, "_guarded_http_stream", stream)

    assert skills_hub._guarded_http_get("https://example.com/SKILL.md", max_bytes=10) is None
    response.iter_bytes.assert_not_called()
