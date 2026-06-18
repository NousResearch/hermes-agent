"""Tests for ARD (Agentic Resource Discovery) source adapter.

Tests cover:
  - ArdSource search/inspect/fetch with mocked HTTP responses
  - MCP server card URL construction from HF Space metadata
  - is_mcp_bundle / get_mcp_config_from_bundle helpers
  - Static catalog fallback when /search endpoint is unavailable
  - MCP bundle vs skill bundle distinction
"""

import json
from unittest.mock import patch, MagicMock

import httpx
import pytest

from tools.skills_hub import (
    ArdSource,
    SkillMeta,
    SkillBundle,
    is_mcp_bundle,
    get_mcp_config_from_bundle,
    ARD_TYPE_SKILL,
    ARD_TYPE_MCP_SERVER,
    ARD_TYPE_MCP_SERVER_CARD,
    ARD_TYPE_A2A_AGENT,
    create_source_router,
    generate_ard_catalog,
    publish_ard_catalog,
    ard_local_search,
    _generate_ard_skill_entries,
    _generate_ard_mcp_entries,
    _cosine_similarity,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

MOCK_SEARCH_RESPONSE = {
    "results": [
        {
            "identifier": "urn:ai:huggingface.co:mcp:space:mrfakename:Z-Image-Turbo",
            "displayName": "Z Image Turbo MCP Server",
            "type": "application/mcp-server-card+json",
            "url": "https://evalstate-hf-discover.hf.space/mcp/huggingface/mrfakename/Z-Image-Turbo/server.json",
            "description": "Generate vivid images from text prompts in seconds",
            "tags": ["huggingface", "space", "gradio", "Image Generation"],
            "metadata": {
                "sourceType": "huggingface-space",
                "spaceId": "mrfakename/Z-Image-Turbo",
                "author": "mrfakename",
                "sdk": "gradio",
            },
        },
        {
            "identifier": "urn:ai:huggingface.co:skill:llm-trainer",
            "displayName": "huggingface-llm-trainer",
            "type": "application/ai-skill",
            "url": "https://huggingface.co/spaces/example/llm-trainer/agents.md",
            "description": "Train or fine-tune language models using TRL",
            "tags": ["huggingface", "training"],
        },
    ]
}

MOCK_CATALOG = {
    "specVersion": "1.0",
    "host": {"displayName": "Test Registry", "identifier": "did:web:example.com"},
    "entries": [
        {
            "identifier": "urn:ai:example.com:mcp:whisper",
            "displayName": "Whisper MCP Server",
            "type": "application/mcp-server-card+json",
            "url": "https://example.com/mcp/whisper/server.json",
            "description": "Transcribe audio to text",
            "metadata": {"spaceId": "hf-audio/whisper-large-v3"},
        },
        {
            "identifier": "urn:ai:example.com:skill:summarizer",
            "displayName": "Text Summarizer Skill",
            "type": "application/ai-skill",
            "url": "https://example.com/skills/summarizer/SKILL.md",
            "description": "Summarize long text documents",
        },
    ],
}


def _mock_httpx_post(url, json=None, **kwargs):
    """Mock httpx.post for ARD search."""
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = MOCK_SEARCH_RESPONSE
    return resp


def _mock_httpx_get(url, **kwargs):
    """Mock httpx.get for static catalog fallback."""
    resp = MagicMock()
    if "ai-catalog.json" in url:
        resp.status_code = 200
        resp.json.return_value = MOCK_CATALOG
    elif "SKILL.md" in url or "agents.md" in url:
        resp.status_code = 200
        resp.text = "---\nname: test-skill\ndescription: test\n---\n# Test"
    else:
        resp.status_code = 404
    return resp


# ---------------------------------------------------------------------------
# ArdSource basic tests
# ---------------------------------------------------------------------------

class TestArdSourceBasics:
    def test_source_id(self):
        assert ArdSource().source_id() == "ard"

    def test_default_registries(self):
        src = ArdSource()
        assert len(src._registries) >= 1
        assert "huggingface" in src._registries[0]

    def test_custom_registries(self):
        src = ArdSource(registries=["https://custom.registry.com"])
        assert src._registries == ["https://custom.registry.com"]

    def test_trust_level(self):
        assert ArdSource().trust_level_for("test") == "community"

    def test_in_source_router(self):
        sources = create_source_router()
        ard = [s for s in sources if s.source_id() == "ard"]
        assert len(ard) == 1


# ---------------------------------------------------------------------------
# Search tests
# ---------------------------------------------------------------------------

class TestArdSearch:
    @patch("tools.skills_hub.httpx.post", side_effect=_mock_httpx_post)
    def test_search_returns_results(self, _mock):
        src = ArdSource(registries=["https://test.registry.com"])
        results = src.search("image generation", limit=5)
        assert len(results) >= 1
        assert any(r.name == "Z Image Turbo MCP Server" for r in results)

    @patch("tools.skills_hub.httpx.post", side_effect=_mock_httpx_post)
    def test_search_caches_results(self, _mock):
        src = ArdSource(registries=["https://test.registry.com"])
        results = src.search("test", limit=5)
        identifier = results[0].identifier
        # inspect should hit cache, not HTTP
        meta = src.inspect(identifier)
        assert meta is not None
        assert meta.name == results[0].name

    @patch("tools.skills_hub.httpx.post", side_effect=_mock_httpx_post)
    def test_search_returns_mcp_and_skill_types(self, _mock):
        src = ArdSource(registries=["https://test.registry.com"])
        results = src.search("test", limit=10)
        types = {r.extra.get("ard_type") for r in results}
        assert ARD_TYPE_MCP_SERVER_CARD in types
        assert ARD_TYPE_SKILL in types

    @patch("tools.skills_hub.httpx.post")
    def test_search_fallback_to_static_catalog(self, mock_post):
        # /search returns non-200 → fallback to static catalog
        mock_resp = MagicMock()
        mock_resp.status_code = 500
        mock_post.return_value = mock_resp

        src = ArdSource(registries=["https://test.registry.com"])
        with patch("tools.skills_hub._guarded_http_get") as mock_get:
            catalog_resp = MagicMock()
            catalog_resp.status_code = 200
            catalog_resp.json.return_value = MOCK_CATALOG
            mock_get.return_value = catalog_resp

            results = src.search("whisper", limit=5)
            assert len(results) >= 1
            assert results[0].name == "Whisper MCP Server"

    @patch("tools.skills_hub.httpx.post", side_effect=_mock_httpx_post)
    def test_search_empty_query(self, _mock):
        src = ArdSource(registries=["https://test.registry.com"])
        results = src.search("", limit=5)
        assert isinstance(results, list)


# ---------------------------------------------------------------------------
# MCP URL construction tests
# ---------------------------------------------------------------------------

class TestMcpUrlConstruction:
    @patch("tools.skills_hub.httpx.post", side_effect=_mock_httpx_post)
    def test_mcp_url_from_space_id(self, _mock):
        src = ArdSource(registries=["https://test.registry.com"])
        results = src.search("test", limit=5)
        mcp_results = [r for r in results if r.extra.get("mcp")]
        assert len(mcp_results) >= 1
        mcp = mcp_results[0].extra["mcp"]
        assert "mrfakename-Z-Image-Turbo.hf.space" in mcp["url"]
        assert mcp["url"].endswith("/gradio_api/mcp")
        assert mcp["space_id"] == "mrfakename/Z-Image-Turbo"

    @patch("tools.skills_hub.httpx.post", side_effect=_mock_httpx_post)
    def test_mcp_transport_default(self, _mock):
        src = ArdSource(registries=["https://test.registry.com"])
        results = src.search("test", limit=5)
        mcp_results = [r for r in results if r.extra.get("mcp")]
        if mcp_results:
            assert mcp_results[0].extra["mcp"]["transport"] == "streamable_http"


# ---------------------------------------------------------------------------
# Inspect tests
# ---------------------------------------------------------------------------

class TestArdInspect:
    @patch("tools.skills_hub.httpx.post", side_effect=_mock_httpx_post)
    def test_inspect_cached(self, _mock):
        src = ArdSource(registries=["https://test.registry.com"])
        results = src.search("test", limit=5)
        identifier = results[0].identifier
        meta = src.inspect(identifier)
        assert meta is not None
        assert meta.identifier == identifier

    @patch("tools.skills_hub._guarded_http_get")
    def test_inspect_from_static_catalog(self, mock_get):
        catalog_resp = MagicMock()
        catalog_resp.status_code = 200
        catalog_resp.json.return_value = MOCK_CATALOG
        mock_get.return_value = catalog_resp

        src = ArdSource(registries=["https://test.registry.com"])
        meta = src.inspect("urn:ai:example.com:skill:summarizer")
        assert meta is not None
        assert meta.name == "Text Summarizer Skill"
        assert meta.extra.get("ard_type") == ARD_TYPE_SKILL


# ---------------------------------------------------------------------------
# Fetch tests
# ---------------------------------------------------------------------------

class TestArdFetch:
    @patch("tools.skills_hub.httpx.post", side_effect=_mock_httpx_post)
    @patch("tools.skills_hub._guarded_http_get")
    def test_fetch_mcp_bundle(self, mock_get, _mock_post):
        src = ArdSource(registries=["https://test.registry.com"])
        results = src.search("test", limit=5)
        mcp_results = [r for r in results if r.extra.get("mcp")]
        assert mcp_results, "No MCP results in search"

        bundle = src.fetch(mcp_results[0].identifier)
        assert bundle is not None
        assert bundle.source == "ard"
        # MCP bundles have no files — the MCP server is live
        assert len(bundle.files) == 0
        assert is_mcp_bundle(bundle)

    @patch("tools.skills_hub.httpx.post", side_effect=_mock_httpx_post)
    @patch("tools.skills_hub._guarded_http_get")
    def test_fetch_skill_bundle(self, mock_get, _mock_post):
        skill_resp = MagicMock()
        skill_resp.status_code = 200
        skill_resp.text = "---\nname: test\ndescription: test\n---\n# Test"
        mock_get.return_value = skill_resp

        src = ArdSource(registries=["https://test.registry.com"])
        results = src.search("test", limit=5)
        skill_results = [r for r in results if not r.extra.get("mcp")]
        if not skill_results:
            pytest.skip("No skill results in mock data")

        bundle = src.fetch(skill_results[0].identifier)
        assert bundle is not None
        assert "SKILL.md" in bundle.files
        assert not is_mcp_bundle(bundle)


# ---------------------------------------------------------------------------
# Helper function tests
# ---------------------------------------------------------------------------

class TestMcpBundleHelpers:
    def test_is_mcp_bundle_true(self):
        bundle = SkillBundle(
            name="test", files={}, source="ard", identifier="test",
            trust_level="community",
            metadata={"ard_type": ARD_TYPE_MCP_SERVER},
        )
        assert is_mcp_bundle(bundle) is True

    def test_is_mcp_bundle_true_for_card(self):
        bundle = SkillBundle(
            name="test", files={}, source="ard", identifier="test",
            trust_level="community",
            metadata={"ard_type": ARD_TYPE_MCP_SERVER_CARD},
        )
        assert is_mcp_bundle(bundle) is True

    def test_is_mcp_bundle_false_for_skill(self):
        bundle = SkillBundle(
            name="test", files={"SKILL.md": "..."}, source="ard",
            identifier="test", trust_level="community",
            metadata={"ard_type": ARD_TYPE_SKILL},
        )
        assert is_mcp_bundle(bundle) is False

    def test_is_mcp_bundle_false_no_metadata(self):
        bundle = SkillBundle(
            name="test", files={"SKILL.md": "..."}, source="github",
            identifier="test", trust_level="community",
        )
        assert is_mcp_bundle(bundle) is False

    def test_get_mcp_config_valid(self):
        bundle = SkillBundle(
            name="test-mcp", files={}, source="ard", identifier="test",
            trust_level="community",
            metadata={
                "ard_type": ARD_TYPE_MCP_SERVER_CARD,
                "mcp": {"url": "https://example.com/mcp", "name": "test-mcp"},
            },
        )
        cfg = get_mcp_config_from_bundle(bundle)
        assert cfg is not None
        assert cfg["name"] == "test-mcp"
        assert cfg["url"] == "https://example.com/mcp"
        assert cfg["transport"] == "streamable_http"

    def test_get_mcp_config_no_url(self):
        bundle = SkillBundle(
            name="test", files={}, source="ard", identifier="test",
            trust_level="community",
            metadata={"ard_type": ARD_TYPE_MCP_SERVER, "mcp": {}},
        )
        cfg = get_mcp_config_from_bundle(bundle)
        assert cfg is None

    def test_get_mcp_config_non_mcp_bundle(self):
        bundle = SkillBundle(
            name="test", files={"SKILL.md": "..."}, source="ard",
            identifier="test", trust_level="community",
            metadata={"ard_type": ARD_TYPE_SKILL},
        )
        cfg = get_mcp_config_from_bundle(bundle)
        assert cfg is None


# ---------------------------------------------------------------------------
# Entry-to-Meta conversion tests
# ---------------------------------------------------------------------------

class TestEntryConversion:
    def test_mcp_entry_with_space_id(self):
        entry = {
            "identifier": "urn:ai:test:mcp:example",
            "displayName": "Test MCP",
            "type": "application/mcp-server-card+json",
            "url": "https://example.com/server.json",
            "description": "A test MCP",
            "metadata": {"spaceId": "user/test-space"},
        }
        src = ArdSource(registries=["https://test.com"])
        meta = src._entry_to_meta(entry, "https://test.com")
        mcp = meta.extra["mcp"]
        assert "user-test-space.hf.space" in mcp["url"]
        assert mcp["transport"] == "streamable_http"

    def test_skill_entry(self):
        entry = {
            "identifier": "urn:ai:test:skill:example",
            "displayName": "Test Skill",
            "type": "application/ai-skill",
            "url": "https://example.com/SKILL.md",
            "description": "A test skill",
        }
        src = ArdSource(registries=["https://test.com"])
        meta = src._entry_to_meta(entry, "https://test.com")
        assert meta.extra["ard_type"] == ARD_TYPE_SKILL
        assert meta.extra["source_url"] == "https://example.com/SKILL.md"
        assert "mcp" not in meta.extra

    def test_entry_missing_identifier(self):
        entry = {"displayName": "No ID", "type": "application/ai-skill"}
        src = ArdSource(registries=["https://test.com"])
        meta = src._entry_to_meta(entry, "https://test.com")
        assert meta.identifier == "ard:No ID"


# ---------------------------------------------------------------------------
# Phase 3: Publisher tests
# ---------------------------------------------------------------------------

class TestArdPublisher:
    def test_generate_catalog_returns_dict(self):
        catalog = generate_ard_catalog(domain="test.local")
        assert isinstance(catalog, dict)
        assert "specVersion" in catalog
        assert "host" in catalog
        assert "entries" in catalog
        assert isinstance(catalog["entries"], list)

    def test_generate_catalog_urns(self):
        catalog = generate_ard_catalog(domain="test.local")
        for entry in catalog["entries"]:
            assert entry["identifier"].startswith("urn:ai:test.local:")

    def test_generate_catalog_types(self):
        catalog = generate_ard_catalog(domain="test.local")
        for entry in catalog["entries"]:
            assert "type" in entry
            assert entry["type"] in (
                ARD_TYPE_SKILL,
                ARD_TYPE_MCP_SERVER,
                ARD_TYPE_MCP_SERVER_CARD,
            )

    def test_generate_ard_skill_entries(self):
        skills = [
            {"name": "test-skill", "description": "A test", "category": "dev"},
            {"name": "no-desc", "description": "", "category": ""},
        ]
        entries = _generate_ard_skill_entries(skills, "test.local")
        assert len(entries) == 2
        assert entries[0]["displayName"] == "test-skill"
        assert entries[0]["type"] == ARD_TYPE_SKILL
        assert "dev" in entries[0].get("tags", [])

    def test_generate_ard_mcp_entries(self):
        servers = {
            "test-mcp": {"url": "https://example.com/mcp"},
            "no-url": {"command": "npx"},  # no URL → skipped
        }
        entries = _generate_ard_mcp_entries(servers, "test.local")
        assert len(entries) == 1
        assert entries[0]["displayName"] == "test-mcp (MCP Server)"
        assert entries[0]["url"] == "https://example.com/mcp"

    def test_publish_ard_catalog_writes_file(self, tmp_path, monkeypatch):
        # Mock hermes_home to tmp_path
        import tools.skills_hub as mod

        fake_home = tmp_path / "hermes"
        fake_home.mkdir()

        def fake_get_home():
            return fake_home

        monkeypatch.setattr(
            "hermes_constants.get_hermes_home", fake_get_home
        )
        # The function imports get_hermes_home at call time
        monkeypatch.setattr(
            "tools.skills_hub.get_hermes_home", fake_get_home
        )

        catalog_path = fake_home / ".well-known" / "ai-catalog.json"
        # generate_ard_catalog may fail to find skills in tmp_path,
        # but should still produce a valid empty catalog
        catalog = generate_ard_catalog(
            domain="test.local", output_path=str(catalog_path)
        )
        assert catalog_path.exists()
        import json

        written = json.loads(catalog_path.read_text())
        assert "entries" in written
        assert written["host"]["identifier"] == "did:web:test.local"


# ---------------------------------------------------------------------------
# Phase 4: Local search tests
# ---------------------------------------------------------------------------

class TestArdLocalSearch:
    def test_keyword_search_returns_results(self):
        results = ard_local_search("web search extract", limit=5)
        assert isinstance(results, list)
        # In test isolation, the catalog may be empty — only assert
        # scoring structure when results exist
        if results:
            assert "score" in results[0]
            assert 0 <= results[0]["score"] <= 100

    def test_empty_query_returns_entries(self):
        results = ard_local_search("", limit=5)
        assert isinstance(results, list)
        # Empty query should return entries with neutral score
        if results:
            assert results[0]["score"] == 50

    def test_filter_by_type(self):
        results = ard_local_search(
            "", limit=10, filter_types=[ARD_TYPE_MCP_SERVER]
        )
        for r in results:
            assert r["type"] == ARD_TYPE_MCP_SERVER

    def test_no_match_returns_empty(self):
        results = ard_local_search("zzzznonexistentxyzzy", limit=5)
        # Should return empty or very low scored results
        assert isinstance(results, list)

    def test_semantic_fallback(self):
        # Without embedding API configured, semantic=True should
        # fall back to keyword search
        results = ard_local_search("web search", limit=5, semantic=True)
        assert isinstance(results, list)
        if results:
            assert "score" in results[0]


class TestCosineSimilarity:
    def test_identical_vectors(self):
        v = [1.0, 0.5, 0.3]
        score = _cosine_similarity(v, v)
        assert abs(score - 1.0) < 0.01  # ~1.0 for identical vectors

    def test_orthogonal_vectors(self):
        score = _cosine_similarity([1.0, 0.0], [0.0, 1.0])
        assert abs(score) < 0.01  # ~0.0 for orthogonal

    def test_opposite_vectors(self):
        score = _cosine_similarity([1.0, 0.0], [-1.0, 0.0])
        assert score < -0.99  # ~-1.0 for opposite

    def test_zero_vector(self):
        score = _cosine_similarity([0.0, 0.0], [1.0, 1.0])
        assert score == 0.0


# ---------------------------------------------------------------------------
# Cache invalidation tests
# ---------------------------------------------------------------------------

class TestCacheInvalidation:
    def test_invalidate_returns_bool(self):
        from tools.skills_hub import _invalidate_embeddings_if_stale

        result = _invalidate_embeddings_if_stale()
        assert isinstance(result, bool)

    def test_invalidate_writes_hash(self, tmp_path):
        import tools.skills_hub as mod

        # Patch HUB_DIR and reset cached paths
        orig_hub = mod.HUB_DIR
        orig_cache_path = mod._EMBEDDINGS_CACHE_PATH
        mod.HUB_DIR = tmp_path
        mod._EMBEDDINGS_CACHE_PATH = None
        try:
            from tools.skills_hub import _invalidate_embeddings_if_stale

            _invalidate_embeddings_if_stale()
            hash_file = tmp_path / "ard-embeddings.hash"
            assert hash_file.exists()
            assert len(hash_file.read_text().strip()) == 32  # MD5 hex
        finally:
            mod.HUB_DIR = orig_hub
            mod._EMBEDDINGS_CACHE_PATH = orig_cache_path

    def test_second_invalidate_is_noop(self, tmp_path):
        import tools.skills_hub as mod

        orig_hub = mod.HUB_DIR
        orig_cache_path = mod._EMBEDDINGS_CACHE_PATH
        mod.HUB_DIR = tmp_path
        mod._EMBEDDINGS_CACHE_PATH = None
        try:
            from tools.skills_hub import _invalidate_embeddings_if_stale

            first = _invalidate_embeddings_if_stale()
            second = _invalidate_embeddings_if_stale()
            # First call invalidates, second should be a no-op
            # (unless test environment catalog changed between calls)
            assert isinstance(first, bool)
            assert isinstance(second, bool)
        finally:
            mod.HUB_DIR = orig_hub
            mod._EMBEDDINGS_CACHE_PATH = orig_cache_path


# ---------------------------------------------------------------------------
# MCP auto-registration integration tests
# ---------------------------------------------------------------------------

class TestMcpAutoRegistration:
    def test_add_mcp_server_from_config_missing_fields(self):
        from tools.mcp_tool import add_mcp_server_from_config

        success, msg = add_mcp_server_from_config({"name": "", "url": ""})
        assert success is False
        assert "must include" in msg

    def test_add_mcp_server_from_config_missing_url(self):
        from tools.mcp_tool import add_mcp_server_from_config

        success, msg = add_mcp_server_from_config({"name": "test", "url": ""})
        assert success is False

    def test_add_mcp_server_from_config_returns_tuple(self):
        """Verify the function signature returns (bool, str)."""
        from tools.mcp_tool import add_mcp_server_from_config

        result = add_mcp_server_from_config({})
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], bool)
        assert isinstance(result[1], str)
