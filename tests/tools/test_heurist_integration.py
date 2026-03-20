#!/usr/bin/env python3
"""
Integration tests: validate HeuristSource works against the live Heurist Marketplace API.
Verifies that skills can be searched, inspected, and fetched successfully.

These tests hit the real API and are marked with @pytest.mark.integration.
Run with: pytest tests/tools/test_heurist_integration.py -v -m integration
"""

import pytest

from tools.skills_hub import HeuristSource


pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def source():
    return HeuristSource()


@pytest.fixture(scope="module")
def search_results(source):
    return source.search("", limit=10)


class TestHeuristIntegrationSearch:
    def test_search_returns_at_least_3_skills(self, search_results):
        assert len(search_results) >= 3

    def test_search_results_have_identifiers(self, search_results):
        for r in search_results[:3]:
            assert r.identifier.startswith("heurist:")
            assert r.source == "heurist"

    def test_search_with_query(self, source):
        results = source.search("crypto", limit=5)
        assert len(results) >= 1


class TestHeuristIntegrationInspect:
    def test_inspect_returns_metadata(self, source, search_results):
        identifier = search_results[0].identifier
        meta = source.inspect(identifier)
        assert meta is not None
        assert meta.name
        assert meta.identifier == identifier
        assert meta.extra.get("approved_sha256") is not None
        assert "is_folder" in meta.extra

    def test_inspect_surfaces_risk_tier(self, source, search_results):
        identifier = search_results[0].identifier
        meta = source.inspect(identifier)
        assert meta.extra.get("risk_tier") in ("low", "medium", "high")

    def test_inspect_surfaces_capabilities(self, source, search_results):
        identifier = search_results[0].identifier
        meta = source.inspect(identifier)
        capabilities = meta.extra.get("capabilities")
        assert isinstance(capabilities, dict)


class TestHeuristIntegrationFetch:
    @pytest.mark.parametrize("index", [0, 1, 2])
    def test_fetch_skill(self, source, search_results, index):
        if index >= len(search_results):
            pytest.skip("Not enough skills in search results")
        identifier = search_results[index].identifier
        bundle = source.fetch(identifier)
        # Bundle may be None if SHA256 verification fails (content updated since approval)
        # That's a valid outcome — the integrity check is working correctly
        if bundle is None:
            pytest.skip(f"Fetch returned None for {identifier} (possible SHA256 mismatch)")
        assert "SKILL.md" in bundle.files
        assert bundle.source == "heurist"
        assert bundle.identifier == identifier
        assert bundle.metadata.get("approved_sha256") is not None

    @pytest.mark.parametrize("index", [0, 1, 2])
    def test_fetch_includes_security_metadata(self, source, search_results, index):
        if index >= len(search_results):
            pytest.skip("Not enough skills in search results")
        identifier = search_results[index].identifier
        bundle = source.fetch(identifier)
        if bundle is None:
            pytest.skip(f"Fetch returned None for {identifier}")
        assert "risk_tier" in bundle.metadata
        assert "capabilities" in bundle.metadata


class TestHeuristIntegrationRiskWarnings:
    def test_format_risk_warnings(self, source, search_results):
        for r in search_results[:5]:
            bundle = source.fetch(r.identifier)
            if bundle is None:
                continue
            warnings = HeuristSource.format_risk_warnings(bundle.metadata)
            assert isinstance(warnings, list)
            # At least verify it runs without error
            break
