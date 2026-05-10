"""Tests for toolset_distributions.py — distribution CRUD, sampling, validation."""

import pytest
from unittest.mock import patch

from toolset_distributions import (
    DISTRIBUTIONS,
    get_distribution,
    list_distributions,
    sample_toolsets_from_distribution,
    validate_distribution,
)


class TestGetDistribution:
    def test_known_distribution(self):
        dist = get_distribution("default")
        assert dist is not None
        assert "description" in dist
        assert "toolsets" in dist

    def test_unknown_returns_none(self):
        assert get_distribution("nonexistent") is None

    def test_all_named_distributions_exist(self):
        expected = [
            "default", "image_gen", "research", "science", "development",
            "safe", "balanced", "minimal", "terminal_only", "terminal_web",
            "creative", "reasoning", "browser_use", "browser_only",
            "browser_tasks", "terminal_tasks", "mixed_tasks",
        ]
        for name in expected:
            assert get_distribution(name) is not None, f"{name} missing"


class TestListDistributions:
    def test_returns_copy(self):
        d1 = list_distributions()
        d2 = list_distributions()
        assert d1 is not d2
        assert d1 == d2

    def test_contains_all(self):
        dists = list_distributions()
        assert len(dists) == len(DISTRIBUTIONS)


class TestValidateDistribution:
    def test_valid(self):
        assert validate_distribution("default") is True
        assert validate_distribution("research") is True

    def test_invalid(self):
        assert validate_distribution("nonexistent") is False
        assert validate_distribution("") is False


class TestSampleToolsetsFromDistribution:
    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown distribution"):
            sample_toolsets_from_distribution("nonexistent")

    def test_default_returns_all_toolsets(self):
        # default has all at 100%, so all should be selected
        result = sample_toolsets_from_distribution("default")
        assert len(result) > 0
        # With 100% probability, all valid toolsets should be present
        dist = get_distribution("default")
        for ts in dist["toolsets"]:
            assert ts in result

    def test_minimal_returns_web_only(self):
        result = sample_toolsets_from_distribution("minimal")
        assert "web" in result

    def test_returns_list_of_strings(self):
        result = sample_toolsets_from_distribution("balanced")
        assert isinstance(result, list)
        for item in result:
            assert isinstance(item, str)

    def test_fallback_guarantees_at_least_one(self):
        # Even with low probabilities, at least one toolset should be selected
        for _ in range(20):
            result = sample_toolsets_from_distribution("reasoning")
            assert len(result) >= 1


class TestDistributionStructure:
    def test_all_have_required_keys(self):
        for name, dist in DISTRIBUTIONS.items():
            assert "description" in dist, f"{name} missing description"
            assert "toolsets" in dist, f"{name} missing toolsets"
            assert isinstance(dist["toolsets"], dict), f"{name} toolsets not a dict"

    def test_probabilities_are_valid_range(self):
        for name, dist in DISTRIBUTIONS.items():
            for ts_name, prob in dist["toolsets"].items():
                assert 0 < prob <= 100, f"{name}.{ts_name} has invalid probability {prob}"

    def test_descriptions_non_empty(self):
        for name, dist in DISTRIBUTIONS.items():
            assert len(dist["description"]) > 5, f"{name} has too short description"
