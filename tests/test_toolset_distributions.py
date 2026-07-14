"""Tests for toolset_distributions.py — distribution CRUD, sampling, validation."""

import pytest
from unittest.mock import patch

from toolset_distributions import (
    DISTRIBUTIONS,
    get_distribution,
    list_distributions,
    print_distribution_info,
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


class TestPrintDistributionInfo:
    def test_known_distribution_prints_info(self, capsys):
        print_distribution_info("default")

        captured = capsys.readouterr()
        assert "default" in captured.out
        assert DISTRIBUTIONS["default"]["description"] in captured.out
        assert "Description:" in captured.out
        assert "Toolsets:" in captured.out
        assert "% chance" in captured.out

    def test_unknown_distribution_prints_error(self, capsys):
        print_distribution_info("nonexistent_distribution_xyz")

        captured = capsys.readouterr()
        assert "Unknown distribution" in captured.out
        assert "nonexistent_distribution_xyz" in captured.out


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


class TestSampleInvalidToolset:
    @patch("toolset_distributions.validate_toolset")
    @patch("toolset_distributions.random.random", return_value=0.0)
    def test_invalid_toolset_skipped_with_warning(self, mock_random, mock_validate, capsys):
        mock_validate.side_effect = lambda toolset_name: toolset_name != "terminal"

        result = sample_toolsets_from_distribution("terminal_only")

        captured = capsys.readouterr()
        assert "terminal" not in result
        assert "file" in result
        assert "Warning" in captured.out
        assert "terminal" in captured.out
        assert "terminal_only" in captured.out

    @patch("toolset_distributions.validate_toolset", return_value=False)
    def test_all_toolsets_invalid_fallback_also_invalid(self, mock_validate):
        dist = get_distribution("terminal_only")
        toolset_count = len(dist["toolsets"])

        result = sample_toolsets_from_distribution("terminal_only")

        assert result == []
        # validate_toolset called once per toolset in the loop + once for the fallback
        assert mock_validate.call_count == toolset_count + 1


class TestSampleFallback:
    @patch("toolset_distributions.validate_toolset", return_value=True)
    @patch("toolset_distributions.random.random", return_value=1.0)
    def test_fallback_picks_highest_probability(self, mock_random, mock_validate):
        result = sample_toolsets_from_distribution("safe")
        dist = get_distribution("safe")
        expected = max(dist["toolsets"], key=dist["toolsets"].get)

        assert result == [expected]

    @patch("toolset_distributions.validate_toolset", return_value=True)
    @patch("toolset_distributions.random.random", return_value=0.0)
    def test_deterministic_all_included(self, mock_random, mock_validate):
        result = sample_toolsets_from_distribution("terminal_only")

        assert set(result) == set(get_distribution("terminal_only")["toolsets"])


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
