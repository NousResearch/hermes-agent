"""Tests for toolset_distributions.py — distribution CRUD, sampling, validation."""

import pytest

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




class TestGetDistributionAnnotation:
    def test_return_annotation_uses_typing_any_not_builtin(self):
        """The return annotation must be typing.Any, not the builtin any().

        Regression for the ``Optional[Dict[str, any]]`` typo: the builtin
        ``any`` is a function, not a type, so the annotation was semantically
        wrong (and static checkers reject it). This asserts the value nested
        in the annotation is ``typing.Any``.
        """
        import typing

        ann = get_distribution.__annotations__["return"]
        # return is Optional[Dict[str, X]] == Union[Dict[str, X], None]
        union_args = typing.get_args(ann)
        dict_type = next(a for a in union_args if a is not type(None))
        key_t, val_t = typing.get_args(dict_type)
        assert val_t is typing.Any, (
            f"expected typing.Any, got {val_t!r} "
            "(the builtin any() is not a valid type annotation)"
        )
        assert val_t is not any


class TestListDistributions:
    def test_returns_copy(self):
        d1 = list_distributions()
        d2 = list_distributions()
        assert d1 is not d2
        assert d1 == d2



class TestValidateDistribution:
    def test_valid(self):
        assert validate_distribution("default") is True
        assert validate_distribution("research") is True



class TestSampleToolsetsFromDistribution:


    def test_minimal_returns_web_only(self):
        result = sample_toolsets_from_distribution("minimal")
        assert "web" in result

    def test_returns_list_of_strings(self):
        result = sample_toolsets_from_distribution("balanced")
        assert isinstance(result, list)
        for item in result:
            assert isinstance(item, str)



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

