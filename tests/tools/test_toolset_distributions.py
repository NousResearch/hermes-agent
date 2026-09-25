"""Tests for toolset_distributions.py — distribution CRUD, sampling, validation."""


from toolset_distributions import (
    DISTRIBUTIONS,
    list_distributions,
    sample_toolsets_from_distribution,
)






class TestListDistributions:
    def test_returns_copy(self):
        d1 = list_distributions()
        d2 = list_distributions()
        assert d1 is not d2
        assert d1 == d2









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


class TestGroupedCompoundEntries:
    """"+"-grouped entries roll once and select all members together (#64503)."""

    def test_hit_roll_selects_all_members_together(self, monkeypatch):
        import toolset_distributions as td

        monkeypatch.setattr(td.random, "random", lambda: 0.0)  # every roll hits
        result = sample_toolsets_from_distribution("browser_tasks")
        assert "browser" in result
        assert "search" in result
        assert "browser+search" not in result  # members, not the raw key

    def test_fallback_expands_compound_members(self, monkeypatch):
        import toolset_distributions as td

        monkeypatch.setitem(
            td.DISTRIBUTIONS,
            "_test_compound_fallback",
            {"description": "t", "toolsets": {"browser+search": 60, "web": 10}},
        )
        monkeypatch.setattr(td.random, "random", lambda: 1.0)  # every roll misses
        result = sample_toolsets_from_distribution("_test_compound_fallback")
        assert set(result) == {"browser", "search"}

    def test_invalid_member_skips_whole_entry(self, monkeypatch):
        import toolset_distributions as td

        monkeypatch.setitem(
            td.DISTRIBUTIONS,
            "_test_compound_invalid",
            {"description": "t", "toolsets": {"browser+__not_a_toolset__": 100, "web": 100}},
        )
        monkeypatch.setattr(td.random, "random", lambda: 0.0)
        result = sample_toolsets_from_distribution("_test_compound_invalid")
        assert "browser" not in result  # invalid member disqualifies the group
        assert "web" in result
