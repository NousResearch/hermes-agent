"""`/reload-mcp` must classify its diff against config, not live sockets (#80771).

The reload diff was built purely from ``tools.mcp_tool._servers`` on either
side of the reconnect, so "removed" answered "was connected, isn't now". A
server whose connect is still in flight when the diff runs — the normal case
past ``mcp_discovery_timeout`` — is still in ``config.yaml`` and connects
moments later, but was reported as ``➖ Removed`` on both the CLI and the
gateway, and the same wrong set was injected into the model's history.
"""
from unittest.mock import patch

import pytest

from tools.mcp_tool import classify_reload_diff


CONFIG = {"docs": {}, "search": {}, "calendar": {}, "maps": {}}


def _diff(before, after, config=CONFIG):
    with patch("tools.mcp_tool._load_mcp_config", return_value=config):
        return classify_reload_diff(set(before), set(after))


class TestReloadDiffClassification:
    def test_still_configured_but_slow_is_not_removed(self):
        """The reported shape: two servers still connecting at diff time."""
        d = _diff(
            before=["docs", "search", "calendar", "maps"],
            after=["docs", "search"],
        )
        assert d["removed"] == set()
        assert d["not_connected"] == {"calendar", "maps"}
        assert d["reconnected"] == {"docs", "search"}

    def test_genuinely_deleted_server_is_removed(self):
        """Dropping a server from config is what the label is for."""
        d = _diff(
            before=["docs", "gone"],
            after=["docs"],
            config={"docs": {}},
        )
        assert d["removed"] == {"gone"}
        assert d["not_connected"] == set()

    def test_disabled_server_is_not_reported_as_removed(self):
        """`enabled: false` is still configured — _load_mcp_config keeps it."""
        d = _diff(before=["docs", "calendar"], after=["docs"])
        assert d["removed"] == set()
        assert d["not_connected"] == {"calendar"}

    def test_new_server_is_added(self):
        d = _diff(before=["docs"], after=["docs", "search"])
        assert d["added"] == {"search"}
        assert d["removed"] == set()
        assert d["reconnected"] == {"docs"}

    def test_sets_are_disjoint_and_cover_the_churn(self):
        """No name may land in two buckets, and none may be dropped."""
        before = {"docs", "search", "calendar", "gone"}
        after = {"docs", "brand_new"}
        d = _diff(before, after)
        buckets = [d["added"], d["removed"], d["reconnected"], d["not_connected"]]
        for i, a in enumerate(buckets):
            for b in buckets[i + 1:]:
                assert not (a & b), f"{a} and {b} overlap"
        assert d["added"] | d["reconnected"] == after
        assert d["removed"] | d["not_connected"] == before - after

    def test_unreadable_config_falls_back_to_connection_only(self):
        """A config read failure must not raise through a reload path."""
        with patch("tools.mcp_tool._load_mcp_config", side_effect=RuntimeError("boom")):
            d = classify_reload_diff({"docs", "calendar"}, {"docs"})
        assert d["removed"] == {"calendar"}   # pre-fix behaviour, not an exception
        assert d["not_connected"] == set()
