# =========================================================================
# Cross-profile discovery — profile param with FTS search
# =========================================================================

import json

import pytest

from hermes_state import SessionDB
from tools.session_search_tool import session_search


@pytest.fixture
def db(tmp_path):
    return SessionDB(tmp_path / "state.db")


class TestCrossProfileDiscovery:
    """session_search(query=..., profile=\"other\") must search the named
    profile's DB, not the current one.  This covers both the SessionDB
    read-only FTS detection (#49554) and the agent-loop parameter passthrough
    (#60789)."""

    @staticmethod
    def _patch_profiles(monkeypatch, home, exists=True):
        from hermes_cli import profiles as profiles_mod
        monkeypatch.setattr(profiles_mod, "normalize_profile_name", lambda n: n)
        monkeypatch.setattr(profiles_mod, "validate_profile_name", lambda n: None)
        monkeypatch.setattr(profiles_mod, "profile_exists", lambda n: exists)
        monkeypatch.setattr(profiles_mod, "get_profile_dir", lambda n: home)

    def test_discovery_with_profile_searches_correct_db(self, db, tmp_path, monkeypatch):
        """Query the other profile's DB — must return hits from that profile only."""
        # ── Seed "other" profile DB ──
        other_home = tmp_path / "other_home"
        other_home.mkdir()
        other_db = SessionDB(other_home / "state.db")
        try:
            other_db.create_session("s_alt", source="cli")
            other_db.append_message("s_alt", role="user", content="koala conservation plan")
            other_db.append_message("s_alt", role="assistant", content="Let's protect the koalas.")
            other_db._conn.commit()
        finally:
            other_db.close()

        # ── Seed current profile DB with a DIFFERENT topic ──
        db.create_session("s_curr", source="cli")
        db.append_message("s_curr", role="user", content="penguin habitat study")
        db._conn.commit()

        self._patch_profiles(monkeypatch, other_home)

        # Query the other profile — should find "koala", not "penguin"
        result = json.loads(
            session_search(query="koala", profile="other", db=db)
        )
        assert result["success"] is True
        assert result["mode"] == "discover"
        assert result["count"] >= 1, "should find koala in other profile"
        sids = [r["session_id"] for r in result["results"]]
        assert "s_alt" in sids, "should include the other profile's session"

    def test_discovery_profile_isolation(self, db, tmp_path, monkeypatch):
        """A query that only matches the current profile must return empty
        when searching the other profile."""
        other_home = tmp_path / "other_home"
        other_home.mkdir()
        other_db = SessionDB(other_home / "state.db")
        try:
            other_db.create_session("s_alt", source="cli")
            other_db.append_message("s_alt", role="user", content="koala")
            other_db._conn.commit()
        finally:
            other_db.close()

        # Current profile has "penguin" — not in other profile
        db.create_session("s_curr", source="cli")
        db.append_message("s_curr", role="user", content="penguin migration patterns")
        db._conn.commit()

        self._patch_profiles(monkeypatch, other_home)

        # Search "penguin" in other profile — should be empty
        result = json.loads(
            session_search(query="penguin", profile="other", db=db)
        )
        assert result["success"] is True
        assert result["count"] == 0, "penguin is only in current profile, not other"
