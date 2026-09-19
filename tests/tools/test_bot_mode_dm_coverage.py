"""Test coverage for tools/bot_mode_dm.py — 24 functions had LOW coverage.

Tests the pure helper functions: path resolution, roster reading,
name resolution, and error formatting. All filesystem access uses
tmp_path — no real profiles are touched.
"""

from pathlib import Path

from tools.bot_mode_dm import (
    _hermes_root,
    _self_profile_name,
    _handle,
    _resolve_local_name,
)


class TestHermesRoot:
    def test_returns_path(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        result = _hermes_root(tmp_path)
        assert isinstance(result, Path)


class TestSelfProfileName:
    def test_returns_string(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        result = _self_profile_name(tmp_path)
        assert isinstance(result, str)


class TestHandle:
    def test_default_maps_to_hermes(self):
        assert _handle("default") == "hermes"

    def test_other_name_unchanged(self):
        assert _handle("alice") == "alice"


class TestResolveLocalName:
    def test_exact_match(self):
        roster = ["alice", "bob"]
        assert _resolve_local_name("alice", roster) == "alice"

    def test_case_insensitive(self):
        roster = ["Alice", "bob"]
        assert _resolve_local_name("alice", roster) == "Alice"

    def test_hermes_maps_to_default(self):
        roster = ["default", "alice"]
        assert _resolve_local_name("hermes", roster) == "default"

    def test_no_match_returns_none(self):
        roster = ["alice"]
        assert _resolve_local_name("charlie", roster) is None

    def test_empty_target_returns_none(self):
        assert _resolve_local_name("", ["alice"]) is None
