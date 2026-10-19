"""Atomic writers must not resurrect a deleted named profile home.

``hermes profile delete`` removes the tree and writes a tombstone under
``profiles/.deleted/<name>``. Background writers that still carry the dead
profile as their Hermes home (reasoning-caps warm thread, models.dev refresh,
gateway lifecycle ledger, MCP OAuth token writes, memory store mutations) used
to re-create ``profiles/<name>/`` with a bare ``mkdir(parents=True)`` right
before an atomic write — the exact resurrection class the tombstone guard
closed for logging and state. These tests lock the same contract for the
writers themselves: a tombstoned home raises ``FileNotFoundError`` and leaves
nothing on disk, while unrelated paths with a ``profiles`` path segment keep
working.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hermes_constants import (
    mark_named_profile_deleted,
    named_profile_home,
    set_hermes_home_override,
)
from utils import atomic_json_write, atomic_write_text


def _tombstoned_profile(tmp_path: Path) -> Path:
    """A real ``<root>/profiles/<name>`` home that has just been deleted."""
    (tmp_path / "config.yaml").write_text("{}\n", encoding="utf-8")
    profile = tmp_path / "profiles" / "p1"
    profile.mkdir(parents=True)
    mark_named_profile_deleted(profile)
    import shutil

    shutil.rmtree(profile)
    assert named_profile_home(profile) is not None
    assert not profile.exists()
    return profile


class TestAtomicWritersRefuseDeletedProfileHome:
    def test_atomic_json_write_does_not_recreate_home(self, tmp_path):
        profile = _tombstoned_profile(tmp_path)
        with pytest.raises(
            FileNotFoundError, match="Named profile home does not exist"
        ):
            atomic_json_write(profile / "cache" / "reasoning_caps.json", {"m": {}})
        assert not profile.exists()

    def test_late_models_cache_save_after_delete(self, tmp_path):
        from hermes_cli.models import _write_json_cache

        profile = _tombstoned_profile(tmp_path)
        token = set_hermes_home_override(profile)
        try:
            with pytest.raises(
                FileNotFoundError, match="Named profile home does not exist"
            ):
                _write_json_cache(
                    profile / "cache" / "reasoning_caps.json",
                    {"m": {}},
                    indent=0,
                    separators=(",", ":"),
                )
        finally:
            from hermes_constants import reset_hermes_home_override

            reset_hermes_home_override(token)
        assert not profile.exists()

    def test_lifecycle_sentinel_write_after_delete(self, tmp_path):
        from gateway.lifecycle_ledger import _write_sentinel

        profile = _tombstoned_profile(tmp_path)
        _write_sentinel({"reason": "clean-exit"}, profile)
        assert not profile.exists()

    def test_oauth_token_write_after_delete(self, tmp_path):
        from tools.mcp_oauth import _write_json

        profile = _tombstoned_profile(tmp_path)
        with pytest.raises(
            FileNotFoundError, match="Named profile home does not exist"
        ):
            _write_json(profile / "mcp-oauth" / "tokens.json", {"access_token": "x"})
        assert not profile.exists()

    def test_memory_store_add_after_delete(self, tmp_path):
        from tools.memory_tool_store import MemoryStore

        profile = _tombstoned_profile(tmp_path)
        token = set_hermes_home_override(profile)
        try:
            store = MemoryStore(memory_char_limit=100, user_char_limit=100)
            with pytest.raises(
                FileNotFoundError, match="Named profile home does not exist"
            ):
                store.add("memory", "entry")
        finally:
            from hermes_constants import reset_hermes_home_override

            reset_hermes_home_override(token)
        assert not profile.exists()

    def test_roundtrip_yaml_update_does_not_recreate_home(self, tmp_path):
        from utils import atomic_roundtrip_yaml_update

        profile = _tombstoned_profile(tmp_path)
        with pytest.raises(
            FileNotFoundError, match="Named profile home does not exist"
        ):
            atomic_roundtrip_yaml_update(profile / "config.yaml", "model", "glm-5.3")
        assert not profile.exists()


class TestUnrelatedProfilesPathsStillWrite:
    def test_custom_home_with_profiles_segment_writes(self, tmp_path):
        custom_home = tmp_path / "srv" / "profiles" / "buildcache"
        atomic_json_write(custom_home / "cache" / "blob.json", {"a": 1})
        assert json.loads(
            (custom_home / "cache" / "blob.json").read_text(encoding="utf-8")
        ) == {"a": 1}
