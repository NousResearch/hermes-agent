"""Tests for Hermes-managed Camofox state helpers."""

import os
import stat

from unittest.mock import patch


def _load_module():
    from tools import browser_camofox_state as state
    return state


class TestCamofoxStatePaths:
    def test_paths_are_profile_scoped(self, tmp_path):
        state = _load_module()
        with patch.object(state, "get_hermes_home", return_value=tmp_path):
            assert state.get_camofox_state_dir() == tmp_path / "browser_auth" / "camofox"


class TestCamofoxIdentity:
    def test_identity_is_deterministic(self, tmp_path):
        state = _load_module()
        with patch.object(state, "get_hermes_home", return_value=tmp_path):
            first = state.get_camofox_identity("task-1")
            second = state.get_camofox_identity("task-1")
            assert first == second

    def test_identity_differs_by_task(self, tmp_path):
        state = _load_module()
        with patch.object(state, "get_hermes_home", return_value=tmp_path):
            a = state.get_camofox_identity("task-a")
            b = state.get_camofox_identity("task-b")
            # Same user (same profile), different session keys
            assert a["user_id"] == b["user_id"]
            assert a["session_key"] != b["session_key"]

    def test_identity_differs_by_profile(self, tmp_path):
        state = _load_module()
        with patch.object(state, "get_hermes_home", return_value=tmp_path / "profile-a"):
            a = state.get_camofox_identity("task-1")
        with patch.object(state, "get_hermes_home", return_value=tmp_path / "profile-b"):
            b = state.get_camofox_identity("task-1")
        assert a["user_id"] != b["user_id"]

    def test_default_task_id(self, tmp_path):
        state = _load_module()
        with patch.object(state, "get_hermes_home", return_value=tmp_path):
            identity = state.get_camofox_identity()
            assert "user_id" in identity
            assert "session_key" in identity
            assert identity["user_id"].startswith("hermes_")
            assert identity["session_key"].startswith("task_")


class TestCamofoxConfigDefaults:
    def test_default_config_includes_managed_persistence_toggle(self):
        from hermes_cli.config import DEFAULT_CONFIG

        browser_cfg = DEFAULT_CONFIG["browser"]
        assert browser_cfg["camofox"]["managed_persistence"] is False


class TestCamofoxIdentitySecret:
    def test_secret_file_is_created_and_reused(self, tmp_path):
        state = _load_module()
        with patch.object(state, "get_hermes_home", return_value=tmp_path):
            first = state.get_camofox_identity("task-1")
            secret_path = state.get_camofox_state_dir() / state.CAMOFOX_SECRET_FILE
            assert secret_path.exists()
            if os.name == "posix":
                assert stat.S_IMODE(secret_path.stat().st_mode) == 0o600
            second = state.get_camofox_identity("task-1")
            assert first == second

    def test_secret_differs_across_profiles(self, tmp_path):
        state = _load_module()

        with patch.object(state, "get_hermes_home", return_value=tmp_path / "a"):
            a_first = state.get_camofox_identity("task-1")
            a_secret_path = state.get_camofox_state_dir() / state.CAMOFOX_SECRET_FILE
            assert a_secret_path.exists()
            a_secret = a_secret_path.read_text()

        with patch.object(state, "get_hermes_home", return_value=tmp_path / "b"):
            b_first = state.get_camofox_identity("task-1")
            b_secret_path = state.get_camofox_state_dir() / state.CAMOFOX_SECRET_FILE
            assert b_secret_path.exists()
            b_secret = b_secret_path.read_text()

        assert a_secret != b_secret

        with patch.object(state, "get_hermes_home", return_value=tmp_path / "a"):
            a_second = state.get_camofox_identity("task-1")
        with patch.object(state, "get_hermes_home", return_value=tmp_path / "b"):
            b_second = state.get_camofox_identity("task-1")

        assert a_first == a_second
        assert b_first == b_second
