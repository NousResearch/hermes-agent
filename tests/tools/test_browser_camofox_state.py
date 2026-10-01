"""Tests for Hermes-managed Camofox state helpers."""

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




