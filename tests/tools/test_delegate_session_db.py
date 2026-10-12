"""Child persistence binds only to a concrete parent database path."""

from pathlib import Path
from unittest.mock import MagicMock, patch

from tools.delegate_tool import _open_child_session_db


def test_mock_generated_parent_path_does_not_open_database():
    parent = MagicMock()
    with patch("hermes_state_registry.acquire") as acquire:
        assert _open_child_session_db(parent) is None
    acquire.assert_not_called()


def test_concrete_parent_path_is_reused(tmp_path):
    parent = MagicMock()
    parent._session_db.db_path = tmp_path / "state.db"
    with patch("hermes_state_registry.acquire") as acquire:
        assert _open_child_session_db(parent) is acquire.return_value
    acquire.assert_called_once_with(Path(parent._session_db.db_path))
