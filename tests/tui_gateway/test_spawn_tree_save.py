"""Regression tests for spawn_tree.save filename timestamp handling."""

import time
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture()
def server():
    with patch.dict(
        "sys.modules",
        {
            "hermes_constants": MagicMock(
                get_hermes_home=MagicMock(return_value="/tmp/hermes_test")
            ),
            "hermes_cli.env_loader": MagicMock(),
            "hermes_cli.banner": MagicMock(),
            "hermes_state": MagicMock(),
        },
    ):
        import importlib

        mod = importlib.import_module("tui_gateway.server")
        yield mod
        mod._sessions.clear()
        mod._pending.clear()
        mod._answers.clear()
        mod._methods.clear()
        importlib.reload(mod)


def test_spawn_tree_save_filename_uses_utc(server, tmp_path, monkeypatch):
    """spawn_tree.save must render its filename timestamp in UTC regardless
    of the runner's local timezone.

    Replacing the deprecated ``datetime.utcfromtimestamp()`` with a naive
    ``datetime.fromtimestamp()`` (no ``tz=timezone.utc``) silently produces
    different filenames in non-UTC environments. Pinning the UTC contract
    here means that pseudo-fix would fail this test rather than diverging
    silently across environments.
    """
    monkeypatch.setenv("TZ", "America/New_York")
    if hasattr(time, "tzset"):
        time.tzset()

    monkeypatch.setattr(server, "_spawn_tree_session_dir", lambda sid: tmp_path)
    monkeypatch.setattr(server, "_append_spawn_tree_index", lambda d, e: None)

    handler = server._methods["spawn_tree.save"]
    finished_at = 1700000000.0  # 2023-11-14T22:13:20Z (17:13:20 EST)
    response = handler(
        1,
        {
            "session_id": "test",
            "subagents": [{"name": "a"}],
            "finished_at": finished_at,
        },
    )

    assert "result" in response, response
    expected_filename = "20231114T221320.json"
    assert response["result"]["path"].endswith("/" + expected_filename)
    assert (tmp_path / expected_filename).exists()
