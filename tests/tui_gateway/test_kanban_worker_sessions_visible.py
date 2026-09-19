"""`kanban.show_worker_sessions` decides whether board workers reach the sidebar tree.

Contract, not snapshot: cron is always excluded; kanban is excluded exactly when the
setting is off. The sidebar's own kanban lane still folds the rows, so "visible" means
reachable, not one row per throwaway run.
"""

from __future__ import annotations

import pytest

from tui_gateway import methods_projects as mp


@pytest.fixture
def kanban_cfg(monkeypatch):
    """Drive the setting through the real config reader the helper calls."""

    def _set(value):
        monkeypatch.setattr(
            "hermes_cli.config.load_config_readonly",
            lambda: {"kanban": {} if value is None else {"show_worker_sessions": value}},
        )

    return _set


def test_workers_hidden_by_default(kanban_cfg):
    kanban_cfg(None)
    assert "kanban" in mp._project_tree_excluded_sources()


def test_workers_visible_when_enabled(kanban_cfg):
    kanban_cfg(True)
    assert "kanban" not in mp._project_tree_excluded_sources()


def test_cron_stays_excluded_either_way(kanban_cfg):
    for value in (None, True, False):
        kanban_cfg(value)
        assert "cron" in mp._project_tree_excluded_sources()


def test_unreadable_config_hides_workers(monkeypatch):
    """Fail closed: a config the reader cannot open must not flood the sidebar."""

    def _boom():
        raise OSError("config unreadable")

    monkeypatch.setattr("hermes_cli.config.load_config_readonly", _boom)
    assert "kanban" in mp._project_tree_excluded_sources()


def test_module_constant_is_not_mutated(kanban_cfg):
    """The helper returns a fresh list; flipping the setting must not edit shared state."""
    kanban_cfg(True)
    mp._project_tree_excluded_sources()
    assert mp._PROJECT_TREE_EXCLUDED_SOURCES == ["cron", "kanban"]
