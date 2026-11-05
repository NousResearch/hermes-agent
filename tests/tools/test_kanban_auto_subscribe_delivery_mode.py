"""New task auto-subscriptions honor the profile's configured delivery policy."""

import json
from pathlib import Path

import pytest


@pytest.fixture
def subscription_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_DB", str(home / "kanban.db"))
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    return home


@pytest.mark.parametrize("config, expected, warns", [
    ("kanban: {}\n", "notify+wake", False),
    ("kanban:\n  auto_subscribe_delivery_mode: notify+wake\n", "notify+wake", False),
    ("kanban:\n  auto_subscribe_delivery_mode: wake\n", "wake", False),
    ("kanban:\n  auto_subscribe_delivery_mode: notify\n", "notify", False),
    ("kanban:\n  auto_subscribe_delivery_mode: invalid\n", "notify+wake", True),
    ("kanban:\n  auto_subscribe_delivery_mode: null\n", "notify+wake", True),
    ("kanban:\n  auto_subscribe_delivery_mode: [wake]\n", "notify+wake", True),
    ("kanban: [\n", "notify+wake", True),
])
def test_create_subscription_delivery_mode(subscription_home, config, expected, warns, caplog):
    from gateway.session_context import clear_session_vars, set_session_vars
    from hermes_cli import kanban_db_connect as kbc, kanban_db_notify as kbn
    from tools import kanban_tools  # noqa: F401 -- registers the real tool handler
    from tools.registry import registry

    (subscription_home / "config.yaml").write_text(config)
    tokens = set_session_vars(platform="telegram", chat_id="chat", profile="default")
    try:
        result = json.loads(registry.dispatch("kanban_create", {"title": "child", "assignee": "peer"}))
    finally:
        clear_session_vars(tokens)
    assert result["ok"] and result["subscribed"], result
    with kbc.connect() as conn:
        sub = kbn.list_notify_subs(conn, result["task_id"])[0]
    assert sub["delivery_mode"] == expected
    assert ("kanban.auto_subscribe_delivery_mode" in caplog.text) == warns


def test_tui_delivery_mode_is_unchanged(subscription_home):
    from gateway.session_context import clear_session_vars, set_session_vars
    from hermes_cli import kanban_db_connect as kbc, kanban_db_notify as kbn
    from tools import kanban_tools
    from tools.registry import registry

    (subscription_home / "config.yaml").write_text("kanban:\n  auto_subscribe_delivery_mode: wake\n")
    tokens = set_session_vars(session_key="tui-origin", profile="default")
    try:
        assert kanban_tools._resolve_notify_target()["delivery_mode"] is None
        result = json.loads(registry.dispatch("kanban_create", {"title": "child", "assignee": "peer"}))
    finally:
        clear_session_vars(tokens)
    assert result["ok"] and result["subscribed"], result
    with kbc.connect() as conn:
        sub = kbn.list_notify_subs(conn, result["task_id"])[0]
    assert sub["platform"] == "tui"
    assert sub["delivery_mode"] == "notify"
