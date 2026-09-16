"""Plugin backend kanban identity forwarding (issue #113304).

_build_plugin_env is the runtime's only bridge to plugin terminal backends. A plugin
that gates on the dispatcher's Kanban identity declares kanban_task_id on
create_environment(); the bridge must forward HERMES_KANBAN_TASK so the guard can
compare it against the task it was asked to serve.
"""

from types import SimpleNamespace
from unittest.mock import patch

import tools.terminal_tool_backends as ttb


def _fake_provider(seen):
    def create_environment(**kwargs):
        seen.update(kwargs)
        return SimpleNamespace()
    return SimpleNamespace(name="gating", create_environment=create_environment)


def _build(seen, **kw):
    with patch.object(ttb, "_get_plugin_env_provider", return_value=_fake_provider(seen)):
        return ttb._build_plugin_env(env_type="gating", image="img", cwd="/tmp", timeout=30,
                                     cc={}, task_id="default", **kw)


def test_kanban_task_id_forwarded_when_exported(monkeypatch):
    """HERMES_KANBAN_TASK set by the dispatcher reaches the plugin's create_environment."""
    monkeypatch.setenv("HERMES_KANBAN_TASK", "task-123")
    seen = {}
    _build(seen)
    assert seen["kanban_task_id"] == "task-123"


def test_kanban_task_id_none_when_not_exported(monkeypatch):
    """No Kanban context: the plugin gets None and falls back to its own binding rule."""
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    seen = {}
    _build(seen)
    assert seen["kanban_task_id"] is None


def test_existing_kwargs_untouched(monkeypatch):
    """The identity kwarg is additive; the rest of the bridge contract is unchanged."""
    monkeypatch.setenv("HERMES_KANBAN_TASK", "task-123")
    seen = {}
    _build(seen)
    assert seen["cwd"] == "/tmp" and seen["timeout"] == 30 and seen["task_id"] == "default"
    assert seen["image"] == "img" and seen["container_config"] == {}
