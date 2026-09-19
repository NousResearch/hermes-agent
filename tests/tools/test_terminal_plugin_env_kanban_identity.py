"""The plugin-terminal bridge forwards the Kanban worker identity to plugin backends (issue #113304).

A Kanban worker's identity exists only as the dispatcher-exported ``HERMES_KANBAN_TASK`` env; the
bridge used to call ``create_environment()`` without it, so a backend gating on that identity fell
back to comparing against the terminal cache key (``"default"``/``"session:*"``) and rejected
every command the worker issued."""

import pytest

import tools.terminal_tool_backends as ttb
from agent import terminal_env_registry as reg
from agent.terminal_env_provider import TerminalEnvironmentProvider


class _Env:
    def execute(self, command, **kwargs):
        return {"returncode": 0, "output": ""}

    def cleanup(self):
        pass


class _RecordingProvider(TerminalEnvironmentProvider):
    name = "recordingbox"
    display_name = "RecordingBox"

    def __init__(self):
        self.calls = []

    def is_available(self):
        return True

    def create_environment(self, *, cwd, timeout, task_id="default", image=None,
                           container_config=None, kanban_task_id=None, **kwargs):
        self.calls.append({"task_id": task_id, "kanban_task_id": kanban_task_id})
        return _Env()


@pytest.fixture(autouse=True)
def _clean_registry():
    reg._reset_for_tests()
    yield
    reg._reset_for_tests()


def test_kanban_worker_identity_reaches_plugin_backend(monkeypatch):
    provider = _RecordingProvider()
    reg.register_provider(provider)
    monkeypatch.setenv("HERMES_KANBAN_TASK", "KT-113304")

    env = ttb._create_environment("recordingbox", image=None, cwd="/tmp", timeout=5,
                                  task_id="session:20260916_153928")

    assert isinstance(env, _Env)
    assert provider.calls == [{"task_id": "session:20260916_153928", "kanban_task_id": "KT-113304"}]
    assert env._hermes_backend_name == "recordingbox"


def test_no_kanban_context_passes_none(monkeypatch):
    provider = _RecordingProvider()
    reg.register_provider(provider)
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)

    ttb._create_environment("recordingbox", image=None, cwd="/tmp", timeout=5, task_id="default")

    assert provider.calls == [{"task_id": "default", "kanban_task_id": None}]
