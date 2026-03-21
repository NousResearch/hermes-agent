import importlib
from pathlib import Path


terminal_tool_module = importlib.import_module("tools.terminal_tool")


class _DummyBackend:
    pass


class _DummyEnv:
    def __init__(self, cwd=None, backend=None):
        self.cwd = cwd
        self.backend = backend


def test_get_active_environments_info_includes_workdirs_and_backends(monkeypatch, tmp_path):
    monkeypatch.setattr(
        terminal_tool_module,
        "_active_environments",
        {
            "task-alpha-1234": _DummyEnv(cwd="/tmp/work-a", backend="local"),
            "task-beta-5678": _DummyEnv(cwd="/tmp/work-b", backend=_DummyBackend),
        },
    )
    monkeypatch.setattr(terminal_tool_module, "_get_scratch_dir", lambda: Path(tmp_path))

    info = terminal_tool_module.get_active_environments_info()

    assert info["count"] == 2
    assert info["workdirs"]["task-alpha-1234"] == "/tmp/work-a"
    assert info["workdirs"]["task-beta-5678"] == "/tmp/work-b"
    assert info["backends"]["task-alpha-1234"] == "local"
    assert info["backends"]["task-beta-5678"] == "_DummyBackend"


def test_get_active_environments_info_handles_non_string_task_ids(monkeypatch, tmp_path):
    class NonStringTaskId:
        def __str__(self):
            return "non-string-id-42"

    task_id = NonStringTaskId()
    monkeypatch.setattr(
        terminal_tool_module,
        "_active_environments",
        {task_id: _DummyEnv(cwd="/tmp/work-c", backend="local")},
    )
    monkeypatch.setattr(terminal_tool_module, "_get_scratch_dir", lambda: Path(tmp_path))

    info = terminal_tool_module.get_active_environments_info()

    assert info["count"] == 1
    assert info["workdirs"][task_id] == "/tmp/work-c"
    assert info["backends"][task_id] == "local"
