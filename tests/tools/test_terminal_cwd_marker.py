from __future__ import annotations

from tools.environments.base import BaseEnvironment


class DummyEnvironment(BaseEnvironment):
    def _run_bash(self, *args, **kwargs):  # pragma: no cover - not used
        raise NotImplementedError

    def cleanup(self):  # pragma: no cover - not used
        pass


def test_wrap_command_creates_marker_parent_before_cwd_file_redirection():
    env = DummyEnvironment(cwd="/work/tree", timeout=10)
    env._snapshot_ready = True
    env._cwd_file = "/missing/tmp/hermes-cwd-test.txt"
    env._snapshot_path = "/missing/tmp/hermes-snap-test.sh"

    wrapped = env._wrap_command("printf ok", "/work/tree")

    assert "mkdir -p /missing/tmp" in wrapped
    assert "pwd -P > /missing/tmp/hermes-cwd-test.txt" in wrapped


def test_wrap_command_translates_windows_artifact_paths_for_bash(monkeypatch):
    monkeypatch.setattr("tools.path_translation.is_windows_host", lambda: True)
    env = DummyEnvironment(cwd="/work/tree", timeout=10)
    env._snapshot_ready = True
    env._cwd_file = r"C:\Users\Admin\AppData\Local\Temp\hermes-cwd-test.txt"
    env._snapshot_path = r"C:\Users\Admin\AppData\Local\Temp\hermes-snap-test.sh"

    wrapped = env._wrap_command("printf ok", "/work/tree")

    assert "C:" not in wrapped
    assert "mkdir -p /c/Users/Admin/AppData/Local/Temp" in wrapped
    assert "pwd -P > /c/Users/Admin/AppData/Local/Temp/hermes-cwd-test.txt" in wrapped


def test_extract_cwd_strips_only_internal_marker_warning():
    env = DummyEnvironment(cwd="/old", timeout=10)
    marker = env._cwd_marker
    result = {
        "output": (
            "real stderr stays\n"
            "/tmp/hermes-cwd-test.txt: No such file or directory\n"
            f"{marker}/new{marker}\n"
        ),
        "returncode": 0,
    }

    env._extract_cwd_from_output(result)

    assert env.cwd == "/new"
    assert "hermes-cwd-test.txt" not in result["output"]
    assert "real stderr stays" in result["output"]
    assert "No such file or directory" not in result["output"]
