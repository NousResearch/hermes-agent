"""Behavior tests for file-search ordering and ripgrep selection."""

import json
import re

import pytest

from tools.environments.local import LocalEnvironment
from tools.file_operations import SearchResult, ShellFileOperations
from tools.file_tools import SEARCH_FILES_SCHEMA, _handle_search_files, search_tool


class RecordingEnvironment:
    is_local = False
    cwd = "/repo"

    def __init__(self, *, rg_output="/repo/one.py\n/repo/two.py\n", rg_code=0):
        self.commands = []
        self.rg_output = rg_output
        self.rg_code = rg_code

    def execute(self, command, **kwargs):
        self.commands.append(command)
        if command.startswith("test -e "):
            return {"output": "exists\n", "returncode": 0}
        if command.startswith("command -v rg"):
            return {"output": "/opt/Rip Grep/rg\n", "returncode": 0}
        if "--version" in command:
            return {"output": "ripgrep 14.1.1\n", "returncode": 0}
        if "--files" in command:
            return {"output": self.rg_output, "returncode": self.rg_code}
        return {"output": "", "returncode": 1}

    @property
    def rg_commands(self):
        return [command for command in self.commands if "--files" in command]


def test_schema_exposes_fast_discovery_default_and_exact_modified_opt_in():
    order = SEARCH_FILES_SCHEMA["parameters"]["properties"]["order"]

    assert order["enum"] == ["discovery", "modified"]
    assert order["default"] == "discovery"
    assert "fast bounded traversal order" in order["description"]
    assert "exact global newest-first" in order["description"]
    assert "ignored for content" in order["description"]


def test_default_file_search_runs_one_bounded_unsorted_rg_command():
    env = RecordingEnvironment()
    ops = ShellFileOperations(env)

    result = ops.search("*.py", path="/repo", target="files", limit=1, offset=1)

    assert result.files == ["/repo/two.py"]
    assert len(env.rg_commands) == 1
    assert "--sortr" not in env.rg_commands[0]
    assert "head -n 3" in env.rg_commands[0]


def test_modified_file_search_runs_one_exact_order_rg_command():
    env = RecordingEnvironment()
    ops = ShellFileOperations(env)

    result = ops.search("*.py", path="/repo", target="files", order="modified")

    assert result.error is None
    assert len(env.rg_commands) == 1
    assert "--sortr=modified" in env.rg_commands[0]


def test_modified_zero_match_exit_one_is_valid_without_capability_error():
    env = RecordingEnvironment(rg_output="", rg_code=1)
    result = ShellFileOperations(env).search(
        "*.missing", path="/repo", target="files", order="modified"
    )
    assert result.error is None
    assert result.files == []
    assert len(env.rg_commands) == 1


@pytest.mark.parametrize("version", ["ripgrep 13.0.0\n", "ripgrep unknown\n"])
def test_modified_requires_parseable_ripgrep_14_before_search(version):
    env = RecordingEnvironment()

    def execute(command, **kwargs):
        env.commands.append(command)
        if command.startswith("test -e "):
            return {"output": "exists\n", "returncode": 0}
        if command.startswith("command -v rg"):
            return {"output": "/opt/Rip Grep/rg\n", "returncode": 0}
        if "--version" in command:
            return {"output": version, "returncode": 0}
        raise AssertionError(f"search must not run: {command}")

    env.execute = execute
    ops = ShellFileOperations(env)
    first = ops.search("*.py", path="/repo", target="files", order="modified")
    second = ops.search("*.py", path="/repo", target="files", order="modified")
    assert "ripgrep 14" in (first.error or "").lower()
    assert second.error == first.error
    assert env.rg_commands == []
    assert len([c for c in env.commands if "--version" in c]) == 1


def test_empty_discovery_output_is_zero_matches_without_retry():
    env = RecordingEnvironment(rg_output="", rg_code=0)
    ops = ShellFileOperations(env)

    result = ops.search("*.missing", path="/repo", target="files")

    assert result.error is None
    assert result.files == []
    assert result.total_count == 0
    assert len(env.rg_commands) == 1


def test_modified_capability_failure_is_actionable_and_not_downgraded():
    env = RecordingEnvironment(rg_output="", rg_code=2)
    ops = ShellFileOperations(env)

    result = ops.search("*.py", path="/repo", target="files", order="modified")

    assert len(env.rg_commands) == 1
    assert result.error is not None
    assert "exact modification-time order" in result.error.lower()
    assert "ripgrep" in result.error


@pytest.mark.parametrize("order", ["discovery", "modified"])
def test_rg_partial_output_with_error_exit_fails_closed(order):
    env = RecordingEnvironment(rg_output="/repo/partial.py\n", rg_code=2)

    result = ShellFileOperations(env).search(
        "*.py", path="/repo", target="files", order=order
    )

    assert result.error is not None
    assert result.files == []
    assert len(env.rg_commands) == 1


def test_invalid_direct_file_order_returns_structured_error():
    env = RecordingEnvironment()
    ops = ShellFileOperations(env)

    result = ops.search("*.py", path="/repo", target="files", order="random")

    assert isinstance(result, SearchResult)
    assert result.error == "Invalid file search order 'random'; expected 'discovery' or 'modified'."
    assert env.rg_commands == []


def test_handler_forwards_modified_order(monkeypatch):
    captured = {}

    def fake_search_tool(**kwargs):
        captured.update(kwargs)
        return "{}"

    monkeypatch.setattr("tools.file_tools.search_tool", fake_search_tool)

    _handle_search_files({"pattern": "*.py", "target": "files", "order": "modified"})

    assert captured["order"] == "modified"


def test_repeated_search_key_distinguishes_order(monkeypatch):
    class StubOperations:
        def search(self, **kwargs):
            return SearchResult()

    monkeypatch.setattr("tools.file_tools._get_file_ops", lambda task_id: StubOperations())
    task_id = "engine-order-key"
    for _ in range(3):
        assert "BLOCKED" not in json.loads(
            search_tool("*.py", target="files", order="discovery", task_id=task_id)
        ).get("error", "")

    changed = json.loads(
        search_tool("*.py", target="files", order="modified", task_id=task_id)
    )

    assert "BLOCKED" not in changed.get("error", "")


class RipgrepInvocationEnvironment(RecordingEnvironment):
    def execute(self, command, **kwargs):
        self.commands.append(command)
        if command.startswith("test -e "):
            return {"output": "exists\n", "returncode": 0}
        if command.startswith("command -v rg"):
            return {"output": "/opt/Rip Grep/rg\n", "returncode": 0}
        if "--version" in command:
            return {"output": "ripgrep 14.1.1\n", "returncode": 0}
        if "--files" in command:
            return {"output": "/repo/a.py\n", "returncode": 0}
        if "--line-number" in command:
            return {"output": "/repo/a.py:1:needle\n", "returncode": 0}
        if "--count-matches" in command:
            return {"output": "", "returncode": 1}
        return {"output": "", "returncode": 1}


def test_resolved_executable_with_spaces_is_used_by_every_rg_invocation():
    env = RipgrepInvocationEnvironment()
    ops = ShellFileOperations(env)

    assert ops.search("*.py", path="/repo", target="files").files
    assert ops.search("needle", path="/repo", target="content").matches
    assert ops._zero_match_probe("absent", "/repo", None) is None

    invocations = [
        command for command in env.commands
        if any(flag in command for flag in ("--files", "--line-number", "--count-matches"))
    ]
    assert invocations
    assert all("'/opt/Rip Grep/rg'" in command for command in invocations)
    assert all(not re.search(r"(?:^|[; ])rg\s", command) for command in invocations)
    assert len([c for c in env.commands if c.startswith("command -v rg")]) == 1


def test_non_rg_command_cache_keeps_cached_misses_and_bool_values():
    env = RecordingEnvironment()
    ops = ShellFileOperations(env)
    assert ops._has_command("find") is False
    assert ops._has_command("find") is False
    assert ops._command_cache == {"find": False}
    assert len([c for c in env.commands if c.startswith("command -v find")]) == 1


@pytest.mark.windows_only
def test_off_path_windows_rg_miss_is_reprobed_then_success_is_cached(
    tmp_path, monkeypatch
):
    local_app_data = tmp_path / "Local Data"
    candidate = local_app_data / "Microsoft" / "WinGet" / "Links" / "rg.exe"
    monkeypatch.setenv("LOCALAPPDATA", str(local_app_data))
    monkeypatch.setenv("USERPROFILE", str(tmp_path / "User Profile"))
    monkeypatch.delenv("SCOOP", raising=False)
    ops = ShellFileOperations(LocalEnvironment(str(tmp_path)))
    probes = []

    def command_v_miss(command, **kwargs):
        probes.append(command)
        from tools.file_operations import ExecuteResult
        return ExecuteResult(stdout="", exit_code=1)

    monkeypatch.setattr(ops, "_exec", command_v_miss)

    assert ops._resolve_command("rg") is None
    candidate.parent.mkdir(parents=True)
    candidate.write_text("")
    expected = str(candidate).replace("\\", "/")
    assert ops._resolve_command("rg") == expected
    assert ops._resolve_command("rg") == expected
    assert probes == ["command -v rg 2>/dev/null", "command -v rg 2>/dev/null"]


def test_remote_resolution_never_probes_controller_host_paths(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "controller-local"))
    monkeypatch.setenv("USERPROFILE", str(tmp_path / "controller-user"))
    env = RecordingEnvironment()

    def miss(command, **kwargs):
        env.commands.append(command)
        return {"output": "", "returncode": 1}

    env.execute = miss
    ops = ShellFileOperations(env)

    assert ops._resolve_command("rg") is None
    assert env.commands == ["command -v rg 2>/dev/null"]
    assert str(tmp_path) not in env.commands[0]


@pytest.mark.windows_only
def test_remote_msys_shaped_executable_is_not_rewritten_as_controller_path():
    env = RecordingEnvironment()

    def execute(command, **kwargs):
        env.commands.append(command)
        if command.startswith("test -e "):
            return {"output": "exists\n", "returncode": 0}
        if command.startswith("command -v rg"):
            return {"output": "/c/remote-tools/rg\n", "returncode": 0}
        if "--files" in command:
            return {"output": "/repo/a.py\n", "returncode": 0}
        return {"output": "", "returncode": 1}

    env.execute = execute

    result = ShellFileOperations(env).search("*.py", path="/repo", target="files")

    assert result.files
    assert "'/c/remote-tools/rg' --files" in env.rg_commands[0]
    assert "C:/remote-tools/rg" not in env.rg_commands[0]


@pytest.mark.windows_only
def test_every_windows_drive_root_is_broad_even_when_home_is_on_another_drive(
    tmp_path, monkeypatch
):
    import tools.file_operations as file_operations

    monkeypatch.setattr(file_operations, "_HOME", "C:/Users/alice")
    ops = ShellFileOperations(LocalEnvironment(str(tmp_path)))

    assert ops._is_broad_local_search_root("D:/") is True
    assert ops._is_broad_local_search_root("D:/repo") is False


def test_modified_multi_path_search_preserves_exact_order_request():
    env = RecordingEnvironment()

    def execute(command, **kwargs):
        env.commands.append(command)
        if command.startswith("test -e "):
            if "'/one /two'" in command:
                return {"output": "not_found\n", "returncode": 0}
            return {"output": "exists\n", "returncode": 0}
        if command.startswith("command -v rg"):
            return {"output": "/usr/bin/rg\n", "returncode": 0}
        if "--version" in command:
            return {"output": "ripgrep 14.1.1\n", "returncode": 0}
        if "--files" in command:
            return {"output": "/two/new.py\n/one/old.py\n", "returncode": 0}
        return {"output": "", "returncode": 1}

    env.execute = execute
    result = ShellFileOperations(env).search(
        "*.py", path="/one /two", target="files", order="modified"
    )

    assert result.files == ["/two/new.py", "/one/old.py"]
    assert len(env.rg_commands) == 1
    assert "--sortr=modified" in env.rg_commands[0]
    assert "'/one'" in env.rg_commands[0]
    assert "'/two'" in env.rg_commands[0]


def test_multi_path_modified_capability_error_propagates():
    env = RecordingEnvironment()

    def execute(command, **kwargs):
        env.commands.append(command)
        if command.startswith("test -e "):
            output = "not_found\n" if "'/one /two'" in command else "exists\n"
            return {"output": output, "returncode": 0}
        if command.startswith("command -v rg"):
            return {"output": "/usr/bin/rg\n", "returncode": 0}
        if "--version" in command:
            return {"output": "ripgrep 13.0.0\n", "returncode": 0}
        raise AssertionError(command)

    env.execute = execute
    result = ShellFileOperations(env).search(
        "*.py", path="/one /two", target="files", order="modified"
    )
    assert "ripgrep 14" in (result.error or "").lower()
    assert env.rg_commands == []


def test_modified_timeout_preserves_partial_results_and_limit_reason():
    env = RecordingEnvironment(
        rg_output="/repo/partial.py\n[Command timed out after 60s]\n",
        rg_code=124,
    )

    result = ShellFileOperations(env).search(
        "*.py", path="/repo", target="files", order="modified"
    )

    assert result.files == ["/repo/partial.py"]
    assert result.truncated is True
    assert result.limit_reason == "search_timeout"


def test_order_is_ignored_for_content_search():
    env = RipgrepInvocationEnvironment()

    result = ShellFileOperations(env).search(
        "needle", path="/repo", target="content", order="not-a-file-order"
    )

    assert result.error is None
    assert result.matches
