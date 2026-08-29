from unittest.mock import MagicMock

import pytest

import tools.file_operations as file_operations
from tools.environments.local import LocalEnvironment
from tools.file_operations import ExecuteResult, ShellFileOperations, _search_stdout_and_limit


TIMEOUT = "[Command timed out after 60s]"


@pytest.fixture()
def ops():
    env = MagicMock(cwd="/tmp/test")
    env.execute.return_value = {"output": "", "returncode": 0}
    return ShellFileOperations(env)


def timeout_output(*lines: str) -> str:
    return "\n".join([*lines, TIMEOUT])


def path_exists_or(output: str, returncode: int = 124):
    def execute(command, **kwargs):
        if "test -e" in command:
            return {"output": "exists", "returncode": 0}
        return {"output": output, "returncode": returncode}

    return execute


def assert_timed_out(result):
    assert result.error is None
    assert result.truncated is True
    assert result.limit_reason == "search_timeout"
    assert result.to_dict()["limit_reason"] == "search_timeout"


def test_timeout_helper_strips_only_trailing_marker():
    assert _search_stdout_and_limit(ExecuteResult(timeout_output("a.py"), 124)) == ("a.py", "search_timeout")
    assert _search_stdout_and_limit(ExecuteResult("a.py\nnot a marker", 0)) == ("a.py\nnot a marker", None)


@pytest.mark.parametrize(
    ("target", "output_mode", "raw", "expected"),
    [
        ("files", "content", timeout_output("src/a.py", "src/b.py"), ["src/a.py", "src/b.py"]),
        ("content", "files_only", timeout_output("src/a.py", "src/b.py"), ["src/a.py", "src/b.py"]),
        ("content", "content", timeout_output("src/a.py:10:foo", "src/b.py:20:foo"), ["src/a.py", "src/b.py"]),
    ],
)
def test_rg_timeout_returns_partial_results_without_marker(ops, monkeypatch, target, output_mode, raw, expected):
    ops.env.execute.side_effect = path_exists_or(raw)
    monkeypatch.setattr(ops, "_has_command", lambda cmd: cmd == "rg")

    result = ops.search("foo", path="/big", target=target, output_mode=output_mode)

    assert_timed_out(result)
    if target == "content" and output_mode == "content":
        assert [match.path for match in result.matches] == expected
        assert all("timed out" not in match.content for match in result.matches)
    else:
        assert result.files == expected
        assert all("timed out" not in path for path in result.files)


def test_real_rg_error_still_hard_fails(ops, monkeypatch):
    ops.env.execute.side_effect = path_exists_or("rg: regex parse error:", returncode=2)
    monkeypatch.setattr(ops, "_has_command", lambda cmd: cmd == "rg")

    result = ops.search("[", path="/big", target="content")

    assert result.error == "Search failed: rg: regex parse error:"
    assert result.limit_reason is None


class FindRecordingEnvironment:
    is_local = False
    cwd = "/narrow"

    def __init__(self, output="", code=0):
        self.output = output
        self.code = code
        self.commands = []

    def execute(self, command, **kwargs):
        self.commands.append((command, kwargs))
        if command.startswith("command -v find"):
            return {"output": "yes\n", "returncode": 0}
        if command.startswith("command -v rg"):
            return {"output": "", "returncode": 1}
        if "find " in command:
            return {"output": self.output, "returncode": self.code}
        return {"output": "", "returncode": 1}

    @property
    def find_commands(self):
        return [
            item for item in self.commands
            if item[0].startswith("find ") or "; find " in item[0]
        ]


class MultiRootFindEnvironment(FindRecordingEnvironment):
    def execute(self, command, **kwargs):
        self.commands.append((command, kwargs))
        if command.startswith("test -e "):
            output = "not_found\n" if "'/one/.hidden /two/.cache'" in command else "exists\n"
            return {"output": output, "returncode": 0}
        if command.startswith("command -v find"):
            return {"output": "yes\n", "returncode": 0}
        if command.startswith("command -v rg"):
            return {"output": "", "returncode": 1}
        if "find " in command:
            return {"output": self.output, "returncode": self.code}
        return {"output": "", "returncode": 1}


def test_find_discovery_is_one_unsorted_pruned_bounded_scan():
    env = FindRecordingEnvironment("/narrow/a.py\n/narrow/b.py\n/narrow/c.py\n/narrow/d.py\n")
    result = ShellFileOperations(env)._search_files(
        "*.py", "/narrow", limit=2, offset=1, order="discovery"
    )
    assert result.files == ["/narrow/b.py", "/narrow/c.py"]
    assert result.truncated is True
    assert len(env.find_commands) == 1
    command, kwargs = env.find_commands[0]
    assert "-printf" not in command
    assert "sort " not in command
    assert "-prune" in command
    assert "head -n 4" in command
    assert kwargs["timeout"] <= 60


def test_find_modified_is_one_exact_scan_without_bsd_retry():
    env = FindRecordingEnvironment("30 /narrow/new.py\n20 /narrow/mid.py\n10 /narrow/old.py\n")
    result = ShellFileOperations(env)._search_files(
        "*.py", "/narrow", limit=1, offset=1, order="modified"
    )
    assert result.files == ["/narrow/mid.py"]
    assert result.truncated is True
    assert len(env.find_commands) == 1
    command, _ = env.find_commands[0]
    assert "-printf '%T@ %p\\n'" in command
    assert "sort -rn" in command
    assert "head -n 3" in command


def test_no_rg_multi_root_modified_is_one_globally_sorted_scan():
    env = MultiRootFindEnvironment(
        "30 /two/.cache/new.py\n10 /one/.hidden/old.py\n"
    )

    result = ShellFileOperations(env).search(
        "*.py",
        path="/one/.hidden /two/.cache",
        target="files",
        order="modified",
        limit=1,
    )

    assert result.error is None
    assert result.files == ["/two/.cache/new.py"]
    assert result.truncated is True
    assert len(env.find_commands) == 1
    command, kwargs = env.find_commands[0]
    assert "find '/one/.hidden' '/two/.cache'" in command
    assert "sort -rn" in command
    assert "head -n 2" in command
    assert "! -path '/one/.hidden'" in command
    assert "! -path '/two/.cache'" in command
    assert kwargs["timeout"] <= 60


def test_find_modified_capability_failure_is_actionable_without_retry():
    env = FindRecordingEnvironment("", code=1)
    result = ShellFileOperations(env)._search_files(
        "*.py", "/narrow", limit=2, offset=0, order="modified"
    )
    assert "modification-time" in (result.error or "")
    assert len(env.find_commands) == 1


@pytest.mark.parametrize(
    ("order", "output", "error_fragment"),
    [
        ("discovery", "/narrow/partial.py\n", "bounded find traversal"),
        ("modified", "/narrow/not-a-timestamp.py\n", "modification-time"),
    ],
)
def test_find_hard_error_discards_partial_output(order, output, error_fragment):
    env = FindRecordingEnvironment(output, code=2)

    result = ShellFileOperations(env)._search_files(
        "*.py", "/narrow", limit=2, offset=0, order=order
    )

    assert error_fragment in (result.error or "")
    assert result.files == []
    assert result.total_count == 0


def test_find_timeout_preserves_partial_results_and_limit_reason():
    env = FindRecordingEnvironment(timeout_output("/narrow/partial.py"), code=124)

    result = ShellFileOperations(env)._search_files(
        "*.py", "/narrow", limit=2, offset=0, order="discovery"
    )

    assert result.files == ["/narrow/partial.py"]
    assert_timed_out(result)


def test_find_zero_match_exit_zero_is_success():
    result = ShellFileOperations(FindRecordingEnvironment("", code=0))._search_files(
        "*.missing", "/narrow", limit=2, offset=0, order="discovery"
    )

    assert result.error is None
    assert result.files == []


def test_local_broad_no_rg_refuses_before_find(monkeypatch, tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    ops = ShellFileOperations(LocalEnvironment(str(home)))
    monkeypatch.setattr(file_operations, "_HOME", str(home))
    monkeypatch.setattr(file_operations.os.path, "isfile", lambda path: False)
    commands = []

    def fake_exec(command, **kwargs):
        commands.append((command, kwargs))
        if command.startswith("command -v rg"):
            return ExecuteResult("", 1)
        if command.startswith("command -v find"):
            return ExecuteResult("yes\n", 0)
        raise AssertionError(f"broad fallback must not execute: {command}")

    monkeypatch.setattr(ops, "_exec", fake_exec)
    result = ops._search_files("*.py", str(home), 10, 0, "discovery")
    assert "ripgrep" in (result.error or "").lower()
    assert not any(command.startswith("find ") for command, _ in commands)
