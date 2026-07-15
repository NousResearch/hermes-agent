from unittest.mock import MagicMock

import pytest

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


def test_rg_count_timeout_returns_partial_counts(ops, monkeypatch):
    ops.env.execute.side_effect = path_exists_or(timeout_output("src/a.py:3", "src/b.py:5"))
    monkeypatch.setattr(ops, "_has_command", lambda cmd: cmd == "rg")

    result = ops.search("foo", path="/big", target="content", output_mode="count")

    assert_timed_out(result)
    assert result.counts == {"src/a.py": 3, "src/b.py": 5}


def test_rg_file_timeout_does_not_retry_unsorted(ops, monkeypatch):
    calls = 0

    def execute(command, **kwargs):
        nonlocal calls
        if "test -e" in command:
            return {"output": "exists", "returncode": 0}
        calls += 1
        return {"output": timeout_output(), "returncode": 124}

    ops.env.execute.side_effect = execute
    monkeypatch.setattr(ops, "_has_command", lambda cmd: cmd == "rg")

    result = ops.search("*.py", path="/big", target="files")

    assert calls == 1
    assert_timed_out(result)
    assert result.files == []


def test_grep_timeout_returns_partial_match(ops, monkeypatch):
    ops.env.execute.side_effect = path_exists_or(timeout_output("src/a.py:10:foo"))
    monkeypatch.setattr(ops, "_has_command", lambda cmd: cmd == "grep")

    result = ops.search("foo", path="/big", target="content")

    assert_timed_out(result)
    assert [match.path for match in result.matches] == ["src/a.py"]


def test_find_timeout_returns_partial_files_and_does_not_retry(ops, monkeypatch):
    calls = 0

    def execute(command, **kwargs):
        nonlocal calls
        if "test -e" in command:
            return {"output": "exists", "returncode": 0}
        calls += 1
        return {"output": timeout_output("1700000000.0 /big/a.py"), "returncode": 124}

    ops.env.execute.side_effect = execute
    monkeypatch.setattr(ops, "_has_command", lambda cmd: cmd == "find")

    result = ops.search("*.py", path="/big", target="files")

    assert calls == 1
    assert_timed_out(result)
    assert result.files == ["/big/a.py"]


def test_real_rg_error_still_hard_fails(ops, monkeypatch):
    ops.env.execute.side_effect = path_exists_or("rg: regex parse error:", returncode=2)
    monkeypatch.setattr(ops, "_has_command", lambda cmd: cmd == "rg")

    result = ops.search("[", path="/big", target="content")

    assert result.error == "Search failed: rg: regex parse error:"
    assert result.limit_reason is None
