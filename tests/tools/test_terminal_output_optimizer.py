from pathlib import Path

from tools.terminal_output_optimizer import classify_command, optimize_terminal_output


class FakeEnv:
    def __init__(self, tmp_path: Path):
        self.tmp_path = tmp_path
        self.writes = {}

    def execute(self, command, timeout=30, stdin_data=None):
        assert stdin_data is not None
        assert "umask 077" in command
        assert "cat >" in command
        # The production path writes through shell quoting. For these tests,
        # it is enough to prove the optimizer requested a permission-restricted
        # sandbox write.
        self.writes[command] = stdin_data
        return {"returncode": 0, "output": ""}


def test_disabled_mode_is_exact_passthrough():
    output = "line\n" * 2000
    result = optimize_terminal_output(
        command="pytest -q",
        output=output,
        returncode=0,
        config={"enabled": False, "min_chars": 10},
    )
    assert result.output == output
    assert result.optimized is False
    assert result.reason == "disabled"


def test_below_min_chars_is_passthrough():
    result = optimize_terminal_output(
        command="git status",
        output="short output",
        config={"enabled": True, "min_chars": 1000},
    )
    assert result.output == "short output"
    assert result.optimized is False
    assert result.reason == "below_min_chars"


def test_classify_common_commands():
    assert classify_command("git status --short") == "git_status"
    assert classify_command("cd repo && git diff") == "git_diff"
    assert classify_command("pytest -q") == "test"
    assert classify_command("pnpm test") == "test"
    assert classify_command("rg pattern .") == "search"


def test_generic_repeated_output_collapses_with_raw_preservation(tmp_path):
    output = ("same noisy line\n" * 600) + ("tail detail\n" * 100)
    result = optimize_terminal_output(
        command="some noisy command",
        output=output,
        returncode=0,
        config={
            "enabled": True,
            "min_chars": 100,
            "target_chars": 1000,
            "raw_output": "preserve",
            "storage_dir": str(tmp_path),
        },
    )
    assert result.optimized is True
    assert "<terminal-output-optimized>" in result.output
    assert "raw_sanitized_output:" in result.output
    assert "repeated" in result.output
    assert len(result.output) < len(output)
    raw_path = Path(result.raw_path)
    assert raw_path.read_text() == output
    assert oct(raw_path.stat().st_mode & 0o777) == "0o600"
    assert oct(tmp_path.stat().st_mode & 0o777) == "0o700"


def test_raw_preserve_failure_fails_open():
    class FailingEnv:
        def execute(self, *args, **kwargs):
            return {"returncode": 1, "output": "nope"}

    output = "line\n" * 1000
    result = optimize_terminal_output(
        command="pytest -q",
        output=output,
        env=FailingEnv(),
        config={"enabled": True, "min_chars": 10, "target_chars": 1000, "raw_output": "preserve"},
    )
    assert result.output == output
    assert result.optimized is False
    assert result.reason == "raw_preserve_failed"


def test_test_output_preserves_failures_and_summary(tmp_path):
    output = "\n".join(
        [f"progress {i}" for i in range(500)]
        + ["FAILED tests/test_x.py::test_case - AssertionError", "short test summary info", "1 failed, 20 passed"]
        + [f"tail {i}" for i in range(100)]
    )
    result = optimize_terminal_output(
        command="pytest -q",
        output=output,
        returncode=1,
        config={
            "enabled": True,
            "min_chars": 100,
            "target_chars": 1000,
            "raw_output": "preserve",
            "storage_dir": str(tmp_path),
        },
    )
    assert result.optimized is True
    assert "FAILED tests/test_x.py::test_case" in result.output
    assert "short test summary" in result.output
    assert "1 failed" in result.output
    assert "exit_code: 1" in result.output
