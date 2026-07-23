"""Terminal executions must not leave pre-command proof reusable."""

import json
from pathlib import Path

import pytest

from agent.verification_evidence import record_terminal_result, verification_status
import tools.terminal_tool as terminal_tool


class _FakeEnvironment:
    env = {}

    def __init__(
        self, cwd: Path, returncode: int = 0, failure: Exception | None = None
    ) -> None:
        self.cwd = str(cwd)
        self.returncode = returncode
        self.failure = failure

    def execute(self, _command: str, **_kwargs: object) -> dict[str, object]:
        if self.failure is not None:
            raise self.failure
        return {"output": "terminal completed", "returncode": self.returncode}


def _node_project(root: Path) -> None:
    (root / "package.json").write_text(
        json.dumps({"scripts": {"test": "vitest", "lint": "eslint ."}})
    )
    (root / "pnpm-lock.yaml").write_text("")


def _run_foreground(
    monkeypatch: pytest.MonkeyPatch,
    root: Path,
    *,
    command: str,
    returncode: int = 0,
    failure: Exception | None = None,
) -> dict[str, object]:
    environment = _FakeEnvironment(root, returncode=returncode, failure=failure)
    monkeypatch.setattr(terminal_tool, "_active_environments", {"default": environment})
    monkeypatch.setattr(terminal_tool, "_last_activity", {"default": 0.0})
    monkeypatch.setattr(terminal_tool, "_task_env_overrides", {})
    monkeypatch.setattr(
        terminal_tool,
        "_get_env_config",
        lambda: {
            "env_type": "local",
            "cwd": str(root),
            "timeout": 60,
            "lifetime_seconds": 3600,
        },
    )

    return json.loads(
        terminal_tool.terminal_tool(
            command=command,
            session_id="session-1",
            workdir=str(root),
            force=True,
        )
    )


@pytest.mark.parametrize("returncode", [0, 1])
def test_foreground_terminal_completion_stales_prior_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, returncode: int
) -> None:
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    _node_project(tmp_path)
    record_terminal_result(
        command="pnpm test",
        cwd=tmp_path,
        session_id="session-1",
        exit_code=0,
        output="all green",
    )
    assert verification_status(session_id="session-1", cwd=tmp_path)["status"] == "passed"

    result = _run_foreground(
        monkeypatch, tmp_path, command="python formatter.py", returncode=returncode
    )

    assert result["exit_code"] == returncode
    assert verification_status(session_id="session-1", cwd=tmp_path)["status"] == "stale"


@pytest.mark.parametrize(
    ("failure", "exit_code"),
    [
        (TimeoutError("backend timeout"), 124),
        (RuntimeError("backend disconnected"), -1),
    ],
)
def test_foreground_terminal_error_stales_prior_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: Exception,
    exit_code: int,
) -> None:
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    monkeypatch.setattr(terminal_tool.time, "sleep", lambda *_args, **_kwargs: None)
    _node_project(tmp_path)
    record_terminal_result(
        command="pnpm test",
        cwd=tmp_path,
        session_id="session-1",
        exit_code=0,
        output="all green",
    )

    result = _run_foreground(
        monkeypatch,
        tmp_path,
        command="python formatter.py",
        failure=failure,
    )

    assert result["exit_code"] == exit_code
    assert verification_status(session_id="session-1", cwd=tmp_path)["status"] == "stale"


def test_fresh_verification_after_foreground_marker_is_reusable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    _node_project(tmp_path)
    record_terminal_result(
        command="pnpm test",
        cwd=tmp_path,
        session_id="session-1",
        exit_code=0,
        output="all green",
    )
    _run_foreground(monkeypatch, tmp_path, command="python formatter.py", returncode=0)

    result = _run_foreground(monkeypatch, tmp_path, command="pnpm test", returncode=0)

    assert result["verification_evidence"]["status"] == "passed"
    assert verification_status(session_id="session-1", cwd=tmp_path)["status"] == "passed"
