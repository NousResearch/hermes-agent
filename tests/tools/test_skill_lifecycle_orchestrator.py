"""End-to-end skill lifecycle orchestration tests."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.skill_lifecycle_orchestrator import (
    SkillTestRequest,
    TestExecutionResult as ExecutionResult,
    run_skill_lifecycle,
)
from tools.skill_test_sandbox import BubblewrapTestExecutor


def _skill(tmp_path: Path, *, with_tests: bool = True) -> Path:
    skill_dir = tmp_path / "demo-skill"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\nname: demo-skill\ndescription: Demo lifecycle skill.\n---\n\n# Demo\n",
        encoding="utf-8",
    )
    if with_tests:
        tests_dir = skill_dir / "tests"
        tests_dir.mkdir()
        (tests_dir / "test_demo.py").write_text(
            "def test_demo():\n    assert True\n",
            encoding="utf-8",
        )
    return skill_dir


def test_passing_tests_are_executed_and_registered(tmp_path: Path) -> None:
    skill_dir = _skill(tmp_path)
    requests: list[SkillTestRequest] = []

    def execute(request: SkillTestRequest) -> ExecutionResult:
        requests.append(request)
        return ExecutionResult(exit_code=0, output="1 passed", isolation="test")

    result = run_skill_lifecycle(skill_dir, execute=execute)

    assert result.status == "passed"
    assert result.registered is True
    assert result.test_attempts == 1
    assert requests[0].cwd == skill_dir
    assert requests[0].argv[-1] == "tests"


def test_text_only_skill_uses_static_validation_without_execution(tmp_path: Path) -> None:
    skill_dir = _skill(tmp_path, with_tests=False)

    def execute(_request: SkillTestRequest) -> ExecutionResult:
        pytest.fail("text-only skills must not execute a test process")

    result = run_skill_lifecycle(skill_dir, execute=execute)

    assert result.status == "static"
    assert result.registered is True
    assert result.test_attempts == 0


def test_failure_triggers_refinement_and_revalidation(tmp_path: Path) -> None:
    skill_dir = _skill(tmp_path)
    executions = iter(
        [
            ExecutionResult(1, "assertion failed", "test"),
            ExecutionResult(0, "1 passed", "test"),
        ]
    )
    refinement_requests = []

    def refine(request) -> bool:
        refinement_requests.append(request)
        (skill_dir / "scripts").mkdir()
        (skill_dir / "scripts" / "fixed.py").write_text("VALUE = 1\n", encoding="utf-8")
        return True

    result = run_skill_lifecycle(
        skill_dir,
        execute=lambda _request: next(executions),
        refine=refine,
        max_refinements=2,
    )

    assert result.status == "passed"
    assert result.registered is True
    assert result.test_attempts == 2
    assert result.refinement_attempts == 1
    assert refinement_requests[0].test_output == "assertion failed"


def test_unchanged_refinement_stops_without_reusing_stale_evidence(tmp_path: Path) -> None:
    skill_dir = _skill(tmp_path)

    result = run_skill_lifecycle(
        skill_dir,
        execute=lambda _request: ExecutionResult(1, "still broken", "test"),
        refine=lambda _request: True,
        max_refinements=2,
    )

    assert result.status == "stalled"
    assert result.registered is False
    assert result.test_attempts == 1
    assert result.refinement_attempts == 1


def test_refinement_budget_bounds_repeated_failures(tmp_path: Path) -> None:
    skill_dir = _skill(tmp_path)
    revisions = 0

    def refine(_request) -> bool:
        nonlocal revisions
        revisions += 1
        (skill_dir / "SKILL.md").write_text(
            "---\nname: demo-skill\ndescription: Demo lifecycle skill.\n---\n\n"
            f"# Revision {revisions}\n",
            encoding="utf-8",
        )
        return True

    result = run_skill_lifecycle(
        skill_dir,
        execute=lambda _request: ExecutionResult(1, "broken", "test"),
        refine=refine,
        max_refinements=2,
    )

    assert result.status == "failed"
    assert result.registered is False
    assert result.test_attempts == 3
    assert result.refinement_attempts == 2


def test_executor_error_keeps_skill_pending(tmp_path: Path) -> None:
    skill_dir = _skill(tmp_path)

    def execute(_request: SkillTestRequest) -> ExecutionResult:
        raise RuntimeError("sandbox unavailable")

    result = run_skill_lifecycle(skill_dir, execute=execute)

    assert result.status == "execution_error"
    assert result.registered is False
    assert "sandbox unavailable" in result.message


@pytest.mark.live_system_guard_bypass
def test_bubblewrap_executor_runs_tests_without_host_tmp_access(tmp_path: Path) -> None:
    executor = BubblewrapTestExecutor.discover()
    if executor is None:
        pytest.skip("bubblewrap is unavailable on this platform")

    host_secret = tmp_path / "host-only-secret"
    host_secret.write_text("secret", encoding="utf-8")
    skill_dir = _skill(tmp_path / "package")
    (skill_dir / "tests" / "test_demo.py").write_text(
        "from pathlib import Path\n\n"
        "def test_sandbox_hides_host_path():\n"
        f"    assert not Path({str(host_secret)!r}).exists()\n"
        "    assert not Path('/lib/systemd/systemd').exists()\n",
        encoding="utf-8",
    )

    result = executor(
        SkillTestRequest(
            argv=(executor.python_executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "tests"),
            cwd=skill_dir,
            timeout=30,
        )
    )

    assert result.exit_code == 0, result.output
    assert result.isolation == "bubblewrap"
    assert "passed" in result.output


@pytest.mark.live_system_guard_bypass
def test_bubblewrap_executor_captures_test_failure(tmp_path: Path) -> None:
    executor = BubblewrapTestExecutor.discover()
    if executor is None:
        pytest.skip("bubblewrap is unavailable on this platform")

    skill_dir = _skill(tmp_path)
    (skill_dir / "tests" / "test_demo.py").write_text(
        "def test_demo():\n    assert False, 'expected failure'\n",
        encoding="utf-8",
    )

    result = executor(
        SkillTestRequest(
            argv=(executor.python_executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "tests"),
            cwd=skill_dir,
            timeout=30,
        )
    )

    assert result.exit_code != 0
    assert "expected failure" in result.output


def test_bubblewrap_executor_uses_prlimit_without_thread_unsafe_preexec(
    tmp_path: Path, monkeypatch
) -> None:
    executor = BubblewrapTestExecutor.discover()
    if executor is None:
        pytest.skip("bubblewrap is unavailable on this platform")
    skill_dir = _skill(tmp_path)
    captured = {}

    def run(command, **kwargs):
        captured["command"] = command
        captured["kwargs"] = kwargs
        return SimpleNamespace(returncode=0, stdout=b"1 passed", stderr=b"")

    monkeypatch.setattr("tools.skill_test_sandbox.subprocess.run", run)

    result = executor(
        SkillTestRequest(
            argv=(executor.python_executable, "-m", "pytest", "-q", "tests"),
            cwd=skill_dir,
            timeout=30,
        )
    )

    assert result.exit_code == 0
    assert captured["command"][0] == executor.systemd_run
    assert "preexec_fn" not in captured["kwargs"]
    assert "--cap-drop" in captured["command"]
    assert "ALL" in captured["command"]
    assert "--property=TasksMax=16" in captured["command"]
    assert "--property=MemoryMax=2147483648" in captured["command"]
    assert not any(arg.startswith("--nproc=") for arg in captured["command"])
    assert "--as=268435456:268435456" in captured["command"]
    tmpfs_index = captured["command"].index("--tmpfs")
    assert captured["command"][tmpfs_index - 2 : tmpfs_index] == [
        "--size",
        "67108864",
    ]
    bind_targets = [
        captured["command"][index + 2]
        for index, arg in enumerate(captured["command"])
        if arg == "--ro-bind"
    ]
    assert "/usr" not in bind_targets
    assert "/bin" not in bind_targets
    assert "/lib" not in bind_targets
    assert executor.python_prefix not in bind_targets
    assert executor.python_base_prefix not in bind_targets


def test_bubblewrap_infrastructure_failure_is_not_reported_as_skill_failure(
    tmp_path: Path, monkeypatch
) -> None:
    executor = BubblewrapTestExecutor.discover()
    if executor is None:
        pytest.skip("bubblewrap is unavailable on this platform")
    skill_dir = _skill(tmp_path)

    def fail_sandbox(_command, **kwargs):
        kwargs["stderr"].write(b"bwrap: Creating new namespace failed\n")
        return SimpleNamespace(returncode=1)

    monkeypatch.setattr(
        "tools.skill_test_sandbox.subprocess.run",
        fail_sandbox,
    )

    with pytest.raises(RuntimeError, match="bubblewrap sandbox failed"):
        executor(
            SkillTestRequest(
                argv=(executor.python_executable, "-m", "pytest", "-q", "tests"),
                cwd=skill_dir,
                timeout=30,
            )
        )


def test_bubblewrap_output_is_file_backed_and_bounded(
    tmp_path: Path, monkeypatch
) -> None:
    executor = BubblewrapTestExecutor.discover()
    if executor is None:
        pytest.skip("bubblewrap is unavailable on this platform")
    skill_dir = _skill(tmp_path)

    def run(_command, **kwargs):
        assert kwargs["stdout"] is not __import__("subprocess").PIPE
        assert kwargs["stderr"] is not __import__("subprocess").PIPE
        kwargs["stdout"].write(b"A" * 100_000)
        kwargs["stderr"].write(b"B" * 100_000)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr("tools.skill_test_sandbox.subprocess.run", run)
    result = executor(
        SkillTestRequest(
            argv=(executor.python_executable, "-m", "pytest", "-q", "tests"),
            cwd=skill_dir,
            timeout=30,
        )
    )

    assert len(result.output.encode()) < 20_000
    assert "bytes omitted" in result.output
