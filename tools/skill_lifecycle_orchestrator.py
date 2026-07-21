"""Deterministic create/evaluate/refine/register loop for skill packages.

The orchestrator owns lifecycle state but not execution.  Callers provide an
isolated test executor and, optionally, a refinement callback.  This keeps
arbitrary generated test code out of ``skill_manage`` while still enforcing the
same digest-bound validation gate used by normal skill discovery.
"""

from __future__ import annotations

import shlex
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Sequence

from tools.skill_validation import (
    LIFECYCLE_LOCK_FILE,
    invalidate_skill_validation,
    record_skill_validation,
    skill_content_digest,
    validation_allows_discovery,
)
from tools.skill_sidecar_io import sidecar_lock


@dataclass(frozen=True)
class SkillTestRequest:
    """A fixed, shell-free test request for an isolated executor."""

    argv: tuple[str, ...]
    cwd: Path
    timeout: int = 300


@dataclass(frozen=True)
class TestExecutionResult:
    """Bounded result returned by an isolated test executor."""

    exit_code: int
    output: str
    isolation: str


@dataclass(frozen=True)
class RefinementRequest:
    """Failure evidence supplied to a package refiner."""

    skill_dir: Path
    attempt: int
    content_digest: str
    test_output: str


@dataclass(frozen=True)
class SkillLifecycleResult:
    """Final state of one bounded lifecycle run."""

    status: str
    registered: bool
    test_attempts: int
    refinement_attempts: int
    message: str = ""
    content_digest: str = ""


TestExecutor = Callable[[SkillTestRequest], TestExecutionResult]
SkillRefiner = Callable[[RefinementRequest], bool]


def _pytest_request(skill_dir: Path, python_executable: str) -> SkillTestRequest:
    return SkillTestRequest(
        argv=(
            python_executable,
            "-m",
            "pytest",
            "-q",
            "-p",
            "no:cacheprovider",
            "tests",
        ),
        cwd=skill_dir,
    )


def run_skill_lifecycle(
    skill_dir: Path,
    *,
    execute: TestExecutor,
    refine: Optional[SkillRefiner] = None,
    max_refinements: int = 2,
    python_executable: Optional[str] = None,
) -> SkillLifecycleResult:
    """Evaluate a skill and refine it through a bounded retry loop.

    ``execute`` must provide isolation appropriate for untrusted generated test
    code.  The orchestrator never invokes a shell and never falls back to local
    execution.  Each test result is tied to a fresh package digest and one-time
    challenge token before the discovery gate is opened.
    """

    skill_dir = Path(skill_dir).resolve()
    if max_refinements < 0:
        raise ValueError("max_refinements must be non-negative")

    # Serialize the entire evaluate → refine → re-evaluate cycle for one package
    # across threads and cooperating processes. Digest/token locking already
    # protects a single evidence submission, but only a whole-cycle lock stops
    # two lifecycle runners (e.g. two background reviews) from interleaving a
    # patch from one with a test run from the other. If secure whole-cycle
    # locking is unavailable, fail closed rather than running an interleavable
    # lifecycle.
    try:
        with sidecar_lock(skill_dir, LIFECYCLE_LOCK_FILE):
            return _run_skill_lifecycle_locked(
                skill_dir,
                execute=execute,
                refine=refine,
                max_refinements=max_refinements,
                python_executable=python_executable,
            )
    except OSError as exc:
        return SkillLifecycleResult(
            status="lock_error",
            registered=False,
            test_attempts=0,
            refinement_attempts=0,
            message=str(exc),
        )


def _run_skill_lifecycle_locked(
    skill_dir: Path,
    *,
    execute: TestExecutor,
    refine: Optional[SkillRefiner] = None,
    max_refinements: int = 2,
    python_executable: Optional[str] = None,
) -> SkillLifecycleResult:
    test_attempts = 0
    refinement_attempts = 0
    executable = python_executable or sys.executable

    while True:
        challenge = record_skill_validation(skill_dir)
        status = str(challenge.get("validation_status") or "invalid")
        digest = str(challenge.get("content_digest") or "")

        if status == "static":
            return SkillLifecycleResult(
                status="static",
                registered=validation_allows_discovery(skill_dir),
                test_attempts=test_attempts,
                refinement_attempts=refinement_attempts,
                content_digest=digest,
            )
        if status != "pending":
            return SkillLifecycleResult(
                status="invalid",
                registered=False,
                test_attempts=test_attempts,
                refinement_attempts=refinement_attempts,
                message=str(challenge.get("error") or "validation challenge failed"),
                content_digest=digest,
            )

        request = _pytest_request(skill_dir, executable)
        try:
            execution = execute(request)
        except Exception as exc:
            return SkillLifecycleResult(
                status="execution_error",
                registered=False,
                test_attempts=test_attempts,
                refinement_attempts=refinement_attempts,
                message=str(exc),
                content_digest=digest,
            )

        if not isinstance(execution.exit_code, int) or isinstance(
            execution.exit_code, bool
        ):
            raise TypeError("executor exit_code must be an integer")
        if not isinstance(execution.output, str):
            raise TypeError("executor output must be a string")
        if not execution.isolation.strip():
            raise ValueError("executor must identify its isolation boundary")

        test_attempts += 1
        evidence = record_skill_validation(
            skill_dir,
            {
                "content_digest": digest,
                "validation_token": challenge.get("validation_token"),
                "command": shlex.join(request.argv),
                "exit_code": execution.exit_code,
                "output": execution.output,
            },
        )
        evidence_status = str(evidence.get("validation_status") or "invalid")

        if evidence_status == "passed":
            return SkillLifecycleResult(
                status="passed",
                registered=validation_allows_discovery(skill_dir),
                test_attempts=test_attempts,
                refinement_attempts=refinement_attempts,
                content_digest=digest,
            )

        # A package mutation during execution makes the evidence stale.  Never
        # refine from or register against an execution of different content.
        if evidence_status != "failed":
            return SkillLifecycleResult(
                status="stale",
                registered=False,
                test_attempts=test_attempts,
                refinement_attempts=refinement_attempts,
                message=str(evidence.get("error") or "validation evidence is stale"),
                content_digest=str(evidence.get("content_digest") or digest),
            )

        if refine is None or refinement_attempts >= max_refinements:
            return SkillLifecycleResult(
                status="failed",
                registered=False,
                test_attempts=test_attempts,
                refinement_attempts=refinement_attempts,
                message=str(evidence.get("error") or "skill tests failed"),
                content_digest=digest,
            )

        before = skill_content_digest(skill_dir)
        refinement_attempts += 1
        changed = refine(
            RefinementRequest(
                skill_dir=skill_dir,
                attempt=refinement_attempts,
                content_digest=before,
                test_output=execution.output,
            )
        )
        after = skill_content_digest(skill_dir)
        if not changed or after == before:
            return SkillLifecycleResult(
                status="stalled",
                registered=False,
                test_attempts=test_attempts,
                refinement_attempts=refinement_attempts,
                message="refinement did not change package content",
                content_digest=after,
            )

        invalidate_skill_validation(skill_dir)


__all__: Sequence[str] = (
    "RefinementRequest",
    "SkillLifecycleResult",
    "SkillTestRequest",
    "TestExecutionResult",
    "run_skill_lifecycle",
)
