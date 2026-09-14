"""Evidence-producing release qualification for the Workstation boundary.

The contract in :mod:`workstation.isolation` deliberately stays small and
read-only.  This module is the executable adapter around that contract: it
runs the local checks, imports clean-install evidence produced elsewhere and
emits one bounded JSON report suitable for CI or a release review.

It never updates a checkout, installs dependencies, changes user state or
attempts rollback.  A clean-machine result must be supplied by the clean
machine that actually performed that stage.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from typing import Any, Callable, Mapping, Sequence

from workstation.isolation import ReleaseQualificationGate
from workstation.session_lifecycle import SessionLifecycleStore


_MAX_OUTPUT_CHARS = 8_000


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(slots=True)
class StageEvidence:
    """Bounded evidence for one release qualification stage."""

    name: str
    passed: bool
    duration_seconds: float
    details: str = ""
    command: list[str] | None = None
    stdout: str = ""
    stderr: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ReleaseQualificationReport:
    """Serializable result of a fail-closed qualification run."""

    accepted: bool
    candidate_root: str
    candidate_revision: str | None
    generated_at: str
    stages: list[StageEvidence] = field(default_factory=list)
    failed_stage: str | None = None
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["stages"] = [stage.to_dict() for stage in self.stages]
        return data


@dataclass(frozen=True, slots=True)
class StageCheck:
    passed: bool
    details: str = ""
    command: list[str] | None = None
    stdout: str = ""
    stderr: str = ""


class ReleaseQualificationRunner:
    """Run the six release gates without mutating the candidate checkout."""

    def __init__(
        self,
        root: Path,
        *,
        python_executable: str | Path | None = None,
        timeout_seconds: float = 300.0,
    ) -> None:
        self.root = Path(root).resolve()
        self.python_executable = str(python_executable or sys.executable)
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self.timeout_seconds = float(timeout_seconds)

    def run(
        self,
        *,
        clean_install_evidence: Path | None = None,
        checks: Mapping[str, Callable[[], bool | StageCheck]] | None = None,
    ) -> ReleaseQualificationReport:
        """Run stages in canonical order and stop at the first failure.

        ``checks`` exists for downstream CI adapters and contract tests.  The
        normal CLI path uses the built-in checks below.  A boolean callback is
        accepted as a deliberately small compatibility surface; production
        adapters should return ``StageCheck`` so their evidence is retained.
        """

        revision, revision_error = self._git_revision()
        check_map = dict(checks or self._default_checks(clean_install_evidence, revision))
        report = ReleaseQualificationReport(
            accepted=True,
            candidate_root=str(self.root),
            candidate_revision=revision,
            generated_at=_utc_now(),
        )
        if revision_error and checks is None:
            check_map["ownership"] = lambda: StageCheck(False, revision_error)

        for stage_name in ReleaseQualificationGate.STAGES:
            check = check_map.get(stage_name)
            started = time.monotonic()
            if check is None:
                outcome = StageCheck(False, "stage missing")
            else:
                try:
                    raw = check()
                    outcome = raw if isinstance(raw, StageCheck) else StageCheck(bool(raw))
                except Exception as exc:
                    outcome = StageCheck(False, str(exc))
            report.stages.append(
                StageEvidence(
                    name=stage_name,
                    passed=outcome.passed,
                    duration_seconds=round(max(0.0, time.monotonic() - started), 3),
                    details=outcome.details,
                    command=outcome.command,
                    stdout=_bound_output(outcome.stdout),
                    stderr=_bound_output(outcome.stderr),
                )
            )
            if not outcome.passed:
                report.accepted = False
                report.failed_stage = stage_name
                report.reason = outcome.details or "stage failed"
                break
        return report

    def _default_checks(
        self,
        clean_install_evidence: Path | None,
        revision: str | None,
    ) -> dict[str, Callable[[], bool | StageCheck]]:
        return {
            "ownership": lambda: self._check_ownership(revision),
            "assets": self._check_assets,
            "clean_install": lambda: self._check_clean_install(clean_install_evidence, revision),
            "workstation_smoke": self._check_workstation_smoke,
            "native_smoke": self._check_native_smoke,
            "migration": self._check_migration,
        }

    def _git_revision(self) -> tuple[str | None, str | None]:
        if not self.root.exists():
            return None, f"candidate root does not exist: {self.root}"
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=self.root,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=min(30.0, self.timeout_seconds),
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            return None, f"could not resolve candidate revision: {exc}"
        revision = result.stdout.strip()
        if result.returncode != 0 or not revision:
            return None, _command_failure("git revision", result)
        return revision, None

    def _check_ownership(self, revision: str | None) -> StageCheck:
        required = self.root / "workstation"
        if revision is None:
            return StageCheck(False, "candidate revision is unavailable")
        if not required.is_dir():
            return StageCheck(False, "workstation directory is missing")
        status_command = ["git", "status", "--porcelain=v1", "--untracked-files=all"]
        try:
            status = subprocess.run(
                status_command,
                cwd=self.root,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=min(30.0, self.timeout_seconds),
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            return StageCheck(False, f"could not verify checkout ownership: {exc}", status_command)
        if status.returncode != 0:
            return StageCheck(False, _command_failure("git status", status), status_command, status.stdout, status.stderr)
        if status.stdout.strip():
            return StageCheck(
                False,
                "candidate checkout is not clean",
                status_command,
                status.stdout,
                status.stderr,
            )
        return StageCheck(True, f"candidate revision {revision}")

    def _check_assets(self) -> StageCheck:
        expected = (
            "workstation/install.ps1",
            "workstation/doctor.ps1",
            "apps/desktop/package.json",
            "apps/desktop/assets/icon.ico",
            "web/package.json",
        )
        missing = [relative for relative in expected if not (self.root / relative).is_file()]
        if missing:
            return StageCheck(False, "missing release assets: " + ", ".join(missing))
        return StageCheck(True, f"verified {len(expected)} release assets")

    def _check_clean_install(self, evidence_path: Path | None, revision: str | None) -> StageCheck:
        if evidence_path is None:
            return StageCheck(False, "clean-install evidence was not supplied")
        path = Path(evidence_path)
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            return StageCheck(False, f"invalid clean-install evidence: {exc}")
        if not isinstance(data, dict):
            return StageCheck(False, "clean-install evidence must be a JSON object")
        workstation_home = data.get("workstation_home")
        if not isinstance(workstation_home, str) or not workstation_home.strip():
            return StageCheck(False, "clean-install evidence must name workstation_home")
        workstation_home_path = Path(workstation_home.strip())
        if not workstation_home_path.is_absolute():
            return StageCheck(False, "clean-install workstation_home must be absolute")
        try:
            if workstation_home_path == self.root or self.root in workstation_home_path.parents:
                return StageCheck(False, "clean-install workstation_home must be outside the candidate checkout")
        except (OSError, ValueError):
            return StageCheck(False, "clean-install workstation_home is not a valid path")
        evidence_revision = str(data.get("candidate_revision", ""))
        stages = data.get("stages", [])
        stage_names = {str(item) for item in stages} if isinstance(stages, list) else set()
        if data.get("accepted") is not True or "clean_install" not in stage_names:
            return StageCheck(False, "clean-install evidence is not an accepted clean_install stage")
        if not revision or evidence_revision != revision:
            return StageCheck(False, "clean-install evidence revision does not match candidate HEAD")
        return StageCheck(True, f"accepted clean-install evidence for {revision}")

    def _check_workstation_smoke(self) -> StageCheck:
        return self._run_command(
            [self.python_executable, "-m", "pytest", "workstation/tests", "-q", "-p", "no:cacheprovider"]
        )

    def _check_native_smoke(self) -> StageCheck:
        probe = self.root / "workstation" / "context" / "engineering-journal" / "probes" / "v3-runtime-hardening-smoke.py"
        if not probe.is_file():
            return StageCheck(False, f"native smoke probe is missing: {probe}")
        return self._run_command([self.python_executable, str(probe)])

    def _check_migration(self) -> StageCheck:
        try:
            with tempfile.TemporaryDirectory(prefix="hermes-release-migration-") as raw_root:
                store = SessionLifecycleStore(Path(raw_root) / "sessions.json")
                store.register("release-qualification", provider="qualification")
                migrated = store.migrate(
                    "release-qualification",
                    target_version=2,
                    transform=lambda data: {**data, "toolset_fingerprint": "release-qualification-v2"},
                    validate=lambda data: data.get("toolset_fingerprint") == "release-qualification-v2",
                )
                if migrated.schema_version != 2:
                    return StageCheck(False, "migration did not reach schema version 2")
        except Exception as exc:
            return StageCheck(False, f"migration preflight failed: {exc}")
        return StageCheck(True, "migration transform, validation and backup completed")

    def _run_command(self, command: Sequence[str]) -> StageCheck:
        try:
            result = subprocess.run(
                list(command),
                cwd=self.root,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=self.timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            return StageCheck(
                False,
                f"command timed out after {self.timeout_seconds:g}s",
                list(command),
                _bound_output(_process_output(exc.stdout)),
                _bound_output(_process_output(exc.stderr)),
            )
        except OSError as exc:
            return StageCheck(False, f"could not start command: {exc}", list(command))
        passed = result.returncode == 0
        details = "command passed" if passed else _command_failure("command", result)
        return StageCheck(
            passed,
            details,
            list(command),
            _bound_output(result.stdout),
            _bound_output(result.stderr),
        )


def _bound_output(value: Any) -> str:
    text = "" if value is None else str(value)
    if len(text) <= _MAX_OUTPUT_CHARS:
        return text
    return "[truncated]\n" + text[-_MAX_OUTPUT_CHARS:]


def _process_output(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return "" if value is None else str(value)


def _command_failure(label: str, result: subprocess.CompletedProcess[str]) -> str:
    return f"{label} failed with exit code {result.returncode}"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the Hermes Workstation release qualification gate")
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="candidate checkout root")
    parser.add_argument("--python", dest="python_executable", default=sys.executable, help="Python used for smoke checks")
    parser.add_argument("--timeout", type=float, default=300.0, help="per-command timeout in seconds")
    parser.add_argument(
        "--clean-install-evidence",
        type=Path,
        help="accepted JSON evidence produced by a clean-machine qualification",
    )
    parser.add_argument("--report", type=Path, help="also write the JSON report to this path")
    args = parser.parse_args(argv)

    report = ReleaseQualificationRunner(
        args.root,
        python_executable=args.python_executable,
        timeout_seconds=args.timeout,
    ).run(clean_install_evidence=args.clean_install_evidence)
    payload = json.dumps(report.to_dict(), ensure_ascii=False, indent=2)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(payload + "\n", encoding="utf-8")
    print(payload)
    return 0 if report.accepted else 1


if __name__ == "__main__":
    raise SystemExit(main())
