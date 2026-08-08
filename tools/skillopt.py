#!/usr/bin/env python3
"""SkillOpt-style validation gate for Hermes skills.

This is not a paper reimplementation. It is the small, boring piece Hermes
needs in core: keep candidate skill edits quarantined, validate them, and only
promote a candidate when it strictly improves a held-out score.

Inspired by: Yang et al., "SkillOpt: Executive Strategy for Self-Evolving
Agent Skills" (arXiv:2605.23904, 2026).
"""

from __future__ import annotations

import difflib
import hashlib
import json
import math
import os
import re
import shlex
import subprocess
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from utils import atomic_replace

Decision = Literal["accepted", "rejected"]


@dataclass(frozen=True)
class SkillOptResult:
    """Outcome returned by :func:`promote_skill_candidate`."""

    decision: Decision
    reason: str
    skill_path: str
    candidate_path: str
    baseline_score: float
    candidate_score: float
    delta: float
    accepted: bool
    dry_run: bool = False
    validator_exit_code: int | None = None
    validator_output: str = ""
    skill_sha256_before: str = ""
    candidate_sha256: str = ""
    backup_path: str | None = None
    history_path: str | None = None
    rejected_path: str | None = None
    diff_preview: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


class SkillOptError(ValueError):
    """Raised for malformed SkillOpt inputs."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def _read_text(path: Path, label: str) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise SkillOptError(f"{label} does not exist: {path}") from exc
    except UnicodeDecodeError as exc:
        raise SkillOptError(f"{label} must be UTF-8 text: {path}") from exc


def _skill_md_path(skill_path: Path) -> Path:
    """Resolve either a skill directory or a SKILL.md path to SKILL.md."""
    path = skill_path.expanduser()
    if path.is_dir():
        path = path / "SKILL.md"
    return path


def _validate_skill_document(content: str) -> str | None:
    # Reuse the same frontmatter/body validation as skill_manage. Import lazily
    # to avoid making every CLI startup pay for PyYAML before skills are touched.
    from tools.skill_manager_tool import _validate_frontmatter, _validate_content_size

    size_error = _validate_content_size(content, "SKILL.md")
    if size_error:
        return size_error
    return _validate_frontmatter(content)


def _frontmatter_name(content: str, label: str) -> str:
    """Return the YAML frontmatter ``name`` value for a validated SKILL.md."""
    import yaml

    end_match = re.search(r"\n---\s*\n", content[3:])
    if not end_match:
        raise SkillOptError(f"{label} frontmatter is not closed")
    parsed = yaml.safe_load(content[3 : end_match.start() + 3]) or {}
    name = parsed.get("name") if isinstance(parsed, dict) else None
    if not isinstance(name, str) or not name.strip():
        raise SkillOptError(f"{label} frontmatter name must be a non-empty string")
    return name.strip()


def _run_validator(command: str, skill_file: Path, candidate_file: Path) -> tuple[int, str]:
    env = os.environ.copy()
    env["HERMES_SKILLOPT_SKILL"] = str(skill_file)
    env["HERMES_SKILLOPT_CANDIDATE"] = str(candidate_file)
    try:
        completed = subprocess.run(
            command,
            shell=True,
            cwd=str(skill_file.parent),
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=600,
        )
    except subprocess.TimeoutExpired as exc:
        output = exc.stdout if isinstance(exc.stdout, str) else ""
        return 124, (output + "\nvalidator timed out after 600s").strip()
    return completed.returncode, (completed.stdout or "").strip()


def _diff_preview(before: str, after: str, *, max_lines: int = 80) -> str:
    lines = list(
        difflib.unified_diff(
            before.splitlines(),
            after.splitlines(),
            fromfile="current/SKILL.md",
            tofile="candidate/SKILL.md",
            lineterm="",
        )
    )
    if len(lines) > max_lines:
        lines = lines[:max_lines] + [f"... diff truncated after {max_lines} lines"]
    return "\n".join(lines)


def _record_result(skill_dir: Path, result: SkillOptResult) -> SkillOptResult:
    payload = asdict(result) | {"timestamp": _utc_now()}
    if result.accepted:
        history_path = skill_dir / ".skillopt" / "history.jsonl"
        _append_jsonl(history_path, payload)
        return SkillOptResult(**(asdict(result) | {"history_path": str(history_path)}))
    rejected_path = skill_dir / ".skillopt" / "rejected.jsonl"
    _append_jsonl(rejected_path, payload)
    return SkillOptResult(**(asdict(result) | {"rejected_path": str(rejected_path)}))


def promote_skill_candidate(
    skill_path: str | Path,
    candidate_path: str | Path,
    *,
    baseline_score: float,
    candidate_score: float,
    validator: str | None = None,
    allow_equal: bool = False,
    dry_run: bool = False,
    metadata: dict[str, Any] | None = None,
) -> SkillOptResult:
    """Promote a candidate SKILL.md only when it passes the validation gate.

    The gate intentionally mirrors the most important production lesson from
    SkillOpt: skills may self-edit, but promotion is conservative. A candidate
    must be valid, pass any external validator, and beat the held-out baseline
    score unless ``allow_equal`` is explicitly set.
    """

    baseline_score = float(baseline_score)
    candidate_score = float(candidate_score)
    if not math.isfinite(baseline_score) or not math.isfinite(candidate_score):
        raise SkillOptError("baseline_score and candidate_score must be finite numbers")

    skill_file = _skill_md_path(Path(skill_path))
    candidate_file = Path(candidate_path).expanduser()
    current = _read_text(skill_file, "skill")
    candidate = _read_text(candidate_file, "candidate")
    skill_dir = skill_file.parent

    base_payload = {
        "skill_path": str(skill_file),
        "candidate_path": str(candidate_file),
        "baseline_score": baseline_score,
        "candidate_score": candidate_score,
        "delta": candidate_score - baseline_score,
        "dry_run": dry_run,
        "skill_sha256_before": _sha256_text(current),
        "candidate_sha256": _sha256_text(candidate),
        "diff_preview": _diff_preview(current, candidate),
        "metadata": metadata or {},
    }

    if skill_file.resolve() == candidate_file.resolve():
        result = SkillOptResult(
            decision="rejected",
            accepted=False,
            reason="candidate must be a quarantined file distinct from the live SKILL.md",
            **base_payload,
        )
        return _record_result(skill_dir, result)

    current_validation_error = _validate_skill_document(current)
    if current_validation_error:
        result = SkillOptResult(
            decision="rejected",
            accepted=False,
            reason=f"current skill validation failed: {current_validation_error}",
            **base_payload,
        )
        return _record_result(skill_dir, result)

    validation_error = _validate_skill_document(candidate)
    if validation_error:
        result = SkillOptResult(
            decision="rejected",
            accepted=False,
            reason=f"candidate validation failed: {validation_error}",
            **base_payload,
        )
        return _record_result(skill_dir, result)

    current_name = _frontmatter_name(current, "skill")
    candidate_name = _frontmatter_name(candidate, "candidate")
    if candidate_name != current_name:
        result = SkillOptResult(
            decision="rejected",
            accepted=False,
            reason=(
                f"candidate skill name mismatch: expected {current_name!r}, "
                f"got {candidate_name!r}"
            ),
            **base_payload,
        )
        return _record_result(skill_dir, result)

    validator_exit_code = None
    validator_output = ""
    if validator:
        validator_exit_code, validator_output = _run_validator(
            validator, skill_file, candidate_file
        )
        if validator_exit_code != 0:
            result = SkillOptResult(
                decision="rejected",
                accepted=False,
                reason=f"validator failed: {shlex.quote(validator)} exited {validator_exit_code}",
                validator_exit_code=validator_exit_code,
                validator_output=validator_output,
                **base_payload,
            )
            return _record_result(skill_dir, result)

    improved = candidate_score > baseline_score or (
        allow_equal and candidate_score == baseline_score
    )
    if not improved:
        comparator = "meet" if allow_equal else "beat"
        result = SkillOptResult(
            decision="rejected",
            accepted=False,
            reason=(
                f"candidate score {candidate_score:g} did not {comparator} "
                f"baseline {baseline_score:g}"
            ),
            validator_exit_code=validator_exit_code,
            validator_output=validator_output,
            **base_payload,
        )
        return _record_result(skill_dir, result)

    backup_path = None
    if not dry_run:
        backup_dir = skill_dir / ".skillopt" / "backups"
        backup_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        backup_file = backup_dir / f"SKILL.{stamp}.{_sha256_text(current)[:12]}.md"
        backup_file.write_text(current, encoding="utf-8")
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=str(skill_file.parent),
            prefix=".skillopt-promote-",
            suffix=".tmp",
            delete=False,
        ) as handle:
            handle.write(candidate)
            tmp_path = Path(handle.name)
        atomic_replace(tmp_path, skill_file)
        backup_path = str(backup_file)

    result = SkillOptResult(
        decision="accepted",
        accepted=True,
        reason="candidate passed validation and improved held-out score",
        validator_exit_code=validator_exit_code,
        validator_output=validator_output,
        backup_path=backup_path,
        **base_payload,
    )
    return _record_result(skill_dir, result)
