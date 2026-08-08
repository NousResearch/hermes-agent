"""Native skill evaluation runner tool.

This tool intentionally mirrors the Aion Phase 8 manual ``skill_eval_run``
prototype without adding LLM judging, skill mutation, candidate promotion, or
runtime config mutation. Dry-runs are always reported as unscored.
"""

from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from hermes_constants import get_hermes_home
from tools.registry import registry


@dataclass
class CaseResult:
    case_id: str
    title: str
    score: int | None
    max_points: int
    notes: str
    hard_fail: bool = False


def _json_error(message: str, *, errors: list[str] | None = None) -> str:
    payload: dict[str, Any] = {"status": "error", "error": message}
    if errors is not None:
        payload["errors"] = errors
    return json.dumps(payload, indent=2)


def _reject_traversal_segments(path: Path, field: str) -> None:
    if any(part in {".", ".."} for part in path.parts):
        raise ValueError(f"{field} contains parent-directory traversal segment")


def _resolve_existing_path(value: str, field: str) -> Path:
    if not value or not str(value).strip():
        raise ValueError(f"{field} is required")
    path = Path(value).expanduser()
    _reject_traversal_segments(path, field)
    absolute = path if path.is_absolute() else Path.cwd() / path
    absolute = absolute.absolute()
    if not absolute.exists():
        raise ValueError(f"{field} does not exist: {path}")
    if not absolute.is_file():
        raise ValueError(f"{field} is not a file: {absolute}")
    return absolute


def _resolve_output_path(value: str | None, default: Path) -> Path:
    path = Path(value).expanduser() if value else default
    _reject_traversal_segments(path, "report_path")
    absolute = path if path.is_absolute() else Path.cwd() / path
    return absolute.absolute()


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _aion_evals_root() -> Path:
    return (get_hermes_home().expanduser().absolute() / "aion-evolution" / "evals").absolute()


def _reports_dir() -> Path:
    return (_aion_evals_root() / "reports").absolute()


def _reject_symlink_component(path: Path, label: str) -> None:
    """Reject if any existing component of ``path`` is a symlink.

    The tool's authorization boundary is the literal Hermes evals/reports tree,
    not the canonical target of symlinks.  ``Path.resolve``-based checks would
    authorize symlink targets outside that tree, so we fail closed instead.
    """
    current = Path(path.anchor) if path.is_absolute() else Path.cwd()
    parts = path.parts[1:] if path.is_absolute() else path.parts
    for part in parts:
        current = current / part
        try:
            if current.is_symlink():
                raise ValueError(f"{label} contains symlink component: {current}")
        except OSError as exc:
            raise ValueError(f"cannot inspect {label} path component: {current}") from exc


def _ensure_eval_path_allowed(path: Path) -> None:
    allowed = _aion_evals_root()
    _reject_symlink_component(allowed, "evals root")
    _reject_symlink_component(path, "eval_path")
    if not _is_relative_to(path, allowed):
        raise ValueError(f"eval_path outside allowed roots: {path}; allowed root: {allowed}")
    if path.parent != allowed:
        raise ValueError(f"eval_path and score_file must be direct children of evals root: {allowed}")


def _ensure_report_path_allowed(path: Path) -> None:
    reports = _reports_dir()
    _reject_symlink_component(reports, "reports directory")
    _reject_symlink_component(path, "report_path")
    _reject_symlink_component(path.parent, "report_path")
    if not _is_relative_to(path, reports):
        raise ValueError(f"report_path outside reports directory: {path}; reports directory: {reports}")
    if path.parent != reports:
        raise ValueError(f"report_path must be a direct child of reports directory: {reports}")


def _write_report_safely(path: Path, content: str) -> None:
    reports = _reports_dir()
    _ensure_report_path_allowed(path)

    dir_fd = _open_dir_chain(reports, create_leaf=True)
    file_fd: int | None = None
    temp_name = f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    try:
        file_fd = os.open(
            temp_name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
            0o600,
            dir_fd=dir_fd,
        )
        with os.fdopen(file_fd, "w", encoding="utf-8") as handle:
            file_fd = None
            handle.write(content)
        os.replace(temp_name, path.name, src_dir_fd=dir_fd, dst_dir_fd=dir_fd)
    except OSError as exc:
        try:
            os.unlink(temp_name, dir_fd=dir_fd)
        except OSError:
            pass
        raise ValueError(f"failed to safely write report_path: {exc}") from exc
    finally:
        if file_fd is not None:
            os.close(file_fd)
        if dir_fd is not None:
            os.close(dir_fd)


def _open_dir_chain(path: Path, *, create_leaf: bool = False) -> int:
    """Open an absolute directory path component-by-component with O_NOFOLLOW.

    Returns a directory file descriptor for ``path``.  Callers own the fd.
    This avoids the pathname-check-then-open race for symlinked ancestors: every
    component is opened relative to the previous trusted directory fd, and
    ``O_NOFOLLOW`` is applied at each step.
    """
    if not path.is_absolute():
        raise ValueError(f"directory path must be absolute: {path}")
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(Path(path.anchor), flags)
    parts = path.parts[1:]
    try:
        for index, part in enumerate(parts):
            is_leaf = index == len(parts) - 1
            if create_leaf and is_leaf:
                try:
                    os.mkdir(part, 0o700, dir_fd=fd)
                except FileExistsError:
                    pass
            next_fd = os.open(part, flags, dir_fd=fd)
            os.close(fd)
            fd = next_fd
        return fd
    except OSError as exc:
        os.close(fd)
        raise ValueError(f"failed to safely open directory path: {path}: {exc}") from exc


def _is_int_not_bool(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def load_json(path: Path) -> dict[str, Any]:
    _ensure_eval_path_allowed(path)
    evals_fd = _open_dir_chain(_aion_evals_root(), create_leaf=False)
    file_fd: int | None = None
    try:
        file_fd = os.open(path.name, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0), dir_fd=evals_fd)
        with os.fdopen(file_fd, "r", encoding="utf-8") as handle:
            file_fd = None
            data = json.load(handle)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {path}: {exc}") from exc
    except OSError as exc:
        raise ValueError(f"failed to safely read eval JSON: {path}: {exc}") from exc
    finally:
        if file_fd is not None:
            os.close(file_fd)
        os.close(evals_fd)
    if not isinstance(data, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return data


def validate_pack(data: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    required_top = ["schema_version", "skill_name", "skill_path", "purpose", "cases", "scoring"]
    for key in required_top:
        if key not in data:
            errors.append(f"missing top-level field: {key}")
    if data.get("schema_version") != "1.0":
        errors.append("schema_version must be 1.0")

    cases = data.get("cases")
    if not isinstance(cases, list) or not cases:
        errors.append("cases must be a non-empty list")
        return errors

    required_case = [
        "id",
        "prompt",
        "expected_trigger",
        "required_behaviors",
        "forbidden_behaviors",
        "evidence_requirements",
        "scoring_rubric",
    ]
    ids: set[str] = set()
    for idx, case in enumerate(cases, start=1):
        if not isinstance(case, dict):
            errors.append(f"case {idx} is not an object")
            continue
        for key in required_case:
            if key not in case:
                errors.append(f"case {idx} missing field: {key}")
        raw_cid = case.get("id", "")
        if not isinstance(raw_cid, str) or not raw_cid:
            cid = str(raw_cid)
            errors.append(f"case {idx} id must be a non-empty string")
        else:
            cid = raw_cid
        if cid and cid in ids:
            errors.append(f"duplicate case id: {cid}")
        if cid:
            ids.add(cid)
        rubric = case.get("scoring_rubric", {})
        if not isinstance(rubric, dict) or "max_points" not in rubric or "criteria" not in rubric:
            errors.append(f"case {cid or idx} has invalid scoring_rubric")
        else:
            if not _is_int_not_bool(rubric.get("max_points")) or rubric["max_points"] <= 0:
                errors.append(f"case {cid or idx} scoring_rubric.max_points must be a positive integer")
            if not isinstance(rubric.get("criteria"), list) or not rubric["criteria"]:
                errors.append(f"case {cid or idx} scoring_rubric.criteria must be a non-empty list")
        for list_key in ["required_behaviors", "forbidden_behaviors", "evidence_requirements"]:
            if list_key in case and not isinstance(case[list_key], list):
                errors.append(f"case {cid or idx} {list_key} must be a list")

    scoring = data.get("scoring", {})
    if not isinstance(scoring, dict):
        errors.append("scoring must be an object")
    else:
        for key in ["case_max_points", "pass_threshold_percent", "hard_fail_conditions"]:
            if key not in scoring:
                errors.append(f"scoring missing field: {key}")
        if "case_max_points" in scoring and (
            not _is_int_not_bool(scoring["case_max_points"]) or scoring["case_max_points"] <= 0
        ):
            errors.append("scoring.case_max_points must be a positive integer")
        threshold = scoring.get("pass_threshold_percent")
        if not isinstance(threshold, (int, float)) or isinstance(threshold, bool) or not 0 <= threshold <= 100:
            errors.append("scoring.pass_threshold_percent must be a number between 0 and 100")
        if "hard_fail_conditions" in scoring and not isinstance(scoring["hard_fail_conditions"], list):
            errors.append("scoring.hard_fail_conditions must be a list")
    return errors


def load_scores(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    raw = load_json(path)
    if "scores" in raw:
        raw = raw["scores"]
    if not isinstance(raw, dict):
        raise ValueError('Score file must be an object keyed by case id, or {"scores": {...}}')

    scores: dict[str, dict[str, Any]] = {}
    for case_id, value in raw.items():
        if _is_int_not_bool(value):
            scores[str(case_id)] = {"score": value, "notes": "", "hard_fail": False}
        elif isinstance(value, dict):
            raw_score = value.get("score")
            if not _is_int_not_bool(raw_score):
                raise ValueError(f"Score for {case_id} must be an integer")
            if "hard_fail" in value and not isinstance(value["hard_fail"], bool):
                raise ValueError(f"hard_fail for {case_id} must be a boolean")
            scores[str(case_id)] = {
                "score": raw_score,
                "notes": value.get("notes", ""),
                "hard_fail": bool(value.get("hard_fail", False)),
            }
        else:
            raise ValueError(f"Invalid score value for {case_id}")
    return scores


def render_case(case: dict[str, Any]) -> str:
    lines = [f"### {case['id']} — {case.get('title', '').strip() or 'Untitled'}", ""]
    lines.extend(["Prompt:", "", f"> {case['prompt']}", ""])
    lines.append(f"Expected trigger: {case['expected_trigger']}")
    if case.get("expected_decision"):
        lines.append(f"Expected decision: `{case['expected_decision']}`")
    lines.append("")
    lines.append("Required behaviors:")
    for item in case.get("required_behaviors", []):
        lines.append(f"- {item}")
    lines.append("")
    lines.append("Forbidden behaviors:")
    for item in case.get("forbidden_behaviors", []):
        lines.append(f"- {item}")
    lines.append("")
    lines.append("Evidence requirements:")
    for item in case.get("evidence_requirements", []):
        lines.append(f"- {item}")
    lines.append("")
    return "\n".join(lines)


def build_results(data: dict[str, Any], scores: dict[str, dict[str, Any]]) -> list[CaseResult]:
    case_ids = {str(case["id"]) for case in data["cases"]}
    unknown_ids = sorted(set(scores) - case_ids)
    if unknown_ids:
        raise ValueError(f"unknown score case id(s): {', '.join(unknown_ids)}")
    if not scores:
        raise ValueError("manual_score requires at least one matching score")
    results: list[CaseResult] = []
    for case in data["cases"]:
        cid = case["id"]
        max_points = int(case["scoring_rubric"]["max_points"])
        score_entry = scores.get(cid)
        if score_entry is None:
            score = None
            notes = "Not scored."
            hard_fail = False
        else:
            raw_score = score_entry.get("score")
            score = None if raw_score is None else int(raw_score)
            if score is not None and (score < 0 or score > max_points):
                raise ValueError(f"Score for {cid} must be between 0 and {max_points}")
            notes = str(score_entry.get("notes", ""))
            hard_fail = bool(score_entry.get("hard_fail", False))
        results.append(CaseResult(cid, case.get("title", ""), score, max_points, notes, hard_fail))
    return results


def summarize_results(data: dict[str, Any], results: list[CaseResult] | None) -> dict[str, Any]:
    if not results:
        return {"status": "ok", "scored": False}
    scored = [r for r in results if r.score is not None]
    if not scored:
        return {
            "status": "incomplete",
            "scored": False,
            "scored_cases": 0,
            "score": 0,
            "max_score": 0,
            "hard_fail": any(r.hard_fail for r in results),
        }
    max_total = sum(r.max_points for r in scored)
    total = sum(r.score or 0 for r in scored)
    hard_fail = any(r.hard_fail for r in results)
    if len(scored) != len(results):
        return {
            "status": "incomplete",
            "scored": False,
            "scored_cases": len(scored),
            "score": total,
            "max_score": max_total,
            "hard_fail": hard_fail,
        }
    percent = (total / max_total * 100) if max_total else None
    threshold = float(data["scoring"]["pass_threshold_percent"])
    status = "passed"
    if hard_fail or (percent is not None and percent < threshold):
        status = "failed"
    return {
        "status": status,
        "scored": len(scored) == len(results) and status != "incomplete",
        "scored_cases": len(scored),
        "score": total,
        "max_score": max_total,
        "hard_fail": hard_fail,
    }


def render_report(data: dict[str, Any], eval_path: Path, mode: str, results: list[CaseResult] | None) -> str:
    now = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
    summary = summarize_results(data, results)
    scored = summary["scored"]
    lines: list[str] = []
    lines.append(f"# Skill Eval Report — {data['skill_name']}")
    lines.append("")
    lines.append(f"- Generated: `{now}`")
    lines.append(f"- Eval pack: `{eval_path}`")
    lines.append(f"- Skill path: `{data['skill_path']}`")
    lines.append(f"- Mode: `{mode}`")
    lines.append(f"- Dry-run: `{str(mode == 'dry_run').lower()}`")
    lines.append(f"- Scored: `{str(scored).lower()}`")
    lines.append(f"- Purpose: {data['purpose']}")
    lines.append("")
    lines.append("## Scoring")
    scoring = data["scoring"]
    lines.append(f"- Case max points: `{scoring['case_max_points']}`")
    lines.append(f"- Pass threshold: `{scoring['pass_threshold_percent']}%`")
    lines.append("- Hard fail conditions:")
    for item in scoring.get("hard_fail_conditions", []):
        lines.append(f"  - {item}")
    lines.append("")

    if results:
        lines.append("## Manual Score Summary")
        lines.append(f"- Scored cases: `{summary['scored_cases']}/{len(results)}`")
        lines.append(f"- Score: `{summary['score']}/{summary['max_score']}`")
        lines.append(f"- Hard fail: `{str(summary['hard_fail']).lower()}`")
        lines.append(f"- Status: `{summary['status']}`")
        lines.append("")
        lines.append("### Case Scores")
        for result in results:
            score_txt = "not scored" if result.score is None else f"{result.score}/{result.max_points}"
            lines.append(
                f"- `{result.case_id}` {result.title}: {score_txt}; "
                f"hard_fail={str(result.hard_fail).lower()}; notes={result.notes or ''}"
            )
        lines.append("")

    lines.append("## Cases")
    lines.append("")
    for case in data["cases"]:
        lines.append(render_case(case))
    return "\n".join(lines).rstrip() + "\n"


def default_report_path(eval_path: Path) -> Path:
    return _reports_dir() / f"{eval_path.stem}-eval-report.md"


def skill_eval_run(args: dict, **_: Any) -> str:
    """Run a local skill-eval pack and write a Markdown report.

    Returns a JSON string because Hermes tool handlers use JSON-encoded results.
    """
    try:
        mode = str(args.get("mode", "dry_run")).strip() or "dry_run"
        if mode not in {"dry_run", "manual_score"}:
            return _json_error("mode must be 'dry_run' or 'manual_score'")

        eval_path = _resolve_existing_path(str(args.get("eval_path", "")), "eval_path")
        _ensure_eval_path_allowed(eval_path)

        score_file = None
        if args.get("score_file"):
            score_file = _resolve_existing_path(str(args.get("score_file")), "score_file")
            _ensure_eval_path_allowed(score_file)
        if mode == "manual_score" and score_file is None:
            return _json_error("score_file is required when mode is manual_score")
        if mode == "dry_run" and score_file is not None:
            return _json_error("score_file is not allowed when mode is dry_run")

        report_path = _resolve_output_path(args.get("report_path"), default_report_path(eval_path))
        _ensure_report_path_allowed(report_path)

        data = load_json(eval_path)
        errors = validate_pack(data)
        if errors:
            return _json_error("eval pack schema validation failed", errors=errors)

        scores = load_scores(score_file) if score_file else {}
        results = build_results(data, scores) if mode == "manual_score" else None
        report = render_report(data, eval_path, mode, results)
        _write_report_safely(report_path, report)

        summary = summarize_results(data, results)
        payload: dict[str, Any] = {
            "status": summary["status"],
            "mode": mode,
            "skill_name": data["skill_name"],
            "eval": str(eval_path),
            "report": str(report_path),
            "case_count": len(data["cases"]),
            "scored": summary["scored"],
            "truthfulness_note": "dry_run reports are unscored and do not test model behavior" if mode == "dry_run" else "manual_score uses provided score_file evidence only",
        }
        for key in ("scored_cases", "score", "max_score", "hard_fail"):
            if key in summary:
                payload[key] = summary[key]
        return json.dumps(payload, indent=2)
    except (ValueError, OSError, TypeError, OverflowError) as exc:
        return _json_error(str(exc))


SKILL_EVAL_RUN_SCHEMA = {
    "description": (
        "Run a local skill evaluation pack and write a Markdown report. "
        "Dry-run mode is unscored and never claims model behavior was tested."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "eval_path": {
                "type": "string",
                "description": "Absolute path to an eval JSON pack under $HERMES_HOME/aion-evolution/evals.",
            },
            "mode": {
                "type": "string",
                "enum": ["dry_run", "manual_score"],
                "default": "dry_run",
            },
            "score_file": {
                "type": "string",
                "description": "Manual score JSON file under the evals root. Required for manual_score; forbidden for dry_run.",
            },
            "report_path": {
                "type": "string",
                "description": "Optional Markdown report path under $HERMES_HOME/aion-evolution/evals/reports.",
            },
            "json_summary": {
                "type": "boolean",
                "description": "Accepted for CLI parity; native tool always returns JSON.",
                "default": True,
            },
        },
        "required": ["eval_path"],
    },
}


registry.register(
    name="skill_eval_run",
    toolset="skills",
    schema=SKILL_EVAL_RUN_SCHEMA,
    handler=lambda args, **kwargs: skill_eval_run(args, **kwargs),
    description=SKILL_EVAL_RUN_SCHEMA["description"],
    emoji="🧪",
)
