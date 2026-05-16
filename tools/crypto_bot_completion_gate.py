#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

import crypto_bot_policy_scanner as policy_scanner


SCHEMA = "hermes.autonomy.crypto_bot_completion_gate.v1"
SIDECAR_AUDIT_SCHEMA = "hermes.autonomy.crypto_bot_sidecar_audit.v1"
DEFAULT_REPO_ROOT = Path("/Users/preston/robinhood/crypto_bot")
DEFAULT_STATE_ROOT = Path("/Users/preston/.local/state/hermes-operator")
DEFAULT_HERMES_ROOT = Path("/Users/preston/.hermes/hermes-agent")
REPORT_DIR_NAME = "completion-gates"

REQUIRED_PROMPT_COMMANDS = (
    "git status --short --branch",
    "git rev-parse --abbrev-ref HEAD",
    "git rev-parse HEAD",
    "git diff --name-only",
    "git diff --check",
)

CANONICAL_SIDECAR_FIELDS = (
    "Schema",
    "Branch observed",
    "Full HEAD observed",
    "Base/head range audited",
    "Changed files",
    "Worktree status",
    "git diff --check exit code",
    "Blocked-surface scan",
    "Final conclusion",
)

SIDE_CAR_AUDIT_ROOT = DEFAULT_STATE_ROOT / "codex-sidecar-audits"


def utc_timestamp() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def safe_read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def truncate(value: str, limit: int = 20000) -> str:
    if len(value) <= limit:
        return value
    return value[:limit] + "\n...[truncated]..."


def run_command(cwd: Path, argv: list[str]) -> dict[str, Any]:
    started = time.time()
    completed = subprocess.run(
        argv,
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    duration_ms = round((time.time() - started) * 1000)
    stdout = completed.stdout
    stderr = completed.stderr
    combined = stdout + stderr
    return {
        "cwd": str(cwd),
        "command": argv,
        "exit_code": completed.returncode,
        "duration_ms": duration_ms,
        "stdout": truncate(stdout),
        "stderr": truncate(stderr),
        "output_sha256": sha256_text(combined),
    }


def git(repo: Path, args: list[str]) -> dict[str, Any]:
    return run_command(repo, ["git", *args])


def command_output(command: dict[str, Any]) -> str:
    return str(command.get("stdout") or "") + str(command.get("stderr") or "")


def output_lines(command: dict[str, Any]) -> list[str]:
    return [line for line in str(command.get("stdout") or "").splitlines() if line]


def resolve_ref(repo: Path, ref: str) -> tuple[str | None, dict[str, Any]]:
    result = git(repo, ["rev-parse", ref])
    if result["exit_code"] != 0:
        return None, result
    return str(result["stdout"]).strip(), result


def branch_ref_exists(repo: Path, branch: str) -> tuple[str | None, dict[str, Any]]:
    return resolve_ref(repo, f"refs/heads/{branch}")


def parse_yaml_allowlist_section(path: Path, task_id: str) -> dict[str, list[str]]:
    result = {"paths": [], "patterns": [], "sources": []}
    if not path.exists():
        return result
    in_section = False
    current_task: str | None = None
    current_key: str | None = None
    for raw_line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        if not raw_line.startswith(" "):
            in_section = raw_line.strip() == "completion_gate_docs_allowlist:"
            current_task = None
            current_key = None
            continue
        if not in_section:
            continue
        stripped = raw_line.strip()
        if raw_line.startswith("  ") and not raw_line.startswith("    "):
            current_task = stripped[:-1] if stripped.endswith(":") else None
            current_key = None
            continue
        if current_task is None or current_task.lower() != task_id.lower():
            continue
        if raw_line.startswith("    ") and not raw_line.startswith("      "):
            current_key = stripped[:-1] if stripped.endswith(":") else None
            continue
        if (
            current_key in {"paths", "patterns"}
            and raw_line.startswith("      - ")
        ):
            value = stripped[2:].strip().strip("'\"")
            result[current_key].append(value)
            result["sources"].append(
                f"{path}:completion_gate_docs_allowlist.{current_task}.{current_key}"
            )
    return result


def load_strategic_plan_allowlist(
    repo_root: Path,
    task_id: str,
) -> dict[str, list[str]]:
    result = {"paths": [], "patterns": [], "sources": []}
    plan = repo_root / "docs/planning/autoresearch_runpod_to_live_trade/plan.json"
    if not plan.exists():
        return result
    try:
        data = json.loads(plan.read_text(encoding="utf-8", errors="replace"))
    except json.JSONDecodeError:
        return result
    fields = (
        "allowed_write_scope",
        "expected_outputs",
        "primary_outputs",
        "completion_evidence_path",
        "expected_evidence_packet_path",
        "suggested_evidence_dir",
        "related_repo_paths",
    )
    for session in data.get("sessions", []):
        if not isinstance(session, dict):
            continue
        ids = {
            str(session.get("session_id") or ""),
            str(session.get("stable_session_id") or ""),
        }
        if task_id not in ids:
            continue
        for field in fields:
            raw_value = session.get(field)
            values = raw_value if isinstance(raw_value, list) else [raw_value]
            for value in values:
                if not isinstance(value, str) or not value:
                    continue
                key = "patterns" if any(char in value for char in "*?[") else "paths"
                result[key].append(value)
                result["sources"].append(f"{plan}:sessions.{task_id}.{field}")
    return result


def merge_allowlists(*allowlists: dict[str, list[str]]) -> dict[str, Any]:
    merged: dict[str, Any] = {"paths": [], "patterns": [], "sources": []}
    for allowlist in allowlists:
        for key in ("paths", "patterns"):
            for value in allowlist.get(key, []):
                if key == "paths" and not policy_scanner.is_safe_docs_path(value):
                    continue
                if key == "patterns" and not policy_scanner.is_safe_docs_pattern(value):
                    continue
                if value not in merged[key]:
                    merged[key].append(value)
        if merged["paths"] or merged["patterns"]:
            for value in allowlist.get("sources", []):
                if value not in merged["sources"]:
                    merged["sources"].append(value)
    return merged


def task_session_id(task_id: str) -> str | None:
    return task_id if re.fullmatch(r"S\d+[A-Z]?", task_id, re.I) else None


def branch_alias(branch: str) -> str | None:
    match = re.search(r"(dev\d+-\d+)", branch, re.I)
    return match.group(1) if match else None


def collect_introduced_text_by_path(
    repo: Path,
    base: str,
    head: str,
    paths: list[str],
) -> dict[str, str]:
    content: dict[str, str] = {}
    for path in paths:
        normalized = policy_scanner.normalize_repo_path(path)
        if not normalized:
            continue
        completed = subprocess.run(
            [
                "git",
                "diff",
                "--unified=0",
                "--no-ext-diff",
                f"{base}..{head}",
                "--",
                normalized,
            ],
            cwd=repo,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if completed.returncode != 0:
            continue
        introduced_lines = [
            line[1:]
            for line in completed.stdout.splitlines()
            if line.startswith("+") and not line.startswith("+++")
        ]
        content[normalized] = "\n".join(introduced_lines)
    return content


def scan_blocked_surfaces(
    changed_files: list[str],
    *,
    allowlisted_paths: list[str] | tuple[str, ...] = (),
    allowlisted_patterns: list[str] | tuple[str, ...] = (),
    content_by_path: dict[str, str] | None = None,
) -> list[dict[str, str]]:
    return policy_scanner.scan_blocked_surfaces(
        changed_files,
        allowlisted_paths=allowlisted_paths,
        allowlisted_patterns=allowlisted_patterns,
        content_by_path=content_by_path,
    )


def looks_like_prefilled_prompt_conclusion(text: str) -> bool:
    lower = text.lower()
    conclusion_patterns = (
        r"conclusion\s*[:*]+\s*(clean|pass|passed|success|successful)",
        r"\*\*conclusion\*\*\s*:\s*(clean|pass|passed|success|successful)",
    )
    if any(re.search(pattern, lower) for pattern in conclusion_patterns):
        return True
    return "blocked surfaces checked**: none" in lower or (
        "blocked surfaces checked" in lower and ": none" in lower
    )


def result_claims_success(text: str) -> bool:
    parsed = parse_final_conclusion(text)
    if parsed["conclusion"] == "PASS":
        return True
    lower = text.lower()
    return bool(re.search(r"conclusion\s*[:*]+\s*(passed|clean|success)", lower))


def has_exit_code_evidence(value: str) -> bool:
    return bool(re.search(r"\b(exit\s*code|exit_code)\b\s*[:=]?\s*\d+", value, re.I))


def is_diagnostic_or_smoke(path: Path | None, text: str) -> bool:
    parts = str(path or "").lower()
    lower = text.lower()
    return any(
        marker in parts or marker in lower
        for marker in ("diagnostic", "smoke", "short-audit", "short audit")
    )


def normalize_result_value(value: str) -> str:
    normalized = value.strip()
    normalized = re.sub(r"\s+", " ", normalized)
    previous = None
    while previous != normalized:
        previous = normalized
        normalized = normalized.strip()
        normalized = normalized.strip(" \t\r\n.,;")
        if (
            len(normalized) >= 2
            and normalized[0] == normalized[-1]
            and normalized[0] in {"`", "'", '"'}
        ):
            normalized = normalized[1:-1].strip()
        if (
            len(normalized) >= 4
            and normalized.startswith("**")
            and normalized.endswith("**")
        ):
            normalized = normalized[2:-2].strip()
    return normalized


def extract_result_values(text: str, label: str) -> list[str]:
    pattern = re.compile(
        rf"^\s*(?:[-*]\s*)?(?:\*\*)?{re.escape(label)}(?:\*\*)?"
        r"[ \t]*:[ \t]*(.+?)[ \t]*$",
        re.IGNORECASE | re.MULTILINE,
    )
    return [normalize_result_value(match.group(1)) for match in pattern.finditer(text)]


def extract_any_result_values(text: str, labels: tuple[str, ...]) -> list[str]:
    values: list[str] = []
    for label in labels:
        values.extend(extract_result_values(text, label))
    return values


def value_names_base_head_range(
    *,
    value: str,
    base_ref: str,
    base_full: str | None,
    head_ref: str,
    head_full: str | None,
) -> bool:
    normalized = normalize_result_value(value)
    canonical = f"{base_full or base_ref}..{head_full or head_ref}"
    return normalized == canonical


def extract_changed_files_from_result(text: str) -> list[str]:
    values = extract_result_values(text, "Changed files")
    files: list[str] = []
    for value in values:
        normalized = normalize_result_value(value)
        if normalized.lower() in {"none", "no files"}:
            continue
        if normalized:
            files.append(policy_scanner.normalize_repo_path(normalized))

    block_pattern = re.compile(
        r"^\s*(?:[-*]\s*)?(?:\*\*)?Changed files(?:\*\*)?\s*:\s*$",
        re.IGNORECASE | re.MULTILINE,
    )
    for match in block_pattern.finditer(text):
        tail = text[match.end() :]
        for line in tail.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("#"):
                break
            if re.match(r"^(?:[-*]\s*)?(?:\*\*)?[A-Za-z][^:]{1,80}:", stripped):
                break
            value = stripped
            if value.startswith(("-", "*")):
                value = value[1:].strip()
            value = normalize_result_value(value)
            if value.lower() in {"none", "no files"}:
                continue
            files.append(policy_scanner.normalize_repo_path(value))
    return sorted(set(path for path in files if path))


def parse_git_diff_check_exit_code(values: list[str]) -> int | None:
    for value in values:
        match = re.search(r"\b(exit\s*code|exit_code)\b\s*[:=]?\s*(\d+)", value, re.I)
        if match:
            return int(match.group(2))
        bare = re.fullmatch(r"\d+", normalize_result_value(value))
        if bare:
            return int(bare.group(0))
    return None


def blocked_surface_scan_passed(values: list[str]) -> bool:
    for value in values:
        normalized = normalize_result_value(value).lower()
        if not normalized:
            continue
        if normalized == "pass" or normalized.startswith("pass "):
            return True
        if re.search(r"\b(exit\s*code|exit_code)\b\s*[:=]?\s*0\b", normalized):
            return True
        if "no blocked" in normalized or "no matches" in normalized:
            return True
    return False


def parse_final_conclusion(text: str) -> dict[str, Any]:
    values: list[str] = []

    final_label_pattern = re.compile(
        r"^\s*(?:[-*]\s*)?(?:\*\*)?Final conclusion(?:\*\*)?\s*:\s*"
        r"(PASS|FAIL|BLOCKED)\b",
        re.IGNORECASE | re.MULTILINE,
    )
    values.extend(
        match.group(1).upper() for match in final_label_pattern.finditer(text)
    )

    label_pattern = re.compile(
        r"^\s*(?:[-*]\s*)?(?:\*\*)?Conclusion(?:\*\*)?\s*:\s*"
        r"(PASS|FAIL|BLOCKED)\b",
        re.IGNORECASE | re.MULTILINE,
    )
    values.extend(match.group(1).upper() for match in label_pattern.finditer(text))

    heading_pattern = re.compile(
        r"^\s*#{1,6}\s+Conclusion\s*$",
        re.IGNORECASE | re.MULTILINE,
    )
    for match in heading_pattern.finditer(text):
        tail = text[match.end() :]
        for line in tail.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("#"):
                break
            conclusion = re.match(r"^(PASS|FAIL|BLOCKED)\b", stripped, re.I)
            if conclusion:
                values.append(conclusion.group(1).upper())
            break

    unique_values = sorted(set(values))
    return {
        "conclusion": unique_values[0] if len(unique_values) == 1 else None,
        "values": values,
        "ambiguous": len(unique_values) > 1,
        "missing": not values,
    }


def discover_sidecar_artifacts(
    *,
    state_root: Path,
    task_id: str,
) -> tuple[Path | None, Path | None, list[str]]:
    warnings: list[str] = []
    audit_root = state_root / "codex-sidecar-audits"
    if not audit_root.exists():
        warnings.append(f"sidecar audit root missing: {audit_root}")
        return None, None, warnings

    candidates = [
        path
        for path in audit_root.rglob("*")
        if path.is_file() and task_id.lower() in str(path).lower()
    ]
    if not candidates:
        return None, None, warnings

    def score(path: Path) -> tuple[int, float]:
        lowered = str(path).lower()
        value = 0
        if "post-commit" in lowered:
            value += 10
        if "final" in lowered:
            value += 5
        return value, path.stat().st_mtime

    prompts = sorted(
        [path for path in candidates if "prompt" in path.name.lower()],
        key=score,
        reverse=True,
    )
    results = sorted(
        [path for path in candidates if "result" in path.name.lower()],
        key=score,
        reverse=True,
    )
    return (
        prompts[0] if prompts else None,
        results[0] if results else None,
        warnings,
    )


def verify_sidecar_prompt(
    *,
    path: Path | None,
) -> tuple[dict[str, Any], list[str], list[str]]:
    blockers: list[str] = []
    warnings: list[str] = []
    info: dict[str, Any] = {
        "path": str(path) if path else None,
        "exists": bool(path and path.exists()),
        "sha256": None,
        "prefilled_conclusion": False,
        "missing_required_commands": list(REQUIRED_PROMPT_COMMANDS),
        "schema_marker_present": False,
        "canonical_fields_present": [],
        "missing_canonical_fields": list(CANONICAL_SIDECAR_FIELDS),
    }

    if path is None:
        blockers.append("sidecar prompt missing")
        return info, blockers, warnings
    if not path.exists():
        blockers.append(f"sidecar prompt path does not exist: {path}")
        return info, blockers, warnings
    if path.stat().st_size == 0:
        blockers.append(f"sidecar prompt is empty: {path}")
        return info, blockers, warnings

    text = safe_read_text(path)
    info["sha256"] = sha256_file(path)
    info["prefilled_conclusion"] = looks_like_prefilled_prompt_conclusion(text)
    missing = [command for command in REQUIRED_PROMPT_COMMANDS if command not in text]
    info["missing_required_commands"] = missing
    info["schema_marker_present"] = SIDECAR_AUDIT_SCHEMA in text
    present_fields = [field for field in CANONICAL_SIDECAR_FIELDS if field in text]
    info["canonical_fields_present"] = present_fields
    info["missing_canonical_fields"] = [
        field for field in CANONICAL_SIDECAR_FIELDS if field not in text
    ]
    if info["prefilled_conclusion"]:
        blockers.append("sidecar prompt prefilled a clean/pass conclusion")
    if missing:
        blockers.append(
            "sidecar prompt missing required commands: " + ", ".join(missing)
        )
    if not info["schema_marker_present"]:
        warnings.append(
            f"sidecar prompt missing canonical schema marker: {SIDECAR_AUDIT_SCHEMA}"
        )
    return info, blockers, warnings


def verify_sidecar_result(
    *,
    path: Path | None,
    branch: str,
    base_ref: str,
    base_full: str | None,
    head_ref: str,
    head_full: str | None,
    changed_files: list[str],
    diff_check_exit_code: int | None,
) -> tuple[dict[str, Any], list[str], list[str]]:
    blockers: list[str] = []
    warnings: list[str] = []
    info: dict[str, Any] = {
        "path": str(path) if path else None,
        "exists": bool(path and path.exists()),
        "sha256": None,
        "branch_match": False,
        "head_match": False,
        "mentions_base_head_range": False,
        "mentions_git_diff_check": False,
        "mentions_blocked_surface_scan": False,
        "has_exit_code_evidence": False,
        "git_diff_check_exit_code": None,
        "blocked_surface_scan_passed": False,
        "diagnostic_or_smoke": False,
        "timed_out_or_incomplete": False,
        "claims_success": False,
        "schema_marker_present": False,
        "final_conclusion": None,
        "final_conclusion_values": [],
        "final_conclusion_ambiguous": False,
    }

    if path is None:
        blockers.append("sidecar result missing")
        return info, blockers, warnings
    if not path.exists():
        blockers.append(f"sidecar result path does not exist: {path}")
        return info, blockers, warnings
    if path.stat().st_size == 0:
        blockers.append(f"sidecar result is empty: {path}")
        return info, blockers, warnings

    text = safe_read_text(path)
    observed_branches = extract_any_result_values(
        text,
        (
            "Branch observed",
            "Branch",
        ),
    )
    observed_heads = extract_any_result_values(
        text,
        (
            "Full HEAD observed",
            "Full HEAD",
            "HEAD observed",
        ),
    )
    audited_ranges = extract_any_result_values(
        text,
        (
            "Base/head range audited",
            "Base/head range",
            "Range audited",
        ),
    )
    diff_check_values = extract_any_result_values(
        text,
        (
            "git diff --check exit code",
            "git diff --check",
        ),
    )
    blocked_surface_values = extract_any_result_values(
        text,
        (
            "Blocked-surface scan",
            "Blocked surface scan",
        ),
    )
    parsed_conclusion = parse_final_conclusion(text)
    observed_changed_files = extract_changed_files_from_result(text)
    info["sha256"] = sha256_file(path)
    info["schema_marker_present"] = SIDECAR_AUDIT_SCHEMA in text
    info["branch_match"] = any(value == branch for value in observed_branches)
    info["head_match"] = any(value == head_full for value in observed_heads)
    info["observed_branches"] = observed_branches
    info["observed_heads"] = observed_heads
    info["audited_ranges"] = audited_ranges
    info["changed_files_match"] = sorted(changed_files) == observed_changed_files
    info["observed_changed_files"] = observed_changed_files
    info["git_diff_check_values"] = diff_check_values
    info["blocked_surface_scan_values"] = blocked_surface_values
    info["mentions_base_head_range"] = any(
        value_names_base_head_range(
            value=value,
            base_ref=base_ref,
            base_full=base_full,
            head_ref=head_ref,
            head_full=head_full,
        )
        for value in audited_ranges
    )
    info["git_diff_check_exit_code"] = parse_git_diff_check_exit_code(
        diff_check_values
    )
    info["mentions_git_diff_check"] = bool(diff_check_values)
    info["mentions_blocked_surface_scan"] = bool(blocked_surface_values)
    info["has_exit_code_evidence"] = info["git_diff_check_exit_code"] is not None
    info["blocked_surface_scan_passed"] = blocked_surface_scan_passed(
        blocked_surface_values
    )
    info["diagnostic_or_smoke"] = is_diagnostic_or_smoke(path, text)
    info["timed_out_or_incomplete"] = bool(
        re.search(r"\b(timed?\s*out|timeout|incomplete|interrupted)\b", text, re.I)
    )
    info["claims_success"] = result_claims_success(text)
    info["final_conclusion"] = parsed_conclusion["conclusion"]
    info["final_conclusion_values"] = parsed_conclusion["values"]
    info["final_conclusion_ambiguous"] = parsed_conclusion["ambiguous"]

    if not info["schema_marker_present"]:
        blockers.append(
            f"sidecar result missing canonical schema marker: {SIDECAR_AUDIT_SCHEMA}"
        )
    if info["diagnostic_or_smoke"]:
        blockers.append("sidecar result is diagnostic/short/smoke evidence")
    if info["timed_out_or_incomplete"]:
        blockers.append("sidecar result reports timeout or incomplete audit")
    if parsed_conclusion["missing"]:
        blockers.append("sidecar result final conclusion is missing")
    elif parsed_conclusion["ambiguous"]:
        blockers.append("sidecar result final conclusion is ambiguous")
    elif parsed_conclusion["conclusion"] != "PASS":
        blockers.append(
            "sidecar result final conclusion is "
            f"{parsed_conclusion['conclusion']}, not PASS"
        )
    if not info["branch_match"]:
        blockers.append("sidecar result does not name the target branch")
    if not info["head_match"]:
        blockers.append("sidecar result does not name the target full HEAD")
    if not info["mentions_base_head_range"]:
        blockers.append("sidecar result does not mention the exact base/head range")
    if not info["changed_files_match"]:
        blockers.append("sidecar result changed files do not match local diff")
    if not info["mentions_git_diff_check"]:
        blockers.append("sidecar result does not mention git diff --check")
    if not info["has_exit_code_evidence"]:
        blockers.append("sidecar result does not include git diff --check exit code")
    elif info["git_diff_check_exit_code"] != 0:
        blockers.append(
            "sidecar result git diff --check exit code is "
            f"{info['git_diff_check_exit_code']}, not 0"
        )
    if not info["mentions_blocked_surface_scan"]:
        blockers.append("sidecar result does not include blocked-surface proof")
    elif not info["blocked_surface_scan_passed"]:
        blockers.append("sidecar result blocked-surface scan is not PASS")
    if diff_check_exit_code not in (None, 0) and info["claims_success"]:
        blockers.append(
            "sidecar result claims pass while local git diff --check fails"
        )

    return info, blockers, warnings


def write_report(report: dict[str, Any], state_root: Path) -> Path:
    task_id = str(report.get("task_id") or "unknown")
    short_head = str(report.get("short_head") or "unknown")
    report_dir = state_root / REPORT_DIR_NAME
    report_dir.mkdir(parents=True, exist_ok=True)
    path = report_dir / f"{utc_timestamp()}-{task_id}-{short_head}.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return path


def evaluate_completion(
    *,
    repo_root: Path,
    base: str,
    head: str,
    branch: str,
    task_id: str,
    hermes_root: Path = DEFAULT_HERMES_ROOT,
    state_root: Path = DEFAULT_STATE_ROOT,
    sidecar_prompt: Path | None = None,
    sidecar_result: Path | None = None,
    write_json_report: bool = True,
) -> dict[str, Any]:
    blockers: list[str] = []
    warnings: list[str] = []
    validator_commands: list[dict[str, Any]] = []

    if sidecar_prompt is None or sidecar_result is None:
        prompt, result, discovery_warnings = discover_sidecar_artifacts(
            state_root=state_root,
            task_id=task_id,
        )
        warnings.extend(discovery_warnings)
        sidecar_prompt = sidecar_prompt or prompt
        sidecar_result = sidecar_result or result

    session_id = task_session_id(task_id)
    alias = branch_alias(branch)
    claim_id = task_id if task_id.lower().startswith("dev") else None
    descriptor_allowlist = parse_yaml_allowlist_section(
        hermes_root / "projects/crypto_bot/crypto_bot.project.yaml",
        task_id,
    )
    strategic_plan_allowlist = load_strategic_plan_allowlist(repo_root, task_id)
    docs_allowlist = merge_allowlists(
        strategic_plan_allowlist,
        descriptor_allowlist,
    )

    if not repo_root.exists():
        blockers.append(f"repo root missing: {repo_root}")
        report = {
            "schema": SCHEMA,
            "timestamp": utc_timestamp(),
            "managed_project_id": "crypto_bot",
            "task_id": task_id,
            "session_id": session_id,
            "branch_alias": alias,
            "claim_id": claim_id,
            "repo_path": str(repo_root),
            "base_ref": base,
            "target_branch": branch,
            "target_head_ref": head,
            "target_full_head": None,
            "short_head": None,
            "worktree_status": None,
            "changed_files": [],
            "changed_python_files": [],
            "validator_commands": [],
            "allowlist": docs_allowlist,
            "blocked_surface_scan": [],
            "sidecar_prompt": {"path": str(sidecar_prompt) if sidecar_prompt else None},
            "sidecar_result": {"path": str(sidecar_result) if sidecar_result else None},
            "conclusion": "BLOCKED",
            "blockers": blockers,
            "warnings": warnings,
            "gate_passed": False,
        }
        if write_json_report:
            report["report_path"] = str(write_report(report, state_root))
        return report

    base_full, base_cmd = resolve_ref(repo_root, base)
    head_full, head_cmd = resolve_ref(repo_root, head)
    validator_commands.extend([base_cmd, head_cmd])
    if base_full is None:
        blockers.append(f"base ref could not be resolved: {base}")
    if head_full is None:
        blockers.append(f"head ref could not be resolved: {head}")

    branch_full, branch_cmd = branch_ref_exists(repo_root, branch)
    validator_commands.append(branch_cmd)
    if branch_full is None:
        blockers.append(f"target branch does not exist: {branch}")
    elif head_full and branch_full != head_full:
        blockers.append(
            f"target branch {branch} points to {branch_full}, not {head_full}"
        )

    status_cmd = git(repo_root, ["status", "--short", "--branch"])
    current_branch_cmd = git(repo_root, ["rev-parse", "--abbrev-ref", "HEAD"])
    current_head_cmd = git(repo_root, ["rev-parse", "HEAD"])
    name_only_cmd = git(repo_root, ["diff", "--name-only", f"{base}..{head}"])
    diff_check_cmd = git(repo_root, ["diff", "--check", f"{base}..{head}"])
    validator_commands.extend(
        [
            status_cmd,
            current_branch_cmd,
            current_head_cmd,
            name_only_cmd,
            diff_check_cmd,
        ]
    )

    worktree_status = str(status_cmd.get("stdout") or "")
    status_lines = worktree_status.splitlines()
    dirty_lines = [line for line in status_lines if not line.startswith("##")]
    if status_cmd["exit_code"] != 0:
        blockers.append("git status failed")
    elif dirty_lines:
        blockers.append("worktree is dirty")

    current_branch = str(current_branch_cmd.get("stdout") or "").strip()
    if current_branch_cmd["exit_code"] == 0 and current_branch != branch:
        warnings.append(
            f"current checkout is {current_branch}; target branch is {branch}"
        )

    changed_files = output_lines(name_only_cmd)
    if name_only_cmd["exit_code"] != 0:
        blockers.append("git diff --name-only failed")

    if diff_check_cmd["exit_code"] != 0:
        blockers.append("git diff --check failed")

    changed_python_files = [
        path for path in changed_files if path.endswith(".py")
    ]
    ruff_result: dict[str, Any] = {
        "applicable": bool(changed_python_files),
        "available": bool(shutil.which("ruff")),
        "changed_python_files": changed_python_files,
        "command": None,
    }
    if changed_python_files and shutil.which("ruff"):
        ruff_cmd = run_command(repo_root, ["ruff", "check", *changed_python_files])
        validator_commands.append(ruff_cmd)
        ruff_result["command"] = ruff_cmd
        if ruff_cmd["exit_code"] != 0:
            blockers.append("ruff check failed for changed Python files")
    elif changed_python_files:
        warnings.append("ruff not available; changed Python files were not linted")

    introduced_text = collect_introduced_text_by_path(
        repo_root,
        base,
        head,
        changed_files,
    )
    blocked_surfaces = scan_blocked_surfaces(
        changed_files,
        allowlisted_paths=docs_allowlist["paths"],
        allowlisted_patterns=docs_allowlist["patterns"],
        content_by_path=introduced_text,
    )
    blocking_surfaces = policy_scanner.block_findings(blocked_surfaces)
    if blocking_surfaces:
        blockers.append("changed files touch blocked surfaces")

    prompt_info, prompt_blockers, prompt_warnings = verify_sidecar_prompt(
        path=sidecar_prompt
    )
    result_info, result_blockers, result_warnings = verify_sidecar_result(
        path=sidecar_result,
        branch=branch,
        base_ref=base,
        base_full=base_full,
        head_ref=head,
        head_full=head_full,
        changed_files=changed_files,
        diff_check_exit_code=diff_check_cmd["exit_code"],
    )
    blockers.extend(prompt_blockers)
    blockers.extend(result_blockers)
    warnings.extend(prompt_warnings)
    warnings.extend(result_warnings)

    if blocking_surfaces:
        conclusion = "BLOCKED"
    elif blockers:
        conclusion = "FAIL"
    else:
        conclusion = "PASS"

    report: dict[str, Any] = {
        "schema": SCHEMA,
        "timestamp": utc_timestamp(),
        "managed_project_id": "crypto_bot",
        "task_id": task_id,
        "session_id": session_id,
        "branch_alias": alias,
        "claim_id": claim_id,
        "task_source": task_id,
        "repo_path": str(repo_root),
        "base_ref": base,
        "base_full": base_full,
        "target_branch": branch,
        "target_head_ref": head,
        "target_full_head": head_full,
        "short_head": head_full[:7] if head_full else None,
        "current_branch": current_branch,
        "current_head": str(current_head_cmd.get("stdout") or "").strip(),
        "worktree_status": worktree_status,
        "changed_files": changed_files,
        "changed_python_files": changed_python_files,
        "validator_commands": validator_commands,
        "validator_working_directories": sorted(
            {str(command.get("cwd")) for command in validator_commands}
        ),
        "git_diff_check": {
            "exit_code": diff_check_cmd["exit_code"],
            "stdout": diff_check_cmd["stdout"],
            "stderr": diff_check_cmd["stderr"],
            "output_sha256": diff_check_cmd["output_sha256"],
        },
        "ruff_check": ruff_result,
        "targeted_tests": {"applicable": False, "commands": []},
        "allowlist": docs_allowlist,
        "allowlisted_path_source": docs_allowlist["sources"],
        "allowlisted_paths": docs_allowlist["paths"],
        "allowlisted_patterns": docs_allowlist["patterns"],
        "blocked_surface_scan": blocked_surfaces,
        "blocked_surface_blockers": blocking_surfaces,
        "sidecar_prompt": prompt_info,
        "sidecar_result": result_info,
        "sidecar_result_branch_head_match": bool(
            result_info.get("branch_match") and result_info.get("head_match")
        ),
        "machine_verifiable_command_transcript_summary": [
            {
                "cwd": command.get("cwd"),
                "command": command.get("command"),
                "exit_code": command.get("exit_code"),
                "output_sha256": command.get("output_sha256"),
            }
            for command in validator_commands
        ],
        "conclusion": conclusion,
        "gate_passed": conclusion == "PASS",
        "blockers": blockers,
        "warnings": warnings,
    }
    if write_json_report:
        report["report_path"] = str(write_report(report, state_root))
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=DEFAULT_REPO_ROOT)
    parser.add_argument("--base", required=True)
    parser.add_argument("--head", required=True)
    parser.add_argument("--branch", required=True)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--hermes-root", type=Path, default=DEFAULT_HERMES_ROOT)
    parser.add_argument("--state-root", type=Path, default=DEFAULT_STATE_ROOT)
    parser.add_argument("--sidecar-prompt", type=Path)
    parser.add_argument("--sidecar-result", type=Path)
    parser.add_argument("--no-write-report", action="store_true")
    parser.add_argument("--format", choices=("json",), default="json")
    args = parser.parse_args()

    report = evaluate_completion(
        repo_root=args.repo_root,
        base=args.base,
        head=args.head,
        branch=args.branch,
        task_id=args.task_id,
        hermes_root=args.hermes_root,
        state_root=args.state_root,
        sidecar_prompt=args.sidecar_prompt,
        sidecar_result=args.sidecar_result,
        write_json_report=not args.no_write_report,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["gate_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
