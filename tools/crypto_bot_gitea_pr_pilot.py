#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import shutil
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Callable, Optional, TextIO

import crypto_bot_remote_lifecycle_common as common


SCHEMA = "hermes.autonomy.crypto_bot_gitea_pr_pilot.v1"
TOOL_VERSION = "2026-05-13.runtime-self-check.v1"
PR_EVIDENCE_SCHEMA = "hermes.autonomy.crypto_bot_pr_evidence.v1"
COMPLETION_GATE_SCHEMA = "hermes.autonomy.crypto_bot_completion_gate.v1"
RESULT_DIR_NAME = "pr-creations"

ApiPost = Callable[[str, dict[str, Any], Optional[str]], dict[str, Any]]


def _api_url(api_base: str, owner: str, repo: str, suffix: str) -> str:
    quoted_owner = urllib.parse.quote(owner, safe="")
    quoted_repo = urllib.parse.quote(repo, safe="")
    return f"{api_base}/repos/{quoted_owner}/{quoted_repo}{suffix}"


def _query_url(base: str, params: dict[str, str]) -> str:
    return base + "?" + urllib.parse.urlencode(params)


def _redact(value: str, token: str | None = None) -> str:
    text = value
    if token:
        text = text.replace(token, "[REDACTED_TOKEN]")
    return text


def api_post_json(
    url: str,
    payload: dict[str, Any],
    token: str | None,
    *,
    timeout: int = 10,
) -> dict[str, Any]:
    record: dict[str, Any] = {"url": url, "status": None, "data": None, "error": None}
    body = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    if token:
        headers["Authorization"] = f"token {token}"
    request = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as resp:
            raw = resp.read(200000).decode("utf-8", errors="replace")
            record["status"] = resp.status
            record["content_type"] = resp.headers.get("content-type")
            try:
                record["data"] = json.loads(raw)
            except json.JSONDecodeError:
                record["body_prefix"] = _redact(raw[:500], token)
    except urllib.error.HTTPError as exc:
        raw = exc.read(2000).decode("utf-8", errors="replace")
        record["status"] = exc.code
        record["content_type"] = exc.headers.get("content-type")
        record["body_prefix"] = _redact(raw[:500], token)
        try:
            record["data"] = json.loads(raw)
        except json.JSONDecodeError:
            pass
    except Exception as exc:  # noqa: BLE001 - API adapter reports probe errors
        record["error"] = _redact(type(exc).__name__ + ": " + str(exc), token)
    return record


def expected_approval_phrase(packet_path: Path, packet: dict[str, Any]) -> str:
    return (
        "APPROVE GITEA PR PILOT "
        f"task_id={packet.get('task_id')} "
        f"source_branch={packet.get('source_branch')} "
        f"source_head={packet.get('source_head')} "
        f"target_branch={packet.get('target_branch')} "
        f"pr_evidence_packet={packet_path} "
        "no_additional_push=true "
        "exactly_one_pr=true "
        "adapter=crypto_bot_gitea_pr_pilot "
        "validated_command_shape="
        "task-source-head-target-evidence-no-push-create-pr-only-execute "
        "self_check_passed_immediately_before_execution=true "
        "dry_run_passed_immediately_before_execution=true "
        "create_pr_only_preflight_passed_immediately_before_execution=true "
        "prohibit_gh_pr_create=true "
        "prohibit_pr_updates_comments_status_checks=true "
        "prohibit_workflow_runner_starts=true "
        "prohibit_merge=true"
    )


def _source_path() -> Path:
    return Path(__file__).resolve()


def _source_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _expr_contains_none_union(node: ast.AST) -> bool:
    for child in ast.walk(node):
        if not isinstance(child, ast.BinOp) or not isinstance(child.op, ast.BitOr):
            continue
        sides = (child.left, child.right)
        if any(
            isinstance(side, ast.Constant) and side.value is None for side in sides
        ):
            return True
    return False


def runtime_type_alias_findings(path: Path) -> list[dict[str, Any]]:
    """Find top-level runtime aliases that evaluate PEP 604 unions on Python 3.9."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except SyntaxError as exc:
        return [{"line": exc.lineno, "name": "<syntax>", "pattern": str(exc)}]

    findings: list[dict[str, Any]] = []
    for node in tree.body:
        value: ast.AST | None = None
        names: list[str] = []
        if isinstance(node, ast.Assign):
            value = node.value
            for target in node.targets:
                if isinstance(target, ast.Name):
                    names.append(target.id)
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            value = node.value
            if isinstance(node.target, ast.Name):
                names.append(node.target.id)
        if value is None or not _expr_contains_none_union(value):
            continue
        findings.append(
            {
                "line": getattr(node, "lineno", None),
                "name": ", ".join(names) if names else "<top-level-expression>",
                "pattern": "runtime-evaluated | None expression",
            }
        )
    return findings


def build_self_check_payload() -> dict[str, Any]:
    source = _source_path()
    findings = runtime_type_alias_findings(source)
    shebang = source.read_text(encoding="utf-8", errors="replace").splitlines()[0]
    python39 = shutil.which("python3.9")
    python39_help_check: dict[str, Any] | None = None
    if python39:
        proc = subprocess.run(
            [python39, str(source), "--help"],
            env={"PATH": os.environ.get("PATH", "")},
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=15,
        )
        python39_help_check = {
            "python": python39,
            "exit_code": proc.returncode,
            "unsupported_operand_type_error": "unsupported operand type"
            in proc.stderr,
        }
    python39_help_pass = (
        python39_help_check is None or python39_help_check["exit_code"] == 0
    )
    return {
        "schema": SCHEMA,
        "tool_version": TOOL_VERSION,
        "source_file_path": str(source),
        "source_sha256": _source_sha256(source),
        "python_executable": sys.executable,
        "python_version": sys.version.replace("\n", " "),
        "shebang": shebang,
        "direct_execution_supported": os.access(source, os.X_OK),
        "runtime_type_alias_findings": findings,
        "python39_static_compatibility_pass": not findings,
        "python39_help_check": python39_help_check,
        "python39_compatibility_checks_pass": not findings and python39_help_pass,
    }


def _read_approval_phrase(
    *,
    approval_phrase: str | None,
    approval_file: Path | None,
) -> str | None:
    if approval_phrase and approval_file:
        return None
    if approval_phrase:
        return approval_phrase.strip()
    if approval_file:
        return approval_file.read_text(encoding="utf-8").strip()
    return None


def _load_body(packet: dict[str, Any], warnings: list[str]) -> str:
    body_path = packet.get("pr_body_path")
    if body_path:
        path = Path(str(body_path))
        if path.exists():
            return path.read_text(encoding="utf-8")
        warnings.append(f"PR body path is missing; generating fallback body: {path}")
    changed_files = [
        str(path)
        for path in (
            packet.get("completion_gate_changed_files")
            or packet.get("changed_files")
            or []
        )
    ]
    changed = "\n".join(f"- `{path}`" for path in changed_files) or "- none"
    validators = packet.get("validators")
    return (
        f"# {packet.get('task_id')}: autonomous branch evidence\n\n"
        f"- Task id: `{packet.get('task_id')}`\n"
        f"- Source branch: `{packet.get('source_branch')}`\n"
        f"- Source full SHA: `{packet.get('source_head')}`\n"
        f"- Target branch: `{packet.get('target_branch')}`\n"
        f"- Base ref: `{packet.get('base_ref')}`\n"
        f"- Completion gate JSON path: `{packet.get('completion_gate_json_path')}`\n"
        f"- Codex sidecar result path: `{packet.get('sidecar_result_path')}`\n"
        f"- Validators: `{json.dumps(validators, sort_keys=True)}`\n"
        f"- Blocked-surface proof: `{packet.get('blocked_surface_proof')}`\n\n"
        "## Changed Files\n\n"
        f"{changed}\n"
    )


def _load_title(packet: dict[str, Any]) -> str:
    title = str(packet.get("pr_title") or "").strip()
    if title:
        return title
    title_path = packet.get("pr_title_path")
    if title_path and Path(str(title_path)).exists():
        return Path(str(title_path)).read_text(encoding="utf-8").strip()
    task_id = str(packet.get("task_id") or "unknown")
    return f"{task_id}: validated autonomous branch evidence"


def _git_ref(repo_root: Path, ref: str) -> str | None:
    return common.git_stdout(repo_root, ["rev-parse", ref])


def _remote_ref(repo_root: Path, branch: str) -> tuple[str | None, dict[str, Any]]:
    result = common.run_git(repo_root, ["ls-remote", "origin", f"refs/heads/{branch}"])
    if result["exit_code"] != 0:
        return None, result
    lines = [line for line in str(result["stdout"]).splitlines() if line]
    if not lines:
        return None, result
    return lines[0].split()[0], result


def _existing_open_prs(
    pulls: list[Any],
    *,
    source_branch: str,
    target_branch: str,
) -> list[dict[str, Any]]:
    matches: list[dict[str, Any]] = []
    for item in pulls:
        if not isinstance(item, dict):
            continue
        head = item.get("head") if isinstance(item.get("head"), dict) else {}
        base = item.get("base") if isinstance(item.get("base"), dict) else {}
        head_ref = (
            head.get("ref")
            or head.get("name")
            or item.get("head_branch")
            or item.get("head_ref")
            or item.get("head")
        )
        base_ref = (
            base.get("ref")
            or base.get("name")
            or item.get("base_branch")
            or item.get("base_ref")
            or item.get("base")
        )
        state = str(item.get("state") or "open").lower()
        if head_ref == source_branch and base_ref == target_branch and state == "open":
            matches.append(
                {
                    "number": item.get("number") or item.get("index"),
                    "url": item.get("html_url") or item.get("url"),
                    "state": state,
                    "head": head_ref,
                    "base": base_ref,
                }
            )
    return matches


def _validate_body_contents(
    *,
    body: str,
    packet: dict[str, Any],
    blockers: list[str],
) -> None:
    required_values = {
        "task id": str(packet.get("task_id") or ""),
        "source branch": str(packet.get("source_branch") or ""),
        "source full head": str(packet.get("source_head") or ""),
        "target branch": str(packet.get("target_branch") or ""),
        "base ref": str(packet.get("base_ref") or ""),
        "completion gate JSON path": str(packet.get("completion_gate_json_path") or ""),
        "sidecar result path": str(packet.get("sidecar_result_path") or ""),
        "blocked-surface proof": str(packet.get("blocked_surface_proof") or ""),
    }
    changed_files = [
        str(path)
        for path in (
            packet.get("completion_gate_changed_files")
            or packet.get("changed_files")
            or []
        )
    ]
    for label, value in required_values.items():
        if value and value not in body:
            blockers.append(f"PR body missing required {label}: {value}")
    for path in changed_files:
        if path not in body:
            blockers.append(f"PR body missing changed file: {path}")
    if "Validators" not in body and "validators" not in body:
        blockers.append("PR body missing validators evidence")
    if "Blocked-surface proof" not in body:
        blockers.append("PR body missing blocked-surface proof label")


def _completion_gate_path(packet: dict[str, Any]) -> Path | None:
    raw = (
        packet.get("completion_gate_json_path")
        or packet.get("completion_gate_path")
        or packet.get("completion_gate")
    )
    return Path(str(raw)) if raw else None


def evaluate_pr_pilot(
    *,
    repo_root: Path,
    pr_evidence_packet: Path,
    gitea_url: str,
    owner: str,
    repo: str,
    expected_task_id: str | None = None,
    expected_source_branch: str | None = None,
    expected_source_head: str | None = None,
    expected_target_branch: str | None = None,
    no_push: bool = False,
    token_env_var: str = "GITEA_TOKEN",
    execute: bool = False,
    create_pr_only_intent: bool = False,
    preflight_only: bool = False,
    approval_phrase: str | None = None,
    approval_file: Path | None = None,
    state_root: Path = common.DEFAULT_STATE_ROOT,
    api_get: common.ApiGet = common.api_get_json,
    api_post: ApiPost = api_post_json,
    write_result: bool = True,
) -> dict[str, Any]:
    blockers: list[str] = []
    warnings: list[str] = []
    execution_blockers: list[str] = []
    api_base = common.normalize_gitea_api_base(gitea_url)
    assert api_base is not None
    create_endpoint = _api_url(api_base, owner, repo, "/pulls")

    packet: dict[str, Any] = {}
    if not pr_evidence_packet.exists():
        blockers.append(f"PR evidence packet missing: {pr_evidence_packet}")
    else:
        try:
            packet = common.read_json(pr_evidence_packet)
        except json.JSONDecodeError as exc:
            blockers.append(f"PR evidence packet is invalid JSON: {exc}")

    if packet.get("schema") != PR_EVIDENCE_SCHEMA:
        blockers.append("PR evidence packet has unexpected schema")
    if packet.get("pr_evidence_ready") is not True:
        blockers.append("PR evidence packet is not ready")
    if packet.get("blockers"):
        blockers.append("PR evidence packet contains blockers")

    task_id = str(packet.get("task_id") or "")
    source_branch = str(packet.get("source_branch") or "")
    source_head = str(packet.get("source_head") or "")
    target_branch = str(packet.get("target_branch") or "")
    base_ref = str(packet.get("base_ref") or "")

    expected_values = {
        "task_id": expected_task_id,
        "source_branch": expected_source_branch,
        "source_head": expected_source_head,
        "target_branch": expected_target_branch,
    }
    packet_values = {
        "task_id": task_id,
        "source_branch": source_branch,
        "source_head": source_head,
        "target_branch": target_branch,
    }
    for key, expected in expected_values.items():
        if expected and packet_values.get(key) != expected:
            blockers.append(
                f"CLI {key.replace('_', ' ')} does not match PR evidence packet"
            )

    gate_path = _completion_gate_path(packet)
    gate_report: dict[str, Any] = {}
    if not gate_path:
        blockers.append("PR evidence packet does not name a completion gate path")
    elif not gate_path.exists():
        blockers.append(f"Completion gate path is missing: {gate_path}")
    else:
        try:
            gate_report = common.read_json(gate_path)
        except json.JSONDecodeError as exc:
            blockers.append(f"Completion gate JSON is invalid: {exc}")
    if gate_report:
        if gate_report.get("schema") != COMPLETION_GATE_SCHEMA:
            blockers.append("Completion gate has unexpected schema")
        if not (
            gate_report.get("conclusion") == "PASS"
            and gate_report.get("gate_passed") is True
        ):
            blockers.append("Completion gate is not PASS")
        if gate_report.get("task_id") != task_id:
            blockers.append("Completion gate task id does not match packet")

    current_branch = common.git_stdout(
        repo_root,
        ["rev-parse", "--abbrev-ref", "HEAD"],
    )
    current_head = common.git_stdout(repo_root, ["rev-parse", "HEAD"])
    source_ref_head = (
        _git_ref(repo_root, f"refs/heads/{source_branch}") if source_branch else None
    )
    if current_branch != source_branch:
        blockers.append("Current local branch does not match packet source branch")
    if current_head != source_head:
        blockers.append("Current local HEAD does not match packet source head")
    if source_ref_head != source_head:
        blockers.append("Local source branch ref does not match packet source head")
    worktree_clean = common.worktree_clean(repo_root)
    if not worktree_clean:
        blockers.append("Local worktree is dirty")

    remote_source_head, remote_source_probe = (
        _remote_ref(repo_root, source_branch) if source_branch else (None, {})
    )
    if remote_source_head != source_head:
        if remote_source_head:
            blockers.append("Remote source branch points to a different SHA")
        else:
            blockers.append("Remote source branch is missing")
    remote_target_head, remote_target_probe = (
        _remote_ref(repo_root, target_branch) if target_branch else (None, {})
    )
    if not remote_target_head:
        blockers.append("Remote target branch is missing")

    pr_list_url = _query_url(
        create_endpoint,
        {"state": "open", "base_branch": target_branch},
    )
    pulls_probe = api_get(pr_list_url)
    pull_request_api_readable = pulls_probe.get("status") == 200
    existing_open_prs: list[dict[str, Any]] = []
    if pull_request_api_readable and isinstance(pulls_probe.get("data"), list):
        existing_open_prs = _existing_open_prs(
            pulls_probe["data"],
            source_branch=source_branch,
            target_branch=target_branch,
        )
        if existing_open_prs:
            blockers.append("An open PR already exists for source branch to target")
    elif pulls_probe.get("status") == 401:
        blockers.append("Open PR list requires authentication")
    else:
        blockers.append("Open PR list is not readable")

    title = _load_title(packet) if packet else ""
    body = _load_body(packet, warnings) if packet else ""
    if not title:
        blockers.append("PR title is missing")
    if not body:
        blockers.append("PR body is missing")
    if body:
        _validate_body_contents(body=body, packet=packet, blockers=blockers)
        secret_findings = common.secret_findings_in_text(body)
        if secret_findings:
            blockers.append("PR body contains secret-looking content")
    else:
        secret_findings = []

    approval_expected = (
        expected_approval_phrase(pr_evidence_packet, packet) if packet else ""
    )
    approval_actual = _read_approval_phrase(
        approval_phrase=approval_phrase,
        approval_file=approval_file,
    )
    approval_matches = bool(approval_actual and approval_actual == approval_expected)
    token_env_var_present = token_env_var in os.environ
    token_env_var_has_value = bool(os.environ.get(token_env_var))

    dry_run_allowed = not blockers
    execute_allowed = False
    created_pr: dict[str, Any] | None = None
    post_probe: dict[str, Any] | None = None
    result_path: Path | None = None
    if execute:
        if not approval_matches:
            execution_blockers.append("Execute mode requires the exact approval phrase")
        if not token_env_var_has_value:
            execution_blockers.append(
                f"Execute mode requires token env var to be present: {token_env_var}"
            )
        execute_allowed = dry_run_allowed and not execution_blockers
        if execute_allowed:
            token = os.environ.get(token_env_var)
            payload = {
                "head": source_branch,
                "base": target_branch,
                "title": title,
                "body": body,
            }
            try:
                post_probe = api_post(create_endpoint, payload, token)
            except Exception as exc:  # noqa: BLE001 - injected adapters are reported
                post_probe = {
                    "status": None,
                    "error": _redact(type(exc).__name__ + ": " + str(exc), token),
                }
            if post_probe.get("status") == 201 and isinstance(
                post_probe.get("data"),
                dict,
            ):
                data = post_probe["data"]
                created_pr = {
                    "number": data.get("number") or data.get("index"),
                    "url": data.get("html_url") or data.get("url"),
                    "head": source_branch,
                    "base": target_branch,
                }
            else:
                execution_blockers.append(
                    "Gitea PR create endpoint did not return 201"
                )
                execute_allowed = False
        if write_result and post_probe is not None:
            result_dir = state_root / RESULT_DIR_NAME
            result_dir.mkdir(parents=True, exist_ok=True)
            short_head = source_head[:7] if source_head else "unknown"
            stamp = common.utc_timestamp()
            result_path = (
                result_dir / f"{stamp}-{task_id or 'unknown'}-{short_head}.json"
            )

    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "mode": (
            "execute"
            if execute
            else "create-pr-only-preflight"
            if preflight_only
            else "dry-run"
        ),
        "repo_path": str(repo_root),
        "pr_evidence_packet_path": str(pr_evidence_packet),
        "gitea_api_base": api_base,
        "owner": owner,
        "repo": repo,
        "task_id": task_id,
        "source_branch": source_branch,
        "source_head": source_head,
        "target_branch": target_branch,
        "base_ref": base_ref,
        "cli_expected_inputs": {
            key: value for key, value in expected_values.items() if value
        },
        "completion_gate_path": str(gate_path) if gate_path else None,
        "completion_gate_pass": bool(
            gate_report.get("conclusion") == "PASS"
            and gate_report.get("gate_passed") is True
        ),
        "current_branch": current_branch,
        "current_head": current_head,
        "worktree_clean": worktree_clean,
        "remote_source_branch_exists": bool(remote_source_head),
        "remote_source_head": remote_source_head,
        "remote_target_branch_exists": bool(remote_target_head),
        "remote_target_head": remote_target_head,
        "remote_source_probe_exit_code": remote_source_probe.get("exit_code"),
        "remote_target_probe_exit_code": remote_target_probe.get("exit_code"),
        "pull_request_api_readable": pull_request_api_readable,
        "pulls_probe_status": pulls_probe.get("status"),
        "existing_open_prs": existing_open_prs,
        "pr_create_endpoint": create_endpoint,
        "pr_create_method": "POST",
        "pr_create_payload_preview": {
            "head": source_branch,
            "base": target_branch,
            "title": title,
            "body_length": len(body),
        },
        "same_repo_branch_head_format": "branch-only",
        "no_push_guard_present": no_push,
        "git_push_allowed": False,
        "git_push_invoked": False,
        "create_pr_only_intent": create_pr_only_intent or execute or preflight_only,
        "preflight_only": preflight_only,
        "mutation_api_allowed": execute_allowed,
        "token_env_var": token_env_var,
        "token_env_var_present_without_secret_inspection": token_env_var_present,
        "auth_env_present": token_env_var_present,
        "execute_auth_ready": token_env_var_has_value,
        "execute_requires_operator_provided_auth": True,
        "approval_expected": approval_expected,
        "approval_matches": approval_matches,
        "body_secret_findings": secret_findings,
        "dry_run_allowed": dry_run_allowed,
        "would_create_pr": dry_run_allowed and not execute,
        "pr_creation_retry_safe_to_request": dry_run_allowed and not execute,
        "create_pr_only_preflight_allowed": dry_run_allowed and preflight_only,
        "execute_allowed": execute_allowed,
        "created_pr": created_pr,
        "post_probe_status": post_probe.get("status") if post_probe else None,
        "post_probe_error": post_probe.get("error") if post_probe else None,
        "result_path": str(result_path) if result_path else None,
        "blockers": blockers + execution_blockers,
        "warnings": warnings,
    }

    if execute and write_result and result_path:
        safe_payload = dict(payload)
        safe_payload.pop("approval_expected", None)
        result_path.write_text(
            json.dumps(safe_payload, indent=2, sort_keys=True) + "\n"
        )

    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=common.DEFAULT_REPO_ROOT)
    parser.add_argument(
        "--pr-evidence-packet",
        "--evidence-packet",
        dest="pr_evidence_packet",
        type=Path,
    )
    parser.add_argument("--gitea-url", default="http://127.0.0.1:3005")
    parser.add_argument("--owner", default="preston")
    parser.add_argument("--repo", default="crypto_bot")
    parser.add_argument("--task-id")
    parser.add_argument("--source-branch")
    parser.add_argument("--source-head")
    parser.add_argument("--target-branch")
    parser.add_argument("--format", choices=("json",), default="json")
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--version", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--create-pr-only", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--no-push", action="store_true")
    parser.add_argument("--token-env-var", default="GITEA_TOKEN")
    parser.add_argument(
        "--approval-phrase",
        help="Exact Operator approval phrase (or use --approval-file)",
    )
    parser.add_argument(
        "--approval-file",
        type=Path,
        help="Path to non-secret file containing the exact Operator approval phrase",
    )
    parser.add_argument("--state-root", type=Path, default=common.DEFAULT_STATE_ROOT)
    return parser


def run_cli(
    argv: list[str] | None = None,
    *,
    stdout: TextIO = sys.stdout,
    api_get: common.ApiGet = common.api_get_json,
    api_post: ApiPost = api_post_json,
) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.self_check or args.version:
        payload = build_self_check_payload()
        print(json.dumps(payload, indent=2, sort_keys=True), file=stdout)
        return 0 if payload["python39_compatibility_checks_pass"] else 1

    if args.pr_evidence_packet is None:
        parser.error("--pr-evidence-packet/--evidence-packet is required")
    if args.execute and not args.create_pr_only:
        parser.error("--execute requires --create-pr-only")
    if args.preflight_only and not args.create_pr_only:
        parser.error("--preflight-only requires --create-pr-only")
    if args.execute and args.preflight_only:
        parser.error("choose either --execute or --preflight-only, not both")
    if args.dry_run and (args.execute or args.preflight_only):
        parser.error("choose --dry-run, --execute, or --preflight-only")
    mode_count = sum(
        bool(flag)
        for flag in (
            args.dry_run,
            args.execute and args.create_pr_only,
            args.preflight_only and args.create_pr_only,
            args.create_pr_only and not args.execute and not args.preflight_only,
        )
    )
    if mode_count != 1:
        parser.error(
            "choose exactly one mode: --dry-run, --create-pr-only --preflight-only, "
            "or --create-pr-only --execute"
        )

    payload = evaluate_pr_pilot(
        repo_root=args.repo_root,
        pr_evidence_packet=args.pr_evidence_packet,
        gitea_url=args.gitea_url,
        owner=args.owner,
        repo=args.repo,
        expected_task_id=args.task_id,
        expected_source_branch=args.source_branch,
        expected_source_head=args.source_head,
        expected_target_branch=args.target_branch,
        no_push=args.no_push,
        token_env_var=args.token_env_var,
        execute=args.execute and args.create_pr_only,
        create_pr_only_intent=args.create_pr_only,
        preflight_only=args.preflight_only and args.create_pr_only,
        approval_phrase=args.approval_phrase,
        approval_file=args.approval_file,
        state_root=args.state_root,
        api_get=api_get,
        api_post=api_post,
    )
    if args.create_pr_only and not args.execute and not args.preflight_only:
        payload["mode"] = "create-pr-only-blocked"
        payload["would_create_pr"] = False
        payload["pr_creation_retry_safe_to_request"] = False
        payload["create_pr_only_preflight_allowed"] = False
        payload["execute_allowed"] = False
        payload["mutation_api_allowed"] = False
        payload["blockers"] = [
            *payload["blockers"],
            "--create-pr-only requires --preflight-only for validation or --execute "
            "for creation",
        ]
    print(json.dumps(payload, indent=2, sort_keys=True), file=stdout)
    if args.execute:
        allowed = payload["execute_allowed"]
    elif args.preflight_only:
        allowed = payload["create_pr_only_preflight_allowed"]
    elif args.create_pr_only:
        allowed = False
    else:
        allowed = payload["dry_run_allowed"]
    return 0 if allowed else 1


def main() -> int:
    return run_cli()


if __name__ == "__main__":
    raise SystemExit(main())
