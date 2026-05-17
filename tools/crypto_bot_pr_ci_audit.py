#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Callable

SCHEMA = "hermes.autonomy.crypto_bot_pr_ci_audit.v1"
DEFAULT_STATE_ROOT = Path("/Users/preston/.local/state/hermes-operator")
DEFAULT_REPO_ROOT = Path("/Users/preston/robinhood/crypto_bot")
DEFAULT_PR_EVIDENCE = (
    DEFAULT_STATE_ROOT / "pr-evidence/20260513T182102Z-S006-8be208b.json"
)
DEFAULT_COMPLETION_GATE = (
    DEFAULT_STATE_ROOT / "completion-gates/20260513T182025Z-S006-8be208b.json"
)
PASSING_CONCLUSIONS = {"PASS", "pass", "passed"}
VALID_CI_STATES = {"absent", "pending", "failed", "passed", "stale", "inaccessible"}
ApiGet = Callable[[str], dict[str, Any]]


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


def read_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def read_text(path: Path | None) -> str:
    if path is None or not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def sha256_file(path: Path) -> str | None:
    if not path.exists():
        return None
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def sanitize_url(value: str) -> str:
    parsed = urllib.parse.urlparse(value)
    if not parsed.scheme or not parsed.netloc or "@" not in parsed.netloc:
        return value
    host = parsed.hostname or ""
    if parsed.port:
        host = f"{host}:{parsed.port}"
    return urllib.parse.urlunparse(
        (parsed.scheme, host, parsed.path, parsed.params, parsed.query, parsed.fragment)
    )


def normalize_gitea_api_base(url: str) -> str:
    stripped = url.rstrip("/")
    if stripped.endswith("/api/v1"):
        return stripped
    return stripped + "/api/v1"


def api_url(api_base: str, owner: str, repo: str, suffix: str) -> str:
    quoted_owner = urllib.parse.quote(owner, safe="")
    quoted_repo = urllib.parse.quote(repo, safe="")
    return f"{api_base}/repos/{quoted_owner}/{quoted_repo}{suffix}"


def query_url(base: str, params: dict[str, str]) -> str:
    return base + "?" + urllib.parse.urlencode(params)


def api_get_json(url: str, *, timeout: int = 5) -> dict[str, Any]:
    record: dict[str, Any] = {
        "url": sanitize_url(url),
        "status": None,
        "data": None,
        "error": None,
    }
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            raw = resp.read(200000).decode("utf-8", errors="replace")
            record["status"] = resp.status
            try:
                record["data"] = json.loads(raw)
            except json.JSONDecodeError:
                record["body_prefix"] = raw[:500]
    except urllib.error.HTTPError as exc:
        raw = exc.read(2000).decode("utf-8", errors="replace")
        record["status"] = exc.code
        try:
            record["data"] = json.loads(raw)
        except json.JSONDecodeError:
            record["body_prefix"] = raw[:500]
    except Exception as exc:  # noqa: BLE001 - audits report probe errors as data
        record["error"] = type(exc).__name__ + ": " + str(exc)
    return record


def gate_passes(
    gate: dict[str, Any] | None,
    *,
    source_branch: str,
    source_head: str,
) -> tuple[bool, list[str]]:
    blockers: list[str] = []
    if not gate:
        return False, ["completion_gate_missing_or_invalid"]
    if gate.get("gate_passed") is not True:
        blockers.append("completion_gate_did_not_pass")
    if str(gate.get("conclusion") or "") not in PASSING_CONCLUSIONS:
        blockers.append("completion_gate_conclusion_not_pass")
    if str(gate.get("task_id") or gate.get("session_id") or "") != "S006":
        blockers.append("completion_gate_task_id_not_s006")
    if str(gate.get("target_branch") or "") != source_branch:
        blockers.append("completion_gate_source_branch_mismatch")
    if str(gate.get("target_full_head") or "") != source_head:
        blockers.append("completion_gate_source_head_mismatch")
    return not blockers, blockers


def evidence_packet_passes(
    packet: dict[str, Any] | None,
    *,
    source_branch: str,
    source_head: str,
    target_branch: str,
) -> tuple[bool, list[str]]:
    blockers: list[str] = []
    if not packet:
        return False, ["pr_evidence_packet_missing_or_invalid"]
    if packet.get("pr_evidence_ready") is not True:
        blockers.append("pr_evidence_packet_not_ready")
    if str(packet.get("task_id") or "") != "S006":
        blockers.append("pr_evidence_task_id_not_s006")
    if str(packet.get("source_branch") or "") != source_branch:
        blockers.append("pr_evidence_source_branch_mismatch")
    if str(packet.get("source_head") or "") != source_head:
        blockers.append("pr_evidence_source_head_mismatch")
    if str(packet.get("target_branch") or "") != target_branch:
        blockers.append("pr_evidence_target_branch_mismatch")
    if packet.get("gate_pass") is not True:
        blockers.append("pr_evidence_gate_not_pass")
    if packet.get("blockers"):
        blockers.append("pr_evidence_packet_has_blockers")
    return not blockers, blockers


def pr_body_link_status(
    *,
    body: str,
    completion_gate_path: Path,
    evidence_packet_path: Path,
    sidecar_path: Path | None,
) -> dict[str, Any]:
    checks = {
        "mentions_s006": "S006" in body,
        "links_completion_gate_path": str(completion_gate_path) in body,
        "links_sidecar_path": bool(sidecar_path and str(sidecar_path) in body),
        "links_pr_evidence_packet_path": str(evidence_packet_path) in body,
    }
    return {
        **checks,
        "ok": checks["mentions_s006"]
        and checks["links_completion_gate_path"]
        and (checks["links_sidecar_path"] or checks["links_pr_evidence_packet_path"]),
    }


def pull_identity(
    pull: dict[str, Any] | None,
    *,
    source_branch: str,
    source_head: str,
    target_branch: str,
) -> dict[str, Any]:
    if not pull:
        return {"matches": False, "head_ref": None, "head_sha": None, "base_ref": None}
    head = pull.get("head") if isinstance(pull.get("head"), dict) else {}
    base = pull.get("base") if isinstance(pull.get("base"), dict) else {}
    head_refs = {
        str(head.get("ref") or ""),
        str(head.get("label") or "").split(":")[-1],
    }
    head_shas = {
        str(head.get("sha") or ""),
        str(pull.get("head_sha") or ""),
    }
    base_refs = {
        str(base.get("ref") or ""),
        str(base.get("label") or "").split(":")[-1],
    }
    matches = (
        source_branch in head_refs
        and source_head in head_shas
        and target_branch in base_refs
    )
    return {
        "matches": matches,
        "head_ref": str(head.get("ref") or ""),
        "head_sha": str(head.get("sha") or pull.get("head_sha") or ""),
        "base_ref": str(base.get("ref") or ""),
        "state": pull.get("state"),
        "title": pull.get("title"),
        "url": pull.get("html_url") or pull.get("url"),
    }


def pull_number(pull: dict[str, Any]) -> int | None:
    raw = pull.get("number") or pull.get("index") or pull.get("id")
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


def discover_matching_pull(
    *,
    api_base: str,
    owner: str,
    repo: str,
    source_branch: str,
    source_head: str,
    target_branch: str,
    api_get: ApiGet,
) -> dict[str, Any]:
    record = api_get(
        query_url(api_url(api_base, owner, repo, "/pulls"), {"state": "all"})
    )
    status = record.get("status")
    result: dict[str, Any] = {
        "readable": status == 200,
        "status": status,
        "error": record.get("error"),
        "number": None,
        "url": None,
        "match_count": 0,
        "pull": None,
    }
    pulls = record.get("data")
    if status != 200 or not isinstance(pulls, list):
        return result
    matches: list[dict[str, Any]] = []
    branch_matches: list[dict[str, Any]] = []
    for pull in pulls:
        if not isinstance(pull, dict):
            continue
        identity = pull_identity(
            pull,
            source_branch=source_branch,
            source_head=source_head,
            target_branch=target_branch,
        )
        if identity["matches"]:
            matches.append(pull)
            continue
        if (
            identity.get("head_ref") == source_branch
            and identity.get("base_ref") == target_branch
        ):
            branch_matches.append(pull)
    result["match_count"] = len(matches)
    result["branch_match_count"] = len(branch_matches)
    candidates = matches or branch_matches
    if not candidates:
        return result
    selected = next(
        (
            pull
            for pull in candidates
            if str(pull.get("state") or "").lower() == "open"
        ),
        candidates[0],
    )
    result.update(
        {
            "number": pull_number(selected),
            "url": selected.get("html_url") or selected.get("url"),
            "pull": selected,
            "matched_by": "exact_head" if matches else "branch_target_fallback",
        }
    )
    return result


def normalize_ci_state(value: str | None) -> str | None:
    state = str(value or "").lower()
    if state in {"success", "passed", "pass"}:
        return "passed"
    if state in {"failure", "failed", "error", "cancelled", "canceled"}:
        return "failed"
    if state in {"pending", "running", "created", "waiting", "blocked"}:
        return "pending"
    return None


def status_sort_key(item: dict[str, Any], index: int) -> tuple[str, int, int]:
    """Return a stable recency key for a Gitea commit status item."""
    timestamp = str(item.get("updated_at") or item.get("created_at") or "")
    raw_id = item.get("id")
    try:
        numeric_id = int(raw_id)
    except (TypeError, ValueError):
        numeric_id = -1
    return (timestamp, numeric_id, index)


def latest_statuses_by_context(statuses: list[Any]) -> list[dict[str, Any]]:
    """Collapse Gitea's append-only statuses to the latest row per context."""
    latest: dict[str, tuple[tuple[str, int, int], dict[str, Any]]] = {}
    anonymous: list[dict[str, Any]] = []
    for index, item in enumerate(statuses):
        if not isinstance(item, dict):
            continue
        context = str(item.get("context") or item.get("name") or "")
        if not context:
            anonymous.append(item)
            continue
        key = status_sort_key(item, index)
        current = latest.get(context)
        if current is None or key > current[0]:
            latest[context] = (key, item)
    return [item for _, item in latest.values()] + anonymous


def classify_ci_state(
    *,
    statuses_record: dict[str, Any],
    combined_record: dict[str, Any],
    pr_head_matches: bool,
) -> dict[str, Any]:
    if not pr_head_matches:
        return {
            "ci_state": "stale",
            "statuses_count": 0,
            "combined_state": None,
            "blocker": "ci_evidence_stale_for_nonmatching_pr_head",
        }
    if statuses_record.get("status") not in {200, 404} or combined_record.get(
        "status"
    ) not in {200, 404}:
        return {
            "ci_state": "inaccessible",
            "statuses_count": 0,
            "combined_state": None,
            "blocker": "ci_evidence_inaccessible",
        }
    statuses_data = statuses_record.get("data")
    raw_statuses = statuses_data if isinstance(statuses_data, list) else []
    statuses = latest_statuses_by_context(raw_statuses)
    combined_data = combined_record.get("data")
    combined_state = (
        normalize_ci_state(combined_data.get("state"))
        if isinstance(combined_data, dict)
        else None
    )
    raw_states = [
        normalize_ci_state(item.get("state") or item.get("status"))
        for item in statuses
        if isinstance(item, dict)
    ]
    states = [state for state in raw_states if state]
    total_count = (
        combined_data.get("total_count")
        if isinstance(combined_data, dict)
        else len(states)
    )
    if not states and not combined_state and not total_count:
        return {
            "ci_state": "absent",
            "statuses_count": 0,
            "combined_state": combined_state,
            "blocker": "ci_evidence_absent",
        }
    if "failed" in states or combined_state == "failed":
        ci_state = "failed"
    elif "pending" in states or combined_state == "pending":
        ci_state = "pending"
    elif states and all(state == "passed" for state in states):
        ci_state = "passed"
    elif combined_state == "passed":
        ci_state = "passed"
    else:
        ci_state = "inaccessible"
    blocker = None if ci_state == "passed" else f"ci_evidence_{ci_state}"
    return {
        "ci_state": ci_state,
        "statuses_count": len(states),
        "combined_state": combined_state,
        "blocker": blocker,
    }


def remote_lifecycle_state(
    *,
    pr_exists: bool,
    ci_state: str,
    ready_for_merge: bool,
) -> str:
    if not pr_exists:
        return "pr_absent"
    if ready_for_merge:
        return "merge_ready"
    if ci_state == "passed":
        return "pr_created_ci_passed_merge_pending"
    if ci_state == "failed":
        return "pr_created_ci_failed"
    if ci_state == "stale":
        return "pr_created_ci_stale_head_mismatch"
    return "pr_created_ci_pending"


def durable_audit_path(state_root: Path) -> Path:
    audit_dir = state_root / "pr-ci-audits"
    audit_dir.mkdir(parents=True, exist_ok=True)
    timestamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = audit_dir / f"{timestamp}-S006-pr-ci-audit.json"
    if not path.exists():
        return path
    for idx in range(1, 100):
        candidate = audit_dir / f"{timestamp}-S006-pr-ci-audit-{idx}.json"
        if not candidate.exists():
            return candidate
    raise RuntimeError("unable to allocate durable PR/CI audit path")


def write_durable_audit(report: dict[str, Any], state_root: Path) -> Path:
    path = durable_audit_path(state_root)
    report["audit_json_path"] = str(path)
    path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return path


def evaluate_pr_ci_audit(
    *,
    repo_root: Path = DEFAULT_REPO_ROOT,
    gitea_url: str = "http://127.0.0.1:3005",
    owner: str = "preston",
    repo: str = "crypto_bot",
    pr_number: int | None = None,
    source_branch: str = "hermes/dev13-006-daemon-trust-contract-mapping",
    source_head: str = "8be208ba317972da03060eb0170a40d2a678aa99",
    target_branch: str = "main",
    pr_evidence_packet: Path = DEFAULT_PR_EVIDENCE,
    completion_gate: Path = DEFAULT_COMPLETION_GATE,
    state_root: Path = DEFAULT_STATE_ROOT,
    api_get: ApiGet | None = None,
    write_artifact: bool = True,
) -> dict[str, Any]:
    api_get = api_get or api_get_json
    api_base = normalize_gitea_api_base(gitea_url)
    blockers: list[str] = []
    warnings: list[str] = []
    packet = read_json(pr_evidence_packet)
    gate = read_json(completion_gate)
    gate_ok, gate_blockers = gate_passes(
        gate,
        source_branch=source_branch,
        source_head=source_head,
    )
    packet_ok, packet_blockers = evidence_packet_passes(
        packet,
        source_branch=source_branch,
        source_head=source_head,
        target_branch=target_branch,
    )
    blockers.extend(gate_blockers)
    blockers.extend(packet_blockers)
    pr_body_path = Path(str((packet or {}).get("pr_body_path") or ""))
    sidecar_path_raw = (packet or {}).get("sidecar_result_path")
    sidecar_path = Path(str(sidecar_path_raw)) if sidecar_path_raw else None
    planned_body_status = pr_body_link_status(
        body=read_text(pr_body_path),
        completion_gate_path=completion_gate,
        evidence_packet_path=pr_evidence_packet,
        sidecar_path=sidecar_path,
    )
    if not planned_body_status["ok"]:
        warnings.append("planned_pr_body_does_not_link_all_evidence_paths")

    discovery = discover_matching_pull(
        api_base=api_base,
        owner=owner,
        repo=repo,
        source_branch=source_branch,
        source_head=source_head,
        target_branch=target_branch,
        api_get=api_get,
    )
    discovered_pr_number = discovery.get("number")
    resolved_pr_number = pr_number if pr_number is not None else discovered_pr_number

    pull_record: dict[str, Any] = {"status": None, "data": None, "error": None}
    if resolved_pr_number is not None:
        pull_record = api_get(
            api_url(api_base, owner, repo, f"/pulls/{resolved_pr_number}")
        )
    pull_data = pull_record.get("data") if pull_record.get("status") == 200 else None
    pull = pull_data if isinstance(pull_data, dict) else None
    pr_exists = pull is not None
    pr_accessible = pull_record.get("status") in {200, 404} or (
        resolved_pr_number is None and discovery["readable"]
    )
    if not pr_accessible:
        blockers.append("s006_pr_inaccessible")
    elif not pr_exists:
        blockers.append("s006_pr_absent")
    identity = pull_identity(
        pull,
        source_branch=source_branch,
        source_head=source_head,
        target_branch=target_branch,
    )
    if pr_exists and not identity["matches"]:
        blockers.append("s006_pr_identity_mismatch")
    actual_body_status = None
    if pr_exists:
        actual_body_status = pr_body_link_status(
            body=str(pull.get("body") or ""),
            completion_gate_path=completion_gate,
            evidence_packet_path=pr_evidence_packet,
            sidecar_path=sidecar_path,
        )
        if not actual_body_status["ok"]:
            blockers.append("s006_pr_body_missing_evidence_links")

    statuses_record = api_get(f"{api_base}/repos/{owner}/{repo}/statuses/{source_head}")
    combined_record = api_get(
        f"{api_base}/repos/{owner}/{repo}/commits/{source_head}/status"
    )
    ci = classify_ci_state(
        statuses_record=statuses_record,
        combined_record=combined_record,
        pr_head_matches=(not pr_exists or bool(identity["matches"])),
    )
    ci_state = str(ci["ci_state"])
    if ci["blocker"]:
        blockers.append(str(ci["blocker"]))

    ready_for_merge = (
        pr_exists
        and bool(identity["matches"])
        and gate_ok
        and packet_ok
        and ci_state == "passed"
        and bool(actual_body_status and actual_body_status["ok"])
    )
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "generated_at": utc_now(),
        "repo_root": str(repo_root),
        "gitea_url": sanitize_url(gitea_url),
        "owner": owner,
        "repo": repo,
        "requested_pr_number": pr_number,
        "discovered_pr_number": discovered_pr_number,
        "pr_number": resolved_pr_number,
        "pr_discovery": {
            "readable": discovery["readable"],
            "status": discovery.get("status"),
            "match_count": discovery.get("match_count"),
            "branch_match_count": discovery.get("branch_match_count"),
            "matched_by": discovery.get("matched_by"),
            "url": discovery.get("url"),
            "error": discovery.get("error"),
        },
        "source_branch": source_branch,
        "source_head": source_head,
        "target_branch": target_branch,
        "pr_evidence_packet_path": str(pr_evidence_packet),
        "pr_evidence_packet_sha256": sha256_file(pr_evidence_packet),
        "completion_gate_path": str(completion_gate),
        "completion_gate_sha256": sha256_file(completion_gate),
        "pr_exists": pr_exists,
        "pr_accessible": pr_accessible,
        "pr_api_status": pull_record.get("status"),
        "pr_identity": identity,
        "pr_matches_spec": bool(pr_exists and identity["matches"]),
        "completion_gate_pass": gate_ok,
        "completion_gate_blockers": gate_blockers,
        "evidence_packet_pass": packet_ok,
        "evidence_packet_blockers": packet_blockers,
        "planned_pr_body_link_status": planned_body_status,
        "actual_pr_body_link_status": actual_body_status,
        "ci_state": ci_state,
        "ci": ci,
        "ci_status_api_status": statuses_record.get("status"),
        "ci_combined_api_status": combined_record.get("status"),
        "s006_remote_lifecycle_state": remote_lifecycle_state(
            pr_exists=pr_exists,
            ci_state=ci_state,
            ready_for_merge=ready_for_merge,
        ),
        "ci_evidence_ready": ci_state == "passed",
        "merge_ready": ready_for_merge,
        "ready_for_merge": ready_for_merge,
        "blockers": sorted(set(blockers)),
        "warnings": warnings,
        "read_only_get_only": True,
    }
    if write_artifact:
        write_durable_audit(report, state_root)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Read-only PR/CI audit for crypto_bot S006."
    )
    parser.add_argument("--repo-root", type=Path, default=DEFAULT_REPO_ROOT)
    parser.add_argument("--gitea-url", default="http://127.0.0.1:3005")
    parser.add_argument("--owner", default="preston")
    parser.add_argument("--repo", default="crypto_bot")
    parser.add_argument("--pr-number", type=int)
    parser.add_argument(
        "--source-branch",
        default="hermes/dev13-006-daemon-trust-contract-mapping",
    )
    parser.add_argument(
        "--source-head",
        default="8be208ba317972da03060eb0170a40d2a678aa99",
    )
    parser.add_argument("--target-branch", default="main")
    parser.add_argument("--pr-evidence-packet", type=Path, default=DEFAULT_PR_EVIDENCE)
    parser.add_argument("--completion-gate", type=Path, default=DEFAULT_COMPLETION_GATE)
    parser.add_argument("--state-root", type=Path, default=DEFAULT_STATE_ROOT)
    parser.add_argument("--no-write-audit", action="store_true")
    parser.add_argument("--format", choices=["json"], default="json")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    report = evaluate_pr_ci_audit(
        repo_root=args.repo_root,
        gitea_url=args.gitea_url,
        owner=args.owner,
        repo=args.repo,
        pr_number=args.pr_number,
        source_branch=args.source_branch,
        source_head=args.source_head,
        target_branch=args.target_branch,
        pr_evidence_packet=args.pr_evidence_packet,
        completion_gate=args.completion_gate,
        state_root=args.state_root,
        write_artifact=not args.no_write_audit,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["merge_ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
