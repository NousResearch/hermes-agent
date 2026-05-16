from __future__ import annotations

import datetime as dt
import json
import os
import re
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Callable

import crypto_bot_policy_scanner as policy_scanner


DEFAULT_HERMES_ROOT = Path("/Users/preston/.hermes/hermes-agent")
DEFAULT_REPO_ROOT = Path("/Users/preston/robinhood/crypto_bot")
DEFAULT_STATE_ROOT = Path("/Users/preston/.local/state/hermes-operator")
DEFAULT_PROJECT_DESCRIPTOR = (
    DEFAULT_HERMES_ROOT / "projects/crypto_bot/crypto_bot.project.yaml"
)

LOCAL_HOSTNAMES = {"127.0.0.1", "localhost", "::1"}
PROTECTED_BRANCH_NAMES = {"main", "master", "develop", "production", "release"}

def utc_timestamp() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def run_git(repo: Path, args: list[str], *, timeout: int = 15) -> dict[str, Any]:
    started = time.time()
    env = os.environ.copy()
    env["GIT_TERMINAL_PROMPT"] = "0"
    try:
        proc = subprocess.run(
            ["git", *args],
            cwd=repo,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=timeout,
        )
        return {
            "command": ["git", *args],
            "exit_code": proc.returncode,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
            "duration_ms": round((time.time() - started) * 1000),
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "command": ["git", *args],
            "exit_code": 124,
            "stdout": exc.stdout or "",
            "stderr": exc.stderr or "git command timed out",
            "duration_ms": round((time.time() - started) * 1000),
        }


def git_stdout(repo: Path, args: list[str]) -> str | None:
    result = run_git(repo, args)
    if result["exit_code"] != 0:
        return None
    return str(result["stdout"]).strip()


def git_lines(repo: Path, args: list[str]) -> list[str]:
    out = git_stdout(repo, args)
    if out is None:
        return []
    return [line for line in out.splitlines() if line]


def worktree_clean(repo: Path) -> bool:
    result = run_git(repo, ["status", "--short"])
    return result["exit_code"] == 0 and not str(result["stdout"]).strip()


def sanitize_url(value: str | None) -> str | None:
    if not value:
        return value
    parsed = urllib.parse.urlparse(value)
    if not parsed.scheme or not parsed.netloc:
        return value
    if "@" not in parsed.netloc:
        return value
    host = parsed.hostname or ""
    if parsed.port:
        host = f"{host}:{parsed.port}"
    return urllib.parse.urlunparse(
        (parsed.scheme, host, parsed.path, parsed.params, parsed.query, parsed.fragment)
    )


def remote_contains_userinfo(value: str | None) -> bool:
    if not value:
        return False
    return "@" in urllib.parse.urlparse(value).netloc


def derive_gitea_from_remote(
    remote_url: str | None,
) -> dict[str, str | bool | None]:
    if not remote_url:
        return {
            "api_base": None,
            "owner": None,
            "repo": None,
            "is_loopback": False,
        }
    parsed = urllib.parse.urlparse(remote_url)
    if parsed.scheme not in {"http", "https"}:
        return {
            "api_base": None,
            "owner": None,
            "repo": None,
            "is_loopback": False,
        }
    hostname = parsed.hostname or ""
    is_loopback = hostname in LOCAL_HOSTNAMES
    parts = [part for part in parsed.path.split("/") if part]
    owner = parts[0] if len(parts) >= 2 else None
    repo = parts[1] if len(parts) >= 2 else None
    if repo and repo.endswith(".git"):
        repo = repo[:-4]
    base = urllib.parse.urlunparse(
        (parsed.scheme, parsed.netloc, "/api/v1", "", "", "")
    )
    return {
        "api_base": base if is_loopback and owner and repo else None,
        "owner": owner,
        "repo": repo,
        "is_loopback": is_loopback,
    }


def normalize_gitea_api_base(url: str | None) -> str | None:
    if not url:
        return None
    stripped = url.rstrip("/")
    if stripped.endswith("/api/v1"):
        return stripped
    return stripped + "/api/v1"


def api_get_json(url: str, *, timeout: int = 5) -> dict[str, Any]:
    record: dict[str, Any] = {"url": url, "status": None, "data": None, "error": None}
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            raw = resp.read(200000).decode("utf-8", errors="replace")
            record["status"] = resp.status
            record["content_type"] = resp.headers.get("content-type")
            try:
                record["data"] = json.loads(raw)
            except json.JSONDecodeError:
                record["body_prefix"] = raw[:500]
    except urllib.error.HTTPError as exc:
        raw = exc.read(2000).decode("utf-8", errors="replace")
        record["status"] = exc.code
        record["content_type"] = exc.headers.get("content-type")
        record["body_prefix"] = raw[:500]
        try:
            record["data"] = json.loads(raw)
        except json.JSONDecodeError:
            pass
    except Exception as exc:  # noqa: BLE001 - tool reports probe errors as data
        record["error"] = type(exc).__name__ + ": " + str(exc)
    return record


ApiGet = Callable[[str], dict[str, Any]]


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_policy_flags(hermes_root: Path = DEFAULT_HERMES_ROOT) -> dict[str, bool]:
    descriptor = hermes_root / "projects/crypto_bot/crypto_bot.project.yaml"
    defaults = {
        "controlled_remote_branch_push_enabled": False,
        "one_pr_creation_pilot_enabled": False,
        "pr_updates_comments_status_mutation_enabled": False,
        "merge_authority_enabled": False,
    }
    if not descriptor.exists():
        return defaults
    text = descriptor.read_text(encoding="utf-8", errors="replace")
    for key in list(defaults):
        match = re.search(rf"^\s*{re.escape(key)}\s*:\s*(\S+)\s*$", text, re.M)
        if not match:
            continue
        defaults[key] = match.group(1).strip().lower() in {"true", "yes", "1"}
    return defaults


def find_workflow_files(repo: Path) -> list[str]:
    paths: list[str] = []
    for root_name in (".gitea", ".github"):
        root = repo / root_name
        if not root.exists():
            continue
        for path in sorted(p for p in root.rglob("*") if p.is_file()):
            paths.append(path.relative_to(repo).as_posix())
    return paths


def workflow_risk_findings(
    repo: Path,
    workflow_files: list[str],
) -> list[dict[str, str]]:
    findings: list[dict[str, str]] = []
    for rel in workflow_files:
        findings.append(
            {
                "path": rel,
                "finding": "workflow file is a read-only controlled surface",
            }
        )
        text = (repo / rel).read_text(encoding="utf-8", errors="replace")
        if "workflow_dispatch" in text:
            findings.append(
                {
                    "path": rel,
                    "finding": "manual workflow dispatch surface is blocked",
                }
            )
        for match in re.finditer(r"^\s*runs-on\s*:\s*([^\n#]+)", text, re.M):
            findings.append(
                {
                    "path": rel,
                    "finding": f"runner label required: {match.group(1).strip()}",
                }
            )
        uses_loopback_action = (
            "uses: http://127.0.0.1" in text
            or "uses: http://localhost" in text
        )
        if uses_loopback_action:
            findings.append(
                {
                    "path": rel,
                    "finding": (
                        "local action mirror reference must remain "
                        "provenance-checked"
                    ),
                }
            )
    return findings


def scan_blocked_surfaces(
    changed_files: list[str],
    *,
    allowlisted_paths: list[str] | tuple[str, ...] = (),
    allowlisted_patterns: list[str] | tuple[str, ...] = (),
) -> list[dict[str, str]]:
    return policy_scanner.scan_blocked_surfaces(
        changed_files,
        allowlisted_paths=allowlisted_paths,
        allowlisted_patterns=allowlisted_patterns,
    )


def block_findings(findings: list[dict[str, str]]) -> list[dict[str, str]]:
    return policy_scanner.block_findings(findings)


def secret_findings_in_text(text: str) -> list[str]:
    return policy_scanner.secret_findings_in_text(text)


def find_matching_completion_gate(
    state_root: Path,
    *,
    branch: str | None,
    head: str | None,
) -> Path | None:
    if not branch or not head:
        return None
    gate_root = state_root / "completion-gates"
    if not gate_root.exists():
        return None
    for path in sorted(gate_root.glob("*.json"), reverse=True):
        try:
            data = read_json(path)
        except (OSError, json.JSONDecodeError):
            continue
        if (
            data.get("conclusion") == "PASS"
            and data.get("gate_passed") is True
            and data.get("target_branch") == branch
            and data.get("target_full_head") == head
        ):
            return path
    return None


def is_protected_branch_name(branch: str, target_branch: str | None = None) -> bool:
    if branch in PROTECTED_BRANCH_NAMES:
        return True
    return bool(target_branch and branch == target_branch)
