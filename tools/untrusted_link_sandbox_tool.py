"""Hermes tools for the local untrusted-link sandbox stack.

These tools are thin adapters around the stable shell contract in
``/home/lucky/docker/untrusted-link-sandbox/bin``. They intentionally do not
mount host paths or run target content directly; the Docker Compose stack owns
isolation, quarantine, and reporting.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional

from tools.registry import registry

DEFAULT_SANDBOX_DIR = Path("/home/lucky/docker/untrusted-link-sandbox")
MAX_TIMEOUT_SECONDS = 600
REQUIRED_WRAPPERS = (
    "triage",
    "audit-url",
    "audit-url-cdp",
    "audit-repo",
    "inspect-download",
)


def _sandbox_dir() -> Path:
    return Path(os.environ.get("HERMES_UNTRUSTED_LINK_SANDBOX_DIR", str(DEFAULT_SANDBOX_DIR))).expanduser()


def _tool_available() -> bool:
    root = _sandbox_dir()
    return (
        root.is_dir()
        and (root / "docker-compose.yml").is_file()
        and all(os.access(root / "bin" / name, os.X_OK) for name in REQUIRED_WRAPPERS)
    )


def _coerce_timeout(value: Any) -> int:
    try:
        timeout = int(value)
    except (TypeError, ValueError):
        timeout = 300
    return max(30, min(timeout, MAX_TIMEOUT_SECONDS))


def _containerish_target(target: str) -> str:
    """Convert host quarantine paths to container paths accepted by bin/triage."""
    root = _sandbox_dir().resolve()
    raw = Path(target).expanduser()
    try:
        resolved = raw.resolve(strict=False)
    except OSError:
        return target
    quarantine_root = root / "quarantine"
    try:
        rel = resolved.relative_to(quarantine_root)
    except ValueError:
        return target
    return "/quarantine/" + rel.as_posix()


def _host_path(path_text: Optional[str]) -> Optional[str]:
    if not path_text:
        return path_text
    root = _sandbox_dir()
    if path_text.startswith("/reports/"):
        return str(root / path_text.lstrip("/"))
    if path_text.startswith("/quarantine/"):
        return str(root / path_text.lstrip("/"))
    return path_text


def _read_summary(report_path: str) -> Dict[str, Any]:
    host_report = Path(_host_path(report_path) or report_path)
    summary_path = host_report.with_suffix(".json")
    if not summary_path.is_file():
        return {}
    try:
        data = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {"summary_error": str(exc), "summary_path": str(summary_path)}
    data["summary_path"] = str(summary_path)
    if "report_path" in data:
        data["report_path_host"] = _host_path(data.get("report_path"))
    if "artifacts_path" in data and data.get("artifacts_path"):
        data["artifacts_path_host"] = _host_path(data.get("artifacts_path"))
    return data


def _read_report_excerpt(report_path: str, max_chars: int = 4000) -> str:
    host_report = Path(_host_path(report_path) or report_path)
    try:
        text = host_report.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    return text[:max_chars]


def _run_sandbox(args: list[str], timeout: int) -> str:
    root = _sandbox_dir()
    proc = subprocess.run(
        args,
        cwd=str(root),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
        check=False,
    )
    stdout = proc.stdout.strip()
    stderr = proc.stderr.strip()
    report_path = ""
    for line in reversed(stdout.splitlines()):
        if line.startswith("/reports/") and line.endswith(".md"):
            report_path = line.strip()
            break
    summary = _read_summary(report_path) if report_path else {}
    result = {
        "success": proc.returncode == 0,
        "exit_code": proc.returncode,
        "command": args,
        "sandbox_dir": str(root),
        "report_path": report_path or None,
        "report_path_host": _host_path(report_path) if report_path else None,
        "summary": summary,
        "stdout_tail": stdout[-4000:],
        "stderr_tail": stderr[-4000:],
    }
    if report_path:
        result["report_excerpt"] = _read_report_excerpt(report_path)
    return json.dumps(result, indent=2)


TRIAGE_SCHEMA: Dict[str, Any] = {
    "name": "untrusted_link_triage",
    "description": (
        "Safely triage an untrusted URL, GitHub/GitLab repository, or quarantined download "
        "using the local Docker untrusted-link sandbox. Prefer this before normal web/browser "
        "tools for unfamiliar user-provided links. Returns the generated markdown report path, "
        "JSON summary, and a short report excerpt."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "target": {
                "type": "string",
                "description": "HTTP(S) URL, GitHub/GitLab repo URL, /quarantine/... path, or host quarantine path.",
            },
            "deep": {
                "type": "boolean",
                "description": "For normal URLs, use deeper CDP telemetry mode. Ignored for repositories and files.",
            },
            "timeout_seconds": {
                "type": "integer",
                "description": "Execution timeout, clamped between 30 and 600 seconds. Default 300.",
            },
        },
        "required": ["target"],
    },
}


def untrusted_link_triage(target: str, deep: bool = False, timeout_seconds: int = 300) -> str:
    args = ["./bin/triage"]
    if deep:
        args.append("--deep")
    args.append(_containerish_target(target))
    return _run_sandbox(args, _coerce_timeout(timeout_seconds))


AUDIT_URL_SCHEMA: Dict[str, Any] = {
    "name": "audit_untrusted_url",
    "description": "Audit an untrusted HTTP(S) URL in the Docker browser sandbox. Use deep=true for CDP telemetry.",
    "parameters": {
        "type": "object",
        "properties": {
            "url": {"type": "string", "description": "HTTP(S) URL to audit."},
            "deep": {"type": "boolean", "description": "Use CDP telemetry mode instead of first-pass Playwright mode."},
            "timeout_seconds": {"type": "integer", "description": "Timeout, clamped 30-600 seconds. Default 300."},
        },
        "required": ["url"],
    },
}


def audit_untrusted_url(url: str, deep: bool = False, timeout_seconds: int = 300) -> str:
    return _run_sandbox(["./bin/audit-url-cdp" if deep else "./bin/audit-url", url], _coerce_timeout(timeout_seconds))


AUDIT_REPO_SCHEMA: Dict[str, Any] = {
    "name": "audit_untrusted_repo",
    "description": "Statically audit an untrusted public GitHub/GitLab repository in the Docker auditor sandbox.",
    "parameters": {
        "type": "object",
        "properties": {
            "url": {"type": "string", "description": "HTTPS GitHub/GitLab repository URL."},
            "timeout_seconds": {"type": "integer", "description": "Timeout, clamped 30-600 seconds. Default 300."},
        },
        "required": ["url"],
    },
}


def audit_untrusted_repo(url: str, timeout_seconds: int = 300) -> str:
    return _run_sandbox(["./bin/audit-repo", url], _coerce_timeout(timeout_seconds))


INSPECT_DOWNLOAD_SCHEMA: Dict[str, Any] = {
    "name": "inspect_untrusted_download",
    "description": "Inspect a file already saved under the untrusted-link sandbox quarantine.",
    "parameters": {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Container /quarantine/... path or matching host quarantine path."},
            "timeout_seconds": {"type": "integer", "description": "Timeout, clamped 30-600 seconds. Default 180."},
        },
        "required": ["path"],
    },
}


def inspect_untrusted_download(path: str, timeout_seconds: int = 180) -> str:
    return _run_sandbox(["./bin/inspect-download", _containerish_target(path)], _coerce_timeout(timeout_seconds))


registry.register(
    name="untrusted_link_triage",
    toolset="untrusted_link_sandbox",
    schema=TRIAGE_SCHEMA,
    handler=lambda args, **kw: untrusted_link_triage(
        target=args.get("target", ""),
        deep=bool(args.get("deep", False)),
        timeout_seconds=args.get("timeout_seconds", 300),
    ),
    check_fn=_tool_available,
    emoji="🧪",
    max_result_size_chars=12000,
)

registry.register(
    name="audit_untrusted_url",
    toolset="untrusted_link_sandbox",
    schema=AUDIT_URL_SCHEMA,
    handler=lambda args, **kw: audit_untrusted_url(
        url=args.get("url", ""),
        deep=bool(args.get("deep", False)),
        timeout_seconds=args.get("timeout_seconds", 300),
    ),
    check_fn=_tool_available,
    emoji="🌐",
    max_result_size_chars=12000,
)

registry.register(
    name="audit_untrusted_repo",
    toolset="untrusted_link_sandbox",
    schema=AUDIT_REPO_SCHEMA,
    handler=lambda args, **kw: audit_untrusted_repo(
        url=args.get("url", ""),
        timeout_seconds=args.get("timeout_seconds", 300),
    ),
    check_fn=_tool_available,
    emoji="📦",
    max_result_size_chars=12000,
)

registry.register(
    name="inspect_untrusted_download",
    toolset="untrusted_link_sandbox",
    schema=INSPECT_DOWNLOAD_SCHEMA,
    handler=lambda args, **kw: inspect_untrusted_download(
        path=args.get("path", ""),
        timeout_seconds=args.get("timeout_seconds", 180),
    ),
    check_fn=_tool_available,
    emoji="🦠",
    max_result_size_chars=12000,
)
