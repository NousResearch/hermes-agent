from __future__ import annotations

import fnmatch
import re
from pathlib import PurePosixPath


SECRET_VALUE_RE = re.compile(
    r"(?i)(api[_-]?key|token|password|passwd|private[_ -]?key|secret|credential)"
    r"\s*[:=]\s*['\"]?([A-Za-z0-9_./+=-]{12,})"
)
SECRET_TOKEN_RE = re.compile(
    r"(?i)(sk-[A-Za-z0-9]{20,}|gh[pousr]_[A-Za-z0-9_]{20,}|"
    r"glpat-[A-Za-z0-9_-]{20,}|xox[baprs]-[A-Za-z0-9-]{20,})"
)
SECRET_PATH_RE = re.compile(
    r"(^|/)(\.env($|[./_-])|.*(token|secret|private[-_]?key|cookie|"
    r"credential|password|passwd|api[-_]?key|keychain).*)",
    re.IGNORECASE,
)

SAFE_DOCS_PATTERNS = (
    "docs/contracts/*.md",
    "docs/contracts/**/*.md",
    "docs/development/*.md",
    "docs/development/**/*.md",
    "docs/architecture/*.md",
    "docs/architecture/**/*.md",
)

BLOCKED_PATH_RULES: tuple[tuple[str, re.Pattern[str], bool], ...] = (
    (
        "workflow surface",
        re.compile(r"(^|/)\.gitea/workflows(/|$)", re.I),
        True,
    ),
    ("runtime database", re.compile(r"(^|/).*\.(db|sqlite|sqlite3)$", re.I), True),
    ("log artifact", re.compile(r"(^|/).*\.(log|out|err)$", re.I), True),
    (
        "broker surface",
        re.compile(r"(^|/).*(broker|robinhood|exchange).*(/|$)", re.I),
        True,
    ),
    (
        "trading or financial surface",
        re.compile(
            r"(^|/).*(trading|financial|live[-_]?market|order|account|"
            r"position|wallet).*(/|$)",
            re.I,
        ),
        True,
    ),
    (
        "deploy or GitOps surface",
        re.compile(
            r"(^|/).*(deploy|gitops|kubernetes|k8s|flux|harbor|openbao|"
            r"rabbitmq|redis|temporal).*(/|$)",
            re.I,
        ),
        True,
    ),
    (
        "runner or workflow surface",
        re.compile(r"(^|/).*(runner|workflow|actions?).*(/|$)", re.I),
        True,
    ),
    (
        "runtime service start surface",
        re.compile(
            r"(^|/).*(launchd|scheduler|worker|service|daemon|server).*(/|$)",
            re.I,
        ),
        False,
    ),
)


def normalize_repo_path(path: str) -> str:
    normalized = path.strip().replace("\\", "/")
    while normalized.startswith("./"):
        normalized = normalized[2:]
    return normalized


def path_matches_pattern(path: str, pattern: str) -> bool:
    normalized = normalize_repo_path(path)
    normalized_pattern = normalize_repo_path(pattern)
    if PurePosixPath(normalized).match(normalized_pattern):
        return True
    return fnmatch.fnmatchcase(normalized, normalized_pattern)


def path_matches_any(path: str, patterns: list[str] | tuple[str, ...]) -> bool:
    return any(path_matches_pattern(path, pattern) for pattern in patterns)


def is_safe_docs_path(path: str) -> bool:
    normalized = normalize_repo_path(path)
    return normalized.endswith(".md") and path_matches_any(
        normalized,
        SAFE_DOCS_PATTERNS,
    )


def is_safe_docs_pattern(pattern: str) -> bool:
    normalized = normalize_repo_path(pattern)
    return normalized.startswith(
        (
            "docs/contracts/",
            "docs/development/",
            "docs/architecture/",
        )
    ) and normalized.endswith(".md")


def allowlist_basis(
    path: str,
    *,
    allowlisted_paths: list[str] | tuple[str, ...] = (),
    allowlisted_patterns: list[str] | tuple[str, ...] = (),
) -> str | None:
    normalized = normalize_repo_path(path)
    for allowed in allowlisted_paths:
        if normalize_repo_path(allowed) == normalized:
            return f"path:{normalize_repo_path(allowed)}"
    for pattern in allowlisted_patterns:
        if path_matches_pattern(normalized, pattern):
            return f"pattern:{normalize_repo_path(pattern)}"
    return None


def secret_findings_in_text(text: str) -> list[str]:
    findings: list[str] = []
    if "BEGIN PRIVATE KEY" in text:
        findings.append("private key marker")
    if SECRET_TOKEN_RE.search(text):
        findings.append("secret token pattern")
    for match in SECRET_VALUE_RE.finditer(text):
        value = match.group(2).lower()
        if value in {"redacted", "placeholder", "example", "changeme"}:
            continue
        findings.append(f"secret-looking assignment: {match.group(1)}")
    return sorted(set(findings))


def scan_blocked_surfaces(
    changed_files: list[str],
    *,
    allowlisted_paths: list[str] | tuple[str, ...] = (),
    allowlisted_patterns: list[str] | tuple[str, ...] = (),
    content_by_path: dict[str, str] | None = None,
) -> list[dict[str, str]]:
    findings: list[dict[str, str]] = []
    content_by_path = content_by_path or {}

    for raw_path in changed_files:
        path = normalize_repo_path(raw_path)
        if not path:
            continue
        if SECRET_PATH_RE.search(path):
            findings.append(
                {
                    "path": path,
                    "reason": "secret-like path",
                    "severity": "block",
                }
            )

        docs_basis = allowlist_basis(
            path,
            allowlisted_paths=allowlisted_paths,
            allowlisted_patterns=allowlisted_patterns,
        )
        safe_docs_allowlisted = bool(docs_basis and is_safe_docs_path(path))
        for reason, pattern, always_block in BLOCKED_PATH_RULES:
            if not pattern.search(path):
                continue
            if reason == "runtime service start surface" and safe_docs_allowlisted:
                findings.append(
                    {
                        "path": path,
                        "reason": reason,
                        "severity": "allowed_docs_reference",
                        "allowlist_basis": docs_basis or "",
                    }
                )
                continue
            severity = (
                "block" if always_block or not safe_docs_allowlisted else "warning"
            )
            findings.append({"path": path, "reason": reason, "severity": severity})

        for finding in secret_findings_in_text(content_by_path.get(path, "")):
            findings.append(
                {
                    "path": path,
                    "reason": f"secret-looking content: {finding}",
                    "severity": "block",
                }
            )

    deduped: list[dict[str, str]] = []
    seen: set[tuple[str, str, str, str]] = set()
    for finding in findings:
        key = (
            finding.get("path", ""),
            finding.get("reason", ""),
            finding.get("severity", ""),
            finding.get("allowlist_basis", ""),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(finding)
    return deduped


def block_findings(findings: list[dict[str, str]]) -> list[dict[str, str]]:
    return [
        finding
        for finding in findings
        if finding.get("severity", "block") == "block"
    ]
