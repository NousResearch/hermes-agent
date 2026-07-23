"""Deterministic, read-only quality checks for Hermes skills.

This module is deliberately diagnostic. It validates skill structure and local
supporting-file references, but never executes a skill-provided command, script,
or URL.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Literal

import yaml

from tools.skill_manager_tool import _validate_frontmatter

Severity = Literal["pass", "warning", "fail"]
_SUPPORTED_LINK_DIRS = {"references", "templates", "scripts", "assets"}
_LINK_RE = re.compile(r"(?<!!)\[[^\]]*\]\(([^)\s]+)(?:\s+[^)]*)?\)")
_CHECKLIST_HEADING_RE = re.compile(r"^##\s+Verification Checklist\s*$", re.MULTILINE | re.IGNORECASE)
_CHECKBOX_RE = re.compile(r"^\s*[-*+]\s+\[[ xX]\]\s+", re.MULTILINE)


@dataclass(frozen=True)
class QualityFinding:
    check: str
    severity: Severity
    message: str
    remediation: str | None = None


@dataclass(frozen=True)
class QualityAuditResult:
    skill_name: str
    path: Path
    source: str
    status: Severity
    findings: tuple[QualityFinding, ...]


def _finding(check: str, severity: Severity, message: str,
             remediation: str | None = None) -> QualityFinding:
    return QualityFinding(check, severity, message, remediation)


def _status(findings: list[QualityFinding]) -> Severity:
    if any(item.severity == "fail" for item in findings):
        return "fail"
    if any(item.severity == "warning" for item in findings):
        return "warning"
    return "pass"


def _frontmatter(content: str) -> dict | None:
    """Parse already-validated frontmatter, returning None for malformed input."""
    normalized = content.lstrip("\ufeff")
    match = re.search(r"\n---\s*\n", normalized[3:])
    if not match:
        return None
    try:
        data = yaml.safe_load(normalized[3:match.start() + 3])
    except yaml.YAMLError:
        return None
    return data if isinstance(data, dict) else None


def _audit_metadata(frontmatter: dict) -> QualityFinding:
    metadata = frontmatter.get("metadata")
    if metadata is None:
        return _finding(
            "verification_metadata", "warning",
            "metadata.hermes.verification.level is not declared.",
            "Add metadata.hermes.verification.level: static for an explicit v1 contract.",
        )
    if not isinstance(metadata, dict):
        return _finding("verification_metadata", "fail", "metadata must be a YAML mapping.")
    hermes = metadata.get("hermes")
    if hermes is None:
        return _finding(
            "verification_metadata", "warning",
            "metadata.hermes.verification.level is not declared.",
            "Add metadata.hermes.verification.level: static for an explicit v1 contract.",
        )
    if not isinstance(hermes, dict):
        return _finding("verification_metadata", "fail", "metadata.hermes must be a YAML mapping.")
    verification = hermes.get("verification")
    if verification is None:
        return _finding(
            "verification_metadata", "warning",
            "metadata.hermes.verification.level is not declared.",
            "Add metadata.hermes.verification.level: static for an explicit v1 contract.",
        )
    if not isinstance(verification, dict):
        return _finding("verification_metadata", "fail", "metadata.hermes.verification must be a YAML mapping.")
    if verification.get("level") != "static" or set(verification) != {"level"}:
        return _finding(
            "verification_metadata", "fail",
            "metadata.hermes.verification accepts only level: static in v1.",
            "Remove unsupported fields and set level to static.",
        )
    return _finding("verification_metadata", "pass", "Static verification metadata is valid.")


def _audit_local_references(content: str, skill_dir: Path) -> QualityFinding:
    skill_root = skill_dir.resolve()
    failures: list[str] = []
    checked = 0
    for match in _LINK_RE.finditer(content):
        raw_target = match.group(1).strip().strip("<>")
        if not raw_target or raw_target.startswith(("#", "http://", "https://", "mailto:")):
            continue
        target = raw_target.split("#", 1)[0].split("?", 1)[0]
        if not target:
            continue
        candidate = Path(target)
        parts = candidate.parts
        if ".." in parts:
            failures.append(f"{raw_target} escapes the skill directory")
            continue
        if not parts or parts[0] not in _SUPPORTED_LINK_DIRS:
            continue
        resolved = (skill_root / candidate).resolve()
        try:
            resolved.relative_to(skill_root)
        except ValueError:
            failures.append(f"{raw_target} escapes the skill directory")
            continue
        checked += 1
        if not resolved.is_file():
            failures.append(f"{raw_target} is missing")
    if failures:
        return _finding(
            "local_references", "fail",
            "Invalid local reference(s): " + "; ".join(failures) + ".",
            "Keep references under references/, templates/, scripts/, or assets/ and ensure each file exists.",
        )
    return _finding("local_references", "pass", f"{checked} supported local reference(s) resolved.")


def _audit_checklist(content: str) -> QualityFinding:
    heading = _CHECKLIST_HEADING_RE.search(content)
    if heading and _CHECKBOX_RE.search(content, heading.end()):
        return _finding("verification_checklist", "pass", "Verification Checklist contains a Markdown checkbox.")
    return _finding(
        "verification_checklist", "warning",
        "No Verification Checklist with a Markdown checkbox was found.",
        "Add a ## Verification Checklist section with at least one - [ ] item.",
    )


def audit_skill_quality(skill_dir: Path, *, skill_name: str, source: str) -> QualityAuditResult:
    """Run static, non-executing quality checks for one skill directory."""
    findings: list[QualityFinding] = []
    root = skill_dir.resolve()
    skill_md = root / "SKILL.md"
    if not skill_md.is_file():
        findings.append(_finding("skill_file", "fail", "SKILL.md is missing."))
        return QualityAuditResult(skill_name, root, source, _status(findings), tuple(findings))
    try:
        content = skill_md.read_text(encoding="utf-8")
    except OSError as exc:
        findings.append(_finding("skill_file", "fail", f"SKILL.md could not be read: {exc}."))
        return QualityAuditResult(skill_name, root, source, _status(findings), tuple(findings))

    error = _validate_frontmatter(content)
    if error:
        findings.append(_finding("frontmatter", "fail", error))
        return QualityAuditResult(skill_name, root, source, _status(findings), tuple(findings))
    findings.append(_finding("frontmatter", "pass", "Frontmatter is valid."))
    frontmatter = _frontmatter(content)
    if frontmatter is None:
        findings.append(_finding("frontmatter", "fail", "Frontmatter could not be parsed."))
        return QualityAuditResult(skill_name, root, source, _status(findings), tuple(findings))

    declared_name = frontmatter.get("name")
    if declared_name != skill_name:
        findings.append(_finding(
            "skill_name", "fail",
            f"Frontmatter name {declared_name!r} does not match resolved name {skill_name!r}.",
        ))
    else:
        findings.append(_finding("skill_name", "pass", "Frontmatter name matches the resolved skill name."))
    findings.append(_audit_metadata(frontmatter))
    findings.append(_audit_local_references(content, root))
    findings.append(_audit_checklist(content))
    return QualityAuditResult(skill_name, root, source, _status(findings), tuple(findings))


def format_quality_report(result: QualityAuditResult) -> str:
    """Format a stable, human-readable quality-audit report."""
    lines = [
        f"Skill quality audit: {result.skill_name} [{result.source}]",
        f"Path: {result.path}",
        f"Result: {result.status.upper()}",
        "",
    ]
    for finding in result.findings:
        lines.append(f"{finding.severity.upper():7} {finding.check}: {finding.message}")
    remediation = [finding.remediation for finding in result.findings if finding.remediation]
    if remediation:
        lines.extend(["", "Remediation:"])
        lines.extend(f"- {item}" for item in remediation)
    return "\n".join(lines)


def save_verification_receipt(receipt_path: Path, result: QualityAuditResult) -> None:
    """Atomically persist a local-only receipt for a completed static audit."""
    try:
        loaded = json.loads(receipt_path.read_text(encoding="utf-8")) if receipt_path.exists() else {}
        payload: dict = loaded if isinstance(loaded, dict) else {}
    except (OSError, json.JSONDecodeError):
        payload = {}
    receipts = payload.get("receipts") if isinstance(payload.get("receipts"), dict) else {}
    skill_md = result.path / "SKILL.md"
    try:
        content_hash = hashlib.sha256(skill_md.read_bytes()).hexdigest()
    except OSError:
        content_hash = ""
    path_identity = hashlib.sha256(str(result.path.resolve()).encode("utf-8")).hexdigest()[:16]
    key = f"{result.source}:{result.skill_name}:{path_identity}"
    receipts[key] = {
        "audited_at": datetime.now(timezone.utc).isoformat(),
        "skill_name": result.skill_name,
        "source": result.source,
        "path": str(result.path),
        "content_hash": content_hash,
        "status": result.status,
        "findings": [
            {
                "check": finding.check,
                "severity": finding.severity,
                "message": finding.message,
                "remediation": finding.remediation,
            }
            for finding in result.findings
        ],
    }
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    data = {"version": 1, "receipts": receipts}
    fd, temporary = tempfile.mkstemp(prefix=".verification-", suffix=".tmp", dir=receipt_path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
        os.replace(temporary, receipt_path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
