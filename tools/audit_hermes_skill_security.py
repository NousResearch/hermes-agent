#!/usr/bin/env python3
"""
Hermes skill security audit layer — operator review workflow.

Complements ``tools.skills_guard`` (install-time regex gate) with categorized
findings for humans reviewing host-side skills before granting trust. Per
SECURITY.md §2.4, neither layer is a security boundary; both are review aids.

Categories:
  - unsafe_instruction   — prompt injection, deception, role hijacking
  - shell_api_mutation     — destructive shell, config/API writes, persistence
  - credential_use         — secret env access, credential files, hardcoded keys
  - hidden_background      — background terminals, cron, silent side effects
  - source_mirror_drift    — content hash mismatch vs expected or mirror copy

Usage::

    from tools.audit_hermes_skill_security import audit_skill_security, format_audit_report

    result = audit_skill_security(Path("~/.hermes/skills/my-skill"))
    print(format_audit_report(result))

CLI::

    python -m tools.audit_hermes_skill_security PATH [--mirror MIRROR] [--expected-hash HASH]
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from tools.skills_guard import SCANNABLE_EXTENSIONS, content_hash

# (regex, check_id, severity, description)
PatternDef = Tuple[str, str, str, str]

AUDIT_CATEGORIES = (
    "unsafe_instruction",
    "shell_api_mutation",
    "credential_use",
    "hidden_background",
    "source_mirror_drift",
)

SEVERITY_ORDER = {"critical": 0, "high": 1, "medium": 2, "low": 3}

# Category-specific patterns (line-scanned; drift is structural).
_CATEGORY_PATTERNS: Dict[str, List[PatternDef]] = {
    "unsafe_instruction": [
        (
            r"ignore\s+(?:\w+\s+)*(previous|all|above|prior)\s+instructions",
            "prompt_injection_ignore",
            "critical",
            "instructs the agent to ignore prior instructions",
        ),
        (
            r"do\s+not\s+(?:\w+\s+)*tell\s+(?:\w+\s+)*the\s+user",
            "deception_hide",
            "critical",
            "instructs the agent to hide actions from the operator",
        ),
        (
            r"disregard\s+(?:\w+\s+)*(your|all|any)\s+(?:\w+\s+)*(instructions|rules|guidelines)",
            "disregard_rules",
            "critical",
            "instructs the agent to disregard safety rules",
        ),
        (
            r"system\s+prompt\s+override",
            "sys_prompt_override",
            "critical",
            "attempts to override the system prompt",
        ),
        (
            r"you\s+are\s+(?:\w+\s+)*now\s+",
            "role_hijack",
            "high",
            "attempts to override the agent role",
        ),
        (
            r"pretend\s+(?:\w+\s+)*(you\s+are|to\s+be)\s+",
            "role_pretend",
            "high",
            "instructs the agent to assume a different identity",
        ),
        (
            r"<!--[^>]*(?:ignore|override|system|secret|hidden)[^>]*-->",
            "html_comment_injection",
            "high",
            "hidden instructions in HTML comments",
        ),
        (
            r"when\s+no\s*one\s+is\s+(watching|looking)",
            "conditional_deception",
            "high",
            "conditional instruction to behave differently when unobserved",
        ),
    ],
    "shell_api_mutation": [
        (
            r"rm\s+-rf\s+/",
            "destructive_root_rm",
            "critical",
            "recursive delete from filesystem root",
        ),
        (
            r"chmod\s+777",
            "insecure_perms",
            "medium",
            "sets world-writable permissions",
        ),
        (
            r"\bsudo\b",
            "sudo_usage",
            "high",
            "uses sudo (privilege escalation)",
        ),
        (
            r"skill_manage\s*\(\s*[^)]*action\s*=\s*[\"']delete",
            "skill_delete",
            "high",
            "instructs deleting skills via skill_manage",
        ),
        (
            r"(write_file|patch)\s*\([^)]*(\.hermes/config\.yaml|SOUL\.md|AGENTS\.md)",
            "hermes_config_write",
            "critical",
            "instructs writing or patching Hermes config/identity files",
        ),
        (
            r"\bcrontab\b|cronjob\s*\(",
            "scheduled_mutation",
            "medium",
            "schedules recurring agent or shell work",
        ),
        (
            r"authorized_keys|/etc/sudoers",
            "persistence_ssh_sudo",
            "critical",
            "modifies SSH keys or sudoers",
        ),
        (
            r"curl\s+[^\n]*\|\s*(ba)?sh",
            "curl_pipe_shell",
            "critical",
            "curl piped to shell (download-and-execute)",
        ),
        (
            r"subprocess\.(run|call|Popen|check_output)\s*\(",
            "python_subprocess",
            "medium",
            "Python subprocess execution in skill scripts",
        ),
        (
            r"os\.system\s*\(",
            "python_os_system",
            "high",
            "os.system() shell execution",
        ),
    ],
    "credential_use": [
        (
            r'\$HOME/\.hermes/\.env|\~/\.hermes/\.env',
            "hermes_env_access",
            "critical",
            "references Hermes secrets file",
        ),
        (
            r'cat\s+[^\n]*(\.env|credentials|\.netrc)',
            "read_secrets_file",
            "critical",
            "reads a known secrets file via shell",
        ),
        (
            r"os\.getenv\s*\(\s*[^\)]*(?:KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL)",
            "python_getenv_secret",
            "critical",
            "reads a secret via os.getenv()",
        ),
        (
            r'os\.environ\b(?!\s*\.get\s*\(\s*["\']PATH)',
            "python_os_environ",
            "high",
            "accesses os.environ (potential credential dump)",
        ),
        (
            r'curl\s+[^\n]*\$\{?\w*(KEY|TOKEN|SECRET|PASSWORD|API)',
            "env_exfil_curl",
            "critical",
            "curl command interpolating a secret environment variable",
        ),
        (
            r'(?:api[_-]?key|token|secret|password)\s*[=:]\s*["\'][A-Za-z0-9+/=_-]{20,}',
            "hardcoded_secret",
            "critical",
            "possible hardcoded API key or secret in skill content",
        ),
        (
            r'ghp_[A-Za-z0-9]{36}|sk-[A-Za-z0-9]{20,}',
            "leaked_token_format",
            "critical",
            "token-shaped secret embedded in skill content",
        ),
    ],
    "hidden_background": [
        (
            r"terminal\s*\([^)]*background\s*=\s*True",
            "terminal_background",
            "high",
            "instructs a background terminal session",
        ),
        (
            r"notify_on_complete\s*=\s*True",
            "terminal_notify_complete",
            "medium",
            "background work with completion notification to the agent",
        ),
        (
            r"\bcronjob\s*\(",
            "cronjob_tool",
            "high",
            "schedules durable background agent work via cronjob",
        ),
        (
            r"\bnohup\b|>\s*/dev/null\s*&|\bdisown\b",
            "shell_background_detach",
            "high",
            "detached shell process (nohup, background, or disown)",
        ),
        (
            r"without\s+(?:\w+\s+)*(informing|telling|notifying)\s+(?:\w+\s+)*the\s+user",
            "silent_to_user",
            "critical",
            "explicit instruction to hide work from the operator",
        ),
        (
            r"do\s+not\s+(?:\w+\s+)*(mention|report|log)\s+(?:\w+\s+)*to\s+the\s+user",
            "omit_from_user",
            "critical",
            "instructs omitting activity from operator-visible output",
        ),
        (
            r"run\s+(?:\w+\s+)*in\s+the\s+background\s+without",
            "background_without_notice",
            "high",
            "background execution framed as hidden from the operator",
        ),
    ],
}


@dataclass
class AuditFinding:
    category: str
    check_id: str
    severity: str
    file: str
    line: int
    match: str
    description: str


@dataclass
class SkillSecurityAuditResult:
    skill_name: str
    skill_path: str
    verdict: str  # pass | review | fail
    findings: List[AuditFinding] = field(default_factory=list)
    content_hash: str = ""
    expected_hash: str = ""
    mirror_hash: str = ""
    drift_detected: bool = False
    scanned_at: str = ""
    summary: str = ""
    category_counts: Dict[str, int] = field(default_factory=dict)


def _truncate_match(text: str, limit: int = 120) -> str:
    text = text.strip()
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def _scan_text_file(
    file_path: Path,
    rel_path: str,
    patterns: Sequence[PatternDef],
    category: str,
) -> List[AuditFinding]:
    if file_path.suffix.lower() not in SCANNABLE_EXTENSIONS and file_path.name != "SKILL.md":
        return []

    try:
        content = file_path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return []

    findings: List[AuditFinding] = []
    seen: set[tuple[str, int, str]] = set()
    for pattern, check_id, severity, description in patterns:
        for i, line in enumerate(content.splitlines(), start=1):
            key = (check_id, i, category)
            if key in seen:
                continue
            if re.search(pattern, line, re.IGNORECASE):
                seen.add(key)
                findings.append(
                    AuditFinding(
                        category=category,
                        check_id=check_id,
                        severity=severity,
                        file=rel_path,
                        line=i,
                        match=_truncate_match(line),
                        description=description,
                    )
                )
    return findings


def _scan_directory_patterns(skill_dir: Path) -> List[AuditFinding]:
    findings: List[AuditFinding] = []
    if not skill_dir.is_dir():
        return findings

    for category, patterns in _CATEGORY_PATTERNS.items():
        for f in skill_dir.rglob("*"):
            if not f.is_file() or f.is_symlink():
                continue
            rel = str(f.relative_to(skill_dir))
            findings.extend(_scan_text_file(f, rel, patterns, category))
    return findings


def _drift_findings(
    skill_path: Path,
    *,
    expected_hash: str = "",
    mirror_path: Optional[Path] = None,
) -> Tuple[List[AuditFinding], str, str, bool]:
    """Return drift findings plus (content_hash, mirror_hash, drift_detected)."""
    current = content_hash(skill_path)
    mirror_h = ""
    drift = False
    findings: List[AuditFinding] = []

    if mirror_path is not None:
        if not mirror_path.exists():
            findings.append(
                AuditFinding(
                    category="source_mirror_drift",
                    check_id="mirror_missing",
                    severity="high",
                    file="(mirror)",
                    line=0,
                    match=str(mirror_path),
                    description="mirror/source path does not exist",
                )
            )
            drift = True
        else:
            mirror_h = content_hash(mirror_path)
            if mirror_h != current:
                drift = True
                findings.append(
                    AuditFinding(
                        category="source_mirror_drift",
                        check_id="mirror_hash_mismatch",
                        severity="critical",
                        file="(directory)",
                        line=0,
                        match=f"installed={current} mirror={mirror_h}",
                        description="installed skill content differs from mirror/source copy",
                    )
                )

    if expected_hash and expected_hash != current:
        drift = True
        findings.append(
            AuditFinding(
                category="source_mirror_drift",
                check_id="expected_hash_mismatch",
                severity="critical",
                file="(directory)",
                line=0,
                match=f"current={current} expected={expected_hash}",
                description="skill content hash does not match recorded expected hash",
            )
        )

    return findings, current, mirror_h, drift


def _determine_verdict(findings: Iterable[AuditFinding], drift_detected: bool) -> str:
    if drift_detected:
        return "fail"
    severities = [f.severity for f in findings]
    if any(s == "critical" for s in severities):
        return "fail"
    if any(s == "high" for s in severities):
        return "review"
    if any(s in ("medium", "low") for s in severities):
        return "review"
    return "pass"


def _category_counts(findings: Iterable[AuditFinding]) -> Dict[str, int]:
    counts = {cat: 0 for cat in AUDIT_CATEGORIES}
    for f in findings:
        counts[f.category] = counts.get(f.category, 0) + 1
    return counts


def _build_summary(name: str, verdict: str, findings: List[AuditFinding], drift: bool) -> str:
    if not findings and not drift:
        return f"{name}: pass — no operator-review findings"
    cats = sorted({f.category for f in findings})
    return f"{name}: {verdict} — {len(findings)} finding(s)" + (
        f", drift detected" if drift else ""
    ) + (f" [{', '.join(cats)}]" if cats else "")


def audit_skill_security(
    skill_path: Path,
    *,
    expected_hash: str = "",
    mirror_path: Optional[Path] = None,
) -> SkillSecurityAuditResult:
    """Run the operator security audit on a skill file or directory.

    Args:
        skill_path: Path to ``SKILL.md`` or a skill directory.
        expected_hash: Optional ``sha256:…`` digest from hub lock / manifest.
        mirror_path: Optional upstream or golden copy to compare via ``content_hash``.
    """
    skill_path = skill_path.resolve()
    if skill_path.is_file():
        skill_dir = skill_path.parent
        skill_name = skill_dir.name
    else:
        skill_dir = skill_path
        skill_name = skill_path.name

    pattern_findings = _scan_directory_patterns(skill_dir) if skill_dir.is_dir() else []
    if skill_path.is_file():
        for category, patterns in _CATEGORY_PATTERNS.items():
            pattern_findings.extend(
                _scan_text_file(skill_path, skill_path.name, patterns, category)
            )

    drift_findings, current_hash, mirror_hash, drift = _drift_findings(
        skill_dir if skill_dir.is_dir() else skill_path.parent,
        expected_hash=expected_hash,
        mirror_path=mirror_path,
    )

    all_findings = pattern_findings + drift_findings
    verdict = _determine_verdict(all_findings, drift)
    scanned_at = datetime.now(timezone.utc).isoformat()

    return SkillSecurityAuditResult(
        skill_name=skill_name,
        skill_path=str(skill_path),
        verdict=verdict,
        findings=all_findings,
        content_hash=current_hash,
        expected_hash=expected_hash or "",
        mirror_hash=mirror_hash,
        drift_detected=drift,
        scanned_at=scanned_at,
        summary=_build_summary(skill_name, verdict, all_findings, drift),
        category_counts=_category_counts(all_findings),
    )


def format_audit_report(result: SkillSecurityAuditResult) -> str:
    """Format an audit result as plain text for CLI or chat display."""
    lines = [
        f"Operator security audit: {result.skill_name}",
        f"  Path: {result.skill_path}",
        f"  Verdict: {result.verdict.upper()}",
        f"  Content hash: {result.content_hash or '(none)'}",
    ]
    if result.expected_hash:
        lines.append(f"  Expected hash: {result.expected_hash}")
    if result.mirror_hash:
        lines.append(f"  Mirror hash: {result.mirror_hash}")
    if result.drift_detected:
        lines.append("  Drift: YES")
    lines.append("  Category counts: " + ", ".join(
        f"{cat}={result.category_counts.get(cat, 0)}" for cat in AUDIT_CATEGORIES
    ))

    if result.findings:
        lines.append("")
        lines.append(f"  {len(result.findings)} finding(s):")
        sorted_findings = sorted(
            result.findings,
            key=lambda f: (
                SEVERITY_ORDER.get(f.severity, 9),
                f.category,
                f.file,
                f.line,
            ),
        )
        current_cat = None
        for f in sorted_findings:
            if f.category != current_cat:
                current_cat = f.category
                lines.append(f"  [{current_cat}]")
            loc = f"{f.file}:{f.line}" if f.line else f.file
            sev = f.severity.upper().ljust(8)
            lines.append(f"    {sev} {f.check_id.ljust(28)} {loc}")
            lines.append(f"             {f.description}")
            if f.match:
                lines.append(f"             \"{f.match[:80]}\"")
    else:
        lines.append("")
        lines.append("  No pattern findings.")

    lines.append("")
    lines.append("  Note: review aid for operators — not an install gate. See SECURITY.md §2.4.")
    return "\n".join(lines)


def _main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Hermes operator skill security audit (review aid, not a boundary).",
    )
    parser.add_argument("skill_path", type=Path, help="Skill directory or SKILL.md path")
    parser.add_argument(
        "--mirror",
        type=Path,
        default=None,
        help="Upstream or golden copy to compare for source/mirror drift",
    )
    parser.add_argument(
        "--expected-hash",
        default="",
        help="Recorded content hash (e.g. from hub lock) to detect drift",
    )
    args = parser.parse_args(argv)

    if not args.skill_path.exists():
        print(f"Error: path not found: {args.skill_path}", file=sys.stderr)
        return 2

    result = audit_skill_security(
        args.skill_path,
        expected_hash=args.expected_hash,
        mirror_path=args.mirror,
    )
    print(format_audit_report(result))
    return 0 if result.verdict == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(_main())
