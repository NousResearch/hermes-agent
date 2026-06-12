#!/usr/bin/env python3
"""Deterministic quality grading for Hermes skill directories.

This module intentionally contains only cheap, local checks.  It is the
Hermes-native counterpart to rubric-style skill evaluation: structural and
safety issues are deterministic, while future semantic/model checks can be
layered on top without changing the report shape.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import yaml

MAX_NAME_LENGTH = 64
MAX_DESCRIPTION_LENGTH = 1024
MAX_BODY_LINES = 500
LONG_REFERENCE_LINES = 100

GRADE_S = "S"
GRADE_A = "A"
GRADE_B = "B"
GRADE_C = "C"
GRADE_F = "F"

SEVERITIES = ("BLOCKER", "MAJOR", "MINOR")
SECTION_NAMES = {
    "1": "validity",
    "2": "structure",
    "3": "trigger",
    "4": "content",
    "5": "resources",
    "6": "safety",
}

# Hermes skill names are broader than strict kebab-case because existing
# user-created skills and the skill manager allow dots/underscores.  Keep this
# aligned with tools.skill_manager_tool.VALID_NAME_RE instead of importing that
# module (which would create a quality<->manager cycle).
NAME_RE = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
TAG_RE = re.compile(r"<\s*/?\s*[A-Za-z][A-Za-z0-9:-]*(?:\s+[^>]*)?>")

WHEN_PATTERNS = [
    r"\buse\s+when\b",
    r"\bwhen\b",
    r"\bwhenever\b",
    r"\basked\s+to\b",
    r"\basks?\s+(?:for|to)\b",
    r"\bfor\s+(?:recurring|repeated|reviewing|creating|debugging|planning|preparing|running|operating)\b",
    r"사용\s*시",
    r"사용할\s*때",
    r"필요할\s*때",
    r"요청(?:받|할|하면|했을)\s*때",
    r"할\s*때",
]

BODY_TRIGGER_PATTERNS = [
    r"^\s*#+\s*(when|when to use|사용\s*시점|사용\s*조건)",
    r"^\s*use\s+when\b",
    r"^\s*when\s+to\s+use\b",
    r"사용\s*시",
    r"사용할\s*때",
]

BROAD_TRIGGER_PATTERNS = [
    r"\bany\s+(?:task|work|review|quality|coding|documentation)\b",
    r"\ball\s+(?:tasks|work|reviews|coding|documentation)\b",
    r"\beverything\b",
    r"\banything\b",
    r"모든\s*(?:작업|리뷰|문서|코딩)",
    r"아무\s*(?:작업|때)",
]

GENERIC_ADVICE_PATTERNS = [
    r"read\s+carefully",
    r"think\s+carefully",
    r"be\s+helpful",
    r"write\s+clean\s+code",
    r"run\s+tests?",
    r"ask\s+clarifying\s+questions?",
    r"코드를\s*잘\s*읽",
    r"테스트를\s*실행",
]

PLACEHOLDER_RE = re.compile(
    r"\b(TODO|TBD|FIXME|PLACEHOLDER|CHANGEME|XXX)\b|<\s*TODO\s*>|\{\{\s*TODO\s*\}\}",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class QualityFinding:
    id: str
    section: str
    item: str
    severity: str
    status: str
    checker: str = "rule"
    why: str = ""
    how_to_fix: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "section": self.section,
            "item": self.item,
            "severity": self.severity,
            "status": self.status,
            "checker": self.checker,
            "why": self.why,
            "how_to_fix": self.how_to_fix,
        }


@dataclass(frozen=True)
class SkillQualityReport:
    skill_name: str
    skill_dir: str
    grade: str
    findings: List[QualityFinding]

    @property
    def failed_findings(self) -> List[QualityFinding]:
        return [finding for finding in self.findings if finding.status == "fail"]

    @property
    def failed_counts(self) -> Dict[str, int]:
        failed = self.failed_findings
        return {
            severity: sum(1 for finding in failed if finding.severity == severity)
            for severity in SEVERITIES
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "skill_name": self.skill_name,
            "skill_dir": self.skill_dir,
            "grade": self.grade,
            "failed_counts": self.failed_counts,
            "findings": [finding.to_dict() for finding in self.findings],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)


def _section(id_: str) -> str:
    return SECTION_NAMES.get(id_.split(".", 1)[0], "unknown")


def _mk(
    id_: str,
    item: str,
    severity: str,
    status: str,
    why: str = "",
    how_to_fix: str = "",
) -> QualityFinding:
    return QualityFinding(
        id=id_,
        section=_section(id_),
        item=item,
        severity=severity,
        status=status,
        why=why,
        how_to_fix=how_to_fix,
    )


def _pass(id_: str, item: str, severity: str) -> QualityFinding:
    return _mk(id_, item, severity, "pass")


def _fail(id_: str, item: str, severity: str, why: str, how_to_fix: str) -> QualityFinding:
    return _mk(id_, item, severity, "fail", why, how_to_fix)


def _na(id_: str, item: str, severity: str, why: str) -> QualityFinding:
    return _mk(id_, item, severity, "na", why, "Fix prerequisite checks first.")


def _normalize_skill_dir(path: Path | str) -> Path:
    p = Path(path).expanduser()
    if p.name == "SKILL.md":
        return p.parent
    return p


def _read_skill(skill_dir: Path) -> Tuple[str, Optional[str]]:
    skill_md = skill_dir / "SKILL.md"
    if not skill_md.exists():
        return "", f"Missing SKILL.md at {skill_md}"
    try:
        return skill_md.read_text(encoding="utf-8"), None
    except Exception as exc:  # pragma: no cover - defensive IO branch
        return "", f"Could not read SKILL.md: {exc}"


def _parse_frontmatter_strict(content: str) -> Tuple[Dict[str, Any], str, bool, Optional[str]]:
    if not content.startswith("---"):
        return {}, content, False, "SKILL.md must start with YAML frontmatter."

    match = re.match(r"^---\s*\r?\n(.*?)\r?\n---\s*\r?\n(.*)$", content, re.DOTALL)
    if not match:
        return {}, content, False, "YAML frontmatter must be closed by a second --- line."

    raw_frontmatter, body = match.group(1), match.group(2)
    try:
        parsed = yaml.safe_load(raw_frontmatter) or {}
    except yaml.YAMLError as exc:
        return {}, body, False, f"YAML frontmatter parse error: {exc}"
    if not isinstance(parsed, dict):
        return {}, body, False, "Frontmatter must be a YAML mapping."
    return parsed, body, True, None


def _has_when(text: str) -> bool:
    lowered = text.lower()
    return any(re.search(pattern, lowered, re.IGNORECASE | re.MULTILINE) for pattern in WHEN_PATTERNS)


def _body_has_trigger(body: str) -> bool:
    return any(re.search(pattern, body, re.IGNORECASE | re.MULTILINE) for pattern in BODY_TRIGGER_PATTERNS)


def _trigger_too_broad(description: str) -> bool:
    return any(re.search(pattern, description, re.IGNORECASE) for pattern in BROAD_TRIGGER_PATTERNS)


def _generic_body_only(body: str) -> bool:
    words = re.findall(r"[A-Za-z가-힣0-9_]+", body)
    if len(words) > 80:
        return False
    hits = sum(1 for pattern in GENERIC_ADVICE_PATTERNS if re.search(pattern, body, re.IGNORECASE))
    return hits >= 1 and not re.search(r"[`$]|\d|references/|scripts/|https?://", body)


def _safe_child_files(root: Path) -> List[Path]:
    """Return regular files under root, excluding symlinks before any reads.

    The security scanner still reports symlink escapes separately.  Quality
    checks should not open or syntax-check files that resolve outside the skill
    directory just to discover that they are unsafe.
    """
    if not root.exists():
        return []
    return sorted(
        path for path in root.rglob("*")
        if not path.is_symlink() and path.is_file()
    )


def _reference_files(skill_dir: Path) -> List[Path]:
    root = skill_dir / "references"
    return [path for path in _safe_child_files(root) if path.suffix == ".md"]


def _script_files(skill_dir: Path) -> List[Path]:
    root = skill_dir / "scripts"
    return [path for path in _safe_child_files(root) if path.suffix not in {".pyc", ".pyo"}]


def _has_toc(text: str) -> bool:
    head = "\n".join(text.splitlines()[:40]).lower()
    return "table of contents" in head or re.search(r"^\s*#{1,3}\s*(contents|목차)\b", head, re.MULTILINE) is not None


def _check_python_syntax(path: Path) -> Optional[str]:
    try:
        compile(path.read_text(encoding="utf-8"), str(path), "exec")
    except SyntaxError as exc:
        return f"{path.name}:{exc.lineno}: {exc.msg}"
    except Exception as exc:  # pragma: no cover - defensive IO branch
        return f"{path.name}: {exc}"
    return None


def _check_shell_syntax(path: Path) -> Optional[str]:
    bash = shutil.which("bash")
    if not bash:
        return None
    result = subprocess.run(
        [bash, "-n", str(path)],
        capture_output=True,
        text=True,
        timeout=10,
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "syntax check failed").strip()
        return f"{path.name}: {detail}"
    return None


def _check_script_syntax(paths: Iterable[Path]) -> List[str]:
    failures: List[str] = []
    for path in paths:
        if path.suffix == ".py":
            failure = _check_python_syntax(path)
        elif path.suffix in {".sh", ".bash"}:
            failure = _check_shell_syntax(path)
        else:
            failure = None
        if failure:
            failures.append(failure)
    return failures


def _safety_findings(skill_dir: Path) -> List[str]:
    try:
        from tools.skills_guard import scan_skill

        result = scan_skill(skill_dir, source="agent-created")
    except Exception:
        return []

    dangerous = []
    for finding in result.findings:
        if finding.severity == "critical" or finding.category in {"exfiltration", "destructive"}:
            dangerous.append(
                f"{finding.file}:{finding.line} {finding.pattern_id} ({finding.category})"
            )
    return dangerous


def compute_grade(findings: Iterable[QualityFinding]) -> str:
    failed = [finding for finding in findings if finding.status == "fail"]
    blockers = sum(1 for finding in failed if finding.severity == "BLOCKER")
    majors = sum(1 for finding in failed if finding.severity == "MAJOR")
    if blockers:
        return GRADE_F
    if majors == 0:
        return GRADE_S
    if majors <= 2:
        return GRADE_A
    if majors <= 4:
        return GRADE_B
    return GRADE_C


def grade_skill(skill_dir: Path | str) -> SkillQualityReport:
    """Run deterministic quality checks for one Hermes skill directory."""
    skill_dir = _normalize_skill_dir(skill_dir).resolve()
    content, read_error = _read_skill(skill_dir)
    frontmatter, body, fm_ok, fm_error = _parse_frontmatter_strict(content) if not read_error else ({}, "", False, read_error)
    name = str(frontmatter.get("name") or skill_dir.name)
    description_raw = frontmatter.get("description", "")
    description = description_raw if isinstance(description_raw, str) else str(description_raw or "")
    references = _reference_files(skill_dir)
    scripts = _script_files(skill_dir)

    findings: List[QualityFinding] = []

    item = "YAML frontmatter can be parsed"
    findings.append(_pass("2.1", item, "BLOCKER") if fm_ok else _fail("2.1", item, "BLOCKER", fm_error or "Frontmatter could not be parsed.", "Use valid YAML frontmatter enclosed by opening and closing --- lines."))

    item = "name is filesystem-safe and at most 64 characters"
    if not fm_ok:
        findings.append(_na("2.2", item, "BLOCKER", "Frontmatter did not parse."))
    elif isinstance(frontmatter.get("name"), str) and NAME_RE.match(name) and len(name) <= MAX_NAME_LENGTH:
        findings.append(_pass("2.2", item, "BLOCKER"))
    else:
        findings.append(_fail("2.2", item, "BLOCKER", f"name {frontmatter.get('name')!r} is missing, too long, or not Hermes-safe.", "Set name to lowercase letters/digits plus optional hyphens, underscores, or dots; max 64 characters."))

    item = "description is 1 to 1024 characters"
    if not fm_ok:
        findings.append(_na("2.4", item, "BLOCKER", "Frontmatter did not parse."))
    elif isinstance(description_raw, str) and 1 <= len(description.strip()) <= MAX_DESCRIPTION_LENGTH:
        findings.append(_pass("2.4", item, "BLOCKER"))
    else:
        findings.append(_fail("2.4", item, "BLOCKER", f"description length is {len(description.strip())} characters.", "Provide a concise frontmatter description between 1 and 1024 characters."))

    item = "description contains no XML or HTML tags"
    if not fm_ok:
        findings.append(_na("2.5", item, "BLOCKER", "Frontmatter did not parse."))
    elif TAG_RE.search(description):
        findings.append(_fail("2.5", item, "BLOCKER", "description contains an XML/HTML-like tag.", "Remove tags from description; keep trigger text plain."))
    else:
        findings.append(_pass("2.5", item, "BLOCKER"))

    item = "SKILL.md body contains instructions"
    if not fm_ok:
        findings.append(_na("2.6", item, "BLOCKER", "Frontmatter did not parse."))
    elif body.strip():
        findings.append(_pass("2.6", item, "BLOCKER"))
    else:
        findings.append(_fail("2.6", item, "BLOCKER", "SKILL.md has no body after frontmatter.", "Add actionable instructions, workflow steps, and verification guidance after frontmatter."))

    desc_has_when = _has_when(description)
    item = "description includes invocation timing"
    if not fm_ok:
        findings.append(_na("3.1", item, "MAJOR", "Frontmatter did not parse."))
    elif desc_has_when:
        findings.append(_pass("3.1", item, "MAJOR"))
    else:
        findings.append(_fail("3.1", item, "MAJOR", "The description does not say when Hermes should load this skill.", "Add WHAT + WHEN to frontmatter description, e.g. 'Use when asked to ...' or 'Use for recurring ...'."))

    item = "body-only trigger anti-pattern absent"
    if not fm_ok:
        findings.append(_na("3.4", item, "BLOCKER", "Frontmatter did not parse."))
    elif not desc_has_when and _body_has_trigger(body):
        findings.append(_fail("3.4", item, "BLOCKER", "Trigger conditions appear in the body but not in description; Hermes sees description before loading the body.", "Move the essential trigger condition into the frontmatter description."))
    else:
        findings.append(_pass("3.4", item, "BLOCKER"))

    item = "trigger scope is bounded"
    if not fm_ok:
        findings.append(_na("3.5", item, "MAJOR", "Frontmatter did not parse."))
    elif _trigger_too_broad(description):
        findings.append(_fail("3.5", item, "MAJOR", "The description is broad enough to match unrelated work.", "Name the concrete target object and workflow instead of using broad terms like any/all/everything."))
    else:
        findings.append(_pass("3.5", item, "MAJOR"))

    item = "body is not merely generic agent advice"
    if not fm_ok:
        findings.append(_na("4.2", item, "MAJOR", "Frontmatter did not parse."))
    elif _generic_body_only(body):
        findings.append(_fail("4.2", item, "MAJOR", "The body mostly repeats generic agent advice without reusable domain procedure.", "Add concrete workflow steps, commands, schemas, thresholds, examples, or domain-specific pitfalls."))
    else:
        findings.append(_pass("4.2", item, "MAJOR"))

    item = f"SKILL.md body is at most {MAX_BODY_LINES} lines"
    body_lines = len(body.splitlines())
    if body_lines <= MAX_BODY_LINES:
        findings.append(_pass("4.3", item, "MINOR"))
    else:
        findings.append(_fail("4.3", item, "MINOR", f"SKILL.md body has {body_lines} lines.", "Move bulky examples/specs into references/ and keep SKILL.md as a concise orchestrator."))

    item = "reference load conditions are clear"
    linked_refs = re.findall(r"\]\((references/[^)#]+)(?:#[^)]+)?\)", body)
    if not linked_refs:
        findings.append(_pass("5.2", item, "MINOR"))
    else:
        unclear = []
        for ref in linked_refs:
            for line in body.splitlines():
                if ref in line:
                    if not re.search(r"\b(when|if|after|before|for|use|read|load)\b|읽|사용|필요", line, re.IGNORECASE):
                        unclear.append(ref)
                    break
        if unclear:
            findings.append(_fail("5.2", item, "MINOR", f"Reference links lack load conditions: {', '.join(sorted(set(unclear)))}.", "State when to read each reference next to its link."))
        else:
            findings.append(_pass("5.2", item, "MINOR"))

    item = "long reference files include a table of contents"
    long_without_toc = []
    for ref in references:
        text = ref.read_text(encoding="utf-8", errors="replace")
        if len(text.splitlines()) >= LONG_REFERENCE_LINES and not _has_toc(text):
            long_without_toc.append(str(ref.relative_to(skill_dir)))
    if long_without_toc:
        findings.append(_fail("5.4", item, "MINOR", f"Long reference file(s) lack TOC: {', '.join(long_without_toc)}.", "Add a short table of contents near the top of each long reference."))
    else:
        findings.append(_pass("5.4", item, "MINOR"))

    item = "script syntax is valid"
    script_failures = _check_script_syntax(scripts)
    if script_failures:
        findings.append(_fail("5.6", item, "MAJOR", "; ".join(script_failures), "Fix script syntax so deterministic workflow helpers can run reliably."))
    else:
        findings.append(_pass("5.6", item, "MAJOR"))

    item = "scripts are mentioned in SKILL.md"
    unmentioned = []
    for script in scripts:
        rel = str(script.relative_to(skill_dir))
        if rel not in body and script.name not in body:
            unmentioned.append(rel)
    if unmentioned:
        findings.append(_fail("5.7", item, "MINOR", f"Script file(s) are not mentioned: {', '.join(unmentioned)}.", "Mention each script path in SKILL.md with when to run it."))
    else:
        findings.append(_pass("5.7", item, "MINOR"))

    item = "placeholder and scaffold residue are absent"
    residue = []
    for path in [skill_dir / "SKILL.md", *references, *scripts]:
        if path.exists() and path.is_file():
            text = path.read_text(encoding="utf-8", errors="replace")
            match = PLACEHOLDER_RE.search(text)
            if match:
                residue.append(f"{path.relative_to(skill_dir)}: {match.group(0)}")
    if residue:
        findings.append(_fail("5.8", item, "MINOR", f"Placeholder residue found: {', '.join(residue)}.", "Remove TODO/FIXME/placeholder text before relying on this skill."))
    else:
        findings.append(_pass("5.8", item, "MINOR"))

    item = "dangerous skill content is absent"
    dangerous = _safety_findings(skill_dir)
    if dangerous:
        findings.append(_fail("6.1", item, "BLOCKER", f"Potential dangerous content: {'; '.join(dangerous[:5])}", "Remove secrets/exfiltration/destructive instructions or move safe setup guidance into explicit user-run steps."))
    else:
        findings.append(_pass("6.1", item, "BLOCKER"))

    grade = compute_grade(findings)
    return SkillQualityReport(
        skill_name=name,
        skill_dir=str(skill_dir),
        grade=grade,
        findings=findings,
    )


def render_quality_report(report: SkillQualityReport) -> str:
    counts = report.failed_counts
    lines = [
        f"TL;DR: [{report.skill_name}] grade {report.grade} | "
        f"BLOCKER {counts['BLOCKER']}, MAJOR {counts['MAJOR']}, MINOR {counts['MINOR']}",
        "",
    ]

    failed = sorted(report.failed_findings, key=lambda f: (SEVERITIES.index(f.severity), tuple(int(p) for p in f.id.split("."))))

    def add_group(title: str, severity: str) -> None:
        lines.append(f"{title}:")
        items = [finding for finding in failed if finding.severity == severity]
        if not items:
            lines.append("- None")
        for finding in items:
            lines.append(f"- {finding.id} {finding.item}")
            lines.append(f"  why: {finding.why or 'No detail provided.'}")
            lines.append(f"  how_to_fix: {finding.how_to_fix or 'No fix provided.'}")
        lines.append("")

    add_group("Blocking issues", "BLOCKER")
    add_group("Priority fixes", "MAJOR")
    add_group("Recommendations", "MINOR")

    lines.append("Section summary:")
    for section in ("validity", "structure", "trigger", "content", "resources", "safety"):
        section_failed = [finding for finding in failed if finding.section == section]
        if not section_failed:
            lines.append(f"- {section}: PASS")
            continue
        parts = []
        for severity in SEVERITIES:
            count = sum(1 for finding in section_failed if finding.severity == severity)
            if count:
                parts.append(f"{count} {severity}")
        lines.append(f"- {section}: {', '.join(parts)}")
    return "\n".join(lines) + "\n"


def main(argv: Optional[List[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Grade a Hermes skill directory with deterministic checks.")
    parser.add_argument("target", help="Skill directory or SKILL.md path")
    parser.add_argument("--json", action="store_true", help="Output JSON")
    args = parser.parse_args(argv)

    report = grade_skill(args.target)
    if args.json:
        sys.stdout.write(report.to_json() + "\n")
    else:
        sys.stdout.write(render_quality_report(report))
    return 0 if report.grade != GRADE_F else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
