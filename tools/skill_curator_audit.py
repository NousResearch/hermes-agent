#!/usr/bin/env python3
"""Deterministic lifecycle audit for Hermes skills.

The audit reports recommended actions only. It deliberately avoids printing raw
reference filenames or file contents because references can contain private
receipts, transcripts, or secret-bearing logs. Paranoia, finally useful.
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import date, datetime
from pathlib import Path
from typing import Any

import yaml

DEFAULT_STALE_LEARNING_DAYS = 30
DEFAULT_MAX_REFERENCES_BYTES = 256 * 1024


def _parse_frontmatter(skill_md: Path) -> tuple[dict[str, Any], str]:
    content = skill_md.read_text(encoding="utf-8")
    if not content.startswith("---"):
        return {}, content
    end_match = re.search(r"\n---\s*\n", content[3:])
    if not end_match:
        return {}, content
    yaml_content = content[3 : end_match.start() + 3]
    body = content[end_match.end() + 3 :]
    try:
        parsed = yaml.safe_load(yaml_content) or {}
    except yaml.YAMLError:
        parsed = {}
    if not isinstance(parsed, dict):
        parsed = {}
    return parsed, body


def _lifecycle(frontmatter: dict[str, Any]) -> dict[str, Any]:
    metadata = frontmatter.get("metadata")
    if not isinstance(metadata, dict):
        return {}
    hermes = metadata.get("hermes")
    if not isinstance(hermes, dict):
        return {}
    lifecycle = hermes.get("lifecycle")
    if not isinstance(lifecycle, dict):
        return {}
    return lifecycle


def _parse_iso_date(value: Any) -> date | None:
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    if not isinstance(value, str) or not value:
        return None
    try:
        return date.fromisoformat(value[:10])
    except ValueError:
        return None


def _references_size(skill_dir: Path) -> tuple[int, int]:
    refs = skill_dir / "references"
    if not refs.exists():
        return 0, 0
    total = 0
    files = 0
    for path in sorted(refs.rglob("*")):
        if path.is_file():
            files += 1
            total += path.stat().st_size
    return total, files


def _has_validation_retention(body: str) -> bool:
    lowered = body.lower()
    return "## validation & retention" in lowered or "## validation and retention" in lowered


def _has_falsifier_note(body: str) -> bool:
    return "falsifier" in body.lower() or "would falsify" in body.lower()


def _finding(skill: str, path: Path, check: str, severity: str, message: str, action: str) -> dict[str, str]:
    return {
        "skill": skill,
        "path": str(path.as_posix()),
        "check": check,
        "severity": severity,
        "message": message,
        "action": action,
    }


def audit_skills(
    skills_dir: str | Path,
    *,
    today: str | date | None = None,
    stale_learning_days: int = DEFAULT_STALE_LEARNING_DAYS,
    max_references_bytes: int = DEFAULT_MAX_REFERENCES_BYTES,
) -> dict[str, Any]:
    """Audit a skills tree and return a deterministic JSON-serializable report."""
    root = Path(skills_dir)
    if today is None:
        today_date = date.today()
    elif isinstance(today, date):
        today_date = today
    else:
        today_date = date.fromisoformat(today[:10])

    findings: list[dict[str, str]] = []
    descriptions: dict[str, list[tuple[str, Path]]] = {}
    skills_audited = 0

    for skill_md in sorted(root.rglob("SKILL.md")):
        skills_audited += 1
        skill_dir = skill_md.parent
        frontmatter, body = _parse_frontmatter(skill_md)
        skill_name = str(frontmatter.get("name") or skill_dir.name)
        rel_dir = skill_dir.relative_to(root)
        lifecycle = _lifecycle(frontmatter)
        status = lifecycle.get("status")

        description = str(frontmatter.get("description") or "").strip().lower()
        if description:
            descriptions.setdefault(description, []).append((skill_name, rel_dir))

        if status == "learning":
            last_validated = _parse_iso_date(lifecycle.get("last_validated"))
            if last_validated is not None:
                age_days = (today_date - last_validated).days
                if age_days > stale_learning_days:
                    findings.append(
                        _finding(
                            skill_name,
                            rel_dir,
                            "stale_learning",
                            "warning",
                            f"learning skill last validated {age_days} days ago (threshold {stale_learning_days}).",
                            "Revalidate, promote to candidate, or deprecate with a replacement pointer.",
                        )
                    )

        if status == "deprecated" and not lifecycle.get("superseded_by"):
            findings.append(
                _finding(
                    skill_name,
                    rel_dir,
                    "deprecated_missing_replacement",
                    "warning",
                    "deprecated skill has no replacement pointer.",
                    "Add metadata.hermes.lifecycle.superseded_by.",
                )
            )

        reference_bytes, reference_files = _references_size(skill_dir)
        if reference_bytes > max_references_bytes:
            findings.append(
                _finding(
                    skill_name,
                    rel_dir,
                    "bloated_references",
                    "warning",
                    f"references/ is {reference_bytes} bytes across {reference_files} files (threshold {max_references_bytes}).",
                    "Compress raw evidence into compact receipts; do not retain raw transcripts by default.",
                )
            )

        if status in {"learning", "candidate", "locked"} and not _has_validation_retention(body):
            findings.append(
                _finding(
                    skill_name,
                    rel_dir,
                    "missing_validation_retention",
                    "info",
                    "lifecycle-aware skill is missing a Validation & Retention section.",
                    "Add evidence, retained/discarded artifacts, and falsifier notes.",
                )
            )
        elif status in {"learning", "candidate", "locked"} and not _has_falsifier_note(body):
            findings.append(
                _finding(
                    skill_name,
                    rel_dir,
                    "missing_falsifier_note",
                    "info",
                    "Validation & Retention section does not mention a falsifier.",
                    "State what would demote, update, or deprecate the skill.",
                )
            )

        if status == "locked":
            receipts = skill_dir / "references"
            if receipts.exists():
                for receipt in sorted(receipts.rglob("*.json")):
                    # Only file-level metadata is inspected; contents are not rendered.
                    try:
                        text = receipt.read_text(encoding="utf-8", errors="ignore").lower()
                    except OSError:
                        continue
                    if '"result": "fail"' in text or '"result":"fail"' in text:
                        findings.append(
                            _finding(
                                skill_name,
                                rel_dir,
                                "locked_recent_failure_receipt",
                                "warning",
                                "locked skill has at least one failure receipt under references/.",
                                "Review failure evidence and demote or update the skill if the failure still reproduces.",
                            )
                        )
                        break

    for description, matches in sorted(descriptions.items()):
        if len(matches) > 1:
            skills = ", ".join(name for name, _ in sorted(matches))
            first_path = sorted(path for _, path in matches)[0]
            findings.append(
                _finding(
                    skills,
                    first_path,
                    "duplicate_description",
                    "info",
                    f"{len(matches)} skills share the same description.",
                    "Review overlap and merge, clarify, or deprecate duplicates.",
                )
            )

    findings = sorted(findings, key=lambda f: (f["skill"], f["check"], f["path"]))
    return {
        "audit": "skill_curator_audit",
        "generated_for_date": today_date.isoformat(),
        "config": {
            "stale_learning_days": stale_learning_days,
            "max_references_bytes": max_references_bytes,
        },
        "summary": {
            "skills_audited": skills_audited,
            "findings": len(findings),
        },
        "findings": findings,
    }


def render_report(report: dict[str, Any]) -> str:
    """Render a stable human-readable report without raw reference details."""
    lines = [
        "Skill Curator Audit",
        f"Date: {report['generated_for_date']}",
        f"Skills audited: {report['summary']['skills_audited']}",
        f"Findings: {report['summary']['findings']}",
        "",
    ]
    if not report["findings"]:
        lines.append("No lifecycle findings.")
        return "\n".join(lines) + "\n"

    for finding in report["findings"]:
        lines.extend(
            [
                f"- [{finding['severity']}] {finding['skill']} :: {finding['check']}",
                f"  Path: {finding['path']}",
                f"  Message: {finding['message']}",
                f"  Recommended action: {finding['action']}",
            ]
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit Hermes skills for lifecycle curation issues.")
    parser.add_argument("--skills-dir", default="skills", help="Skills directory to audit (default: skills)")
    parser.add_argument("--today", default=None, help="Override today's date for deterministic tests (YYYY-MM-DD)")
    parser.add_argument("--stale-learning-days", type=int, default=DEFAULT_STALE_LEARNING_DAYS)
    parser.add_argument("--max-references-bytes", type=int, default=DEFAULT_MAX_REFERENCES_BYTES)
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of text")
    args = parser.parse_args()

    report = audit_skills(
        args.skills_dir,
        today=args.today,
        stale_learning_days=args.stale_learning_days,
        max_references_bytes=args.max_references_bytes,
    )
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(render_report(report), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
