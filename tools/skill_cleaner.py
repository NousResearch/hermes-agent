"""Report-only Hermes skill cleaner/audit helpers.

This module intentionally performs **no mutations**. It inventories skill cards,
runs the existing skills guard, checks a minimal verified-card contract, estimates
prompt bloat, and reports likely duplicate/overlapping skills so a human/curator
can decide what to consolidate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agent.skill_utils import get_all_skills_dirs, iter_skill_index_files, parse_frontmatter
from hermes_constants import display_hermes_home, get_bundled_skills_dir, get_hermes_home, get_skills_dir
from tools.skills_guard import scan_skill

_TOKEN_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9_-]{2,}")
_NAME_RE = re.compile(r"^[a-z][a-z0-9_-]*$")
_NUMBERED_STEP_RE = re.compile(r"(?m)^\s*\d+\.\s+\S+")
_SESSION_ARTIFACT_RE = re.compile(
    r"\b(?:PR\s*#?\d+|pull request\s*#?\d+|issue\s*#?\d+|(?=[0-9a-f]{7,40}\b)(?=[0-9a-f]*[a-f])[0-9a-f]{7,40})\b",
    re.IGNORECASE,
)


@dataclass
class CardFinding:
    severity: str
    code: str
    message: str


@dataclass
class SkillCardReport:
    name: str
    source: str
    path: str
    rel_path: str
    char_count: int
    estimated_tokens: int
    description: str = ""
    guard_verdict: str = "safe"
    guard_findings: int = 0
    card_findings: list[CardFinding] = field(default_factory=list)


@dataclass
class DuplicateCandidate:
    left: str
    right: str
    similarity: float
    reason: str


@dataclass
class SkillCleanerReport:
    generated_at: str
    hermes_home: str
    scanned_roots: list[str]
    skills: list[SkillCardReport]
    duplicates: list[DuplicateCandidate]

    @property
    def total_estimated_tokens(self) -> int:
        return sum(s.estimated_tokens for s in self.skills)

    @property
    def finding_counts(self) -> dict[str, int]:
        counts = {"error": 0, "warning": 0, "info": 0}
        for skill in self.skills:
            for finding in skill.card_findings:
                counts[finding.severity] = counts.get(finding.severity, 0) + 1
        return counts


@dataclass(frozen=True)
class _SkillSource:
    label: str
    root: Path


def _estimate_tokens(text: str) -> int:
    # Cheap prompt-size approximation; intentionally deterministic and dependency-free.
    return max(1, (len(text) + 3) // 4)


def _token_set(text: str) -> set[str]:
    stop = {
        "the", "and", "for", "with", "that", "this", "when", "from", "into", "your",
        "skill", "skills", "hermes", "agent", "use", "using", "will", "must", "should",
    }
    return {tok.lower() for tok in _TOKEN_RE.findall(text) if tok.lower() not in stop}


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _relative_to_root(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def _source_roots(include_bundled: bool) -> list[_SkillSource]:
    active_root = get_skills_dir().resolve()
    roots: list[_SkillSource] = []
    seen: set[Path] = set()
    for root in get_all_skills_dirs():
        resolved = root.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        label = "active-profile" if resolved == active_root else "external"
        roots.append(_SkillSource(label, resolved))
    if include_bundled:
        repo_default = Path(__file__).resolve().parent.parent / "skills"
        bundled = get_bundled_skills_dir(default=repo_default).resolve()
        if bundled not in seen:
            roots.append(_SkillSource("bundled", bundled))
    return roots


def _nested_get(data: dict[str, Any], dotted: str) -> Any:
    cur: Any = data
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def _check_required_card_field(findings: list[CardFinding], card: dict[str, Any], dotted: str) -> None:
    value = _nested_get(card, dotted)
    missing = value is None or value == "" or value == [] or value == {}
    if missing:
        findings.append(CardFinding("warning", "missing_skill_card_field", f"skill_card.{dotted} is required by the verified-card contract."))


def _skill_card_findings(card: Any, raw: str) -> list[CardFinding]:
    findings: list[CardFinding] = []
    if not isinstance(card, dict):
        return [
            CardFinding(
                "warning",
                "missing_skill_card",
                "Add skill_card provenance/risk/scope/verification/dependency metadata before trusting this skill as verified.",
            )
        ]

    required_fields = (
        "card_version",
        "source.origin",
        "source.upstream_url",
        "source.author",
        "source.imported_at",
        "scope.summary",
        "scope.allowed_surfaces",
        "scope.denied_surfaces",
        "risk.level",
        "risk.reasons",
        "risk.approval_required_for",
        "verification.reviewed_by",
        "verification.reviewed_at",
        "verification.spec_reference",
        "verification.content_sha256",
        "verification.modified_since_review",
        "dependencies.required_env_vars",
        "dependencies.required_commands",
        "dependencies.network_access",
    )
    for field_name in required_fields:
        _check_required_card_field(findings, card, field_name)

    level = str(_nested_get(card, "risk.level") or "").strip().lower()
    if level and level not in {"low", "medium", "high", "critical"}:
        findings.append(CardFinding("warning", "invalid_risk_level", "skill_card.risk.level must be low, medium, high, or critical."))

    for dotted in (
        "scope.allowed_surfaces",
        "scope.denied_surfaces",
        "risk.reasons",
        "risk.approval_required_for",
        "dependencies.required_env_vars",
        "dependencies.required_commands",
    ):
        value = _nested_get(card, dotted)
        if value is not None and not isinstance(value, list):
            findings.append(CardFinding("warning", "invalid_skill_card_field", f"skill_card.{dotted} should be a list."))

    modified = _nested_get(card, "verification.modified_since_review")
    if modified is True:
        findings.append(CardFinding("warning", "modified_since_review", "Skill card says content changed since review; re-review before trusting."))

    expected_hash = _nested_get(card, "verification.content_sha256")
    if isinstance(expected_hash, str) and len(expected_hash.strip()) == 64:
        actual_hash = hashlib.sha256(raw.encode("utf-8")).hexdigest()
        if expected_hash.strip().lower() != actual_hash:
            findings.append(CardFinding("warning", "content_hash_mismatch", "skill_card verification hash does not match current SKILL.md content."))

    return findings


def _card_contract_findings(frontmatter: dict[str, Any], body: str, raw: str, skill_dir: Path) -> list[CardFinding]:
    findings: list[CardFinding] = []
    name = str(frontmatter.get("name") or skill_dir.name).strip()
    description = str(frontmatter.get("description") or "").strip()

    if not raw.startswith("---"):
        findings.append(CardFinding("error", "missing_frontmatter", "SKILL.md must start with YAML frontmatter."))
    if not name or not _NAME_RE.match(name):
        findings.append(CardFinding("error", "invalid_name", "Frontmatter `name` must be lowercase kebab/underscore identifier."))
    if not description:
        findings.append(CardFinding("error", "missing_description", "Frontmatter `description` is required for discoverability."))
    elif len(description) > 220:
        findings.append(CardFinding("warning", "long_description", "Description is too long for a compact skill card."))

    lower_body = body.lower()
    if len(body.strip()) < 400:
        findings.append(CardFinding("warning", "thin_body", "Body is thin; class-level skills need actionable workflow detail."))
    if not any(marker in lower_body for marker in ("when to use", "trigger", "use when", "scope")):
        findings.append(CardFinding("warning", "missing_trigger", "Add trigger/when-to-use guidance."))
    if not (_NUMBERED_STEP_RE.search(body) or any(marker in lower_body for marker in ("steps", "workflow", "procedure"))):
        findings.append(CardFinding("warning", "missing_workflow", "Add concrete workflow steps or a procedure section."))
    if not any(marker in lower_body for marker in ("verify", "verification", "validate", "checks", "tests")):
        findings.append(CardFinding("warning", "missing_verification", "Add verification/checks guidance."))
    if _SESSION_ARTIFACT_RE.search(body):
        findings.append(CardFinding("warning", "session_artifact", "Body appears to include PR/issue/SHA session artifacts; move stale detail to references or remove."))

    estimated_tokens = _estimate_tokens(raw)
    if estimated_tokens > 4000:
        findings.append(CardFinding("error", "prompt_bloat", f"Estimated {estimated_tokens:,} tokens; split details into references/templates/scripts."))
    elif estimated_tokens > 2000:
        findings.append(CardFinding("warning", "large_card", f"Estimated {estimated_tokens:,} tokens; review for prompt bloat."))

    return findings


def _scan_one(skill_md: Path, root: Path, source: str) -> tuple[SkillCardReport, set[str]]:
    raw = skill_md.read_text(encoding="utf-8")
    frontmatter, body = parse_frontmatter(raw)
    skill_dir = skill_md.parent
    name = str(frontmatter.get("name") or skill_dir.name).strip() or skill_dir.name
    guard = scan_skill(skill_dir, source="official" if source == "bundled" else "agent-created")
    rel = _relative_to_root(skill_dir, root)
    card_findings = _card_contract_findings(frontmatter, body, raw, skill_dir)
    card_findings.extend(_skill_card_findings(frontmatter.get("skill_card"), raw))
    report = SkillCardReport(
        name=name,
        source=source,
        path=str(skill_dir),
        rel_path=rel,
        char_count=len(raw),
        estimated_tokens=_estimate_tokens(raw),
        description=str(frontmatter.get("description") or "").strip(),
        guard_verdict=guard.verdict,
        guard_findings=len(guard.findings),
        card_findings=card_findings,
    )
    tokens = _token_set(f"{name} {report.description}\n{body}")
    return report, tokens


def audit_skills(*, include_bundled: bool = False, similarity_threshold: float = 0.42) -> SkillCleanerReport:
    """Build a read-only skill-cleaner report for the active Hermes profile."""
    reports: list[SkillCardReport] = []
    token_sets: dict[str, set[str]] = {}
    roots = _source_roots(include_bundled)

    for source in roots:
        if not source.root.is_dir():
            continue
        for skill_md in iter_skill_index_files(source.root, "SKILL.md"):
            try:
                report, tokens = _scan_one(skill_md, source.root, source.label)
            except (OSError, UnicodeDecodeError):
                continue
            key = f"{report.source}:{report.rel_path}:{report.name}"
            reports.append(report)
            token_sets[key] = tokens

    duplicates: list[DuplicateCandidate] = []
    keyed = [(f"{r.source}:{r.rel_path}:{r.name}", r) for r in reports]
    for idx, (left_key, left) in enumerate(keyed):
        for right_key, right in keyed[idx + 1 :]:
            if left.name == right.name and left.path != right.path:
                duplicates.append(DuplicateCandidate(left.name, right.name, 1.0, "same skill name in multiple locations"))
                continue
            sim = _jaccard(token_sets.get(left_key, set()), token_sets.get(right_key, set()))
            if sim >= similarity_threshold:
                duplicates.append(DuplicateCandidate(left.name, right.name, round(sim, 3), "high content-term overlap"))

    duplicates.sort(key=lambda d: d.similarity, reverse=True)
    reports.sort(key=lambda s: (s.estimated_tokens, s.name), reverse=True)
    return SkillCleanerReport(
        generated_at=datetime.now(timezone.utc).isoformat(),
        hermes_home=display_hermes_home(),
        scanned_roots=[str(s.root) for s in roots if s.root.is_dir()],
        skills=reports,
        duplicates=duplicates,
    )


def report_to_dict(report: SkillCleanerReport) -> dict[str, Any]:
    data = asdict(report)
    data["summary"] = {
        "skill_count": len(report.skills),
        "total_estimated_tokens": report.total_estimated_tokens,
        "finding_counts": report.finding_counts,
        "duplicate_candidates": len(report.duplicates),
    }
    return data


def format_markdown_report(report: SkillCleanerReport, *, duplicate_limit: int = 30, finding_limit: int = 80) -> str:
    counts = report.finding_counts
    lines = [
        "# Hermes Skill Cleaner Report",
        "",
        "Read-only audit. No skills were modified, archived, deleted, or rewritten.",
        "",
        f"- Generated: `{report.generated_at}`",
        f"- Hermes home: `{report.hermes_home}`",
        f"- Scanned roots: {', '.join(f'`{r}`' for r in report.scanned_roots) or '(none)' }",
        f"- Skills scanned: **{len(report.skills)}**",
        f"- Estimated loaded skill-card tokens: **{report.total_estimated_tokens:,}**",
        f"- Card findings: errors={counts.get('error', 0)}, warnings={counts.get('warning', 0)}, info={counts.get('info', 0)}",
        f"- Duplicate candidates: **{len(report.duplicates)}**",
        "",
        "## Largest skill cards",
        "",
        "| Skill | Source | Est. tokens | Guard | Path |",
        "|---|---:|---:|---|---|",
    ]
    for skill in report.skills[:20]:
        lines.append(
            f"| `{skill.name}` | {skill.source} | {skill.estimated_tokens:,} | "
            f"{skill.guard_verdict} ({skill.guard_findings}) | `{skill.rel_path}` |"
        )

    lines.extend(["", "## Verified skill-card findings", ""])
    emitted = 0
    for skill in report.skills:
        for finding in skill.card_findings:
            if emitted >= finding_limit:
                break
            lines.append(f"- **{finding.severity.upper()}** `{skill.name}` `{finding.code}` — {finding.message}")
            emitted += 1
        if emitted >= finding_limit:
            break
    if emitted == 0:
        lines.append("- No verified-card findings.")
    elif sum(len(s.card_findings) for s in report.skills) > emitted:
        lines.append(f"- ...truncated at {finding_limit} findings; see JSON for all findings.")

    lines.extend(["", "## Duplicate / consolidation candidates", ""])
    if not report.duplicates:
        lines.append("- No duplicate candidates above threshold.")
    else:
        lines.extend(["| Similarity | Left | Right | Reason |", "|---:|---|---|---|"])
        for dup in report.duplicates[:duplicate_limit]:
            lines.append(f"| {dup.similarity:.3f} | `{dup.left}` | `{dup.right}` | {dup.reason} |")
        if len(report.duplicates) > duplicate_limit:
            lines.append(f"\n_Truncated at {duplicate_limit} duplicate candidates; see JSON for all pairs._")

    lines.extend([
        "",
        "## Recommended next action",
        "",
        "Use this as scout data only. Consolidation, archive, delete, profile changes, or prompt-loader changes still require Workman approval gates.",
        "",
    ])
    return "\n".join(lines)


def write_report_files(report: SkillCleanerReport, output_dir: Path | None = None) -> tuple[Path, Path]:
    """Write markdown and JSON report artifacts under the active profile reports dir."""
    if output_dir is None:
        output_dir = get_hermes_home() / "reports" / "skill-cleaner"
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    md_path = output_dir / f"skill-cleaner-{stamp}.md"
    json_path = output_dir / f"skill-cleaner-{stamp}.json"
    md_path.write_text(format_markdown_report(report), encoding="utf-8")
    json_path.write_text(json.dumps(report_to_dict(report), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return md_path, json_path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a read-only Hermes skill cleaner audit for the active profile.",
    )
    parser.add_argument(
        "--include-bundled",
        action="store_true",
        help="Also scan bundled repo/package skills in addition to the active profile skills.",
    )
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=0.42,
        help="Jaccard token-overlap threshold for duplicate candidates (default: 0.42).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for markdown+JSON artifacts (default: <HERMES_HOME>/reports/skill-cleaner).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print JSON summary instead of markdown after writing artifacts.",
    )
    parser.add_argument(
        "--no-write",
        action="store_true",
        help="Print the report only; do not write artifacts.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    report = audit_skills(
        include_bundled=args.include_bundled,
        similarity_threshold=args.similarity_threshold,
    )

    md_path: Path | None = None
    json_path: Path | None = None
    if not args.no_write:
        md_path, json_path = write_report_files(report, output_dir=args.output_dir)

    if args.json:
        payload = report_to_dict(report)
        if md_path and json_path:
            payload["artifacts"] = {"markdown": str(md_path), "json": str(json_path)}
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    else:
        print(format_markdown_report(report))
        if md_path and json_path:
            print(f"\nArtifacts written:\n- {md_path}\n- {json_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through CLI smoke tests
    raise SystemExit(main())
