"""Read-only integrity checks for bundled skills."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any, Callable, Sequence


@dataclass(frozen=True)
class SkillValidationIssue:
    severity: str
    code: str
    skill: str
    path: Path
    line: int
    target: str
    message: str


@dataclass(frozen=True)
class _SkillRecord:
    name: str
    path: Path
    skill_md: Path
    text: str
    frontmatter: dict[str, Any]


@dataclass(frozen=True)
class SkillValidationContext:
    root: Path
    records: tuple[_SkillRecord, ...]
    known_names: frozenset[str]


SkillValidationCheck = Callable[
    [_SkillRecord, SkillValidationContext], list[SkillValidationIssue]
]


_LOCAL_REF_RE = re.compile(
    r"(?<![\w./-])((?:scripts|references|templates)/[A-Za-z0-9][A-Za-z0-9._/-]*)"
)
_SKILL_VIEW_RE = re.compile(r"\bskill_view\(\s*[\"']([^\"']+)[\"']")
_SKILL_MANAGE_RE = re.compile(
    r"\bskill_manage\(\s*(?:[\"'][^\"']+[\"']\s*,\s*)?[\"']([^\"']+)[\"']"
)
_BASH_FENCE_RE = re.compile(r"```(?:bash|sh|shell)\s*\n(.*?)```", re.DOTALL)
_HERMES_TOOL_NAMES = {
    "read_file",
    "search_files",
    "write_file",
    "edit_file",
    "list_files",
}


def validate_bundled_skills(
    bundled_dir: str | Path,
    *,
    checks: Sequence[str] | None = None,
) -> list[SkillValidationIssue]:
    """Return deterministic integrity issues for bundled skill markdown.

    These checks intentionally stay local and read-only. They catch drift that
    does not require model judgment: missing bundled assets, broken skill
    references, and Hermes tool calls mistakenly presented as shell commands.
    """
    root = Path(bundled_dir)
    records = tuple(_discover_skills(root))
    context = SkillValidationContext(
        root=root,
        records=records,
        known_names=frozenset(record.name for record in records),
    )
    active_checks = _resolve_checks(checks)
    issues: list[SkillValidationIssue] = []

    for record in records:
        for check in active_checks:
            issues.extend(check(record, context))

    return sorted(
        issues,
        key=lambda issue: (
            str(issue.path),
            issue.line,
            issue.code,
            issue.target,
        ),
    )


def available_bundled_skill_checks() -> tuple[str, ...]:
    """Return the stable names of bundled-skill validation checks."""
    return tuple(_CHECKS)


def _resolve_checks(checks: Sequence[str] | None) -> tuple[SkillValidationCheck, ...]:
    if checks is None:
        return tuple(_CHECKS.values())
    unknown = [name for name in checks if name not in _CHECKS]
    if unknown:
        choices = ", ".join(available_bundled_skill_checks())
        raise ValueError(f"Unknown bundled skill validation check(s): {', '.join(unknown)}. Choices: {choices}")
    return tuple(_CHECKS[name] for name in checks)


def _discover_skills(root: Path) -> list[_SkillRecord]:
    records: list[_SkillRecord] = []
    if not root.exists():
        return records

    for skill_md in sorted(root.rglob("SKILL.md")):
        parts = set(skill_md.parts)
        if ".git" in parts or ".github" in parts or ".hub" in parts:
            continue
        text = skill_md.read_text(encoding="utf-8", errors="replace")
        frontmatter = _read_frontmatter(text)
        name = str(frontmatter.get("name") or skill_md.parent.name).strip()
        records.append(
            _SkillRecord(
                name=name,
                path=skill_md.parent,
                skill_md=skill_md,
                text=text,
                frontmatter=frontmatter,
            )
        )
    return records


def _read_frontmatter(text: str) -> dict[str, Any]:
    if not text.startswith("---"):
        return {}
    end = text.find("\n---", 3)
    if end == -1:
        return {}
    body = text[3:end].strip()
    if not body:
        return {}
    try:
        import yaml

        parsed = yaml.safe_load(body)
    except Exception:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _validate_local_file_refs(
    record: _SkillRecord, context: SkillValidationContext
) -> list[SkillValidationIssue]:
    issues: list[SkillValidationIssue] = []
    seen: set[tuple[str, int]] = set()
    for match in _LOCAL_REF_RE.finditer(record.text):
        target = match.group(1).rstrip(".,;:)]}")
        line = _line_for_match(record.text, match.start(1))
        if (target, line) in seen:
            continue
        seen.add((target, line))
        if not (record.path / target).exists():
            issues.append(
                SkillValidationIssue(
                    severity="high",
                    code="missing-file-reference",
                    skill=record.name,
                    path=record.skill_md,
                    line=line,
                    target=target,
                    message=f"References bundled file '{target}', but it does not exist.",
                )
            )
    return issues


def _validate_related_skills(
    record: _SkillRecord, context: SkillValidationContext
) -> list[SkillValidationIssue]:
    related = _related_skills(record.frontmatter)

    issues: list[SkillValidationIssue] = []
    for target in related:
        target = target.strip()
        if target and target not in context.known_names:
            issues.append(
                SkillValidationIssue(
                    severity="medium",
                    code="unknown-related-skill",
                    skill=record.name,
                    path=record.skill_md,
                    line=_frontmatter_line(record.text, "related_skills"),
                    target=target,
                    message=f"related_skills references unknown bundled skill '{target}'.",
                )
            )
    return issues


def _related_skills(frontmatter: dict[str, Any]) -> list[str]:
    """Read related skills using the same metadata.hermes fallback as the loader."""
    hermes_meta: dict[str, Any] = {}
    metadata = frontmatter.get("metadata")
    if isinstance(metadata, dict):
        raw_hermes_meta = metadata.get("hermes", {}) or {}
        if isinstance(raw_hermes_meta, dict):
            hermes_meta = raw_hermes_meta

    return _parse_string_list(
        hermes_meta.get("related_skills") or frontmatter.get("related_skills", "")
    )


def _parse_string_list(value: Any) -> list[str]:
    if not value:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]

    text = str(value).strip()
    if text.startswith("[") and text.endswith("]"):
        text = text[1:-1]
    return [item.strip().strip("\"'") for item in text.split(",") if item.strip()]


def _validate_skill_tool_refs(
    record: _SkillRecord, context: SkillValidationContext
) -> list[SkillValidationIssue]:
    issues: list[SkillValidationIssue] = []
    for match in _SKILL_VIEW_RE.finditer(record.text):
        target = match.group(1).strip()
        if _should_check_skill_target(target) and target not in context.known_names:
            issues.append(
                SkillValidationIssue(
                    severity="medium",
                    code="unknown-skill-view-reference",
                    skill=record.name,
                    path=record.skill_md,
                    line=_line_for_match(record.text, match.start(1)),
                    target=target,
                    message=f"skill_view references unknown bundled skill '{target}'.",
                )
            )

    for match in _SKILL_MANAGE_RE.finditer(record.text):
        target = match.group(1).strip()
        if _should_check_skill_target(target) and target not in context.known_names:
            issues.append(
                SkillValidationIssue(
                    severity="medium",
                    code="unknown-skill-manage-reference",
                    skill=record.name,
                    path=record.skill_md,
                    line=_line_for_match(record.text, match.start(1)),
                    target=target,
                    message=f"skill_manage references unknown bundled skill '{target}'.",
                )
            )
    return issues


def _validate_bash_fences(
    record: _SkillRecord, context: SkillValidationContext
) -> list[SkillValidationIssue]:
    issues: list[SkillValidationIssue] = []
    for fence in _BASH_FENCE_RE.finditer(record.text):
        body = fence.group(1)
        for tool_name in sorted(_HERMES_TOOL_NAMES):
            tool_match = re.search(rf"^\s*{re.escape(tool_name)}\b", body, re.MULTILINE)
            if not tool_match:
                continue
            issues.append(
                SkillValidationIssue(
                    severity="medium",
                    code="hermes-tool-in-bash-fence",
                    skill=record.name,
                    path=record.skill_md,
                    line=_line_for_match(record.text, fence.start(1) + tool_match.start()),
                    target=tool_name,
                    message=(
                        f"Hermes tool '{tool_name}' appears in a bash fence; "
                        "readers may execute it as a shell command."
                    ),
                )
            )
    return issues


_CHECKS: dict[str, SkillValidationCheck] = {
    "local-file-references": _validate_local_file_refs,
    "related-skills": _validate_related_skills,
    "skill-tool-references": _validate_skill_tool_refs,
    "bash-fence-tools": _validate_bash_fences,
}


def _should_check_skill_target(target: str) -> bool:
    if not target:
        return False
    if "/" in target or ":" in target:
        return False
    if target.startswith(("http://", "https://")):
        return False
    return True


def _line_for_match(text: str, index: int) -> int:
    return text.count("\n", 0, index) + 1


def _frontmatter_line(text: str, key: str) -> int:
    for lineno, line in enumerate(text.splitlines(), start=1):
        if line.strip().startswith(f"{key}:"):
            return lineno
    return 1
