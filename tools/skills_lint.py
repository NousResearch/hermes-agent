#!/usr/bin/env python3
"""
Skills Lint — Structure validator for SKILL.md files.

Complements skills_guard (security) and the hub update checker: this module
validates the *shape* of a skill — frontmatter syntax, required fields, field
constraints, and requirement declarations — surfacing problems the runtime
otherwise hides (lenient YAML fallback, silently dropped env-var entries,
silent name/description truncation).

Rule IDs are stable (HSLxxx) so JSON output can be consumed by CI and so
individual rules can be suppressed in the future.

Usage:
    from tools.skills_lint import lint_skill, format_lint_report

    result = lint_skill(Path("skills/category/my-skill"))
    if result.errors:
        print(format_lint_report(result))
"""

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, FrozenSet, Iterable, List, Optional, Tuple


SEVERITY_ERROR = "error"
SEVERITY_WARNING = "warning"

# Top-level frontmatter keys actually read somewhere in the codebase.
KNOWN_FRONTMATTER_KEYS = frozenset(
    {
        "name",
        "description",
        "version",
        "author",
        "license",
        "platforms",
        "environments",
        "compatibility",
        "dependencies",
        "prerequisites",
        "required_environment_variables",
        "required_credential_files",
        "setup",
        "metadata",
        "tags",
        "triggers",
    }
)

VALID_PLATFORMS = frozenset({"linux", "macos", "windows"})

_BODY_PATH_RE = re.compile(
    r"(?<![\w/.])((?:\./)?[A-Za-z0-9_-]+(?:/[A-Za-z0-9_-]+)*"
    r"/[A-Za-z0-9_.-]+\.(?:py|sh|bash|js|ts|rb|ps1|json|ya?ml|toml|csv|md|txt))\b"
)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class LintFinding:
    rule_id: str        # "HSL0xx"
    severity: str       # "error" | "warning"
    field: str          # frontmatter key or relative file path
    message: str

    def to_dict(self) -> Dict[str, str]:
        return {
            "rule_id": self.rule_id,
            "severity": self.severity,
            "field": self.field,
            "message": self.message,
        }


@dataclass
class LintResult:
    skill_path: Path
    skill_name: str
    findings: List[LintFinding] = field(default_factory=list)

    @property
    def errors(self) -> List[LintFinding]:
        return [f for f in self.findings if f.severity == SEVERITY_ERROR]

    @property
    def warnings(self) -> List[LintFinding]:
        return [f for f in self.findings if f.severity == SEVERITY_WARNING]

    def exit_relevant(self, fail_on: str) -> bool:
        """True when findings meet/exceed the --fail-on threshold."""
        if fail_on == SEVERITY_WARNING:
            return bool(self.findings)
        return bool(self.errors)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "skill_path": str(self.skill_path),
            "skill_name": self.skill_name,
            "errors": len(self.errors),
            "warnings": len(self.warnings),
            "findings": [f.to_dict() for f in self.findings],
        }


@dataclass
class LintContext:
    skill_dir: Path
    raw: str
    frontmatter: Dict[str, Any]
    body: str
    yaml_error: Optional[str]
    known_skill_names: Optional[FrozenSet[str]] = None


# ---------------------------------------------------------------------------
# Rule registry
# ---------------------------------------------------------------------------

# Each rule yields (field, message) pairs; severity lives in the registration.
RuleFn = Callable[[LintContext], Iterable[Tuple[str, str]]]
RULES: List[Tuple[str, str, RuleFn]] = []

# Structural rules run first; when any of them fires the field rules are
# skipped, because an empty/garbage frontmatter dict would cascade into
# misleading "name missing"/"description missing" noise.
_STRUCTURAL_RULE_IDS = frozenset({"HSL001", "HSL002", "HSL003"})


def rule(rule_id: str, severity: str) -> Callable[[RuleFn], RuleFn]:
    def register(fn: RuleFn) -> RuleFn:
        RULES.append((rule_id, severity, fn))
        return fn

    return register


# ── Structural rules ───────────────────────────────────────────────────────

@rule("HSL001", SEVERITY_ERROR)
def _check_fence(ctx: LintContext):
    if not ctx.raw.startswith("---"):
        yield "frontmatter", "SKILL.md has no YAML frontmatter block (must start with '---')"
    elif not re.search(r"\n---\s*\n", ctx.raw[3:]):
        yield "frontmatter", "frontmatter fence is unterminated (no closing '---' line)"


@rule("HSL002", SEVERITY_ERROR)
def _check_yaml(ctx: LintContext):
    if ctx.yaml_error and "expected a mapping" not in ctx.yaml_error:
        yield (
            "frontmatter",
            f"YAML parse failure: {ctx.yaml_error.splitlines()[0]} — at runtime this "
            "silently falls back to naive key:value parsing, dropping nested fields",
        )


@rule("HSL003", SEVERITY_ERROR)
def _check_mapping(ctx: LintContext):
    if ctx.yaml_error and "expected a mapping" in ctx.yaml_error:
        yield "frontmatter", f"frontmatter must be a YAML mapping: {ctx.yaml_error}"


# ── name / version / description ───────────────────────────────────────────

@rule("HSL010", SEVERITY_ERROR)
def _check_name_present(ctx: LintContext):
    name = ctx.frontmatter.get("name")
    if not isinstance(name, str) or not name.strip():
        yield "name", "'name' is missing or empty"


@rule("HSL011", SEVERITY_ERROR)
def _check_name_length(ctx: LintContext):
    from tools.skills_tool import MAX_NAME_LENGTH

    name = ctx.frontmatter.get("name")
    if isinstance(name, str) and len(name) > MAX_NAME_LENGTH:
        yield (
            "name",
            f"'name' is {len(name)} chars (max {MAX_NAME_LENGTH}); "
            "the runtime truncates it silently",
        )


@rule("HSL012", SEVERITY_ERROR)
def _check_name_safe(ctx: LintContext):
    from tools.skills_hub import _validate_skill_name

    name = ctx.frontmatter.get("name")
    if isinstance(name, str) and name.strip():
        try:
            _validate_skill_name(name.strip())
        except ValueError as exc:
            yield "name", f"'name' would be rejected at install time: {exc}"


@rule("HSL013", SEVERITY_WARNING)
def _check_name_matches_dir(ctx: LintContext):
    name = ctx.frontmatter.get("name")
    if isinstance(name, str) and name.strip() and name.strip() != ctx.skill_dir.name:
        yield (
            "name",
            f"'name' ({name.strip()!r}) differs from directory name "
            f"({ctx.skill_dir.name!r}); list/uninstall key on the frontmatter name",
        )


@rule("HSL014", SEVERITY_WARNING)
def _check_version(ctx: LintContext):
    from packaging.version import InvalidVersion, Version

    version = ctx.frontmatter.get("version")
    if version is None:
        return
    try:
        Version(str(version))
    except InvalidVersion:
        yield "version", f"'version' ({version!r}) is not a valid version string"


@rule("HSL015", SEVERITY_ERROR)
def _check_description_present(ctx: LintContext):
    desc = ctx.frontmatter.get("description")
    if not isinstance(desc, str) or not desc.strip():
        yield "description", "'description' is missing or empty"


@rule("HSL016", SEVERITY_ERROR)
def _check_description_length(ctx: LintContext):
    from tools.skills_tool import MAX_DESCRIPTION_LENGTH

    desc = ctx.frontmatter.get("description")
    if isinstance(desc, str) and len(desc) > MAX_DESCRIPTION_LENGTH:
        yield (
            "description",
            f"'description' is {len(desc)} chars (max {MAX_DESCRIPTION_LENGTH}); "
            "the runtime truncates it silently",
        )


# ── platforms / unknown keys ───────────────────────────────────────────────

@rule("HSL020", SEVERITY_ERROR)
def _check_platforms(ctx: LintContext):
    platforms = ctx.frontmatter.get("platforms")
    if platforms is None:
        return
    if isinstance(platforms, str):
        platforms = [platforms]
    if not isinstance(platforms, list):
        yield "platforms", f"'platforms' must be a list or string, got {type(platforms).__name__}"
        return
    for entry in platforms:
        normalized = str(entry).strip().lower()
        if normalized not in VALID_PLATFORMS:
            yield (
                "platforms",
                f"unknown platform {entry!r} (valid: {', '.join(sorted(VALID_PLATFORMS))})",
            )


@rule("HSL021", SEVERITY_WARNING)
def _check_unknown_keys(ctx: LintContext):
    for key in ctx.frontmatter:
        if str(key) not in KNOWN_FRONTMATTER_KEYS:
            yield str(key), f"unknown frontmatter key {key!r} is ignored by the runtime"


# ── requirement declarations ───────────────────────────────────────────────

def _iter_declared_env_names(ctx: LintContext):
    """Yield (source_field, raw_name) for every declared env var, all formats."""
    raw = ctx.frontmatter.get("required_environment_variables")
    if isinstance(raw, dict):
        raw = [raw]
    if isinstance(raw, list):
        for entry in raw:
            if isinstance(entry, str):
                yield "required_environment_variables", entry
            elif isinstance(entry, dict):
                name = entry.get("name") or entry.get("env_var")
                if name:
                    yield "required_environment_variables", str(name)

    setup = ctx.frontmatter.get("setup")
    if isinstance(setup, dict):
        secrets = setup.get("collect_secrets")
        if isinstance(secrets, list):
            for entry in secrets:
                if isinstance(entry, dict) and entry.get("env_var"):
                    yield "setup.collect_secrets", str(entry["env_var"])

    prereq = ctx.frontmatter.get("prerequisites")
    if isinstance(prereq, dict):
        env_vars = prereq.get("env_vars")
        if isinstance(env_vars, list):
            for entry in env_vars:
                yield "prerequisites.env_vars", str(entry)


@rule("HSL030", SEVERITY_ERROR)
def _check_env_var_entries(ctx: LintContext):
    raw = ctx.frontmatter.get("required_environment_variables")
    if raw is None:
        return
    if isinstance(raw, dict):
        raw = [raw]
    if not isinstance(raw, list):
        yield (
            "required_environment_variables",
            f"must be a list, got {type(raw).__name__}; the runtime ignores it entirely",
        )
        return
    for i, entry in enumerate(raw):
        if isinstance(entry, str):
            continue
        if not isinstance(entry, dict):
            yield (
                f"required_environment_variables[{i}]",
                f"entry must be a string or mapping, got {type(entry).__name__}; "
                "the runtime drops it silently",
            )
        elif not str(entry.get("name") or entry.get("env_var") or "").strip():
            yield (
                f"required_environment_variables[{i}]",
                "entry has no 'name' (or 'env_var'); the runtime drops it silently",
            )


@rule("HSL031", SEVERITY_ERROR)
def _check_env_var_names(ctx: LintContext):
    from tools.skills_tool import _ENV_VAR_NAME_RE

    for source_field, name in _iter_declared_env_names(ctx):
        candidate = name.strip()
        if candidate and not _ENV_VAR_NAME_RE.match(candidate):
            yield (
                source_field,
                f"invalid environment variable name {candidate!r}; "
                "the runtime drops it silently",
            )


@rule("HSL032", SEVERITY_WARNING)
def _check_legacy_prerequisites(ctx: LintContext):
    if "prerequisites" in ctx.frontmatter:
        yield (
            "prerequisites",
            "legacy 'prerequisites' block; migrate env_vars to "
            "'required_environment_variables' entries with prompt/help text",
        )


@rule("HSL040", SEVERITY_WARNING)
def _check_dependencies(ctx: LintContext):
    from packaging.requirements import InvalidRequirement, Requirement

    deps = ctx.frontmatter.get("dependencies")
    if deps is None:
        return
    if not isinstance(deps, list):
        yield "dependencies", f"'dependencies' must be a list, got {type(deps).__name__}"
        return
    for entry in deps:
        try:
            Requirement(str(entry))
        except InvalidRequirement:
            yield "dependencies", f"invalid pip requirement spec {entry!r}"


# ── cross-references ───────────────────────────────────────────────────────

@rule("HSL050", SEVERITY_WARNING)
def _check_related_skills(ctx: LintContext):
    if ctx.known_skill_names is None:
        return
    metadata = ctx.frontmatter.get("metadata")
    if not isinstance(metadata, dict):
        return
    hermes_meta = metadata.get("hermes")
    if not isinstance(hermes_meta, dict):
        return
    related = hermes_meta.get("related_skills")
    if not isinstance(related, list):
        return
    for entry in related:
        name = str(entry).strip()
        if name and name not in ctx.known_skill_names:
            yield (
                "metadata.hermes.related_skills",
                f"related skill {name!r} not found among installed skills",
            )


@rule("HSL060", SEVERITY_WARNING)
def _check_body_file_refs(ctx: LintContext):
    seen: set = set()
    for match in _BODY_PATH_RE.finditer(ctx.body):
        candidate = match.group(1)
        if candidate in seen or "://" in candidate:
            continue
        seen.add(candidate)
        # Anchor heuristic: only flag paths that are clearly skill-relative —
        # either explicitly "./"-prefixed, or rooted at a directory the skill
        # actually ships. Bare paths like "src/auth.py" in prose are usually
        # examples about the *user's* project, not skill resources.
        relative = candidate[2:] if candidate.startswith("./") else candidate
        first_segment = relative.split("/", 1)[0]
        anchored = (
            candidate.startswith("./")
            or (ctx.skill_dir / first_segment).is_dir()
        )
        if anchored and not (ctx.skill_dir / relative).exists():
            yield candidate, f"referenced file {candidate!r} does not exist in the skill directory"


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------

def lint_skill(
    skill_path: Path,
    *,
    known_skill_names: Optional[FrozenSet[str]] = None,
) -> LintResult:
    """Lint one skill directory (or a SKILL.md path) and return all findings."""
    skill_path = Path(skill_path)
    skill_dir = skill_path.parent if skill_path.name == "SKILL.md" else skill_path
    # Resolve so Path(".") and trailing-separator inputs get a real directory
    # name — HSL013 compares against it and the report displays it.
    skill_dir = skill_dir.resolve()
    skill_md = skill_dir / "SKILL.md"

    result = LintResult(skill_path=skill_dir, skill_name=skill_dir.name)

    try:
        raw = skill_md.read_text(encoding="utf-8")
    except FileNotFoundError:
        result.findings.append(
            LintFinding("HSL001", SEVERITY_ERROR, "SKILL.md", "SKILL.md not found")
        )
        return result
    except (UnicodeDecodeError, PermissionError) as exc:
        result.findings.append(
            LintFinding("HSL001", SEVERITY_ERROR, "SKILL.md", f"cannot read SKILL.md: {exc}")
        )
        return result

    from agent.skill_utils import parse_frontmatter_strict

    frontmatter, body, yaml_error = parse_frontmatter_strict(raw)
    ctx = LintContext(
        skill_dir=skill_dir,
        raw=raw,
        frontmatter=frontmatter,
        body=body,
        yaml_error=yaml_error,
        known_skill_names=known_skill_names,
    )

    name = frontmatter.get("name")
    if isinstance(name, str) and name.strip():
        result.skill_name = name.strip()

    structural = [r for r in RULES if r[0] in _STRUCTURAL_RULE_IDS]
    field_rules = [r for r in RULES if r[0] not in _STRUCTURAL_RULE_IDS]

    for rule_id, severity, fn in structural:
        for finding_field, message in fn(ctx):
            result.findings.append(LintFinding(rule_id, severity, finding_field, message))

    if result.findings:
        return result

    for rule_id, severity, fn in field_rules:
        for finding_field, message in fn(ctx):
            result.findings.append(LintFinding(rule_id, severity, finding_field, message))

    return result


def collect_installed_skill_names() -> FrozenSet[str]:
    """Names of all installed skills (including disabled), for HSL050."""
    from tools.skills_tool import _find_all_skills

    return frozenset(s["name"] for s in _find_all_skills(skip_disabled=True))


def find_installed_skill_paths() -> Dict[str, Path]:
    """Map installed skill names (frontmatter name, dirname fallback) to dirs.

    Mirrors the enumeration order of skills_tool._find_all_skills: the local
    skills dir wins over external dirs, first occurrence of a name wins.
    """
    from agent.skill_utils import (
        get_external_skills_dirs,
        is_excluded_skill_path,
        iter_skill_index_files,
        parse_frontmatter,
    )
    from tools.skills_tool import SKILLS_DIR

    paths: Dict[str, Path] = {}
    dirs_to_scan = []
    if SKILLS_DIR.exists():
        dirs_to_scan.append(SKILLS_DIR)
    dirs_to_scan.extend(get_external_skills_dirs())

    for scan_dir in dirs_to_scan:
        for skill_md in iter_skill_index_files(scan_dir, "SKILL.md"):
            if is_excluded_skill_path(skill_md):
                continue
            try:
                frontmatter, _ = parse_frontmatter(
                    skill_md.read_text(encoding="utf-8")[:4000]
                )
            except Exception:
                frontmatter = {}
            name = str(frontmatter.get("name") or skill_md.parent.name).strip()
            paths.setdefault(name, skill_md.parent)
            paths.setdefault(skill_md.parent.name, skill_md.parent)

    return paths


def format_lint_report(result: LintResult) -> str:
    """Compact multi-line report for one skill, suitable for CLI display."""
    if not result.findings:
        status = "OK"
    elif result.errors:
        status = f"{len(result.errors)} error(s), {len(result.warnings)} warning(s)"
    else:
        status = f"{len(result.warnings)} warning(s)"

    lines = [f"Lint: {result.skill_name} ({result.skill_path})  {status}"]
    severity_order = {SEVERITY_ERROR: 0, SEVERITY_WARNING: 1}
    for f in sorted(result.findings, key=lambda f: (severity_order.get(f.severity, 2), f.rule_id)):
        sev = f.severity.upper().ljust(8)
        lines.append(f"  {f.rule_id}  {sev} {f.field}: {f.message}")
    return "\n".join(lines)
