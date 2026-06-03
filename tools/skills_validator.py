"""Static validation of installed skills against a configurable ruleset.

Complementary to the Curator (`agent/curator.py`). Where Curator performs
LLM-driven mutation of agent-authored skills, this module is a read-only
structural linter that operates on any skill — bundled, hub-installed, or
agent-authored — and reports findings without modifying anything.

Two built-in rulesets:
  - `hermes`      — Matches the existing `_validate_frontmatter` contract
                    in `tools/skill_manager_tool.py` (name + description +
                    non-empty body). Lenient.
  - `agentskills` — Full agentskills.io frontmatter + section + line-count
                    spec. Strict.

Custom rulesets can be loaded from a YAML file (see docs).

The validator never writes to disk. It returns a `ValidationReport` that
CLI / tooling can render as needed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set

import yaml


SEMVER_RE = re.compile(r"^\d+\.\d+(\.\d+)?$")
KEBAB_RE = re.compile(r"^[a-z][a-z0-9-]*$")

DEFAULT_MAX_DESCRIPTION = 1024
DEFAULT_MAX_LINES = 500

KNOWN_COMPLEXITIES = {"basic", "intermediate", "advanced"}
TRIGGER_VALUES = {"manual", "cron", "slash", "preload"}

SEVERITY_BLOCKING = "BLOCKING"
SEVERITY_SUGGEST = "SUGGEST"
SEVERITY_OK = "OK"
_VALID_SEVERITIES = {SEVERITY_BLOCKING, SEVERITY_SUGGEST, SEVERITY_OK}

# Common keywords/CLI commands/false-positives in body backtick refs
_BODY_DEAD_REF_SKIP = {
"sh", "md", "yml", "yaml", "json", "toml", "env", "git",
"py", "js", "ts", "go", "rs", "rb", "php", "html", "css",
"txt", "csv", "xml", "png", "jpg", "gif", "svg", "pdf",
"http", "https", "api", "url", "uri", "cli", "ui", "ux",
"stdout", "stdin", "stderr", "config", "debug", "release",
"curl", "wget", "grep", "sed", "awk", "cat", "less", "tail",
"head", "find", "ls", "cp", "mv", "rm", "mkdir", "chmod",
"chown", "ps", "top", "kill", "ssh", "scp", "rsync",
"docker", "kubectl", "helm",
"pip", "npm", "yarn", "apt", "yum", "brew", "cargo",
"python3", "node", "ruby", "perl", "php", "go",
"ffmpeg", "ffprobe", "sox", "convert", "magick",
"jq", "yq", "xmllint", "csvtool", "pandoc",
"left", "right", "top", "bottom", "up", "down",
"name", "type", "kind", "size", "path", "mode",
"start", "end", "next", "prev", "last", "first",
"true", "false", "yes", "no", "on", "off",
"age", "message", "user", "bot",
"str", "int", "bool", "float", "var", "val", "func",
"v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9",
}


BUILTIN_RULESETS: Dict[str, Dict[str, str]] = {
    "hermes": {
        "frontmatter.present": SEVERITY_BLOCKING,
        "frontmatter.parseable": SEVERITY_BLOCKING,
        "frontmatter.name.present": SEVERITY_BLOCKING,
        "frontmatter.description.present": SEVERITY_BLOCKING,
        "frontmatter.description.length": SEVERITY_BLOCKING,
        "frontmatter.metadata.hermes.present": SEVERITY_SUGGEST,
        "frontmatter.metadata.hermes.trigger.valid": SEVERITY_SUGGEST,
        "frontmatter.related_skills.valid": SEVERITY_SUGGEST,
        "body.non_empty": SEVERITY_BLOCKING,
        "body.dead_references": SEVERITY_SUGGEST,
    },
    "agentskills": {
        "frontmatter.present": SEVERITY_BLOCKING,
        "frontmatter.parseable": SEVERITY_BLOCKING,
        "frontmatter.name.present": SEVERITY_BLOCKING,
        "frontmatter.name.kebab_case": SEVERITY_BLOCKING,
        "frontmatter.name.matches_dir": SEVERITY_BLOCKING,
        "frontmatter.description.present": SEVERITY_BLOCKING,
        "frontmatter.description.length": SEVERITY_BLOCKING,
        "frontmatter.license.present": SEVERITY_SUGGEST,
        "frontmatter.allowed_tools.present": SEVERITY_SUGGEST,
        "frontmatter.metadata.author.present": SEVERITY_SUGGEST,
        "frontmatter.metadata.version.present": SEVERITY_SUGGEST,
        "frontmatter.metadata.version.semver": SEVERITY_SUGGEST,
        "frontmatter.metadata.domain.present": SEVERITY_SUGGEST,
        "frontmatter.metadata.complexity.valid": SEVERITY_SUGGEST,
        "frontmatter.metadata.tags.present": SEVERITY_SUGGEST,
        "frontmatter.metadata.hermes.present": SEVERITY_SUGGEST,
        "frontmatter.metadata.hermes.trigger.valid": SEVERITY_SUGGEST,
        "frontmatter.related_skills.valid": SEVERITY_BLOCKING,
        "section.when_to_use.present": SEVERITY_SUGGEST,
        "section.procedure.present": SEVERITY_SUGGEST,
        "body.non_empty": SEVERITY_BLOCKING,
        "body.line_count": SEVERITY_SUGGEST,
        "body.dead_references": SEVERITY_SUGGEST,
    },
}


@dataclass
class Finding:
    rule: str
    severity: str
    message: str


@dataclass
class ValidationReport:
    skill_path: Path
    skill_name: str
    ruleset_name: str
    findings: List[Finding] = field(default_factory=list)

    @property
    def blocking(self) -> List[Finding]:
        return [f for f in self.findings if f.severity == SEVERITY_BLOCKING]

    @property
    def suggestions(self) -> List[Finding]:
        return [f for f in self.findings if f.severity == SEVERITY_SUGGEST]

    @property
    def has_blocking(self) -> bool:
        return any(f.severity == SEVERITY_BLOCKING for f in self.findings)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _deep_get(d: dict, *keys: str, default=None):
    """Safely traverse nested dicts."""
    for k in keys:
        if isinstance(d, dict):
            d = d.get(k)
        else:
            return default
    return d


def _find_related_skills(fm: dict) -> List[str]:
    """Extract related_skills list, checking metadata.hermes first, then top-level."""
    # Try metadata.hermes.related_skills first
    rs = _deep_get(fm, "metadata", "hermes", "related_skills")
    if isinstance(rs, list):
        return rs
    if isinstance(rs, str):
        return [rs]

    # Fallback to top-level related_skills
    rs = fm.get("related_skills", [])
    if isinstance(rs, list):
        return rs
    if isinstance(rs, str):
        return [rs]

    return []


def _find_body_dead_refs(body: str, valid_names: Set[str]) -> List[str]:
    """Find backtick-wrapped strings in body that look like skill names but don't exist."""
    refs = re.findall(r"`([a-z][a-z0-9-]*[a-z0-9])`", body)
    dead = []
    for ref in refs:
        if ref in _BODY_DEAD_REF_SKIP:
            continue
        if ref not in valid_names:
            dead.append(ref)
    return dead


def build_skill_name_index(skills_root: Path) -> Set[str]:
    """Build set of valid skill YAML names from all SKILL.md files.

    Scans every SKILL.md under `skills_root`, parses frontmatter, and
    collects the ``name`` field.  Skips anything under an ``.archive``
    segment.
    """
    names: Set[str] = set()
    for p in sorted(skills_root.rglob("SKILL.md")):
        if any(part == ".archive" for part in p.parts):
            continue
        try:
            content = p.read_text(encoding="utf-8")
        except Exception:
            continue
        fm, _, _ = _parse_frontmatter(content)
        if fm and isinstance(fm, dict) and "name" in fm:
            n = str(fm["name"]).strip()
            if n:
                names.add(n)
    return names


# ---------------------------------------------------------------------------
# Rulesets
# ---------------------------------------------------------------------------

def load_ruleset(name_or_path: str) -> Dict[str, str]:
    """Resolve a built-in ruleset name or a YAML file path to {rule: severity}.

    Built-in names: 'hermes', 'agentskills'.
    Custom files must be valid YAML with a top-level 'rules' mapping
    of `rule_id: severity`. Severities are case-insensitive and must be
    one of BLOCKING, SUGGEST, OK.
    """
    if name_or_path in BUILTIN_RULESETS:
        return dict(BUILTIN_RULESETS[name_or_path])

    p = Path(name_or_path)
    if not p.exists():
        builtins = sorted(BUILTIN_RULESETS)
        raise ValueError(
            f"Ruleset '{name_or_path}' is not a built-in and not a readable file. "
            f"Built-ins: {builtins}."
        )

    try:
        data = yaml.safe_load(p.read_text(encoding="utf-8"))
    except yaml.YAMLError as e:
        raise ValueError(f"Custom ruleset '{p}' has invalid YAML: {e}") from e

    if not isinstance(data, dict) or "rules" not in data:
        raise ValueError(
            f"Custom ruleset '{p}' must be a mapping with a 'rules' key."
        )

    rules_raw = data["rules"]
    if not isinstance(rules_raw, dict):
        raise ValueError(
            f"Custom ruleset '{p}': 'rules' must be a mapping of rule_id -> severity."
        )

    resolved: Dict[str, str] = {}
    for rule_id, sev in rules_raw.items():
        sev_norm = str(sev).strip().upper()
        if sev_norm not in _VALID_SEVERITIES:
            raise ValueError(
                f"Custom ruleset '{p}': rule '{rule_id}' has invalid severity "
                f"'{sev}'. Must be one of {sorted(_VALID_SEVERITIES)}."
            )
        resolved[str(rule_id)] = sev_norm
    return resolved


# ---------------------------------------------------------------------------
# Frontmatter parsing
# ---------------------------------------------------------------------------

def _parse_frontmatter(content: str):
    """Returns (frontmatter_dict_or_None, body_str, error_message_or_None)."""
    if not content.strip():
        return None, "", "file is empty"
    if not content.startswith("---"):
        return None, content, "missing opening '---' delimiter"
    end = re.search(r"\n---\s*\n", content[3:])
    if not end:
        return None, content, "missing closing '---' delimiter"
    yaml_text = content[3 : end.start() + 3]
    body = content[end.end() + 3 :]
    try:
        fm = yaml.safe_load(yaml_text)
    except yaml.YAMLError as e:
        return None, body, f"YAML parse error: {e}"
    if not isinstance(fm, dict):
        return None, body, "frontmatter is not a YAML mapping"
    return fm, body, None


# ---------------------------------------------------------------------------
# Core validation
# ---------------------------------------------------------------------------

def validate_skill(
    skill_path: Path,
    ruleset: Dict[str, str],
    ruleset_name: str = "custom",
    valid_names: Optional[Set[str]] = None,
) -> ValidationReport:
    """Validate a single skill against the supplied ruleset.

    `skill_path` may point at a directory containing SKILL.md or directly at
    the SKILL.md file.

    ``valid_names`` is an optional set of known skill YAML names (from
    :func:`build_skill_name_index`).  When supplied, cross-reference checks
    (related_skills validity, body-level dead refs) are enabled.
    """
    skill_md = (
        skill_path / "SKILL.md" if skill_path.is_dir() else skill_path
    )
    skill_dir_name = skill_md.parent.name
    report = ValidationReport(
        skill_path=skill_md,
        skill_name=skill_dir_name,
        ruleset_name=ruleset_name,
    )

    def add(rule_id: str, message: str) -> None:
        sev = ruleset.get(rule_id)
        if sev and sev != SEVERITY_OK:
            report.findings.append(Finding(rule_id, sev, message))

    if not skill_md.exists():
        report.findings.append(
            Finding("file.exists", SEVERITY_BLOCKING, f"SKILL.md not found at {skill_md}")
        )
        return report

    content = skill_md.read_text(encoding="utf-8")
    fm, body, fm_err = _parse_frontmatter(content)

    if fm is None:
        rule = (
            "frontmatter.present"
            if fm_err and "missing" in fm_err
            else "frontmatter.parseable"
        )
        add(rule, fm_err or "frontmatter could not be parsed")
        if not body.strip():
            add("body.non_empty", "body is empty after frontmatter")
        return report

    # name.*
    name = fm.get("name")
    if not name:
        add("frontmatter.name.present", "missing 'name' field")
    else:
        name_str = str(name)
        if not KEBAB_RE.match(name_str):
            add(
                "frontmatter.name.kebab_case",
                f"name '{name_str}' is not kebab-case (expected ^[a-z][a-z0-9-]*$)",
            )
        if name_str != skill_dir_name:
            add(
                "frontmatter.name.matches_dir",
                f"name '{name_str}' does not match directory '{skill_dir_name}'",
            )

    # description.*
    desc = fm.get("description")
    if not desc:
        add("frontmatter.description.present", "missing 'description' field")
    else:
        desc_str = str(desc)
        if len(desc_str) > DEFAULT_MAX_DESCRIPTION:
            add(
                "frontmatter.description.length",
                f"description is {len(desc_str)} characters, max {DEFAULT_MAX_DESCRIPTION}",
            )

    # license / allowed-tools
    if not fm.get("license"):
        add("frontmatter.license.present", "missing 'license' field (e.g. MIT)")
    if not fm.get("allowed-tools"):
        add("frontmatter.allowed_tools.present", "missing 'allowed-tools' field")

    # metadata.*
    meta_raw = fm.get("metadata") or {}
    meta = meta_raw if isinstance(meta_raw, dict) else {}

    if not meta.get("author"):
        add("frontmatter.metadata.author.present", "missing metadata.author")
    if not meta.get("domain"):
        add("frontmatter.metadata.domain.present", "missing metadata.domain")
    if not meta.get("tags"):
        add("frontmatter.metadata.tags.present", "missing metadata.tags")

    version = meta.get("version")
    if not version:
        add("frontmatter.metadata.version.present", "missing metadata.version")
    elif not SEMVER_RE.match(str(version)):
        add(
            "frontmatter.metadata.version.semver",
            f"metadata.version '{version}' is not semver (M.m or M.m.p)",
        )

    complexity = meta.get("complexity")
    if not complexity:
        add("frontmatter.metadata.complexity.valid", "missing metadata.complexity")
    elif str(complexity) not in KNOWN_COMPLEXITIES:
        add(
            "frontmatter.metadata.complexity.valid",
            f"metadata.complexity '{complexity}' is not in "
            f"{sorted(KNOWN_COMPLEXITIES)}",
        )

    # metadata.hermes.* — Hermes-specific operational metadata
    hermes_meta_raw = _deep_get(fm, "metadata", "hermes", default={})
    hermes_meta = hermes_meta_raw if isinstance(hermes_meta_raw, dict) else {}
    if not hermes_meta:
        add(
            "frontmatter.metadata.hermes.present",
            "missing metadata.hermes section (tags, related_skills, trigger)",
        )
    else:
        if not hermes_meta.get("tags"):
            add(
                "frontmatter.metadata.hermes.present",
                "metadata.hermes is present but missing 'tags'",
            )
        trigger = hermes_meta.get("trigger", "")
        if not trigger:
            add(
                "frontmatter.metadata.hermes.trigger.valid",
                "missing metadata.hermes.trigger",
            )
        elif trigger not in TRIGGER_VALUES:
            add(
                "frontmatter.metadata.hermes.trigger.valid",
                f"metadata.hermes.trigger '{trigger}' is not one of "
                f"{sorted(TRIGGER_VALUES)}",
            )

    # related_skills.* — cross-reference integrity (requires valid_names)
    if valid_names is not None:
        rs_list = _find_related_skills(fm)
        broken = [r for r in rs_list if r not in valid_names]
        if broken:
            add(
                "frontmatter.related_skills.valid",
                f"Broken related_skills references (skill not found): {broken}",
            )

    # body.*
    if not body.strip():
        add("body.non_empty", "body is empty after frontmatter")

    line_count = len(body.splitlines())
    if line_count > DEFAULT_MAX_LINES:
        add(
            "body.line_count",
            f"body is {line_count} lines (suggested max {DEFAULT_MAX_LINES}) — "
            f"consider extracting examples or splitting compound procedures",
        )

    # body.dead_references — backtick-wrapped skill names that don't resolve
    if valid_names is not None:
        body_dead = _find_body_dead_refs(body, valid_names)
        # Deduplicate for cleaner output
        seen: Set[str] = set()
        unique_dead: List[str] = []
        for r in body_dead:
            if r not in seen:
                seen.add(r)
                unique_dead.append(r)
        if unique_dead:
            add(
                "body.dead_references",
                f"Body text references skill names not found in any SKILL.md: {unique_dead}",
            )

    # section.*
    body_lower = body.lower()
    if "## when to use" not in body_lower:
        add(
            "section.when_to_use.present",
            "missing '## When to Use' section (recommended by agentskills.io)",
        )
    if not re.search(r"^##\s+procedure\b", body, re.MULTILINE | re.IGNORECASE):
        add(
            "section.procedure.present",
            "missing '## Procedure' section (recommended by agentskills.io)",
        )

    return report


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def find_skills(skills_root: Path) -> List[Path]:
    """Return all skill directories under `skills_root` that contain a SKILL.md.

    Skips anything under an `.archive` segment so archived skills don't show
    up in audit runs.
    """
    found: List[Path] = []
    for p in skills_root.rglob("SKILL.md"):
        if any(part == ".archive" for part in p.parts):
            continue
        found.append(p.parent)
    return found
