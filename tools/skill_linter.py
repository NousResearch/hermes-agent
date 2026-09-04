"""Structural + convention linter for SKILL.md files.

The hard validator (``skill_manager_tool._validate_frontmatter``) blocks the
non-negotiables; this is the advisory companion encoding the CONTRIBUTING.md
"Skill authoring standards" a human reviewer would otherwise catch. Findings
never block by themselves — ``lint_skill`` returns ``LintFinding`` rows and the
caller decides. Frontmatter parsing is delegated to ``agent.skill_utils`` so
BOM handling and the prompt description budget stay in one place.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from agent.skill_utils import SKILL_PROMPT_DESC_LIMIT, parse_frontmatter

# Shell utilities already wrapped as native tools; naming them in prose steers
# the model to a raw shell call. banned token -> native tool to name instead.
_SHELL_UTIL_TO_TOOL: Dict[str, str] = {
    "grep": "search_files", "rg": "search_files", "cat": "read_file", "head": "read_file",
    "tail": "read_file", "sed": "patch", "awk": "patch",
    "find": "search_files (target='files')", "ls": "search_files (target='files')"}
_MARKETING_WORDS = (
    "powerful", "comprehensive", "seamless", "advanced", "cutting-edge", "state-of-the-art",
    "revolutionary", "robust")
# POSIX-only primitives that require ``platforms:`` when a bundled script uses
# them. Detected in scripts/, not in prose.
_POSIX_PRIMITIVES = (
    "fcntl",
    "termios",
    "os.setsid",  # windows-footgun: ok  (search-pattern string, not a call)
    "signal.SIGKILL",  # windows-footgun: ok  (search-pattern string, not a call)
    "osascript",
    "/proc/",
    "apt-get",
    "systemctl")
# Scaffolding files a skill should not ship (noise, not skill content).
_FORBIDDEN_FILES = ("README.md", "CHANGELOG.md", "install.sh", ".env", ".env.example", ".gitignore")
# Presence of the load-bearing section is checked, not exact ordering, so the
# linter is not a change-detector.
_EXPECTED_SECTIONS = ("When to Use", "When to use")

ERROR = "error"
WARNING = "warning"


@dataclass
class LintFinding:
    """A single lint result. ``severity`` is advisory metadata for the caller."""

    severity: str  # ERROR | WARNING
    rule: str
    message: str


def _err(rule: str, message: str) -> LintFinding:
    return LintFinding(ERROR, rule, message)


def _warn(rule: str, message: str) -> LintFinding:
    return LintFinding(WARNING, rule, message)


def _strip_code_blocks(body: str) -> str:
    """Remove fenced code blocks so prose-only checks don't fire on examples."""
    return re.sub(r"```.*?```", "", body, flags=re.S)


def _check_frontmatter(frontmatter: Dict[str, Any], skill_dir: Optional[Path]) -> Iterator[LintFinding]:
    name = str(frontmatter.get("name", "")).strip()
    if name and not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", name):
        yield _err("name-format", f"name '{name}' must be lowercase letters, digits, hyphens, "
                   f"and underscores only.")
    if skill_dir is not None and name and name != skill_dir.name:
        yield _err("name-dir-mismatch", f"frontmatter name '{name}' does not match directory "
                   f"'{skill_dir.name}'; they must be identical.")
    # Measure the raw authored value: extract_skill_description() already
    # truncates to the prompt budget, so it can never exceed the limit.
    desc = str(frontmatter.get("description", "")).strip().strip("'\"")
    if len(desc) > SKILL_PROMPT_DESC_LIMIT:
        yield _warn("description-length",
                    f"description is {len(desc)} chars; the skill index truncates past "
                    f"{SKILL_PROMPT_DESC_LIMIT} chars + '...', losing routing "
                    f"signal. Keep it to one sentence.")
    hits = [w for w in _MARKETING_WORDS if re.search(rf"\b{re.escape(w)}\b", desc.lower())]
    if hits:
        yield _warn("description-marketing",
                    f"description contains marketing words {hits}; state the capability, not adjectives.")
    for key in ("version", "author", "license"):
        if key not in frontmatter:
            yield _warn("missing-metadata", f"frontmatter is missing '{key}'; every peer skill has it.")
    meta = frontmatter.get("metadata")
    hermes_meta = meta.get("hermes") if isinstance(meta, dict) else None
    if not isinstance(hermes_meta, dict):
        yield _warn("missing-metadata", "frontmatter is missing metadata.hermes.{tags, related_skills}.")
    elif "tags" not in hermes_meta:
        yield _warn("missing-metadata", "metadata.hermes.tags is missing.")
    author = str(frontmatter.get("author", ""))
    if author and author.strip().lower() in ("hermes", "agent", "hermes agent") and (
        author != "Hermes Agent"):
        yield _warn("author-caps", f"author '{author}' should be 'Hermes Agent' (proper caps) "
                    f"or a real contributor name.")
    platforms = frontmatter.get("platforms")
    if platforms:
        valid = {"linux", "macos", "windows", "darwin"}
        items = platforms if isinstance(platforms, list) else [platforms]
        bad = [p for p in items if str(p).lower() not in valid]
        if bad:
            yield _warn("platforms-value", f"platforms contains unrecognized value(s) {bad}; "
                        f"expected a subset of {sorted(valid)}.")


def _check_body(body: str, skill_dir: Optional[Path]) -> Iterator[LintFinding]:
    # Only backtick-wrapped mentions in PROSE (not fenced code): bare words are too noisy.
    prose = _strip_code_blocks(body)
    for util, tool in _SHELL_UTIL_TO_TOOL.items():
        if re.search(rf"`{re.escape(util)}`", prose):
            yield _warn("shell-utility-reference",
                        f"prose references `{util}`; name the native tool `{tool}` instead.")
    if not any(re.search(rf"^#+\s+{re.escape(s)}", body, re.M) for s in _EXPECTED_SECTIONS):
        yield _warn("missing-section", "no '## When to Use' section found; skills need explicit "
                    "trigger conditions near the top.")
    if skill_dir is None:
        return
    # Dangling links. Only references/, templates/, assets/ are reliably skill-owned;
    # `scripts/` is excluded because dev skills legitimately cite repo-root scripts.
    seen: set[str] = set()
    for match in re.finditer(r"(references|templates|assets)/[\w./-]+", body):
        rel = match.group(0)
        if rel in seen or "*" in rel or rel.endswith("/"):  # dupes, placeholders, globs
            continue
        seen.add(rel)
        if not (skill_dir / rel).exists():
            yield _warn("dangling-reference", f"body references '{rel}' but that file "
                        f"does not exist in the skill directory.")


def _check_files(frontmatter: Dict[str, Any], skill_dir: Path) -> Iterator[LintFinding]:
    # Bundled scripts using POSIX-only primitives require a platforms: declaration.
    scripts_dir = skill_dir / "scripts"
    offenders: Dict[str, List[str]] = {}
    if not frontmatter.get("platforms") and scripts_dir.is_dir():
        for script in scripts_dir.rglob("*"):
            if not script.is_file() or script.suffix not in (".py", ".sh", ".bash"):
                continue
            try:
                text = script.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            hit = [p for p in _POSIX_PRIMITIVES if p in text]
            if hit:
                offenders[script.name] = hit
    if offenders:
        detail = "; ".join(f"{k}: {v}" for k, v in offenders.items())
        yield _warn("platforms-gating",
                    f"scripts use POSIX-only primitives ({detail}) but no 'platforms:' frontmatter is "
                    f"declared. Fix cross-platform or gate with platforms: [linux, macos].")
    for fname in _FORBIDDEN_FILES:
        if (skill_dir / fname).exists():
            yield _warn("forbidden-file",
                        f"skill ships '{fname}'; skills should not include scaffolding/config files.")


def lint_content(content: str, *, skill_dir: Optional[Path] = None) -> List[LintFinding]:
    """Lint raw SKILL.md *content*.

    ``skill_dir`` enables on-disk checks (name/dir match, dangling links, POSIX
    gating, forbidden files); without it only content checks run, which is what
    the create path needs before the file exists.
    """
    frontmatter, body = parse_frontmatter(content)
    findings = list(_check_frontmatter(frontmatter, skill_dir)) + list(_check_body(body, skill_dir))
    if skill_dir is not None:
        findings += _check_files(frontmatter, skill_dir)
    return findings


def lint_skill(skill_md_path: Path) -> List[LintFinding]:
    """Lint a SKILL.md file on disk, with all on-disk checks enabled."""
    skill_md_path = Path(skill_md_path)
    content = skill_md_path.read_text(encoding="utf-8", errors="ignore")
    return lint_content(content, skill_dir=skill_md_path.parent)


# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# Names external plugins imported from this module before the Sep 2026 decomposition.
# Internal code MUST NOT use these (scripts/check_compat_pointers.py fails CI if it does).
# The whole block is removed by reverting the commit that added it.

def format_findings(findings: List[LintFinding]) -> str:
    """Render findings as a newline-joined human-readable block."""
    return "\n".join(f.format() for f in findings)

def has_errors(findings: List[LintFinding]) -> bool:
    return any(f.severity == ERROR for f in findings)
# ---- END PLUGIN-COMPAT ----
