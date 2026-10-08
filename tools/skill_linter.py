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
from urllib.parse import unquote

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
# incident-log-shape: at least this many PR/issue refs AND this density (per 1k chars of prose).
_INCIDENT_REF_MIN = 4
_INCIDENT_REF_PER_KCHAR = 0.5  # the 100k incident-log SKILL.md this targets sat at ~0.7
# references-sprawl: a skill carrying more reference files than this is hoarding per-session notes.
# Calibration: a deliberately curated large workflow skill sits near 50 topical files; the hoarding
# shape this catches was 443 one-per-session files.
_MAX_REFERENCE_FILES = 60
# oversized-body: SKILL.md is loaded whole by skill_view and then rides in context for every later
# call of the session, so body size is paid per turn, not once. The authoring standard is ~200 lines;
# this budget is ~3x that (bundled skills average ~20k chars). The hard cap in skill_manager_tool
# (100k) is a safety stop, not a target — agent-authored skills grew to sit right under it.
_BODY_SOFT_BUDGET_CHARS = 24_000

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
    if len(body) > _BODY_SOFT_BUDGET_CHARS:
        yield _warn("oversized-body",
                    f"SKILL.md body is {len(body):,} chars (~{len(body) // 4:,} tokens); skill_view loads "
                    f"all of it and it stays in context for every later call of the session. Keep the "
                    f"always-on rules here (~200 lines) and move topic depth into references/<topic>.md, "
                    f"linked from the body.")
    # Only backtick-wrapped mentions in PROSE (not fenced code): bare words are too noisy.
    prose = _strip_code_blocks(body)
    for util, tool in _SHELL_UTIL_TO_TOOL.items():
        if re.search(rf"`{re.escape(util)}`", prose):
            yield _warn("shell-utility-reference",
                        f"prose references `{util}`; name the native tool `{tool}` instead.")
    if not any(re.search(rf"^#+\s+{re.escape(s)}", body, re.M) for s in _EXPECTED_SECTIONS):
        yield _warn("missing-section", "no '## When to Use' section found; skills need explicit "
                    "trigger conditions near the top.")
    # Incident-log shape: a skill body dense in PR/issue numbers is narrating history instead of
    # stating rules. Threshold is per 1k chars so a long body with one citation is fine.
    refs = len(re.findall(r"(?<![\w/])#\d{3,6}\b|\b(?:PR|issue)\s*#?\d{3,6}\b", prose))
    if refs >= _INCIDENT_REF_MIN and refs / max(len(prose), 1) * 1000 >= _INCIDENT_REF_PER_KCHAR:
        yield _warn("incident-log-shape", f"{refs} PR/issue references in prose; write the generalizable "
                    "rule + why and drop the incident numbers — the rule must stand without the story.")
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
                text = script.read_text(encoding="utf-8-sig", errors="ignore")
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
    refs_dir = skill_dir / "references"
    if refs_dir.is_dir():
        n_refs = sum(1 for p in refs_dir.rglob("*.md") if not any(part.startswith("_") for part in p.parts))
        if n_refs > _MAX_REFERENCE_FILES:
            yield _warn("references-sprawl",
                        f"{n_refs} files under references/; that is a per-session log, not topical depth. "
                        "Merge same-topic files into one rule set and drop incident narration.")



# Keep navigation visible to readers that only load the first screen of a support file.
_REFERENCE_TOC_LINE_LIMIT = 100
_REFERENCE_TOC_PREFIX_LINES = 40
_CONTENTS_HEADING = re.compile(
    r"^\s{0,3}#{1,6}\s+(?:table\s+of\s+)?contents\s*:?[\s#]*$", re.I)
_CONTENTS_ITEM = re.compile(
    r"^\s*(?:[-+*]|\d+[.)])\s+(?:\[[ xX]\]\s+)?\[[^\]]+\]\(#[^)]+\)")
_MARKDOWN_LINK = re.compile(r"\[[^\]]*\]\(\s*(?:<([^>]+)>|([^\s)]+))")
_SUPPORT_PATH = re.compile(r"(?:references|templates|assets|scripts|examples)/[\w./-]+\.md")


def _has_reference_toc(lines: List[str]) -> bool:
    fence = None
    items = 0
    for line in lines[:_REFERENCE_TOC_PREFIX_LINES]:
        stripped = line.lstrip()
        marker = re.match(r"(`{3,}|~{3,})", stripped)
        if marker:
            token = marker.group(0)
            if fence is None:
                fence = token
            elif token[0] == fence[0] and len(token) >= len(fence):
                fence = None
            continue
        if fence is not None:
            continue
        if _CONTENTS_HEADING.match(line):
            return True
        if _CONTENTS_ITEM.match(line):
            items += 1
    return items >= 2


def _linked_support_files(text: str, base: Path, files: Dict[Path, str]) -> set[Path]:
    targets = [m.group(1) or m.group(2) for m in _MARKDOWN_LINK.finditer(text)]
    targets += re.findall(r"`([^`\n]+\.md(?:#[^`\n]*)?)`", text)
    targets += _SUPPORT_PATH.findall(text)
    linked = set()
    for target in targets:
        target = unquote(target.split("#", 1)[0].split("?", 1)[0])
        candidate = (base / target).resolve()
        if candidate in files:
            linked.add(candidate)
    return linked


def _check_reference_docs(body: str, skill_dir: Path) -> Iterator[LintFinding]:
    root = skill_dir.resolve()
    files: Dict[Path, str] = {}
    contents: Dict[Path, str] = {}
    for path in sorted(skill_dir.rglob("*.md")):
        rel = path.relative_to(skill_dir)
        if rel.as_posix() == "SKILL.md" or any(p.startswith(("_", ".")) for p in rel.parts):
            continue
        resolved = path.resolve()
        if not resolved.is_relative_to(root) or not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8-sig", errors="ignore")
        except OSError:
            continue
        files[resolved] = rel.as_posix()
        contents[resolved] = text
        lines = text.splitlines()
        if len(lines) > _REFERENCE_TOC_LINE_LIMIT and not _has_reference_toc(lines):
            yield _warn("reference-toc", f"'{rel.as_posix()}' has {len(lines)} lines; add a Contents "
                        f"heading or anchor list in the first {_REFERENCE_TOC_PREFIX_LINES} lines "
                        "so a partial read can find the remaining sections.")
    direct = _linked_support_files(body, root, files)
    reached = set()
    pending = list(direct)
    while pending:
        current = pending.pop()
        if current in reached:
            continue
        reached.add(current)
        pending.extend(_linked_support_files(contents[current], current.parent, files) - reached)
    for path in sorted(reached - direct):
        yield _warn("reference-depth", f"'{files[path]}' is only linked through another support file; "
                    "link it directly from SKILL.md so readers can discover it in one step.")


def lint_content(content: str, *, skill_dir: Optional[Path] = None) -> List[LintFinding]:
    """Lint raw SKILL.md *content*.

    ``skill_dir`` enables on-disk checks (name/dir match, dangling links, POSIX
    gating, forbidden files, reference navigation); without it only content checks run, which is what
    the create path needs before the file exists.
    """
    frontmatter, body = parse_frontmatter(content)
    findings = list(_check_frontmatter(frontmatter, skill_dir)) + list(_check_body(body, skill_dir))
    if skill_dir is not None:
        findings += _check_files(frontmatter, skill_dir)
        findings += _check_reference_docs(body, skill_dir)
    return findings


def lint_skill(skill_md_path: Path) -> List[LintFinding]:
    """Lint a SKILL.md file on disk, with all on-disk checks enabled."""
    skill_md_path = Path(skill_md_path)
    content = skill_md_path.read_text(encoding="utf-8-sig", errors="ignore")
    return lint_content(content, skill_dir=skill_md_path.parent)
