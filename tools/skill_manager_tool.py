#!/usr/bin/env python3
"""Skill Manager Tool — agent-managed skill creation & editing.

Skills are the agent's procedural memory (narrow "how to do X"; MEMORY.md/USER.md are
broad, declarative). New skills land in ~/.hermes/skills/ (or ``skills.create_dir``);
existing skills (bundled, hub, user) are modified in place. Layout:
``<skills>/[category/]<skill>/SKILL.md`` + optional ``references/ templates/ scripts/ assets/``.
"""

import contextvars as _ctxvars
import json
from contextlib import suppress
import logging
import re
import shutil
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

from hermes_constants import get_hermes_home, display_hermes_home
from utils import atomic_write_text, is_truthy_value
from hermes_cli.config import cfg_get
from agent.skill_utils import (
    extract_skill_description,
    is_skill_description_truncated_for_prompt,
    parse_frontmatter as _parse_frontmatter,
    SKILL_PROMPT_DESC_LIMIT)
from tools.skill_manager_guards import (
    _background_review_preflight, _background_review_read_before_write_guard, _background_review_write_guard,
    _containing_skills_root, _curator_consolidation_delete_guard, _maybe_auto_propose_org_edit,
    _org_mirror_write_guard, _pinned_guard, _validate_delete_target, _is_background_review, _refusal as _err)
from tools.skill_manager_batch import _skill_manage_batch
from tools.skills_guard import scan_skill, should_allow_install, format_scan_report

logger = logging.getLogger(__name__)


def _guard_agent_created_enabled() -> bool:
    """skills.guard_agent_created (default False): opt-in — terminal() runs the same code ungated."""
    try:
        from hermes_cli.config import load_config
        return is_truthy_value(cfg_get(load_config(), "skills", "guard_agent_created"), default=False)
    except Exception:
        return False


def _security_scan_skill(skill_dir: Path) -> Optional[str]:
    """Post-write scan (opt-in); error string if blocked, else None. An "ask" verdict
    (dangerous findings) is surfaced as an error so the agent can retry without them."""
    if not _guard_agent_created_enabled():
        return None
    try:
        result = scan_skill(skill_dir, source="agent-created")
        allowed, reason = should_allow_install(result)
        if allowed is None:
            logger.warning("Agent-created skill blocked (dangerous findings): %s", reason)
        if allowed is not True:
            return f"Security scan blocked this skill ({reason}):\n{format_scan_report(result)}"
    except Exception as e:
        logger.warning("Security scan failed for %s: %s", skill_dir, e, exc_info=True)
    return None


# All skills live in ~/.hermes/skills/ (single source of truth)
HERMES_HOME = get_hermes_home()
SKILLS_DIR = HERMES_HOME / "skills"
_SKILLS_DIR_AT_IMPORT = SKILLS_DIR


def _skills_dir() -> Path:
    """Active profile's skills dir at call time (multi-profile runtimes rebind per session).
    An explicitly patched module-level ``SKILLS_DIR`` (tests) wins over the live HERMES_HOME.

    Long-lived multi-profile runtimes (Dashboard/TUI/Desktop backend, cron, kanban workers) import this
    module once under the launch HERMES_HOME and later bind a different profile per session (#40677).
    """
    configured = Path(SKILLS_DIR)
    return configured if configured != _SKILLS_DIR_AT_IMPORT else get_hermes_home() / "skills"


MAX_NAME_LENGTH = 64
MAX_DESCRIPTION_LENGTH = 1024
MAX_SKILL_CONTENT_CHARS = 100_000   # ~36k tokens at 2.75 chars/token
MAX_SKILL_FILE_BYTES = 1_048_576    # 1 MiB per supporting file
VALID_NAME_RE = re.compile(r'^[a-z0-9][a-z0-9._-]*$')  # filesystem-safe, URL-friendly
ALLOWED_SUBDIRS = {"references", "templates", "scripts", "assets"}  # for write_file/remove_file
_FRONTMATTER_END_RE = re.compile(r'\n---\s*\n')
_NAME_RULE = "Use lowercase letters, numbers, hyphens, dots, and underscores."


def _display_create_dir() -> str:
    """Skill-creation dir for schema/instruction text; follows ``skills.create_dir``."""
    try:
        from agent.skill_utils import display_skill_create_dir
        return display_skill_create_dir()
    except Exception:
        return f"{display_hermes_home()}/skills/"


# --- Validation helpers -------------------------------------------------------

def _check_identifier(value: str, label: str, invalid: str) -> Optional[str]:
    if len(value) > MAX_NAME_LENGTH:
        return f"{label} exceeds {MAX_NAME_LENGTH} characters."
    return None if VALID_NAME_RE.match(value) else invalid


def _validate_name(name: str) -> Optional[str]:
    if not name:
        return "Skill name is required."
    return _check_identifier(
        name, "Skill name", f"Invalid skill name '{name}'. {_NAME_RULE} Must start with a letter or digit.")


def _validate_category(category: Optional[str]) -> Optional[str]:
    if category is None or (isinstance(category, str) and not category.strip()):
        return None
    if not isinstance(category, str):
        return "Category must be a string."
    category = category.strip()
    invalid = (f"Invalid category '{category}'. {_NAME_RULE} "
               "Categories must be a single directory name.")
    if "/" in category or "\\" in category:
        return invalid
    return _check_identifier(category, "Category", invalid)


def _validate_frontmatter(content: str, *, new_skill: bool = False) -> Optional[str]:
    """Validate frontmatter (name + description) and a non-empty body. ``new_skill`` (create
    only) also enforces SKILL_PROMPT_DESC_LIMIT so new skills never lose routing signal to
    index truncation; edit/patch skip it so existing over-limit skills stay maintainable."""
    if not content.strip():
        return "Content cannot be empty."
    content = content.lstrip("\ufeff")  # tolerate a Windows UTF-8 BOM
    if not content.startswith("---"):
        return "SKILL.md must start with YAML frontmatter (---). See existing skills for format."
    end_match = _FRONTMATTER_END_RE.search(content[3:])
    if not end_match:
        return "SKILL.md frontmatter is not closed. Ensure you have a closing '---' line."
    try:
        parsed = yaml.safe_load(content[3:end_match.start() + 3])
    except yaml.YAMLError as e:
        return f"YAML frontmatter parse error: {e}"
    if not isinstance(parsed, dict):
        return "Frontmatter must be a YAML mapping (key: value pairs)."
    for field in ("name", "description"):
        if field not in parsed:
            return f"Frontmatter must include '{field}' field."
    desc = str(parsed["description"])
    if len(desc) > MAX_DESCRIPTION_LENGTH:
        return f"Description exceeds {MAX_DESCRIPTION_LENGTH} characters."
    if new_skill and len(desc.strip().strip("'\"")) > SKILL_PROMPT_DESC_LIMIT:
        return (
            f"Description is {len(desc.strip())} chars — new skills must fit the "
            f"{SKILL_PROMPT_DESC_LIMIT}-char system-prompt budget (one sentence, trigger first, "
            f"ends with a period). The skill index truncates longer descriptions to "
            f"{SKILL_PROMPT_DESC_LIMIT - 3} chars + '...', destroying the routing signal. "
            f"Move detail into the skill body.")
    if not content[end_match.end() + 3:].strip():
        return "SKILL.md must have content after the frontmatter (instructions, procedures, etc.)."
    return None


def _validate_content_size(content: str, label: str = "SKILL.md") -> Optional[str]:
    if len(content) > MAX_SKILL_CONTENT_CHARS:
        return (
            f"{label} content is {len(content):,} characters (limit: {MAX_SKILL_CONTENT_CHARS:,}). "
            f"Consider splitting into a smaller SKILL.md with supporting files in references/ "
            f"or templates/.")
    return None


def _description_preview(content: str) -> str:
    """First 120 chars of the frontmatter description; '' on any failure."""
    with suppress(Exception):
        fm_end = _FRONTMATTER_END_RE.search(content[3:])
        if fm_end:
            return str(yaml.safe_load(content[3:fm_end.start() + 3]).get("description", ""))[:120]
    return ""


def _resolve_skill_dir(name: str, category: str = None) -> Path:
    """New-skill dir; honors ``skills.create_dir`` (e.g. a shared fleet dir)."""
    base = _skills_dir()
    try:
        from agent.skill_utils import get_skill_create_dir
        base = get_skill_create_dir() or base
    except Exception:
        logger.debug("skills.create_dir lookup failed", exc_info=True)
    return base / (category or "") / name


def _iter_skill_dirs(root: Path):
    from agent.skill_utils import is_excluded_skill_path
    for skill_md in root.rglob("SKILL.md"):
        if not is_excluded_skill_path(skill_md):
            yield skill_md.parent


def _find_skill(name: str) -> Optional[Dict[str, Any]]:
    """Find a skill (local skills dir, then skills.external_dirs) -> ``{"path": Path}`` | None.

    Accepts the bare dir name (``axolotl``; matches category-nested skills too) and the
    categorized relative path (``mlops/axolotl``) — the two forms skill_view resolves. The
    categorized form matches RELATIVE to the local root only (relative_to raises for external dirs)."""
    from agent.skill_utils import get_all_skills_dirs
    local_root = None
    if "/" in name or "\\" in name:
        try:
            local_root = _skills_dir().resolve()
        except OSError:
            logger.debug(
                "skills dir resolve failed; categorized lookups fall back to the unresolved path",
                exc_info=True)
            local_root = _skills_dir()
    for skills_dir in get_all_skills_dirs():
        if not skills_dir.exists():
            continue
        for skill_dir in _iter_skill_dirs(skills_dir):
            if skill_dir.name == name:
                return {"path": skill_dir}
            if local_root is not None:
                resolved = skill_dir.resolve()
                if (resolved.is_relative_to(local_root)
                        and resolved.relative_to(local_root).as_posix() == name):  # POSIX form
                    return {"path": skill_dir}
    return None


def _find_skill_in_other_profiles(name: str) -> List[Tuple[str, Path]]:
    """``(profile, skill_dir)`` pairs for OTHER profiles holding ``name`` (so the not-found
    error can explain a wrong-profile mistake). Fail-quiet."""
    matches: List[Tuple[str, Path]] = []
    try:
        from hermes_constants import get_default_hermes_root
        root = get_default_hermes_root()
    except Exception:
        return matches
    _active = _skills_dir()
    active_dir = _active.resolve() if _active.exists() else _active
    # Every profile's skills dir EXCEPT the active one (already searched). A candidate whose
    # path cannot be resolved is skipped (not a fatal error); is_dir() checks stay unguarded.
    candidates: List[Tuple[str, Path]] = []
    with suppress(OSError, RuntimeError):
        if (root / "skills").resolve() != active_dir:
            candidates.append(("default", root / "skills"))
    if (root / "profiles").is_dir():
        with suppress(OSError):
            for entry in (root / "profiles").iterdir():
                if not entry.is_dir():
                    continue
                try:
                    if (entry / "skills").resolve() == active_dir:
                        continue
                except (OSError, RuntimeError):
                    continue
                candidates.append((entry.name, entry / "skills"))
    for profile_name, skills_dir in candidates:
        if not skills_dir.is_dir():
            continue
        with suppress(OSError):
            hit = next((d for d in _iter_skill_dirs(skills_dir) if d.name == name), None)
            if hit is not None:
                matches.append((profile_name, hit))  # one match per profile is enough
    return matches


def _skill_not_found_error(name: str, suffix: str = "") -> str:
    """Not-found error naming other profiles that hold the skill, plus ``suffix``."""
    from agent.file_safety import _resolve_active_profile_name
    base = f"Skill '{name}' not found in active profile '{_resolve_active_profile_name()}'."
    others = _find_skill_in_other_profiles(name)
    if len(others) == 1:
        other_profile, other_path = others[0]
        base += (
            f" A skill by that name exists in profile '{other_profile}' ({other_path}). To edit "
            f"it, switch profiles (`hermes -p {other_profile}`) or edit the file directly "
            f"(file tools / terminal).")
    elif others:
        names = ", ".join(f"'{p}'" for p, _ in others)
        base += (
            f" Skills by that name exist in other profiles: {names}. Switch profiles (`hermes -p "
            f"<name>`) to edit there, or edit the files directly (file tools / terminal).")
    else:
        base += " Use skills_list() to see available skills."
    return base + suffix


def _validate_file_path(file_path: str) -> Optional[str]:
    """Validate a write_file/remove_file path: under an allowed subdir, no escape."""
    from tools.path_security import has_traversal_component
    if not file_path:
        return "file_path is required."
    parts = Path(file_path).parts
    # Traversal first, so the SKILL.md exception is unreachable by a traversal-laden path.
    if has_traversal_component(file_path):
        return "Path traversal ('..') is not allowed."
    # SKILL.md lives at the skill root; accept 'SKILL.md' and '<skill>/SKILL.md'.
    if parts and parts[-1] == "SKILL.md" and len(parts) in (1, 2):
        return None
    if not parts or parts[0] not in ALLOWED_SUBDIRS:
        allowed = ", ".join(sorted(ALLOWED_SUBDIRS))
        return f"File must be under one of: {allowed}. Got: '{file_path}'"
    if len(parts) < 2:
        return f"Provide a file path, not just a directory. Example: '{parts[0]}/myfile.md'"
    return None


def _resolve_supporting_file(skill_dir: Path, file_path: str):
    """Validate ``file_path`` and resolve it inside ``skill_dir``
    -> ``(target, None)`` | ``(None, error_dict)``."""
    from tools.path_security import validate_within_dir
    target = skill_dir / (file_path or "")
    err = _validate_file_path(file_path) or validate_within_dir(target, skill_dir)
    return (None, _err(err)) if err else (target, None)


def _locate_for_write(name: str, action: str, not_found_suffix: str = "", *,
                      org_guard: bool = True):
    """Find the skill; run the org-mirror (unless ``org_guard=False``) and background-review
    write guards -> ``(skill_dir, None)`` | ``(None, error_dict)``."""
    existing = _find_skill(name)
    if not existing:
        return None, _err(_skill_not_found_error(name, not_found_suffix))
    skill_dir = existing["path"]
    guard = ((org_guard and _org_mirror_write_guard(name, skill_dir, action))
             or _background_review_write_guard(name, skill_dir, action))
    return (None, guard) if guard else (skill_dir, None)


def _guarded_write(name: str, skill_dir: Path, target: Path, action: str, label: str,
                   content: str) -> Optional[Dict[str, Any]]:
    """Read-before-write guard (existing targets only), atomic write, then the security scan;
    a blocked scan restores the original (or unlinks a new file). Error dict or None."""
    original = None
    if target.exists():
        if read_guard := _background_review_read_before_write_guard(name, target, action, label):
            return read_guard
        original = target.read_text(encoding="utf-8")
    target.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(target, content, preserve_mode=True, create_mode=0o644)
    scan_error = _security_scan_skill(skill_dir)
    if not scan_error:
        return None
    if original is not None:
        atomic_write_text(target, original, preserve_mode=True)
    else:
        target.unlink(missing_ok=True)
    return _err(scan_error)


def _attach_org_note(result: Dict[str, Any], name: str, skill_dir: Path) -> Dict[str, Any]:
    if org_note := _maybe_auto_propose_org_edit(name, skill_dir):
        result["org_sharing"] = org_note
        result["message"] = f"{result['message']} {org_note}"
    return result


def _add_description_prompt_preview(result: Dict[str, Any], content: str) -> Dict[str, Any]:
    fm, _ = _parse_frontmatter(content)
    if is_skill_description_truncated_for_prompt(fm):
        result["system_prompt_preview"] = (
            f"System prompt will show: \"{extract_skill_description(fm)}\" — keep the trigger "
            f"self-contained in the first {SKILL_PROMPT_DESC_LIMIT - 3} chars.")
    return result


def _attach_lint_findings(result: Dict[str, Any], skill_md: Path) -> None:
    """Attach ADVISORY authoring findings (hard rejects already ran in _validate_frontmatter)."""
    try:
        from tools.skill_linter import lint_skill  # local import: optional path
        findings = lint_skill(skill_md)
    except Exception:
        findings = None
    if not findings:
        return
    result["lint_warnings"] = [
        {"severity": f.severity, "rule": f.rule, "message": f.message} for f in findings]
    result["lint_hint"] = (
        "The skill was created. These are advisory authoring-convention findings (not blockers) "
        "— fix them with skill_manage(action='patch') to match Hermes skill standards.")


def _clip(text: str, n: int, ellipsis: str) -> str:
    return text[:n] + (ellipsis if len(text) > n else "")


# --- Core actions -------------------------------------------------------------

def _create_skill(name: str, content: str, category: str = None) -> Dict[str, Any]:
    if err := (_validate_name(name) or _validate_category(category)
               or _validate_frontmatter(content, new_skill=True) or _validate_content_size(content)):
        return _err(err)
    if existing := _find_skill(name):
        return _err(f"A skill named '{name}' already exists at {existing['path']}.")
    skill_dir = _resolve_skill_dir(name, category)
    skill_dir.mkdir(parents=True, exist_ok=True)
    skill_md = skill_dir / "SKILL.md"
    atomic_write_text(skill_md, content, preserve_mode=True, create_mode=0o644)
    if scan_error := _security_scan_skill(skill_dir):
        shutil.rmtree(skill_dir, ignore_errors=True)
        return _err(scan_error)
    root = _skills_dir()  # display relative under the profile dir; absolute under skills.create_dir
    display = skill_dir.relative_to(root) if skill_dir.is_relative_to(root) else skill_dir
    result = {
        "success": True, "message": f"Skill '{name}' created.", "path": str(display),
        "skill_md": str(skill_md), "_change": {"description": _description_preview(content)},
        **({"category": category} if category else {}),
        "hint": "To add reference files, templates, or scripts, use "
                f"skill_manage(action='write_file', name='{name}', file_path='references/example.md', "
                "file_content='...')"}
    _attach_lint_findings(_add_description_prompt_preview(result, content), skill_md)
    return result


def _edit_skill(name: str, content: str) -> Dict[str, Any]:
    """Replace the SKILL.md of any existing skill (full rewrite)."""
    if err := _validate_frontmatter(content) or _validate_content_size(content):
        return _err(err)
    skill_dir, guard = _locate_for_write(name, "edit")
    # SKILL.md always exists here (_find_skill requires it), so a blocked scan restores it.
    if guard := guard or _guarded_write(name, skill_dir, skill_dir / "SKILL.md", "edit", "SKILL.md", content):
        return guard
    result = {
        "success": True, "message": f"Skill '{name}' updated (full rewrite).",
        "path": str(skill_dir), "_change": {"description": _description_preview(content)}}
    return _add_description_prompt_preview(_attach_org_note(result, name, skill_dir), content)


def _patch_skill(name: str, old_string: str, new_string: str, file_path: str = None,
                 replace_all: bool = False) -> Dict[str, Any]:
    """Targeted find-and-replace in SKILL.md (default) or a supporting file; unique match unless replace_all."""
    if not old_string:
        # A bare "required" error is a dead end: the model retries blindly and often
        # escapes to action='write_file', clobbering the whole file.
        return _err(
            "old_string is required for 'patch' and must be the EXACT text currently in the file. "
            "Read the target file first (read_file on the skill's SKILL.md, or the file named by "
            "file_path) and copy the snippet verbatim, then retry 'patch'. Do NOT fall back to "
            "action='write_file' — that rewrites the entire file and destroys unrelated content.")
    if new_string is None:
        return _err("new_string is required for 'patch'. Use an empty string to delete matched text.")
    # No old_string == new_string guard here: fuzzy_find_and_replace rejects that with a
    # richer error (file_preview) this layer cannot produce.
    skill_dir, guard = _locate_for_write(name, "patch")
    if guard:
        return guard
    target_label = file_path or "SKILL.md"
    if file_path:
        target, err = _resolve_supporting_file(skill_dir, file_path)
        if err:
            return err
    else:
        target = skill_dir / "SKILL.md"
    if not target.exists():
        return _err(f"File not found: {target.relative_to(skill_dir)}")
    if read_guard := _background_review_read_before_write_guard(name, target, "patch", target_label):
        return read_guard
    content = target.read_text(encoding="utf-8")
    # Same fuzzy engine as the file patch tool (whitespace/indent/escape normalization,
    # block anchors) so minor formatting mismatches don't fail.
    from tools.fuzzy_match import fuzzy_find_and_replace
    new_content, match_count, _strategy, match_error = fuzzy_find_and_replace(
        content, old_string, new_string, replace_all)
    if match_error:
        with suppress(Exception):
            from tools.fuzzy_match import format_no_match_hint
            match_error += format_no_match_hint(match_error, match_count, old_string, content)
        return _err(match_error) | {"file_preview": _clip(content, 500, "...")}
    if err := _validate_content_size(new_content, label=target_label):
        return _err(err)
    if not file_path and (err := _validate_frontmatter(new_content)):
        return _err(f"Patch would break SKILL.md structure: {err}")
    if guard := _guarded_write(name, skill_dir, target, "patch", target_label, new_content):
        return guard
    result = {
        "success": True,
        "message": f"Patched {target_label} in skill '{name}' ({match_count} replacement{'s' if match_count > 1 else ''}).",
        "_change": {"old": _clip(old_string, 200, "…"), "new": _clip(new_string, 200, "…")}}
    return _attach_org_note(result, name, skill_dir)


def _delete_skill(name: str, absorbed_into: Optional[str] = None) -> Dict[str, Any]:
    """Delete a skill. ``absorbed_into``: None = undeclared (legacy, accepted); "" = explicit prune;
    "<skill>" = absorbed into that umbrella, which must exist (so the model can't claim one)."""
    skill_dir, guard = _locate_for_write(name, "delete")
    if guard := guard or _curator_consolidation_delete_guard(name, absorbed_into):
        return guard
    if pinned_err := _pinned_guard(name):
        return _err(pinned_err)
    absorbed_target = absorbed_into.strip() if isinstance(absorbed_into, str) else ""
    if absorbed_target:
        if absorbed_target == name:
            return _err(f"absorbed_into='{absorbed_target}' cannot equal the skill being deleted.")
        if not _find_skill(absorbed_target):
            return _err(f"absorbed_into='{absorbed_target}' does not exist. "
                        f"Create or patch the umbrella skill first, then retry the delete.")
    skills_root = _containing_skills_root(skill_dir)
    if unsafe := _validate_delete_target(skill_dir):  # defense-in-depth before rmtree
        return _err(unsafe)
    # Curator consolidations must be RECOVERABLE (`hermes curator restore`): archive instead
    # of rmtree. Foreground deletes keep hard-delete semantics.
    absorbed_note = f" Content absorbed into '{absorbed_target}'." if absorbed_target else ""
    if _is_background_review():
        try:
            from tools.skill_usage import archive_skill
            ok, archive_msg = archive_skill(name)
        except Exception as e:
            return _err(f"failed to archive '{name}': {e}")
        if not ok:
            return _err(archive_msg)
        return {"success": True,
                "message": f"Skill '{name}' archived ({archive_msg}).{absorbed_note}",
                "_archived": True}
    shutil.rmtree(skill_dir)
    _rmdir_if_empty(skill_dir.parent, skills_root)  # empty category dir, never the root
    return {"success": True, "message": f"Skill '{name}' deleted.{absorbed_note}"}


def _rmdir_if_empty(parent: Path, stop: Path) -> None:
    if parent != stop and parent.exists() and not any(parent.iterdir()):
        parent.rmdir()


def _write_file(name: str, file_path: str, file_content: str) -> Dict[str, Any]:
    """Add or overwrite a supporting file within any skill directory."""
    if err := _validate_file_path(file_path):
        return _err(err)
    if not file_content and file_content != "":
        return _err("file_content is required.")
    if (content_bytes := len(file_content.encode("utf-8"))) > MAX_SKILL_FILE_BYTES:
        return _err(f"File content is {content_bytes:,} bytes (limit: {MAX_SKILL_FILE_BYTES:,} "
                    f"bytes / 1 MiB). Consider splitting into smaller files.")
    if err := _validate_content_size(file_content, label=file_path):
        return _err(err)
    skill_dir, guard = _locate_for_write(name, "write_file", " Create it first with action='create'.")
    if guard:
        return guard
    target, err = _resolve_supporting_file(skill_dir, file_path)
    if guard := err or _guarded_write(name, skill_dir, target, "write_file", file_path, file_content):
        return guard
    result = _attach_org_note({"success": True, "message": f"File '{file_path}' written to skill '{name}'.",
                               "path": str(target)}, name, skill_dir)
    # references/ is where per-session hoarding shows up; surface the sprawl finding on the write
    # that crosses the line so the review fork sees it in the same turn.
    if file_path.startswith("references/") and (skill_dir / "SKILL.md").exists():
        _attach_lint_findings(result, skill_dir / "SKILL.md")
    return result


def _remove_file(name: str, file_path: str) -> Dict[str, Any]:
    """Remove a supporting file from any skill directory."""
    if err := _validate_file_path(file_path):
        return _err(err)
    skill_dir, guard = _locate_for_write(name, "remove_file", org_guard=False)
    if guard:
        return guard
    target, err = _resolve_supporting_file(skill_dir, file_path)
    if err:
        return err
    if not target.exists():  # list what IS there so the model can pick the right path
        available = [str(f.relative_to(skill_dir)) for subdir in ALLOWED_SUBDIRS
                     if (skill_dir / subdir).exists() for f in (skill_dir / subdir).rglob("*") if f.is_file()]
        return _err(f"File '{file_path}' not found in skill '{name}'.", available_files=available or None)
    if read_guard := _background_review_read_before_write_guard(name, target, "remove_file", file_path):
        return read_guard
    target.unlink()
    _rmdir_if_empty(target.parent, skill_dir)
    return {"success": True, "message": f"File '{file_path}' removed from skill '{name}'."}


# --- Main entry point ---------------------------------------------------------

# Set while replaying an approved staged skill write so skill_manage() does not re-gate it.
_skill_gate_bypass: "_ctxvars.ContextVar[bool]" = _ctxvars.ContextVar(
    "skill_gate_bypass", default=False)


def _run_write_gate(build_staging):
    """Shared write gate: None to proceed, else a JSON tool result (blocked/staged).
    ``build_staging(wa) -> (payload, gist)`` runs only when staging. Fails open if
    write_approval cannot be imported."""
    try:
        from tools import write_approval as wa
    except Exception:
        return None  # fail open
    decision = wa.evaluate_gate(wa.SKILLS)
    if decision.allow:
        return None
    if decision.blocked:
        return tool_error(decision.message, success=False)
    payload, gist = build_staging(wa)
    record = wa.stage_write(wa.SKILLS, payload, summary=gist, origin=wa.current_origin())
    return json.dumps({"success": True, "staged": True, "pending_id": record["id"],
                       "gist": gist, "message": decision.message}, ensure_ascii=False)


def _apply_skill_write_gate(action, name, **payload_kwargs):
    """Flat-shape gate: stage the full kwargs so approval can replay them; bypassed during replay."""
    if action not in _ACTION_HANDLERS or _skill_gate_bypass.get():
        return None
    def _staging(wa):
        payload = {"action": action, "name": name,
                   **{k: v for k, v in payload_kwargs.items() if v is not None}}
        gist_kw = {k: payload_kwargs.get(k) or ""
                   for k in ("content", "file_path", "old_string", "new_string")}
        return payload, wa.skill_gist(action, name, **gist_kw)
    return _run_write_gate(_staging)


_FLAT_OP_KEYS = ("content", "category", "file_path", "file_content", "old_string", "new_string",
                 "absorbed_into", "operations")


def _skill_manage_from(payload: Dict[str, Any], **extra) -> str:
    """Call ``skill_manage`` with the flat-shape fields (and absorbed_into/operations) of ``payload``."""
    return skill_manage(
        action=payload.get("action", ""), name=payload.get("name", ""),
        replace_all=payload.get("replace_all", False),
        **{k: payload.get(k) for k in _FLAT_OP_KEYS}, **extra)


def apply_skill_pending(payload: Dict[str, Any]) -> str:
    """Replay a staged skill write, bypassing the gate (the /skills approve handler)."""
    token = _skill_gate_bypass.set(True)
    try:
        return _skill_manage_from(payload)
    finally:
        _skill_gate_bypass.reset(token)


# Sync push debounce: a burst of skill_manage writes collapses into one push on a daemon timer.
_sync_push_timer = None
_sync_push_lock = threading.Lock()
_SYNC_PUSH_DEBOUNCE_S = 5.0


def _maybe_debounced_sync_push(skill_name: str) -> None:
    """Debounced best-effort sync push after a skill write; never blocks the caller. Skills not
    opted into sync do nothing (no auth/network); ``maybe_push_skills`` enforces the access gate."""
    global _sync_push_timer
    try:
        from tools.skill_usage import is_sync_enabled
        if not is_sync_enabled(skill_name):
            return
    except Exception:
        return
    def _fire():
        with suppress(Exception):
            from tools.skills_sync_client import maybe_push_skills
            maybe_push_skills(message=f"sync: {skill_name}")
    with _sync_push_lock:
        if _sync_push_timer is not None:
            _sync_push_timer.cancel()  # only sets an Event; never raises
        _sync_push_timer = threading.Timer(_SYNC_PUSH_DEBOUNCE_S, _fire)
        _sync_push_timer.daemon = True
        _sync_push_timer.start()


def _act_patch(a):
    """Two shapes: old_string/new_string = targeted replacement (validated in _patch_skill so the
    tool and the helper give the same guidance); content alone = full rewrite (the old 'edit')."""
    if a["content"] and (a["old_string"] or a["new_string"] is not None):
        return tool_error("Pass EITHER content (full SKILL.md rewrite) OR "
                          "old_string/new_string (targeted replacement), not both.", success=False)
    if a["content"]:
        return _edit_skill(a["name"], a["content"])
    return _patch_skill(a["name"], a["old_string"], a["new_string"], a["file_path"], a["replace_all"])


# action -> handler(args dict) returning a result dict, or a tool_error JSON string for
# argument-shape errors. "edit" is a legacy alias for a full rewrite (not in the schema).
_ACTION_HANDLERS = {
    "create": lambda a: _create_skill(a["name"], a["content"], a["category"]),
    "edit": lambda a: _edit_skill(a["name"], a["content"]),
    "patch": _act_patch,
    "delete": lambda a: _delete_skill(a["name"], absorbed_into=a["absorbed_into"]),
    "write_file": lambda a: _write_file(a["name"], a["file_path"], a["file_content"]),
    "remove_file": lambda a: _remove_file(a["name"], a["file_path"])}
# action -> (arg, is_missing, error) argument-shape checks run before the handler.
_MISSING, _IS_NONE = (lambda v: not v), (lambda v: v is None)
_REQUIRED_ARGS = {
    "create": [("content", _MISSING,
                "content is required for 'create'. Provide the full SKILL.md text (frontmatter + body).")],
    "edit": [("content", _MISSING,
              "content is required for a full rewrite. Provide the full updated SKILL.md text.")],
    "write_file": [
        ("file_path", _MISSING, "file_path is required for 'write_file'. Example: 'references/api-guide.md'"),
        ("file_content", _IS_NONE, "file_content is required for 'write_file'.")],
    "remove_file": [("file_path", _MISSING, "file_path is required for 'remove_file'.")]}


def _record_success(action, name, result, *, file_path, absorbed_into, task_id,
                    session_id, ledger_before) -> None:
    """Best-effort post-mutation side effects (never break the tool): ledger, prompt-cache
    clear, curator telemetry, debounced sync push."""
    with suppress(Exception):
        from tools import skill_ledger as _ledger
        _post = _find_skill(name)
        # delete: consolidation vs prune, and whether the recoverable archive handled it
        _evidence = ({"absorbed_into": absorbed_into, "archived": bool(result.get("_archived"))}
                     if action == "delete" else {})
        _evidence.update({k: v for k, v in (("session_id", session_id), ("file_path", file_path)) if v})
        _ledger.record_mutation(
            action, name, before=ledger_before if ledger_before is not None else [],
            after_root=_post["path"] if _post else None, evidence=_evidence)
    with suppress(Exception):
        from agent.prompt_builder import clear_skills_system_prompt_cache
        clear_skills_system_prompt_cache(clear_snapshot=True)
    # Curator telemetry: only the background review fork marks a skill agent-created
    # (foreground creates belong to the user). A recoverable curator archive keeps its
    # record as STATE_ARCHIVED (`hermes curator status`/`restore`); only a hard delete forgets.
    with suppress(Exception):
        from tools.skill_usage import bump_patch, forget, record_created
        # During the curator consolidation pass, a verified consolidation must be RECOVERABLE: archival into
        # ~/.hermes/skills/.archive/ is documented as the maximum destructive action the curator may take,
        # and `hermes curator restore` promises the skill can be brought back. Route through the recoverable
        # archive primitive instead of permanent rmtree so a misjudged consolidation can be undone (#29912).
        # Foreground, user-directed deletes keep their existing hard-delete semantics.
        from tools.skill_provenance import is_background_review
        if action == "create":
            record_created(name, agent_created=is_background_review(),
                           task_id=task_id, session_id=session_id)
        elif action in {"patch", "edit", "write_file", "remove_file"}:
            bump_patch(name, action=action, task_id=task_id, session_id=session_id)
        elif action == "delete" and not result.get("_archived"):
            forget(name)
    # Only AFTER the write gate passed (staged writes returned early): never push un-reviewed content.
    with suppress(Exception):
        _maybe_debounced_sync_push(name)


def skill_manage(
    action: str, name: str, content: str = None, category: str = None, file_path: str = None,
    file_content: str = None, old_string: str = None, new_string: str = None,
    replace_all: bool = False, absorbed_into: str = None, task_id: str = None,
    session_id: str = None, operations=None) -> str:
    """Dispatch to the action handler -> JSON string. ``operations`` (atomic batch shape,
    see _skill_manage_batch) overrides the flat fields."""
    if operations is not None:
        return _skill_manage_batch(
            operations, default_name=name or None, task_id=task_id, session_id=session_id)
    if (preflight := _background_review_preflight(action, name)) is not None:
        return json.dumps(preflight, ensure_ascii=False)
    # Approval gate: skills are too large to review inline, so they always stage regardless
    # of origin; bypassed when replaying an approved staged write.
    args = dict(content=content, category=category, file_path=file_path, file_content=file_content,
                old_string=old_string, new_string=new_string, replace_all=replace_all,
                absorbed_into=absorbed_into)
    if (gate_result := _apply_skill_write_gate(action, name, **args)) is not None:
        return gate_result
    # Ledger pre-capture: telemetry, not a gate — failures must NEVER block the mutation. delete
    # destroys the whole package (consolidation may have re-homed support files first), so
    # complete it from the newest curator backup or a restore is hollow.
    # Audit ledger (tracker #79686 P3): capture the pre-mutation state of the skill directory so every
    # mutation — any actor — lands in the append-only JSONL ledger with before/after blobs.
    _ledger_before = None
    with suppress(Exception):
        from tools import skill_ledger as _ledger
        _pre = _find_skill(name)
        _ledger_before = _ledger.capture_before(
            _pre["path"] if _pre else None, complete_package=(action == "delete"), skill=name)
    for arg, missing, message in _REQUIRED_ARGS.get(action, ()):
        if missing(args[arg]):
            return tool_error(message, success=False)
    handler = _ACTION_HANDLERS.get(action, lambda a: _err(
        f"Unknown action '{action}'. Use: create, edit, patch, delete, write_file, remove_file"))
    result = handler({"name": name, **args})
    if isinstance(result, str):
        return result  # tool_error JSON for argument-shape problems (patch)
    if result.get("success"):
        _record_success(
            action, name, result, file_path=file_path, absorbed_into=absorbed_into,
            task_id=task_id, session_id=session_id, ledger_before=_ledger_before)
    return json.dumps(result, ensure_ascii=False)


# --- OpenAI Function-Calling Schema -------------------------------------------

SKILL_MANAGE_SCHEMA = {
    "name": "skill_manage",
    # ONE advertised call shape (memory-tool pattern): the call IS an operations
    # array. The legacy flat shape (top-level action/name/content/...) is still
    # ACCEPTED for old transcripts and staged-write replay, but not advertised.
    "description": (
        "Create, update, or delete skills — your procedural memory for "
        "recurring task types. The call is an operations array (a single "
        "edit is a list of one); it applies atomically — any failure rolls "
        "every touched skill back. Ops: create (full SKILL.md; lands in "
        f"{_display_create_dir()}; must precede that skill's other "
        "ops), patch (targeted old_string/new_string fix — preferred; "
        "content alone REPLACES the whole file, read it via skill_view() "
        "first), write_file/remove_file (supporting files), delete (sole "
        "op only). Existing skills are modified wherever they live. Keep "
        "the description's first 57 chars a self-contained trigger: 'Use "
        "when <trigger>. <one-line behavior>.' Write lessons, not logs: "
        "imperative rule + why, no PR numbers/dates/incident narration, one "
        "rule per lesson, references/ named by topic (extend before adding). "
        "skill_view() shows format conventions."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "operations": {
                "type": "array",
                "description": "Ordered ops; each names its target skill.",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": (
                                "Skill name (lowercase, hyphens/underscores, "
                                "max 64 chars); an existing skill's name "
                                "unless creating."
                            )
                        },
                        "action": {
                            "type": "string",
                            "enum": ["create", "patch", "delete", "write_file", "remove_file"]
                        },
                        "content": {
                            "type": "string",
                            "description": (
                                "Full SKILL.md text (YAML frontmatter + "
                                "markdown body) for create, or a full "
                                "rewrite on patch."
                            )
                        },
                        "category": {
                            "type": "string",
                            "description": "Optional category subdir for create (e.g. 'devops')."
                        },
                        # patch args: same fuzzy-matching semantics as the
                        # `patch` tool — teach only skill-specific facts here.
                        "old_string": {
                            "type": "string",
                            "description": "Text to find (patch; same matching semantics as the patch tool)."
                        },
                        "new_string": {
                            "type": "string",
                            "description": "Replacement (patch); empty string deletes the match."
                        },
                        "replace_all": {
                            "type": "boolean",
                            "description": "patch: replace all occurrences (default false)."
                        },
                        "file_path": {
                            "type": "string",
                            "description": (
                                "Path RELATIVE to the skill's own directory, "
                                "e.g. 'references/api.md' — no leading slash, "
                                "never absolute. write_file/remove_file: "
                                "required; first segment references/, "
                                "templates/, scripts/, or assets/. patch: "
                                "optional (default SKILL.md)."
                            )
                        },
                        "file_content": {
                            "type": "string",
                            "description": "Content for write_file."
                        }
                    },
                    "required": ["name", "action"]
                }
            },
            # Also accepted, never advertised: the legacy flat single-op fields, and
            # `absorbed_into` on delete ops (curator-only vocabulary; the curator's
            # prompt documents it and the delete guard's error re-teaches it).
        },
        "required": ["operations"],
    },
}


# --- Registry ---
from tools.registry import registry, tool_error

registry.register(
    name="skill_manage", toolset="skills", schema=SKILL_MANAGE_SCHEMA, emoji="📝",
    handler=lambda args, **kw: _skill_manage_from(
        args, task_id=kw.get("task_id"), session_id=kw.get("session_id")))


# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# Names external plugins imported from this module before the Sep 2026 decomposition.
# Internal code MUST NOT use these (scripts/check_compat_pointers.py fails CI if it does).
# The whole block is removed by reverting the commit that added it.


_PLUGIN_COMPAT_LAZY = {
    'mark_background_review_skill_read': ('tools.skill_manager_guards', 'mark_background_review_skill_read'),
}


def __getattr__(name):  # PEP 562 — lazy so no import cycles
    target = _PLUGIN_COMPAT_LAZY.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib
    from hermes_cli.plugin_compat import warn_once
    warn_once(__name__, name, *target)
    return getattr(importlib.import_module(target[0]), target[1])
# ---- END PLUGIN-COMPAT ----
