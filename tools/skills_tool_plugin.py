"""Plugin-provided skill serving for ``skill_view`` (``plugin:skill`` names) plus the JSON /
file-serving helpers shared with the local-skill path. Helpers tests patch on the origin module
(``_is_skill_disabled``, ``_parse_frontmatter``, ``skill_matches_platform``) resolve lazily."""

import json
import logging
from contextlib import suppress
from pathlib import Path
from typing import Any, Dict, List

from tools.skills_tool_setup import SkillReadinessStatus

logger = logging.getLogger("tools.skills_tool")

MAX_NAME_LENGTH = 64  # Anthropic-recommended progressive-disclosure limits
MAX_DESCRIPTION_LENGTH = 1024
_INJECTION_PATTERNS: list = [  # shared by local-skill and plugin-skill paths
    "ignore previous instructions", "ignore all previous", "you are now",
    "disregard your", "forget your instructions", "new instructions:",
    "system prompt:", "<system>", "]]>"]
_SUPPORT_DIRS = ("references", "templates", "assets", "scripts")
_SKILL_FILE_EXTS = {".md", ".py", ".yaml", ".yml", ".json", ".tex", ".sh"}


def _json(payload: dict) -> str:
    return json.dumps(payload, ensure_ascii=False)


def _fail(error: str, **extra) -> str:
    return _json({"success": False, "error": error, **extra})


def _read_skill_text(path: Path) -> str:
    """utf-8-sig + errors="replace": user-authored SKILL.md may carry a Notepad BOM or stray
    bytes; pinning UTF-8 keeps skill_view deterministic across host locales."""
    return path.read_text(encoding="utf-8-sig", errors="replace")


def _truncate_description(description: str) -> str:
    if len(description) <= MAX_DESCRIPTION_LENGTH:
        return description
    return description[: MAX_DESCRIPTION_LENGTH - 3] + "..."


def _safe_frontmatter(path: Path | None = None, *, content: str | None = None) -> Dict[str, Any]:
    """Frontmatter of *path* (or of *content*), ``{}`` on any read/parse failure.
    Parses via ``tools.skills_tool._parse_frontmatter`` so test patches are honored."""
    from tools import skills_tool as _st
    with suppress(Exception):
        return _st._parse_frontmatter(_read_skill_text(path) if content is None else content)[0]
    return {}


def _available_skill_files(skill_dir: Path) -> Dict[str, List[str]]:
    """Non-SKILL.md files grouped by support dir (+ "other" for known source extensions
    elsewhere); empty groups dropped."""
    groups: Dict[str, List[str]] = {}
    for f in skill_dir.rglob("*"):
        if not f.is_file() or f.name == "SKILL.md":
            continue
        rel = str(f.relative_to(skill_dir))
        top = rel.split("/", 1)[0] if "/" in rel else None
        if top in _SUPPORT_DIRS or f.suffix in _SKILL_FILE_EXTS:
            groups.setdefault(top if top in _SUPPORT_DIRS else "other", []).append(rel)
    return {k: groups[k] for k in (*_SUPPORT_DIRS, "other") if k in groups}


def _serve_skill_file(
    skill_root: Path, file_path: str, label: str, *, hint: str | None = None,
    list_available: bool = False, read_error_prefix: bool = False, mark_read: bool = False) -> str:
    """Serve one linked file from a skill directory as a skill_view JSON result. ``hint``
    decorates traversal/containment errors; ``list_available`` adds the available-files listing
    on not-found (local); ``read_error_prefix`` wraps non-decode read errors (plugin) instead
    of propagating to the caller's handler."""
    from tools.path_security import has_traversal_component, validate_within_dir
    extra = {"hint": hint} if hint else {}
    if has_traversal_component(file_path):
        return _fail("Path traversal ('..') is not allowed.", **extra)
    target = skill_root / file_path
    if path_error := validate_within_dir(target, skill_root):
        return _fail(path_error, **extra)
    # is_file(), not exists(): a bare directory must take the not-found branch.
    if not target.is_file():
        listing = {} if not list_available else {
            "available_files": _available_skill_files(skill_root),
            "hint": "Use one of the available file paths listed above"}
        return _fail(f"File '{file_path}' not found in skill '{label}'.", **listing)
    try:
        content = _read_skill_text(target)
    except UnicodeDecodeError:
        return _json({
            "success": True, "name": label, "file": file_path, "is_binary": True,
            "content": f"[Binary file: {target.name}, size: {target.stat().st_size} bytes]"})
    except Exception as exc:
        if not read_error_prefix:
            raise
        return _fail(f"Failed to read '{file_path}': {exc}")
    if mark_read:
        _mark_background_review_read(target)
    return _json({  # _source_path: internal, feeds the repeat-view dedup fingerprint
        "success": True, "name": label, "file": file_path, "content": content,
        "file_type": target.suffix, "_source_path": str(target)})


def _mark_background_review_read(path: Path) -> None:
    try:
        from tools.skill_manager_guards import mark_background_review_skill_read
        mark_background_review_skill_read(path)
    except Exception:
        logger.debug("Could not record background-review skill read for %s", path, exc_info=True)


def _preprocess_skill(content: str, skill_dir, session_id, debug_msg: str, *args) -> str:
    """Apply the configured SKILL.md preprocessing; on failure log and serve raw."""
    try:
        from agent.skill_preprocessing import preprocess_skill_content
        return preprocess_skill_content(content, skill_dir, session_id=session_id)
    except Exception:
        logger.debug(debug_msg, *args, exc_info=True)
        return content


def _serve_plugin_skill(
    skill_md: Path, namespace: str, bare: str, file_path: str | None = None, *,
    preprocess: bool = True, session_id: str | None = None) -> str:
    """Read a plugin-provided skill, apply guards, return JSON."""
    from hermes_cli.plugins import _get_disabled_plugins, get_plugin_manager
    from tools import skills_tool as _st
    if namespace in _get_disabled_plugins():
        return _fail(f"Plugin '{namespace}' is disabled. Re-enable with: hermes plugins enable {namespace}")
    qualified_name = f"{namespace}:{bare}"
    try:
        content = _read_skill_text(skill_md)
    except Exception as e:
        return _fail(f"Failed to read skill '{qualified_name}': {e}")
    parsed_frontmatter = _safe_frontmatter(content=content)
    if _st._is_skill_disabled(qualified_name):
        return _fail(f"Skill '{qualified_name}' is disabled.")
    if not _st.skill_matches_platform(parsed_frontmatter):
        return _fail(f"Skill '{qualified_name}' is not supported on this platform.",
                     readiness_status=SkillReadinessStatus.UNSUPPORTED.value)
    if file_path:
        return _serve_skill_file(skill_md.parent, file_path, qualified_name, read_error_prefix=True)
    if any(p in content.lower() for p in _INJECTION_PATTERNS):
        logger.warning(
            "Plugin skill '%s:%s' contains patterns that may indicate prompt injection", namespace, bare)
    banner = ""
    with suppress(Exception):  # bundle-context banner: sibling skills of the same plugin
        siblings = [s for s in get_plugin_manager().list_plugin_skills(namespace) if s != bare]
        banner = f"[Bundle context: This skill is part of the '{namespace}' plugin." + (
            f"\nSibling skills: {', '.join(siblings)}.\nUse qualified form to invoke siblings "
            f"(e.g. {namespace}:{siblings[0]})." if siblings else "") + "]\n\n"
    rendered_content = content if not preprocess else _preprocess_skill(
        content, skill_md.parent, session_id, "Could not preprocess plugin skill %s:%s",
        namespace, bare)
    return _json({
        "success": True, "name": qualified_name, "content": banner + rendered_content,
        "description": _truncate_description(str(parsed_frontmatter.get("description", ""))),
        "linked_files": _plugin_skill_linked_files(skill_md.parent),
        "readiness_status": SkillReadinessStatus.AVAILABLE.value})


def _plugin_skill_linked_files(skill_root: Path) -> Dict[str, List[str]] | None:
    from tools.path_security import validate_within_dir
    linked: Dict[str, List[str]] = {}
    for category in _SUPPORT_DIRS:
        files = [
            str(path.relative_to(skill_root)) for path in sorted((skill_root / category).rglob("*"))
            if path.is_file() and validate_within_dir(path, skill_root) is None]
        if files:
            linked[category] = files
    return linked or None
