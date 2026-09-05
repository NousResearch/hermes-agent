#!/usr/bin/env python3
"""Skills Tool — list and view skill documents (progressive disclosure). A skill is a directory
holding SKILL.md (YAML frontmatter + instructions) plus optional references/, templates/, assets/,
scripts/. `skills_list` returns name/description only; `skill_view` returns full content and
linked files. Sibling modules (skills_tool_setup / _plugin / _dedup) re-export here."""

import json
import logging
import os
import time
from contextlib import suppress
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Dict, List, Optional, Tuple

from hermes_constants import get_hermes_home
from tools.registry import registry, tool_error
from hermes_cli.config import cfg_get
from agent.skill_utils import (
    EXCLUDED_SKILL_DIRS as _EXCLUDED_SKILL_DIRS, is_skill_support_path as _is_skill_support_path)
from tools.skills_tool_setup import (  # noqa: F401
    SkillReadinessStatus, _build_setup_note, _capture_required_environment_variables,
    _get_required_environment_variables, _is_env_var_persisted, _is_remote_env_backend)
from tools.skills_tool_plugin import (  # noqa: F401
    MAX_DESCRIPTION_LENGTH, MAX_NAME_LENGTH, _INJECTION_PATTERNS, _fail, _json,
    _mark_background_review_read, _preprocess_skill, _read_skill_text, _safe_frontmatter,
    _serve_plugin_skill, _serve_skill_file, _truncate_description)
from tools.skills_tool_dedup import (  # noqa: F401
    _check_skill_view_dedup, _record_skill_view, reset_skill_view_dedup)

logger = logging.getLogger(__name__)

# Per-session discovery cache: {cache_key: (signature, timestamp, skills_list)}. Signature =
# per-dir max mtime of the dir and its immediate children (add/remove inside a category does
# NOT bump the root mtime) + the disabled set (config-only change, no mtime) + platform; the
# TTL bounds staleness from in-place SKILL.md edits, which no directory signature can see.
_SKILLS_CACHE: dict = {}
_SKILLS_CACHE_TTL_SECONDS = 30.0


def _skills_scan_signature(dirs_to_scan, disabled) -> tuple:
    """O(#dirs + #categories) stat-based change signature; platform is read via
    ``agent.skill_utils.sys`` so test patches are honored."""
    from agent import skill_utils as _skill_utils
    platform = getattr(getattr(_skill_utils, "sys", None), "platform", "")
    sig = []
    for d in dirs_to_scan:
        try:
            m = d.stat().st_mtime
        except OSError:
            continue
        with suppress(OSError), os.scandir(d) as it:
            for entry in it:
                with suppress(OSError):
                    if entry.is_dir(follow_symlinks=False):
                        m = max(m, entry.stat(follow_symlinks=False).st_mtime)
        sig.append((str(d), m))
    return (tuple(sig), frozenset(disabled), platform)


HERMES_HOME = get_hermes_home()  # all skills live in ~/.hermes/skills/ (seeded from bundled)
SKILLS_DIR = HERMES_HOME / "skills"
_SKILLS_DIR_AT_IMPORT = SKILLS_DIR


def _skills_dir() -> Path:
    """Active profile's skills dir at call time: the patched ``SKILLS_DIR`` when a patcher changed
    it, else live profile-scoped HERMES_HOME (long-lived runtimes may import before profile set)."""
    configured = Path(SKILLS_DIR)
    return configured if configured != _SKILLS_DIR_AT_IMPORT else get_hermes_home() / "skills"


_secret_capture_callback = None
_LOOKUP_HINT = "Use a skill name or relative path within the skills directory."


def _skill_lookup_path_error(name: str) -> Optional[str]:
    """Error if lookup *name* could escape the search roots it is joined onto. Windows drive
    paths are rejected too: their ``:`` would be misread as a plugin namespace separator."""
    from tools.path_security import has_traversal_component
    if not isinstance(name, str):
        return "Skill name must be a string."
    win = PureWindowsPath(candidate := name.strip())
    if PurePosixPath(candidate).is_absolute() or win.is_absolute() or win.drive:
        return "Skill name must be a relative path within the skills directory."
    if has_traversal_component(candidate):
        return "Skill name cannot contain '..' path traversal components."
    return None


def load_env() -> Dict[str, str]:
    """Load profile-scoped environment variables from HERMES_HOME/.env."""
    env_path = get_hermes_home() / ".env"
    env_vars: Dict[str, str] = {}
    if env_path.exists():
        # utf-8-sig: a Notepad BOM would otherwise glue U+FEFF onto the first key.
        with env_path.open(encoding="utf-8-sig", errors="replace") as f:
            for line in map(str.strip, f):
                if line and not line.startswith("#") and "=" in line:
                    key, _, value = line.removeprefix("export ").partition("=")
                    env_vars[key.strip()] = value.strip().strip("\"'")
    return env_vars


def set_secret_capture_callback(callback) -> None:
    global _secret_capture_callback
    _secret_capture_callback = callback


def _skill_utils_delegate(attr: str):
    """Lazy call-time delegate to ``agent.skill_utils.<attr>`` (re-export; patches honored)."""
    def _delegate(*args):
        from agent import skill_utils
        return getattr(skill_utils, attr)(*args)
    _delegate.__name__ = _delegate.__qualname__ = attr
    return _delegate


skill_matches_platform = _skill_utils_delegate("skill_matches_platform")
# Offer-time relevance gate (kanban/docker/s6), NOT hard compatibility; explicit loads bypass it.
skill_matches_environment = _skill_utils_delegate("skill_matches_environment")
_parse_frontmatter = _skill_utils_delegate("parse_frontmatter")
_get_disabled_skill_names = _skill_utils_delegate("get_disabled_skill_names")


def check_skills_requirements() -> bool:
    return True  # always available: the directory is created on first use


def _get_category_from_path(skill_path: Path) -> Optional[str]:
    """``~/.hermes/skills/mlops/axolotl/SKILL.md`` -> ``"mlops"``; active profile dir first
    (respects test monkeypatching), then skills.external_dirs."""
    dirs_to_check = [_skills_dir()]
    with suppress(Exception):
        from agent.skill_utils import get_external_skills_dirs
        dirs_to_check.extend(get_external_skills_dirs())
    for skills_dir in dirs_to_check:
        with suppress(ValueError):
            if len(parts := skill_path.relative_to(skills_dir).parts) >= 3:
                return parts[0]
    return None


def _parse_tags(tags_value) -> List[str]:
    """Tags from frontmatter: a parsed list, "[a, b]", or "a, b"."""
    if not tags_value:
        return []
    if isinstance(tags_value, list):
        return [str(t).strip() for t in tags_value if t]
    tags_value = str(tags_value).strip()
    if tags_value.startswith("[") and tags_value.endswith("]"):
        tags_value = tags_value[1:-1]
    return [t.strip().strip("\"'") for t in tags_value.split(",") if t.strip()]


def _is_skill_disabled(name: str, platform: str = None) -> bool:
    """Disabled in config? Platform precedence: explicit arg, ``HERMES_PLATFORM``, session
    ``HERMES_SESSION_PLATFORM``. A globally-disabled skill stays disabled on every platform
    (keep in sync with agent.skill_utils.get_disabled_skill_names)."""
    try:
        from hermes_cli.config import load_config
        skills_cfg = load_config().get("skills", {})
        resolved_platform = platform or os.getenv("HERMES_PLATFORM")
        if not resolved_platform:
            with suppress(Exception):
                from gateway.session_context import get_session_env
                resolved_platform = get_session_env("HERMES_SESSION_PLATFORM") or ""
        platform_disabled = None
        if resolved_platform:
            platform_disabled = cfg_get(skills_cfg, "platform_disabled", resolved_platform)
        in_platform = platform_disabled is not None and name in platform_disabled
        return in_platform or name in skills_cfg.get("disabled", [])
    except Exception:
        return False


def _skill_search_dirs() -> Tuple[list, list, Path]:
    """(project_dirs, all_dirs, active_skills_dir); trusted project-local dirs come FIRST so
    first-wins dedup / the collision resolver prefer them."""
    from agent.skill_utils import get_external_skills_dirs, get_project_skills_dirs
    project_dirs = list(get_project_skills_dirs())
    active_skills_dir = _skills_dir()
    all_dirs = project_dirs + ([active_skills_dir] if active_skills_dir.exists() else [])
    all_dirs += get_external_skills_dirs()
    return project_dirs, all_dirs, active_skills_dir


def _find_all_skills(*, skip_disabled: bool = False) -> List[Dict[str, Any]]:
    """All skills (name, description, category) across project/local/external dirs, first-wins
    by name; cached per session. ``skip_disabled=True`` ignores disabled state (config UI)."""
    from agent.skill_utils import iter_project_skill_files, iter_skill_index_files
    cache_key = "with_disabled" if skip_disabled else "filtered"
    disabled = set() if skip_disabled else _get_disabled_skill_names()
    project_dirs, dirs_to_scan, _ = _skill_search_dirs()
    signature = _skills_scan_signature(dirs_to_scan, disabled)
    now = time.monotonic()
    cached = _SKILLS_CACHE.get(cache_key)
    if cached is not None and cached[0] == signature and (now - cached[1]) < _SKILLS_CACHE_TTL_SECONDS:
        # Shallow copies: callers mutate the returned dicts (web_server annotates
        # s["enabled"]/s["usage"]); handing out cached objects would poison the cache.
        return [dict(s) for s in cached[2]]
    skills = []
    seen_names: set = set()
    for scan_dir in dirs_to_scan:  # project dirs go through the quarantine chokepoint
        _iter = iter_project_skill_files if scan_dir in project_dirs else lambda d: iter_skill_index_files(d, "SKILL.md")
        for skill_md in _iter(scan_dir):
            if any(part in _EXCLUDED_SKILL_DIRS for part in skill_md.parts):
                continue
            try:
                frontmatter, body = _parse_frontmatter(_read_skill_text(skill_md)[:4000])
                if not skill_matches_platform(frontmatter) or not skill_matches_environment(frontmatter):
                    continue
                name = frontmatter.get("name", skill_md.parent.name)[:MAX_NAME_LENGTH]
                if name in seen_names or name in disabled:
                    continue
                description = frontmatter.get("description", "")
                if not description:  # first non-heading body line (a null value stays null)
                    description = next((ln for ln in map(str.strip, body.strip().split("\n"))
                                        if ln and not ln.startswith("#")), description)
                seen_names.add(name)
                skills.append({"name": name, "description": _truncate_description(description),
                               "category": _get_category_from_path(skill_md)})
            except (UnicodeDecodeError, PermissionError) as e:
                logger.debug("Failed to read skill file %s: %s", skill_md, e)
            except Exception as e:
                logger.debug("Skipping skill at %s: failed to parse: %s", skill_md, e, exc_info=True)
    # Keyed by the signature computed BEFORE the scan: a write racing the scan changes the
    # signature, so the next call re-scans instead of serving a torn result.
    _SKILLS_CACHE[cache_key] = (signature, now, skills)
    return [dict(s) for s in skills]


def _sort_skills(skills: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Keep every skill listing path ordered the same way."""
    return sorted(skills, key=lambda s: (s.get("category") or "", s["name"]))


def skills_list(category: str = None, task_id: str = None) -> str:
    """Tier 1 listing: name + description (+ category) only; ``task_id`` is handler parity."""
    try:
        _skills_dir().mkdir(parents=True, exist_ok=True)
        all_skills = _find_all_skills()
        try:
            from hermes_cli.plugins import discover_plugins, get_plugin_manager
            discover_plugins()
            for plugin_skill in get_plugin_manager().list_plugin_skill_metadata():
                frontmatter = plugin_skill.pop("frontmatter", {})
                if not skill_matches_platform(frontmatter) or _is_skill_disabled(plugin_skill["name"]):
                    continue
                all_skills.append(plugin_skill)
        except Exception:
            logger.debug("Plugin skill listing failed", exc_info=True)
        if not all_skills:
            return _json({"success": True, "skills": [], "categories": [],
                          "message": "No skills found in skills/ directory."})
        if category:
            all_skills = [s for s in all_skills if s.get("category") == category]
        all_skills = _sort_skills(all_skills)
        categories = sorted({s.get("category") for s in all_skills if s.get("category")})
        return _json({
            "success": True, "skills": all_skills, "categories": categories,
            "count": len(all_skills),
            "hint": "Use skill_view(name) to see full content, tags, and linked files"})
    except Exception as e:
        return tool_error(str(e), success=False)


def _resolve_plugin_skill(name, file_path, task_id, preprocess):
    """``plugin:skill`` dispatch: ``(result_json, None)`` when answered, else ``(None,
    local_category_name)`` to fall through to the flat-tree scan — categorized local skills also use
    ``category:skill`` in config/gateway prompts, so the on-disk ``category/skill`` form returns."""
    from agent.skill_utils import is_valid_namespace, parse_qualified_name
    from hermes_cli.plugins import discover_plugins, get_plugin_manager
    namespace, bare = parse_qualified_name(name)
    if not is_valid_namespace(namespace):
        return _fail(f"Invalid namespace '{namespace}' in '{name}'. Namespaces must match [a-zA-Z0-9_-]+."), None
    discover_plugins()  # idempotent
    pm = get_plugin_manager()
    active_memory_provider = None
    try:
        from plugins.memory import _get_active_memory_provider, _prune_inactive_memory_provider_skills
        active_memory_provider = _get_active_memory_provider()
        _prune_inactive_memory_provider_skills(active_memory_provider)
    except Exception as exc:
        logger.debug("Failed pruning inactive memory-provider skills: %s", exc)
    plugin_skill_md = pm.find_plugin_skill(name)
    # Memory providers load through plugins.memory, not the general PluginManager: load the
    # namespaced provider once so its collector can forward its skills into the registry.
    if plugin_skill_md is None and namespace == active_memory_provider:
        try:
            from plugins.memory import load_memory_provider
            load_memory_provider(namespace)
            plugin_skill_md = pm.find_plugin_skill(name)
        except Exception as exc:
            logger.debug("Failed lazy memory-provider skill load for %s: %s", namespace, exc)
    if plugin_skill_md is not None and not plugin_skill_md.exists():
        pm.remove_plugin_skill(name)  # stale registry entry — file deleted out of band
        return _fail(
            f"Skill '{name}' file no longer exists at {plugin_skill_md}. The registry entry "
            f"has been cleaned up — try again after the plugin is reloaded."), None
    if plugin_skill_md is not None:
        return _serve_plugin_skill(
            plugin_skill_md, namespace, bare, file_path=file_path, preprocess=preprocess, session_id=task_id), None
    if available := pm.list_plugin_skills(namespace):  # plugin exists but this specific skill is missing
        return _fail(
            f"Skill '{bare}' not found in plugin '{namespace}'.",
            available_skills=[f"{namespace}:{s}" for s in available],
            hint=f"The '{namespace}' plugin provides {len(available)} skill(s)."), None
    return None, (f"{namespace}/{bare}" if bare else None)  # plugin not found → local scan


def _under_any(path: Path, dirs) -> bool:
    """True when ``path`` (resolved where possible) lives under one of ``dirs``."""
    resolved = path
    with suppress(Exception):
        resolved = path.resolve()
    return any(resolved.is_relative_to(d) for d in dirs)


def _collect_skill_candidates(name, local_category_name, all_dirs):
    """ALL (skill_dir, skill_md) candidates across every dir and lookup strategy (direct path,
    recursive by dir / frontmatter name, legacy flat <name>.md), deduped by resolved path.
    Collision detection is the point: silent shadowing of a local skill by a same-named
    external one is a real bug class, so the caller refuses >1."""
    from agent.skill_utils import iter_skill_index_files
    candidates: List[Tuple[Optional[Path], Path]] = []
    seen_md: set = set()

    def _record(sd: Optional[Path], smd: Path) -> None:
        key = smd
        with suppress(Exception):
            key = smd.resolve()
        if key not in seen_md:
            seen_md.add(key)
            candidates.append((sd, smd))

    def _record_direct(direct_path: Path) -> None:  # "mlops/axolotl" / "axolotl" or its flat .md sibling
        flat = direct_path.with_suffix(".md")
        if not _is_skill_support_path(direct_path) and direct_path.is_dir() and (direct_path / "SKILL.md").exists():
            _record(direct_path, direct_path / "SKILL.md")
        elif flat.exists() and not _is_skill_support_path(flat):
            _record(None, flat)

    for search_dir in all_dirs:
        for direct in filter(None, (name, local_category_name)):  # "p:x" with no plugin p → "p/x"
            _record_direct(search_dir / direct)
        # Recursive by directory name plus frontmatter `name:` — skills_list()
        # exposes the frontmatter name, so skill_view(name) must accept it too.
        for found_skill_md in iter_skill_index_files(search_dir, "SKILL.md"):
            if (found_skill_md.parent.name == name
                    or _safe_frontmatter(found_skill_md).get("name") == name):
                _record(found_skill_md.parent, found_skill_md)
        # Legacy flat <name>.md anywhere under the dir; support docs are excluded
        # (they load via file_path and must not shadow real skills sharing the basename).
        for found_md in search_dir.rglob(f"{name}.md"):
            if found_md.name != "SKILL.md" and not _is_skill_support_path(found_md):
                _record(None, found_md)
    return candidates


# (support dir, globs, recursive, files only) — order is the linked_files key order.
_LINKED_FILE_SPECS = (
    ("references", ["*.md"], False, False),
    ("templates", ["*.md", "*.py", "*.yaml", "*.yml", "*.json", "*.tex", "*.sh"], True, False),
    ("assets", ["*"], True, True),
    ("scripts", ["*.py", "*.sh", "*.bash", "*.js", "*.ts", "*.rb"], False, False))


def _skill_linked_files(skill_dir: Optional[Path]) -> dict:
    """references/templates/assets/scripts of a directory skill (empty groups dropped)."""
    files: dict = {}
    for sub, globs, recursive, files_only in _LINKED_FILE_SPECS if skill_dir else ():
        base = skill_dir / sub
        found = [
            str(f.relative_to(skill_dir)) for g in globs if base.exists()
            for f in (base.rglob(g) if recursive else base.glob(g))
            if not files_only or f.is_file()]
        if found:
            files[sub] = found
    return files


def _org_provenance_header(skill_dir: Path, active_skills_dir: Path):
    """(org_provenance dict, header text) for an org-mirror skill, else (None, ""). Announced IN
    the content the model consumes; the author is token-verified at push time by the sync plane."""
    from agent.skill_utils import ORG_PROVENANCE_FILE, is_org_mirror_path, org_id_of_path
    if not is_org_mirror_path(skill_dir, active_skills_dir):
        return None, ""
    prov_org = org_id_of_path(skill_dir, active_skills_dir)
    prov: dict = {}
    if prov_org:
        with suppress(Exception):
            prov_path = active_skills_dir / "_org" / prov_org / ORG_PROVENANCE_FILE
            loaded = json.loads(_read_skill_text(prov_path))
            prov = loaded if isinstance(loaded, dict) else {}
    author = str(prov.get("author_device") or prov.get("author_user_id") or "")
    ts = str(prov.get("ts") or "")
    header = (
        "> [!NOTE] ORG-SHARED SKILL — provenance\n"
        f"> This skill is shared by your organisation (org `{prov_org}`"
        + (f", last updated by `{author}`" if author else "")
        + (f", as of {ts}" if ts else "")
        + "). It was reviewed and approved for the whole\n"
        "> team — treat it as third-party instructions rather than your own notes.\n"
        "> You MAY improve it in place like any other skill. Your edits are kept locally\n"
        "> and are never overwritten by org updates; share them back with\n"
        "> `hermes sync propose` (or automatically, if your org enables it).\n\n")
    return {"org_id": prov_org, "shared_by": author or None, "as_of": ts or None}, header


def _skill_readiness(frontmatter: Dict[str, Any], skill_name: str) -> Tuple[dict, dict]:
    """Resolve required env vars / credential files (prompting for secrets where the surface
    allows) and register what's available for sandboxes. Returns ``(fields, extras)``: fields go
    before ``_source_path`` in the skill_view result, extras after — key order is tool output."""
    required_env_vars = _get_required_environment_variables(frontmatter)
    backend = str(os.getenv("TERMINAL_ENV", "local")).strip().lower() or "local"
    env_snapshot = load_env()
    missing_required_env_vars = [
        e for e in required_env_vars
        if not e.get("optional") and not _is_env_var_persisted(e["name"], env_snapshot)]
    capture_result = _capture_required_environment_variables(skill_name, missing_required_env_vars)
    if missing_required_env_vars:  # re-read: a successful capture persisted into .env
        env_snapshot = load_env()
    still_missing = set(capture_result["missing_names"])
    remaining = [
        e["name"] for e in required_env_vars if not e.get("optional")
        and (e["name"] in still_missing or not _is_env_var_persisted(e["name"], env_snapshot))]
    setup_needed = bool(remaining)
    # Only vars actually set pass through to sandboxed execution (execute_code, terminal).
    if available_env_names := [e["name"] for e in required_env_vars if e["name"] not in remaining]:
        try:
            from tools.env_passthrough import register_env_passthrough
            register_env_passthrough(available_env_names)
        except Exception:
            logger.debug("Could not register env passthrough for skill %s", skill_name, exc_info=True)
    # Credential files for remote sandboxes: existing host files are registered,
    # missing ones flag setup_needed.
    required_cred_files_raw = frontmatter.get("required_credential_files", [])
    missing_cred_files: list = []
    if isinstance(required_cred_files_raw, list) and required_cred_files_raw:
        try:
            from tools.credential_files import register_credential_files
            missing_cred_files = register_credential_files(required_cred_files_raw)
            setup_needed = setup_needed or bool(missing_cred_files)
        except Exception:
            logger.debug("Could not register credential files for skill %s", skill_name, exc_info=True)
    status = SkillReadinessStatus.SETUP_NEEDED if setup_needed else SkillReadinessStatus.AVAILABLE
    fields = {
        "required_environment_variables": required_env_vars, "required_commands": [],
        "missing_required_environment_variables": remaining,
        "missing_credential_files": missing_cred_files, "missing_required_commands": [],
        "setup_needed": setup_needed, "setup_skipped": capture_result["setup_skipped"],
        "readiness_status": status.value}
    extras: dict = {}
    if setup_help := next((e["help"] for e in required_env_vars if e.get("help")), None):
        extras["setup_help"] = setup_help
    if capture_result["gateway_setup_hint"]:
        extras["gateway_setup_hint"] = capture_result["gateway_setup_hint"]
    missing_items = [f"env ${n}" for n in remaining] + [f"file {p}" for p in missing_cred_files]
    if setup_needed and (setup_note := _build_setup_note(status, missing_items, setup_help)):
        if _is_remote_env_backend(backend):
            setup_note = f"{setup_note} {backend.upper()}-backed skills need these requirements available inside the remote environment as well."
        extras["setup_note"] = setup_note
    return fields, extras


def _locate_skill(name: str, local_category_name: Optional[str], project_dirs: list, all_dirs):
    """Unique on-disk skill for *name*: collision refusal, project-tier precedence, quarantine
    gate, not-found listing. ``(error_json, skill_dir, skill_md)``; skill_md set iff no error."""
    if not all_dirs:
        return _fail(
            "Skills directory does not exist yet. It will be created on first install."), None, None
    candidates = _collect_skill_candidates(name, local_category_name, all_dirs)
    if len(candidates) > 1 and project_dirs:
        # A project skill intentionally overrides a same-named local/external skill;
        # ambiguity WITHIN the project tier still refuses.
        candidates = [c for c in candidates if _under_any(c[1], project_dirs)] or candidates
    if len(candidates) > 1:
        paths = [str(smd) for _, smd in candidates]
        logger.warning("Skill name collision for '%s': %d candidates — %s", name, len(candidates), "; ".join(paths))
        return _fail(
            f"Ambiguous skill name '{name}': {len(candidates)} skills match across your local skills dir "
            "and external_dirs. Refusing to guess — load one explicitly by its categorized path.",
            matches=paths,
            hint="Pass the full relative path instead of the bare name (e.g., 'category/skill-name'), "
            "or rename one of the colliding skills so each name is unique."), None, None
    skill_dir, skill_md = candidates[0] if candidates else (None, None)
    # Quarantine gate: a project-tier skill with a dangerous scan verdict must not
    # load even by explicit name (same chokepoint the index and skills_list use).
    if skill_md is not None and project_dirs:
        from agent.skill_utils import is_quarantined_project_skill
        if _under_any(skill_md, project_dirs) and is_quarantined_project_skill(skill_md):
            return _fail(
                f"Project skill '{name}' is quarantined: the security scan flagged its content as "
                "dangerous. It will not load until the repo's skill content changes and passes a re-scan.",
                hint="Inspect the skill in the repo checkout, or untrust the repo with "
                "`hermes skills untrust`."), None, None
    if not skill_md or not skill_md.exists():
        available = [s["name"] for s in _sort_skills(_find_all_skills())[:20]]
        return _fail(f"Skill '{name}' not found.", available_skills=available,
                     hint="Use skills_list to see all available skills"), None, None
    return None, skill_dir, skill_md


def _log_security_warnings(name: str, skill_md: Path, content: str, all_dirs, active_skills_dir):
    """Warn (never block) when loaded from outside the trusted dirs (project + local + external)
    and/or when common prompt-injection patterns appear."""
    trusted_dirs = [active_skills_dir.resolve()]
    with suppress(Exception):
        trusted_dirs.extend(d.resolve() for d in all_dirs)
    warnings = []
    if not _under_any(skill_md, trusted_dirs):
        warnings.append(f"skill file is outside the trusted skills directory (~/.hermes/skills/): {skill_md}")
    if any(p in content.lower() for p in _INJECTION_PATTERNS):
        warnings.append("skill content contains patterns that may indicate prompt injection")
    if warnings:
        logger.warning("Skill security warning for '%s': %s", name, "; ".join(warnings))


def skill_view(
    name: str, file_path: str = None, task_id: str = None, preprocess: bool = True) -> str:
    """View a skill (SKILL.md) or a file within its directory, as JSON. ``name`` is a skill name
    or path ("axolotl", "03-fine-tuning/axolotl"); "plugin:skill" resolves plugin-provided
    skills. ``preprocess`` applies the configured SKILL.md template / inline shell rendering;
    slash/preload callers render the message themselves."""
    try:
        # Validate before the ':' dispatch so a Windows drive path (C:\skills\foo) can't be
        # reinterpreted as a plugin namespace.
        if lookup_error := _skill_lookup_path_error(name):
            return _fail(lookup_error, hint=_LOOKUP_HINT)
        local_category_name: str | None = None
        if ":" in name:  # plugin registry; bare names use the flat-tree scan below
            served, local_category_name = _resolve_plugin_skill(name, file_path, task_id, preprocess)
            if served is not None:
                return served
        # The fall-through form (namespace/bare) joins onto each search dir too; re-validate it
        # since `bare` is not namespace-checked.
        if local_category_name and (lookup_error := _skill_lookup_path_error(local_category_name)):
            return _fail(lookup_error, hint=_LOOKUP_HINT)
        project_dirs, all_dirs, active_skills_dir = _skill_search_dirs()
        error, skill_dir, skill_md = _locate_skill(
            name, local_category_name, project_dirs, all_dirs)
        if error is not None:
            return error
        try:  # read once — reused for platform check and main content
            content = _read_skill_text(skill_md)
        except Exception as e:
            return _fail(f"Failed to read skill '{name}': {e}")
        _log_security_warnings(name, skill_md, content, all_dirs, active_skills_dir)
        frontmatter = _safe_frontmatter(content=content)
        if not skill_matches_platform(frontmatter):
            return _fail(f"Skill '{name}' is not supported on this platform.", readiness_status=SkillReadinessStatus.UNSUPPORTED.value)
        resolved_name = frontmatter.get("name", skill_md.parent.name)
        if _is_skill_disabled(resolved_name):
            return _fail(f"Skill '{resolved_name}' is disabled. Enable it with `hermes skills` or inspect the files directly on disk.")
        if file_path and skill_dir:
            return _serve_skill_file(
                skill_dir, file_path, name, list_available=True, mark_read=True,
                hint="Use a relative path within the skill directory")
        # tags/related_skills: metadata.hermes.* (agentskills.io) first, then top-level.
        metadata = frontmatter.get("metadata")
        hermes_meta = (metadata.get("hermes", {}) or {}) if isinstance(metadata, dict) else {}
        tags, related_skills = (
            _parse_tags(hermes_meta.get(k) or frontmatter.get(k, "")) for k in ("tags", "related_skills"))
        linked_files = _skill_linked_files(skill_dir)
        try:
            rel_path = str(skill_md.relative_to(active_skills_dir))
        except ValueError:  # external skill — relative to its own parent dir
            rel_path = str(skill_md.relative_to(skill_md.parent.parent)) if skill_md.parent.parent else skill_md.name
        skill_name = frontmatter.get("name", skill_md.stem if not skill_dir else skill_dir.name)
        readiness, readiness_extras = _skill_readiness(frontmatter, skill_name)
        rendered_content = content if not preprocess else _preprocess_skill(
            content, skill_dir, task_id, "Could not preprocess skill content for %s", skill_name)
        org_provenance, header = None, ""
        if skill_dir:
            try:
                org_provenance, header = _org_provenance_header(skill_dir, active_skills_dir)
            except Exception:
                logger.debug("Could not resolve org provenance for %s", skill_name, exc_info=True)
        result = {
            "success": True, "name": skill_name, "description": frontmatter.get("description", ""),
            "tags": tags, "related_skills": related_skills, "content": header + rendered_content,
            "path": rel_path, "skill_dir": str(skill_dir) if skill_dir else None,
            "org_provenance": org_provenance,
            "linked_files": linked_files if linked_files else None,
            "usage_hint": "To view linked files, call skill_view(name, file_path) where file_path is e.g. 'references/api.md' or 'assets/config.yaml'" if linked_files else None,
            **readiness,
            # Internal: absolute source path for the repeat-view dedup fingerprint.
            "_source_path": str(skill_md),
            **readiness_extras}
        _mark_background_review_read(skill_md)
        if frontmatter.get("compatibility"):  # agentskills.io optional fields
            result["compatibility"] = frontmatter["compatibility"]
        if isinstance(metadata, dict):
            result["metadata"] = metadata
        return _json(result)
    except Exception as e:
        return tool_error(str(e), success=False)


SKILLS_LIST_SCHEMA = {
    "name": "skills_list",
    "description": "List available skills (name + description). Use skill_view(name) to load full content.",
    "parameters": {
        "type": "object",
        "properties": {
            "category": {
                "type": "string",
                "description": "Optional category filter to narrow results",
            }
        },
        "required": [],
    },
}

SKILL_VIEW_SCHEMA = {
    "name": "skill_view",
    "description": "Skills allow for loading information about specific tasks and workflows, as well as scripts and templates. Load a skill's full content or access its linked files (references, templates, scripts). First call returns SKILL.md content plus a 'linked_files' dict showing available references/templates/scripts. To access those, call again with file_path parameter. Do not repeat the same name/file_path after a successful load; act on the loaded content instead.",
    "parameters": {
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "The skill name (use skills_list to see available skills). For plugin-provided skills, use the qualified form 'plugin:skill' (e.g. 'superpowers:writing-plans').",
            },
            "file_path": {
                "type": "string",
                "description": "OPTIONAL: Path to a linked file within the skill (e.g., 'references/api.md', 'templates/config.yaml', 'scripts/validate.py'). Omit to get the main SKILL.md content.",
            },
        },
        "required": ["name"],
    },
}

registry.register(
    name="skills_list", toolset="skills", schema=SKILLS_LIST_SCHEMA,
    handler=lambda args, **kw: skills_list(category=args.get("category"), task_id=kw.get("task_id")),
    check_fn=check_skills_requirements, emoji="📚")


def _skill_view_with_bump(args, **kw):
    """Invoke skill_view, then bump view_count/use on success (best-effort). Repeat-view dedup
    mirrors read_file's unchanged-stub: a SAME, unchanged skill file already loaded in this
    session returns a short stub (cache cleared on context compression)."""
    name = args.get("name", "")
    task_id = kw.get("task_id")
    if (stub := _check_skill_view_dedup(task_id, name, args.get("file_path"))) is not None:
        return stub
    result = skill_view(name, file_path=args.get("file_path"), task_id=task_id)
    with suppress(Exception):
        parsed = json.loads(result)
        if isinstance(parsed, dict) and parsed.get("success"):
            _record_skill_view(task_id, name, args.get("file_path"), parsed)
            if resolved := parsed.get("name") or name:  # qualified forms return the canonical name
                from tools.skill_usage import bump_use, bump_view
                bump_view(str(resolved))
                # Viewing is actively loading the skill to act on it — that counts as use
                # (the curator's stale timer keys off last_used_at).
                bump_use(str(resolved), task_id=kw.get("task_id"), session_id=kw.get("session_id"))
    return result


registry.register(
    name="skill_view", toolset="skills", schema=SKILL_VIEW_SCHEMA, handler=_skill_view_with_bump,
    check_fn=check_skills_requirements, emoji="📚")


# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# Names external plugins imported from this module before the Sep 2026 decomposition.
# Internal code MUST NOT use these (scripts/check_compat_pointers.py fails CI if it does).
# The whole block is removed by reverting the commit that added it.
from enum import Enum  # noqa: F401,E402
from typing import Set  # noqa: F401,E402
import re  # noqa: F401,E402
import threading  # noqa: F401,E402


_PLUGIN_COMPAT_LAZY = {
    'display_hermes_home': ('hermes_constants', 'display_hermes_home'),
    'env_var_enabled': ('utils', 'env_var_enabled'),
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
