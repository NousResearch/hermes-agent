#!/usr/bin/env python3
"""list_skills — enriched L0 skill catalog for the desktop read-only surface.

Progressive disclosure tier 1 (L0): metadata only. Returns one entry per
visible skill with the fields the desktop catalog UI needs to render:

    {
        "id":                     stable identifier (qualified name for
                                  plugin skills, bare name otherwise),
        "slug":                   url-safe identifier (bare name),
        "name":                   display name (bare),
        "description":            short description (frontmatter or first
                                  non-empty body line),
        "source_badge":           one of "bundled" | "plugin" | "user",
        "requires_toolsets":      list[str] from metadata.hermes,
        "fallback_for_toolsets":  list[str] from metadata.hermes,
    }

Conditional visibility — a skill whose ``requires_toolsets`` are not all
registered is filtered out (e.g. a triage skill that requires a messaging
toolset stays hidden when the messaging connector has not been set up).
``fallback_for_toolsets`` is informational only and never hides a skill.

The full SKILL.md body is intentionally NOT crossed over the IPC boundary
by this tool — the desktop renders the catalog from this payload alone and
opens the detail view (L1) via the existing ``skill_view`` tool. That keeps
the L0 cost bounded to a directory scan + small frontmatter parses.

Registered in the shared tool inventory under the ``skills`` toolset so
it is exposed alongside ``skills_list`` / ``skill_view``.
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from tools.registry import registry, tool_error

logger = logging.getLogger(__name__)


_SLUG_RE = re.compile(r"[^a-z0-9-]+")
_MAX_NAME_LEN = 64
_MAX_DESCRIPTION_LEN = 1024


def _slugify(name: str) -> str:
    """Return a lowercase, hyphen-delimited slug for a display name."""
    s = name.strip().lower().replace("_", "-").replace(" ", "-")
    s = _SLUG_RE.sub("-", s)
    s = re.sub(r"-+", "-", s).strip("-")
    return s or "skill"


def _truncate(text: str, limit: int) -> str:
    text = (text or "").strip()
    if len(text) > limit:
        return text[: max(0, limit - 3)] + "..."
    return text


def _registered_toolsets() -> Set[str]:
    """Return the set of toolset names currently registered on the runtime."""
    try:
        names = set(registry.get_registered_toolset_names())
    except Exception:
        names = set()
    try:
        aliases = registry.get_registered_toolset_aliases()
        for alias, target in aliases.items():
            names.add(alias)
            names.add(target)
    except Exception:
        pass
    return names


def _local_skills_dir() -> Optional[Path]:
    """Return the user-local ``~/.hermes/skills`` directory if available."""
    try:
        from agent.skill_utils import get_skills_dir  # type: ignore

        return Path(get_skills_dir())
    except Exception:
        return None


def _external_skills_dirs() -> List[Path]:
    """Return external skill directories declared in config.yaml."""
    try:
        from agent.skill_utils import get_external_skills_dirs  # type: ignore

        return list(get_external_skills_dirs())
    except Exception:
        return []


def _bundled_skills_dirs() -> List[Path]:
    """Return directories shipped alongside Hermes / sidecar bundle.

    Honours ``HERMES_BUNDLED_SKILLS_DIR`` (set by the desktop bundle so the
    sidecar can find seed skills outside the user's HERMES_HOME). Falls back
    to a sibling ``skills/`` directory at the repo root, which is the source
    of truth checked in to the repository.
    """
    dirs: List[Path] = []
    env_dir = os.environ.get("HERMES_BUNDLED_SKILLS_DIR")
    if env_dir:
        p = Path(env_dir).expanduser()
        if p.is_dir():
            dirs.append(p)
    repo_skills = Path(__file__).resolve().parent.parent / "skills"
    if repo_skills.is_dir() and repo_skills not in dirs:
        dirs.append(repo_skills)
    return dirs


def _iter_skill_files(scan_dir: Path) -> Iterable[Path]:
    try:
        from agent.skill_utils import iter_skill_index_files  # type: ignore

        return list(iter_skill_index_files(scan_dir, "SKILL.md"))
    except Exception:
        # Defensive fallback — bounded recursive scan.
        return [p for p in scan_dir.rglob("SKILL.md") if p.is_file()]


def _parse_frontmatter(text: str) -> Tuple[Dict[str, Any], str]:
    """Reuse the existing frontmatter parser to stay byte-compatible."""
    try:
        from tools.skills_tool import _parse_frontmatter as _pf  # type: ignore

        return _pf(text)
    except Exception:
        return {}, text


def _extract_conditions(frontmatter: Dict[str, Any]) -> Dict[str, List[str]]:
    try:
        from agent.skill_utils import extract_skill_conditions  # type: ignore

        result = extract_skill_conditions(frontmatter) or {}
    except Exception:
        result = {}
    return {
        "requires_toolsets": list(result.get("requires_toolsets") or []),
        "fallback_for_toolsets": list(result.get("fallback_for_toolsets") or []),
    }


def _platform_ok(frontmatter: Dict[str, Any]) -> bool:
    try:
        from tools.skills_tool import skill_matches_platform  # type: ignore

        return bool(skill_matches_platform(frontmatter))
    except Exception:
        return True


def _disabled_names() -> Set[str]:
    try:
        from agent.skill_utils import get_disabled_skill_names  # type: ignore

        return set(get_disabled_skill_names() or [])
    except Exception:
        return set()


def _scan_filesystem_skills(
    scan_dirs: List[Tuple[Path, str]],
    *,
    disabled: Set[str],
) -> List[Dict[str, Any]]:
    """Walk each (dir, source_badge) pair and collect L0 entries.

    The first occurrence of a given name wins, mirroring the precedence
    used by the existing ``skills_list`` tool (user > external/bundled).
    """
    seen: Set[str] = set()
    out: List[Dict[str, Any]] = []
    for scan_dir, badge in scan_dirs:
        for skill_md in _iter_skill_files(scan_dir):
            try:
                text = skill_md.read_text(encoding="utf-8")[:4000]
                fm, body = _parse_frontmatter(text)
            except (UnicodeDecodeError, PermissionError, OSError) as e:
                logger.debug("Could not read skill %s: %s", skill_md, e)
                continue
            except Exception as e:  # pragma: no cover - defensive
                logger.debug("Could not parse skill %s: %s", skill_md, e)
                continue

            if not _platform_ok(fm):
                continue

            name = str(fm.get("name") or skill_md.parent.name)[:_MAX_NAME_LEN]
            if not name or name in disabled or name in seen:
                continue

            description = _truncate(fm.get("description") or "", _MAX_DESCRIPTION_LEN)
            if not description:
                for line in body.splitlines():
                    s = line.strip()
                    if s and not s.startswith("#"):
                        description = _truncate(s, _MAX_DESCRIPTION_LEN)
                        break

            conds = _extract_conditions(fm)
            seen.add(name)
            out.append({
                "id": name,
                "slug": _slugify(name),
                "name": name,
                "description": description,
                "source_badge": badge,
                "requires_toolsets": conds["requires_toolsets"],
                "fallback_for_toolsets": conds["fallback_for_toolsets"],
            })
    return out


def _scan_plugin_skills(disabled: Set[str]) -> List[Dict[str, Any]]:
    """Collect plugin-registered skills (source_badge="plugin")."""
    try:
        from hermes_cli.plugins import (  # type: ignore
            _get_disabled_plugins,
            get_plugin_manager,
        )
    except Exception:
        return []

    out: List[Dict[str, Any]] = []
    try:
        pm = get_plugin_manager()
    except Exception:
        return out
    try:
        plugin_skills = dict(getattr(pm, "_plugin_skills", {}) or {})
    except Exception:
        return out
    try:
        disabled_plugins = set(_get_disabled_plugins() or [])
    except Exception:
        disabled_plugins = set()

    for qualified, meta in plugin_skills.items():
        if not isinstance(meta, dict):
            continue
        plugin = str(meta.get("plugin") or "")
        bare = str(meta.get("bare_name") or "")
        if not plugin or not bare:
            continue
        if plugin in disabled_plugins:
            continue
        if qualified in disabled or bare in disabled:
            continue

        description = ""
        fm: Dict[str, Any] = {}
        path = meta.get("path")
        if isinstance(path, (str, Path)):
            try:
                text = Path(path).read_text(encoding="utf-8")[:4000]
                fm, _body = _parse_frontmatter(text)
                if not _platform_ok(fm):
                    continue
                description = _truncate(fm.get("description") or "", _MAX_DESCRIPTION_LEN)
            except Exception:
                description = ""

        if not description:
            description = _truncate(str(meta.get("description") or ""), _MAX_DESCRIPTION_LEN)

        conds = _extract_conditions(fm)
        out.append({
            "id": qualified,
            "slug": _slugify(bare),
            "name": bare,
            "description": description,
            "source_badge": "plugin",
            "requires_toolsets": conds["requires_toolsets"],
            "fallback_for_toolsets": conds["fallback_for_toolsets"],
        })
    return out


def _visible(entry: Dict[str, Any], registered: Set[str]) -> bool:
    """A skill is visible only when every required toolset is registered."""
    required = entry.get("requires_toolsets") or []
    return all(t in registered for t in required)


def list_skills() -> str:
    """Return the enriched L0 skill catalog as a JSON string.

    The catalog spans three sources:
      * ``bundled`` — skills shipped inside the Hermes bundle / repo seed.
      * ``user``    — skills under the user's local ``~/.hermes/skills``.
      * ``plugin``  — skills registered by loaded plugins.

    Skills whose ``requires_toolsets`` are not satisfied by the live
    toolset registry are dropped before serialization, so the desktop UI
    never has to know about hidden gates.
    """
    try:
        disabled = _disabled_names()
        registered = _registered_toolsets()

        scan_pairs: List[Tuple[Path, str]] = []
        local = _local_skills_dir()
        if local and local.is_dir():
            scan_pairs.append((local, "user"))
        for ext in _external_skills_dirs():
            if ext.is_dir():
                scan_pairs.append((ext, "user"))
        for bundled in _bundled_skills_dirs():
            scan_pairs.append((bundled, "bundled"))

        fs_entries = _scan_filesystem_skills(scan_pairs, disabled=disabled)
        plugin_entries = _scan_plugin_skills(disabled)

        all_entries = fs_entries + plugin_entries
        visible = [e for e in all_entries if _visible(e, registered)]
        visible.sort(key=lambda e: (e["source_badge"], e["name"].lower()))

        return json.dumps(
            {
                "success": True,
                "skills": visible,
                "count": len(visible),
                "sources": ["bundled", "plugin", "user"],
            },
            ensure_ascii=False,
        )
    except Exception as e:  # pragma: no cover - defensive
        logger.debug("list_skills failed: %s", e, exc_info=True)
        return tool_error(str(e), success=False)


LIST_SKILLS_SCHEMA = {
    "name": "list_skills",
    "description": (
        "Return the read-only L0 skill catalog with source badges "
        "(bundled / plugin / user) and toolset visibility metadata. "
        "Skills whose required toolsets are not registered are hidden. "
        "Use skill_view(name) to load the full SKILL.md body (L1)."
    ),
    "parameters": {
        "type": "object",
        "properties": {},
        "required": [],
    },
}


registry.register(
    name="list_skills",
    toolset="skills",
    schema=LIST_SKILLS_SCHEMA,
    handler=lambda args, **kw: list_skills(),
    emoji="📚",
)
