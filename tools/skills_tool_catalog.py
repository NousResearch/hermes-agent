"""Filesystem catalog reads and live visibility selection for the skills tool."""

import logging

logger = logging.getLogger(__name__)


def _read_catalog(roots):
    from agent.skill_utils import (
        TIER_PROJECT, iter_project_skill_files, iter_skill_index_files, resolve_skill_catalog)
    from tools import skills_tool as st

    scanned = []
    for tier, scan_dir in roots:
        files = (iter_project_skill_files(scan_dir) if tier == TIER_PROJECT
                 else iter_skill_index_files(scan_dir, "SKILL.md"))
        for skill_md in files:
            if any(part in st._EXCLUDED_SKILL_DIRS for part in skill_md.parts):
                continue
            try:
                frontmatter, body = st._parse_frontmatter(st._read_skill_text(skill_md)[:4000])
                description = frontmatter.get("description", "")
                if not description:
                    description = next((ln for ln in map(str.strip, body.strip().split("\n"))
                                        if ln and not ln.startswith("#")), description)
                scanned.append({
                    "name": frontmatter.get("name", skill_md.parent.name)[:st.MAX_NAME_LENGTH],
                    "description": st._truncate_description(description),
                    "category": st._get_category_from_path(skill_md), "tier": tier, "root": scan_dir,
                    "path": skill_md, "_frontmatter": frontmatter})
            except (UnicodeDecodeError, PermissionError) as exc:
                logger.debug("Failed to read skill file %s: %s", skill_md, exc)
            except Exception as exc:
                logger.debug("Skipping skill at %s: failed to parse: %s", skill_md, exc, exc_info=True)
    return resolve_skill_catalog(scanned)


def _select_catalog(records, disabled, include_hidden):
    from agent.skill_utils import is_disabled_entry, resolve_skill_catalog
    from tools import skills_tool as st

    selected = []
    invalid = False
    for record in records:
        frontmatter = record["_frontmatter"]
        try:
            visible = (st.skill_matches_platform(frontmatter)
                       and st.skill_matches_environment(frontmatter)
                       and st.skill_matches_apps(frontmatter))
        except Exception as exc:
            logger.debug("Skipping skill at %s: visibility check failed: %s", record["path"], exc, exc_info=True)
            invalid = True
            continue
        row = {key: value for key, value in record.items() if key != "_frontmatter"}
        row["visible"] = visible
        selected.append(row)
    if invalid:
        # A malformed entry must not make a valid same-name entry ambiguous.
        selected = resolve_skill_catalog(selected)
    return [row for row in selected
            if (row.pop("visible") or include_hidden) and not is_disabled_entry(row, disabled)]
