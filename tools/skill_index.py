#!/usr/bin/env python3
"""Skill Bank index — build, query, and maintain the Hermes skill index.

Provides:
  - build_index()       — Walk all skill dirs, parse SKILL.md frontmatter,
                           extract SkillBank fields, compute embeddings.
  - merge_usage_data()  — Merge quality scores / usage into the index.
  - _index_path()       — Path to the persisted index JSON file.

Schema (version 2):
  {
    "version": 2,
    "model": "paraphrase-multilingual-MiniLM-L12-v2",
    "built_at": "ISO-8601",
    "count": N,
    "skills": [
      {
        "name": "...",           // from frontmatter or dir name
        "description": "...",    // from frontmatter
        "tags": [...],           // from frontmatter metadata.hermes.tags
        "triggers": [...],       // from body (## Triggers section or frontmatter)
        "body_preview": "...",   // first 300 chars of body
        "path": "...",           // relative path from skills dir
        "provides": [...],       // SkillBank: skills this provides
        "requires": [...],       // SkillBank: skills/prereqs this requires
        "chain_with": [...],     // SkillBank: skills commonly used together
        "principle": "...",      // SkillBank: core principle
        "when_to_apply": "...",  // SkillBank: when to use this skill
        "common_mistakes": [...],// SkillBank: common pitfalls
        "embedding": [...],      // 384-dim float vector (or [] if unavailable)
        "success_rate": 0.5,    // from quality-scores
        "total_uses": 0,        // from quality-scores
      }
    ]
  }
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

logger = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────

HERMES_HOME = Path.home() / ".hermes"
SKILLS_DIR = HERMES_HOME / "skills"
SKILL_INDEX_DIR = SKILLS_DIR / ".skill-index"
QUALITY_SCORES_FILE = SKILL_INDEX_DIR / "quality-scores.json"

# Repo-bundled skills
BUNDLED_SKILLS_DIR = Path(__file__).resolve().parent.parent / "skills"

# Model used for embeddings
EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
EMBEDDING_DIM = 384

# Skip these well-known non-skill directories
EXCLUDED_DIRS = frozenset({
    ".git", ".svn", "__pycache__", ".archive", ".hub",
    "node_modules", ".venv", "venv", "env", ".tox",
    ".skill-index", "index-cache", "learned",
})

SKILL_SUPPORT_DIRS = frozenset({
    "references", "templates", "assets", "scripts",
})

# ── Helpers ────────────────────────────────────────────────────────────────


def _index_path() -> Path:
    """Return the path to the persisted skill index JSON file."""
    SKILL_INDEX_DIR.mkdir(parents=True, exist_ok=True)
    return SKILL_INDEX_DIR / "skill-index.json"


def _parse_frontmatter(content: str) -> Tuple[Dict[str, Any], str]:
    """Parse YAML frontmatter from SKILL.md content.

    Returns (frontmatter_dict, body_string).
    """
    content = content.strip()
    if not content.startswith("---"):
        return {}, content

    # Find the closing ---
    end_idx = content.find("---", 3)
    if end_idx == -1:
        return {}, content

    fm_block = content[3:end_idx].strip()
    body = content[end_idx + 3:].strip()

    try:
        frontmatter = yaml.safe_load(fm_block)
        if not isinstance(frontmatter, dict):
            frontmatter = {}
    except yaml.YAMLError:
        frontmatter = {}

    return frontmatter, body


def _extract_triggers(frontmatter: Dict[str, Any], body: str) -> List[str]:
    """Extract trigger patterns from frontmatter or body."""
    # Check frontmatter first
    triggers = frontmatter.get("triggers", [])
    if isinstance(triggers, list) and triggers:
        return triggers

    # Try metadata.hermes.triggers
    meta = frontmatter.get("metadata", {})
    if isinstance(meta, dict):
        hermes = meta.get("hermes", {})
        if isinstance(hermes, dict):
            triggers = hermes.get("triggers", [])
            if isinstance(triggers, list) and triggers:
                return triggers

    # Scan body for a ## Triggers section
    lines = body.split("\n")
    in_triggers = False
    collected = []
    for line in lines:
        if re.match(r"^#{2,3}\s+Triggers", line.strip()):
            in_triggers = True
            continue
        if in_triggers:
            if line.strip().startswith("#"):
                break
            stripped = line.strip().strip("-*").strip()
            if stripped:
                collected.append(stripped)
    return collected


def _extract_tags(frontmatter: Dict[str, Any]) -> List[str]:
    """Extract tags from frontmatter metadata."""
    tags = frontmatter.get("tags", [])
    if isinstance(tags, list) and tags:
        return [str(t) for t in tags]

    # Try metadata.hermes.tags
    meta = frontmatter.get("metadata", {})
    if isinstance(meta, dict):
        hermes = meta.get("hermes", {})
        if isinstance(hermes, dict):
            tags = hermes.get("tags", [])
            if isinstance(tags, list):
                return [str(t) for t in tags]

    # Try metadata.tags
    if isinstance(meta, dict):
        tags = meta.get("tags", [])
        if isinstance(tags, list):
            return [str(t) for t in tags]

    return []


def _extract_skillbank_field(frontmatter: Dict[str, Any], key: str, default: Any = None) -> Any:
    """Extract a SkillBank field from frontmatter metadata.skillbank or top-level."""
    meta = frontmatter.get("metadata", {})
    if isinstance(meta, dict):
        skillbank = meta.get("skillbank", {})
        if isinstance(skillbank, dict) and key in skillbank:
            return skillbank[key]

    # Fall back to top-level (convention: provides, requires, chain_with
    # can be top-level or under hermes sub-key)
    hermes = meta.get("hermes", {})
    if isinstance(hermes, dict):
        hermes_val = hermes.get(key)
        if hermes_val is not None:
            return hermes_val

    return frontmatter.get(key, default)


def _compute_embeddings(texts: List[str]) -> List[List[float]]:
    """Compute embeddings for a list of texts.

    Returns a list of float lists (384-dim). Falls back to empty lists
    when sentence-transformers is not available.
    """
    try:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(EMBEDDING_MODEL)
        embeddings = model.encode(texts, show_progress_bar=False)
        return [emb.tolist() for emb in embeddings]
    except ImportError:
        logger.warning(
            "sentence-transformers not available; embeddings will be empty. "
            "Install with: pip install sentence-transformers"
        )
        return [[] for _ in texts]
    except Exception as exc:
        logger.warning("Embedding computation failed: %s", exc)
        return [[] for _ in texts]


# ── Index building ─────────────────────────────────────────────────────────


def _iter_skill_dirs() -> List[Path]:
    """Return all skill directories to scan."""
    dirs = []

    # Primary: user's ~/.hermes/skills/
    if SKILLS_DIR.is_dir():
        dirs.append(SKILLS_DIR)

    # Bundled: repo skills/
    if BUNDLED_SKILLS_DIR.is_dir():
        dirs.append(BUNDLED_SKILLS_DIR)

    # External: from config (via skill_utils)
    try:
        sys.path.insert(0, str(BUNDLED_SKILLS_DIR.parent))
        from agent.skill_utils import get_external_skills_dirs

        dirs.extend(get_external_skills_dirs())
    except ImportError:
        pass

    return dirs


def build_index() -> Dict[str, Any]:
    """Walk all skill directories and build a complete SkillBank index.

    Returns the full index dict (ready for JSON serialisation).
    """
    index_path = _index_path()
    old_index = {}
    if index_path.exists():
        try:
            old_index = json.loads(index_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            old_index = {}

    old_skills = {s.get("name", ""): s for s in old_index.get("skills", [])}

    skills: List[Dict[str, Any]] = []
    seen_names: set = set()
    all_descriptions: List[str] = []
    all_skill_paths: List[str] = []
    skills_needing_embedding: List[int] = []

    for base_dir in _iter_skill_dirs():
        if not base_dir.is_dir():
            continue
        for root, dirs, files in os.walk(str(base_dir), followlinks=True):
            # Prune excluded and support dirs
            root_path = Path(root)
            has_skill_md = "SKILL.md" in files
            dirs[:] = [
                d
                for d in dirs
                if d not in EXCLUDED_DIRS
                and not (has_skill_md and d in SKILL_SUPPORT_DIRS)
            ]

            if "SKILL.md" not in files:
                continue

            skill_md = root_path / "SKILL.md"
            skill_dir = skill_md.parent

            try:
                content = skill_md.read_text(encoding="utf-8")
                frontmatter, body = _parse_frontmatter(content)
            except (UnicodeDecodeError, PermissionError, OSError):
                continue
            except Exception as exc:
                logger.debug("Skipping %s: %s", skill_md, exc)
                continue

            # Determine skill name
            name = str(frontmatter.get("name", "")).strip() or skill_dir.name
            if name in seen_names:
                continue
            seen_names.add(name)

            description = str(frontmatter.get("description", "")).strip()
            if not description:
                for line in body.strip().split("\n"):
                    line = line.strip()
                    if line and not line.startswith("#"):
                        description = line
                        break

            # Relative path from first matching base dir
            try:
                rel_path = str(skill_dir.relative_to(base_dir))
            except ValueError:
                rel_path = skill_dir.name

            # Body preview (first 300 chars)
            body_preview = body[:300].strip()

            # Tags
            tags = _extract_tags(frontmatter)

            # Triggers
            triggers = _extract_triggers(frontmatter, body)

            # SkillBank fields
            provides = _extract_skillbank_field(frontmatter, "provides", [])
            if not isinstance(provides, list):
                provides = []
            requires = _extract_skillbank_field(frontmatter, "requires", [])
            if not isinstance(requires, list):
                requires = []
            chain_with = _extract_skillbank_field(frontmatter, "chain_with", [])
            if not isinstance(chain_with, list):
                chain_with = []
            principle = _extract_skillbank_field(frontmatter, "principle", "")
            if not isinstance(principle, str):
                principle = str(principle) if principle else ""
            when_to_apply = _extract_skillbank_field(frontmatter, "when_to_apply", "")
            if not isinstance(when_to_apply, str):
                when_to_apply = str(when_to_apply) if when_to_apply else ""
            common_mistakes = _extract_skillbank_field(frontmatter, "common_mistakes", [])
            if not isinstance(common_mistakes, list):
                common_mistakes = []

            # Preserve old embedding if content hasn't changed
            old_skill = old_skills.get(name, {})
            old_embedding = old_skill.get("embedding", [])
            needs_new_embedding = not old_embedding or old_skill.get("description") != description

            skill_entry = {
                "name": name,
                "description": description,
                "tags": tags,
                "triggers": triggers,
                "body_preview": body_preview,
                "path": rel_path,
                "provides": provides,
                "requires": requires,
                "chain_with": chain_with,
                "principle": principle,
                "when_to_apply": when_to_apply,
                "common_mistakes": common_mistakes,
                "embedding": [],
                "success_rate": old_skill.get("success_rate", 0.5),
                "total_uses": old_skill.get("total_uses", 0),
            }

            if needs_new_embedding:
                all_descriptions.append(description)
                skills_needing_embedding.append(len(skills))
            else:
                skill_entry["embedding"] = old_embedding

            skills.append(skill_entry)

    # Compute embeddings for new/changed skills
    if skills_needing_embedding:
        texts = [all_descriptions[i] for i in range(len(all_descriptions))]
        if texts:
            new_embeddings = _compute_embeddings(texts)
            for list_idx, skill_idx in enumerate(skills_needing_embedding):
                if list_idx < len(new_embeddings):
                    skills[skill_idx]["embedding"] = new_embeddings[list_idx]

    skills.sort(key=lambda s: s["name"])

    index = {
        "version": 2,
        "model": EMBEDDING_MODEL,
        "built_at": datetime.now(timezone.utc).isoformat(),
        "count": len(skills),
        "skills": skills,
    }

    return index


def merge_usage_data(index: Dict[str, Any]) -> Dict[str, Any]:
    """Merge quality scores and usage data into the index.

    Reads quality-scores.json and updates each skill's success_rate
    and total_uses.
    """
    if not QUALITY_SCORES_FILE.exists():
        logger.info("No quality-scores.json found; skipping usage merge")
        return index

    try:
        scores = json.loads(QUALITY_SCORES_FILE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Failed to read quality-scores.json: %s", exc)
        return index

    name_to_skill = {}
    for s in index.get("skills", []):
        name_to_skill[s["name"]] = s

    for skill_name, data in scores.items():
        if skill_name not in name_to_skill:
            continue
        if not isinstance(data, dict):
            continue

        skill = name_to_skill[skill_name]
        # success_rate from quality-scores
        total = data.get("total", 0.5)
        skill["success_rate"] = float(total) if isinstance(total, (int, float)) else 0.5

        # total_uses from quality-scores
        use_count = data.get("use_count", 0)
        skill["total_uses"] = int(use_count) if isinstance(use_count, (int, float)) else 0

    return index


def rebuild_index() -> Dict[str, Any]:
    """Convenience: build + merge + write in one call.

    Returns the written index dict.
    """
    index = build_index()
    index = merge_usage_data(index)
    path = _index_path()
    path.write_text(json.dumps(index, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Rebuilt skill index: %d skills -> %s", index["count"], path)
    return index


# ── CLI ────────────────────────────────────────────────────────────────────


def main():
    """CLI entry point: rebuild the index and print summary."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    index = rebuild_index()
    print(f"Rebuilt: {index['count']} skills")
    print(f"  version: {index['version']}")
    print(f"  model: {index['model']}")
    print(f"  built_at: {index['built_at']}")
    print(f"  path: {_index_path()}")


if __name__ == "__main__":
    main()
