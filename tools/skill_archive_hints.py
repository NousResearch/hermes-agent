"""Archived skills that look like a skill being created, for the ``skill_manage`` create result.

The curator archives unused agent-created skills into ``skills/.archive/``, which takes them out of the
skill index. A later create on the same topic then starts from scratch without knowing a prior skill
existed (#58975). Create is the one moment the agent is deliberately writing a new procedure, so it is the
cheap place to say "an archived skill covers similar ground". Matching compares the new skill's own name +
description (not chat text) against each archived skill's, weighting words by how rare they are across the
whole local library so shared boilerplate ("research", "workflow") never counts as a topic. Read-only.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from agent.skill_utils import iter_skill_index_files, parse_frontmatter
from tools.skill_usage import _archive_dir, _parse_iso_timestamp, _skills_dir, load_usage

_STOPWORDS = frozenset(
    "a an and are as at be by for from how in into is it its of on or that the this to use used using "
    "when with without your you via do does any all new skill skills".split())
# Rarity-weighted cosine similarity. Calibrated on a real ~150-skill library: at 0.35 only true near-duplicates
# (the claude-code/codex/opencode family, two versions of one workflow) cross; a plain shared-word count flagged
# about half the library against unrelated skills.
_MIN_SIMILARITY = 0.35
_MAX_HINTS = 3


def _topic_words(text: str) -> set[str]:
    words = (w.rstrip("s") if len(w) > 4 else w for w in re.findall(r"[a-z0-9]+", text.lower()))
    return {w for w in words if len(w) > 2 and w not in _STOPWORDS}


def _name_and_description(skill_md: Path) -> tuple[str, str] | None:
    try:
        frontmatter, _ = parse_frontmatter(skill_md.read_text(encoding="utf-8-sig", errors="replace"))
    except OSError:
        return None
    return str(frontmatter.get("name") or skill_md.parent.name), str(frontmatter.get("description") or "")


def similar_archived_skills(name: str, description: str) -> list[dict[str, Any]]:
    """Up to ``_MAX_HINTS`` archived skills whose name + description resemble the new skill's, best first.
    ``[]`` when there is no archive or nothing is close."""
    archive = _archive_dir()
    if not archive.is_dir():
        return []
    archived = [(d, nd) for d in sorted(archive.iterdir()) if (nd := _name_and_description(d / "SKILL.md"))]
    if not archived:
        return []
    active = [nd for md in iter_skill_index_files(_skills_dir(), "SKILL.md") if (nd := _name_and_description(md))]
    docs = [_topic_words(f"{n.replace('-', ' ')} {d}") for n, d in active + [nd for _, nd in archived]]
    # Smoothed IDF over the whole local library: stays positive for a tiny archive, still discounts common words.
    df = Counter(w for words in docs for w in words)

    def weight(word: str) -> float:
        return (math.log((1 + len(docs)) / (1 + df[word])) + 1) ** 2

    wanted = _topic_words(f"{name.replace('-', ' ')} {description}")
    wanted_norm = math.sqrt(sum(weight(w) for w in wanted))
    usage, now, matches = load_usage(), datetime.now(UTC), []
    for skill_dir, (archived_name, archived_desc) in archived:
        words = _topic_words(f"{archived_name.replace('-', ' ')} {archived_desc}")
        norm = wanted_norm * math.sqrt(sum(weight(w) for w in words))
        score = sum(weight(w) for w in wanted & words) / norm if norm else 0.0
        if score < _MIN_SIMILARITY:
            continue
        record = usage.get(archived_name) or usage.get(skill_dir.name) or {}
        archived_at = _parse_iso_timestamp(record.get("archived_at")) if isinstance(record, dict) else None
        matches.append((score, {
            "name": archived_name,
            "description": archived_desc[:200],
            "path": str(Path(".archive") / skill_dir.name / "SKILL.md"),
            "archived_days_ago": (now - archived_at).days if archived_at else None,
        }))
    matches.sort(key=lambda m: (-m[0], m[1]["name"]))
    return [m[1] for m in matches[:_MAX_HINTS]]


def archived_hint_text(hints: list[dict[str, Any]]) -> str:
    names = ", ".join(
        f"'{h['name']}'" + (f" (archived {h['archived_days_ago']}d ago)" if h["archived_days_ago"] is not None else "")
        for h in hints)
    return (f"Archived skill(s) on a similar topic: {names}. They may hold lessons worth keeping — read "
            "the archived SKILL.md and verify it against the current setup before reusing anything (older "
            "archives are more likely outdated). To bring one back instead of keeping this new skill: "
            "`hermes curator restore <name>`.")
