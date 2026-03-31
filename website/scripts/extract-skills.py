#!/usr/bin/env python3
"""Extract skill metadata from SKILL.md files and index caches into JSON for the dashboard.

Sources:
  1. skills/         — built-in skills (shipped with hermes-agent)
  2. optional-skills/ — optional skills (shipped but not active by default)
  3. skills/index-cache/*.json — cached indexes from external registries

Usage:
    python3 website/scripts/extract-skills.py

Outputs:
    website/src/data/skills.json
"""

import json
import os
import sys

import yaml

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LOCAL_SKILL_DIRS = [
    ("skills", "built-in"),
    ("optional-skills", "optional"),
]
INDEX_CACHE_DIR = os.path.join(REPO_ROOT, "skills", "index-cache")
OUTPUT = os.path.join(REPO_ROOT, "website", "src", "data", "skills.json")

# Friendly display names for categories
CATEGORY_LABELS = {
    "apple": "Apple",
    "autonomous-ai-agents": "AI Agents",
    "blockchain": "Blockchain",
    "communication": "Communication",
    "creative": "Creative",
    "data-science": "Data Science",
    "devops": "DevOps",
    "dogfood": "Dogfood",
    "domain": "Domain",
    "email": "Email",
    "feeds": "Feeds",
    "gaming": "Gaming",
    "gifs": "GIFs",
    "github": "GitHub",
    "health": "Health",
    "inference-sh": "Inference",
    "leisure": "Leisure",
    "mcp": "MCP",
    "media": "Media",
    "migration": "Migration",
    "mlops": "MLOps",
    "note-taking": "Note-Taking",
    "productivity": "Productivity",
    "red-teaming": "Red Teaming",
    "research": "Research",
    "security": "Security",
    "smart-home": "Smart Home",
    "social-media": "Social Media",
    "software-development": "Software Dev",
    "translation": "Translation",
    "other": "Other",
}

# Map external source identifiers to friendly labels
SOURCE_LABELS = {
    "anthropics_skills": "Anthropic",
    "openai_skills": "OpenAI",
    "claude_marketplace": "Claude Marketplace",
    "lobehub": "LobeHub",
}


def extract_local_skills():
    """Parse SKILL.md frontmatter from local skill directories."""
    skills = []

    for base_dir, source_label in LOCAL_SKILL_DIRS:
        base_path = os.path.join(REPO_ROOT, base_dir)
        if not os.path.isdir(base_path):
            continue

        for root, _dirs, files in os.walk(base_path):
            if "SKILL.md" not in files:
                continue

            skill_path = os.path.join(root, "SKILL.md")
            with open(skill_path) as f:
                content = f.read()

            if not content.startswith("---"):
                continue

            parts = content.split("---", 2)
            if len(parts) < 3:
                continue

            try:
                fm = yaml.safe_load(parts[1])
            except yaml.YAMLError:
                continue

            if not fm or not isinstance(fm, dict):
                continue

            # Category from directory structure
            rel = os.path.relpath(root, base_path)
            category = rel.split(os.sep)[0]

            # Tags — can be in metadata.hermes.tags or top-level tags
            tags = []
            metadata = fm.get("metadata")
            if isinstance(metadata, dict):
                hermes_meta = metadata.get("hermes", {})
                if isinstance(hermes_meta, dict):
                    tags = hermes_meta.get("tags", [])
            if not tags:
                tags = fm.get("tags", [])
            if isinstance(tags, str):
                tags = [tags]

            skills.append({
                "name": fm.get("name", os.path.basename(root)),
                "description": fm.get("description", ""),
                "category": category,
                "categoryLabel": CATEGORY_LABELS.get(category, category.replace("-", " ").title()),
                "source": source_label,
                "tags": tags or [],
                "platforms": fm.get("platforms", []),
                "author": fm.get("author", ""),
                "version": fm.get("version", ""),
            })

    return skills


def extract_cached_index_skills():
    """Parse skills from index cache JSON files (external registries)."""
    skills = []

    if not os.path.isdir(INDEX_CACHE_DIR):
        return skills

    for filename in os.listdir(INDEX_CACHE_DIR):
        if not filename.endswith(".json"):
            continue

        filepath = os.path.join(INDEX_CACHE_DIR, filename)
        try:
            with open(filepath) as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue

        # Determine source label from filename
        stem = filename.replace(".json", "")
        source_label = "community"
        for key, label in SOURCE_LABELS.items():
            if key in stem:
                source_label = label
                break

        # LobeHub format: {"agents": [...], "tags": [...]}
        if isinstance(data, dict) and "agents" in data:
            for agent in data["agents"]:
                if not isinstance(agent, dict):
                    continue
                skills.append({
                    "name": agent.get("identifier", agent.get("meta", {}).get("title", "unknown")),
                    "description": _first_line(agent.get("meta", {}).get("description", "")),
                    "category": _guess_category(agent.get("meta", {}).get("tags", [])),
                    "categoryLabel": "",  # filled below
                    "source": source_label,
                    "tags": agent.get("meta", {}).get("tags", []),
                    "platforms": [],
                    "author": agent.get("author", ""),
                    "version": "",
                })
            continue

        # SkillMeta list format (Anthropic, OpenAI caches)
        if isinstance(data, list):
            for entry in data:
                if not isinstance(entry, dict) or not entry.get("name"):
                    continue
                # Skip meta-packages (claude marketplace bundles)
                if "skills" in entry and isinstance(entry["skills"], list):
                    continue
                skills.append({
                    "name": entry.get("name", ""),
                    "description": entry.get("description", ""),
                    "category": _category_from_identifier(entry.get("identifier", "")),
                    "categoryLabel": "",
                    "source": source_label,
                    "tags": entry.get("tags", []),
                    "platforms": [],
                    "author": "",
                    "version": "",
                })

    # Fill in category labels
    for s in skills:
        if not s["categoryLabel"]:
            s["categoryLabel"] = CATEGORY_LABELS.get(
                s["category"],
                s["category"].replace("-", " ").title() if s["category"] else "Uncategorized",
            )

    return skills


def _first_line(text: str) -> str:
    """Return the first sentence or first 200 chars."""
    if not text:
        return ""
    line = text.split("\n")[0].strip()
    return line[:200]


def _guess_category(tags: list) -> str:
    """Map LobeHub tags to our category names (best effort)."""
    if not tags:
        return "uncategorized"
    tag_lower = [t.lower() for t in tags]
    mapping = {
        "programming": "software-development",
        "code": "software-development",
        "coding": "software-development",
        "software-development": "software-development",
        "frontend-development": "software-development",
        "backend-development": "software-development",
        "web-development": "software-development",
        "react": "software-development",
        "python": "software-development",
        "typescript": "software-development",
        "java": "software-development",
        "rust": "software-development",
        "writing": "creative",
        "design": "creative",
        "creative": "creative",
        "art": "creative",
        "image-generation": "creative",
        "education": "research",
        "academic": "research",
        "research": "research",
        "marketing": "social-media",
        "seo": "social-media",
        "social-media": "social-media",
        "productivity": "productivity",
        "business": "productivity",
        "data": "data-science",
        "data-science": "data-science",
        "machine-learning": "mlops",
        "deep-learning": "mlops",
        "devops": "devops",
        "gaming": "gaming",
        "game": "gaming",
        "game-development": "gaming",
        "music": "media",
        "media": "media",
        "video": "media",
        "health": "health",
        "fitness": "health",
        "translation": "translation",
        "language-learning": "translation",
        "security": "security",
        "cybersecurity": "security",
    }
    for tag in tag_lower:
        if tag in mapping:
            return mapping[tag]
    return tags[0].lower().replace(" ", "-") if tags else "uncategorized"


def _category_from_identifier(identifier: str) -> str:
    """Extract a rough category from source identifier like 'anthropics/skills/skills/name'.

    For external registries we can't reliably infer a category from the path,
    so we return 'uncategorized' and let the consolidation step handle it.
    """
    return "uncategorized"


# Minimum number of skills a category must have to be shown independently.
# Categories below this threshold are merged into "Other".
MIN_CATEGORY_SIZE = 4


def _consolidate_small_categories(skills: list) -> list:
    """Merge categories with fewer than MIN_CATEGORY_SIZE skills into 'other'."""
    from collections import Counter

    # First, fold "uncategorized" into "other"
    for s in skills:
        if s["category"] in ("uncategorized", ""):
            s["category"] = "other"
            s["categoryLabel"] = "Other"

    counts = Counter(s["category"] for s in skills)
    small_cats = {cat for cat, n in counts.items() if n < MIN_CATEGORY_SIZE}

    for s in skills:
        if s["category"] in small_cats:
            s["category"] = "other"
            s["categoryLabel"] = "Other"

    return skills


def main():
    local = extract_local_skills()
    external = extract_cached_index_skills()

    all_skills = local + external

    # Roll small / one-off categories into "Other"
    all_skills = _consolidate_small_categories(all_skills)

    # Sort: local first (built-in, optional), then external; "other" last within each source
    source_order = {"built-in": 0, "optional": 1}
    all_skills.sort(key=lambda s: (
        source_order.get(s["source"], 2),
        1 if s["category"] == "other" else 0,
        s["category"],
        s["name"],
    ))

    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
    with open(OUTPUT, "w") as f:
        json.dump(all_skills, f, indent=2)

    print(f"Extracted {len(all_skills)} skills to {OUTPUT}")
    print(f"  {len(local)} local ({sum(1 for s in local if s['source'] == 'built-in')} built-in, "
          f"{sum(1 for s in local if s['source'] == 'optional')} optional)")
    print(f"  {len(external)} from external indexes")


if __name__ == "__main__":
    main()
