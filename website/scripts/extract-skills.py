#!/usr/bin/env python3
"""Extract skill metadata from SKILL.md files and index caches into JSON."""

import json
import os
from collections import Counter

import yaml

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LOCAL_SKILL_DIRS = [
    ("skills", "built-in"),
    ("optional-skills", "optional"),
]
INDEX_CACHE_DIR = os.path.join(REPO_ROOT, "skills", "index-cache")
OUTPUT = os.path.join(REPO_ROOT, "website", "src", "data", "skills.json")

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

SOURCE_LABELS = {
    "anthropics_skills": "Anthropic",
    "openai_skills": "OpenAI",
    "claude_marketplace": "Claude Marketplace",
    "lobehub": "LobeHub",
}


def extract_local_skills():
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

            rel = os.path.relpath(root, base_path)
            category = rel.split(os.sep)[0]

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

        stem = filename.replace(".json", "")
        source_label = "community"
        for key, label in SOURCE_LABELS.items():
            if key in stem:
                source_label = label
                break

        if isinstance(data, dict) and "agents" in data:
            for agent in data["agents"]:
                if not isinstance(agent, dict):
                    continue
                skills.append({
                    "name": agent.get("identifier", agent.get("meta", {}).get("title", "unknown")),
                    "description": (agent.get("meta", {}).get("description", "") or "").split("\n")[0][:200],
                    "category": _guess_category(agent.get("meta", {}).get("tags", [])),
                    "categoryLabel": "",  # filled below
                    "source": source_label,
                    "tags": agent.get("meta", {}).get("tags", []),
                    "platforms": [],
                    "author": agent.get("author", ""),
                    "version": "",
                })
            continue

        if isinstance(data, list):
            for entry in data:
                if not isinstance(entry, dict) or not entry.get("name"):
                    continue
                if "skills" in entry and isinstance(entry["skills"], list):
                    continue
                skills.append({
                    "name": entry.get("name", ""),
                    "description": entry.get("description", ""),
                    "category": "uncategorized",
                    "categoryLabel": "",
                    "source": source_label,
                    "tags": entry.get("tags", []),
                    "platforms": [],
                    "author": "",
                    "version": "",
                })

    for s in skills:
        if not s["categoryLabel"]:
            s["categoryLabel"] = CATEGORY_LABELS.get(
                s["category"],
                s["category"].replace("-", " ").title() if s["category"] else "Uncategorized",
            )

    return skills


TAG_TO_CATEGORY = {}
for _cat, _tags in {
    "software-development": [
        "programming", "code", "coding", "software-development",
        "frontend-development", "backend-development", "web-development",
        "react", "python", "typescript", "java", "rust",
    ],
    "creative": ["writing", "design", "creative", "art", "image-generation"],
    "research": ["education", "academic", "research"],
    "social-media": ["marketing", "seo", "social-media"],
    "productivity": ["productivity", "business"],
    "data-science": ["data", "data-science"],
    "mlops": ["machine-learning", "deep-learning"],
    "devops": ["devops"],
    "gaming": ["gaming", "game", "game-development"],
    "media": ["music", "media", "video"],
    "health": ["health", "fitness"],
    "translation": ["translation", "language-learning"],
    "security": ["security", "cybersecurity"],
}.items():
    for _t in _tags:
        TAG_TO_CATEGORY[_t] = _cat


def _guess_category(tags: list) -> str:
    if not tags:
        return "uncategorized"
    for tag in tags:
        cat = TAG_TO_CATEGORY.get(tag.lower())
        if cat:
            return cat
    return tags[0].lower().replace(" ", "-")


MIN_CATEGORY_SIZE = 4


def _consolidate_small_categories(skills: list) -> list:
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

    all_skills = _consolidate_small_categories(local + external)

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
