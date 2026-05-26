#!/usr/bin/env python3
"""Extract skill metadata into website/src/data/skills.json for the Skills Hub page.

Two data sources:

1. Local SKILL.md files under ``skills/`` (built-in) and ``optional-skills/``
   (official optional). These give us full metadata — overview prose, version,
   license, env vars, commands — that the unified index doesn't carry.

2. The unified Hermes Skills Index at ``website/static/api/skills-index.json``,
   built twice daily by ``scripts/build_skills_index.py`` (workflow
   ``.github/workflows/skills-index.yml``). Covers skills.sh, ClawHub, browse.sh,
   LobeHub, Claude Marketplace, well-known endpoints, and the GitHub taps
   (openai/skills, anthropics/skills, huggingface/skills, VoltAgent, etc.).

Legacy fallback: if the unified index is missing AND ``skills/index-cache/``
contains pre-baked JSON dumps, we read those (preserves behaviour from before
the unified index existed).
"""

import json
import os
from collections import Counter

import yaml

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LOCAL_SKILL_DIRS = [
    ("skills", "built-in"),
    ("optional-skills", "optional"),
]
UNIFIED_INDEX_PATH = os.path.join(REPO_ROOT, "website", "static", "api", "skills-index.json")
LEGACY_INDEX_CACHE_DIR = os.path.join(REPO_ROOT, "skills", "index-cache")
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

# Map the source ids the unified index emits to the friendly labels the
# Skills Hub UI uses. Keep these in sync with the SOURCE_CONFIG dict in
# website/src/pages/skills/index.tsx.
UNIFIED_SOURCE_LABELS = {
    "official": "official",   # treated as our "optional" tier in the UI
    "skills.sh": "skills.sh",
    "skills-sh": "skills.sh",
    "clawhub": "ClawHub",
    "browse-sh": "browse.sh",
    "lobehub": "LobeHub",
    "claude-marketplace": "Claude Marketplace",
    "well-known": "Well-Known",
    "github": "GitHub",  # default for non-named GitHub taps
}

# Repo-specific labels for the unified index's "github" source. Lets us
# call out the well-known taps with their vendor name instead of a generic
# "GitHub" pill. Match is checked against the leading "owner/repo/" prefix
# of the identifier.
GITHUB_TAP_LABELS = {
    "openai/skills": "OpenAI",
    "anthropics/skills": "Anthropic",
    "huggingface/skills": "HuggingFace",
    "VoltAgent/awesome-agent-skills": "VoltAgent",
    "garrytan/gstack": "gstack",
    "MiniMax-AI/cli": "MiniMax",
}

# Legacy filename -> label mapping for the deprecated skills/index-cache/
# fallback. Used only when website/static/api/skills-index.json is absent.
LEGACY_SOURCE_LABELS = {
    "anthropics_skills": "Anthropic",
    "openai_skills": "OpenAI",
    "claude_marketplace": "Claude Marketplace",
    "lobehub": "LobeHub",
}


def _extract_overview(body: str) -> str:
    """Pull the first non-heading paragraph from a SKILL.md body."""
    if not body:
        return ""
    paragraphs = [p.strip() for p in body.split("\n\n") if p.strip()]
    for p in paragraphs[:6]:
        if p.startswith("#"):
            lines = [ln for ln in p.split("\n") if ln.strip() and not ln.lstrip().startswith("#")]
            if lines:
                p = "\n".join(lines).strip()
            else:
                continue
        if p.startswith(":::"):
            continue
        if p.startswith("```") or p.startswith("~~~"):
            continue
        if len(p) > 500:
            cut = p[:500]
            last_period = cut.rfind(". ")
            if last_period > 200:
                p = cut[: last_period + 1]
            else:
                p = cut.rstrip() + "…"
        return p
    return ""


def _docs_page_path(rel_dir: str, source_label: str) -> str:
    """Compute the per-skill docs-site URL slug for a given SKILL.md location.

    Mirrors the slug logic in website/scripts/generate-skill-docs.py:
      bundled  + skills/<cat>/<slug>/SKILL.md          -> bundled/<cat>/<cat>-<slug>
      bundled  + skills/<cat>/<sub>/<slug>/SKILL.md    -> bundled/<cat>/<cat>-<sub>-<slug>
      optional + optional-skills/<cat>/<slug>/SKILL.md -> optional/<cat>/<cat>-<slug>
    """
    parts = [p for p in rel_dir.split(os.sep) if p]
    if not parts:
        return ""
    source_dir = "bundled" if source_label == "built-in" else "optional"
    if len(parts) == 1:
        category, slug = parts[0], parts[0]
        return f"{source_dir}/{category}/{category}-{slug}"
    if len(parts) == 2:
        category, slug = parts
        return f"{source_dir}/{category}/{category}-{slug}"
    if len(parts) == 3:
        category, sub, slug = parts
        return f"{source_dir}/{category}/{category}-{sub}-{slug}"
    return ""


def _install_command(source: str, identifier: str, name: str) -> str:
    """Build the ``hermes skills install …`` command for a unified-index entry.

    These show up in the SkillCard panel so users can copy-paste them. We try
    to use the most idiomatic identifier per source.
    """
    if not identifier:
        return f"hermes skills install {name}"
    src = source.lower()
    if src in {"official", "built-in", "optional"}:
        # OptionalSkillSource emits identifiers like "official/security/1password"
        return f"hermes skills install {identifier}"
    if src in {"skills.sh", "skills-sh"}:
        # Already wrapped as "skills-sh/owner/repo/skill" by the source
        return f"hermes skills install {identifier}"
    if src == "clawhub":
        return f"hermes skills install clawhub/{identifier}"
    if src == "browse-sh":
        # Identifier already includes the "browse-sh/" prefix from BrowseShSource
        return f"hermes skills install {identifier}"
    if src == "lobehub":
        return f"hermes skills install {identifier}"
    if src == "claude-marketplace":
        return f"hermes skills install {identifier}"
    if src == "github":
        return f"hermes skills install {identifier}"
    if src == "well-known":
        return f"hermes skills install {identifier}"
    return f"hermes skills install {identifier}"


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
            with open(skill_path, encoding="utf-8") as f:
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

            body = parts[2].strip()
            overview = _extract_overview(body)

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

            prereq = fm.get("prerequisites") or {}
            env_vars = []
            commands = []
            if isinstance(prereq, dict):
                ev = prereq.get("env_vars")
                if isinstance(ev, list):
                    env_vars = [str(x) for x in ev if x]
                elif isinstance(ev, str) and ev.strip():
                    env_vars = [ev.strip()]
                cmds = prereq.get("commands")
                if isinstance(cmds, list):
                    commands = [str(x) for x in cmds if x]
                elif isinstance(cmds, str) and cmds.strip():
                    commands = [cmds.strip()]

            skills.append({
                "name": fm.get("name", os.path.basename(root)),
                "description": fm.get("description", ""),
                "overview": overview,
                "category": category,
                "categoryLabel": CATEGORY_LABELS.get(category, category.replace("-", " ").title()),
                "source": source_label,
                "tags": tags or [],
                "platforms": fm.get("platforms", []),
                "author": fm.get("author", ""),
                "version": fm.get("version", ""),
                "license": fm.get("license", ""),
                "envVars": env_vars,
                "commands": commands,
                "docsPath": _docs_page_path(rel, source_label),
            })

    return skills


def _label_for_github_identifier(identifier: str) -> str:
    """Return a friendly source label for a unified-index 'github' entry."""
    if not identifier:
        return "GitHub"
    for prefix, label in GITHUB_TAP_LABELS.items():
        if identifier.startswith(prefix + "/") or identifier == prefix:
            return label
    return "GitHub"


def extract_unified_index_skills():
    """Read website/static/api/skills-index.json — the canonical multi-source index."""
    if not os.path.isfile(UNIFIED_INDEX_PATH):
        return None

    try:
        with open(UNIFIED_INDEX_PATH, encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        print(f"[extract-skills] Failed to read unified index: {e}")
        return None

    if not isinstance(data, dict) or "skills" not in data:
        return None

    out = []
    for entry in data.get("skills", []):
        if not isinstance(entry, dict):
            continue
        source_id = (entry.get("source") or "").lower()
        identifier = entry.get("identifier", "") or ""
        name = entry.get("name") or identifier.split("/")[-1] or "unknown"
        description = (entry.get("description") or "").split("\n")[0]
        if len(description) > 280:
            description = description[:277] + "…"
        tags = entry.get("tags", []) or []
        if not isinstance(tags, list):
            tags = []

        # Skip official entries here — extract_local_skills() already covered
        # those from optional-skills/ with full metadata (overview, version, etc.).
        if source_id == "official":
            continue

        # Map source id -> display label
        if source_id == "github":
            source_label = _label_for_github_identifier(identifier)
        else:
            source_label = UNIFIED_SOURCE_LABELS.get(source_id, source_id or "community")

        # Guess a category from tags so the UI's category filter has a chance.
        category = _guess_category(tags)
        extra = entry.get("extra", {}) or {}

        # Author hint from extras when available (skills.sh has installs;
        # clawhub doesn't expose author).
        author = ""
        if source_id in {"skills.sh", "skills-sh"}:
            repo = entry.get("repo", "")
            if repo:
                author = repo.split("/")[0]

        install_cmd = _install_command(source_id, identifier, name)

        out.append({
            "name": name,
            "description": description,
            "overview": "",
            "category": category,
            "categoryLabel": "",  # filled in _consolidate_small_categories
            "source": source_label,
            "tags": tags,
            "platforms": [],
            "author": author,
            "version": "",
            "license": "",
            "envVars": [],
            "commands": [],
            "docsPath": "",
            "identifier": identifier,
            "installCmd": install_cmd,
        })

    return out


def extract_legacy_cache_skills():
    """Read the deprecated skills/index-cache/ snapshots — fallback only."""
    skills = []

    if not os.path.isdir(LEGACY_INDEX_CACHE_DIR):
        return skills

    for filename in os.listdir(LEGACY_INDEX_CACHE_DIR):
        if not filename.endswith(".json"):
            continue

        filepath = os.path.join(LEGACY_INDEX_CACHE_DIR, filename)
        try:
            with open(filepath, encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue

        stem = filename.replace(".json", "")
        source_label = "community"
        for key, label in LEGACY_SOURCE_LABELS.items():
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
                    "categoryLabel": "",
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
        if not isinstance(tag, str):
            continue
        cat = TAG_TO_CATEGORY.get(tag.lower())
        if cat:
            return cat
    first = tags[0] if isinstance(tags[0], str) else ""
    return first.lower().replace(" ", "-") if first else "uncategorized"


MIN_CATEGORY_SIZE = 4


def _consolidate_small_categories(skills: list) -> list:
    for s in skills:
        if s["category"] in {"uncategorized", ""}:
            s["category"] = "other"
            s["categoryLabel"] = "Other"

    counts = Counter(s["category"] for s in skills)
    small_cats = {cat for cat, n in counts.items() if n < MIN_CATEGORY_SIZE}

    for s in skills:
        if s["category"] in small_cats:
            s["category"] = "other"
            s["categoryLabel"] = "Other"
        elif not s["categoryLabel"]:
            s["categoryLabel"] = CATEGORY_LABELS.get(
                s["category"],
                s["category"].replace("-", " ").title() if s["category"] else "Uncategorized",
            )

    return skills


def main():
    local = extract_local_skills()

    unified = extract_unified_index_skills()
    if unified is not None:
        external = unified
        external_source = "unified index"
    else:
        external = extract_legacy_cache_skills()
        external_source = "legacy index-cache"
        print(
            f"[extract-skills] WARNING: unified index not found at "
            f"{UNIFIED_INDEX_PATH}; falling back to {external_source}. "
            f"Run `python3 scripts/build_skills_index.py` to refresh."
        )

    all_skills = _consolidate_small_categories(local + external)

    source_order = {"built-in": 0, "optional": 1}
    all_skills.sort(key=lambda s: (
        source_order.get(s["source"], 2),
        1 if s["category"] == "other" else 0,
        s["category"],
        s["name"],
    ))

    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
    with open(OUTPUT, "w", encoding="utf-8") as f:
        json.dump(all_skills, f, indent=2)

    print(f"Extracted {len(all_skills)} skills to {OUTPUT}")
    print(f"  {len(local)} local ({sum(1 for s in local if s['source'] == 'built-in')} built-in, "
          f"{sum(1 for s in local if s['source'] == 'optional')} optional)")
    print(f"  {len(external)} from {external_source}")

    # Breakdown by source
    by_source = Counter(s["source"] for s in all_skills)
    print("By source:")
    for src, count in by_source.most_common():
        print(f"  {src}: {count}")


if __name__ == "__main__":
    main()
