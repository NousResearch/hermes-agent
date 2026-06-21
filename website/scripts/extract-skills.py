#!/usr/bin/env python3
"""Extract skill metadata into website/static/api/skills.json for the Skills Hub page.

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
from datetime import datetime, timezone

import yaml

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LOCAL_SKILL_DIRS = [
    ("skills", "built-in"),
    ("optional-skills", "optional"),
]
UNIFIED_INDEX_PATH = os.path.join(REPO_ROOT, "website", "static", "api", "skills-index.json")
LEGACY_INDEX_CACHE_DIR = os.path.join(REPO_ROOT, "skills", "index-cache")
# Output to static/api/ so the file is CDN-served at /api/skills.json
# rather than bundled into the page's JS chunk. At 50k+ skills the
# bundled payload was ~26 MB; lazy-fetch keeps the initial page load
# fast and shrinks the JS chunk back to a few hundred KB.
OUTPUT = os.path.join(REPO_ROOT, "website", "static", "api", "skills.json")
META_OUTPUT = os.path.join(REPO_ROOT, "website", "static", "api", "skills-meta.json")

CATEGORY_LABELS = {
    "apple": "Apple",
    "autonomous-ai-agents": "AI Agents",
    "blockchain": "Blockchain",
    "communication": "Communication",
    "creative": "Creative",
    "data-science": "Data Science",
    "devops": "DevOps",
    "dogfood": "Dogfood",
    "domain": "Business & Finance",
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
    "NVIDIA/skills": "NVIDIA",
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


def _source_url(source: str, identifier: str, extra: dict) -> str:
    """Best-effort clickable URL to the skill's origin (repo / detail page).

    Community skills have no generated docs page, so without this the
    expanded card on the Skills Hub gives users nowhere to go to read the
    actual SKILL.md before installing. We prefer an explicit URL the source
    adapter already collected (``extra.detail_url`` / ``extra.repo_url``),
    then fall back to synthesizing one from the identifier shape.
    """
    extra = extra or {}
    for key in ("detail_url", "source_url", "repo_url", "url", "index_url"):
        val = extra.get(key)
        if isinstance(val, str) and val.startswith("http"):
            return val

    if not identifier:
        return ""
    src = (source or "").lower()

    # GitHub-backed taps (openai/anthropic/nvidia/hf/gstack/VoltAgent/...):
    # identifier is "owner/repo/<path...>" — link to the directory on GitHub.
    if src in {"github", "openai", "anthropic", "huggingface", "nvidia",
               "gstack", "voltagent", "minimax", "claude marketplace",
               "claude-marketplace"}:
        parts = [p for p in identifier.split("/") if p]
        if len(parts) >= 2:
            owner, repo = parts[0], parts[1]
            sub = "/".join(parts[2:])
            base = f"https://github.com/{owner}/{repo}"
            return f"{base}/tree/main/{sub}" if sub else base
        return ""

    if src == "clawhub":
        # identifier is a bare slug (the "clawhub/" prefix is added at install time)
        slug = identifier[len("clawhub/"):] if identifier.startswith("clawhub/") else identifier
        return f"https://clawhub.ai/skills/{slug}"

    if src in {"skills.sh", "skills-sh"}:
        # "skills-sh/owner/repo/skill" -> the skills.sh detail page
        rest = identifier[len("skills-sh/"):] if identifier.startswith("skills-sh/") else identifier
        return f"https://skills.sh/skills/{rest}"

    if src == "lobehub":
        slug = identifier[len("lobehub/"):] if identifier.startswith("lobehub/") else identifier
        return f"https://lobehub.com/agent/{slug}"

    if src in {"browse.sh", "browse-sh"}:
        # "browse-sh/<hostname>/<task-id>" -> browse.sh task page
        rest = identifier[len("browse-sh/"):] if identifier.startswith("browse-sh/") else identifier
        return f"https://browse.sh/skills/{rest}"

    return ""


def extract_local_skills():
    skills = []

    for base_dir, source_label in LOCAL_SKILL_DIRS:
        base_path = os.path.join(REPO_ROOT, base_dir)
        if not os.path.isdir(base_path):
            continue

        for root, dirs, files in os.walk(base_path):
            dirs[:] = [d for d in dirs if d != ".archive"]
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
    """Read website/static/api/skills-index.json — the canonical multi-source index.

    Returns ``(skills, meta)`` where ``meta`` carries the index's
    ``generated_at`` timestamp and total count so the Skills Hub page can
    show a "Last refreshed …" badge. Returns ``(None, None)`` when the
    index file is absent or malformed (caller falls back to the legacy
    cache).
    """
    if not os.path.isfile(UNIFIED_INDEX_PATH):
        return None, None

    try:
        with open(UNIFIED_INDEX_PATH, encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        print(f"[extract-skills] Failed to read unified index: {e}")
        return None, None

    if not isinstance(data, dict) or "skills" not in data:
        return None, None

    meta = {
        "indexGeneratedAt": data.get("generated_at", ""),
        "indexSkillCount": data.get("skill_count", 0),
        "indexVersion": data.get("version", 0),
    }

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

        # A skills.sh.json grouping sidecar (if the tap ships one) gives us a
        # real, human-readable category — prefer it over the tag heuristic.
        # extra["category"] holds the grouping title, e.g. "Inference AI".
        sidecar_category = extra.get("category") if isinstance(extra, dict) else None
        category_label_override = ""
        if isinstance(sidecar_category, str) and sidecar_category.strip():
            category_label_override = sidecar_category.strip()
            category = category_label_override.lower().replace(" ", "-")

        # Author hint from extras when available (skills.sh has installs;
        # clawhub doesn't expose author).
        author = ""
        if source_id in {"skills.sh", "skills-sh"}:
            repo = entry.get("repo", "")
            if repo:
                author = repo.split("/")[0]

        install_cmd = _install_command(source_id, identifier, name)
        source_url = _source_url(source_id, identifier, extra)

        out.append({
            "name": name,
            "description": description,
            "overview": "",
            "category": category,
            "categoryLabel": category_label_override,  # set from sidecar, else filled in _consolidate_small_categories
            "fixedCategory": bool(category_label_override),  # sidecar categories are exempt from small-cat collapse
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
            "sourceUrl": source_url,
        })

    return out, meta


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
        "react", "python", "typescript", "java", "rust", "cli",
        "developer-tools", "development", "api", "database", "debugging",
        "documentation", "testing", "test", "architecture",
    ],
    "autonomous-ai-agents": [
        "ai", "agent", "agents", "ai-agent", "ai-agents", "agentic",
        "agentic-ai", "ai-assistant", "assistant", "multi-agent",
        "autonomous", "llm", "rag", "prompt", "prompts", "a2a", "acp",
    ],
    "creative": [
        "writing", "design", "creative", "art", "image-generation",
        "image", "content", "video-editing", "content-creation",
    ],
    "research": ["education", "academic", "academic-writing", "research", "knowledge"],
    "social-media": ["marketing", "seo", "social-media", "advertising", "creator"],
    "productivity": [
        "productivity", "business", "automation", "calendar", "email",
        "document", "documents", "office", "notes", "note-taking",
        "collaboration", "workflow", "crm",
    ],
    "data-science": ["data", "data-science", "analytics", "analysis", "visualization"],
    "mlops": ["machine-learning", "deep-learning", "mlops", "training", "fine-tuning"],
    "devops": ["devops", "docker", "kubernetes", "infrastructure", "deployment", "monitoring", "ci-cd"],
    "gaming": ["gaming", "game", "game-development"],
    "media": ["music", "media", "video", "audio", "podcast", "youtube"],
    "health": ["health", "fitness", "medical", "wellness"],
    "translation": ["translation", "language-learning", "i18n", "localization"],
    "security": ["security", "cybersecurity", "auth", "compliance", "audit", "privacy"],
    "blockchain": [
        "blockchain", "crypto", "cryptocurrency", "defi", "web3",
        "bitcoin", "ethereum", "nft", "trading", "arbitrage",
    ],
    "communication": ["communication", "chat", "messaging", "slack", "discord"],
    "domain": [
        "finance", "accounting", "banking", "ecommerce", "e-commerce",
        "shopping", "travel", "booking", "real-estate", "legal",
        "government", "b2b", "b2b-sales", "entrepreneur", "budget",
    ],
}.items():
    for _t in _tags:
        TAG_TO_CATEGORY[_t] = _cat


def _guess_category(tags: list) -> str:
    """Map a skill's tags to a curated category, or 'uncategorized'.

    Previously this fell back to ``tags[0]`` verbatim, which produced
    hundreds of junk one-off "categories" in the sidebar (e.g.
    "Doramagic Crystal", "0.10.7 Dev", "Ap2") — version strings, brand
    names, and tag noise. We now ONLY accept categories that map to a
    known curated bucket; everything else becomes "uncategorized", which
    _consolidate_small_categories folds into "Other". Sidecar-declared
    categories (skills.sh groupings) bypass this entirely via fixedCategory.
    """
    if not tags:
        return "uncategorized"
    for tag in tags:
        if not isinstance(tag, str):
            continue
        cat = TAG_TO_CATEGORY.get(tag.lower())
        if cat:
            return cat
        # Also accept a tag that's already a known curated category key
        # (e.g. a skill tagged literally "security" or "devops").
        normalized = tag.lower().replace(" ", "-")
        if normalized in CATEGORY_LABELS and normalized != "other":
            return normalized
    return "uncategorized"


MIN_CATEGORY_SIZE = 4


def _consolidate_small_categories(skills: list) -> list:
    for s in skills:
        if s["category"] in {"uncategorized", ""}:
            s["category"] = "other"
            s["categoryLabel"] = "Other"

    # Skills with a sidecar-declared category (skills.sh.json grouping) keep
    # their category even if it's the only skill in it — the tap explicitly
    # chose that label, so it's not a heuristic guess to collapse away.
    counts = Counter(
        s["category"] for s in skills if not s.get("fixedCategory")
    )
    small_cats = {cat for cat, n in counts.items() if n < MIN_CATEGORY_SIZE}

    for s in skills:
        if s.get("fixedCategory"):
            continue
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

    unified, index_meta = extract_unified_index_skills()
    if unified is not None:
        external = unified
        external_source = "unified index"
    else:
        external = extract_legacy_cache_skills()
        external_source = "legacy index-cache"
        index_meta = None
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
        # Minified — file is served over the wire, not read by humans.
        # At 50k+ skills the indented version was ~30% larger.
        json.dump(all_skills, f, separators=(",", ":"), ensure_ascii=False)

    # Sidecar meta file so the page can render a "Last refreshed" badge
    # without changing the shape of skills.json.
    by_source = Counter(s["source"] for s in all_skills)
    meta = {
        "extractedAt": datetime.now(timezone.utc).isoformat(),
        "totalSkills": len(all_skills),
        "localSkills": len(local),
        "externalSkills": len(external),
        "externalSource": external_source,
        "bySource": dict(by_source.most_common()),
    }
    if index_meta:
        meta.update(index_meta)
    with open(META_OUTPUT, "w", encoding="utf-8") as f:
        json.dump(meta, f, separators=(",", ":"), ensure_ascii=False)

    print(f"Extracted {len(all_skills)} skills to {OUTPUT}")
    print(f"  {len(local)} local ({sum(1 for s in local if s['source'] == 'built-in')} built-in, "
          f"{sum(1 for s in local if s['source'] == 'optional')} optional)")
    print(f"  {len(external)} from {external_source}")

    print("By source:")
    for src, count in by_source.most_common():
        print(f"  {src}: {count}")
    if index_meta and index_meta.get("indexGeneratedAt"):
        print(f"Unified index built at: {index_meta['indexGeneratedAt']}")


if __name__ == "__main__":
    main()
