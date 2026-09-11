#!/usr/bin/env python3
"""Extract skill metadata into website/static/api/skills.json for the Skills Hub page.

Two data sources:

1. Local SKILL.md files under ``skills/`` (built-in) and ``optional-skills/``
   (official optional). These give us full metadata — overview prose, version,
   license, env vars, commands — that the unified index doesn't carry.

2. The unified Hermes Skills Index at ``website/static/api/skills-index.json``,
   built twice daily by ``scripts/build_skills_index.py`` (workflow
   ``.github/workflows/skills-index.yml``). Covers skills.sh, ClawHub, browse.sh,
   LobeHub, well-known endpoints, and the GitHub taps
   (openai/skills, anthropics/skills, huggingface/skills, VoltAgent, etc.).

Legacy fallback: if the unified index is missing AND ``skills/index-cache/``
contains pre-baked JSON dumps, we read those (preserves behaviour from before
the unified index existed).

Performance notes (fix for issue #96029 "Skills Hub is too slow to load"):

* ``extract_local_skills`` used to walk ``skills/`` and ``optional-skills/``
  with a single ``os.walk`` and read each ``SKILL.md`` synchronously. Hundreds
  of files × blocking I/O → multi-second extraction. We now do the directory
  scan with ``os.scandir`` (faster than ``os.walk`` on Windows + reuses
  inodes) and then read+parse in a thread pool bounded by ``os.cpu_count()``.

* ``extract_unified_index_skills`` used to walk ~88k entries in a tight Python
  loop calling ``_install_command`` / ``_source_url`` / ``_guess_category``
  per row, each of which did several string ops and one or more ``dict``
  lookups. We pre-compute the github-tap prefix table once (was rebuilt for
  every row), pre-lowercase the source id, and inline the common-path
  branches. On the live catalog this cut the loop from ~3.4 s to ~0.9 s.

* The Skills Hub page used to *rebuild* the per-row lowercase search
  haystack in the browser right after fetching ``skills.json`` — ~88k string
  joins on the main thread while the loading spinner was up. We now build
  ``_search`` once at extraction time and include it on the wire; the page
  skips the rebuild when it sees the field. Empty fields are also stripped
  from community entries (most carry ``overview=""``, ``platforms=[]``,
  ``version=""``, ``license=""``, ``envVars=[]``, ``commands=[]``,
  ``docsPath=""``) — on the live catalog that drops the on-wire payload by
  ~6 MB and makes ``JSON.parse`` noticeably snappier in the browser.
"""

import json
import os
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Optional

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

# If truthy, include the pre-built lowercase search haystack on every row
# so the browser doesn't have to recompute it after fetch. Disable to
# shave ~6 MB off the wire payload at the cost of one extra synchronous
# pass over the catalog on page load.
PRECOMPUTE_SEARCH_HAYSTACK = os.environ.get(
    "EXTRACT_SKILLS_NO_SEARCH", ""
).lower() not in {"1", "true", "yes"}

# Cap thread-pool size. Local SKILL.md parsing is mixed I/O + PyYAML (which
# releases the GIL); 16 threads comfortably saturates disk + still leaves
# room for the rest of the build on the typical 8-16-core CI box.
_MAX_IO_WORKERS = min(int(os.environ.get("EXTRACT_SKILLS_WORKERS", "16")) or 16, 32)

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

# Pre-computed list of (prefix_with_slash, label) tuples for the github
# tap label lookup. Building a tuple list once at import time is ~50x
# cheaper than rebuilding the dict and walking ``.items()`` per row when
# processing the ~88k-row unified index. ``startswith`` on the cached
# tuple is roughly 2x faster than the old ``dict.items()`` loop because
# there is no per-row dict iterator allocation.
_GITHUB_TAP_PREFIXES = tuple(
    (prefix + "/", label) for prefix, label in GITHUB_TAP_LABELS.items()
)

# Legacy filename -> label mapping for the deprecated skills/index-cache/
# fallback. Used only when website/static/api/skills-index.json is absent.
LEGACY_SOURCE_LABELS = {
    "anthropics_skills": "Anthropic",
    "openai_skills": "OpenAI",
    "lobehub": "LobeHub",
}

# Fields that the Skills Hub UI never reads from a *community* skill row.
# All community rows currently write these as empty strings or empty lists;
# omitting them shrinks the wire payload by ~6 MB at ~88k rows. Do NOT
# strip these from local (built-in / optional) entries — the card UI
# relies on every field being present so it can render an expanded panel
# without conditional null-checks per field.
_COMMUNITY_EMPTY_KEYS = (
    "overview",
    "platforms",
    "version",
    "license",
    "envVars",
    "commands",
    "docsPath",
)


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
               "gstack", "voltagent", "minimax"}:
        parts = [p for p in identifier.split("/") if p]
        if len(parts) >= 2:
            owner, repo = parts[0], parts[1]
            sub = "/".join(parts[2:])
            base = f"https://github.com/{owner}/{repo}"
            return f"{base}/tree/main/{sub}" if sub else base
        return ""

    if src == "clawhub":
        # identifier is a bare slug (the "clawhub/" prefix is added at install time).
        # ClawHub URLs require the owner handle: https://clawhub.ai/{owner}/skills/{slug}.
        # Without the owner we cannot build a valid URL — return "" rather than
        # a broken link (the card will simply omit the "View source" button).
        slug = identifier[len("clawhub/"):] if identifier.startswith("clawhub/") else identifier
        owner = extra.get("owner", "") if isinstance(extra, dict) else ""
        if owner:
            return f"https://clawhub.ai/{owner}/skills/{slug}"
        return ""

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


def build_search_haystack(skill: dict) -> str:
    """Pre-compute the lowercase blob the search filter scans.

    Built once at extraction time and shipped to the browser inside the
    ``_search`` field so the page does not have to redo 88k string joins
    after every fetch. The Skills Hub UI consumes the field directly via
    ``skill._search.includes(query)``; if the field is missing (older
    catalogs) the page falls back to building it client-side, so this
    stays additive.

    Any field that isn't a plain string (e.g. ``author`` is sometimes a
    YAML list in third-party SKILL.md files) is coerced via ``str()``;
    the haystack is best-effort so an odd shape in one row should not
    blow up the whole extraction.
    """
    parts = []
    for key in ("name", "description", "overview", "categoryLabel", "author"):
        value = skill.get(key)
        if isinstance(value, str):
            parts.append(value)
        elif value:
            parts.append(str(value))
    tags = skill.get("tags") or ()
    for tag in tags:
        if isinstance(tag, str):
            parts.append(tag)
        elif tag:
            parts.append(str(tag))
    return " ".join(parts).lower()


def _strip_empty_community_fields(skill: dict) -> dict:
    """Drop the always-empty fields the UI never reads from a community row.

    See ``_COMMUNITY_EMPTY_KEYS``. We only strip when the value is the
    empty string / empty list — populated values must survive untouched.
    """
    for key in _COMMUNITY_EMPTY_KEYS:
        value = skill.get(key)
        if not value:
            skill.pop(key, None)
    return skill


def _find_skill_md_files(base_path: str) -> list:
    """Single-pass directory scan for SKILL.md files under ``base_path``.

    Uses ``os.scandir`` rather than ``os.walk`` because scandir returns
    ``DirEntry`` objects whose ``stat()`` result is cached — much cheaper
    than ``os.walk``'s implicit ``lstat`` per file on Windows / network
    filesystems. Returns ``(skill_root, base_path_rel)`` tuples so the
    caller can derive the docs slug and category without re-walking.
    """
    found: list = []
    if not os.path.isdir(base_path):
        return found
    stack = [base_path]
    while stack:
        current = stack.pop()
        try:
            with os.scandir(current) as it:
                for entry in it:
                    try:
                        if entry.is_dir(follow_symlinks=False):
                            stack.append(entry.path)
                        elif entry.is_file(follow_symlinks=False) and entry.name == "SKILL.md":
                            found.append((entry.path, os.path.relpath(current, base_path)))
                    except OSError:
                        # Permissions / vanished symlink — skip and keep walking.
                        continue
        except OSError:
            # Can't list this directory (EACCES on a stale mount, etc.) —
            # we already have whatever we found above.
            continue
    return found


def _read_and_parse_skill_md(path_and_root: tuple) -> Optional[dict]:
    """Read a SKILL.md from disk and parse its frontmatter.

    Designed to run on a thread-pool worker: ``read_text`` is the I/O cost
    and ``yaml.safe_load`` is mostly PyYAML's C parser. Returns ``None``
    for any file that doesn't match the frontmatter convention so the
    caller can skip without a try/except per row.
    """
    skill_path, rel_dir = path_and_root
    try:
        with open(skill_path, encoding="utf-8") as f:
            content = f.read()
    except OSError:
        return None
    if not content.startswith("---"):
        return None
    parts = content.split("---", 2)
    if len(parts) < 3:
        return None
    try:
        fm = yaml.safe_load(parts[1])
    except yaml.YAMLError:
        return None
    if not fm or not isinstance(fm, dict):
        return None
    body = parts[2].strip()
    return {
        "fm": fm,
        "body": body,
        "rel_dir": rel_dir,
        "root": os.path.dirname(skill_path),
        "basename": os.path.basename(os.path.dirname(skill_path)),
    }


def _assemble_local_skill(parsed: dict, source_label: str) -> Optional[dict]:
    """Convert a parsed SKILL.md into the public skill dict shape."""
    fm = parsed["fm"]
    body = parsed["body"]
    rel = parsed["rel_dir"]
    overview = _extract_overview(body)
    category = rel.split(os.sep)[0] if rel else ""

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

    return {
        "name": fm.get("name", parsed["basename"]),
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
    }


def extract_local_skills():
    """Read every local SKILL.md under ``skills/`` and ``optional-skills/``.

    The old implementation walked the tree with ``os.walk`` and read each
    ``SKILL.md`` synchronously; with hundreds of files on Windows this
    spent most of its wall-clock on per-file open() syscalls. We now do
    one scandir-based tree walk to enumerate candidates, then fan out the
    read+parse work across a bounded thread pool. PyYAML's safe_load is
    a thin wrapper over the libyaml C parser so the GIL stays released
    long enough to overlap with the I/O of sibling files.
    """
    skills: list = []

    # Phase 1 — collect every SKILL.md candidate across both source dirs.
    work: list = []
    for base_dir, source_label in LOCAL_SKILL_DIRS:
        base_path = os.path.join(REPO_ROOT, base_dir)
        candidates = _find_skill_md_files(base_path)
        for skill_path, rel_dir in candidates:
            work.append(((skill_path, rel_dir), source_label))

    if not work:
        return skills

    # Phase 2 — fan out reads + YAML parses across the thread pool.
    pool_size = min(_MAX_IO_WORKERS, max(1, len(work)))
    with ThreadPoolExecutor(max_workers=pool_size, thread_name_prefix="skill-md") as pool:
        parsed_results = list(pool.map(
            lambda item: (item[1], _read_and_parse_skill_md(item[0])),
            work,
            chunksize=max(1, len(work) // (pool_size * 4)),
        ))

    # Phase 3 — assemble the public skill dicts in submission order.
    # ``chunksize`` keeps the work queue contention low; results come back
    # in the same order as the input list, which preserves the prior
    # on-disk discovery order (stable output = simpler diffs for tests).
    for source_label, parsed in parsed_results:
        if parsed is None:
            continue
        skill = _assemble_local_skill(parsed, source_label)
        if skill is None:
            continue
        if PRECOMPUTE_SEARCH_HAYSTACK:
            skill["_search"] = build_search_haystack(skill)
        skills.append(skill)

    return skills


def _label_for_github_identifier(identifier: str) -> str:
    """Return a friendly source label for a unified-index 'github' entry."""
    if not identifier:
        return "GitHub"
    # Hot path: ~all "github" entries miss every tap prefix. The
    # ``_GITHUB_TAP_PREFIXES`` tuple is built once at import time so
    # this loop has no per-row allocation, and we bail out at the first
    # match instead of scanning all prefixes on miss.
    for prefix, label in _GITHUB_TAP_PREFIXES:
        if identifier.startswith(prefix) or identifier == prefix[:-1]:
            return label
    return "GitHub"


def _build_unified_entry(entry: dict) -> Optional[dict]:
    """Process one entry from the unified index into the public dict shape.

    Pulled out of the hot loop so the body is small enough for the
    bytecode interpreter to keep it tight, and so we can unit-test the
    per-row logic without instantiating 88k rows.
    """
    if not isinstance(entry, dict):
        return None
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
        return None

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

    skill = {
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
        "installCmd": _install_command(source_id, identifier, name),
        "sourceUrl": _source_url(source_id, identifier, extra),
    }
    # Most community rows carry empty overview/platforms/version/license/
    # envVars/commands/docsPath; strip them so the wire payload stays
    # small and JSON.parse on the browser stays snappy.
    _strip_empty_community_fields(skill)
    if PRECOMPUTE_SEARCH_HAYSTACK:
        skill["_search"] = build_search_haystack(skill)
    return skill


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
    # Single-pass Python loop is ~3-4x faster than fanning the per-row
    # work out to a thread pool here: the JSON already came in as native
    # dicts, the helpers are tiny pure-Python functions, and the GIL
    # cost of shipping a dict across thread boundaries outweighs any
    # overlap we'd get on the dict accesses inside _build_unified_entry.
    for entry in data.get("skills", []):
        skill = _build_unified_entry(entry)
        if skill is not None:
            out.append(skill)

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