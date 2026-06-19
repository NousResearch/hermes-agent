"""Skill Index — Semantic search, quality scoring, and eviction for Hermes skills.

This module provides:
1. Vector-based semantic skill search (sentence-transformers with hash fallback)
2. Multi-dimensional quality scoring (use frequency, completeness, trust, freshness, activity)
3. Quality-driven eviction thresholds for the Curator

Usage:
    from tools.skill_index import build_index, semantic_search, calculate_scores, load_quality_scores
"""
import hashlib
import json
import re
import time
import threading
from pathlib import Path
from typing import Optional

# ── Configuration ──

SKILLS_DIR = Path.home() / ".hermes" / "skills"
INDEX_DIR = SKILLS_DIR / ".skill-index"
EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
EMBEDDING_DIM = 384  # Model output dimension

_model = None
_model_lock = threading.Lock()


def _index_path() -> Path:
    return INDEX_DIR / "skill-index.json"


def _scores_path() -> Path:
    return INDEX_DIR / "quality-scores.json"


# ── Embedding ──

def _get_model():
    """Lazy-load the sentence-transformers model (thread-safe)."""
    global _model
    if _model is not None:
        return _model
    with _model_lock:
        if _model is not None:
            return _model
        try:
            from sentence_transformers import SentenceTransformer
            _model = SentenceTransformer(EMBEDDING_MODEL)
            return _model
        except Exception:
            return None


def _metadata_to_text(metadata: dict) -> str:
    """Convert skill metadata to a single text string for embedding."""
    return " ".join([
        metadata.get("name", "").replace("-", " "),
        metadata.get("description", ""),
        " ".join(metadata.get("tags", [])),
        metadata.get("triggers", "") if isinstance(metadata.get("triggers"), str) else " ".join(metadata.get("triggers", [])),
        metadata.get("body_preview", ""),
    ]).lower()


def _hash_embedding(text: str) -> list[float]:
    """Fallback hash-based embedding when sentence-transformers unavailable."""
    words = text.split()
    if not words:
        return [0.0] * EMBEDDING_DIM
    vec = [0.0] * EMBEDDING_DIM
    for word in words:
        h = int(hashlib.md5(word.encode()).hexdigest(), 16)
        idx = h % EMBEDDING_DIM
        sign = 1.0 if (h // EMBEDDING_DIM) % 2 == 0 else -1.0
        vec[idx] += sign * (1.0 / len(words))
    norm = sum(x * x for x in vec) ** 0.5
    if norm > 0:
        vec = [x / norm for x in vec]
    return vec


def compute_text_embedding(metadata: dict) -> list[float]:
    """Compute embedding for skill metadata using sentence-transformers.

    Falls back to hash-based embedding if sentence-transformers unavailable.

    Args:
        metadata: Dict with keys: name, description, tags, body_preview

    Returns:
        384-dimensional normalized vector
    """
    text = _metadata_to_text(metadata)
    model = _get_model()
    if model is not None:
        emb = model.encode(text, normalize_embeddings=True)
        return emb.tolist()
    return _hash_embedding(text)


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(x * x for x in b) ** 0.5
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


# ── SkillBank Body Extraction (Phase 1) ──

_SECTION_HEADERS = re.compile(r"^##\s+(.+)$", re.MULTILINE)

def _extract_section(body: str, *heading_keywords: str) -> str:
    """Extract text under the first matching ## heading.

    heading_keywords: case-insensitive substrings to match against heading text.
    Returns the section body (between this heading and the next ## or end).
    """
    if not body:
        return ""
    matches = list(_SECTION_HEADERS.finditer(body))
    for i, m in enumerate(matches):
        heading = m.group(1).strip().lower()
        if any(kw.lower() in heading for kw in heading_keywords):
            start = m.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(body)
            return body[start:end].strip()
    return ""


def _extract_principle(body: str) -> str:
    """Extract core approach from SKILL.md body.

    Priority: ## Steps / ## How > ## Core Pattern > first paragraph.
    """
    # 1. Try ## Steps / ## How
    section = _extract_section(body, "steps", "how it works", "how", "core pattern", "approach")
    if section:
        # Take first 500 chars or first 3 lines, whichever is shorter
        lines = [l.strip() for l in section.split("\n") if l.strip() and not l.strip().startswith("#")]
        result = " ".join(lines[:5])
        return result[:500]

    # 2. Fallback: first non-heading paragraph after the title
    lines = body.split("\n")
    past_title = False
    paragraph_lines = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("# ") and not past_title:
            past_title = True
            continue
        if not past_title:
            continue
        if stripped.startswith("##"):
            break
        if stripped:
            paragraph_lines.append(stripped)
        elif paragraph_lines:
            break
    return " ".join(paragraph_lines)[:500]


def _extract_when_to_apply(body: str) -> str:
    """Extract trigger/usage conditions from SKILL.md body.

    Looks for: ## When to Use, ## Trigger Conditions, ## When to Load, ## Triggers
    """
    section = _extract_section(body, "when to use", "trigger", "when to load", "triggers", "activation")
    if section:
        lines = [l.strip() for l in section.split("\n") if l.strip()]
        return "\n".join(lines[:10])[:500]
    return ""


def _extract_common_mistakes(body: str) -> list[str]:
    """Extract common mistakes/pitfalls from SKILL.md body.

    Looks for: ## Notes, ## Pitfalls, ## Anti-patterns, ## Common Mistakes
    """
    section = _extract_section(body, "pitfall", "anti-pattern", "common mistake", "notes", "warning")
    if not section:
        return []
    mistakes = []
    for line in section.split("\n"):
        stripped = line.strip()
        if stripped.startswith("- ") or stripped.startswith("* "):
            mistakes.append(stripped[2:].strip())
        elif stripped.startswith(("1.", "2.", "3.", "4.", "5.")):
            # Numbered list
            text = stripped.split(".", 1)[1].strip()
            if text:
                mistakes.append(text)
    return mistakes[:10]


# ── Index Building ──

def extract_skill_metadata(skill_dir: Path) -> dict | None:
    """Extract metadata from a skill directory's SKILL.md frontmatter."""
    skill_md = skill_dir / "SKILL.md"
    if not skill_md.exists():
        return None

    try:
        content = skill_md.read_text(errors="replace")
    except Exception:
        return None

    if not content.startswith("---"):
        return None

    try:
        end = content.find("---", 3)
        if end <= 0:
            return None

        frontmatter_text = content[3:end].strip()
        frontmatter = _parse_simple_yaml(frontmatter_text)
        full_body = content[end + 3:].strip()
        body = full_body[:500]  # Keep truncated for embedding/backward compat

        tags = []
        if "metadata" in frontmatter and isinstance(frontmatter["metadata"], dict):
            hermes_meta = frontmatter["metadata"].get("hermes", {})
            if isinstance(hermes_meta, dict):
                tags = hermes_meta.get("tags", [])

        triggers = frontmatter.get("triggers", [])
        if isinstance(triggers, str):
            triggers = [triggers]

        # Chaining metadata
        provides = frontmatter.get("provides", [])
        requires = frontmatter.get("requires", [])
        chain_with = frontmatter.get("chain_with", [])
        # Also check metadata.hermes for compatibility
        if not provides and "metadata" in frontmatter:
            hermes_meta = frontmatter["metadata"].get("hermes", {})
            if isinstance(hermes_meta, dict):
                provides = hermes_meta.get("provides", [])
                requires = hermes_meta.get("requires", [])
                chain_with = hermes_meta.get("chain_with", [])

        # SkillBank fields (Phase 1): extract from full body
        principle = _extract_principle(full_body)
        when_to_apply = _extract_when_to_apply(full_body)
        common_mistakes = _extract_common_mistakes(full_body)

        return {
            "name": frontmatter.get("name", skill_dir.name),
            "description": frontmatter.get("description", ""),
            "tags": tags if isinstance(tags, list) else [],
            "triggers": triggers if isinstance(triggers, list) else [],
            "body_preview": body[:200],
            "path": str(skill_dir.relative_to(SKILLS_DIR)) if SKILLS_DIR in skill_dir.parents else skill_dir.name,
            "provides": provides if isinstance(provides, list) else [],
            "requires": requires if isinstance(requires, list) else [],
            "chain_with": chain_with if isinstance(chain_with, list) else [],
            # SkillBank fields
            "principle": principle,
            "when_to_apply": when_to_apply,
            "common_mistakes": common_mistakes,
        }
    except Exception:
        return None


def merge_usage_data(index: dict, usage_path: Optional[Path] = None) -> dict:
    """Merge .usage.json Thompson data into skill-index.json.

    Adds success_rate (0-1) and total_uses to each skill in the index.
    Skills without usage data get neutral defaults (0.5 / 0).
    """
    if usage_path is None:
        usage_path = SKILLS_DIR / ".usage.json"

    usage = {}
    if usage_path.exists():
        try:
            usage = json.loads(usage_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass

    for skill in index.get("skills", []):
        name = skill.get("name", "")
        skill_usage = usage.get(name, {})
        skill["success_rate"] = skill_usage.get("success_rate", 0.5)
        skill["total_uses"] = skill_usage.get("total_outcomes", 0)

    return index


def _parse_simple_yaml(text: str) -> dict:
    """Minimal YAML parser for frontmatter (handles nested dicts, lists, and top-level lists)."""
    result = {}
    current_key = None
    sub_dict = None
    sub_key = None
    pending_list_key = None  # Track top-level key expecting list items

    for line in text.split("\n"):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue

        indent = len(line) - len(line.lstrip())

        # Handle list items (lines starting with "- ")
        if stripped.startswith("- "):
            val = stripped[2:].strip()
            if val and pending_list_key and indent > 0:
                # Top-level list item
                if not isinstance(result.get(pending_list_key), list):
                    result[pending_list_key] = []
                result[pending_list_key].append(_yaml_value(val))
                continue
            elif val and sub_dict is not None and sub_key and current_key:
                # Nested list item
                parent = result.get(current_key, {}).get(sub_key)
                if isinstance(parent, list):
                    parent.append(_yaml_value(val))
                elif isinstance(parent, dict) and not parent:
                    result[current_key][sub_key] = [_yaml_value(val)]
                continue

        if ":" in stripped:
            parts = stripped.split(":", 1)
            key = parts[0].strip()
            value = parts[1].strip()

            if indent == 0:
                current_key = key
                sub_dict = None
                if value:
                    result[key] = _yaml_value(value)
                    pending_list_key = None
                else:
                    # Could be a dict or a list — assume dict, switch to list if items follow
                    result[key] = {}
                    pending_list_key = key  # Will be converted to list if items follow
            elif indent > 0 and current_key:
                if isinstance(result.get(current_key), dict):
                    if value:
                        result[current_key][key] = _yaml_value(value)
                    else:
                        sub_dict = {}
                        result[current_key][key] = sub_dict
                        sub_key = key
        elif indent == 0:
            # Not a colon line and not indented — reset pending list
            pending_list_key = None

    # Post-process: convert {} to [] for keys that have list items
    for key in list(result.keys()):
        if isinstance(result[key], dict) and not result[key]:
            # Empty dict that might have been intended as a list
            # Check if it was followed by list items (already handled above)
            pass

    return result


def _yaml_value(s: str):
    """Convert a YAML value string to Python type."""
    if not s:
        return ""
    if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
        return s[1:-1]
    if s.lower() in ("true", "yes"):
        return True
    if s.lower() in ("false", "no"):
        return False
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        pass
    return s


def build_index() -> dict:
    """Build or rebuild the skill index from filesystem.

    Returns:
        Index dict with metadata and embeddings for all skills.
    """
    INDEX_DIR.mkdir(exist_ok=True)
    skills = []

    # Pre-load model once for batch encoding
    model = _get_model()
    texts = []
    skill_metas = []

    for item in sorted(SKILLS_DIR.rglob("SKILL.md")):
        skill_dir = item.parent
        if skill_dir.name.startswith(".") or skill_dir.name.startswith("_"):
            continue
        if skill_dir.parent != SKILLS_DIR and skill_dir.parent.name not in (".", ""):
            if (skill_dir.parent / "SKILL.md").exists():
                continue

        meta = extract_skill_metadata(skill_dir)
        if meta:
            skill_metas.append(meta)
            texts.append(_metadata_to_text(meta))

    # Batch encode for efficiency
    if model is not None and texts:
        import numpy as np
        embeddings = model.encode(texts, normalize_embeddings=True, batch_size=32, show_progress_bar=False)
        for meta, emb in zip(skill_metas, embeddings):
            meta["embedding"] = emb.tolist()
            skills.append(meta)
    else:
        for meta in skill_metas:
            meta["embedding"] = _hash_embedding(_metadata_to_text(meta))
            skills.append(meta)

    index = {
        "version": 2,
        "model": EMBEDDING_MODEL if model is not None else "hash-fallback",
        "built_at": __import__("datetime").datetime.now().isoformat(),
        "count": len(skills),
        "skills": skills,
    }

    _index_path().write_text(json.dumps(index, indent=2, ensure_ascii=False))
    return index


def _load_index() -> dict:
    """Load the skill index from disk."""
    if not _index_path().exists():
        return {"skills": []}
    return json.loads(_index_path().read_text(errors="replace"))


# ── Semantic Search ──

def semantic_search(query: str, top_k: int = 5) -> list[dict]:
    """Search skills by semantic similarity.

    Args:
        query: Natural language query
        top_k: Number of results to return

    Returns:
        List of dicts with keys: name, path, description, score
    """
    index = _load_index()
    if not index.get("skills"):
        return []

    query_emb = compute_text_embedding({
        "name": query,
        "description": query,
        "tags": [],
        "body_preview": "",
    })

    results = []
    for skill in index["skills"]:
        sim = cosine_similarity(query_emb, skill["embedding"])
        results.append({
            "name": skill["name"],
            "path": skill["path"],
            "description": skill["description"][:100],
            "score": round(sim, 4),
        })

    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_k]


# ── Quality Scoring ──

def score_completeness(skill_dir: Path) -> float:
    """Score SKILL.md completeness (0.0 - 1.0)."""
    skill_md = skill_dir / "SKILL.md"
    if not skill_md.exists():
        return 0.0

    try:
        content = skill_md.read_text(errors="replace")
    except Exception:
        return 0.0

    score = 0.0

    if content.startswith("---"):
        score += 0.3
        header = content[:500]
        if "description:" in header:
            score += 0.2
        if "tags:" in header:
            score += 0.1
        if "triggers:" in header or "prerequisites:" in header:
            score += 0.1

    body_len = len(content)
    if body_len > 2000:
        score += 0.3
    elif body_len > 500:
        score += 0.15

    return min(score, 1.0)


def score_trust(skill_dir: Path) -> float:
    """Score based on trust level using skill_usage.provenance() API.

    Returns: 1.0 (bundled), 0.7 (hub), 0.3 (agent-created)
    """
    try:
        from tools.skill_usage import provenance
        prov = provenance(skill_dir.name)
        return {"bundled": 1.0, "hub": 0.7, "agent": 0.3}.get(prov, 0.3)
    except (ImportError, Exception):
        bundled = SKILLS_DIR / ".bundled_manifest"
        if bundled.exists() and skill_dir.name in bundled.read_text():
            return 1.0
        return 0.3


def score_freshness(skill_dir: Path) -> float:
    """Score based on last update time (0.0 - 1.0)."""
    skill_md = skill_dir / "SKILL.md"
    if not skill_md.exists():
        return 0.0
    try:
        mtime = skill_md.stat().st_mtime
    except Exception:
        return 0.0
    days_old = (time.time() - mtime) / 86400
    return max(0.0, 1.0 - days_old / 365)


def calculate_scores() -> dict:
    """Calculate quality scores for all skills.

    Dimensions (weights sum to 1.0):
        use_frequency (0.20): Normalized use count
        completeness  (0.20): Documentation quality
        trust         (0.15): Authorship trust level
        freshness     (0.15): Recency of last use
        activity      (0.10): Combined use+view+patch activity
        success_rate  (0.20): Rolling success rate from feedback loop (NEW)

    Returns:
        Dict of {skill_name: {"total": float, "dimensions": {...}, "use_count": int}}
    """
    usage_path = SKILLS_DIR / ".usage.json"
    usage = {}
    if usage_path.exists():
        try:
            usage = json.loads(usage_path.read_text())
        except Exception:
            pass

    max_use = max((u.get("use_count", 0) for u in usage.values()), default=1) or 1
    max_activity = max(
        (u.get("use_count", 0) + u.get("view_count", 0) + u.get("patch_count", 0) for u in usage.values()),
        default=1
    ) or 1

    scores = {}
    for item in SKILLS_DIR.rglob("SKILL.md"):
        skill_dir = item.parent
        if skill_dir.name.startswith(".") or skill_dir.name.startswith("_"):
            continue
        if skill_dir.parent != SKILLS_DIR and skill_dir.parent.name not in (".", ""):
            if (skill_dir.parent / "SKILL.md").exists():
                continue

        name = skill_dir.name
        skill_usage = usage.get(name, {})

        use_freq = skill_usage.get("use_count", 0) / max_use
        completeness = score_completeness(skill_dir)
        trust = score_trust(skill_dir)
        freshness = score_freshness(skill_dir)

        activity = skill_usage.get("use_count", 0) + skill_usage.get("view_count", 0) + skill_usage.get("patch_count", 0)
        activity_score = activity / max_activity

        # Success rate from feedback loop (0.5 neutral default when no data)
        raw_rate = skill_usage.get("success_rate")
        if raw_rate is not None:
            try:
                success_rate = float(raw_rate)
            except (TypeError, ValueError):
                success_rate = 0.5
        else:
            success_rate = 0.5  # Neutral: no data yet

        total = (
            0.20 * use_freq +
            0.20 * completeness +
            0.15 * trust +
            0.15 * freshness +
            0.10 * activity_score +
            0.20 * success_rate
        )

        scores[name] = {
            "total": round(total, 4),
            "dimensions": {
                "use_frequency": round(use_freq, 4),
                "completeness": round(completeness, 4),
                "trust": round(trust, 4),
                "freshness": round(freshness, 4),
                "activity": round(activity_score, 4),
                "success_rate": round(success_rate, 4),
            },
            "use_count": skill_usage.get("use_count", 0),
            "total_outcomes": skill_usage.get("total_outcomes", 0),
        }

    INDEX_DIR.mkdir(exist_ok=True)
    _scores_path().write_text(json.dumps(scores, indent=2, ensure_ascii=False))

    return scores


def load_quality_scores() -> dict:
    """Load quality scores from disk. Returns empty dict if not available."""
    if not _scores_path().exists():
        return {}
    try:
        return json.loads(_scores_path().read_text())
    except Exception:
        return {}


# ── Eviction Helpers ──


# ── Skill Chaining ──

def discover_chain(goal_provides: list[str], max_depth: int = 3) -> list[list[dict]]:
    """Discover skill chains that can produce the goal outputs.

    Args:
        goal_provides: What we want to achieve (e.g., ["test-passing", "deployment-plan"])
        max_depth: Maximum chain length

    Returns:
        List of chains, each chain is a list of skill dicts
    """
    index = _load_index()
    skills = index.get("skills", [])

    # Build lookup: provide_name -> [skills that provide it]
    provides_map: dict[str, list[dict]] = {}
    for skill in skills:
        for p in skill.get("provides", []):
            provides_map.setdefault(p, []).append(skill)

    # BFS to find chains
    chains = []
    queue: list[list[dict]] = [[]]

    for depth in range(max_depth):
        next_queue = []
        for chain in queue:
            # What does the current chain provide?
            current_provides = set()
            for skill in chain:
                current_provides.update(skill.get("provides", []))

            # What do we still need?
            needed = set(goal_provides) - current_provides

            if not needed:
                chains.append(chain)
                continue

            # Find skills that provide what we need
            for need in needed:
                for skill in provides_map.get(need, []):
                    if skill["name"] not in {s["name"] for s in chain}:
                        # Check if skill's requirements are satisfied
                        skill_requires = set(skill.get("requires", []))
                        if skill_requires.issubset(current_provides | {"ssh-access", "project-structure"}):
                            # Allow common implicit requirements
                            next_queue.append(chain + [skill])

        queue = next_queue

    # Sort by chain length (shorter = better)
    chains.sort(key=len)
    return chains[:5]


def suggest_next_skills(current_skill: str) -> list[dict]:
    """Suggest skills that commonly chain after the current skill.

    Args:
        current_skill: Name of the currently loaded skill

    Returns:
        List of suggested skills with reasons
    """
    index = _load_index()
    skills_by_name = {s["name"]: s for s in index.get("skills", [])}

    current = skills_by_name.get(current_skill, {})
    if not current:
        return []

    suggestions = []

    # 1. Explicit chain_with
    for name in current.get("chain_with", []):
        if name in skills_by_name:
            suggestions.append({
                "name": name,
                "reason": "explicitly chained",
                "score": 1.0,
            })

    # 2. provides/requires matching
    current_provides = set(current.get("provides", []))
    for name, skill in skills_by_name.items():
        if name == current_skill:
            continue
        skill_requires = set(skill.get("requires", []))
        overlap = current_provides & skill_requires
        if overlap:
            suggestions.append({
                "name": name,
                "reason": f"provides: {', '.join(overlap)}",
                "score": len(overlap) / max(len(skill_requires), 1),
            })

    # Deduplicate
    seen = set()
    unique = []
    for s in suggestions:
        if s["name"] not in seen:
            seen.add(s["name"])
            unique.append(s)

    unique.sort(key=lambda x: x["score"], reverse=True)
    return unique[:5]


def get_stale_threshold(skill_name: str, quality_scores: dict | None = None) -> int:
    """Get the stale threshold in days for a skill based on quality score.

    Args:
        skill_name: Name of the skill
        quality_scores: Quality scores dict (from load_quality_scores)

    Returns:
        Number of days before skill becomes stale
    """
    if quality_scores is None:
        return 30

    quality = quality_scores.get(skill_name, {}).get("total", 0.5)

    if quality < 0.2:
        return 14
    elif quality < 0.3:
        return 30
    elif quality >= 0.7:
        return 90
    elif quality >= 0.5:
        return 60
    else:
        return 30


# ── CLI Entry Point ──

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m tools.skill_index <command> [args]")
        print("  build         — Build/rebuild skill index")
        print("  search <q>    — Semantic search")
        print("  scores        — Calculate quality scores")
        print("  top [N]       — Show top N skills by quality")
        print("  chain <goal>  — Discover skill chains for a goal")
        print("  suggest <s>   — Suggest next skills to chain")
        print("  chaining      — Show skills with chaining metadata")
        sys.exit(1)

    cmd = sys.argv[1]

    if cmd == "build":
        index = build_index()
        print(f"✅ Indexed {index['count']} skills (model: {index['model']})")

    elif cmd == "search":
        query = " ".join(sys.argv[2:])
        results = semantic_search(query, top_k=5)
        for i, r in enumerate(results, 1):
            print(f"{i}. [{r['score']}] {r['name']} — {r['description'][:60]}")

    elif cmd == "scores":
        scores = calculate_scores()
        ranked = sorted(scores.items(), key=lambda x: x[1]["total"], reverse=True)
        print("=== Skill Quality Scores ===")
        for i, (name, s) in enumerate(ranked[:15], 1):
            print(f"{i:2d}. [{s['total']:.3f}] {name} (uses={s['use_count']})")
        print(f"\nTotal: {len(scores)} skills scored")

    elif cmd == "top":
        n = int(sys.argv[2]) if len(sys.argv) > 2 else 10
        scores = load_quality_scores()
        if not scores:
            print("No scores found. Run: python -m tools.skill_index scores")
            sys.exit(1)
        ranked = sorted(scores.items(), key=lambda x: x[1]["total"], reverse=True)
        for i, (name, s) in enumerate(ranked[:n], 1):
            dims = s["dimensions"]
            print(f"{i:2d}. [{s['total']:.3f}] {name}")
            print(f"     use={dims['use_frequency']:.2f} comp={dims['completeness']:.2f} "
                  f"trust={dims['trust']:.2f} fresh={dims['freshness']:.2f} act={dims['activity']:.2f}")

    elif cmd == "chain":
        goal = " ".join(sys.argv[2:])
        goal_map = {
            "deploy": ["server-ready", "deployment-plan"],
            "review": ["code-review-report"],
            "test": ["test-passing"],
            "api": ["api-endpoint"],
        }
        provides = []
        for kw, provs in goal_map.items():
            if kw in goal.lower():
                provides.extend(provs)
        if not provides:
            provides = ["documentation"]
        chains = discover_chain(provides)
        if chains:
            print(f"=== Skill Chains for '{goal}' ===")
            for i, chain in enumerate(chains, 1):
                names = " → ".join(s["name"] for s in chain)
                print(f"{i}. {names}")
        else:
            print(f"No chains found for '{goal}'")

    elif cmd == "suggest":
        skill_name = sys.argv[2] if len(sys.argv) > 2 else ""
        if not skill_name:
            print("Usage: skill_index suggest <skill-name>")
            sys.exit(1)
        suggestions = suggest_next_skills(skill_name)
        if suggestions:
            print(f"=== Suggested chains after '{skill_name}' ===")
            for i, s in enumerate(suggestions, 1):
                print(f"{i}. {s['name']} ({s['reason']}, score={s['score']:.2f})")
        else:
            print(f"No suggestions for '{skill_name}'")

    elif cmd == "chaining":
        index = _load_index()
        with_chain = [s for s in index.get("skills", []) if s.get("provides") or s.get("requires") or s.get("chain_with")]
        if with_chain:
            print(f"=== Skills with Chaining Metadata ({len(with_chain)}) ===")
            for s in with_chain:
                p = ", ".join(s.get("provides", [])) or "-"
                r = ", ".join(s.get("requires", [])) or "-"
                c = ", ".join(s.get("chain_with", [])) or "-"
                print(f"  {s['name']}:")
                print(f"    provides: {p}")
                print(f"    requires: {r}")
                print(f"    chain_with: {c}")
        else:
            print("No skills with chaining metadata. Run 'build' first.")

    else:
        print(f"Unknown command: {cmd}")
        sys.exit(1)
