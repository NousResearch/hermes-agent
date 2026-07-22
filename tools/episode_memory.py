"""Evidence-gated episode memory + FTS recall.

Episodes are durable lessons stored under ``$HERMES_HOME/memories/episodes/``.
They are **not** injected into the system prompt (unlike MEMORY.md / USER.md).
Writes require evidence (path, commit SHA, PR/issue URL, or log path) so
unverified claims cannot re-enter later wearing authority.

Recall is local SQLite FTS5 over episodes (and, when enabled, other
HERMES_HOME markdown roots). Stdlib only. No embeddings, no external service,
no Honcho dependency.

Design intent (user demand): fill the gap between tiny always-on MEMORY.md,
chat-only session_search, and external memory providers. Related prior art
outside core: joelbrilliant/agent-memory-kit (edge install). Non-overlap with
open PRs for shared FastAPI memory (#23684), portable pack skill (#20692),
and git-as-memory skill (#28636).
"""

from __future__ import annotations

import json
import logging
import os
import re
import sqlite3
import time
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from hermes_constants import display_hermes_home, get_hermes_home
from tools.registry import registry, tool_error

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------


def get_episodes_dir() -> Path:
    return get_hermes_home() / "memories" / "episodes"


def get_episode_index_path() -> Path:
    return get_hermes_home() / "memories" / "episodes.db"


WRITE_DISCIPLINE = (
    "Episodes are durable lessons that stay useful in a month, not status "
    "snapshots. evidence is mandatory (file path, commit SHA, PR/issue URL, "
    "or log path). Queue/phase/completion notes belong in a tracker or "
    "session_search, not here. An unverified claim in a memory store is worse "
    "than no memory."
)

_STATUS_SHAPED = re.compile(
    r"(?i)\b("
    r"phase\s+\d+\s+(done|complete|finished)|"
    r"queue\s+(is\s+)?at|"
    r"status\s*update|"
    r"todo\s*:\s*|"
    r"wip\b|"
    r"in\s+progress\b"
    r")\b"
)

_SECRET_PATTERNS = [
    re.compile(r"sk-ant-[A-Za-z0-9_\-]{20,}"),
    re.compile(r"xox[baprs]-[A-Za-z0-9\-]{10,}"),
    re.compile(r"AKIA[0-9A-Z]{16}"),
    re.compile(r"ghp_[A-Za-z0-9]{36,}"),
    re.compile(r"github_pat_[A-Za-z0-9_]{20,}"),
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY"),
    re.compile(
        r"(?im)^(?:export\s+)?[A-Z0-9_]*(?:API_KEY|TOKEN|SECRET|PASSWORD)\s*=\s*\S{16,}"
    ),
]

MAX_EPISODE_CHARS = 4000
MAX_INDEX_FILE_BYTES = 1024 * 1024
DEFAULT_CORPUS_ROOTS = ("episodes", "memories")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _slugify(content: str) -> str:
    words = re.findall(r"[A-Za-z0-9]+", content.lower())
    slug = "-".join(words[:6]) or "episode"
    slug = re.sub(r"[^a-z0-9-]", "", slug)
    slug = re.sub(r"-+", "-", slug).strip("-")
    return slug or "episode"


def _unique_episode_path(base_dir: Path, date_str: str, slug: str) -> Path:
    path = base_dir / f"{date_str}-{slug}.md"
    if not path.exists():
        return path
    n = 2
    while True:
        candidate = base_dir / f"{date_str}-{slug}-{n}.md"
        if not candidate.exists():
            return candidate
        n += 1


def _has_secret(content: str) -> bool:
    return any(p.search(content) for p in _SECRET_PATTERNS)


def _parse_frontmatter(text: str) -> Tuple[Dict[str, str], str]:
    if not text.startswith("---\n"):
        return {}, text
    end = text.find("\n---\n", 4)
    if end < 0:
        return {}, text
    meta_block = text[4:end]
    body = text[end + 5 :]
    meta: Dict[str, str] = {}
    for line in meta_block.splitlines():
        if ":" not in line:
            continue
        key, _, val = line.partition(":")
        meta[key.strip()] = val.strip()
    return meta, body


def _looks_status_shaped(content: str) -> bool:
    return bool(_STATUS_SHAPED.search(content))


def _episodes_enabled() -> bool:
    """Read memory.episodes_enabled from config (default True)."""
    try:
        from hermes_cli.config import load_config

        cfg = load_config() or {}
        mem = cfg.get("memory") or {}
        if "episodes_enabled" not in mem:
            return True
        return bool(mem.get("episodes_enabled"))
    except Exception:
        return True


def _corpus_roots() -> Sequence[str]:
    try:
        from hermes_cli.config import load_config

        cfg = load_config() or {}
        mem = cfg.get("memory") or {}
        roots = mem.get("episode_corpus_roots")
        if isinstance(roots, list) and roots:
            return tuple(str(r) for r in roots)
    except Exception:
        pass
    return DEFAULT_CORPUS_ROOTS


# ---------------------------------------------------------------------------
# Index
# ---------------------------------------------------------------------------


def _ensure_schema(conn: sqlite3.Connection) -> None:
    conn.execute(
        "CREATE VIRTUAL TABLE IF NOT EXISTS docs USING fts5("
        "content, path UNINDEXED, source UNINDEXED, mtime UNINDEXED, "
        "tokenize='porter unicode61')"
    )


def _connect_index(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    _ensure_schema(conn)
    return conn


def _iter_corpus_files(roots: Sequence[str]) -> List[Tuple[Path, str]]:
    """Return (absolute_path, source_tag) pairs under HERMES_HOME."""
    home = get_hermes_home()
    out: List[Tuple[Path, str]] = []
    seen: set = set()

    for root_name in roots:
        root_name = root_name.strip().strip("/")
        if not root_name or ".." in root_name:
            continue
        if root_name == "episodes":
            base = get_episodes_dir()
            source = "episodes"
        else:
            base = home / root_name
            source = root_name.split("/")[0]

        if not base.exists():
            continue

        if base.is_file() and base.suffix == ".md":
            candidates = [base]
        else:
            candidates = sorted(base.rglob("*.md"))

        for path in candidates:
            try:
                resolved = path.resolve()
            except OSError:
                continue
            # Stay inside HERMES_HOME
            try:
                resolved.relative_to(home.resolve())
            except ValueError:
                continue
            # Skip backups / nested junk
            parts = set(resolved.parts)
            if any(p in parts for p in (".git", "node_modules", ".archive", "backups")):
                continue
            key = str(resolved)
            if key in seen:
                continue
            seen.add(key)
            out.append((resolved, source))
    return out


def rebuild_episode_index(
    index_path: Optional[Path] = None,
    roots: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Full rebuild of the FTS index. Safe to call anytime."""
    index_path = index_path or get_episode_index_path()
    roots = roots or _corpus_roots()
    files = _iter_corpus_files(roots)

    # Atomic replace via temp file
    tmp_path = index_path.with_suffix(".db.tmp")
    if tmp_path.exists():
        tmp_path.unlink()

    conn = _connect_index(tmp_path)
    indexed = 0
    skipped = 0
    try:
        for path, source in files:
            try:
                size = path.stat().st_size
                if size <= 0 or size > MAX_INDEX_FILE_BYTES:
                    skipped += 1
                    continue
                text = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                skipped += 1
                continue
            if _has_secret(text):
                skipped += 1
                logger.info("episode index skipped secret-shaped file: %s", path)
                continue
            mtime = int(path.stat().st_mtime)
            conn.execute(
                "INSERT INTO docs (content, path, source, mtime) VALUES (?, ?, ?, ?)",
                (text, str(path), source, mtime),
            )
            indexed += 1
        conn.commit()
    finally:
        conn.close()

    os.replace(str(tmp_path), str(index_path))
    return {
        "success": True,
        "indexed": indexed,
        "skipped": skipped,
        "index": str(index_path),
        "roots": list(roots),
    }


def _index_episode_file(path: Path, index_path: Optional[Path] = None) -> None:
    index_path = index_path or get_episode_index_path()
    try:
        text = path.read_text(encoding="utf-8")
        mtime = int(path.stat().st_mtime)
    except OSError as exc:
        logger.debug("episode index insert failed to read %s: %s", path, exc)
        return

    conn = _connect_index(index_path)
    try:
        # Drop prior row for same path if any, then insert.
        conn.execute("DELETE FROM docs WHERE path = ?", (str(path.resolve()),))
        conn.execute(
            "INSERT INTO docs (content, path, source, mtime) VALUES (?, ?, ?, ?)",
            (text, str(path.resolve()), "episodes", mtime),
        )
        conn.commit()
    except sqlite3.Error as exc:
        logger.warning("episode index insert failed: %s", exc)
    finally:
        conn.close()


def _fts_match_expr(query: str) -> str:
    terms = []
    for term in query.split():
        cleaned = term.replace('"', '""')
        if cleaned:
            terms.append(f'"{cleaned}"')
    return " ".join(terms)


def recall_episodes(
    query: str,
    k: int = 5,
    source: Optional[str] = None,
    index_path: Optional[Path] = None,
    auto_rebuild: bool = True,
) -> Dict[str, Any]:
    """BM25 recall over the local episode/corpus index."""
    query = (query or "").strip()
    if not query:
        return {"success": False, "error": "query is required for recall."}

    k = max(1, min(int(k or 5), 20))
    index_path = index_path or get_episode_index_path()

    if not index_path.exists() and auto_rebuild:
        rebuild_episode_index(index_path=index_path)

    if not index_path.exists():
        return {
            "success": True,
            "hits": [],
            "message": (
                f"No episode index at {display_hermes_home()}/memories/episodes.db yet. "
                "Write an episode with action=remember first."
            ),
        }

    match_expr = _fts_match_expr(query)
    if not match_expr:
        return {"success": True, "hits": [], "query": query}

    conn = sqlite3.connect(str(index_path))
    try:
        sql = (
            "SELECT path, source, mtime, bm25(docs) AS score, "
            "snippet(docs, 0, '', '', ' ... ', 40) AS snip "
            "FROM docs WHERE docs MATCH ?"
        )
        params: List[Any] = [match_expr]
        if source:
            sql += " AND source = ?"
            params.append(source)
        sql += " ORDER BY bm25(docs) LIMIT ?"
        params.append(k)

        try:
            rows = conn.execute(sql, params).fetchall()
        except sqlite3.Error as exc:
            return {
                "success": False,
                "error": f"FTS query failed: {exc}",
                "hint": "Try simpler terms, or action=reindex.",
            }
    finally:
        conn.close()

    hits = []
    for path, src, mtime, score, snip in rows:
        hits.append(
            {
                "path": path,
                "source": src,
                "mtime": mtime,
                "score": float(score) if score is not None else None,
                "snippet": (snip or "").strip(),
            }
        )
    return {"success": True, "query": query, "hits": hits, "count": len(hits)}


# ---------------------------------------------------------------------------
# Remember / list / get
# ---------------------------------------------------------------------------


def remember_episode(
    content: str,
    evidence: str,
    source: str = "agent",
    tags: Optional[Sequence[str]] = None,
    episodes_dir: Optional[Path] = None,
    index_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Write an evidence-gated episode markdown file and index it."""
    content = (content or "").strip()
    evidence = (evidence or "").strip()
    source = (source or "agent").strip() or "agent"

    if not evidence:
        return {
            "success": False,
            "error": "REFUSED: evidence is required.",
            "discipline": WRITE_DISCIPLINE,
        }
    if not content:
        return {"success": False, "error": "REFUSED: empty content."}
    if len(content) > MAX_EPISODE_CHARS:
        return {
            "success": False,
            "error": (
                f"Episode content is {len(content)} chars; max is {MAX_EPISODE_CHARS}. "
                "Write a tighter lesson."
            ),
        }
    if _has_secret(content) or _has_secret(evidence):
        return {
            "success": False,
            "error": "REFUSED: content or evidence looks secret-shaped. Do not store secrets.",
        }

    # Soft warn on status-shaped content (still allow if evidence present —
    # the model may be wrong about the heuristic). Surface the warning.
    warnings: List[str] = []
    if _looks_status_shaped(content):
        warnings.append(
            "Content looks status-shaped (phase/queue/WIP). Prefer a durable lesson. "
            + WRITE_DISCIPLINE
        )

    tag_list = []
    if tags:
        for t in tags:
            t = str(t).strip()
            if t:
                tag_list.append(t)

    episodes_dir = episodes_dir or get_episodes_dir()
    episodes_dir.mkdir(parents=True, exist_ok=True)

    date_str = date.today().isoformat()
    path = _unique_episode_path(episodes_dir, date_str, _slugify(content))

    frontmatter = [
        "---",
        f"date: {date_str}",
        f"source: {source}",
        f"evidence: {evidence}",
        f"tags: [{', '.join(tag_list)}]",
        "---",
        "",
    ]
    file_text = "\n".join(frontmatter) + content + "\n"
    path.write_text(file_text, encoding="utf-8")

    _index_episode_file(path, index_path=index_path)

    result: Dict[str, Any] = {
        "success": True,
        "path": str(path),
        "date": date_str,
        "source": source,
        "evidence": evidence,
        "tags": tag_list,
        "message": (
            f"Episode written to {path.name}. It is searchable via episode(action='recall') "
            "and is NOT injected into the system prompt."
        ),
    }
    if warnings:
        result["warnings"] = warnings
    return result


def list_episodes(
    limit: int = 20,
    episodes_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    episodes_dir = episodes_dir or get_episodes_dir()
    limit = max(1, min(int(limit or 20), 100))
    if not episodes_dir.exists():
        return {"success": True, "episodes": [], "count": 0}

    files = sorted(episodes_dir.glob("*.md"), key=lambda p: p.stat().st_mtime, reverse=True)
    items = []
    for path in files[:limit]:
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        meta, body = _parse_frontmatter(text)
        preview = " ".join(body.strip().split())[:160]
        items.append(
            {
                "path": str(path),
                "date": meta.get("date"),
                "source": meta.get("source"),
                "evidence": meta.get("evidence"),
                "tags": meta.get("tags"),
                "preview": preview,
            }
        )
    return {"success": True, "episodes": items, "count": len(items)}


def get_episode(path: str) -> Dict[str, Any]:
    if not path or not str(path).strip():
        return {"success": False, "error": "path is required for get."}
    p = Path(path).expanduser()
    if not p.is_absolute():
        # Allow bare filename under episodes dir
        p = get_episodes_dir() / p
    try:
        resolved = p.resolve()
        resolved.relative_to(get_hermes_home().resolve())
    except (OSError, ValueError):
        return {"success": False, "error": "path must stay under HERMES_HOME."}
    if not resolved.exists() or not resolved.is_file():
        return {"success": False, "error": f"Episode not found: {resolved}"}
    try:
        text = resolved.read_text(encoding="utf-8")
    except OSError as exc:
        return {"success": False, "error": f"Failed to read episode: {exc}"}
    meta, body = _parse_frontmatter(text)
    return {
        "success": True,
        "path": str(resolved),
        "meta": meta,
        "content": body.strip(),
    }


# ---------------------------------------------------------------------------
# Tool entry point
# ---------------------------------------------------------------------------


def episode_tool(
    action: str = "recall",
    content: Optional[str] = None,
    evidence: Optional[str] = None,
    source: Optional[str] = None,
    tags: Optional[Any] = None,
    query: Optional[str] = None,
    k: int = 5,
    path: Optional[str] = None,
    limit: int = 20,
) -> str:
    """Dispatch episode tool actions. Always returns a JSON string."""
    if not _episodes_enabled():
        return tool_error(
            "Episode memory is disabled (memory.episodes_enabled: false).",
            success=False,
        )

    action = (action or "").strip().lower()
    tag_list: Optional[List[str]] = None
    if isinstance(tags, str):
        tag_list = [t.strip() for t in tags.split(",") if t.strip()]
    elif isinstance(tags, list):
        tag_list = [str(t).strip() for t in tags if str(t).strip()]

    if action == "remember":
        result = remember_episode(
            content=content or "",
            evidence=evidence or "",
            source=source or "agent",
            tags=tag_list,
        )
    elif action == "recall":
        result = recall_episodes(query=query or "", k=k, source=source)
    elif action == "list":
        result = list_episodes(limit=limit)
    elif action == "get":
        result = get_episode(path or "")
    elif action == "reindex":
        result = rebuild_episode_index()
    else:
        result = {
            "success": False,
            "error": f"Unknown action '{action}'. Use: remember, recall, list, get, reindex.",
        }
    return json.dumps(result, ensure_ascii=False)


def check_episode_requirements() -> bool:
    return True


EPISODE_SCHEMA = {
    "name": "episode",
    "description": (
        "Evidence-gated episode memory for durable lessons. Episodes are stored under "
        f"{display_hermes_home()}/memories/episodes/ and are NOT injected into the system "
        "prompt (unlike the memory tool's MEMORY.md / USER.md). Use this when a lesson "
        "should survive for months without burning the always-on memory budget.\n\n"
        "ACTIONS:\n"
        "- remember: write a lesson. Requires content + evidence (file path, commit SHA, "
        "PR/issue URL, or log path). Optional source and tags.\n"
        "- recall: FTS search over episodes (and configured HERMES_HOME markdown roots). "
        "Pass query, optional k (default 5), optional source filter "
        "(episodes|memories|skills).\n"
        "- list: recent episodes.\n"
        "- get: read one episode by path or filename.\n"
        "- reindex: rebuild the local FTS index.\n\n"
        "WHEN: verified checkpoints only (after a fix is proven, a review is accepted, "
        "or the user asks to remember a durable lesson). The one-month test: will this "
        "still be true and useful in a month?\n\n"
        "SKIP: status/queue/phase snapshots, unverified claims, secrets, chat replay "
        "(use session_search), standing preferences (use memory), procedures (use skills).\n\n"
        "Does not require Honcho or any external memory provider."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["remember", "recall", "list", "get", "reindex"],
                "description": "Episode action to perform.",
            },
            "content": {
                "type": "string",
                "description": "Lesson text for action=remember.",
            },
            "evidence": {
                "type": "string",
                "description": (
                    "REQUIRED for remember: file path, commit SHA, PR/issue URL, or log path "
                    "that proves the lesson."
                ),
            },
            "source": {
                "type": "string",
                "description": (
                    "For remember: who wrote it (e.g. agent, user, review). "
                    "For recall: optional source filter (episodes, memories, skills)."
                ),
            },
            "tags": {
                "description": "Optional tags for remember (array of strings, or comma-separated).",
            },
            "query": {
                "type": "string",
                "description": "Search query for action=recall.",
            },
            "k": {
                "type": "integer",
                "description": "Max recall hits (default 5, max 20).",
            },
            "path": {
                "type": "string",
                "description": "Episode path or filename for action=get.",
            },
            "limit": {
                "type": "integer",
                "description": "Max items for action=list (default 20).",
            },
        },
        "required": ["action"],
    },
}


registry.register(
    name="episode",
    toolset="memory",
    schema=EPISODE_SCHEMA,
    handler=lambda args, **kw: episode_tool(
        action=args.get("action", "recall"),
        content=args.get("content"),
        evidence=args.get("evidence"),
        source=args.get("source"),
        tags=args.get("tags"),
        query=args.get("query"),
        k=args.get("k", 5) or 5,
        path=args.get("path"),
        limit=args.get("limit", 20) or 20,
    ),
    check_fn=check_episode_requirements,
    emoji="📎",
)
