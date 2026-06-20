#!/usr/bin/env python3
"""
Semantic (hybrid BM25 + vector) retrieval for session_search — #44075.

FTS5 keyword search only finds past conversations when the query shares
exact words with them ("smart home" never finds the Tuya-bulbs session).
This module adds an optional vector half:

    query ──► embedding ──► cosine KNN over message_embeddings (sqlite-vec)
          └─► FTS5 BM25 (existing search_messages)
                          └─► weighted Reciprocal Rank Fusion ──► ranked rows

Everything is opt-in and degrades to the existing FTS5-only behaviour:
``hybrid_search()`` returns ``None`` — caller falls back — whenever

  - ``session_search.semantic`` is false in config.yaml (the default),
  - sqlite-vec is not installed and cannot be lazy-installed,
  - no embedding provider resolves, or
  - the embedding call fails (network, auth, unsupported endpoint).

Embeddings are generated outside the message-write hot path: a bounded
batch of pending messages is embedded opportunistically per search call
(``index_pending``), and ``scripts/backfill_session_embeddings.py`` does
the one-time full backfill. Embedding providers reuse the auxiliary
provider resolution (any OpenAI-compatible ``/embeddings`` endpoint).
"""

import logging
import struct
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Defaults mirrored in hermes_cli/config.py DEFAULT_CONFIG["session_search"].
_DEFAULTS: Dict[str, Any] = {
    "semantic": False,
    "embedding_provider": "",          # "" = auto-detect via provider chain
    "embedding_model": "text-embedding-3-small",
    "embedding_base_url": "",          # direct OpenAI-compatible endpoint
    "embedding_api_key": "",           # key for embedding_base_url
    "hybrid_weight_vector": 0.7,
    "hybrid_weight_bm25": 0.3,
    "min_similarity": 0.25,            # drop vector hits below this cosine similarity
    "index_batch_size": 96,            # pending messages embedded per search
}

# Embedding endpoints reject over-long inputs (text-embedding-3-* caps at
# 8191 tokens). Truncating loses tail content of giant messages but keeps
# the request valid; the FTS5 half still matches the full text.
_MAX_EMBED_CHARS = 8000

_RRF_K = 60  # standard reciprocal-rank-fusion damping constant

# Warn-once flags so a misconfigured provider logs one line per process,
# not one per search call.
_warned_no_client = False
_warned_embed_failed = False

_client_cache: Dict[tuple, Any] = {}


def get_semantic_config() -> Dict[str, Any]:
    """``session_search`` section of config.yaml merged over defaults."""
    cfg = dict(_DEFAULTS)
    try:
        from hermes_cli.config import load_config_readonly
        section = load_config_readonly().get("session_search") or {}
        if isinstance(section, dict):
            for key in cfg:
                if section.get(key) is not None and key in section:
                    cfg[key] = section[key]
    except Exception as exc:
        logger.debug("Could not load session_search config: %s", exc)
    return cfg


def semantic_enabled(cfg: Optional[Dict[str, Any]] = None) -> bool:
    cfg = cfg or get_semantic_config()
    return bool(cfg.get("semantic"))


def _ensure_sqlite_vec() -> bool:
    """Import sqlite-vec, lazy-installing it if the user allows that."""
    try:
        import sqlite_vec  # noqa: F401
        return True
    except ImportError:
        pass
    try:
        from tools.lazy_deps import ensure
        # prompt=False: never raise a blocking input() prompt mid-session.
        ensure("tool.session_semantic", prompt=False)
        import sqlite_vec  # noqa: F401
        return True
    except Exception as exc:
        logger.debug("sqlite-vec unavailable: %s", exc)
        return False


def _get_embedding_client(cfg: Dict[str, Any]):
    """Resolve an OpenAI-compatible client exposing ``.embeddings``.

    Reuses the auxiliary provider router; an explicit
    ``embedding_base_url``/``embedding_api_key`` pair takes precedence over
    the named provider, matching the auxiliary.<task> slot convention.
    Returns None when nothing resolves (caller falls back to FTS-only).
    """
    cache_key = (
        cfg.get("embedding_provider") or "auto",
        cfg.get("embedding_model"),
        cfg.get("embedding_base_url") or "",
    )
    if cache_key in _client_cache:
        return _client_cache[cache_key]

    client = None
    try:
        from agent.auxiliary_client import resolve_provider_client
        # Empty strings mean "unset" — resolve_provider_client truthy-checks
        # both explicit_* params, so "" routes the same as None.
        client, _model = resolve_provider_client(
            str(cfg.get("embedding_provider") or "auto"),
            model=str(cfg.get("embedding_model") or ""),
            explicit_base_url=str(cfg.get("embedding_base_url") or ""),
            explicit_api_key=str(cfg.get("embedding_api_key") or ""),
        )
    except Exception as exc:
        logger.debug("Embedding provider resolution failed: %s", exc)
    # Codex/Responses adapters only expose chat completions — an embeddings
    # endpoint is mandatory here.
    if client is not None and not hasattr(client, "embeddings"):
        logger.debug(
            "Resolved provider client has no embeddings endpoint; "
            "semantic session search needs an OpenAI-compatible /embeddings."
        )
        client = None
    _client_cache[cache_key] = client
    return client


def serialize_f32(vector: List[float]) -> bytes:
    """Pack a float list into the little-endian float32 BLOB sqlite-vec reads."""
    return struct.pack(f"<{len(vector)}f", *vector)


def embed_texts(
    texts: List[str], cfg: Optional[Dict[str, Any]] = None
) -> Optional[List[bytes]]:
    """Embed texts, returning float32 BLOBs in input order, or None on any
    failure (missing provider, network error, unsupported endpoint)."""
    global _warned_no_client, _warned_embed_failed
    if not texts:
        return []
    cfg = cfg or get_semantic_config()
    client = _get_embedding_client(cfg)
    if client is None:
        if not _warned_no_client:
            _warned_no_client = True
            logger.warning(
                "session_search.semantic is enabled but no embedding provider "
                "resolved — falling back to keyword-only search. Configure "
                "session_search.embedding_provider or embedding_base_url."
            )
        return None
    try:
        response = client.embeddings.create(
            model=cfg["embedding_model"],
            input=[t[:_MAX_EMBED_CHARS] for t in texts],
        )
        ordered = sorted(response.data, key=lambda d: d.index)
        if len(ordered) != len(texts):
            raise ValueError(
                f"embedding count mismatch: sent {len(texts)}, got {len(ordered)}"
            )
        return [serialize_f32(d.embedding) for d in ordered]
    except Exception as exc:
        if not _warned_embed_failed:
            _warned_embed_failed = True
            logger.warning(
                "Embedding call failed (%s) — session search falling back to "
                "keyword-only. Further failures logged at debug level.", exc
            )
        else:
            logger.debug("Embedding call failed: %s", exc)
        return None


def index_pending(
    db,
    cfg: Optional[Dict[str, Any]] = None,
    budget: Optional[int] = None,
) -> int:
    """Embed up to ``budget`` messages that have no embedding yet.

    Returns rows written. Read-only DB handles (cross-profile search opens
    other profiles' state.db with mode=ro) make the upsert fail — that is
    swallowed so searching existing embeddings still works.
    """
    cfg = cfg or get_semantic_config()
    model = cfg["embedding_model"]
    limit = budget if budget is not None else int(cfg["index_batch_size"])
    if limit <= 0:
        return 0
    rows = db.get_unembedded_messages(model, limit=limit)
    if not rows:
        return 0
    blobs = embed_texts([r["content"] for r in rows], cfg)
    if blobs is None:
        return 0
    payload = []
    for row, blob in zip(rows, blobs):
        dim = len(blob) // 4
        if dim:
            payload.append((row["id"], model, dim, blob))
    try:
        return db.upsert_message_embeddings(payload)
    except Exception as exc:
        logger.debug("Embedding upsert skipped (read-only DB?): %s", exc)
        return 0


def rrf_merge(
    fts_rows: List[Dict[str, Any]],
    vector_rows: List[Dict[str, Any]],
    weight_bm25: float,
    weight_vector: float,
    k: int = _RRF_K,
) -> List[Dict[str, Any]]:
    """Weighted Reciprocal Rank Fusion of two ranked result lists.

    score(msg) = w_bm25 / (k + rank_fts) + w_vec / (k + rank_vec)

    Rank-based fusion sidesteps normalising BM25 (unbounded, lower=better
    via FTS5 ``rank``) against cosine distance — the two scales never meet.
    Rows are deduped by message id; the FTS row wins ties for field content
    (it carries the highlighted snippet) and each surviving row is annotated
    with ``match_type``: "keyword", "semantic", or "both".
    """
    scores: Dict[Any, float] = {}
    merged: Dict[Any, Dict[str, Any]] = {}
    sources: Dict[Any, set] = {}

    for rank, row in enumerate(fts_rows):
        mid = row.get("id")
        scores[mid] = scores.get(mid, 0.0) + weight_bm25 / (k + rank + 1)
        merged.setdefault(mid, row)
        sources.setdefault(mid, set()).add("keyword")

    for rank, row in enumerate(vector_rows):
        mid = row.get("id")
        scores[mid] = scores.get(mid, 0.0) + weight_vector / (k + rank + 1)
        merged.setdefault(mid, row)
        sources.setdefault(mid, set()).add("semantic")

    out = []
    for mid, row in merged.items():
        entry = dict(row)
        kinds = sources[mid]
        entry["match_type"] = "both" if len(kinds) == 2 else next(iter(kinds))
        out.append((scores[mid], entry))
    out.sort(key=lambda pair: pair[0], reverse=True)
    return [entry for _score, entry in out]


def hybrid_search(
    db,
    query: str,
    role_filter: Optional[List[str]] = None,
    exclude_sources: Optional[List[str]] = None,
    limit: int = 20,
    sort: Optional[str] = None,
) -> Optional[List[Dict[str, Any]]]:
    """Hybrid BM25 + vector search over session messages.

    Returns merged rows shaped like ``SessionDB.search_messages()`` output,
    or ``None`` whenever the semantic path is unavailable — the caller then
    runs the existing FTS5-only search, preserving current behaviour
    exactly.
    """
    cfg = get_semantic_config()
    if not semantic_enabled(cfg):
        return None
    if not query or not query.strip():
        return None
    if not _ensure_sqlite_vec() or not db.vector_search_available():
        return None

    # Keep the index converging toward full coverage — bounded batch so a
    # search is never blocked behind a deep-history backfill.
    try:
        index_pending(db, cfg)
    except Exception as exc:
        logger.debug("Opportunistic embedding indexing failed: %s", exc)

    query_blobs = embed_texts([query.strip()], cfg)
    if not query_blobs:
        return None
    query_blob = query_blobs[0]

    fts_rows: List[Dict[str, Any]] = []
    try:
        fts_rows = db.search_messages(
            query=query,
            role_filter=role_filter,
            exclude_sources=exclude_sources,
            limit=limit,
        )
    except Exception as exc:
        logger.warning("FTS half of hybrid search failed: %s", exc)

    vector_rows = db.search_messages_by_vector(
        query_blob,
        model=cfg["embedding_model"],
        dim=len(query_blob) // 4,
        role_filter=role_filter,
        exclude_sources=exclude_sources,
        limit=limit,
    )
    # KNN always returns the nearest rows no matter how far they are — an
    # off-topic database would answer every query. Keep only hits above the
    # similarity floor so "no semantic match" stays expressible.
    min_similarity = float(cfg.get("min_similarity", 0.25))
    vector_rows = [
        row for row in vector_rows
        if (1.0 - (row.get("distance") or 0.0)) >= min_similarity
    ]
    if not vector_rows and not fts_rows:
        return []

    merged = rrf_merge(
        fts_rows,
        vector_rows,
        weight_bm25=float(cfg["hybrid_weight_bm25"]),
        weight_vector=float(cfg["hybrid_weight_vector"]),
    )
    # `sort` biases by time like the FTS-only path: timestamp primary,
    # fused relevance (already the list order) as tiebreaker.
    if sort in ("newest", "oldest"):
        merged.sort(
            key=lambda r: r.get("timestamp") or 0,
            reverse=(sort == "newest"),
        )
    return merged[:limit]


def backfill(
    db,
    cfg: Optional[Dict[str, Any]] = None,
    batch_size: int = 64,
    progress: Optional[Any] = None,
) -> int:
    """Embed every pending message in batches. Returns total rows embedded.

    Used by scripts/backfill_session_embeddings.py; stops cleanly when the
    provider starts failing (index_pending returns 0) rather than spinning.
    """
    cfg = cfg or get_semantic_config()
    total = 0
    while True:
        written = index_pending(db, cfg, budget=batch_size)
        if written == 0:
            break
        total += written
        if progress is not None:
            progress(total)
    return total
