#!/usr/bin/env python3
"""Tiny first slice of the profile-scoped semantic index.

This tool deliberately starts with a single staged text payload instead of a
repo crawler.  It proves the contract: dry-run first, explicit external OpenAI
embedding, local LanceDB persistence, and inspectable DB state.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from hermes_constants import get_hermes_home

logger = logging.getLogger(__name__)

DEFAULT_TABLE = "semantic_chunks"
DEFAULT_MODEL = "text-embedding-3-small"
DEFAULT_PAYLOAD = (
    "Hermes semantic index smoke payload: a tiny codebase memory chunk used "
    "to verify OpenAI embeddings and local LanceDB storage."
)
MAX_STAGE_CHARS = 2_000

_SECRET_PATTERNS = (
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    re.compile(r"\bsk-[A-Za-z0-9_\-]{16,}"),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9_]{20,}"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"(?i)\b(api[_-]?key|token|secret|password)\s*[:=]\s*['\"]?[^'\"\s]{12,}"),
)


def _json(data: Dict[str, Any]) -> str:
    return json.dumps(data, ensure_ascii=False)


def _load_config() -> Dict[str, Any]:
    try:
        from hermes_cli.config import load_config
        cfg = load_config() or {}
        return cfg.get("semantic_index") or {}
    except Exception:
        logger.debug("semantic_index config load failed", exc_info=True)
        return {}


def _index_dir(path_override: Optional[str] = None) -> Path:
    if path_override and str(path_override).strip():
        return Path(path_override).expanduser()
    cfg_path = str(_load_config().get("path") or "").strip()
    if cfg_path:
        return Path(cfg_path).expanduser()
    return get_hermes_home() / "semantic" / "lancedb"


def _table_name(table: Optional[str] = None) -> str:
    raw = str(table or _load_config().get("table") or DEFAULT_TABLE).strip()
    if not raw:
        return DEFAULT_TABLE
    return re.sub(r"[^A-Za-z0-9_]", "_", raw)[:64] or DEFAULT_TABLE


def _embedding_model(model: Optional[str] = None) -> str:
    return str(model or _load_config().get("model") or DEFAULT_MODEL).strip() or DEFAULT_MODEL


def _embedding_dimensions(dimensions: Optional[int] = None) -> Optional[int]:
    value = dimensions if dimensions is not None else _load_config().get("dimensions")
    if value in (None, "", 0):
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _payload_text(payload: Optional[str]) -> str:
    text = DEFAULT_PAYLOAD if payload is None or not str(payload).strip() else str(payload)
    return text.strip()


def _content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def _stable_id(corpus: str, uri: str, text: str) -> str:
    digest = _content_hash(f"{corpus}\n{uri}\n{_content_hash(text)}")
    return digest[:32]


def _secret_warning(text: str) -> Optional[str]:
    for pattern in _SECRET_PATTERNS:
        if pattern.search(text):
            return "payload looks like it may contain credentials; refusing to send it to OpenAI"
    return None


def _preview(text: str, limit: int = 240) -> str:
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3] + "..."


def _record_for_payload(
    *,
    payload: str,
    vector: List[float],
    corpus: str,
    uri: str,
    model: str,
    provider: str = "openai",
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    now = time.time()
    return {
        "id": _stable_id(corpus, uri, payload),
        "corpus": corpus,
        "source_type": "manual_payload",
        "uri": uri,
        "content_hash": _content_hash(payload),
        "embedding_provider": provider,
        "embedding_model": model,
        "dimensions": len(vector),
        "text": payload,
        "metadata_json": json.dumps(metadata or {}, sort_keys=True, ensure_ascii=False),
        "vector": [float(v) for v in vector],
        "indexed_at": now,
    }


def _import_lancedb():
    try:
        import lancedb  # type: ignore
        return lancedb
    except ImportError:
        from tools.lazy_deps import ensure
        ensure("tool.semantic_index", prompt=False)
        import lancedb  # type: ignore
        return lancedb


def _table_names(db: Any) -> List[str]:
    names = db.table_names()
    return list(names() if callable(names) else names)


def _sample_rows(table: Any, limit: int = 3) -> List[Dict[str, Any]]:
    if hasattr(table, "head"):
        raw = table.head(limit)
        if hasattr(raw, "to_pylist"):
            return [dict(row) for row in raw.to_pylist()]
        if hasattr(raw, "to_dict"):
            return [dict(row) for row in raw.to_dict(orient="records")]
        return [dict(row) for row in raw]
    return [dict(row) for row in table.limit(limit).to_list()]


def _add_record(lancedb_module: Any, db_path: Path, table: str, record: Dict[str, Any]) -> Dict[str, Any]:
    db_path.mkdir(parents=True, exist_ok=True)
    db = lancedb_module.connect(str(db_path))
    if table in _table_names(db):
        tbl = db.open_table(table)
        tbl.add([record])
    else:
        tbl = db.create_table(table, data=[record])
    try:
        count = int(tbl.count_rows())
    except Exception:
        count = None
    return {"table": table, "row_count": count}


def _embed_openai(
    text: str,
    *,
    model: str,
    dimensions: Optional[int],
    base_url: Optional[str],
    api_key: Optional[str],
    timeout: int,
    openai_client_cls: Any = None,
) -> Dict[str, Any]:
    key = (api_key or os.getenv("OPENAI_API_KEY") or "").strip()
    if not key:
        raise RuntimeError("OPENAI_API_KEY is required for action='stage' or action='query'")

    if openai_client_cls is None:
        from openai import OpenAI
        openai_client_cls = OpenAI

    client = openai_client_cls(
        api_key=key,
        base_url=(base_url or os.getenv("OPENAI_BASE_URL") or None),
        timeout=timeout,
        max_retries=0,
    )
    kwargs: Dict[str, Any] = {"model": model, "input": [text]}
    if dimensions:
        kwargs["dimensions"] = dimensions
    response = client.embeddings.create(**kwargs)
    data = sorted(response.data, key=lambda item: item.index)
    vector = list(data[0].embedding)
    usage = getattr(response, "usage", None)
    return {
        "vector": vector,
        "usage": {
            "prompt_tokens": getattr(usage, "prompt_tokens", None),
            "total_tokens": getattr(usage, "total_tokens", None),
        },
    }


def _stage_payload(
    *,
    payload: str,
    corpus: str,
    uri: str,
    table: str,
    db_path: Path,
    model: str,
    dimensions: Optional[int],
    metadata: Optional[Dict[str, Any]],
    base_url: Optional[str],
    api_key: Optional[str],
    timeout: int,
    lancedb_module: Any = None,
    openai_client_cls: Any = None,
) -> Dict[str, Any]:
    warning = _secret_warning(payload)
    if warning:
        return {"success": False, "error": warning}
    if len(payload) > MAX_STAGE_CHARS:
        return {
            "success": False,
            "error": f"payload is {len(payload)} chars; max for this first-stage tool is {MAX_STAGE_CHARS}",
        }

    embedded = _embed_openai(
        payload,
        model=model,
        dimensions=dimensions,
        base_url=base_url,
        api_key=api_key,
        timeout=timeout,
        openai_client_cls=openai_client_cls,
    )
    vector = embedded["vector"]
    record = _record_for_payload(
        payload=payload,
        vector=vector,
        corpus=corpus,
        uri=uri,
        model=model,
        metadata=metadata,
    )
    lancedb_module = lancedb_module or _import_lancedb()
    storage = _add_record(lancedb_module, db_path, table, record)
    return {
        "success": True,
        "action": "stage",
        "dry_run": False,
        "provider": "openai",
        "model": model,
        "dimensions": len(vector),
        "usage": embedded["usage"],
        "db_path": str(db_path),
        "table": table,
        "row_count": storage["row_count"],
        "record": {
            "id": record["id"],
            "corpus": corpus,
            "uri": uri,
            "content_hash": record["content_hash"],
            "text_preview": _preview(payload),
        },
    }


def _query_index(
    *,
    query: str,
    table: str,
    db_path: Path,
    model: str,
    dimensions: Optional[int],
    limit: int,
    base_url: Optional[str],
    api_key: Optional[str],
    timeout: int,
    lancedb_module: Any = None,
    openai_client_cls: Any = None,
) -> Dict[str, Any]:
    if not query.strip():
        return {"success": False, "error": "query is required for action='query'"}
    lancedb_module = lancedb_module or _import_lancedb()
    db = lancedb_module.connect(str(db_path))
    if table not in _table_names(db):
        return {"success": False, "error": f"table '{table}' does not exist", "db_path": str(db_path)}
    embedded = _embed_openai(
        query,
        model=model,
        dimensions=dimensions,
        base_url=base_url,
        api_key=api_key,
        timeout=timeout,
        openai_client_cls=openai_client_cls,
    )
    rows = db.open_table(table).search(embedded["vector"]).limit(max(1, min(int(limit or 5), 20))).to_list()
    results = []
    for row in rows:
        vector = row.pop("vector", None)
        results.append({
            **row,
            "text_preview": _preview(str(row.get("text") or "")),
            "vector_dimensions": len(vector) if isinstance(vector, list) else row.get("dimensions"),
        })
    return {
        "success": True,
        "action": "query",
        "provider": "openai",
        "model": model,
        "db_path": str(db_path),
        "table": table,
        "count": len(results),
        "results": results,
        "usage": embedded["usage"],
    }


def _stats(table: str, db_path: Path, lancedb_module: Any = None) -> Dict[str, Any]:
    try:
        lancedb_module = lancedb_module or _import_lancedb()
    except Exception as exc:
        return {
            "success": True,
            "action": "stats",
            "lancedb_available": False,
            "db_path": str(db_path),
            "table": table,
            "error": str(exc),
        }
    db = lancedb_module.connect(str(db_path))
    tables = _table_names(db)
    if table not in tables:
        return {
            "success": True,
            "action": "stats",
            "lancedb_available": True,
            "db_path": str(db_path),
            "tables": tables,
            "table": table,
            "table_exists": False,
            "row_count": 0,
        }
    tbl = db.open_table(table)
    sample = _sample_rows(tbl, limit=3)
    for row in sample:
        vector = row.pop("vector", None)
        row["vector_dimensions"] = len(vector) if hasattr(vector, "__len__") else row.get("dimensions")
        row["text_preview"] = _preview(str(row.get("text") or ""))
        row.pop("text", None)
    return {
        "success": True,
        "action": "stats",
        "lancedb_available": True,
        "db_path": str(db_path),
        "tables": tables,
        "table": table,
        "table_exists": True,
        "row_count": int(tbl.count_rows()),
        "sample": sample,
    }


def semantic_index(
    action: str = "dry_run",
    payload: str = None,
    query: str = "",
    corpus: str = "manual",
    uri: str = "manual://semantic-index-smoke",
    metadata: Dict[str, Any] = None,
    table: str = None,
    db_path: str = None,
    model: str = None,
    dimensions: int = None,
    limit: int = 5,
    embed: bool = False,
    base_url: str = None,
    api_key: str = None,
    timeout: int = 30,
    lancedb_module: Any = None,
    openai_client_cls: Any = None,
) -> str:
    """Dry-run, stage, query, or inspect the local LanceDB semantic index."""
    action_norm = str(action or "dry_run").strip().lower()
    table_name = _table_name(table)
    model_name = _embedding_model(model)
    dims = _embedding_dimensions(dimensions)
    path = _index_dir(db_path)
    text = _payload_text(payload)
    corpus_norm = str(corpus or "manual").strip() or "manual"
    uri_norm = str(uri or "manual://semantic-index-smoke").strip() or "manual://semantic-index-smoke"

    if action_norm in {"dryrun", "preview"}:
        action_norm = "dry_run"

    if action_norm == "dry_run":
        return _json({
            "success": True,
            "action": "dry_run",
            "dry_run": True,
            "would_call_openai": False,
            "would_write_lancedb": False,
            "provider": "openai",
            "model": model_name,
            "dimensions": dims,
            "db_path": str(path),
            "table": table_name,
            "payload": {
                "corpus": corpus_norm,
                "uri": uri_norm,
                "char_count": len(text),
                "content_hash": _content_hash(text),
                "id": _stable_id(corpus_norm, uri_norm, text),
                "text_preview": _preview(text),
                "secret_blocker": _secret_warning(text),
            },
            "next_step": "Call semantic_index(action='stage', embed=true, payload=...) to send this tiny payload to OpenAI and store it in LanceDB.",
        })

    if action_norm == "stage":
        if not embed:
            return _json({
                "success": False,
                "error": "stage requires embed=true so external OpenAI usage is explicit",
                "dry_run_hint": "Call semantic_index(action='dry_run', payload=...) first to inspect the payload.",
            })
        return _json(_stage_payload(
            payload=text,
            corpus=corpus_norm,
            uri=uri_norm,
            table=table_name,
            db_path=path,
            model=model_name,
            dimensions=dims,
            metadata=metadata,
            base_url=base_url,
            api_key=api_key,
            timeout=int(timeout or 30),
            lancedb_module=lancedb_module,
            openai_client_cls=openai_client_cls,
        ))

    if action_norm == "query":
        return _json(_query_index(
            query=str(query or ""),
            table=table_name,
            db_path=path,
            model=model_name,
            dimensions=dims,
            limit=limit,
            base_url=base_url,
            api_key=api_key,
            timeout=int(timeout or 30),
            lancedb_module=lancedb_module,
            openai_client_cls=openai_client_cls,
        ))

    if action_norm == "stats":
        return _json(_stats(table_name, path, lancedb_module=lancedb_module))

    return _json({"success": False, "error": f"Unknown action '{action}'. Use dry_run, stage, query, or stats."})


def check_semantic_index_requirements() -> bool:
    return get_hermes_home().exists()


SEMANTIC_INDEX_SCHEMA = {
    "name": "semantic_index",
    "description": (
        "Dry-run and stage tiny text payloads into the local LanceDB semantic index. "
        "The default action is dry_run and never calls OpenAI or writes LanceDB. "
        "Use action='stage' with embed=true only after inspecting the payload; this "
        "sends the text to OpenAI embeddings and stores the resulting vector locally. "
        "Use action='query' to embed a query and search the table, and action='stats' "
        "to inspect the vector database."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["dry_run", "stage", "query", "stats"],
                "description": "dry_run previews only; stage embeds+writes; query searches; stats inspects LanceDB.",
                "default": "dry_run",
            },
            "payload": {
                "type": "string",
                "description": "Tiny text payload to preview or stage. Omit for a built-in smoke payload.",
            },
            "query": {
                "type": "string",
                "description": "Semantic query text for action='query'.",
            },
            "embed": {
                "type": "boolean",
                "description": "Required true for action='stage' to make OpenAI usage explicit.",
                "default": False,
            },
            "corpus": {"type": "string", "description": "Logical corpus name, e.g. 'manual' or 'codebase'."},
            "uri": {"type": "string", "description": "Stable source URI for the staged payload."},
            "metadata": {"type": "object", "description": "Small JSON metadata object stored with the chunk."},
            "table": {"type": "string", "description": "LanceDB table name. Defaults to semantic_chunks."},
            "db_path": {"type": "string", "description": "Override LanceDB path. Defaults to profile semantic/lancedb."},
            "model": {"type": "string", "description": "OpenAI embedding model. Defaults to text-embedding-3-small."},
            "dimensions": {"type": "integer", "description": "Optional OpenAI embedding dimensions override."},
            "limit": {"type": "integer", "description": "Max query results, clamped to [1, 20].", "default": 5},
        },
        "required": [],
    },
}


from tools.registry import registry

registry.register(
    name="semantic_index",
    toolset="semantic_index",
    schema=SEMANTIC_INDEX_SCHEMA,
    handler=lambda args, **kw: semantic_index(
        action=args.get("action", "dry_run"),
        payload=args.get("payload"),
        query=args.get("query", ""),
        corpus=args.get("corpus", "manual"),
        uri=args.get("uri", "manual://semantic-index-smoke"),
        metadata=args.get("metadata"),
        table=args.get("table"),
        db_path=args.get("db_path"),
        model=args.get("model"),
        dimensions=args.get("dimensions"),
        limit=args.get("limit", 5),
        embed=args.get("embed", False),
        base_url=args.get("base_url"),
        api_key=args.get("api_key"),
        timeout=args.get("timeout", 30),
        lancedb_module=kw.get("lancedb_module"),
        openai_client_cls=kw.get("openai_client_cls"),
    ),
    check_fn=check_semantic_index_requirements,
    emoji="",
)
