"""MongoDB tools for querying and writing documents via pymongo.

Registers four LLM-callable tools (gated on ``MONGODB_URI``):

- ``mongo_list_collections`` -- list databases or collections
- ``mongo_find`` -- find documents with filter/projection/sort/limit
- ``mongo_aggregate`` -- run an aggregation pipeline
- ``mongo_write`` -- insert, update, or delete documents

The connection string is read from ``MONGODB_URI`` in ``~/.hermes/.env``.
Non-secret limits live under ``mongodb:`` in ``config.yaml``.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Dict, List, Optional

from hermes_cli.config import cfg_get, load_config
from tools.lazy_deps import FeatureUnavailable, ensure
from tools.registry import registry, tool_error

logger = logging.getLogger(__name__)

_NAME_RE = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")
_HARD_MAX_ROWS = 1000
_client = None


def _check_mongodb_available() -> bool:
    """Tool is only available when MONGODB_URI is set."""
    return bool(os.getenv("MONGODB_URI"))


def _check_mongodb_write_available() -> bool:
    if not _check_mongodb_available():
        return False
    try:
        cfg = load_config()
        return not bool(cfg_get(cfg, "mongodb", "read_only", default=False))
    except Exception:
        return True


def _validate_name(name: str, label: str) -> Optional[str]:
    if not name or not name.strip():
        return f"{label} is required"
    if not _NAME_RE.fullmatch(name.strip()):
        return f"Invalid {label}: use letters, digits, and underscores only"
    return None


def _mongodb_settings() -> Dict[str, Any]:
    try:
        cfg = load_config()
    except Exception:
        cfg = {}
    max_rows = cfg_get(cfg, "mongodb", "max_rows", default=100)
    try:
        max_rows = int(max_rows)
    except (TypeError, ValueError):
        max_rows = 100
    max_rows = max(1, min(max_rows, _HARD_MAX_ROWS))
    default_db = str(cfg_get(cfg, "mongodb", "default_database", default="") or "").strip()
    return {"max_rows": max_rows, "default_database": default_db}


def _resolve_database(database: Optional[str]) -> tuple[Optional[str], Optional[str]]:
    db = (database or "").strip()
    if not db:
        db = _mongodb_settings()["default_database"]
    if not db:
        return None, "database is required (or set mongodb.default_database in config.yaml)"
    err = _validate_name(db, "database")
    if err:
        return None, err
    return db, None


def _parse_json(raw: Optional[str], field: str, default: Any) -> tuple[Any, Optional[str]]:
    if raw is None or not str(raw).strip():
        return default, None
    try:
        return json.loads(raw), None
    except json.JSONDecodeError as exc:
        return default, f"Invalid JSON in {field}: {exc}"


def _get_client():
    global _client
    try:
        ensure("tool.mongodb", prompt=False)
    except FeatureUnavailable as exc:
        raise RuntimeError(str(exc)) from exc
    from pymongo import MongoClient

    uri = os.getenv("MONGODB_URI", "")
    if not uri:
        raise RuntimeError("MONGODB_URI is not set")
    if _client is None:
        _client = MongoClient(uri, serverSelectionTimeoutMS=5000)
    return _client


def _serialize_doc(doc: Any) -> Any:
    from bson import json_util

    return json.loads(json_util.dumps(doc))


def _serialize_docs(docs: List[Any]) -> List[Any]:
    return [_serialize_doc(doc) for doc in docs]


def _mongo_error(exc: Exception) -> str:
    return tool_error(f"MongoDB error: {exc}")


def mongo_list_collections(database: str = "") -> str:
    """List databases or collections."""
    if not _check_mongodb_available():
        return tool_error("MONGODB_URI is not configured")

    db_name = (database or "").strip()
    try:
        client = _get_client()
        if not db_name:
            names = sorted(client.list_database_names())
            return json.dumps({"databases": names, "count": len(names)}, ensure_ascii=False)

        err = _validate_name(db_name, "database")
        if err:
            return tool_error(err)
        names = sorted(client[db_name].list_collection_names())
        return json.dumps(
            {"database": db_name, "collections": names, "count": len(names)},
            ensure_ascii=False,
        )
    except RuntimeError as exc:
        return tool_error(str(exc))
    except Exception as exc:
        logger.exception("mongo_list_collections failed")
        return _mongo_error(exc)


def mongo_find(
    database: str = "",
    collection: str = "",
    filter: str = "",
    projection: str = "",
    sort: str = "",
    limit: int = 0,
) -> str:
    """Find documents in a collection."""
    if not _check_mongodb_available():
        return tool_error("MONGODB_URI is not configured")

    db_name, err = _resolve_database(database)
    if err:
        return tool_error(err)
    coll_err = _validate_name(collection, "collection")
    if coll_err:
        return tool_error(coll_err)

    query, err = _parse_json(filter, "filter", {})
    if err:
        return tool_error(err)
    if not isinstance(query, dict):
        return tool_error("filter must be a JSON object")

    proj, err = _parse_json(projection, "projection", None)
    if err:
        return tool_error(err)
    if proj is not None and not isinstance(proj, dict):
        return tool_error("projection must be a JSON object")

    sort_spec, err = _parse_json(sort, "sort", None)
    if err:
        return tool_error(err)
    if sort_spec is not None and not isinstance(sort_spec, dict):
        return tool_error("sort must be a JSON object")

    settings = _mongodb_settings()
    row_limit = int(limit) if limit else settings["max_rows"]
    row_limit = max(1, min(row_limit, settings["max_rows"]))

    try:
        client = _get_client()
        cursor = client[db_name][collection].find(query, proj)
        if sort_spec:
            cursor = cursor.sort(list(sort_spec.items()))
        docs = list(cursor.limit(row_limit))
        return json.dumps(
            {
                "database": db_name,
                "collection": collection,
                "count": len(docs),
                "limit": row_limit,
                "documents": _serialize_docs(docs),
            },
            ensure_ascii=False,
        )
    except RuntimeError as exc:
        return tool_error(str(exc))
    except Exception as exc:
        logger.exception("mongo_find failed")
        return _mongo_error(exc)


def mongo_aggregate(
    database: str = "",
    collection: str = "",
    pipeline: str = "",
    limit: int = 0,
) -> str:
    """Run an aggregation pipeline."""
    if not _check_mongodb_available():
        return tool_error("MONGODB_URI is not configured")

    db_name, err = _resolve_database(database)
    if err:
        return tool_error(err)
    coll_err = _validate_name(collection, "collection")
    if coll_err:
        return tool_error(coll_err)

    stages, err = _parse_json(pipeline, "pipeline", None)
    if err:
        return tool_error(err)
    if not isinstance(stages, list):
        return tool_error("pipeline must be a JSON array of aggregation stages")

    settings = _mongodb_settings()
    row_limit = int(limit) if limit else settings["max_rows"]
    row_limit = max(1, min(row_limit, settings["max_rows"]))

    safe_pipeline = list(stages)
    if not safe_pipeline or "$limit" not in json.dumps(safe_pipeline):
        safe_pipeline.append({"$limit": row_limit})

    try:
        client = _get_client()
        docs = list(client[db_name][collection].aggregate(safe_pipeline))
        if len(docs) > row_limit:
            docs = docs[:row_limit]
        return json.dumps(
            {
                "database": db_name,
                "collection": collection,
                "count": len(docs),
                "limit": row_limit,
                "results": _serialize_docs(docs),
            },
            ensure_ascii=False,
        )
    except RuntimeError as exc:
        return tool_error(str(exc))
    except Exception as exc:
        logger.exception("mongo_aggregate failed")
        return _mongo_error(exc)


def mongo_write(
    action: str = "",
    database: str = "",
    collection: str = "",
    filter: str = "",
    document: str = "",
    documents: str = "",
) -> str:
    """Insert, update, or delete documents."""
    if not _check_mongodb_write_available():
        if _check_mongodb_available():
            return tool_error("MongoDB writes are disabled (mongodb.read_only=true)")
        return tool_error("MONGODB_URI is not configured")

    act = (action or "").strip().lower()
    if act not in {"insert", "update", "delete"}:
        return tool_error("action must be one of: insert, update, delete")

    db_name, err = _resolve_database(database)
    if err:
        return tool_error(err)
    coll_err = _validate_name(collection, "collection")
    if coll_err:
        return tool_error(coll_err)

    try:
        client = _get_client()
        coll = client[db_name][collection]

        if act == "insert":
            if documents.strip():
                payload, err = _parse_json(documents, "documents", None)
                if err:
                    return tool_error(err)
                if not isinstance(payload, list):
                    return tool_error("documents must be a JSON array")
                result = coll.insert_many(payload)
                return json.dumps(
                    {
                        "action": "insert",
                        "inserted_count": len(result.inserted_ids),
                        "inserted_ids": [str(i) for i in result.inserted_ids],
                    },
                    ensure_ascii=False,
                )

            payload, err = _parse_json(document, "document", None)
            if err:
                return tool_error(err)
            if not isinstance(payload, dict):
                return tool_error("document must be a JSON object")
            result = coll.insert_one(payload)
            return json.dumps(
                {
                    "action": "insert",
                    "inserted_count": 1,
                    "inserted_id": str(result.inserted_id),
                },
                ensure_ascii=False,
            )

        query, err = _parse_json(filter, "filter", None)
        if err:
            return tool_error(err)
        if not isinstance(query, dict) or not query:
            return tool_error("filter must be a non-empty JSON object for update/delete")

        if act == "update":
            update_doc, err = _parse_json(document, "document", None)
            if err:
                return tool_error(err)
            if not isinstance(update_doc, dict):
                return tool_error("document must be a JSON object for update")
            result = coll.update_many(query, update_doc)
            return json.dumps(
                {
                    "action": "update",
                    "matched_count": result.matched_count,
                    "modified_count": result.modified_count,
                },
                ensure_ascii=False,
            )

        result = coll.delete_many(query)
        return json.dumps(
            {
                "action": "delete",
                "deleted_count": result.deleted_count,
            },
            ensure_ascii=False,
        )
    except RuntimeError as exc:
        return tool_error(str(exc))
    except Exception as exc:
        logger.exception("mongo_write failed")
        return _mongo_error(exc)


MONGO_LIST_COLLECTIONS_SCHEMA = {
    "name": "mongo_list_collections",
    "description": (
        "List MongoDB databases or collections. Omit database to list databases; "
        "provide database to list collections in that database."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "database": {
                "type": "string",
                "description": (
                    "Database name. Omit to list all databases; set to list "
                    "collections within that database."
                ),
            },
        },
        "required": [],
    },
}

MONGO_FIND_SCHEMA = {
    "name": "mongo_find",
    "description": (
        "Find documents in a MongoDB collection. Returns matching documents as JSON."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "database": {
                "type": "string",
                "description": "Database name (defaults to mongodb.default_database in config).",
            },
            "collection": {
                "type": "string",
                "description": "Collection name.",
            },
            "filter": {
                "type": "string",
                "description": "MongoDB query filter as a JSON object string (default: {}).",
            },
            "projection": {
                "type": "string",
                "description": "Optional field projection as a JSON object string.",
            },
            "sort": {
                "type": "string",
                "description": "Optional sort spec as a JSON object string (field -> 1 or -1).",
            },
            "limit": {
                "type": "integer",
                "description": "Max documents to return (capped by mongodb.max_rows in config).",
            },
        },
        "required": ["collection"],
    },
}

MONGO_AGGREGATE_SCHEMA = {
    "name": "mongo_aggregate",
    "description": "Run a MongoDB aggregation pipeline and return results as JSON.",
    "parameters": {
        "type": "object",
        "properties": {
            "database": {
                "type": "string",
                "description": "Database name (defaults to mongodb.default_database in config).",
            },
            "collection": {
                "type": "string",
                "description": "Collection name.",
            },
            "pipeline": {
                "type": "string",
                "description": "Aggregation pipeline as a JSON array of stage objects.",
            },
            "limit": {
                "type": "integer",
                "description": "Max documents to return (capped by mongodb.max_rows in config).",
            },
        },
        "required": ["collection", "pipeline"],
    },
}

MONGO_WRITE_SCHEMA = {
    "name": "mongo_write",
    "description": (
        "Insert, update, or delete documents in a MongoDB collection. "
        "Disabled when mongodb.read_only is true in config."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "description": "One of: insert, update, delete.",
            },
            "database": {
                "type": "string",
                "description": "Database name (defaults to mongodb.default_database in config).",
            },
            "collection": {
                "type": "string",
                "description": "Collection name.",
            },
            "filter": {
                "type": "string",
                "description": "Query filter as JSON object (required for update/delete).",
            },
            "document": {
                "type": "string",
                "description": "Document or update operators as JSON object (insert/update).",
            },
            "documents": {
                "type": "string",
                "description": "Array of documents as JSON (insert_many alternative to document).",
            },
        },
        "required": ["action", "collection"],
    },
}


registry.register(
    name="mongo_list_collections",
    toolset="mongodb",
    schema=MONGO_LIST_COLLECTIONS_SCHEMA,
    handler=lambda args, **kw: mongo_list_collections(database=args.get("database", "")),
    check_fn=_check_mongodb_available,
    requires_env=["MONGODB_URI"],
    emoji="🍃",
)

registry.register(
    name="mongo_find",
    toolset="mongodb",
    schema=MONGO_FIND_SCHEMA,
    handler=lambda args, **kw: mongo_find(
        database=args.get("database", ""),
        collection=args.get("collection", ""),
        filter=args.get("filter", ""),
        projection=args.get("projection", ""),
        sort=args.get("sort", ""),
        limit=args.get("limit", 0),
    ),
    check_fn=_check_mongodb_available,
    requires_env=["MONGODB_URI"],
    emoji="🍃",
)

registry.register(
    name="mongo_aggregate",
    toolset="mongodb",
    schema=MONGO_AGGREGATE_SCHEMA,
    handler=lambda args, **kw: mongo_aggregate(
        database=args.get("database", ""),
        collection=args.get("collection", ""),
        pipeline=args.get("pipeline", ""),
        limit=args.get("limit", 0),
    ),
    check_fn=_check_mongodb_available,
    requires_env=["MONGODB_URI"],
    emoji="🍃",
)

registry.register(
    name="mongo_write",
    toolset="mongodb",
    schema=MONGO_WRITE_SCHEMA,
    handler=lambda args, **kw: mongo_write(
        action=args.get("action", ""),
        database=args.get("database", ""),
        collection=args.get("collection", ""),
        filter=args.get("filter", ""),
        document=args.get("document", ""),
        documents=args.get("documents", ""),
    ),
    check_fn=_check_mongodb_write_available,
    requires_env=["MONGODB_URI"],
    emoji="🍃",
)
