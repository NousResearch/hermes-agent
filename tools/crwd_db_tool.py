"""CRWD database tool -- read-only lookups for the CRWD Coach agent.

Registers a single LLM-callable tool ``crwd_db`` (gated on ``CRWD_MONGO_URI``)
that reads CRWD's MongoDB data through a handful of purpose-built actions plus
one guarded custom-query escape hatch:

- ``list_active_gigs`` -- open gigs sorted by soonest end_date
- ``get_gig_details``  -- fuzzy-match gigs by name / free text, ranked candidates
- ``get_user``         -- look up one user by email, phone, or _id
- ``get_user_gigs``    -- campaigns a user is an active member of
- ``custom_query``     -- guarded find/count on the three known collections

Connection string comes from ``CRWD_MONGO_URI`` (in ``~/.hermes/.env``); the
database name from ``CRWD_MONGO_DB`` (default ``crwd_staging``). Read-only by
construction: there is no insert/update/delete code path in this module.
"""

from __future__ import annotations

import difflib
import json
import logging
import os
import re
from typing import Any, Dict, List, Optional

from tools.lazy_deps import FeatureUnavailable, ensure
from tools.registry import registry, tool_error

logger = logging.getLogger(__name__)

# --- Constants ---

_DB_DEFAULT = "crwd_staging"
_COLL_CRWDS = "crwds"
_COLL_USERS = "users"
_COLL_MEMBERS = "added_crwd_members"
_COLL_PURCHASES = "user_product_purchases"
_COLL_RECEIPTS = "receipt_upload_history"
_COLL_NOTIFS = "notifications"
_ALLOWED_COLLECTIONS = {
    _COLL_CRWDS, _COLL_USERS, _COLL_MEMBERS,
    _COLL_PURCHASES, _COLL_RECEIPTS, _COLL_NOTIFS,
}

_HARD_LIMIT = 20
_MAX_TIME_MS = 5000
_GIG_TOPN_CAP = 10
_MATCH_FLOOR = 0.3

_OBJECTID_RE = re.compile(r"^[a-fA-F0-9]{24}$")

# Fields that must never be returned from ``users``, regardless of projection.
_USER_SECRET_RE = re.compile(r"password|token|otp|secret", re.IGNORECASE)

# Explicit projections -- never return whole documents.
_USER_FIELDS = {
    "full_name": 1, "first_name": 1, "last_name": 1, "email": 1, "phone": 1,
    "bio": 1, "status": 1, "city": 1, "state": 1, "country": 1,
    "isBlocked": 1, "isDeleted": 1,
}
_GIG_FIELDS = {
    "name": 1, "description": 1, "gig_type": 1, "payout": 1, "price": 1,
    "gig_stores": 1, "start_date": 1, "end_date": 1, "type_of_work_proof": 1,
    "status": 1, "address": 1, "city": 1, "state": 1, "postal_code": 1,
    "image": 1, "isDeleted": 1,
}
_MEMBER_FIELDS = {
    "member": 1, "user_id": 1, "worker_id": 1, "crwd_id": 1, "status": 1,
    "isCompleted": 1, "hasPaid": 1, "isDeleted": 1, "createdAt": 1, "updatedAt": 1,
}
# What product a member is approved to buy for a gig (name + buy link).
_PURCHASE_FIELDS = {
    "product_name": 1, "product_url": 1, "store_name": 1, "crwd_id": 1,
    "crwd_name": 1, "gig_type": 1, "source": 1, "purchasedAt": 1, "createdAt": 1,
}
# Receipt/proof validation status (current pipeline). Omits the S3 key.
_RECEIPT_FIELDS = {
    "status": 1, "fail_reason": 1, "receipt_type": 1, "order_number": 1,
    "campaign_id": 1, "extracted_data": 1, "fraud_band_after": 1,
    "created_at": 1, "updated_at": 1,
}
# Account notifications. Never project the device/chat token fields.
_NOTIF_FIELDS = {
    "title": 1, "description": 1, "notificationType": 1, "isSeen": 1,
    "date": 1, "status": 1, "createdAt": 1,
}

# Noise words stripped before fuzzy scoring gig names.
_NOISE_WORDS = {
    "the", "a", "an", "gig", "campaign", "crwd", "and", "for", "with",
    "supplement", "supplements", "review", "reviews",
}

_client = None


# --- Availability / connection ---

def check_crwd_db_requirements() -> bool:
    """Tool is only available when CRWD_MONGO_URI is set."""
    return bool(os.getenv("CRWD_MONGO_URI"))


def _get_client():
    global _client
    try:
        ensure("tool.mongodb", prompt=False)
    except FeatureUnavailable as exc:
        raise RuntimeError(str(exc)) from exc
    from pymongo import MongoClient

    uri = os.getenv("CRWD_MONGO_URI", "")
    if not uri:
        raise RuntimeError("CRWD_MONGO_URI is not set")
    if _client is None:
        _client = MongoClient(uri, serverSelectionTimeoutMS=5000)
    return _client


def _db():
    db_name = os.getenv("CRWD_MONGO_DB", _DB_DEFAULT).strip() or _DB_DEFAULT
    return _get_client()[db_name]


def _oid(value: Any):
    """Return an ObjectId for a 24-hex string, else None."""
    from bson import ObjectId

    if isinstance(value, str) and _OBJECTID_RE.match(value):
        return ObjectId(value)
    return None


# --- Serialization ---

def _serialize_doc(doc: Any) -> Any:
    from bson import json_util

    return json.loads(json_util.dumps(doc))


def _serialize_docs(docs: List[Any]) -> List[Any]:
    return [_serialize_doc(doc) for doc in docs]


def _now():
    import datetime

    return datetime.datetime.now()


def _open_gig_filter() -> Dict[str, Any]:
    """Filter for currently-open gigs: not deleted, Active, end_date in future."""
    return {
        "isDeleted": {"$ne": True},
        "status": {"$regex": r"^active$", "$options": "i"},
        "end_date": {"$gte": _now()},
    }


def _effective_payout(gig: Dict[str, Any]) -> Any:
    """Top-level payout when set, else the max per-store payout_amount."""
    payout = gig.get("payout")
    try:
        if payout and float(payout) > 0:
            return payout
    except (TypeError, ValueError):
        pass
    amounts = []
    for store in gig.get("gig_stores") or []:
        amt = store.get("payout_amount")
        if isinstance(amt, (int, float)):
            amounts.append(amt)
    return max(amounts) if amounts else payout


def _slim_gig(gig: Dict[str, Any]) -> Dict[str, Any]:
    """Clean, coach-friendly gig summary (product names + links included)."""
    stores = []
    for store in gig.get("gig_stores") or []:
        stores.append({
            "store_name": store.get("store_name"),
            "payout_amount": store.get("payout_amount"),
            "products": [
                {"name": p.get("name"), "product_url": p.get("product_url")}
                for p in (store.get("products") or [])
            ],
        })
    out = {
        "_id": gig.get("_id"),
        "name": gig.get("name"),
        "description": gig.get("description"),
        "gig_type": gig.get("gig_type"),
        "status": gig.get("status"),
        "start_date": gig.get("start_date"),
        "end_date": gig.get("end_date"),
        "effective_payout": _effective_payout(gig),
        "type_of_work_proof": gig.get("type_of_work_proof"),
        "image": gig.get("image"),
        "stores": stores,
    }
    if gig.get("gig_type") == "irl":
        out["location"] = {
            "address": gig.get("address"), "city": gig.get("city"),
            "state": gig.get("state"), "postal_code": gig.get("postal_code"),
        }
    return _serialize_doc(out)


# --- Actions ---

def _list_active_gigs(limit: int = 5) -> str:
    row_limit = max(1, min(int(limit or 5), _HARD_LIMIT))
    cursor = (
        _db()[_COLL_CRWDS]
        .find(_open_gig_filter(), _GIG_FIELDS, max_time_ms=_MAX_TIME_MS)
        .sort("end_date", 1)
        .limit(row_limit)
    )
    items = [_slim_gig(g) for g in cursor]
    return json.dumps(
        {"_type": "gig_list", "items": items, "error": None}, ensure_ascii=False
    )


def _normalize(text: str) -> str:
    words = re.findall(r"[a-z0-9]+", (text or "").lower())
    kept = [w for w in words if w not in _NOISE_WORDS]
    return " ".join(kept or words)


def _score(query_norm: str, name: str, description: str = "") -> float:
    """Fuzzy score in [0, 1] of query against a gig name (+ description)."""
    if not query_norm:
        return 0.0
    name_norm = _normalize(name)
    ratio = difflib.SequenceMatcher(None, query_norm, name_norm).ratio()
    substring = 1.0 if name_norm and query_norm in name_norm else 0.0
    score = 0.6 * ratio + 0.4 * substring
    if description:
        desc_norm = _normalize(description)
        if desc_norm and query_norm in desc_norm:
            score = max(score, 0.5)
    return round(min(score, 1.0), 4)


def _get_gig_details(query: str, top_n: int = 3) -> str:
    query = (query or "").strip()
    top_n = max(1, min(int(top_n or 3), _GIG_TOPN_CAP))
    if not query:
        return tool_error("query is required for get_gig_details")

    # Exact _id short-circuit.
    oid = _oid(query)
    if oid is not None:
        gig = _db()[_COLL_CRWDS].find_one({"_id": oid}, _GIG_FIELDS, max_time_ms=_MAX_TIME_MS)
        if gig:
            item = _slim_gig(gig)
            item["score"] = 1.0
            return json.dumps(
                {"_type": "gig_match_candidates", "query": query, "items": [item]},
                ensure_ascii=False,
            )

    query_norm = _normalize(query)
    cursor = _db()[_COLL_CRWDS].find(
        _open_gig_filter(),
        {"name": 1, "description": 1, "status": 1, "end_date": 1},
        max_time_ms=_MAX_TIME_MS,
    )
    scored = []
    for gig in cursor:
        s = _score(query_norm, gig.get("name", ""), gig.get("description", ""))
        if s >= _MATCH_FLOOR:
            scored.append((s, gig))
    scored.sort(key=lambda t: t[0], reverse=True)

    items = []
    for s, gig in scored[:top_n]:
        items.append({
            "score": s,
            "_id": str(gig.get("_id")),
            "name": gig.get("name"),
            "status": gig.get("status"),
            "end_date": _serialize_doc(gig.get("end_date")),
        })
    return json.dumps(
        {"_type": "gig_match_candidates", "query": query, "items": items},
        ensure_ascii=False,
    )


def _get_user(identifier: str) -> str:
    identifier = (identifier or "").strip()
    if not identifier:
        return tool_error("identifier is required for get_user")

    oid = _oid(identifier)
    if oid is not None:
        query: Dict[str, Any] = {"_id": oid}
    elif "@" in identifier:
        query = {"email": identifier}
    else:
        query = {"phone": identifier}

    user = _db()[_COLL_USERS].find_one(query, _USER_FIELDS, max_time_ms=_MAX_TIME_MS)
    return json.dumps(
        {"_type": "user", "items": [_serialize_doc(user)] if user else [], "error": None},
        ensure_ascii=False,
    )


def _get_user_gigs(user_id: str, limit: int = 10) -> str:
    user_id = (user_id or "").strip()
    if not user_id:
        return tool_error("user_id is required for get_user_gigs")
    row_limit = max(1, min(int(limit or 10), _HARD_LIMIT))

    oid = _oid(user_id)
    id_values = [oid, user_id] if oid is not None else [user_id]
    member_filter = {
        "$or": [
            {"member": {"$in": id_values}},
            {"user_id": {"$in": id_values}},
            {"worker_id": {"$in": id_values}},
        ],
        "status": {"$regex": r"^active$", "$options": "i"},
        "isDeleted": {"$ne": True},
    }
    members = list(
        _db()[_COLL_MEMBERS]
        .find(member_filter, _MEMBER_FIELDS, max_time_ms=_MAX_TIME_MS)
        .limit(row_limit)
    )
    crwd_ids = [m["crwd_id"] for m in members if m.get("crwd_id") is not None]
    gigs_by_id = {}
    if crwd_ids:
        for gig in _db()[_COLL_CRWDS].find(
            {"_id": {"$in": crwd_ids}}, _GIG_FIELDS, max_time_ms=_MAX_TIME_MS
        ):
            gigs_by_id[str(gig["_id"])] = _slim_gig(gig)

    items = []
    for m in members:
        items.append({
            "membership": _serialize_doc(m),
            "gig": gigs_by_id.get(str(m.get("crwd_id"))),
        })
    return json.dumps(
        {"_type": "user_gigs", "items": items, "error": None}, ensure_ascii=False
    )


def _id_values(user_id: str) -> list:
    """Match values for a user id stored as either ObjectId or string."""
    oid = _oid(user_id)
    return [oid, user_id] if oid is not None else [user_id]


def _get_user_products(user_id: str, limit: int = 10) -> str:
    """Products a member is approved to buy for a gig (name + buy link)."""
    user_id = (user_id or "").strip()
    if not user_id:
        return tool_error("user_id is required for get_user_products")
    row_limit = max(1, min(int(limit or 10), _HARD_LIMIT))
    cursor = (
        _db()[_COLL_PURCHASES]
        .find(
            {"user_id": {"$in": _id_values(user_id)}, "isDeleted": {"$ne": True}},
            _PURCHASE_FIELDS, max_time_ms=_MAX_TIME_MS,
        )
        .sort("purchasedAt", -1)
        .limit(row_limit)
    )
    items = _serialize_docs(list(cursor))
    return json.dumps(
        {"_type": "user_products", "items": items, "error": None}, ensure_ascii=False
    )


def _get_user_receipts(user_id: str, limit: int = 10) -> str:
    """Receipt/proof upload validation status (pass/fail + reason)."""
    user_id = (user_id or "").strip()
    if not user_id:
        return tool_error("user_id is required for get_user_receipts")
    row_limit = max(1, min(int(limit or 10), _HARD_LIMIT))
    cursor = (
        _db()[_COLL_RECEIPTS]
        .find(
            {"user_id": {"$in": _id_values(user_id)}},
            _RECEIPT_FIELDS, max_time_ms=_MAX_TIME_MS,
        )
        .sort("created_at", -1)
        .limit(row_limit)
    )
    items = _serialize_docs(list(cursor))
    return json.dumps(
        {"_type": "user_receipts", "items": items, "error": None}, ensure_ascii=False
    )


def _get_user_notifications(user_id: str, limit: int = 10) -> str:
    """Recent account notifications for a member (secret fields excluded)."""
    user_id = (user_id or "").strip()
    if not user_id:
        return tool_error("user_id is required for get_user_notifications")
    row_limit = max(1, min(int(limit or 10), _HARD_LIMIT))
    cursor = (
        _db()[_COLL_NOTIFS]
        .find(
            {"to": {"$in": _id_values(user_id)}, "isDeleted": {"$ne": True}},
            _NOTIF_FIELDS, max_time_ms=_MAX_TIME_MS,
        )
        .sort("createdAt", -1)
        .limit(row_limit)
    )
    items = _serialize_docs(list(cursor))
    return json.dumps(
        {"_type": "user_notifications", "items": items, "error": None},
        ensure_ascii=False,
    )


# --- custom_query escape hatch ---

def _has_where(obj: Any) -> bool:
    if isinstance(obj, dict):
        if "$where" in obj:
            return True
        return any(_has_where(v) for v in obj.values())
    if isinstance(obj, list):
        return any(_has_where(v) for v in obj)
    return False


def _redact_secrets(doc: Any) -> Any:
    """Strip any password/token/otp/secret-looking key at any depth.

    Applied to every custom_query result, not just ``users`` -- e.g.
    ``notifications`` carries device/chat tokens.
    """
    if isinstance(doc, dict):
        return {
            k: _redact_secrets(v)
            for k, v in doc.items()
            if not _USER_SECRET_RE.search(str(k))
        }
    if isinstance(doc, list):
        return [_redact_secrets(v) for v in doc]
    return doc


def _custom_query(
    collection: str,
    operation: str,
    filter: Optional[Dict[str, Any]] = None,
    projection: Optional[Dict[str, Any]] = None,
    sort: Optional[Dict[str, Any]] = None,
    limit: int = 20,
) -> str:
    if collection not in _ALLOWED_COLLECTIONS:
        return tool_error(
            f"collection must be one of {sorted(_ALLOWED_COLLECTIONS)}"
        )
    if operation not in {"find", "count"}:
        return tool_error("operation must be 'find' or 'count'")
    filter = filter or {}
    if not isinstance(filter, dict):
        return tool_error("filter must be an object")
    if _has_where(filter):
        return tool_error("$where is not allowed")

    coll = _db()[collection]
    if operation == "count":
        total = coll.count_documents(filter, maxTimeMS=_MAX_TIME_MS)
        return json.dumps(
            {"_type": "custom_query_result", "operation": "count",
             "collection": collection, "count": total, "error": None},
            ensure_ascii=False,
        )

    row_limit = max(1, min(int(limit or _HARD_LIMIT), _HARD_LIMIT))
    proj = projection if isinstance(projection, dict) else None
    cursor = coll.find(filter, proj, max_time_ms=_MAX_TIME_MS)
    if isinstance(sort, dict) and sort:
        cursor = cursor.sort(list(sort.items()))
    docs = [_redact_secrets(d) for d in _serialize_docs(list(cursor.limit(row_limit)))]
    return json.dumps(
        {"_type": "custom_query_result", "operation": "find",
         "collection": collection, "items": docs, "count": len(docs), "error": None},
        ensure_ascii=False,
    )


# --- Router ---

def crwd_db_tool(args: Dict[str, Any], **_kw: Any) -> str:
    if not check_crwd_db_requirements():
        return tool_error("CRWD_MONGO_URI is not configured")

    action = str(args.get("action", "")).strip()
    try:
        if action == "list_active_gigs":
            return _list_active_gigs(limit=args.get("limit", 5))
        if action == "get_gig_details":
            return _get_gig_details(query=args.get("query", ""), top_n=args.get("top_n", 3))
        if action == "get_user":
            return _get_user(identifier=args.get("identifier", ""))
        if action == "get_user_gigs":
            return _get_user_gigs(user_id=args.get("user_id", ""), limit=args.get("limit", 10))
        if action == "get_user_products":
            return _get_user_products(user_id=args.get("user_id", ""), limit=args.get("limit", 10))
        if action == "get_user_receipts":
            return _get_user_receipts(user_id=args.get("user_id", ""), limit=args.get("limit", 10))
        if action == "get_user_notifications":
            return _get_user_notifications(user_id=args.get("user_id", ""), limit=args.get("limit", 10))
        if action == "custom_query":
            return _custom_query(
                collection=str(args.get("collection", "")),
                operation=str(args.get("operation", "")),
                filter=args.get("filter"),
                projection=args.get("projection"),
                sort=args.get("sort"),
                limit=args.get("limit", 20),
            )
        return tool_error(
            "Unknown action. Use: list_active_gigs, get_gig_details, get_user, "
            "get_user_gigs, custom_query"
        )
    except RuntimeError as exc:
        # Config/connection problems -- safe to surface the short message.
        return tool_error(str(exc))
    except Exception:
        logger.exception("crwd_db action %r failed", action)
        return tool_error("query failed")


# --- Schema ---

CRWD_DB_SCHEMA = {
    "name": "crwd_db",
    "description": (
        "Query CRWD's MongoDB data: gigs/campaigns, users, campaign "
        "membership, a member's approved products (buy links), their receipt/"
        "proof upload status, and their account notifications. Read-only. Use "
        "the specific action if it fits (list_active_gigs, get_gig_details, "
        "get_user, get_user_gigs, get_user_products, get_user_receipts, "
        "get_user_notifications); use custom_query only when none of the "
        "others answer the question. get_gig_details fuzzy-matches gig names "
        "and returns ranked candidates -- pick the _id you mean before using "
        "it elsewhere."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": [
                    "list_active_gigs", "get_gig_details", "get_user",
                    "get_user_gigs", "get_user_products", "get_user_receipts",
                    "get_user_notifications", "custom_query",
                ],
            },
            "limit": {"type": "integer", "description": "max rows (capped at 20)"},
            "identifier": {"type": "string", "description": "email, phone, or user _id (get_user)"},
            "user_id": {"type": "string", "description": "users._id (get_user_gigs, get_user_products, get_user_receipts, get_user_notifications)"},
            "query": {"type": "string", "description": "gig _id, name, or free text to fuzzy-match (get_gig_details)"},
            "top_n": {"type": "integer", "description": "max candidates to return, default 3, max 10 (get_gig_details)"},
            "collection": {"type": "string", "enum": [
                "crwds", "users", "added_crwd_members",
                "user_product_purchases", "receipt_upload_history", "notifications",
            ]},
            "operation": {"type": "string", "enum": ["find", "count"]},
            "filter": {"type": "object"},
            "projection": {"type": "object"},
            "sort": {"type": "object"},
        },
        "required": ["action"],
    },
}


# --- Registration ---

registry.register(
    name="crwd_db",
    toolset="crwd",
    schema=CRWD_DB_SCHEMA,
    handler=crwd_db_tool,
    check_fn=check_crwd_db_requirements,
    requires_env=["CRWD_MONGO_URI"],
    emoji="🛍️",
)
