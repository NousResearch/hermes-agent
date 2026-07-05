"""CRWD database tool -- read-only lookups for the CRWD Coach agent.

Registers a single LLM-callable tool ``crwd_db`` (gated on ``CRWD_MONGO_URI``)
that reads CRWD's MongoDB data through a handful of purpose-built actions plus
one guarded custom-query escape hatch:

- ``list_active_gigs`` -- open gigs sorted by soonest end_date; pass ``user_id`` to
  exclude gigs the member already has a membership for
- ``get_gig_details``  -- fuzzy-match gigs by name / free text, ranked candidates
- ``get_user``         -- look up one user by email, phone, or _id
- ``get_user_gigs``    -- campaigns a user is an active member of
- ``get_user_gig_status`` -- per-gig stage + personalized next_step from progress data
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
_COLL_GIG_STORE_ORDERS = "gig_store_orders"
_COLL_GIG_PRODUCT_REVIEWS = "gig_product_reviews"
_COLL_ORDER_RECEIPT_REVIEWS = "order_receipt_reviews"
_ALLOWED_COLLECTIONS = {
    _COLL_CRWDS, _COLL_USERS, _COLL_MEMBERS,
    _COLL_PURCHASES, _COLL_RECEIPTS, _COLL_NOTIFS,
}
_GIG_STATUS_CAP = 5

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
    "isAccepted": 1, "isApproved": 1, "isCompleted": 1, "hasPaid": 1,
    "isDeleted": 1, "createdAt": 1, "updatedAt": 1,
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

def _get_enrolled_gig_ids(user_id: str) -> set[str]:
    """Gig _ids the user has any non-deleted membership row for."""
    user_id = (user_id or "").strip()
    if not user_id:
        return set()
    id_values = _id_values(user_id)
    member_filter = {
        "$or": [
            {"member": {"$in": id_values}},
            {"user_id": {"$in": id_values}},
            {"worker_id": {"$in": id_values}},
        ],
        "isDeleted": {"$ne": True},
    }
    cursor = _db()[_COLL_MEMBERS].find(
        member_filter, {"crwd_id": 1}, max_time_ms=_MAX_TIME_MS
    )
    enrolled: set[str] = set()
    for row in cursor:
        crwd_id = row.get("crwd_id")
        if crwd_id is not None:
            enrolled.add(str(crwd_id))
    return enrolled


def _list_active_gigs(limit: int = 5, user_id: str = "", offset: int = 0) -> str:
    row_limit = max(1, min(int(limit or 5), _HARD_LIMIT))
    row_offset = max(0, int(offset or 0))
    query: Dict[str, Any] = dict(_open_gig_filter())
    user_id = (user_id or "").strip()
    excluded_count = 0
    if user_id:
        enrolled_ids = _get_enrolled_gig_ids(user_id)
        excluded_count = len(enrolled_ids)
        enrolled_oids = [oid for gid in enrolled_ids if (oid := _oid(gid)) is not None]
        if enrolled_oids:
            query["_id"] = {"$nin": enrolled_oids}
    coll = _db()[_COLL_CRWDS]
    total = coll.count_documents(query, maxTimeMS=_MAX_TIME_MS)
    cursor = (
        coll.find(query, _GIG_FIELDS, max_time_ms=_MAX_TIME_MS)
        .sort("end_date", 1)
        .skip(row_offset)
        .limit(row_limit)
    )
    items = [_slim_gig(g) for g in cursor]
    next_offset = row_offset + len(items)
    has_more = next_offset < total
    payload: Dict[str, Any] = {
        "_type": "gig_list",
        "items": items,
        "error": None,
        "offset": row_offset,
        "limit": row_limit,
        "total": total,
        "has_more": has_more,
        "next_offset": next_offset if has_more else None,
    }
    if user_id:
        payload["excluded_enrolled_count"] = excluded_count
    return json.dumps(payload, ensure_ascii=False)


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


def _get_waitlisted_gigs(user_id: str, limit: int = 10) -> str:
    """Gigs the member applied for but has not been accepted into yet."""
    user_id = (user_id or "").strip()
    if not user_id:
        return tool_error("user_id is required for get_waitlisted_gigs")
    row_limit = max(1, min(int(limit or 10), _HARD_LIMIT))

    oid = _oid(user_id)
    id_values = [oid, user_id] if oid is not None else [user_id]
    member_filter = {
        "$or": [
            {"member": {"$in": id_values}},
            {"user_id": {"$in": id_values}},
            {"worker_id": {"$in": id_values}},
        ],
        "isDeleted": {"$ne": True},
        "isAccepted": False,
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
        {"_type": "waitlisted_gigs", "items": items, "error": None}, ensure_ascii=False
    )


def _get_user_gigs(user_id: str, limit: int = 10) -> str:
    user_id = (user_id or "").strip()
    if not user_id:
        return tool_error("user_id is required for get_user_gigs")
    row_limit = max(1, min(int(limit or 10), _HARD_LIMIT))

    members = list(
        _db()[_COLL_MEMBERS]
        .find(_joined_member_filter(user_id), _MEMBER_FIELDS, max_time_ms=_MAX_TIME_MS)
        .sort("updatedAt", -1)
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


def _member_or_filter(user_id: str) -> Dict[str, Any]:
    """Filter fragment matching a user id on member/user_id/worker_id fields."""
    id_values = _id_values(user_id)
    return {
        "$or": [
            {"member": {"$in": id_values}},
            {"user_id": {"$in": id_values}},
            {"worker_id": {"$in": id_values}},
        ],
    }


def _joined_member_filter(user_id: str) -> Dict[str, Any]:
    """Active/joined memberships — aligned with app-chatbot get_user_joined_gigs."""
    return {
        "$and": [
            _member_or_filter(user_id),
            {"isDeleted": {"$ne": True}},
            {
                "$or": [
                    {"isAccepted": True},
                    {"isApproved": True},
                    {"status": {"$in": ["Active", "Accepted", "Approved", "Joined"]}},
                ],
            },
        ],
    }


def _waitlisted_member_filter(user_id: str) -> Dict[str, Any]:
    return {
        **_member_or_filter(user_id),
        "isDeleted": {"$ne": True},
        "isAccepted": False,
    }


def _gig_type_key(gig: Dict[str, Any]) -> str:
    gt = str(gig.get("gig_type") or "").strip().lower()
    if gt in ("irl", "in_store", "live"):
        return "irl"
    if gt in ("web_based", "web", "online", "amazon"):
        return "web"
    return gt or "unknown"


def _first_buy_link(gig: Dict[str, Any], purchases: List[Dict[str, Any]]) -> Optional[str]:
    for row in purchases:
        url = row.get("product_url")
        if url:
            return str(url)
    for store in gig.get("gig_stores") or []:
        for product in store.get("products") or []:
            url = product.get("product_url")
            if url:
                return str(url)
    return None


def compute_gig_stage(
    membership: Dict[str, Any],
    gig: Dict[str, Any],
    *,
    purchases: List[Dict[str, Any]],
    store_orders: List[Dict[str, Any]],
    product_reviews: List[Dict[str, Any]],
    order_receipt_reviews: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Derive machine-readable stage + coach-facing next_step for one membership."""
    gig_name = str(gig.get("name") or "this gig").strip()
    gig_type = _gig_type_key(gig)
    buy_link = _first_buy_link(gig, purchases)

    is_accepted = membership.get("isAccepted")
    is_approved = membership.get("isApproved")
    has_paid = membership.get("hasPaid")
    rejection = membership.get("rejectionReason") or membership.get("rejectionNotes")

    progress: Dict[str, Any] = {
        "purchase_confirmed": bool(purchases),
        "receipt_submitted": False,
        "receipt_approved": False,
        "review_submitted": False,
        "review_approved": False,
    }

    if is_accepted is False:
        return {
            "stage": "waitlisted",
            "next_step": (
                f"You're on the waitlist for {gig_name} — we'll notify you once "
                "you're accepted."
            ),
            "progress": progress,
            "buy_link": buy_link,
            "handoff_recommended": False,
        }

    if rejection:
        return {
            "stage": "rejected",
            "next_step": (
                f"Your enrollment for {gig_name} was not approved. "
                "I'll loop in a human who can help."
            ),
            "progress": progress,
            "buy_link": buy_link,
            "handoff_recommended": True,
        }

    if not is_approved:
        return {
            "stage": "pending_approval",
            "next_step": (
                f"Your application for {gig_name} is pending approval — "
                "hang tight, we'll notify you when you're in."
            ),
            "progress": progress,
            "buy_link": buy_link,
            "handoff_recommended": False,
        }

    if not purchases:
        link_hint = f" Use your buy link: {buy_link}." if buy_link else ""
        return {
            "stage": "need_purchase",
            "next_step": (
                f"You're approved for {gig_name} — next, buy the product using "
                f"the gig's link in the app.{link_hint}"
            ),
            "progress": progress,
            "buy_link": buy_link,
            "handoff_recommended": False,
        }

    progress["purchase_confirmed"] = True

    if gig_type == "irl":
        if not store_orders:
            return {
                "stage": "need_receipt",
                "next_step": (
                    f"For {gig_name}, visit the store, buy the product, then upload "
                    "your receipt in the app."
                ),
                "progress": progress,
                "buy_link": buy_link,
                "handoff_recommended": False,
            }
        latest_order = store_orders[0]
        progress["receipt_submitted"] = bool(
            latest_order.get("receipt_file") or latest_order.get("receipt_files")
        )
        if latest_order.get("rejectionReason"):
            return {
                "stage": "receipt_rejected",
                "next_step": (
                    f"Your receipt for {gig_name} needs a human review — "
                    "I'll connect you with support."
                ),
                "progress": progress,
                "buy_link": buy_link,
                "handoff_recommended": True,
            }
        if progress["receipt_submitted"] and not latest_order.get("isApproved"):
            progress["receipt_submitted"] = True
            return {
                "stage": "receipt_review",
                "next_step": (
                    f"Your receipt for {gig_name} is being reviewed — "
                    "we'll notify you when it's approved."
                ),
                "progress": progress,
                "buy_link": buy_link,
                "handoff_recommended": False,
            }
        if latest_order.get("isApproved"):
            progress["receipt_approved"] = True

        if not product_reviews:
            return {
                "stage": "need_review",
                "next_step": (
                    f"Receipt approved for {gig_name}! Next: post your review/UGC "
                    "and submit the links in the app."
                ),
                "progress": progress,
                "buy_link": buy_link,
                "handoff_recommended": False,
            }
        latest_review = product_reviews[0]
        progress["review_submitted"] = bool(
            latest_review.get("review_link") or latest_review.get("ugc_post_link")
        )
        if latest_review.get("rejectionReason"):
            return {
                "stage": "review_rejected",
                "next_step": (
                    f"Your review submission for {gig_name} needs support — "
                    "I'll loop in a human."
                ),
                "progress": progress,
                "buy_link": buy_link,
                "handoff_recommended": True,
            }
        if progress["review_submitted"] and not latest_review.get("isApproved"):
            return {
                "stage": "review_review",
                "next_step": (
                    f"Your review for {gig_name} is under review — "
                    "we'll notify you when it's approved."
                ),
                "progress": progress,
                "buy_link": buy_link,
                "handoff_recommended": False,
            }
        if latest_review.get("isApproved"):
            progress["review_approved"] = True

    else:
        order_rows = [r for r in order_receipt_reviews if r.get("type") == "order_receipt"]
        review_rows = [r for r in order_receipt_reviews if r.get("type") == "review"]

        if not order_rows:
            return {
                "stage": "need_receipt",
                "next_step": (
                    f"For {gig_name}, order the product, then upload your order "
                    "receipt screenshot in the app."
                ),
                "progress": progress,
                "buy_link": buy_link,
                "handoff_recommended": False,
            }
        latest_order = order_rows[0]
        progress["receipt_submitted"] = bool(latest_order.get("order_receipt_file"))
        if not latest_order.get("isOrderApproved") and progress["receipt_submitted"]:
            return {
                "stage": "receipt_review",
                "next_step": (
                    f"Your order receipt for {gig_name} is being reviewed — "
                    "we'll notify you when it's approved."
                ),
                "progress": progress,
                "buy_link": buy_link,
                "handoff_recommended": False,
            }
        if latest_order.get("isOrderApproved"):
            progress["receipt_approved"] = True

        if not review_rows:
            return {
                "stage": "need_review",
                "next_step": (
                    f"Order approved for {gig_name}! Leave your review, then upload "
                    "the order + review screenshots in the app."
                ),
                "progress": progress,
                "buy_link": buy_link,
                "handoff_recommended": False,
            }
        latest_review = review_rows[0]
        progress["review_submitted"] = bool(
            latest_review.get("review") or latest_review.get("review_file")
        )
        if progress["review_submitted"] and str(latest_review.get("status") or "").lower() not in (
            "approved", "complete", "completed",
        ):
            if not latest_review.get("isOrderApproved"):
                return {
                    "stage": "review_review",
                    "next_step": (
                        f"Your review for {gig_name} is under review — "
                        "we'll notify you when it's approved."
                    ),
                    "progress": progress,
                    "buy_link": buy_link,
                    "handoff_recommended": False,
                }
        if latest_review.get("isOrderApproved") or str(
            latest_review.get("status") or ""
        ).lower() in ("approved", "complete", "completed"):
            progress["review_approved"] = True

    if has_paid:
        return {
            "stage": "paid",
            "next_step": (
                f"Payout for {gig_name} has been issued. If you don't see it yet, "
                "check your Dot payout link or ask me to loop in support."
            ),
            "progress": progress,
            "buy_link": buy_link,
            "handoff_recommended": False,
        }

    return {
        "stage": "awaiting_payout",
        "next_step": (
            f"All proof for {gig_name} is approved — payout typically lands in "
            "1–2 business days via Dot."
        ),
        "progress": progress,
        "buy_link": buy_link,
        "handoff_recommended": False,
    }


def _progress_for_crwd(
    user_id: str,
    crwd_id: Any,
) -> Dict[str, List[Dict[str, Any]]]:
    """Fetch purchase + proof rows for one gig."""
    id_values = _id_values(user_id)
    crwd_values = [crwd_id]
    if isinstance(crwd_id, str):
        oid = _oid(crwd_id)
        if oid is not None:
            crwd_values = [oid, crwd_id]

    db = _db()
    purchases = list(
        db[_COLL_PURCHASES]
        .find(
            {
                "user_id": {"$in": id_values},
                "crwd_id": {"$in": crwd_values},
                "isDeleted": {"$ne": True},
            },
            _PURCHASE_FIELDS,
            max_time_ms=_MAX_TIME_MS,
        )
        .sort("purchasedAt", -1)
        .limit(5)
    )
    store_orders = list(
        db[_COLL_GIG_STORE_ORDERS]
        .find(
            {"user_id": {"$in": id_values}, "crwd_id": {"$in": crwd_values}},
            {
                "receipt_file": 1, "receipt_files": 1, "isApproved": 1,
                "rejectionReason": 1, "reviewedAt": 1,
            },
            max_time_ms=_MAX_TIME_MS,
        )
        .sort("reviewedAt", -1)
        .limit(5)
    )
    product_reviews = list(
        db[_COLL_GIG_PRODUCT_REVIEWS]
        .find(
            {"user_id": {"$in": id_values}, "crwd_id": {"$in": crwd_values}},
            {
                "review_link": 1, "ugc_post_link": 1, "isApproved": 1,
                "rejectionReason": 1, "reviewedAt": 1,
            },
            max_time_ms=_MAX_TIME_MS,
        )
        .sort("reviewedAt", -1)
        .limit(5)
    )
    order_receipt_reviews = list(
        db[_COLL_ORDER_RECEIPT_REVIEWS]
        .find(
            {
                "order_generated_by": {"$in": id_values},
                "crwd_id": {"$in": crwd_values},
            },
            {
                "type": 1, "order_receipt_file": 1, "review": 1, "review_file": 1,
                "isOrderApproved": 1, "status": 1,
            },
            max_time_ms=_MAX_TIME_MS,
        )
        .limit(10)
    )
    return {
        "purchases": purchases,
        "store_orders": store_orders,
        "product_reviews": product_reviews,
        "order_receipt_reviews": order_receipt_reviews,
    }


def _filter_membership_by_gig_ref(
    members: List[Dict[str, Any]],
    gigs_by_id: Dict[str, Dict[str, Any]],
    *,
    crwd_id: str = "",
    gig_name: str = "",
) -> List[Dict[str, Any]]:
    """Narrow memberships to one gig when crwd_id or fuzzy gig_name is provided."""
    crwd_id = (crwd_id or "").strip()
    gig_name = (gig_name or "").strip()
    if crwd_id:
        return [m for m in members if str(m.get("crwd_id")) == crwd_id]
    if not gig_name:
        return members
    query_norm = _normalize(gig_name)
    matched = []
    for m in members:
        gid = str(m.get("crwd_id"))
        gig = gigs_by_id.get(gid) or {}
        name = gig.get("name") or ""
        if _score(query_norm, name) >= _MATCH_FLOOR:
            matched.append(m)
    return matched or members


def build_user_gig_status(
    user_id: str,
    *,
    crwd_id: str = "",
    gig_name: str = "",
    include_waitlisted: bool = False,
    limit: int = _GIG_STATUS_CAP,
) -> Dict[str, Any]:
    """Build gig status payload (dict) for one member — used by tool + prefetch hook."""
    user_id = (user_id or "").strip()
    if not user_id:
        return {"_type": "user_gig_status", "items": [], "error": "user_id is required"}

    row_limit = max(1, min(int(limit or _GIG_STATUS_CAP), _HARD_LIMIT))
    db = _db()

    member_filter = _joined_member_filter(user_id)
    members = list(
        db[_COLL_MEMBERS]
        .find(member_filter, _MEMBER_FIELDS, max_time_ms=_MAX_TIME_MS)
        .sort("updatedAt", -1)
        .limit(row_limit * 2)
    )

    waitlisted: List[Dict[str, Any]] = []
    if include_waitlisted:
        waitlisted = list(
            db[_COLL_MEMBERS]
            .find(_waitlisted_member_filter(user_id), _MEMBER_FIELDS, max_time_ms=_MAX_TIME_MS)
            .sort("updatedAt", -1)
            .limit(row_limit)
        )
        members = members + waitlisted

    crwd_ids = [m["crwd_id"] for m in members if m.get("crwd_id") is not None]
    gigs_by_id: Dict[str, Dict[str, Any]] = {}
    if crwd_ids:
        for gig in db[_COLL_CRWDS].find(
            {"_id": {"$in": crwd_ids}}, _GIG_FIELDS, max_time_ms=_MAX_TIME_MS
        ):
            gigs_by_id[str(gig["_id"])] = gig

    members = _filter_membership_by_gig_ref(
        members, gigs_by_id, crwd_id=crwd_id, gig_name=gig_name
    )[:row_limit]

    items = []
    for m in members:
        gid = m.get("crwd_id")
        gig = gigs_by_id.get(str(gid)) if gid is not None else None
        if not gig:
            continue
        prog = _progress_for_crwd(user_id, gid)
        stage_info = compute_gig_stage(
            m, gig,
            purchases=prog["purchases"],
            store_orders=prog["store_orders"],
            product_reviews=prog["product_reviews"],
            order_receipt_reviews=prog["order_receipt_reviews"],
        )
        items.append({
            "gig_id": str(gid),
            "gig_name": gig.get("name"),
            "gig_type": _gig_type_key(gig),
            "end_date": _serialize_doc(gig.get("end_date")),
            "membership": {
                "isAccepted": m.get("isAccepted"),
                "isApproved": m.get("isApproved"),
                "hasPaid": m.get("hasPaid"),
                "status": m.get("status"),
            },
            "progress": stage_info["progress"],
            "stage": stage_info["stage"],
            "next_step": stage_info["next_step"],
            "buy_link": stage_info.get("buy_link"),
            "handoff_recommended": stage_info.get("handoff_recommended", False),
        })

    return {
        "_type": "user_gig_status",
        "items": items,
        "active_gigs": items,
        "count": len(items),
        "error": None,
    }


def _get_user_gig_status(
    user_id: str,
    crwd_id: str = "",
    gig_name: str = "",
    include_waitlisted: bool = False,
    limit: int = _GIG_STATUS_CAP,
) -> str:
    payload = build_user_gig_status(
        user_id,
        crwd_id=crwd_id,
        gig_name=gig_name,
        include_waitlisted=include_waitlisted,
        limit=limit,
    )
    if payload.get("error"):
        return tool_error(str(payload["error"]))
    return json.dumps(payload, ensure_ascii=False)


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
            return _list_active_gigs(
                limit=args.get("limit", 5),
                user_id=args.get("user_id", ""),
                offset=args.get("offset", 0),
            )
        if action == "get_gig_details":
            return _get_gig_details(query=args.get("query", ""), top_n=args.get("top_n", 3))
        if action == "get_user":
            return _get_user(identifier=args.get("identifier", ""))
        if action == "get_user_gigs":
            return _get_user_gigs(user_id=args.get("user_id", ""), limit=args.get("limit", 10))
        if action == "get_waitlisted_gigs":
            return _get_waitlisted_gigs(
                user_id=args.get("user_id", ""), limit=args.get("limit", 10)
            )
        if action == "get_user_products":
            return _get_user_products(user_id=args.get("user_id", ""), limit=args.get("limit", 10))
        if action == "get_user_receipts":
            return _get_user_receipts(user_id=args.get("user_id", ""), limit=args.get("limit", 10))
        if action == "get_user_notifications":
            return _get_user_notifications(user_id=args.get("user_id", ""), limit=args.get("limit", 10))
        if action == "get_user_gig_status":
            return _get_user_gig_status(
                user_id=args.get("user_id", ""),
                crwd_id=args.get("crwd_id", ""),
                gig_name=args.get("gig_name", ""),
                include_waitlisted=bool(args.get("include_waitlisted")),
                limit=args.get("limit", _GIG_STATUS_CAP),
            )
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
            "get_user_gigs, get_waitlisted_gigs, get_user_gig_status, "
            "get_user_products, get_user_receipts, get_user_notifications, custom_query"
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
        "get_user, get_user_gigs, get_waitlisted_gigs, get_user_gig_status, "
        "get_user_products, "
        "get_user_receipts, get_user_notifications); use custom_query only when none of the "
        "others answer the question. list_active_gigs accepts user_id to "
        "exclude gigs the member already has a membership for, and offset for "
        "pagination; it returns has_more and next_offset for the next page. "
        "get_gig_details fuzzy-matches gig names and returns ranked candidates "
        "-- pick the _id you mean before using it elsewhere. "
        "get_waitlisted_gigs returns gigs the member applied for but is not "
        "yet accepted into (isAccepted false / pending approval). "
        "get_user_gig_status returns per-gig stage and personalized next_step "
        "from membership + proof progress."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": [
                    "list_active_gigs", "get_gig_details", "get_user",
                    "get_user_gigs", "get_waitlisted_gigs", "get_user_gig_status",
                    "get_user_products",
                    "get_user_receipts", "get_user_notifications", "custom_query",
                ],
            },
            "limit": {"type": "integer", "description": "max rows per page (capped at 20; list_active_gigs default 5)"},
            "offset": {
                "type": "integer",
                "description": (
                    "skip N results for pagination (list_active_gigs). "
                    "Use next_offset from the previous result to get the next page."
                ),
            },
            "identifier": {"type": "string", "description": "email, phone, or user _id (get_user)"},
            "user_id": {
                "type": "string",
                "description": (
                    "users._id. For list_active_gigs: exclude gigs the member "
                    "already has a membership for. Also used by get_user_gigs, "
                    "get_waitlisted_gigs, get_user_products, get_user_receipts, "
                    "get_user_notifications, get_user_gig_status."
                ),
            },
            "crwd_id": {
                "type": "string",
                "description": "Optional gig _id filter (get_user_gig_status)",
            },
            "gig_name": {
                "type": "string",
                "description": "Optional fuzzy gig name filter (get_user_gig_status)",
            },
            "include_waitlisted": {
                "type": "boolean",
                "description": "Include waitlisted memberships (get_user_gig_status)",
            },
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
