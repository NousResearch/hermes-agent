"""Observed extraction delivery facts, independent of page freshness claims."""

from datetime import datetime, timezone


def retrieval_time():
    return datetime.now(timezone.utc).isoformat()


def new_extract_facts(requested_count):
    return {
        "requested_backend": None, "requested_count": requested_count,
        "cache_status": "bypass", "provider_call_attempted": False,
        "fallback_attempted": False, "fallback_used": False,
    }


def mark_fetched(results, provider, facts, *, rescued=False):
    """Annotate private row facts before cache writes and final projection."""
    timestamp = retrieval_time()
    for row in results:
        if row.get("error") or not (row.get("raw_content") or row.get("content")):
            continue
        metadata = row.get("metadata")
        served = metadata.get("_hermes_served_by") if isinstance(metadata, dict) else None
        row["_hermes_extract_retrieved_at"] = timestamp
        # A third-party rescue without a reported vendor is explicitly unknown.
        row["_hermes_extract_served_by"] = served or (None if rescued else provider)
        if rescued or (served and served != provider):
            facts["fallback_attempted"] = True
            facts["fallback_used"] = True


def build_extract_provenance(results, facts):
    successful = [row for row in results if not row.get("error") and (row.get("content") or row.get("raw_content"))]
    times = [row.get("_hermes_extract_retrieved_at") for row in successful]
    known_times = [value for value in times if isinstance(value, str) and value]
    vendors = {row.get("_hermes_extract_served_by") for row in successful}
    known_vendors = sorted(value for value in vendors if isinstance(value, str) and value)
    return {
        **facts,
        "served_by": (known_vendors[0] if len(known_vendors) == 1 else known_vendors or None),
        "serving_backend_complete": bool(successful) and None not in vendors,
        "served_by_source": "wrapper_or_provider_reported",
        "retrieved_at": min(known_times) if successful and len(known_times) == len(successful) else None,
        "served_at": retrieval_time(),
        "fetch_succeeded": any(not row.get("cached") for row in successful),
        "returned_count": len(results), "success_count": len(successful),
        "failure_count": len(results) - len(successful),
        "evidence_scope": "extracted_content", "page_freshness_verified": False,
    }
