"""Source identity and request-time provenance for web result envelopes."""
import json
import re
from datetime import datetime
from typing import Any, Optional

_CORE_SEARCH_PROVENANCE_FIELDS = frozenset(
    {
        "requested_backend",
        "served_by",
        "served_by_source",
        "fallback_used",
        "retrieved_at",
        "served_at",
        "cache",
        "evidence_scope",
        "page_fetched",
        "result_scope",
        "requested_limit",
        "fetched_result_count",
        "returned_count",
        "result_set_truncated",
        "result_set_truncation_scope",
        "upstream_cache_timestamp",
        "upstream_cache_timestamp_status",
        "limitations",
        "transformations",
    }
)


_PROVIDER_SELF_CERTIFICATION_FIELDS = frozenset(
    {"confidence", "fresh", "current", "verified", "authoritative"}
)


_MAX_PROVIDER_RESPONSE_NESTING = 100


_RFC3339_TIMESTAMP_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2}[Tt]\d{2}:\d{2}:\d{2}"
    r"(?:\.\d+)?(?:[Zz]|[+-]\d{2}:\d{2})$"
)


def _provenance_string_list(value: Any) -> list[str]:
    """Return only non-empty strings from a provider-owned list field."""
    if not isinstance(value, list):
        return []
    return [item.strip() for item in value if isinstance(item, str) and item.strip()]


def _merge_provenance_lists(*values: Any) -> list[str]:
    """Stable, duplicate-free merge for limitations and transformations."""
    merged = []
    seen = set()
    for value in values:
        for item in _provenance_string_list(value):
            if item not in seen:
                seen.add(item)
                merged.append(item)
    return merged


def _search_web_count(response: dict) -> int:
    data = response.get("data") if isinstance(response, dict) else None
    web = data.get("web") if isinstance(data, dict) else None
    return len(web) if isinstance(web, list) else 0


def _invalid_search_provider_response(reason: str) -> dict:
    """Return a bounded failure without reflecting an untrusted payload."""
    return {
        "success": False,
        "error": f"Invalid web search provider response: {reason}",
    }


def _validate_search_provider_response(response: Any) -> dict:
    """Fail closed unless a provider response satisfies the search envelope.

    Failure envelopes retain their legacy provider-defined shape. Successful
    envelopes must use the literal JSON boolean ``true`` and contain a list of
    result objects at ``data.web`` before they can be cached or receive the
    Hermes-owned truth contract.
    """
    if not isinstance(response, dict):
        return _invalid_search_provider_response("expected an object")
    if type(response.get("success")) is not bool:
        return _invalid_search_provider_response(
            "'success' must be a JSON boolean"
        )
    if response["success"] is False:
        return response
    data = response.get("data")
    if not isinstance(data, dict):
        return _invalid_search_provider_response(
            "successful response must contain an object at 'data'"
        )
    web = data.get("web")
    if not isinstance(web, list):
        return _invalid_search_provider_response(
            "successful response must contain a list at 'data.web'"
        )
    if any(not isinstance(item, dict) for item in web):
        return _invalid_search_provider_response(
            "every 'data.web' item must be an object"
        )
    try:
        # Normalize exactly what the JSON tool boundary can represent. This
        # converts nested tuples to arrays and rejects cycles, unsupported
        # objects, and NaN/Infinity before caching or provenance injection.
        return json.loads(
            json.dumps(response, ensure_ascii=False, allow_nan=False)
        )
    except (TypeError, ValueError, RecursionError):
        return _invalid_search_provider_response(
            "successful response must be JSON-compatible"
        )


def _is_rfc3339_timestamp(value: str) -> bool:
    """Validate Hermes' RFC 3339 subset (leap-second notation excluded)."""
    if not _RFC3339_TIMESTAMP_RE.fullmatch(value):
        return False
    if value[17:19] == "60":
        return False
    candidate = value[:10] + "T" + value[11:]
    if candidate[-1:] in {"Z", "z"}:
        candidate = candidate[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(candidate)
    except ValueError:
        return False
    return parsed.tzinfo is not None and parsed.utcoffset() is not None


def _strip_provider_self_certification_fields(
    value: Any,
    *,
    _depth: int = 0,
) -> tuple[Any, bool]:
    """Copy a provider payload while removing ambiguous bare truth claims.

    Exact, case-insensitive keys are removed at every level. Namespaced fields
    such as ``provider_confidence`` remain available for explicitly documented
    upstream metrics. Excessively nested or cyclic provider payloads fail
    closed instead of consuming unbounded recursion.
    """
    if _depth > _MAX_PROVIDER_RESPONSE_NESTING:
        raise ValueError("web search provider response nesting exceeds safety limit")
    if isinstance(value, dict):
        cleaned = {}
        omitted = False
        for key, item in value.items():
            if (
                isinstance(key, str)
                and key.strip().casefold()
                in _PROVIDER_SELF_CERTIFICATION_FIELDS
            ):
                omitted = True
                continue
            cleaned_item, item_omitted = _strip_provider_self_certification_fields(
                item,
                _depth=_depth + 1,
            )
            cleaned[key] = cleaned_item
            omitted = omitted or item_omitted
        return cleaned, omitted
    if isinstance(value, (list, tuple)):
        cleaned_list = []
        omitted = False
        for item in value:
            cleaned_item, item_omitted = _strip_provider_self_certification_fields(
                item,
                _depth=_depth + 1,
            )
            cleaned_list.append(cleaned_item)
            omitted = omitted or item_omitted
        return cleaned_list, omitted
    return value, False


def _inject_search_provenance(
    response: dict,
    *,
    requested_backend: str,
    requested_limit: int,
    fetched_result_count: int,
    retrieved_at: str,
    cache_status: str,
    cache_age_seconds: Optional[float] = None,
    cache_ttl_seconds: Optional[float] = None,
    fallback_used: bool = False,
) -> dict:
    """Add the core-owned truth contract to a successful search response.

    Providers may contribute stable fields (for example ``engine``, source
    date semantics, upstream cache reporting, limitations, and
    transformations). Hermes owns request-time routing, cache, timestamps,
    scope, and counts, and overwrites those fields on every call. Bare provider
    ``confidence``, ``current``, ``fresh``, ``verified``, and
    ``authoritative`` self-certification fields are removed.
    """
    if not isinstance(response, dict) or response.get("success") is not True:
        return response

    response, omitted_self_certification = (
        _strip_provider_self_certification_fields(response)
    )
    data = response.get("data")
    if not isinstance(data, dict):
        data = {}
    provider_provenance = data.get("provenance")
    if not isinstance(provider_provenance, dict):
        provider_provenance = {}

    legacy_served_by = data.get("served_by")
    served_by = (
        legacy_served_by.strip()
        if isinstance(legacy_served_by, str) and legacy_served_by.strip()
        else requested_backend
    )
    served_by_source = (
        "provider_reported"
        if isinstance(legacy_served_by, str) and legacy_served_by.strip()
        else "requested_backend_default"
    )
    raw_upstream_cache_timestamp = provider_provenance.get(
        "upstream_cache_timestamp"
    )
    if raw_upstream_cache_timestamp is None:
        upstream_cache_timestamp = None
        upstream_cache_timestamp_status = "not_reported_in_response"
    elif isinstance(raw_upstream_cache_timestamp, str):
        candidate_timestamp = raw_upstream_cache_timestamp.strip()
        if (
            _RFC3339_TIMESTAMP_RE.fullmatch(candidate_timestamp)
            and candidate_timestamp[17:19] == "60"
        ):
            upstream_cache_timestamp = None
            upstream_cache_timestamp_status = (
                "reported_unsupported_second_60"
            )
        elif _is_rfc3339_timestamp(candidate_timestamp):
            upstream_cache_timestamp = candidate_timestamp
            upstream_cache_timestamp_status = "reported_in_response"
        else:
            upstream_cache_timestamp = None
            upstream_cache_timestamp_status = "reported_invalid_rfc3339"
    else:
        upstream_cache_timestamp = None
        upstream_cache_timestamp_status = "reported_invalid_rfc3339"

    returned_count = _search_web_count(response)
    result_set_truncated = fetched_result_count > returned_count
    limitations = _merge_provenance_lists(
        provider_provenance.get("limitations"),
        ["page_not_fetched", "not_exhaustive"],
        ["upstream_cache_time_not_reported"]
        if upstream_cache_timestamp_status == "not_reported_in_response"
        else [],
        ["upstream_cache_timestamp_invalid_rfc3339"]
        if upstream_cache_timestamp_status == "reported_invalid_rfc3339"
        else [],
        ["upstream_cache_timestamp_second_60_unsupported"]
        if upstream_cache_timestamp_status
        == "reported_unsupported_second_60"
        else [],
    )
    transformations = _merge_provenance_lists(
        provider_provenance.get("transformations"),
        ["limit_slice"] if result_set_truncated else [],
        ["provider_self_certification_fields_omitted"]
        if omitted_self_certification
        else [],
    )

    cache_age = (
        float(cache_age_seconds)
        if cache_status == "hit" and cache_age_seconds is not None
        else None
    )
    cache_ttl = (
        float(cache_ttl_seconds)
        if cache_status in {"hit", "miss"} and cache_ttl_seconds is not None
        else None
    )
    provenance = {
        "requested_backend": requested_backend,
        "served_by": served_by,
        "served_by_source": served_by_source,
        "fallback_used": bool(fallback_used or served_by != requested_backend),
        "retrieved_at": retrieved_at,
        "served_at": _search_provenance_now(),
        "cache": {
            "layer": "hermes_process_memory",
            "status": cache_status,
            "age_seconds": cache_age,
            "ttl_seconds": cache_ttl,
            "key_dimensions": [
                "provider_name",
                "normalized_query",
                "bucketed_limit",
            ],
            "credential_identity_in_key": False,
            "locale_in_key": False,
            "provider_configuration_in_key": False,
        },
        "evidence_scope": "search_result_metadata_only",
        "page_fetched": False,
        "result_scope": "top_n",
        "requested_limit": requested_limit,
        "fetched_result_count": fetched_result_count,
        "returned_count": returned_count,
        "result_set_truncated": result_set_truncated,
        "result_set_truncation_scope": "hermes_bucket_slice_only",
        "upstream_cache_timestamp": upstream_cache_timestamp,
        "upstream_cache_timestamp_status": upstream_cache_timestamp_status,
        "limitations": limitations,
        "transformations": transformations,
    }
    for key, value in provider_provenance.items():
        if (
            key not in _CORE_SEARCH_PROVENANCE_FIELDS
            and not (
                isinstance(key, str)
                and key.strip().casefold() in _PROVIDER_SELF_CERTIFICATION_FIELDS
            )
        ):
            provenance[key] = value

    # Rebuild ``data`` so provenance stays ahead of potentially large result
    # lists when downstream context storage has to keep only a prefix.
    response["data"] = {
        "provenance": provenance,
        **{key: value for key, value in data.items() if key != "provenance"},
    }
    return response


def _search_provenance_now() -> str:
    """Late import avoids coupling web provider discovery to cache config."""
    from tools.web_result_cache import utc_now_iso

    return utc_now_iso()
