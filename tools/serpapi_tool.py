#!/usr/bin/env python3
"""Read-only SerpApi credential/capability and travel search tools.

The tools verify/use the ``SERPAPI_API_KEY`` credential without exposing it in
logs or tool output. Account probing calls SerpApi's free Account API endpoint,
which the SerpApi docs state does not count toward monthly search quota. Travel
searches are read-only SerpApi Search API calls and never book, reserve, or
purchase anything.
"""

from __future__ import annotations

import json
import logging
import os
import re
from datetime import datetime
from typing import Any, Dict, List, Optional

import requests

from tools.registry import registry

logger = logging.getLogger(__name__)

SERPAPI_KEY_ENV = "SERPAPI_API_KEY"
SERPAPI_ACCOUNT_URL = "https://serpapi.com/account.json"
SERPAPI_SEARCH_URL = "https://serpapi.com/search.json"
SERPAPI_TIMEOUT_SECONDS = 20
SERPAPI_SEARCH_TIMEOUT_SECONDS = 30
TRAVEL_PRICE_VOLATILITY_CAVEAT = (
    "Travel prices and availability are volatile and must be rechecked with the "
    "airline or provider before purchase. This tool is search-only and cannot "
    "book, reserve, hold, or purchase travel."
)
HOTELS_VOLATILITY_CAVEAT = (
    "Hotel prices and availability are volatile and must be rechecked with the "
    "provider before purchase. This tool is search-only and cannot book, "
    "reserve, or purchase hotels."
)

_CABIN_CLASS_TO_SERPAPI = {
    "economy": "1",
    "premium_economy": "2",
    "premium economy": "2",
    "business": "3",
    "first": "4",
}
_TRIP_TYPE_TO_SERPAPI = {
    "round_trip": "1",
    "round trip": "1",
    "return": "1",
    "one_way": "2",
    "one way": "2",
    "multi_city": "3",
    "multi city": "3",
}


def _missing_secret_payload(tool: str) -> Dict[str, Any]:
    """Return an actionable missing-secret error without secret values."""
    return {
        "success": False,
        "provider": "serpapi",
        "tool": tool,
        "error": (
            f"Missing {SERPAPI_KEY_ENV}. Store the SerpApi API key in "
            "Bitwarden Secrets Manager using that exact secret name, then "
            "restart or reload Hermes so the secret is present in the environment."
        ),
        "missing_secret": SERPAPI_KEY_ENV,
    }


def _redacted_account_payload(data: Dict[str, Any]) -> Dict[str, Any]:
    """Return non-secret account/capability fields from SerpApi's response."""
    allowed_fields = [
        "account_id",
        "account_email",
        "plan_id",
        "plan_name",
        "searches_per_month",
        "plan_searches_left",
        "extra_credits",
        "total_searches_left",
        "this_month_usage",
        "last_hour_searches",
        "account_rate_limit_per_hour",
    ]
    return {field: data[field] for field in allowed_fields if field in data}


def _normalize_iata(value: str, field_name: str) -> str:
    normalized = str(value or "").strip().upper()
    if not re.fullmatch(r"[A-Z0-9, ]{3,64}", normalized):
        raise ValueError(f"{field_name} must be an airport/city code such as AMS or JFK.")
    return normalized.replace(" ", "")


def _normalize_iso_date(value: str, field_name: str) -> str:
    text = str(value or "").strip()
    try:
        datetime.strptime(text, "%Y-%m-%d")
    except ValueError as exc:
        raise ValueError(f"{field_name} must be in YYYY-MM-DD format.") from exc
    return text


def _normalize_positive_int(value: Any, field_name: str, *, allow_zero: bool = False) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be an integer.") from exc
    minimum = 0 if allow_zero else 1
    if parsed < minimum:
        raise ValueError(f"{field_name} must be at least {minimum}.")
    return parsed


def _normalize_cabin_class(value: str) -> tuple[str, str]:
    key = str(value or "economy").strip().lower()
    if key not in _CABIN_CLASS_TO_SERPAPI:
        supported = ", ".join(sorted(_CABIN_CLASS_TO_SERPAPI))
        raise ValueError(f"cabin_class must be one of: {supported}.")
    canonical = key.replace(" ", "_")
    return canonical, _CABIN_CLASS_TO_SERPAPI[key]


def _normalize_trip_type(value: str, has_return_date: bool) -> tuple[str, str]:
    key = str(value or ("round_trip" if has_return_date else "one_way")).strip().lower()
    if key not in _TRIP_TYPE_TO_SERPAPI:
        supported = ", ".join(sorted(_TRIP_TYPE_TO_SERPAPI))
        raise ValueError(f"trip_type must be one of: {supported}.")
    canonical = key.replace(" ", "_")
    return canonical, _TRIP_TYPE_TO_SERPAPI[key]


def _extract_airport(segment: Dict[str, Any], key: str) -> Dict[str, Any]:
    airport = segment.get(key) or {}
    if not isinstance(airport, dict):
        airport = {}
    return {
        "name": airport.get("name"),
        "id": airport.get("id"),
        "time": airport.get("time"),
    }


def _normalize_flight_option(option: Dict[str, Any], *, currency: str, bucket: str, index: int) -> Dict[str, Any]:
    flights = option.get("flights") if isinstance(option.get("flights"), list) else []
    segments: List[Dict[str, Any]] = []
    airlines: List[str] = []
    for segment in flights:
        if not isinstance(segment, dict):
            continue
        airline = segment.get("airline")
        if airline and airline not in airlines:
            airlines.append(airline)
        segments.append(
            {
                "departure": _extract_airport(segment, "departure_airport"),
                "arrival": _extract_airport(segment, "arrival_airport"),
                "airline": airline,
                "flight_number": segment.get("flight_number"),
                "airplane": segment.get("airplane"),
                "duration_minutes": segment.get("duration"),
                "travel_class": segment.get("travel_class"),
            }
        )

    booking_link = option.get("booking_link") or option.get("link")
    return {
        "rank": index,
        "source_bucket": bucket,
        "price": {"amount": option.get("price"), "currency": currency},
        "type": option.get("type"),
        "provider": "Google Flights via SerpApi",
        "airlines": airlines,
        "total_duration_minutes": option.get("total_duration"),
        "segments": segments,
        "booking_link": booking_link,
        "booking_token_available": bool(option.get("booking_token")),
        "raw_metadata": {
            "extensions": option.get("extensions"),
            "carbon_emissions": option.get("carbon_emissions"),
            "airline_logo": option.get("airline_logo"),
            "layovers": option.get("layovers"),
        },
    }


def _safe_serpapi_error(payload: Dict[str, Any], status_code: int, fallback_context: str) -> str:
    message = str(payload.get("error") or payload.get("message") or "").strip()
    return message or f"SerpApi {fallback_context} failed with HTTP {status_code}."


def _safe_search_parameters(params: Dict[str, Any]) -> Dict[str, Any]:
    """Remove secret parameters before returning/debug-printing request metadata."""
    return {key: value for key, value in params.items() if key != "api_key"}


def _compact_raw_metadata(item: Dict[str, Any]) -> Dict[str, Any]:
    """Keep enough source data for debugging without echoing the full payload."""
    keys = [
        "type",
        "property_token",
        "serpapi_property_details_link",
        "gps_coordinates",
        "amenities",
        "images",
        "thumbnail",
    ]
    return {key: item[key] for key in keys if key in item}


def _extract_rate(item: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize SerpApi's Google Hotels rate shapes."""
    source = "rate_per_night"
    rate = item.get("rate_per_night")
    if rate is None:
        source = "total_rate"
        rate = item.get("total_rate")
    if rate is None:
        source = "price"
        rate = item.get("price")

    if isinstance(rate, dict):
        return {
            "lowest": rate.get("lowest"),
            "extracted_lowest": rate.get("extracted_lowest"),
            "currency": rate.get("currency"),
            "source": source,
        }
    return {
        "lowest": rate,
        "extracted_lowest": item.get("extracted_price"),
        "currency": None,
        "source": source,
    }


def _normalize_hotel(item: Dict[str, Any], *, check_in_date: str, check_out_date: str) -> Dict[str, Any]:
    """Normalize a SerpApi Google Hotels property into Hermes' stable shape."""
    return {
        "name": item.get("name"),
        "price_rate": _extract_rate(item),
        "dates": {"check_in": check_in_date, "check_out": check_out_date},
        "provider_source": item.get("source") or item.get("provider") or "Google Hotels via SerpApi",
        "link": item.get("link") or item.get("property_token"),
        "location": item.get("address") or item.get("location") or item.get("gps_coordinates"),
        "rating": item.get("overall_rating") or item.get("rating"),
        "reviews": item.get("reviews"),
        "hotel_class": item.get("hotel_class"),
        "raw_metadata": _compact_raw_metadata(item),
    }


def _serpapi_error_message(response, payload: Dict[str, Any], *, default: str) -> str:
    """Extract SerpApi error text while preserving clean fallback messages."""
    message = str(payload.get("error") or payload.get("message") or "").strip()
    if response.status_code in {401, 403}:
        return message or "SerpApi rejected the credential."
    return message or default


def check_serpapi_requirements() -> bool:
    """Return True when the SerpApi secret is present in the environment."""
    return bool(os.getenv(SERPAPI_KEY_ENV, "").strip())


def serpapi_read_only_probe() -> str:
    """Verify SerpApi credentials with a read-only, free account probe.

    Returns JSON. The API key is never included in the returned payload.
    """
    api_key = os.getenv(SERPAPI_KEY_ENV, "").strip()
    if not api_key:
        return json.dumps(_missing_secret_payload("serpapi_read_only_probe"))

    try:
        response = requests.get(
            SERPAPI_ACCOUNT_URL,
            params={"api_key": api_key},
            timeout=SERPAPI_TIMEOUT_SECONDS,
        )

        try:
            payload = response.json()
        except ValueError:
            payload = {}

        if response.status_code == 200:
            return json.dumps(
                {
                    "success": True,
                    "provider": "serpapi",
                    "tool": "serpapi_read_only_probe",
                    "capability": "SerpApi account/read-only search capability confirmed",
                    "endpoint": SERPAPI_ACCOUNT_URL,
                    "quota_cost": "free account API; not counted toward monthly search quota",
                    "account": _redacted_account_payload(payload),
                },
                ensure_ascii=False,
            )

        return json.dumps(
            {
                "success": False,
                "provider": "serpapi",
                "tool": "serpapi_read_only_probe",
                "status_code": response.status_code,
                "error": _serpapi_error_message(
                    response,
                    payload,
                    default=f"SerpApi account probe failed with HTTP {response.status_code}.",
                ),
            },
            ensure_ascii=False,
        )
    except requests.Timeout as exc:
        logger.warning("SerpApi read-only probe timed out: %s", exc)
        return json.dumps(
            {
                "success": False,
                "provider": "serpapi",
                "tool": "serpapi_read_only_probe",
                "error": f"SerpApi account probe timed out after {SERPAPI_TIMEOUT_SECONDS} seconds.",
                "error_type": type(exc).__name__,
            }
        )
    except requests.RequestException as exc:
        logger.warning("SerpApi read-only probe failed: %s", exc)
        return json.dumps(
            {
                "success": False,
                "provider": "serpapi",
                "tool": "serpapi_read_only_probe",
                "error": "SerpApi account probe could not reach SerpApi.",
                "error_type": type(exc).__name__,
            }
        )


def serpapi_google_flights_search(
    origin: str,
    destination: str,
    departure_date: str,
    *,
    return_date: Optional[str] = None,
    trip_type: Optional[str] = None,
    adults: int = 1,
    children: int = 0,
    infants_in_seat: int = 0,
    infants_on_lap: int = 0,
    cabin_class: str = "economy",
    currency: str = "USD",
    gl: Optional[str] = None,
    hl: str = "en",
    filters: Optional[Dict[str, Any]] = None,
    limit: int = 10,
) -> str:
    """Search Google Flights through SerpApi and return normalized JSON.

    This is search-only: it never books, reserves, purchases, or mutates vendor state.
    """
    api_key = os.getenv(SERPAPI_KEY_ENV, "").strip()
    if not api_key:
        return json.dumps(_missing_secret_payload("serpapi_google_flights_search"))

    try:
        normalized_origin = _normalize_iata(origin, "origin")
        normalized_destination = _normalize_iata(destination, "destination")
        normalized_departure = _normalize_iso_date(departure_date, "departure_date")
        normalized_return = _normalize_iso_date(return_date, "return_date") if return_date else None
        normalized_trip_type, serpapi_trip_type = _normalize_trip_type(trip_type or "", bool(normalized_return))
        normalized_cabin, serpapi_cabin = _normalize_cabin_class(cabin_class)
        normalized_adults = _normalize_positive_int(adults, "adults")
        normalized_children = _normalize_positive_int(children, "children", allow_zero=True)
        normalized_infants_seat = _normalize_positive_int(infants_in_seat, "infants_in_seat", allow_zero=True)
        normalized_infants_lap = _normalize_positive_int(infants_on_lap, "infants_on_lap", allow_zero=True)
        capped_limit = max(1, min(int(limit or 10), 50))
    except ValueError as exc:
        return json.dumps(
            {
                "success": False,
                "provider": "serpapi",
                "tool": "serpapi_google_flights_search",
                "error": str(exc),
                "error_type": "validation_error",
                "caveat": TRAVEL_PRICE_VOLATILITY_CAVEAT,
            }
        )

    params: Dict[str, Any] = {
        "engine": "google_flights",
        "api_key": api_key,
        "departure_id": normalized_origin,
        "arrival_id": normalized_destination,
        "outbound_date": normalized_departure,
        "type": serpapi_trip_type,
        "adults": normalized_adults,
        "children": normalized_children,
        "infants_in_seat": normalized_infants_seat,
        "infants_on_lap": normalized_infants_lap,
        "travel_class": serpapi_cabin,
        "currency": str(currency or "USD").strip().upper(),
        "hl": hl or "en",
    }
    if normalized_return and serpapi_trip_type == "1":
        params["return_date"] = normalized_return
    if gl:
        params["gl"] = gl
    for key, value in (filters or {}).items():
        if value is not None and key != "api_key":
            params[key] = value

    safe_params = _safe_search_parameters(params)
    try:
        response = requests.get(SERPAPI_SEARCH_URL, params=params, timeout=SERPAPI_SEARCH_TIMEOUT_SECONDS)
        try:
            payload = response.json()
        except ValueError:
            payload = {}

        if response.status_code != 200 or payload.get("error"):
            return json.dumps(
                {
                    "success": False,
                    "provider": "serpapi",
                    "tool": "serpapi_google_flights_search",
                    "status_code": response.status_code,
                    "error": _serpapi_error_message(
                        response,
                        payload,
                        default=f"SerpApi Google Flights search failed with HTTP {response.status_code}.",
                    ),
                    "query": safe_params,
                    "caveat": TRAVEL_PRICE_VOLATILITY_CAVEAT,
                },
                ensure_ascii=False,
            )

        results: List[Dict[str, Any]] = []
        for bucket in ("best_flights", "other_flights"):
            options = payload.get(bucket) if isinstance(payload.get(bucket), list) else []
            for option in options:
                if isinstance(option, dict):
                    results.append(
                        _normalize_flight_option(
                            option,
                            currency=params["currency"],
                            bucket=bucket,
                            index=len(results) + 1,
                        )
                    )
                if len(results) >= capped_limit:
                    break
            if len(results) >= capped_limit:
                break

        search_metadata = payload.get("search_metadata") or {}
        return json.dumps(
            {
                "success": True,
                "provider": "serpapi",
                "tool": "serpapi_google_flights_search",
                "engine": "google_flights",
                "query": {
                    "origin": normalized_origin,
                    "destination": normalized_destination,
                    "departure_date": normalized_departure,
                    "return_date": normalized_return,
                    "trip_type": normalized_trip_type,
                    "adults": normalized_adults,
                    "children": normalized_children,
                    "infants_in_seat": normalized_infants_seat,
                    "infants_on_lap": normalized_infants_lap,
                    "cabin_class": normalized_cabin,
                    "currency": params["currency"],
                    "request_parameters": safe_params,
                },
                "results_count": len(results),
                "results": results,
                "source_metadata": {
                    "search_id": search_metadata.get("id"),
                    "status": search_metadata.get("status"),
                    "json_endpoint": search_metadata.get("json_endpoint"),
                    "search_parameters": _safe_search_parameters(payload.get("search_parameters", {})),
                },
                "caveat": TRAVEL_PRICE_VOLATILITY_CAVEAT,
            },
            ensure_ascii=False,
        )
    except requests.Timeout as exc:
        logger.warning("SerpApi Google Flights search timed out: %s", exc)
        return json.dumps(
            {
                "success": False,
                "provider": "serpapi",
                "tool": "serpapi_google_flights_search",
                "error": f"SerpApi Google Flights search timed out after {SERPAPI_SEARCH_TIMEOUT_SECONDS} seconds.",
                "error_type": type(exc).__name__,
                "query": safe_params,
                "caveat": TRAVEL_PRICE_VOLATILITY_CAVEAT,
            }
        )
    except requests.RequestException as exc:
        logger.warning("SerpApi Google Flights search failed: %s", exc)
        return json.dumps(
            {
                "success": False,
                "provider": "serpapi",
                "tool": "serpapi_google_flights_search",
                "error": "SerpApi Google Flights search could not reach SerpApi.",
                "error_type": type(exc).__name__,
                "query": safe_params,
                "caveat": TRAVEL_PRICE_VOLATILITY_CAVEAT,
            }
        )
def serpapi_google_hotels_search(
    destination: str,
    check_in_date: str,
    check_out_date: str,
    *,
    adults: int = 2,
    children: Optional[int] = None,
    rooms: int = 1,
    currency: str = "USD",
    gl: Optional[str] = None,
    hl: Optional[str] = None,
    filters: Optional[Dict[str, Any]] = None,
    limit: int = 10,
) -> str:
    """Search Google Hotels through SerpApi and return normalized JSON.

    This is search-only: it never books, reserves, purchases, or mutates vendor state.
    """
    api_key = os.getenv(SERPAPI_KEY_ENV, "").strip()
    if not api_key:
        return json.dumps(_missing_secret_payload("serpapi_google_hotels_search"))

    if not destination or not check_in_date or not check_out_date:
        return json.dumps(
            {
                "success": False,
                "provider": "serpapi",
                "tool": "serpapi_google_hotels_search",
                "error": "destination, check_in_date, and check_out_date are required.",
                "caveat": HOTELS_VOLATILITY_CAVEAT,
            }
        )

    params: Dict[str, Any] = {
        "engine": "google_hotels",
        "q": destination,
        "check_in_date": check_in_date,
        "check_out_date": check_out_date,
        "adults": max(1, int(adults or 1)),
        "rooms": max(1, int(rooms or 1)),
        "currency": currency,
        "api_key": api_key,
    }
    if children is not None:
        params["children"] = max(0, int(children))
    if gl:
        params["gl"] = gl
    if hl:
        params["hl"] = hl
    for key, value in (filters or {}).items():
        if value is not None and key != "api_key":
            params[key] = value

    safe_params = _safe_search_parameters(params)
    try:
        response = requests.get(SERPAPI_SEARCH_URL, params=params, timeout=SERPAPI_TIMEOUT_SECONDS)
        try:
            payload = response.json()
        except ValueError:
            payload = {}

        if response.status_code != 200 or payload.get("error"):
            return json.dumps(
                {
                    "success": False,
                    "provider": "serpapi",
                    "tool": "serpapi_google_hotels_search",
                    "status_code": response.status_code,
                    "error": _serpapi_error_message(
                        response,
                        payload,
                        default=f"SerpApi Google Hotels search failed with HTTP {response.status_code}.",
                    ),
                    "query": safe_params,
                    "caveat": HOTELS_VOLATILITY_CAVEAT,
                },
                ensure_ascii=False,
            )

        properties: List[Dict[str, Any]] = payload.get("properties") or []
        capped_limit = max(1, min(int(limit or 10), 50))
        hotels = [
            _normalize_hotel(item, check_in_date=check_in_date, check_out_date=check_out_date)
            for item in properties[:capped_limit]
        ]
        return json.dumps(
            {
                "success": True,
                "provider": "serpapi",
                "tool": "serpapi_google_hotels_search",
                "engine": "google_hotels",
                "query": safe_params,
                "results_count": len(hotels),
                "hotels": hotels,
                "search_metadata": payload.get("search_metadata", {}),
                "search_parameters": _safe_search_parameters(payload.get("search_parameters", {})),
                "caveat": HOTELS_VOLATILITY_CAVEAT,
            },
            ensure_ascii=False,
        )
    except requests.Timeout as exc:
        logger.warning("SerpApi Google Hotels search timed out: %s", exc)
        return json.dumps(
            {
                "success": False,
                "provider": "serpapi",
                "tool": "serpapi_google_hotels_search",
                "error": f"SerpApi Google Hotels search timed out after {SERPAPI_TIMEOUT_SECONDS} seconds.",
                "error_type": type(exc).__name__,
                "query": safe_params,
                "caveat": HOTELS_VOLATILITY_CAVEAT,
            }
        )
    except requests.RequestException as exc:
        logger.warning("SerpApi Google Hotels search failed: %s", exc)
        return json.dumps(
            {
                "success": False,
                "provider": "serpapi",
                "tool": "serpapi_google_hotels_search",
                "error": "SerpApi Google Hotels search could not reach SerpApi.",
                "error_type": type(exc).__name__,
                "query": safe_params,
                "caveat": HOTELS_VOLATILITY_CAVEAT,
            }
        )


SERPAPI_READ_ONLY_PROBE_SCHEMA = {
    "name": "serpapi_read_only_probe",
    "description": (
        "Verify that SERPAPI_API_KEY is present and accepted by SerpApi using "
        "SerpApi's read-only Account API. The probe is safe: it never books, "
        "purchases, mutates external state, or returns the API key."
    ),
    "parameters": {"type": "object", "properties": {}},
}


SERPAPI_GOOGLE_FLIGHTS_SEARCH_SCHEMA = {
    "name": "serpapi_google_flights_search",
    "description": (
        "Search Google Flights via SerpApi. Search-only: never books, reserves, "
        "purchases, mutates external state, or returns the API key. Returns "
        "normalized flight options plus source metadata for debugging."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "origin": {"type": "string", "description": "Origin airport/city code, e.g. AMS."},
            "destination": {"type": "string", "description": "Destination airport/city code, e.g. JFK."},
            "departure_date": {"type": "string", "description": "Departure date in YYYY-MM-DD format."},
            "return_date": {"type": "string", "description": "Optional return date in YYYY-MM-DD format."},
            "trip_type": {"type": "string", "description": "round_trip or one_way. Defaults from return_date."},
            "adults": {"type": "integer", "description": "Adult passengers.", "default": 1},
            "children": {"type": "integer", "description": "Child passengers.", "default": 0},
            "infants_in_seat": {"type": "integer", "description": "Infants in their own seat.", "default": 0},
            "infants_on_lap": {"type": "integer", "description": "Infants on lap.", "default": 0},
            "cabin_class": {"type": "string", "description": "economy, premium_economy, business, or first.", "default": "economy"},
            "currency": {"type": "string", "description": "Currency code such as USD or EUR.", "default": "USD"},
            "gl": {"type": "string", "description": "Optional Google country code, e.g. us or nl."},
            "hl": {"type": "string", "description": "Optional Google UI language, e.g. en or nl.", "default": "en"},
            "filters": {"type": "object", "description": "Optional SerpApi google_flights-compatible parameters."},
            "limit": {"type": "integer", "description": "Maximum normalized flight options to return (1-50).", "default": 10},
        },
        "required": ["origin", "destination", "departure_date"],
    },
}


SERPAPI_GOOGLE_HOTELS_SEARCH_SCHEMA = {
    "name": "serpapi_google_hotels_search",
    "description": (
        "Search Google Hotels via SerpApi. Search-only: never books, reserves, "
        "purchases, mutates external state, or returns the API key. Returns "
        "normalized hotel results plus raw metadata for debugging."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "destination": {
                "type": "string",
                "description": "Destination/city/query, e.g. Amsterdam or hotels near Schiphol.",
            },
            "check_in_date": {"type": "string", "description": "Check-in date in YYYY-MM-DD format."},
            "check_out_date": {"type": "string", "description": "Check-out date in YYYY-MM-DD format."},
            "adults": {"type": "integer", "description": "Number of adult guests.", "default": 2},
            "children": {"type": "integer", "description": "Number of children, if any."},
            "rooms": {"type": "integer", "description": "Number of rooms.", "default": 1},
            "currency": {"type": "string", "description": "Currency code such as USD or EUR.", "default": "USD"},
            "gl": {"type": "string", "description": "Optional Google country code, e.g. us or nl."},
            "hl": {"type": "string", "description": "Optional Google UI language, e.g. en or nl."},
            "filters": {"type": "object", "description": "Optional SerpApi google_hotels-compatible filters/parameters."},
            "limit": {"type": "integer", "description": "Maximum normalized hotels to return (1-50).", "default": 10},
        },
        "required": ["destination", "check_in_date", "check_out_date"],
    },
}


def _handle_serpapi_read_only_probe(args, **kwargs):
    return serpapi_read_only_probe()


def _handle_serpapi_google_flights_search(args, **kwargs):
    return serpapi_google_flights_search(
        origin=args.get("origin", ""),
        destination=args.get("destination", ""),
        departure_date=args.get("departure_date", ""),
        return_date=args.get("return_date"),
        trip_type=args.get("trip_type"),
        adults=args.get("adults", 1),
        children=args.get("children", 0),
        infants_in_seat=args.get("infants_in_seat", 0),
        infants_on_lap=args.get("infants_on_lap", 0),
        cabin_class=args.get("cabin_class", "economy"),
        currency=args.get("currency", "USD"),
        gl=args.get("gl"),
        hl=args.get("hl", "en"),
        filters=args.get("filters"),
        limit=args.get("limit", 10),
    )


def _handle_serpapi_google_hotels_search(args, **kwargs):
    return serpapi_google_hotels_search(
        destination=args.get("destination", ""),
        check_in_date=args.get("check_in_date", ""),
        check_out_date=args.get("check_out_date", ""),
        adults=args.get("adults", 2),
        children=args.get("children"),
        rooms=args.get("rooms", 1),
        currency=args.get("currency", "USD"),
        gl=args.get("gl"),
        hl=args.get("hl"),
        filters=args.get("filters"),
        limit=args.get("limit", 10),
    )


registry.register(
    name="serpapi_read_only_probe",
    toolset="serpapi",
    schema=SERPAPI_READ_ONLY_PROBE_SCHEMA,
    handler=_handle_serpapi_read_only_probe,
    requires_env=[SERPAPI_KEY_ENV],
    emoji="🔎",
    max_result_size_chars=20_000,
)


registry.register(
    name="serpapi_google_flights_search",
    toolset="serpapi",
    schema=SERPAPI_GOOGLE_FLIGHTS_SEARCH_SCHEMA,
    handler=_handle_serpapi_google_flights_search,
    requires_env=[SERPAPI_KEY_ENV],
    emoji="✈️",
    max_result_size_chars=40_000,
)


registry.register(
    name="serpapi_google_hotels_search",
    toolset="serpapi",
    schema=SERPAPI_GOOGLE_HOTELS_SEARCH_SCHEMA,
    handler=_handle_serpapi_google_hotels_search,
    requires_env=[SERPAPI_KEY_ENV],
    emoji="🏨",
    max_result_size_chars=40_000,
)
