"""Google Places API (New) tool for Hermes Agent.

Provides text search, nearby search, and place details via the
Google Places API (New). Requires GOOGLE_PLACES_API_KEY env var.

Endpoints:
- Text Search: POST /v1/places:searchText
- Nearby Search: POST /v1/places:searchNearby
- Place Details: GET /v1/places/{placeId}

Docs: https://developers.google.com/maps/documentation/places/web-service/op-overview
"""

import json
import logging
import os
from typing import Any, Dict, List, Optional

import requests

from tools.registry import registry, tool_error

logger = logging.getLogger(__name__)

BASE_URL = "https://places.googleapis.com/v1"
DEFAULT_FIELD_MASK = (
    "places.id,places.displayName,places.formattedAddress,places.location,"
    "places.types,places.primaryType,places.rating,places.userRatingCount,"
    "places.priceLevel,places.websiteUri,places.nationalPhoneNumber,"
    "places.regularOpeningHours,places.googleMapsUri"
)
DETAILS_FIELD_MASK = (
    "id,displayName,formattedAddress,location,types,primaryType,"
    "rating,userRatingCount,priceLevel,websiteUri,nationalPhoneNumber,"
    "regularOpeningHours,googleMapsUri"
)


def _get_api_key() -> str:
    return os.getenv("GOOGLE_PLACES_API_KEY", "")


def _check_places_available() -> bool:
    return bool(_get_api_key())


def _headers(field_mask: str) -> Dict[str, str]:
    return {
        "Content-Type": "application/json",
        "X-Goog-Api-Key": _get_api_key(),
        "X-Goog-FieldMask": field_mask,
    }


def _normalize_place(place: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten displayName and location for easier LLM consumption."""
    out = dict(place)
    # displayName comes as {"text": "...", "languageCode": "..."}
    if isinstance(out.get("displayName"), dict):
        out["displayName"] = out["displayName"].get("text", "")
    # location comes as {"latitude": ..., "longitude": ...}
    loc = out.get("location", {})
    if isinstance(loc, dict):
        out["latitude"] = loc.get("latitude")
        out["longitude"] = loc.get("longitude")
    return out


def _compact_results(places: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Return a compact list of normalized places."""
    return [_normalize_place(p) for p in places]


# ---------------------------------------------------------------------------
# Text Search
# ---------------------------------------------------------------------------

def places_search_tool(query: str, limit: int = 5) -> str:
    """Search for places using a text query."""
    if not _check_places_available():
        return tool_error("GOOGLE_PLACES_API_KEY not set")

    payload = {"textQuery": query, "pageSize": max(1, min(limit, 20))}
    try:
        resp = requests.post(
            f"{BASE_URL}/places:searchText",
            headers=_headers(DEFAULT_FIELD_MASK),
            json=payload,
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
        places = data.get("places", [])
        return json.dumps(
            {"count": len(places), "places": _compact_results(places)}, indent=2
        )
    except requests.HTTPError as e:
        logger.error("Google Places text search HTTP error: %s", e)
        return tool_error(f"Google Places text search failed: {e.response.text if e.response else e}")
    except Exception as e:
        logger.error("Google Places text search error: %s", e)
        return tool_error(f"Google Places text search failed: {e}")


PLACES_SEARCH_SCHEMA = {
    "name": "places_search",
    "description": (
        "Search for places using Google Places API (New). "
        "Returns places matching a text query (e.g. 'pizza in New York', 'gas station near me'). "
        "Each result includes name, address, coordinates, rating, price level, phone, hours, and Google Maps link."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The text search query (e.g. 'coffee shops in Seattle').",
            },
            "limit": {
                "type": "integer",
                "description": "Maximum number of results to return (1-20). Default: 5.",
                "minimum": 1,
                "maximum": 20,
                "default": 5,
            },
        },
        "required": ["query"],
    },
}


# ---------------------------------------------------------------------------
# Nearby Search
# ---------------------------------------------------------------------------

def places_nearby_tool(
    latitude: float,
    longitude: float,
    radius: float = 500.0,
    included_types: Optional[List[str]] = None,
    limit: int = 10,
) -> str:
    """Find places near a geographic coordinate."""
    if not _check_places_available():
        return tool_error("GOOGLE_PLACES_API_KEY not set")

    payload: Dict[str, Any] = {
        "locationRestriction": {
            "circle": {
                "center": {"latitude": latitude, "longitude": longitude},
                "radius": max(0.0, min(radius, 50000.0)),
            }
        },
        "maxResultCount": max(1, min(limit, 20)),
    }
    if included_types:
        payload["includedTypes"] = included_types

    try:
        resp = requests.post(
            f"{BASE_URL}/places:searchNearby",
            headers=_headers(DEFAULT_FIELD_MASK),
            json=payload,
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
        places = data.get("places", [])
        return json.dumps(
            {"count": len(places), "places": _compact_results(places)}, indent=2
        )
    except requests.HTTPError as e:
        logger.error("Google Places nearby search HTTP error: %s", e)
        return tool_error(f"Google Places nearby search failed: {e.response.text if e.response else e}")
    except Exception as e:
        logger.error("Google Places nearby search error: %s", e)
        return tool_error(f"Google Places nearby search failed: {e}")


PLACES_NEARBY_SCHEMA = {
    "name": "places_nearby",
    "description": (
        "Find places near a specific latitude/longitude using Google Places API (New). "
        "Each result includes name, address, coordinates, rating, price level, phone, hours, and Google Maps link."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "latitude": {
                "type": "number",
                "description": "Latitude of the search center.",
            },
            "longitude": {
                "type": "number",
                "description": "Longitude of the search center.",
            },
            "radius": {
                "type": "number",
                "description": "Search radius in meters (0-50000). Default: 500.",
                "minimum": 0,
                "maximum": 50000,
                "default": 500,
            },
            "included_types": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Optional list of place types to filter by (e.g. ['restaurant', 'cafe']). "
                    "See https://developers.google.com/maps/documentation/places/web-service/place-types"
                ),
            },
            "limit": {
                "type": "integer",
                "description": "Maximum number of results (1-20). Default: 10.",
                "minimum": 1,
                "maximum": 20,
                "default": 10,
            },
        },
        "required": ["latitude", "longitude"],
    },
}


# ---------------------------------------------------------------------------
# Place Details
# ---------------------------------------------------------------------------

def place_details_tool(place_id: str) -> str:
    """Get detailed information about a place by its Google Place ID."""
    if not _check_places_available():
        return tool_error("GOOGLE_PLACES_API_KEY not set")

    try:
        resp = requests.get(
            f"{BASE_URL}/places/{place_id}",
            headers=_headers(DETAILS_FIELD_MASK),
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
        return json.dumps(_normalize_place(data), indent=2)
    except requests.HTTPError as e:
        logger.error("Google Places details HTTP error: %s", e)
        return tool_error(f"Google Places details failed: {e.response.text if e.response else e}")
    except Exception as e:
        logger.error("Google Places details error: %s", e)
        return tool_error(f"Google Places details failed: {e}")


PLACE_DETAILS_SCHEMA = {
    "name": "place_details",
    "description": (
        "Get detailed information about a place using its Google Place ID. "
        "Returns name, address, coordinates, rating, price level, phone, hours, and Google Maps link."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "place_id": {
                "type": "string",
                "description": "The Google Place ID (e.g. 'ChIJj61dQgK6j4AR4GeTYWZsKWw').",
            },
        },
        "required": ["place_id"],
    },
}


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

registry.register(
    name="places_search",
    toolset="google_places",
    schema=PLACES_SEARCH_SCHEMA,
    handler=lambda args, **kw: places_search_tool(
        args.get("query", ""), limit=args.get("limit", 5)
    ),
    check_fn=_check_places_available,
    requires_env=["GOOGLE_PLACES_API_KEY"],
    emoji="",
)

registry.register(
    name="places_nearby",
    toolset="google_places",
    schema=PLACES_NEARBY_SCHEMA,
    handler=lambda args, **kw: places_nearby_tool(
        latitude=args.get("latitude", 0.0),
        longitude=args.get("longitude", 0.0),
        radius=args.get("radius", 500.0),
        included_types=args.get("included_types") if isinstance(args.get("included_types"), list) else None,
        limit=args.get("limit", 10),
    ),
    check_fn=_check_places_available,
    requires_env=["GOOGLE_PLACES_API_KEY"],
    emoji="",
)

registry.register(
    name="place_details",
    toolset="google_places",
    schema=PLACE_DETAILS_SCHEMA,
    handler=lambda args, **kw: place_details_tool(args.get("place_id", "")),
    check_fn=_check_places_available,
    requires_env=["GOOGLE_PLACES_API_KEY"],
    emoji="",
)
