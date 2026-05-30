"""Marketplace listing extraction helpers.

This module provides a narrow tool for purchase-sensitive marketplace pages
where generic ``web_search`` snippets and ``web_extract`` markdown are often
insufficient. Amazon/eBay use SerpApi as the primary marketplace provider when
configured, eBay preserves the official Browse API as a secondary path, and
Firecrawl structured extraction remains the generic provider fallback.
"""

from __future__ import annotations

import asyncio
import base64
import os
import re
import time
from typing import Any, Dict, Iterable, List, Optional
from urllib.parse import urlparse

import requests

from plugins.web.firecrawl.provider import FirecrawlWebSearchProvider, check_firecrawl_api_key
from tools.registry import registry
from tools.url_safety import is_safe_url
from tools.website_policy import check_website_access

_MARKETPLACE_HOST_SUFFIXES = {
    "amazon": ("amazon.com",),
    "ebay": ("ebay.com",),
}

_AMAZON_ASIN_RE = re.compile(r"/(?:dp|gp/product|exec/obidos/ASIN)/([A-Z0-9]{10})(?:[/?#]|$)", re.I)
_EBAY_ITEM_RE = re.compile(r"/(?:itm)(?:/[^/?#]+)?/(\d{9,14})(?:[/?#]|$)|/(?:itm)/(\d{9,14})(?:[/?#]|$)", re.I)
_EBAY_SCOPE = "https://api.ebay.com/oauth/api_scope"
_EBAY_TOKEN_CACHE: Dict[str, Any] = {"access_token": None, "expires_at": 0.0, "cache_key": None}
_SERPAPI_BASE_URL = "https://serpapi.com/search.json"

MARKETPLACE_LISTING_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "price": {"type": "string"},
        "currency": {"type": "string"},
        "availability": {"type": "string"},
        "condition": {"type": "string"},
        "seller": {"type": "string"},
        "seller_rating": {"type": "string"},
        "shipping": {"type": "string"},
        "returns": {"type": "string"},
        "item_location": {"type": "string"},
        "product_specs": {
            "type": "object",
            "additionalProperties": {"type": "string"},
        },
        "warnings": {
            "type": "array",
            "items": {"type": "string"},
        },
    },
}

MARKETPLACE_LISTING_EXTRACT_SCHEMA = {
    "name": "marketplace_listing_extract",
    "description": (
        "Extract structured product/listing details from Amazon or eBay listing URLs. "
        "Use for shopping/product comparisons when web_search snippets or generic "
        "web_extract are insufficient. Amazon/eBay URLs use SerpApi product extraction "
        "when SERPAPI_API_KEY is configured, then the official eBay Browse API for "
        "eBay when EBAY_CLIENT_ID and EBAY_CLIENT_SECRET are configured, then a "
        "configured commerce-capable extractor such as Firecrawl."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "urls": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Amazon/eBay listing URLs to extract (max 5).",
                "maxItems": 5,
            },
            "marketplace": {
                "type": "string",
                "enum": ["auto", "amazon", "ebay"],
                "default": "auto",
                "description": "Expected marketplace. Use auto unless the URL is ambiguous.",
            },
        },
        "required": ["urls"],
    },
}

MARKETPLACE_LISTING_SEARCH_SCHEMA = {
    "name": "marketplace_listing_search",
    "description": (
        "Search Amazon or eBay marketplace listings by keyword using SerpApi. Use this for "
        "shopping/product comparisons when you need current marketplace candidate listings "
        "before extracting a specific listing URL. Requires SERPAPI_API_KEY."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Marketplace keyword search query, e.g. 'LG Gram 17'.",
            },
            "marketplace": {
                "type": "string",
                "enum": ["amazon", "ebay"],
                "default": "ebay",
                "description": "Marketplace to search.",
            },
            "max_results": {
                "type": "integer",
                "minimum": 1,
                "maximum": 20,
                "default": 10,
                "description": "Maximum number of normalized listings to return.",
            },
        },
        "required": ["query"],
    },
}


def _host_matches(host: str, suffixes: Iterable[str]) -> bool:
    host = host.lower().strip(".")
    return any(host == suffix or host.endswith(f".{suffix}") for suffix in suffixes)


def detect_marketplace(url: str) -> Optional[str]:
    """Return ``amazon``/``ebay`` for supported listing hosts, else None."""
    host = urlparse(url).hostname or ""
    for marketplace, suffixes in _MARKETPLACE_HOST_SUFFIXES.items():
        if _host_matches(host, suffixes):
            return marketplace
    return None


def parse_listing_id(url: str, marketplace: str) -> Optional[str]:
    """Extract the stable listing/product id when it is visible in the URL."""
    parsed = urlparse(url)
    path = parsed.path or ""
    if marketplace == "amazon":
        match = _AMAZON_ASIN_RE.search(path)
        return match.group(1).upper() if match else None
    if marketplace == "ebay":
        match = _EBAY_ITEM_RE.search(path)
        if not match:
            return None
        return next(group for group in match.groups() if group)
    return None


def _configuration_error() -> Dict[str, Any]:
    return {
        "success": False,
        "error": (
            "Marketplace listing extraction requires a commerce-capable backend. "
            "For Amazon/eBay, configure SERPAPI_API_KEY for SerpApi search/product "
            "extraction, or EBAY_CLIENT_ID and EBAY_CLIENT_SECRET for the official "
            "Browse API. For Amazon/eBay provider fallback, configure Firecrawl via "
            "FIRECRAWL_API_KEY/FIRECRAWL_API_URL or the Nous Tool Gateway."
        ),
        "configuration_required": {
            "supported_now": ["serpapi_amazon", "serpapi_ebay", "ebay_browse_api", "firecrawl_structured_extract"],
            "future_official_routes": ["amazon_creators_or_pa_api"],
        },
    }


def _url_error(url: str, error: str) -> Dict[str, Any]:
    return {"url": url, "title": "", "content": "", "error": error}


def _has_ebay_browse_config() -> bool:
    return bool(os.getenv("EBAY_CLIENT_ID", "").strip() and os.getenv("EBAY_CLIENT_SECRET", "").strip())


def _serpapi_api_key() -> str:
    """Return the configured SerpApi key without logging or exposing it."""
    return os.getenv("SERPAPI_API_KEY", "").strip() or os.getenv("SERPAPI_KEY", "").strip()


def _has_serpapi_config() -> bool:
    return bool(_serpapi_api_key())


def _serpapi_base_url() -> str:
    return os.getenv("SERPAPI_BASE_URL", _SERPAPI_BASE_URL).strip() or _SERPAPI_BASE_URL


def _ebay_api_base_url() -> str:
    return os.getenv("EBAY_API_BASE_URL", "https://api.ebay.com").strip().rstrip("/")


def _ebay_token_url() -> str:
    configured = os.getenv("EBAY_OAUTH_TOKEN_URL", "").strip()
    if configured:
        return configured
    return f"{_ebay_api_base_url()}/identity/v1/oauth2/token"


def _get_ebay_access_token() -> str:
    client_id = os.getenv("EBAY_CLIENT_ID", "").strip()
    client_secret = os.getenv("EBAY_CLIENT_SECRET", "").strip()
    if not client_id or not client_secret:
        raise RuntimeError("EBAY_CLIENT_ID and EBAY_CLIENT_SECRET are required for eBay Browse API")

    cache_key = (_ebay_token_url(), client_id, client_secret)
    now = time.time()
    cached = _EBAY_TOKEN_CACHE.get("access_token")
    if cached and _EBAY_TOKEN_CACHE.get("cache_key") == cache_key and now < float(_EBAY_TOKEN_CACHE.get("expires_at", 0)):
        return str(cached)

    basic = base64.b64encode(f"{client_id}:{client_secret}".encode("utf-8")).decode("ascii")
    response = requests.post(
        _ebay_token_url(),
        headers={
            "Authorization": f"Basic {basic}",
            "Content-Type": "application/x-www-form-urlencoded",
            "Accept": "application/json",
        },
        data={"grant_type": "client_credentials", "scope": _EBAY_SCOPE},
        timeout=20,
    )
    if not response.ok:
        raise RuntimeError(f"eBay OAuth token request failed: HTTP {response.status_code} {response.text[:300]}")
    payload = response.json()
    token = payload.get("access_token")
    if not token:
        raise RuntimeError("eBay OAuth token response did not include access_token")
    expires_in = int(payload.get("expires_in") or 7200)
    _EBAY_TOKEN_CACHE.update(
        {
            "access_token": token,
            "expires_at": now + max(60, expires_in - 120),
            "cache_key": cache_key,
        }
    )
    return str(token)


def _first_present(*values: Any) -> str:
    for value in values:
        if value is None:
            continue
        if isinstance(value, str) and value.strip():
            return value.strip()
        if not isinstance(value, str):
            return str(value)
    return ""


def _currency_from_price(price: str) -> str:
    if price.startswith("$") or " US $" in price or price.endswith(" USD"):
        return "USD"
    if price.startswith("£") or price.endswith(" GBP"):
        return "GBP"
    if price.startswith("€") or price.endswith(" EUR"):
        return "EUR"
    return ""


def _first_dict(*values: Any) -> Dict[str, Any]:
    for value in values:
        if isinstance(value, dict):
            return value
    return {}


def _serpapi_money(value: Any) -> tuple[str, str]:
    """Normalize SerpApi money shapes from search/product responses."""
    if isinstance(value, dict):
        amount = _first_present(value.get("amount"), value.get("extracted"), value.get("value"), value.get("raw"))
        currency = _first_present(value.get("currency"), _currency_from_price(amount))
        if amount and value.get("raw") and not str(amount).startswith(("$", "£", "€")):
            amount = _first_present(value.get("raw"), amount)
        return amount, currency
    price = _first_present(value)
    return price, _currency_from_price(price)


def _money(value: Any) -> tuple[str, str]:
    if not isinstance(value, dict):
        return "", ""
    amount = _first_present(value.get("value"), value.get("convertedFromValue"))
    currency = _first_present(value.get("currency"), value.get("convertedFromCurrency"))
    return amount, currency


def _format_shipping(options: Any) -> str:
    if not isinstance(options, list) or not options:
        return ""
    option = options[0] if isinstance(options[0], dict) else {}
    cost, currency = _money(option.get("shippingCost"))
    parts = []
    if option.get("type"):
        parts.append(str(option["type"]))
    if cost:
        parts.append(f"{cost} {currency}".strip())
    return "; ".join(parts)


def _format_returns(return_terms: Any) -> str:
    if not isinstance(return_terms, dict):
        return ""
    accepted = return_terms.get("returnsAccepted")
    period = return_terms.get("refundPeriod") or {}
    parts = []
    if accepted is not None:
        parts.append("returns accepted" if accepted else "returns not accepted")
    if isinstance(period, dict) and period.get("value") and period.get("unit"):
        parts.append(f"{period['value']} {period['unit']}")
    return "; ".join(parts)


def _format_location(location: Any) -> str:
    if not isinstance(location, dict):
        return ""
    return ", ".join(
        part
        for part in [
            _first_present(location.get("city")),
            _first_present(location.get("stateOrProvince")),
            _first_present(location.get("country")),
        ]
        if part
    )


def _aspects_to_specs(aspects: Any) -> Dict[str, str]:
    specs: Dict[str, str] = {}
    if not isinstance(aspects, list):
        return specs
    for aspect in aspects:
        if not isinstance(aspect, dict):
            continue
        name = _first_present(aspect.get("name"))
        values = aspect.get("value") or aspect.get("values")
        if not name:
            continue
        if isinstance(values, list):
            specs[name] = ", ".join(str(v) for v in values if v is not None)
        else:
            specs[name] = _first_present(values)
    return specs


def _serpapi_specs(specifications: Any) -> Dict[str, str]:
    if isinstance(specifications, dict):
        grouped: Dict[str, str] = {}
        groups = specifications.get("groups")
        if isinstance(groups, list):
            for group in groups:
                if not isinstance(group, dict):
                    continue
                sections = group.get("sections")
                if not isinstance(sections, list):
                    continue
                for section in sections:
                    if not isinstance(section, dict):
                        continue
                    fields = section.get("fields")
                    if not isinstance(fields, list):
                        continue
                    for field in fields:
                        if not isinstance(field, dict):
                            continue
                        title = _first_present(field.get("title"), field.get("name"), field.get("type"))
                        value = _first_present(field.get("value"), field.get("text"))
                        if title and value:
                            grouped[title] = value
        if grouped:
            return grouped
        return {str(k): _first_present(v) for k, v in specifications.items() if _first_present(v)}
    specs: Dict[str, str] = {}
    if isinstance(specifications, list):
        for item in specifications:
            if not isinstance(item, dict):
                continue
            name = _first_present(item.get("name"), item.get("label"), item.get("key"))
            value = _first_present(item.get("value"), item.get("text"))
            if name and value:
                specs[name] = value
    return specs


def _serpapi_snippets_text(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, dict):
        snippets = value.get("snippets")
        if isinstance(snippets, list):
            return " ".join(
                _first_present(snippet.get("text") if isinstance(snippet, dict) else snippet)
                for snippet in snippets
            ).strip()
        return _first_present(value.get("status"))
    if isinstance(value, list):
        parts: List[str] = []
        for item in value:
            text = _serpapi_snippets_text(item)
            if text:
                parts.append(text)
        return "; ".join(parts)
    return _first_present(value)


def _serpapi_shipping_text(value: Any) -> str:
    if isinstance(value, dict):
        options = value.get("options")
        if isinstance(options, list) and options:
            option = options[0] if isinstance(options[0], dict) else {}
            if option.get("free") is True:
                return _first_present(option.get("via"), "Free shipping")
            price, currency = _serpapi_money(option.get("price") or option.get("cost"))
            return "; ".join(part for part in [_first_present(option.get("via")), f"{price} {currency}".strip()] if part)
        return _serpapi_snippets_text(value)
    return _serpapi_snippets_text(value)


def _serpapi_media_links(media: Any) -> List[str]:
    links: List[str] = []
    if not isinstance(media, list):
        return links
    for item in media:
        if not isinstance(item, dict):
            continue
        images = item.get("image")
        if not isinstance(images, list):
            continue
        candidates = [image for image in images if isinstance(image, dict) and image.get("link")]
        if candidates:
            candidates.sort(key=lambda image: ((image.get("size") or {}).get("width") or 0), reverse=True)
            links.append(str(candidates[0]["link"]))
    return links


def _normalize_serpapi_search_result(item: Dict[str, Any], marketplace: str) -> Dict[str, Any]:
    price, currency = _serpapi_money(item.get("price"))
    if not price:
        price, currency = _serpapi_money(item.get("extracted_price"))
    seller = _first_dict(item.get("seller"), item.get("seller_results"))
    seller_name = _first_present(seller.get("name"), seller.get("username"))
    if not seller_name and isinstance(item.get("seller"), str):
        seller_name = str(item["seller"]).strip()
    return {
        "title": _first_present(item.get("title"), item.get("name")),
        "price": price,
        "currency": _first_present(item.get("currency"), currency),
        "condition": _first_present(item.get("condition")),
        "shipping": _first_present(item.get("shipping"), item.get("delivery")),
        "seller": seller_name,
        "seller_rating": _first_present(seller.get("rating"), seller.get("feedback"), seller.get("feedback_percentage")),
        "url": _first_present(item.get("link"), item.get("url")),
        "link": _first_present(item.get("link"), item.get("url")),
        "product_id": _first_present(item.get("product_id"), item.get("asin"), item.get("item_id"), item.get("id")),
        "source": f"serpapi_{marketplace}",
        "metadata": {
            "thumbnail": item.get("thumbnail"),
            "extensions": item.get("extensions"),
            "rating": item.get("rating"),
            "reviews": item.get("reviews"),
        },
    }


def _serpapi_result_items(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    for key in ("organic_results", "ebay_results", "shopping_results", "product_results", "results"):
        value = payload.get(key)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
    return []


def _search_ebay_with_serpapi(query: str, max_results: int) -> Dict[str, Any]:
    api_key = _serpapi_api_key()
    if not api_key:
        return {
            "success": False,
            "error": "marketplace_listing_search requires SERPAPI_API_KEY for SerpApi eBay search.",
            "configuration_required": {"supported_now": ["serpapi_ebay"]},
        }

    response = requests.get(
        _serpapi_base_url(),
        params={"engine": "ebay", "_nkw": query, "api_key": api_key},
        timeout=30,
    )
    if not response.ok:
        raise RuntimeError(f"SerpApi eBay search failed: HTTP {response.status_code} {response.text[:300]}")
    payload = response.json()
    listings = [_normalize_serpapi_search_result(item, "ebay") for item in _serpapi_result_items(payload)[:max_results]]
    return {
        "success": True,
        "marketplace": "ebay",
        "query": query,
        "source": "serpapi_ebay",
        "results": listings,
        "metadata": {"search_metadata": payload.get("search_metadata"), "search_parameters": payload.get("search_parameters")},
    }


def _search_amazon_with_serpapi(query: str, max_results: int) -> Dict[str, Any]:
    api_key = _serpapi_api_key()
    if not api_key:
        return {
            "success": False,
            "error": "marketplace_listing_search requires SERPAPI_API_KEY for SerpApi Amazon search.",
            "configuration_required": {"supported_now": ["serpapi_amazon"]},
        }
    response = requests.get(
        _serpapi_base_url(),
        params={"engine": "amazon", "k": query, "amazon_domain": "amazon.com", "api_key": api_key},
        timeout=30,
    )
    if not response.ok:
        raise RuntimeError(f"SerpApi Amazon search failed: HTTP {response.status_code} {response.text[:300]}")
    payload = response.json()
    listings = [_normalize_serpapi_search_result(item, "amazon") for item in _serpapi_result_items(payload)[:max_results]]
    return {
        "success": True,
        "marketplace": "amazon",
        "query": query,
        "source": "serpapi_amazon",
        "results": listings,
        "metadata": {"search_metadata": payload.get("search_metadata"), "search_parameters": payload.get("search_parameters")},
    }


def _normalize_serpapi_ebay_product(payload: Dict[str, Any], url: str, legacy_id: str) -> Dict[str, Any]:
    product = _first_dict(payload.get("product_results"), payload)
    buy = _first_dict(product.get("buy"), payload.get("buy"))
    buy_it_now = _first_dict(buy.get("buy_it_now"))
    seller = _first_dict(product.get("seller_results"), payload.get("seller_results"), product.get("seller"))
    price, currency = _serpapi_money(buy_it_now.get("price"))
    if not price:
        price, currency = _serpapi_money(buy.get("price"))
    if not price:
        price, currency = _serpapi_money(product.get("price"))
    structured = {
        "title": _first_present(product.get("title"), payload.get("title")),
        "price": price,
        "currency": _first_present(buy_it_now.get("currency"), buy.get("currency"), product.get("currency"), currency),
        "availability": _first_present(buy.get("availability"), product.get("availability"), product.get("stock")),
        "condition": _first_present(product.get("condition"), buy.get("condition")),
        "seller": _first_present(seller.get("username"), seller.get("name"), seller.get("seller_name")),
        "seller_rating": _first_present(seller.get("rating"), seller.get("feedback_percentage"), seller.get("feedback_score")),
        "shipping": _serpapi_shipping_text(buy.get("shipping") or product.get("shipping")),
        "returns": _serpapi_snippets_text(buy.get("returns") or product.get("returns")),
        "item_location": _first_present(product.get("item_location"), seller.get("location")),
        "product_specs": _serpapi_specs(product.get("specifications")),
        "warnings": [],
    }
    return {
        "url": _first_present(product.get("link"), product.get("url"), url),
        "title": structured["title"],
        "content": _first_present(product.get("short_description"), product.get("description")),
        "raw_content": "",
        "marketplace": "ebay",
        "listing_id": legacy_id,
        "source": "serpapi_ebay",
        "structured_data": structured,
        "metadata": {
            "product_id": _first_present(product.get("product_id"), payload.get("product_id"), legacy_id),
            "watch_count": product.get("watch_count"),
            "media": product.get("media"),
            "image_links": _serpapi_media_links(product.get("media")),
            "full_description_link": product.get("full_description_link"),
        },
    }


def _fetch_ebay_product_with_serpapi(url: str, legacy_id: str) -> Dict[str, Any]:
    api_key = _serpapi_api_key()
    if not api_key:
        raise RuntimeError("SERPAPI_API_KEY is required for SerpApi eBay product extraction")
    response = requests.get(
        _serpapi_base_url(),
        params={"engine": "ebay_product", "product_id": legacy_id, "api_key": api_key},
        timeout=30,
    )
    if not response.ok:
        raise RuntimeError(f"SerpApi eBay product extraction failed: HTTP {response.status_code} {response.text[:300]}")
    payload = response.json()
    return _normalize_serpapi_ebay_product(payload, url, legacy_id)


def _normalize_serpapi_amazon_product(payload: Dict[str, Any], url: str, asin: str) -> Dict[str, Any]:
    product = _first_dict(payload.get("product_results"), payload)
    buybox = _first_dict(product.get("buybox_winner"))
    price, currency = _serpapi_money(product.get("price") or buybox.get("price"))
    if not price:
        price, currency = _serpapi_money(product.get("extracted_price"))
    seller = _first_dict(product.get("seller"), buybox.get("seller"))
    specs = _serpapi_specs(product.get("specifications"))
    feature_bullets = product.get("feature_bullets")
    if not specs and isinstance(feature_bullets, list):
        specs = {"feature_bullets": "; ".join(_first_present(item) for item in feature_bullets if _first_present(item))}
    images = product.get("images") or []
    image_links = [str(image) for image in images if isinstance(image, str)]
    structured = {
        "title": _first_present(product.get("title")),
        "price": price,
        "currency": _first_present(product.get("currency"), currency),
        "availability": _first_present(product.get("availability"), product.get("stock")),
        "condition": _first_present(product.get("condition")),
        "seller": _first_present(seller.get("name"), seller.get("seller_name"), product.get("brand")),
        "seller_rating": _first_present(product.get("rating"), product.get("reviews")),
        "shipping": _serpapi_shipping_text(product.get("shipping") or product.get("delivery")),
        "returns": _serpapi_snippets_text(product.get("returns")),
        "item_location": "",
        "product_specs": specs,
        "warnings": [],
    }
    return {
        "url": _first_present(product.get("link"), product.get("url"), url),
        "title": structured["title"],
        "content": _first_present(product.get("description"), feature_bullets),
        "raw_content": "",
        "marketplace": "amazon",
        "listing_id": asin,
        "source": "serpapi_amazon",
        "structured_data": structured,
        "metadata": {
            "asin": _first_present(product.get("asin"), asin),
            "brand": product.get("brand"),
            "rating": product.get("rating"),
            "reviews": product.get("reviews"),
            "image_links": image_links,
        },
    }


def _fetch_amazon_product_with_serpapi(url: str, asin: str) -> Dict[str, Any]:
    api_key = _serpapi_api_key()
    if not api_key:
        raise RuntimeError("SERPAPI_API_KEY is required for SerpApi Amazon product extraction")
    response = requests.get(
        _serpapi_base_url(),
        params={"engine": "amazon_product", "asin": asin, "amazon_domain": "amazon.com", "api_key": api_key},
        timeout=30,
    )
    if not response.ok:
        raise RuntimeError(f"SerpApi Amazon product extraction failed: HTTP {response.status_code} {response.text[:300]}")
    payload = response.json()
    return _normalize_serpapi_amazon_product(payload, url, asin)


def _normalize_ebay_item(payload: Dict[str, Any], url: str, legacy_id: str) -> Dict[str, Any]:
    price, currency = _money(payload.get("price"))
    seller_raw = payload.get("seller")
    seller: Dict[str, Any] = seller_raw if isinstance(seller_raw, dict) else {}
    structured = {
        "title": _first_present(payload.get("title")),
        "price": price,
        "currency": currency,
        "availability": _first_present(payload.get("estimatedAvailabilities")),
        "condition": _first_present(payload.get("condition"), payload.get("conditionId")),
        "seller": _first_present(seller.get("username"), payload.get("sellerUsername")),
        "seller_rating": _first_present(seller.get("feedbackPercentage"), seller.get("feedbackScore")),
        "shipping": _format_shipping(payload.get("shippingOptions")),
        "returns": _format_returns(payload.get("returnTerms")),
        "item_location": _format_location(payload.get("itemLocation")),
        "product_specs": _aspects_to_specs(payload.get("localizedAspects")),
        "warnings": [],
    }
    return {
        "url": _first_present(payload.get("itemWebUrl"), url),
        "title": structured["title"],
        "content": "",
        "raw_content": "",
        "marketplace": "ebay",
        "listing_id": legacy_id,
        "source": "ebay_browse_api",
        "structured_data": structured,
        "metadata": {
            "itemId": payload.get("itemId"),
            "legacyItemId": legacy_id,
            "categoryPath": payload.get("categoryPath"),
            "image": payload.get("image"),
        },
    }


def _fetch_ebay_item_by_legacy_id(url: str, legacy_id: str) -> Dict[str, Any]:
    token = _get_ebay_access_token()
    marketplace_id = os.getenv("EBAY_MARKETPLACE_ID", "EBAY_US").strip() or "EBAY_US"
    response = requests.get(
        f"{_ebay_api_base_url()}/buy/browse/v1/item/get_item_by_legacy_id",
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/json",
            "X-EBAY-C-MARKETPLACE-ID": marketplace_id,
        },
        params={"legacy_item_id": legacy_id},
        timeout=30,
    )
    if not response.ok:
        raise RuntimeError(f"eBay Browse API getItemByLegacyId failed: HTTP {response.status_code} {response.text[:300]}")
    payload = response.json()
    return _normalize_ebay_item(payload, url, legacy_id)


async def _extract_with_firecrawl(prepared: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    provider = FirecrawlWebSearchProvider()
    extracted = await provider.extract(
        [item["url"] for item in prepared],
        json_schema=MARKETPLACE_LISTING_SCHEMA,
        include_markdown=True,
    )

    by_url = {item["url"]: item for item in prepared}
    results: List[Dict[str, Any]] = []
    for item in extracted:
        source_url = item.get("url", "")
        original = by_url.get(source_url) or next(
            (entry for entry in prepared if entry["url"] == source_url), None
        )
        enriched = dict(item)
        enriched.setdefault("source", "firecrawl_structured_extract")
        if original:
            enriched.setdefault("marketplace", original["marketplace"])
            enriched.setdefault("listing_id", original["listing_id"])
        results.append(enriched)
    return results


async def marketplace_listing_extract_tool(
    urls: List[str], marketplace: str = "auto"
) -> Dict[str, Any]:
    """Extract structured details from supported marketplace listing URLs."""
    if not isinstance(urls, list) or not urls:
        return {"success": False, "error": "Provide at least one Amazon or eBay listing URL."}

    requested_marketplace = (marketplace or "auto").lower().strip()
    if requested_marketplace not in {"auto", "amazon", "ebay"}:
        return {"success": False, "error": "marketplace must be one of: auto, amazon, ebay"}

    prepared: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []
    for url in urls[:5]:
        if not isinstance(url, str) or not url.strip():
            errors.append(_url_error(str(url), "URL must be a non-empty string"))
            continue
        url = url.strip()
        if not is_safe_url(url):
            errors.append(_url_error(url, "URL failed safety checks"))
            continue
        blocked = check_website_access(url)
        if blocked:
            errors.append(_url_error(url, blocked["message"]))
            continue
        detected = detect_marketplace(url)
        if detected is None:
            errors.append(_url_error(url, "Only Amazon and eBay listing URLs are supported"))
            continue
        if requested_marketplace != "auto" and detected != requested_marketplace:
            errors.append(_url_error(url, f"URL appears to be {detected}, not {requested_marketplace}"))
            continue
        prepared.append(
            {
                "url": url,
                "marketplace": detected,
                "listing_id": parse_listing_id(url, detected),
            }
        )

    if not prepared:
        return {"success": False, "results": errors, "error": "No supported marketplace URLs to extract."}

    results: List[Dict[str, Any]] = errors[:]
    remaining_for_browse: List[Dict[str, Any]] = []
    remaining_for_firecrawl: List[Dict[str, Any]] = []

    if _has_serpapi_config():
        for item in prepared:
            if item["marketplace"] == "amazon" and item.get("listing_id"):
                try:
                    results.append(
                        await asyncio.to_thread(
                            _fetch_amazon_product_with_serpapi,
                            item["url"],
                            str(item["listing_id"]),
                        )
                    )
                except Exception:  # noqa: BLE001 — fall through to Firecrawl per URL
                    remaining_for_firecrawl.append(item)
            elif item["marketplace"] == "ebay" and item.get("listing_id"):
                try:
                    results.append(
                        await asyncio.to_thread(
                            _fetch_ebay_product_with_serpapi,
                            item["url"],
                            str(item["listing_id"]),
                        )
                    )
                except Exception:  # noqa: BLE001 — fall through to secondary providers per URL
                    remaining_for_browse.append(item)
            else:
                remaining_for_firecrawl.append(item)
    else:
        remaining_for_browse = [item for item in prepared if item["marketplace"] == "ebay" and item.get("listing_id")]
        remaining_for_firecrawl = [item for item in prepared if item not in remaining_for_browse]

    if _has_ebay_browse_config():
        for item in remaining_for_browse:
            try:
                results.append(
                    await asyncio.to_thread(
                        _fetch_ebay_item_by_legacy_id,
                        item["url"],
                        str(item["listing_id"]),
                    )
                )
            except Exception:  # noqa: BLE001 — fall through to Firecrawl per URL
                remaining_for_firecrawl.append(item)
    else:
        remaining_for_firecrawl.extend(remaining_for_browse)

    if remaining_for_firecrawl and check_firecrawl_api_key():
        results.extend(await _extract_with_firecrawl(remaining_for_firecrawl))
    elif remaining_for_firecrawl:
        config = _configuration_error()
        results.extend(
            {
                "url": item["url"],
                "marketplace": item["marketplace"],
                "listing_id": item["listing_id"],
                "error": config["error"],
            }
            for item in remaining_for_firecrawl
        )
        if len(results) == len(remaining_for_firecrawl) + len(errors):
            config["results"] = results
            return config

    success = any(not item.get("error") for item in results)
    return {"success": success, "results": results}


async def marketplace_listing_search_tool(
    query: str, marketplace: str = "ebay", max_results: int = 10
) -> Dict[str, Any]:
    """Search supported marketplace listings."""
    if not isinstance(query, str) or not query.strip():
        return {"success": False, "error": "Provide a non-empty marketplace search query."}
    requested_marketplace = (marketplace or "ebay").lower().strip()
    if requested_marketplace not in {"amazon", "ebay"}:
        return {"success": False, "error": "marketplace_listing_search currently supports amazon and ebay"}
    try:
        limit = max(1, min(int(max_results), 20))
    except (TypeError, ValueError):
        limit = 10
    try:
        search_fn = _search_amazon_with_serpapi if requested_marketplace == "amazon" else _search_ebay_with_serpapi
        return await asyncio.to_thread(search_fn, query.strip(), limit)
    except Exception as exc:  # noqa: BLE001 — normalize provider failure for tool callers
        return {"success": False, "error": str(exc), "source": f"serpapi_{requested_marketplace}"}


registry.register(
    name="marketplace_listing_search",
    toolset="web",
    schema=MARKETPLACE_LISTING_SEARCH_SCHEMA,
    handler=lambda args, **kw: marketplace_listing_search_tool(
        args.get("query", ""),
        marketplace=args.get("marketplace", "ebay"),
        max_results=args.get("max_results", 10),
    ),
    check_fn=lambda: True,
    requires_env=[],
    is_async=True,
    emoji="🛒",
    max_result_size_chars=100_000,
)


registry.register(
    name="marketplace_listing_extract",
    toolset="web",
    schema=MARKETPLACE_LISTING_EXTRACT_SCHEMA,
    handler=lambda args, **kw: marketplace_listing_extract_tool(
        args.get("urls", [])[:5] if isinstance(args.get("urls"), list) else [],
        marketplace=args.get("marketplace", "auto"),
    ),
    check_fn=lambda: True,
    requires_env=[],
    is_async=True,
    emoji="🛒",
    max_result_size_chars=100_000,
)
