"""Shared price parsing utilities for multi-currency support.

Handles US/UK (1,299.00) and European (1 299,00 or 1.299,00) formats.
Detects currency from URL domain, HTML content, or explicit currency symbols.
"""

import re
from typing import Optional, Tuple


# ---------------------------------------------------------------------------
# Currency detection
# ---------------------------------------------------------------------------

_DOMAIN_CURRENCY_MAP = {
    "amazon.com": "USD", "walmart.com": "USD", "target.com": "USD",
    "bestbuy.com": "USD", "newegg.com": "USD", "ebay.com": "USD",
    "amazon.co.uk": "GBP", "ebay.co.uk": "GBP", "argos.co.uk": "GBP",
    "currys.co.uk": "GBP", "johnlewis.com": "GBP",
    "amazon.de": "EUR", "amazon.fr": "EUR", "amazon.it": "EUR",
    "amazon.es": "EUR", "ebay.de": "EUR", "ebay.fr": "EUR",
    "idealo.de": "EUR", "otto.de": "EUR", "mediamarkt.de": "EUR",
    "zalando.com": "EUR", "cdiscount.com": "EUR", "fnac.com": "EUR",
    "bol.com": "EUR",
    "elgiganten.dk": "DKK", "elgiganten.se": "SEK",
    "aliexpress.com": "USD", "rakuten.com": "USD",
}

_SYMBOL_CURRENCY_MAP = {
    "$": "USD", "€": "EUR", "£": "GBP",
    "kr": "SEK",  # also DKK/NOK — ambiguous
}


def detect_currency(url: str = "", html: str = "", default: str = "USD") -> str:
    """Detect currency from URL domain or HTML content."""
    url_lower = url.lower()
    for domain, currency in _DOMAIN_CURRENCY_MAP.items():
        if domain in url_lower:
            return currency

    # Check HTML for currency symbols
    for symbol, currency in _SYMBOL_CURRENCY_MAP.items():
        if symbol in html[:5000]:  # Only check early HTML
            return currency

    return default


# ---------------------------------------------------------------------------
# Multi-format price parsing
# ---------------------------------------------------------------------------

def parse_price(text: str, currency: str = "auto") -> Optional[float]:
    """Parse a price string to float, auto-detecting format.

    Supports:
        European:  1.299,00 €   → 1299.0
        US/UK:     $1,299.00    → 1299.0
        Space:     1 299,00 €   → 1299.0
        Plain:     1299         → 1299.0
    """
    if not text:
        return None

    # Strip currency symbols, whitespace, and common prefixes
    cleaned = re.sub(r'[^\d.,\s]', '', text.strip())
    # Remove leading/trailing spaces
    cleaned = cleaned.strip()
    if not cleaned:
        return None

    # Determine the locale format
    if currency in ("EUR", "SEK", "DKK"):
        # European: dot=thousands, comma=decimal
        return _parse_european_price(cleaned)
    elif currency in ("USD", "GBP"):
        # US/UK: comma=thousands, dot=decimal
        return _parse_english_price(cleaned)
    else:
        # Auto-detect: heuristic based on the text
        return _parse_auto_price(cleaned)


def _parse_european_price(text: str) -> Optional[float]:
    """Parse European price: 1.299,00 → 1299.0"""
    text = text.replace(" ", "")  # Remove spaces (e.g., 1 299,00)
    text = text.replace(".", "").replace(",", ".")
    try:
        return float(text)
    except ValueError:
        return None


def _parse_english_price(text: str) -> Optional[float]:
    """Parse US/UK price: 1,299.00 → 1299.0"""
    text = text.replace(" ", "").replace(",", "")
    try:
        return float(text)
    except ValueError:
        return None


def _parse_auto_price(text: str) -> Optional[float]:
    """Auto-detect price format from the number itself."""
    text = text.replace(" ", "")

    has_dot = "." in text
    has_comma = "," in text

    if has_dot and has_comma:
        # Both present: check order
        if text.rfind(".") < text.rfind(","):
            # 1.299,00 → European
            return _parse_european_price(text)
        else:
            # 1,299.00 → English
            return _parse_english_price(text)

    if has_comma:
        # Single comma: check position from end
        after_comma = len(text) - text.rfind(",") - 1
        if after_comma <= 2:
            # 499,90 → European decimal
            return _parse_european_price(text)
        else:
            # 1,299 → English thousands
            return _parse_english_price(text)

    if has_dot:
        # Single dot: check position from end
        after_dot = len(text) - text.rfind(".") - 1
        if after_dot <= 2:
            # 499.90 → English decimal
            return _parse_english_price(text)
        else:
            # 1.299 → European thousands
            return _parse_european_price(text)

    # No separators at all
    try:
        return float(text)
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# HTML helpers
# ---------------------------------------------------------------------------

def extract_text(html: str, patterns: list) -> str:
    """Try multiple regex patterns and return the first match, stripped of tags."""
    for pattern in patterns:
        match = re.search(pattern, html, re.DOTALL | re.IGNORECASE)
        if match:
            text = re.sub(r'<[^>]+>', '', match.group(1)).strip()
            if text:
                return text
    return ""


def extract_json_ld(html: str) -> list:
    """Extract all JSON-LD blocks from HTML."""
    import json
    blocks = []
    for match in re.finditer(
        r'<script[^>]*type=["\']application/ld\+json["\'][^>]*>(.*?)</script>',
        html, re.DOTALL | re.IGNORECASE
    ):
        try:
            data = json.loads(match.group(1).strip())
            if isinstance(data, list):
                blocks.extend(data)
            else:
                blocks.append(data)
        except (json.JSONDecodeError, ValueError):
            continue
    return blocks


def extract_meta(html: str, property_name: str) -> str:
    """Extract content from a <meta> tag by property or name."""
    match = re.search(
        rf'<meta[^>]*(?:property|name)=["\'](?:og:)?{re.escape(property_name)}["\'][^>]*content=["\']([^"\']*)["\']',
        html, re.IGNORECASE
    )
    if not match:
        # Try reversed attribute order
        match = re.search(
            rf'<meta[^>]*content=["\']([^"\']*)["\'][^>]*(?:property|name)=["\'](?:og:)?{re.escape(property_name)}["\']',
            html, re.IGNORECASE
        )
    return match.group(1).strip() if match else ""
