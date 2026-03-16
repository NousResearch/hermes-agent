"""GenericParser — universal product data extraction via structured data.

Uses JSON-LD (Schema.org Product), Open Graph meta tags, and microdata
to extract product information from *any* modern e-commerce site.

This is the fallback parser tried when no site-specific parser matches.
Supports ~70% of modern e-commerce sites out of the box.
"""

import json
import logging
import re
from typing import List, Optional

from tools.price_tracker.parsers.base import BaseSiteParser, ProductData
from tools.price_tracker.parsers.price_utils import (
    detect_currency, parse_price, extract_json_ld, extract_meta, extract_text,
)

logger = logging.getLogger(__name__)


class GenericParser(BaseSiteParser):
    """Extracts product data using JSON-LD, OpenGraph, and common HTML patterns."""

    def get_site_name(self) -> str:
        return "Generic (Structured Data)"

    def get_domains(self) -> List[str]:
        return []  # Matches everything as fallback

    def can_handle(self, url: str) -> bool:
        return True  # Always true — used as last resort

    def parse(self, html: str, url: str = "") -> ProductData:
        data = ProductData()
        currency = detect_currency(url, html)
        data.currency = currency

        # --- Try JSON-LD first (most reliable) ---
        product_ld = self._parse_json_ld(html, currency)
        if product_ld:
            return product_ld

        # --- Try Open Graph meta tags ---
        data.name = extract_meta(html, "title") or extract_meta(html, "og:title")
        data.image_url = extract_meta(html, "image") or extract_meta(html, "og:image")

        # OG product price
        price_str = extract_meta(html, "product:price:amount") or extract_meta(html, "og:price:amount")
        if price_str:
            data.price = parse_price(price_str, currency)

        og_currency = extract_meta(html, "product:price:currency") or extract_meta(html, "og:price:currency")
        if og_currency:
            data.currency = og_currency.upper()

        availability = extract_meta(html, "product:availability") or extract_meta(html, "og:availability")
        if availability:
            avail_lower = availability.lower()
            if "instock" in avail_lower or "in_stock" in avail_lower:
                data.stock_status = "in_stock"
            elif "outofstock" in avail_lower or "out_of_stock" in avail_lower:
                data.stock_status = "out_of_stock"
            elif "limited" in avail_lower or "preorder" in avail_lower:
                data.stock_status = "limited"

        # --- Try Schema.org microdata ---
        if not data.price:
            price_text = extract_text(html, [
                r'itemprop="price"[^>]*content="([^"]+)"',
                r'itemprop="price"[^>]*>\s*([^<]+)',
                r'itemprop="lowPrice"[^>]*content="([^"]+)"',
            ])
            if price_text:
                data.price = parse_price(price_text, currency)

        if not data.name:
            data.name = extract_text(html, [
                r'itemprop="name"[^>]*>\s*(.*?)\s*</(?:h1|span|div)',
            ])

        # --- Fallback: <title> tag ---
        if not data.name:
            title_match = re.search(r'<title[^>]*>(.*?)</title>', html, re.DOTALL | re.IGNORECASE)
            if title_match:
                title = re.sub(r'<[^>]+>', '', title_match.group(1)).strip()
                # Remove common suffixes
                for sep in [' | ', ' - ', ' – ', ' — ', ' :: ']:
                    if sep in title:
                        title = title.split(sep)[0].strip()
                data.name = title

        # --- Fallback: price from common patterns ---
        if not data.price:
            price_text = extract_text(html, [
                r'class="[^"]*price[^"]*"[^>]*>\s*([^<]+)',
                r'class="[^"]*Price[^"]*"[^>]*>\s*([^<]+)',
                r'data-price="([^"]+)"',
            ])
            if price_text:
                data.price = parse_price(price_text, currency)

        # Stock status fallback
        if data.stock_status == "unknown" and data.price:
            data.stock_status = "in_stock"

        return data

    def _parse_json_ld(self, html: str, currency: str) -> Optional[ProductData]:
        """Extract product data from JSON-LD blocks."""
        blocks = extract_json_ld(html)

        for block in blocks:
            if isinstance(block, dict):
                # Handle @graph arrays
                if "@graph" in block:
                    for item in block["@graph"]:
                        result = self._extract_from_ld_product(item, currency)
                        if result:
                            return result
                else:
                    result = self._extract_from_ld_product(block, currency)
                    if result:
                        return result
        return None

    def _extract_from_ld_product(self, item: dict, currency: str) -> Optional[ProductData]:
        """Extract ProductData from a single JSON-LD product object."""
        item_type = item.get("@type", "")
        if isinstance(item_type, list):
            item_type = " ".join(item_type)

        if "Product" not in item_type:
            return None

        data = ProductData()
        data.currency = currency
        data.name = item.get("name", "")
        data.image_url = item.get("image", "")
        if isinstance(data.image_url, list):
            data.image_url = data.image_url[0] if data.image_url else ""

        # Rating
        agg_rating = item.get("aggregateRating", {})
        if agg_rating:
            try:
                data.rating = float(agg_rating.get("ratingValue", 0))
                data.review_count = int(agg_rating.get("reviewCount", 0))
            except (ValueError, TypeError):
                pass

        # Category
        cat = item.get("category", "")
        if isinstance(cat, list):
            data.category = " > ".join(cat)
        elif isinstance(cat, str):
            data.category = cat

        # Brand as seller fallback
        brand = item.get("brand", {})
        if isinstance(brand, dict):
            data.seller = brand.get("name", "")
        elif isinstance(brand, str):
            data.seller = brand

        # Offers
        offers = item.get("offers", item.get("offer", {}))
        if isinstance(offers, list):
            offers = offers[0] if offers else {}

        if isinstance(offers, dict):
            # Price
            price_val = offers.get("price", offers.get("lowPrice", ""))
            if price_val:
                try:
                    data.price = float(price_val)
                except (ValueError, TypeError):
                    data.price = parse_price(str(price_val), currency)

            # Currency from offers
            offer_currency = offers.get("priceCurrency", "")
            if offer_currency:
                data.currency = offer_currency.upper()

            # Availability
            availability = offers.get("availability", "")
            if isinstance(availability, str):
                avail_lower = availability.lower()
                if "instock" in avail_lower:
                    data.stock_status = "in_stock"
                elif "outofstock" in avail_lower:
                    data.stock_status = "out_of_stock"
                elif "limited" in avail_lower or "preorder" in avail_lower:
                    data.stock_status = "limited"

            # Seller from offers
            seller_obj = offers.get("seller", {})
            if isinstance(seller_obj, dict) and seller_obj.get("name"):
                data.seller = seller_obj["name"]

        if data.name:
            return data
        return None
