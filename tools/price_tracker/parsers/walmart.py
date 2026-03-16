"""Walmart parser — handles walmart.com product pages.

Extracts price from JSON-LD Product schema and HTML fallbacks.
"""

import logging
import re
from typing import List

from tools.price_tracker.parsers.base import BaseSiteParser, ProductData
from tools.price_tracker.parsers.price_utils import (
    parse_price, extract_text, extract_json_ld, extract_meta,
)

logger = logging.getLogger(__name__)


class WalmartParser(BaseSiteParser):
    """Parser for walmart.com product pages."""

    def get_site_name(self) -> str:
        return "Walmart"

    def get_domains(self) -> List[str]:
        return ["walmart.com", "www.walmart.com"]

    def can_handle(self, url: str) -> bool:
        return "walmart.com" in url.lower()

    def parse(self, html: str, url: str = "") -> ProductData:
        data = ProductData()
        data.currency = "USD"

        # --- JSON-LD extraction ---
        for block in extract_json_ld(html):
            if isinstance(block, dict) and "Product" in str(block.get("@type", "")):
                data.name = block.get("name", "")
                offers = block.get("offers", {})
                if isinstance(offers, list):
                    offers = offers[0] if offers else {}
                if isinstance(offers, dict):
                    price_val = offers.get("price", offers.get("lowPrice", ""))
                    if price_val:
                        try:
                            data.price = float(price_val)
                        except (ValueError, TypeError):
                            data.price = parse_price(str(price_val), "USD")
                    avail = offers.get("availability", "").lower()
                    if "instock" in avail:
                        data.stock_status = "in_stock"
                    elif "outofstock" in avail:
                        data.stock_status = "out_of_stock"
                if data.name:
                    break

        # --- HTML fallbacks ---
        if not data.name:
            data.name = extract_text(html, [
                r'<h1[^>]*itemprop="name"[^>]*>(.*?)</h1>',
                r'<h1[^>]*class="[^"]*prod-ProductTitle[^"]*"[^>]*>(.*?)</h1>',
                r'<title>(.*?)(?:\s*[-|])',
            ])

        if not data.price:
            price_text = extract_text(html, [
                r'itemprop="price"[^>]*content="([^"]+)"',
                r'class="[^"]*price-group[^"]*"[^>]*>\s*\$([\d,.]+)',
                r'"currentPrice":\s*([\d.]+)',
                r'"price":\s*([\d.]+)',
            ])
            if price_text:
                data.price = parse_price(price_text, "USD")

        # Original price
        if not data.original_price:
            orig_text = extract_text(html, [
                r'"wasPrice":\s*([\d.]+)',
                r'class="[^"]*price-old[^"]*"[^>]*>\s*\$([\d,.]+)',
            ])
            if orig_text:
                data.original_price = parse_price(orig_text, "USD")

        # Stock
        if data.stock_status == "unknown":
            html_lower = html.lower()
            if "out of stock" in html_lower or "not available" in html_lower:
                data.stock_status = "out_of_stock"
            elif "add to cart" in html_lower:
                data.stock_status = "in_stock"
            elif data.price:
                data.stock_status = "in_stock"

        data.seller = "Walmart"
        data.image_url = extract_meta(html, "image") or ""
        return data
