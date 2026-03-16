"""Newegg parser — handles newegg.com product pages.

Extracts price from JSON-LD Product schema, price-current class, and script data.
"""

import logging
import re
from typing import List

from tools.price_tracker.parsers.base import BaseSiteParser, ProductData
from tools.price_tracker.parsers.price_utils import (
    parse_price, extract_text, extract_json_ld, extract_meta,
)

logger = logging.getLogger(__name__)


class NeweggParser(BaseSiteParser):
    """Parser for newegg.com product pages."""

    def get_site_name(self) -> str:
        return "Newegg"

    def get_domains(self) -> List[str]:
        return ["newegg.com", "www.newegg.com"]

    def can_handle(self, url: str) -> bool:
        return "newegg.com" in url.lower()

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
                    price_val = offers.get("price", "")
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
                r'class="product-title"[^>]*>(.*?)</h1>',
                r'<h1[^>]*>\s*(.*?)\s*</h1>',
                r'<title>(.*?)(?:\s*[-|])',
            ])

        if not data.price:
            price_text = extract_text(html, [
                r'class="price-current"[^>]*>\s*<strong>([\d,]+)</strong>\s*<sup>\.(\d+)</sup>',
                r'class="price-current"[^>]*>(.*?)</li>',
                r'"price":\s*"?\$([\d,.]+)"?',
            ])
            if price_text:
                # Handle the strong + sup pattern: "1,299" + "99"
                match = re.search(r'([\d,]+)\.(\d+)', price_text.replace(' ', ''))
                if match:
                    data.price = parse_price(match.group(0), "USD")
                else:
                    data.price = parse_price(price_text, "USD")

        # Original / list price
        if not data.original_price:
            orig_text = extract_text(html, [
                r'class="price-was[^"]*"[^>]*>\s*\$([\d,.]+)',
            ])
            if orig_text:
                data.original_price = parse_price(orig_text, "USD")

        # Stock
        if data.stock_status == "unknown":
            html_lower = html.lower()
            if "out of stock" in html_lower or "currently unavailable" in html_lower:
                data.stock_status = "out_of_stock"
            elif "add to cart" in html_lower:
                data.stock_status = "in_stock"
            elif data.price:
                data.stock_status = "in_stock"

        data.seller = "Newegg"
        data.image_url = extract_meta(html, "image") or ""
        return data
