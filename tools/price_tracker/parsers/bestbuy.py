"""Best Buy parser — handles bestbuy.com product pages.

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


class BestBuyParser(BaseSiteParser):
    """Parser for bestbuy.com product pages."""

    def get_site_name(self) -> str:
        return "Best Buy"

    def get_domains(self) -> List[str]:
        return ["bestbuy.com", "www.bestbuy.com"]

    def can_handle(self, url: str) -> bool:
        return "bestbuy.com" in url.lower()

    def parse(self, html: str, url: str = "") -> ProductData:
        data = ProductData()
        data.currency = "USD"

        # --- JSON-LD extraction (Best Buy uses structured data) ---
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
                r'class="sku-title"[^>]*>\s*<h1[^>]*>(.*?)</h1>',
                r'<h1[^>]*class="[^"]*heading[^"]*"[^>]*>(.*?)</h1>',
                r'<title>(.*?)(?:\s*[-|])',
            ])

        if not data.price:
            price_text = extract_text(html, [
                r'class="priceView-hero-price[^"]*"[^>]*>\s*<span[^>]*>(.*?)</span>',
                r'class="priceView-customer-price[^"]*"[^>]*>\s*<span[^>]*>(.*?)</span>',
                r'data-testid="customer-price"[^>]*>\s*<span[^>]*>(.*?)</span>',
                r'"currentPrice":\s*([\d.]+)',
            ])
            if price_text:
                data.price = parse_price(price_text, "USD")

        # Original price
        if not data.original_price:
            orig_text = extract_text(html, [
                r'class="pricing-price__regular-price[^"]*"[^>]*>.*?<span[^>]*>(.*?)</span>',
                r'"regularPrice":\s*([\d.]+)',
            ])
            if orig_text:
                data.original_price = parse_price(orig_text, "USD")

        # Stock status from HTML
        if data.stock_status == "unknown":
            html_lower = html.lower()
            if "sold out" in html_lower or "coming soon" in html_lower:
                data.stock_status = "out_of_stock"
            elif "add to cart" in html_lower:
                data.stock_status = "in_stock"
            elif data.price:
                data.stock_status = "in_stock"

        data.seller = "Best Buy"
        data.image_url = extract_meta(html, "image") or ""
        return data
