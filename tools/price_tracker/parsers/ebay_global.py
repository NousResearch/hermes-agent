"""eBay Global parser — handles all eBay domains.

Supports: ebay.com, ebay.co.uk, ebay.de, ebay.fr
Extracts current price, Buy It Now, bid info, and seller ratings.
"""

import logging
import re
from typing import List

from tools.price_tracker.parsers.base import BaseSiteParser, ProductData
from tools.price_tracker.parsers.price_utils import (
    detect_currency, parse_price, extract_text, extract_json_ld,
)

logger = logging.getLogger(__name__)

_EBAY_DOMAINS = ["ebay.com", "ebay.co.uk", "ebay.de", "ebay.fr"]


class EbayGlobalParser(BaseSiteParser):
    """Parser for all eBay product/listing pages worldwide."""

    def get_site_name(self) -> str:
        return "eBay (Global)"

    def get_domains(self) -> List[str]:
        domains = []
        for d in _EBAY_DOMAINS:
            domains.extend([d, f"www.{d}"])
        return domains

    def can_handle(self, url: str) -> bool:
        return any(d in url.lower() for d in _EBAY_DOMAINS)

    def parse(self, html: str, url: str = "") -> ProductData:
        data = ProductData()
        currency = detect_currency(url, html)
        data.currency = currency

        # --- Try JSON-LD first ---
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
                            data.price = parse_price(str(price_val), currency)
                    pc = offers.get("priceCurrency", "")
                    if pc:
                        data.currency = pc.upper()
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
                r'class="x-item-title__mainTitle"[^>]*>\s*<span[^>]*>(.*?)</span>',
                r'id="itemTitle"[^>]*>\s*(?:<span[^>]*>.*?</span>)?\s*(.*?)\s*</h1>',
                r'<h1[^>]*class="[^"]*it-ttl[^"]*"[^>]*>(.*?)</h1>',
            ])

        # Buy It Now / current price
        if not data.price:
            price_text = extract_text(html, [
                r'class="x-price-primary"[^>]*>\s*<span[^>]*>(.*?)</span>',
                r'id="prcIsum"[^>]*>(.*?)</span>',
                r'class="[^"]*notranslate[^"]*"[^>]*>([\d.,]+)',
                r'itemprop="price"[^>]*content="([^"]+)"',
            ])
            if price_text:
                data.price = parse_price(price_text, currency)

        # Bid price fallback
        if not data.price:
            bid_text = extract_text(html, [
                r'id="prcIsum_bidPrice"[^>]*>(.*?)</span>',
                r'class="x-price-primary"[^>]*>.*?<span[^>]*>(.*?)</span>',
            ])
            if bid_text:
                data.price = parse_price(bid_text, currency)

        # Seller
        data.seller = extract_text(html, [
            r'class="x-sellercard-atf__info__about-seller"[^>]*>\s*<span[^>]*>(.*?)</span>',
            r'class="[^"]*si-content[^"]*"[^>]*>\s*<a[^>]*>(.*?)</a>',
            r'class="mbg-nw"[^>]*>(.*?)</span>',
        ])

        # Image
        img_match = re.search(
            r'class="[^"]*ux-image-carousel-item[^"]*"[^>]*>\s*<img[^>]+src="([^"]+)"',
            html
        ) or re.search(
            r'id="icImg"[^>]*src="([^"]+)"', html
        )
        if img_match:
            data.image_url = img_match.group(1)

        # Stock
        if data.stock_status == "unknown":
            html_lower = html.lower()
            if "ended" in html_lower or "bidding has ended" in html_lower:
                data.stock_status = "out_of_stock"
            elif data.price:
                data.stock_status = "in_stock"

        return data
