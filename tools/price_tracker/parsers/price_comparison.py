"""Price comparison site parsers — Idealo, PriceSpy, CamelCamelCamel.

These sites aggregate prices from multiple sellers. The parsers extract
the lowest available price and seller information.
"""

import logging
import re
from typing import List

from tools.price_tracker.parsers.base import BaseSiteParser, ProductData
from tools.price_tracker.parsers.price_utils import (
    detect_currency, parse_price, extract_text, extract_json_ld, extract_meta,
)

logger = logging.getLogger(__name__)


class IdealoParser(BaseSiteParser):
    """Parser for idealo.de price comparison pages."""

    def get_site_name(self) -> str:
        return "Idealo"

    def get_domains(self) -> List[str]:
        return ["idealo.de", "www.idealo.de", "idealo.co.uk", "www.idealo.co.uk",
                "idealo.fr", "www.idealo.fr", "idealo.it", "www.idealo.it",
                "idealo.es", "www.idealo.es"]

    def can_handle(self, url: str) -> bool:
        return "idealo." in url.lower()

    def parse(self, html: str, url: str = "") -> ProductData:
        data = ProductData()
        currency = detect_currency(url, html)
        data.currency = currency

        # Try JSON-LD first (Idealo uses it)
        for block in extract_json_ld(html):
            if isinstance(block, dict) and "Product" in str(block.get("@type", "")):
                data.name = block.get("name", "")
                offers = block.get("offers", {})
                if isinstance(offers, dict):
                    low = offers.get("lowPrice", offers.get("price", ""))
                    if low:
                        try:
                            data.price = float(low)
                        except (ValueError, TypeError):
                            data.price = parse_price(str(low), currency)
                    pc = offers.get("priceCurrency", "")
                    if pc:
                        data.currency = pc.upper()
                    data.stock_status = "in_stock" if data.price else "out_of_stock"
                if data.name:
                    break

        # HTML fallbacks
        if not data.name:
            data.name = extract_text(html, [
                r'<h1[^>]*>\s*(.*?)\s*</h1>',
                r'class="[^"]*productName[^"]*"[^>]*>(.*?)</',
            ])

        if not data.price:
            price_text = extract_text(html, [
                r'class="[^"]*productOffers-listItemOfferPrice[^"]*"[^>]*>(.*?)</',
                r'class="[^"]*offerList-item-priceMin[^"]*"[^>]*>(.*?)</',
            ])
            if price_text:
                data.price = parse_price(price_text, currency)

        data.seller = "Multiple Sellers (Idealo)"
        if data.price and data.stock_status == "unknown":
            data.stock_status = "in_stock"

        data.image_url = extract_meta(html, "image") or ""
        return data


class PriceSpyParser(BaseSiteParser):
    """Parser for pricespy.com / prisjakt / pricerunner."""

    def get_site_name(self) -> str:
        return "PriceSpy"

    def get_domains(self) -> List[str]:
        return ["pricespy.com", "www.pricespy.com",
                "pricespy.co.uk", "www.pricespy.co.uk",
                "prisjakt.nu", "www.prisjakt.nu",
                "pricerunner.com", "www.pricerunner.com",
                "pricerunner.se", "www.pricerunner.se",
                "pricerunner.dk", "www.pricerunner.dk"]

    def can_handle(self, url: str) -> bool:
        url_lower = url.lower()
        return any(d in url_lower for d in ["pricespy.", "prisjakt.", "pricerunner."])

    def parse(self, html: str, url: str = "") -> ProductData:
        data = ProductData()
        currency = detect_currency(url, html)
        data.currency = currency

        # JSON-LD extraction
        for block in extract_json_ld(html):
            if isinstance(block, dict) and "Product" in str(block.get("@type", "")):
                data.name = block.get("name", "")
                offers = block.get("offers", {})
                if isinstance(offers, dict):
                    low = offers.get("lowPrice", offers.get("price", ""))
                    if low:
                        try:
                            data.price = float(low)
                        except (ValueError, TypeError):
                            data.price = parse_price(str(low), currency)
                    pc = offers.get("priceCurrency", "")
                    if pc:
                        data.currency = pc.upper()
                if data.name:
                    break

        if not data.name:
            data.name = extract_text(html, [r'<h1[^>]*>(.*?)</h1>'])

        if not data.price:
            price_text = extract_text(html, [
                r'class="[^"]*price[^"]*"[^>]*>([\d.,\s]+)',
            ])
            if price_text:
                data.price = parse_price(price_text, currency)

        data.seller = "Multiple Sellers (PriceSpy)"
        data.image_url = extract_meta(html, "image") or ""

        if data.price:
            data.stock_status = "in_stock"
        return data


class CamelParser(BaseSiteParser):
    """Parser for camelcamelcamel.com (Amazon price history)."""

    def get_site_name(self) -> str:
        return "CamelCamelCamel"

    def get_domains(self) -> List[str]:
        return ["camelcamelcamel.com", "www.camelcamelcamel.com"]

    def can_handle(self, url: str) -> bool:
        return "camelcamelcamel.com" in url.lower()

    def parse(self, html: str, url: str = "") -> ProductData:
        data = ProductData()
        data.currency = "USD"

        # Product name
        data.name = extract_text(html, [
            r'<h1[^>]*class="[^"]*product_title[^"]*"[^>]*>(.*?)</h1>',
            r'<title>(.*?)(?:\s*\||\s*-)',
        ])

        # Current Amazon price
        price_text = extract_text(html, [
            r'id="[^"]*lowest_price[^"]*"[^>]*>(.*?)</',
            r'class="[^"]*product_pricetag[^"]*"[^>]*>(.*?)</',
        ])
        if price_text:
            data.price = parse_price(price_text, "USD")

        # Image
        img_match = re.search(r'class="[^"]*product_image[^"]*"[^>]*src="([^"]+)"', html)
        if img_match:
            data.image_url = img_match.group(1)

        data.seller = "Amazon (via CamelCamelCamel)"
        if data.price:
            data.stock_status = "in_stock"
        return data
