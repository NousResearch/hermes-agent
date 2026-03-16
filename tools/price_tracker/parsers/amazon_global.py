"""Amazon Global parser — handles all Amazon domains.

Supports: amazon.com, amazon.co.uk, amazon.de, amazon.fr, amazon.it,
          amazon.es, amazon.com.tr
"""

import logging
import re
from typing import List, Optional

from tools.price_tracker.parsers.base import BaseSiteParser, ProductData
from tools.price_tracker.parsers.price_utils import (
    detect_currency, parse_price, extract_text,
)

logger = logging.getLogger(__name__)

_AMAZON_DOMAINS = [
    "amazon.com", "amazon.co.uk", "amazon.de", "amazon.fr",
    "amazon.it", "amazon.es", "amazon.com.tr",
]

_DOMAIN_NAMES = {
    "amazon.com": "Amazon US",
    "amazon.co.uk": "Amazon UK",
    "amazon.de": "Amazon Deutschland",
    "amazon.fr": "Amazon France",
    "amazon.it": "Amazon Italia",
    "amazon.es": "Amazon España",
    "amazon.com.tr": "Amazon Türkiye",
}


class AmazonGlobalParser(BaseSiteParser):
    """Parser for all Amazon product pages worldwide."""

    def get_site_name(self) -> str:
        return "Amazon (Global)"

    def get_domains(self) -> List[str]:
        domains = []
        for d in _AMAZON_DOMAINS:
            domains.extend([d, f"www.{d}"])
        return domains

    def can_handle(self, url: str) -> bool:
        url_lower = url.lower()
        return any(d in url_lower for d in _AMAZON_DOMAINS)

    def _detect_amazon_domain(self, url: str) -> str:
        """Return the specific Amazon domain from a URL."""
        url_lower = url.lower()
        # Check longer domains first (amazon.com.tr before amazon.com)
        for d in sorted(_AMAZON_DOMAINS, key=len, reverse=True):
            if d in url_lower:
                return d
        return "amazon.com"

    def parse(self, html: str, url: str = "") -> ProductData:
        data = ProductData()
        domain = self._detect_amazon_domain(url)
        currency = detect_currency(url, html)
        data.currency = currency

        # --- Product name ---
        data.name = extract_text(html, [
            r'id="productTitle"[^>]*>\s*(.*?)\s*</span>',
            r'id="title"[^>]*>\s*<span[^>]*>\s*(.*?)\s*</span>',
            r'<title>(.*?)(?:\s*:\s*Amazon|\s*-\s*Amazon)',
        ])

        # --- Current price ---
        # Handle whole + fraction pattern (used across all Amazon domains)
        fraction_match = re.search(
            r'class="a-price-whole">([\d.]+)</span>.*?class="a-price-fraction">([\d]+)',
            html, re.DOTALL
        )
        if fraction_match:
            whole = fraction_match.group(1).replace(".", "").replace(",", "")
            frac = fraction_match.group(2)
            try:
                data.price = float(f"{whole}.{frac}")
            except ValueError:
                pass

        if not data.price:
            price_text = extract_text(html, [
                r'class="a-price-whole"[^>]*>([\d.,]+)',
                r'class="a-price aok-align-center"[^>]*>.*?<span[^>]*>([\d.,]+\s*(?:TL|€|£|\$))',
                r'id="priceblock_ourprice"[^>]*>(.*?)</span>',
                r'id="priceblock_dealprice"[^>]*>(.*?)</span>',
                r'id="priceblock_saleprice"[^>]*>(.*?)</span>',
                r'class="a-price"[^>]*>.*?<span class="a-offscreen">([\d.,]+\s*(?:TL|€|£|\$))',
                r'<span class="a-offscreen">([\d.,]+\s*(?:TL|€|£|\$))</span>',
                r'id="corePrice_feature_div"[^>]*>.*?<span class="a-offscreen">([\d.,]+\s*(?:TL|€|£|\$))',
                r'id="corePriceDisplay_desktop_feature_div"[^>]*>.*?<span class="a-offscreen">([\d.,]+\s*(?:TL|€|£|\$))',
                r'data-a-color="price"[^>]*>\s*<span[^>]*>([\d.,]+\s*(?:TL|€|£|\$))',
            ])
            if price_text:
                data.price = parse_price(price_text, currency)

        # Fallback 2: Check embedded JSON data (often used by Amazon before JS renders)
        if not data.price:
            import json
            # Look for priceAmount in any twister/display JS setup
            price_amount_match = re.search(r'"priceAmount"\s*:\s*"?([\d.]+)"?', html)
            if price_amount_match:
                try:
                    data.price = float(price_amount_match.group(1))
                except ValueError:
                    pass

        # Fallback 3: Check twister-plus-price-data-price
        if not data.price:
            twister_match = re.search(r'data-price="([^"]+)"', html)
            if twister_match:
                data.price = parse_price(twister_match.group(1), currency)

        # --- Original (list) price ---
        orig_text = extract_text(html, [
            r'class="a-text-price"[^>]*>.*?<span[^>]*>([\d.,]+\s*(?:TL|€|£|\$))',
            r'class="a-text-price"[^>]*>\s*<span[^>]*>(.*?)</span>',
            r'class="a-price a-text-price"[^>]*>.*?<span class="a-offscreen">([\d.,]+\s*(?:TL|€|£|\$))',
            r'priceBlockStrikePriceString["\']:\s*["\']([^"\']+)',
            r'basisPrice["\']:\s*["\']([^"\']+)',
        ])
        if orig_text:
            data.original_price = parse_price(orig_text, currency)

        # --- Stock status ---
        availability_text = extract_text(html, [
            r'id="availability"[^>]*>\s*(.*?)\s*</(?:span|div)',
        ]).lower()

        # Multi-language out-of-stock keywords
        oos_keywords = [
            "stokta yok", "tükendi", "mevcut değil",  # TR
            "currently unavailable", "out of stock",  # EN
            "nicht verfügbar", "nicht auf lager",  # DE
            "actuellement indisponible", "en rupture",  # FR
            "non disponibile", "esaurito",  # IT
            "no disponible", "agotado",  # ES
        ]
        limited_keywords = ["sadece", "son", "sınırlı", "only", "left in stock", "nur noch"]
        in_stock_keywords = [
            "stokta", "mevcut", "kargoya",
            "in stock", "in den einkaufswagen", "ajouter au panier",
            "aggiungi al carrello", "añadir a la cesta",
        ]

        if any(w in availability_text for w in oos_keywords):
            data.stock_status = "out_of_stock"
        elif any(w in availability_text for w in limited_keywords):
            data.stock_status = "limited"
        elif any(w in availability_text for w in in_stock_keywords):
            data.stock_status = "in_stock"
        elif "addtocart" in html.lower() or "add-to-cart" in html.lower():
            data.stock_status = "in_stock"
        else:
            data.stock_status = "unknown"

        # --- Seller ---
        data.seller = extract_text(html, [
            r'id="sellerProfileTriggerId"[^>]*>(.*?)</a>',
            r'id="merchant-info"[^>]*>.*?<a[^>]*>(.*?)</a>',
            r'"merchantName"\s*:\s*"([^"]+)"',
        ])
        if not data.seller:
            html_lower = html.lower()
            # Sort by domain length descending to match amazon.com.tr before amazon.com
            sorted_domains = sorted(_DOMAIN_NAMES.items(), key=lambda x: len(x[0]), reverse=True)
            if url:
                for d, name in sorted_domains:
                    if d in url.lower():
                        if f"{d} tarafından" in html_lower or "ships from and sold by amazon" in html_lower:
                            data.seller = name
                        break
            else:
                # No URL provided — check all domains against HTML
                for d, name in sorted_domains:
                    if f"{d} tarafından" in html_lower:
                        data.seller = name
                        break
                if not data.seller and "ships from and sold by amazon" in html_lower:
                    data.seller = "Amazon"

        # --- Image ---
        img_match = re.search(
            r'id="landingImage"[^>]*src="([^"]+)"', html
        ) or re.search(
            r'"hiRes"\s*:\s*"([^"]+)"', html
        ) or re.search(
            r'id="imgBlkFront"[^>]*src="([^"]+)"', html
        )
        if img_match:
            data.image_url = img_match.group(1)

        # --- Category ---
        cats = re.findall(
            r'class="a-link-normal a-color-tertiary"[^>]*>\s*(.*?)\s*</a>',
            html,
        )
        if cats:
            data.category = " > ".join(c.strip() for c in cats if c.strip())

        # --- Rating ---
        rating_text = extract_text(html, [
            r'class="a-icon-alt"[^>]*>([\d,\.]+)\s*(?:üzerinden|out of|von|sur|su|de)',
        ])
        if rating_text:
            try:
                data.rating = float(rating_text.replace(",", "."))
            except ValueError:
                pass

        # --- Review count ---
        review_text = extract_text(html, [
            r'id="acrCustomerReviewText"[^>]*>([\d.,]+)',
        ])
        if review_text:
            try:
                data.review_count = int(review_text.replace(".", "").replace(",", ""))
            except ValueError:
                pass

        return data
