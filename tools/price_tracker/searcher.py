"""Multi-source product searcher — searches Amazon, eBay, Best Buy, Newegg, Walmart concurrently.

Returns price data from ALL available sources so the caller can build a
Market Overview comparison table and pick the Best Deal.
"""

import logging
import re
import urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, Dict, Any, List

from tools.price_tracker.scraper import StealthScraper

logger = logging.getLogger(__name__)


class ProductSearcher:
    """Searches for products across multiple international stores."""

    # Maximum concurrent search threads
    MAX_WORKERS = 5

    def __init__(self, scraper: Optional[StealthScraper] = None):
        if scraper is None:
            from tools.price_tracker.scraper import StealthScraper
            self.scraper = StealthScraper()
        else:
            self.scraper = scraper

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def search_all_sources(self, query: str) -> List[Dict[str, Any]]:
        """Search for a product across all supported stores concurrently.

        Returns a list of results sorted by price (cheapest first).
        Each result: {success, site, url, name, price, stock_status}
        Failed/blocked sources are omitted from the list.
        """
        logger.info("Multi-source search for: %s", query)

        search_fns = [
            ("Amazon", self._search_amazon),
            ("eBay", self._search_ebay),
            ("Best Buy", self._search_bestbuy),
            ("Newegg", self._search_newegg),
            ("Walmart", self._search_walmart),
        ]

        # Modular Expansion: Add any registered parsers that support search
        from tools.price_tracker.parsers import _PARSERS
        for parser in _PARSERS:
            # Avoid duplicating hardcoded ones if they are moved later
            site_name = parser.get_site_name()
            if site_name not in [s[0] for s in search_fns]:
                # Wrap search in a lambda that passes the scraper
                search_fns.append((site_name, lambda q, p=parser: p.search(q, self.scraper)))

        results: List[Dict[str, Any]] = []

        with ThreadPoolExecutor(max_workers=self.MAX_WORKERS) as executor:
            future_to_site = {
                executor.submit(fn, query): site_name
                for site_name, fn in search_fns
            }
            for future in as_completed(future_to_site):
                site_name = future_to_site[future]
                try:
                    result = future.result()
                    if result:
                        results.append(result)
                    else:
                        logger.info("No result from %s", site_name)
                except Exception as e:
                    logger.debug("Search failed for %s: %s", site_name, e)

        # Sort by price (cheapest first), put None prices at the end
        results.sort(key=lambda r: r.get("price") or float("inf"))
        return results

    def search(self, query: str) -> Dict[str, Any]:
        """Compatibility wrapper — returns the cheapest result (or first found).

        Used by the old single-source flow.
        """
        all_results = self.search_all_sources(query)
        if not all_results:
            return {
                "success": False,
                "error": "Product not found across any store, or blocked by anti-bot systems.",
            }

        best = all_results[0]
        best["all_results"] = all_results  # attach full list for downstream use
        return best

    # ------------------------------------------------------------------
    # Per-store search methods
    # ------------------------------------------------------------------

    def _search_amazon(self, query: str) -> Optional[Dict[str, Any]]:
        """Search via Amazon.com and scrape the first product page for price."""
        encoded_query = urllib.parse.quote_plus(query)
        search_url = f"https://www.amazon.com/s?k={encoded_query}"

        result = self.scraper.scrape(search_url)
        if not result.success:
            return None

        html = result.html

        # Check for captcha or sign-in wall
        if "api-services-support@amazon.com" in html or "Type the characters you see" in html \
                or "ap_email" in html:
            logger.debug("Amazon returned captcha/sign-in wall during search.")
            return {
                "success": True,
                "site": "Amazon",
                "url": search_url,
                "name": "",
                "price": None,
                "stock_status": "unknown",
                "note": "Amazon bot protection active.",
            }

        # Find first product link
        matches = re.finditer(
            r'<a\s+[^>]*href="(/[^"]*?/dp/[A-Z0-9]{10}[^"]*|/dp/[A-Z0-9]{10}[^"]*)"',
            html, re.IGNORECASE
        )

        product_url = None
        for match in matches:
            product_path = match.group(1)
            if "slredirect" not in product_path:
                base_path = product_path.split("?", 1)[0]
                product_url = f"https://www.amazon.com{base_path}"
                break

        if not product_url:
            return None

        # Scrape the product page for price
        return self._scrape_product_page(product_url, "Amazon")

    def _search_ebay(self, query: str) -> Optional[Dict[str, Any]]:
        """Search via eBay.com with stealth-mode headers."""
        encoded_query = urllib.parse.quote_plus(query)
        # Buy It Now only, sorted by price + shipping
        search_url = f"https://www.ebay.com/sch/i.html?_nkw={encoded_query}&LH_BIN=1&_sop=15"

        result = self.scraper.scrape(search_url)
        if not result.success:
            return None

        html = result.html
        if "captcha" in html.lower() or "px-captcha" in html.lower():
            logger.debug("eBay returned captcha during search — shielded.")
            # Return a blocked result so the UI can inform the user
            return {
                "success": True,
                "site": "eBay",
                "url": search_url,
                "name": "",
                "price": None,
                "stock_status": "unknown",
                "note": "eBay is currently shielding, results from other stores shown.",
            }

        # Find first listing URL
        matches = re.finditer(
            r'<a\s+[^>]*href="(https://www\.ebay\.com/itm/[0-9]+[^"]*)"',
            html, re.IGNORECASE,
        )
        for match in matches:
            product_url = match.group(1).split("?", 1)[0]
            return self._scrape_product_page(product_url, "eBay")

        return None

    def _search_bestbuy(self, query: str) -> Optional[Dict[str, Any]]:
        """Search via Best Buy."""
        encoded_query = urllib.parse.quote_plus(query)
        search_url = f"https://www.bestbuy.com/site/searchpage.jsp?st={encoded_query}"

        result = self.scraper.scrape(search_url)
        if not result.success:
            return None

        html = result.html

        # Find first product link: /site/product-name/SKUID.p
        matches = re.finditer(
            r'<a\s+[^>]*href="(/site/[^"]+/\d+\.p[^"]*)"',
            html, re.IGNORECASE,
        )
        for match in matches:
            product_path = match.group(1).split("?", 1)[0]
            product_url = f"https://www.bestbuy.com{product_path}"
            return self._scrape_product_page(product_url, "Best Buy")

        return None

    def _search_newegg(self, query: str) -> Optional[Dict[str, Any]]:
        """Search via Newegg."""
        encoded_query = urllib.parse.quote_plus(query)
        search_url = f"https://www.newegg.com/p/pl?d={encoded_query}"

        result = self.scraper.scrape(search_url)
        if not result.success:
            return None

        html = result.html

        # Find first product link: /product-name/p/ITEMID
        matches = re.finditer(
            r'<a\s+[^>]*href="(https://www\.newegg\.com/[^"]+/p/[^"]+)"',
            html, re.IGNORECASE,
        )
        for match in matches:
            product_url = match.group(1).split("?", 1)[0]
            # Skip search page links
            if "/p/pl" in product_url:
                continue
            return self._scrape_product_page(product_url, "Newegg")

        return None

    def _search_walmart(self, query: str) -> Optional[Dict[str, Any]]:
        """Search via Walmart."""
        encoded_query = urllib.parse.quote_plus(query)
        search_url = f"https://www.walmart.com/search?q={encoded_query}"

        result = self.scraper.scrape(search_url)
        if not result.success:
            return None

        html = result.html

        # Find first product link: /ip/product-name/ITEMID
        matches = re.finditer(
            r'<a\s+[^>]*href="(/ip/[^"]+)"',
            html, re.IGNORECASE,
        )
        for match in matches:
            product_path = match.group(1).split("?", 1)[0]
            product_url = f"https://www.walmart.com{product_path}"
            return self._scrape_product_page(product_url, "Walmart")

        return None

    # ------------------------------------------------------------------
    # Shared product page scraper
    # ------------------------------------------------------------------

    def _scrape_product_page(self, url: str, site: str) -> Optional[Dict[str, Any]]:
        """Scrape the actual product page and extract price/name via the parser."""
        from tools.price_tracker.parsers import get_parser

        try:
            result = self.scraper.scrape(url)
            if not result.success:
                return {"success": True, "site": site, "url": url,
                        "name": "", "price": None, "stock_status": "unknown"}

            parser = get_parser(url)
            if not parser:
                return {"success": True, "site": site, "url": url,
                        "name": "", "price": None, "stock_status": "unknown"}

            data = parser.parse(result.html, url)
            return {
                "success": True,
                "site": site,
                "url": url,
                "name": data.name or "",
                "price": data.price,
                "stock_status": data.stock_status or "unknown",
                "original_price": data.original_price,
                "seller": data.seller or site,
            }
        except Exception as e:
            logger.debug("Failed to scrape %s (%s): %s", site, url, e)
            return {"success": True, "site": site, "url": url,
                    "name": "", "price": None, "stock_status": "unknown"}
