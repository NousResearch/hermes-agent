"""Abstract base class for site-specific HTML parsers."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Any


@dataclass
class ProductData:
    """Parsed product information from a web page."""
    name: str = ""
    price: Optional[float] = None
    original_price: Optional[float] = None
    currency: str = "auto"  # auto-detected from URL/HTML
    stock_status: str = "unknown"  # in_stock, out_of_stock, limited, unknown
    seller: str = ""
    image_url: str = ""
    category: str = ""
    rating: Optional[float] = None
    review_count: Optional[int] = None
    source_url: str = ""  # URL this data was scraped from


class BaseSiteParser(ABC):
    """Base class for site-specific parsers.

    To add support for a new site:
    1. Create a new file in parsers/ (e.g., ``trendyol.py``)
    2. Subclass ``BaseSiteParser``
    3. Implement all abstract methods
    4. Register with ``register_parser()`` from ``parsers/__init__.py``
    """

    @abstractmethod
    def get_site_name(self) -> str:
        """Return the human-readable site name (e.g., 'Amazon Türkiye')."""

    @abstractmethod
    def get_domains(self) -> List[str]:
        """Return list of domains this parser handles (e.g., ['amazon.com.tr'])."""

    @abstractmethod
    def can_handle(self, url: str) -> bool:
        """Return True if this parser can handle the given URL."""

    @abstractmethod
    def parse(self, html: str, url: str = "") -> ProductData:
        """Parse HTML and return structured product data."""

    def search(self, query: str, scraper: Any) -> Optional[dict]:
        """Search for a product on this site. Returns {name, url, price, site, ...}
        
        Default implementation returns None (no search support).
        """
        return None
