"""Parser registry and factory for site-specific HTML parsers.

Registration order matters: site-specific parsers are tried first,
GenericParser is the fallback (always matches).
"""

from typing import Optional, List
from tools.price_tracker.parsers.base import BaseSiteParser, ProductData

# The global registry of parser instances
_PARSERS: List[BaseSiteParser] = []


def register_parser(parser: BaseSiteParser):
    """Register a site parser (appended to the end of the list)."""
    _PARSERS.append(parser)


def register_fallback(parser: BaseSiteParser):
    """Register a fallback parser (always tried last)."""
    _PARSERS.append(parser)


# ---------------------------------------------------------------------------
# Import and register all parsers (order = priority)
# ---------------------------------------------------------------------------

# Layer 3: Site-specific parsers (highest priority)
# (Currently no standalone L3 parsers. Global sites belong in Layer 2 or Layer 1)

# Layer 2: Family parsers
from .amazon_global import AmazonGlobalParser
from .ebay_global import EbayGlobalParser
from .bestbuy import BestBuyParser
from .newegg import NeweggParser
from .walmart import WalmartParser
from .price_comparison import IdealoParser, PriceSpyParser, CamelParser

register_parser(AmazonGlobalParser())
register_parser(EbayGlobalParser())
register_parser(BestBuyParser())
register_parser(NeweggParser())
register_parser(WalmartParser())
register_parser(IdealoParser())
register_parser(PriceSpyParser())
register_parser(CamelParser())

# Layer 1: GenericParser (fallback — always matches)
from .generic import GenericParser

register_fallback(GenericParser())


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_parser(url: str) -> Optional[BaseSiteParser]:
    """Return the first parser that can handle *url*, or None.

    Site-specific parsers are tried before GenericParser.
    GenericParser always matches, so this effectively never returns None
    unless the registry is empty.
    """
    for parser in _PARSERS:
        if parser.can_handle(url):
            return parser
    return None


def list_supported_sites() -> list:
    """Return metadata for all registered parsers (excluding GenericParser)."""
    return [
        {"site": p.get_site_name(), "domains": p.get_domains()}
        for p in _PARSERS
        if p.get_domains()  # GenericParser has empty domains list
    ]
