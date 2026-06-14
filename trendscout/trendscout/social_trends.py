#!/usr/bin/env python3
"""
Social trending-topics signal via trends24.in — a public aggregator that
mirrors X/Twitter's regional trending hashtags/terms (no X account or
auth involved), scraped via the existing self-hosted Firecrawl instance.

These terms don't become `posts` rows (no permalink/engagement metrics of
their own) — they're used purely as a term-frequency signal for emergence
detection and as a "trending now" digest section.
"""

import re

from . import firecrawl_client

_BASE_URL = 'https://trends24.in/'

# trends24.in serves worldwide trends from the site root, all other regions
# from a slug path (e.g. /australia/, /united-states/).
_WORLDWIDE_REGION = 'worldwide'

_HEADING_RE = re.compile(
    r'^###\s+(?:(?:a\s+few|few|\d+)\s+(?:minute|hour)s?\s+ago|now)\s*$', re.IGNORECASE
)
_ITEM_RE = re.compile(r'^\d+\.\s+\[(?P<term>.+?)\]\(https://twitter\.com/search')


def _parse_trends24(markdown: str, top_n: int) -> list[str]:
    """Extract trending terms from the most recent hourly block."""
    terms = []
    in_block = False
    for line in markdown.splitlines():
        if _HEADING_RE.match(line):
            if in_block:
                break  # second heading reached => end of the most-recent block
            in_block = True
            continue
        if not in_block:
            continue
        match = _ITEM_RE.match(line.strip())
        if match:
            terms.append(match.group('term'))
            if len(terms) >= top_n:
                break
    return terms


def fetch_trends(regions: list[str], top_n: int = 10) -> dict[str, list[str]]:
    """Fetch current trending terms for each region from trends24.in."""
    trends = {}
    for region in regions:
        url = _BASE_URL if region == _WORLDWIDE_REGION else f'{_BASE_URL}{region}/'
        markdown = firecrawl_client.scrape(url)
        if markdown:
            trends[region] = _parse_trends24(markdown, top_n)
    return trends


def normalize_term(term: str) -> str:
    """Normalize a trending term for term_frequency tracking."""
    term = term.strip().lstrip('#').strip().lower()
    return re.sub(r'\s+', ' ', term)
