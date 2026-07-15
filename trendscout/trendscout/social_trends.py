#!/usr/bin/env python3
"""
Social trending-topics signals, scraped via the existing self-hosted
Firecrawl instance — no platform accounts/auth involved.

Sources:
  - trends24.in: public aggregator mirroring X/Twitter's regional trending
    hashtags/terms.
  - TikTok Creative Center: TikTok's public "Popular Hashtags" tool (no
    login required), geo defaults to whatever region Firecrawl's egress IP
    resolves to.

These terms don't become `posts` rows (no permalink/engagement metrics of
their own) — they're used purely as a term-frequency signal for emergence
detection and as a "trending now" digest section.
"""

import re

from . import firecrawl_client

# ── trends24.in (X/Twitter) ─────────────────────────────────────────────────

_TRENDS24_BASE_URL = 'https://trends24.in/'

# trends24.in serves worldwide trends from the site root, all other regions
# from a slug path (e.g. /australia/, /united-states/).
_WORLDWIDE_REGION = 'worldwide'

_TRENDS24_HEADING_RE = re.compile(
    r'^###\s+(?:(?:a\s+few|few|\d+)\s+(?:minute|hour)s?\s+ago|now)\s*$', re.IGNORECASE
)
_TRENDS24_ITEM_RE = re.compile(r'^\d+\.\s+\[(?P<term>.+?)\]\(https://twitter\.com/search')


def _parse_trends24(markdown: str, top_n: int) -> list[str]:
    """Extract trending terms from the most recent hourly block."""
    terms = []
    in_block = False
    for line in markdown.splitlines():
        if _TRENDS24_HEADING_RE.match(line):
            if in_block:
                break  # second heading reached => end of the most-recent block
            in_block = True
            continue
        if not in_block:
            continue
        match = _TRENDS24_ITEM_RE.match(line.strip())
        if match:
            terms.append(match.group('term'))
            if len(terms) >= top_n:
                break
    return terms


def fetch_x_trends(regions: list[str], top_n: int = 10) -> dict[str, list[str]]:
    """Fetch current X/Twitter trending terms for each region from trends24.in."""
    trends = {}
    for region in regions:
        url = _TRENDS24_BASE_URL if region == _WORLDWIDE_REGION else f'{_TRENDS24_BASE_URL}{region}/'
        markdown = firecrawl_client.scrape(url)
        if markdown:
            trends[region] = _parse_trends24(markdown, top_n)
    return trends


# ── TikTok Creative Center (Popular Hashtags) ───────────────────────────────

_TIKTOK_URL = 'https://ads.tiktok.com/business/creativecenter/inspiration/popular/hashtag/pc/en'
_TIKTOK_WAIT_MS = 6000

_TIKTOK_HASHTAG_RE = re.compile(r'^#(\S+)$')


def _parse_tiktok_hashtags(markdown: str, top_n: int) -> list[str]:
    """Extract hashtags from the Popular Hashtags table (the page only
    renders a handful before a JS-driven "View more" expansion)."""
    terms = []
    started = False
    for line in markdown.splitlines():
        line = line.strip()
        if line == 'Action':
            started = True  # last column header before the hashtag rows
            continue
        if not started:
            continue
        if line == 'View more':
            break
        match = _TIKTOK_HASHTAG_RE.match(line)
        if match:
            terms.append(f'#{match.group(1)}')
            if len(terms) >= top_n:
                break
    return terms


def fetch_tiktok_trends(top_n: int = 10) -> list[str]:
    """Fetch current popular hashtags from TikTok's Creative Center."""
    markdown = firecrawl_client.scrape(_TIKTOK_URL, wait_for=_TIKTOK_WAIT_MS)
    if not markdown:
        return []
    return _parse_tiktok_hashtags(markdown, top_n)


# ── Combined ─────────────────────────────────────────────────────────────────

def fetch_all(config: dict) -> dict[str, list[str]]:
    """Fetch all enabled social-trend sources, keyed by label for the digest."""
    top_n = config.get('top_n', 10)
    trends = {}

    x_config = config.get('x', {})
    if x_config.get('regions'):
        for region, terms in fetch_x_trends(x_config['regions'], top_n).items():
            trends[f'x/{region}'] = terms

    if config.get('tiktok', {}).get('enabled'):
        terms = fetch_tiktok_trends(top_n)
        if terms:
            trends['tiktok'] = terms

    return trends


def normalize_term(term: str) -> str:
    """Normalize a trending term for term_frequency tracking."""
    term = term.strip().lstrip('#').strip().lower()
    return re.sub(r'\s+', ' ', term)
