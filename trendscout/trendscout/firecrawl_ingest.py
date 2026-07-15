#!/usr/bin/env python3
"""
Firecrawl-based ingestion for non-Reddit sources listed in config/urls.txt.

Scrapes each URL via the self-hosted Firecrawl instance (formats=["markdown"])
and parses recognised listing formats into the same post dict shape used by
reddit_ingest.py: id, source, subreddit, title, selftext, author, permalink,
url, created_utc, score, num_comments.

Currently understands Hacker News-style listing pages (item?id=... links,
"N points by X ... N comments"). URLs whose domain isn't recognised are
skipped — add a new _PARSERS entry to support more sources.
"""

import re
import time
from datetime import datetime, timedelta, timezone

from . import firecrawl_client

REQUEST_DELAY_SECONDS = 1.0

_HN_TITLE_RE = re.compile(
    r'^\|\s*\d+\.\s*\|\s*\[.*?\]\([^)]*\)\s*\|\s*\[(?P<title>.+?)\]\((?P<url>https?://[^)]+)\)'
)
_HN_META_RE = re.compile(
    r'(?P<points>\d+)\s+points by \[(?P<author>[^\]]+)\].*?'
    r'\[(?P<age>\d+)\s+(?P<unit>minute|hour|day)s?\s+ago\]'
    r'\(https://news\.ycombinator\.com/item\?id=(?P<id>\d+)\)'
    r'(?:.*?\[(?P<comments>\d+)\s+comments?\])?'
)


def _age_to_created_utc(age: int, unit: str, now: datetime) -> float:
    delta = {
        'minute': timedelta(minutes=age),
        'hour': timedelta(hours=age),
        'day': timedelta(days=age),
    }[unit]
    return (now - delta).timestamp()


def _parse_hackernews(markdown: str) -> list[dict]:
    now = datetime.now(timezone.utc)
    posts = []
    pending = None

    for line in markdown.splitlines():
        title_match = _HN_TITLE_RE.match(line)
        if title_match:
            pending = title_match
            continue
        if pending is None:
            continue
        meta_match = _HN_META_RE.search(line)
        if meta_match:
            item_id = meta_match.group('id')
            posts.append({
                'id': f'hn_{item_id}',
                'source': 'hackernews',
                'subreddit': None,
                'title': pending.group('title'),
                'selftext': '',
                'author': meta_match.group('author'),
                'permalink': f'https://news.ycombinator.com/item?id={item_id}',
                'url': pending.group('url'),
                'created_utc': _age_to_created_utc(
                    int(meta_match.group('age')), meta_match.group('unit'), now
                ),
                'score': int(meta_match.group('points')),
                'num_comments': int(meta_match.group('comments') or 0),
            })
        pending = None

    return posts


_PARSERS = {
    'news.ycombinator.com': _parse_hackernews,
}


def fetch_url(url: str) -> list[dict]:
    """Scrape a single URL via Firecrawl and parse it into post dicts.

    Returns an empty list for unrecognised domains or failed scrapes.
    """
    parser = next((fn for domain, fn in _PARSERS.items() if domain in url), None)
    if parser is None:
        return []

    markdown = firecrawl_client.scrape(url)
    if not markdown:
        return []
    return parser(markdown)


def fetch_all(urls: list[str]) -> list[dict]:
    """Fetch posts for every URL in the list, in order."""
    all_posts: list[dict] = []
    for url in urls:
        all_posts.extend(fetch_url(url))
        time.sleep(REQUEST_DELAY_SECONDS)
    return all_posts
