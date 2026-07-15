#!/usr/bin/env python3
"""Shared Firecrawl scrape helper for trendscout ingestion modules.

The Firecrawl API URL is resolved at call time with the following precedence:
1. Explicit `api_url` argument to scrape()
2. FIRECRAWL_API_URL environment variable
3. Default fallback: http://localhost:3002

This avoids the import-order bug where the URL was read at module import
(before config.load_config() ran).
"""
import json
import os
import urllib.error
import urllib.request

REQUEST_TIMEOUT = 60
DEFAULT_FIRECRAWL_URL = 'http://localhost:3002'
DEFAULT_FIRECRAWL_KEY = 'local_secret_key'


def scrape(url: str, wait_for: int | None = None, api_url: str | None = None, api_key: str | None = None) -> str:
    """Scrape a URL with Firecrawl, return markdown or empty string.

    `wait_for` (milliseconds) lets JS-rendered SPAs finish loading before
    the page is captured.

    Args:
        url: The URL to scrape
        wait_for: Optional milliseconds to wait for JS rendering
        api_url: Firecrawl API endpoint (default: FIRECRAWL_API_URL env or localhost:3002)
        api_key: API key (default: FIRECRAWL_API_KEY env or 'local_secret_key')
    """
    # Resolve at call time — not module import
    base_url = api_url or os.getenv('FIRECRAWL_API_URL', DEFAULT_FIRECRAWL_URL)
    key = api_key or os.getenv('FIRECRAWL_API_KEY', DEFAULT_FIRECRAWL_KEY)

    try:
        body = {'url': url, 'formats': ['markdown']}
        if wait_for:
            body['waitFor'] = wait_for
        payload = json.dumps(body).encode()
        req = urllib.request.Request(
            f'{base_url}/v1/scrape',
            data=payload,
            headers={
                'Content-Type': 'application/json',
                'Authorization': f'Bearer {key}',
            },
            method='POST',
        )
        with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as resp:
            data = json.loads(resp.read())
            if not data.get('success'):
                return ''
            return data.get('data', {}).get('markdown', '') or ''
    except (urllib.error.URLError, Exception):
        return ''
