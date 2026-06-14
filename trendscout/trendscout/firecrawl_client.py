#!/usr/bin/env python3
"""Shared Firecrawl scrape helper for trendscout ingestion modules."""

import json
import os
import urllib.error
import urllib.request

REQUEST_TIMEOUT = 60

FIRECRAWL_URL = os.getenv('FIRECRAWL_API_URL', 'http://localhost:3002')
FIRECRAWL_KEY = os.getenv('FIRECRAWL_API_KEY', 'local_secret_key')


def scrape(url: str) -> str:
    """Scrape a URL with Firecrawl, return markdown or empty string."""
    try:
        payload = json.dumps({'url': url, 'formats': ['markdown']}).encode()
        req = urllib.request.Request(
            f'{FIRECRAWL_URL}/v1/scrape',
            data=payload,
            headers={
                'Content-Type': 'application/json',
                'Authorization': f'Bearer {FIRECRAWL_KEY}',
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
