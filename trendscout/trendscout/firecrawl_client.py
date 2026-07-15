#!/usr/bin/env python3
#!/usr/bin/env python3
"""Shared Firecrawl scrape helper for trendscout ingestion modules.

Firecrawl URL and API key are read from environment variables, which should be
set by trendscout.config.load_config() from the firecrawl.api_url config value.
"""
import json
import os
import urllib.error
import urllib.request

REQUEST_TIMEOUT = 60

# These are set by config.load_config() from firecrawl.api_url
# Default fallback only for standalone usage (not recommended)
FIRECRAWL_URL = os.getenv('FIRECRAWL_API_URL')
FIRECRAWL_KEY = os.getenv('FIRECRAWL_API_KEY', 'local_secret_key')


def scrape(url: str, wait_for: int = None) -> str:
    """Scrape a URL with Firecrawl, return markdown or empty string.

    `wait_for` (milliseconds) lets JS-rendered SPAs finish loading before
    the page is captured.
    """
    try:
        body = {'url': url, 'formats': ['markdown']}
        if wait_for:
            body['waitFor'] = wait_for
        payload = json.dumps(body).encode()
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
