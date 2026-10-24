"""Preserve the delivery metadata of a checksummed real Firecrawl extraction through disk caching."""
import asyncio
import hashlib
import json
import os
from pathlib import Path

import pytest


@pytest.fixture
def extraction(tmp_path):
    previous = os.environ.get('HERMES_HOME')
    os.environ['HERMES_HOME'] = str(tmp_path)
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    token = set_hermes_home_override(tmp_path)
    from hermes_cli.config import atomic_config_write

    atomic_config_write(tmp_path / 'config.yaml', {
        'web': {'extract_backend': 'firecrawl', 'cache_enabled': True, 'cache_ttl_minutes': 20},
    })
    path = Path(__file__).with_name('actual-public-extract-response.json')
    assert hashlib.sha256(path.read_bytes()).hexdigest() == '6f73f7b88335a8fc8417745e8ec3e51dfdb90323467692b5bbc7af274558a104'
    recorded = json.loads(path.read_bytes())
    assert recorded['provenance']['served_by'] == 'firecrawl'
    assert recorded['provenance']['provider_call_attempted'] is True
    try:
        yield recorded, tmp_path
    finally:
        reset_hermes_home_override(token)
        if previous is None:
            os.environ.pop('HERMES_HOME')
        else:
            os.environ['HERMES_HOME'] = previous


def test_actual_disk_cache_retains_the_recorded_fetch_time(extraction):
    from tools.web_result_cache import extract_cache_get, extract_cache_put

    recorded, home = extraction
    page, facts = recorded['results'][0], recorded['provenance']
    assert extract_cache_get(page['url'], provider='firecrawl') is None
    extract_cache_put(page['url'], page['content'], page['title'], provider='firecrawl', retrieved_at=facts['retrieved_at'])
    cached = extract_cache_get(page['url'], provider='firecrawl')
    assert cached is not None
    assert cached['content'] == page['content'] and cached['title'] == page['title']
    assert cached['_hermes_extract_retrieved_at'] == facts['retrieved_at']
    assert cached['_hermes_extract_served_by'] == 'firecrawl'
    assert list((home / 'cache/web').glob('*.json'))
    assert extract_cache_get(page['url'], provider='exa') is None
    assert extract_cache_get(page['url'], provider='firecrawl', format='text') is None


def test_real_extract_cache_stage_serves_recorded_content_without_a_new_fetch(extraction):
    from tools.web_result_cache import extract_cache_put
    from tools.web_tools_extract import _extract_safe_urls
    from tools.web_extract_provenance import build_extract_provenance, new_extract_facts
    from plugins.web.firecrawl.provider import FirecrawlWebSearchProvider

    recorded, _home = extraction
    page, original = recorded['results'][0], recorded['provenance']
    extract_cache_put(page['url'], page['content'], page['title'], provider='firecrawl', retrieved_at=original['retrieved_at'])
    provider = FirecrawlWebSearchProvider()
    facts = new_extract_facts(1)
    facts['requested_backend'] = provider.name
    results = asyncio.run(_extract_safe_urls(provider, [page['url']], None, facts=facts))
    facts = build_extract_provenance(results, facts)
    assert facts['cache_status'] == 'hit' and facts['provider_call_attempted'] is False
    assert facts['retrieved_at'] == original['retrieved_at']
    assert facts['fetch_succeeded'] is False and facts['page_freshness_verified'] is False
    assert facts['requested_backend'] == facts['served_by'] == 'firecrawl'
    assert facts['serving_backend_complete'] is True
    assert facts['success_count'] == facts['returned_count'] == facts['requested_count'] == 1
    assert facts['failure_count'] == 0 and facts['fallback_attempted'] is False
    assert results[0]['content'] == page['content']
