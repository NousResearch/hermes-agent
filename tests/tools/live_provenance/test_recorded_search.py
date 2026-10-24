"""Cache a checksummed response captured from the configured real search provider."""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path

import pytest


@pytest.fixture
def search_data(tmp_path):
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    home = tmp_path / 'home'
    home.mkdir()
    (home / 'config.yaml').write_text('web:\n  cache_enabled: true\n  cache_ttl_minutes: 20\n', encoding='utf-8')
    previous = os.environ.get('HERMES_HOME')
    os.environ['HERMES_HOME'] = str(home)
    token = set_hermes_home_override(home)
    path = Path(__file__).with_name('actual-public-search-response.json')
    assert hashlib.sha256(path.read_bytes()).hexdigest() == '6d5d229fda3ec19abd8a519647b131e5b682c4faf249081992eca286adf6e2b2'
    try:
        yield json.loads(path.read_bytes())
    finally:
        reset_hermes_home_override(token)
        if previous is None:
            os.environ.pop('HERMES_HOME')
        else:
            os.environ['HERMES_HOME'] = previous


def test_real_response_cache_retains_retrieval_time_and_defensive_copy(search_data):
    from tools.web_result_cache import SearchMemo

    memo = SearchMemo()
    query = 'NousResearch Hermes Agent official GitHub'
    provenance = search_data['data']['provenance']
    provider = provenance['served_by']
    retrieved = provenance['retrieved_at']
    assert memo.lookup_with_metadata(provider, query, 3) is None
    before = deepcopy(search_data)
    memo.store(provider, query, 10, search_data, retrieved_at=retrieved)
    first, observed, age, ttl = memo.lookup_with_metadata(provider, query, 3)
    assert observed == retrieved and age >= 0 and ttl == 1200
    first['data']['web'].clear()
    second, observed_again, next_age, _ = memo.lookup_with_metadata(provider, query, 5)
    assert second == before and observed_again == retrieved and next_age >= age
    assert search_data == before


def test_real_search_scope_and_bucket_slice_preserve_original_facts(search_data):
    from tools.web_result_cache import SearchMemo, slice_search_response
    from tools.web_tools_provenance import _inject_search_provenance, _validate_search_provider_response

    before = deepcopy(search_data)
    normalized = _validate_search_provider_response(search_data)
    original = normalized['data']['provenance']
    memo = SearchMemo()
    query = 'NousResearch Hermes Agent official GitHub'
    memo.store(original['served_by'], query, 10, normalized, retrieved_at=original['retrieved_at'])
    cached, retrieved, age, ttl = memo.lookup_with_metadata(original['served_by'], query, 3)
    result = _inject_search_provenance(
        slice_search_response(cached, 3), requested_backend=original['requested_backend'],
        requested_limit=3, fetched_result_count=len(normalized['data']['web']),
        retrieved_at=retrieved, cache_status='hit',
        cache_age_seconds=age, cache_ttl_seconds=ttl,
    )
    provenance = result['data']['provenance']
    assert result['data']['web'] == before['data']['web'][:3]
    assert provenance['retrieved_at'] == original['retrieved_at']
    assert provenance['page_fetched'] is False
    assert provenance['evidence_scope'] == 'search_result_metadata_only'
    assert provenance['returned_count'] == 3
    assert provenance['fetched_result_count'] == len(before['data']['web'])
    assert provenance['result_set_truncated'] is True
    assert provenance['cache']['status'] == 'hit'
    assert search_data == before
