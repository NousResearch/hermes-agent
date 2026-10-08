"""ClawHub keyword discovery uses the search API and preserves publisher identity."""

import json
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlsplit

from tools.skills_hub_clawhub import ClawHubSource


@contextmanager
def search_registry(results):
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            url = urlsplit(self.path)
            query = parse_qs(url.query)
            requests.append((url.path, query))
            if url.path.endswith('/search'):
                payload = {'results': results}
            elif query.get('owner') == ['alice']:
                payload = {'skill': {'slug': 'collision', 'displayName': 'Alice skill'},
                           'owner': {'handle': 'alice'}, 'latestVersion': {'version': '1.0'}}
            elif url.path.endswith('/skills'):
                # The listing endpoint ignores keyword filters and only returns recent skills.
                payload = {'items': [{'slug': 'unrelated', 'displayName': 'Unrelated'}]}
            else:
                self.send_error(404)
                return
            body = json.dumps(payload).encode()
            self.send_response(200)
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    source = ClawHubSource()
    source.BASE_URL = f'http://127.0.0.1:{server.server_port}/api/v1'
    try:
        yield source, requests
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_search_discovers_non_recent_skills_and_keeps_same_slug_publishers():
    rows = [{'slug': 'collision', 'displayName': 'Phone calls',
             'summary': 'Place phone calls', 'ownerHandle': owner}
            for owner in ('alice', 'bob')]
    with search_registry(rows) as (source, requests):
        results = source.search('phone calls', limit=5)
        assert [meta.identifier for meta in results] == ['@alice/collision', '@bob/collision']
        assert all(meta.source == 'clawhub' and meta.trust_level == 'community' for meta in results)
        assert requests == [('/api/v1/search', {'q': ['phone calls'], 'limit': ['5']})]
        before = len(requests)
        assert source.search('phone calls', limit=5) == results
        assert len(requests) == before
        assert source.inspect(results[0].identifier).identifier == results[0].identifier
        assert requests[-1][1]['owner'] == ['alice']


def test_search_empty_result_is_authoritative_and_does_not_walk_catalog():
    with search_registry([]) as (source, requests):
        assert source.search('missing capability', limit=5) == []
        assert requests == [('/api/v1/search', {'q': ['missing capability'], 'limit': ['5']})]
