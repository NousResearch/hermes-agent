"""A navigation-only listing reads committed rows; accounting-aware listings still drain usage."""
from concurrent.futures import ThreadPoolExecutor
from threading import Event

import pytest

from tests.fixtures.storage_holder import raw_holder
from tests.fixtures.storage_seed import seed_session
from persistence_assembly import open_persistence, profile_scope
from persistence.conversation.sessions import SessionQuery, SessionWindow
from persistence.conversation.usage import TokenCounts, UsageDelta


@pytest.mark.parametrize('with_tip', [False, True])
def test_navigation_list_does_not_wait_for_uncommitted_usage(tmp_path, monkeypatch, with_tip):
    from tui_gateway import server
    store = open_persistence(profile_scope(home=tmp_path))
    monkeypatch.setattr(server, '_db', store)
    monkeypatch.setattr(server, '_hermes_home', tmp_path)
    holder = raw_holder(store)
    hold, release = Event(), Event()

    def unrelated_write(conn):
        assert conn.in_transaction
        conn.execute("UPDATE sessions SET title='new' WHERE id='other'")
        hold.set()
        assert release.wait(10)

    def rpc():
        response = server.handle_request({'jsonrpc': '2.0', 'id': 17, 'method': 'session.list', 'params': {'limit': 5}})
        assert 'result' in response, response
        return response['result']['sessions']

    try:
        seed_session(store, 'other', 'cli', title='old')
        seed_session(store, 'root', 'cli')
        if with_tip:
            seed_session(store, 'tip', 'cli', parent_session_id='root')
            holder._write_sql("UPDATE sessions SET ended_at=started_at, end_reason='compression' WHERE id='root'")
        expected = rpc()
        assert {r['id'] for r in expected} == {'other', 'tip' if with_tip else 'root'}
        if with_tip:
            assert next(r for r in expected if r['id'] == 'tip')['_lineage_root_id'] == 'root'
        with ThreadPoolExecutor(max_workers=2) as pool:
            writer = pool.submit(holder._execute_write, unrelated_write)
            try:
                assert hold.wait(3)
                store.usage.queue_usage('other', UsageDelta(tokens=TokenCounts(input_tokens=19), source='cli'))
                assert holder.flush_token_counts(timeout=0) is False
                assert pool.submit(rpc).result(timeout=2) == expected
                assert not writer.done()
            finally:
                release.set()
            writer.result(3)
        rows = store.sessions.list_sessions(SessionQuery(), SessionWindow(limit=5), compact_rows=True)
        other = next(row.as_list_row() for row in rows if row.session.session_id == 'other')
        assert other['input_tokens'] == 19 and other['title'] == 'new'
    finally:
        store.close()
