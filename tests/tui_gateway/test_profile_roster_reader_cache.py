"""The canonical gateway's roster memo (``cached_reader_session_fields``, #139227) keys on the reader
as well as the store: one principal's answer is never served to another, and ``invalidate`` drops it
like the other memos."""
from __future__ import annotations

import pytest

from tui_gateway import profile_roster_cache as cache


@pytest.fixture
def store(tmp_path, monkeypatch):
    (tmp_path / "state.db").write_bytes(b"x" * 16)
    # The store reads as settled at once: the memo is exercised without waiting out the racy window.
    monkeypatch.setattr(cache, "mtime_is_racy", lambda mtime_ns: False)
    cache.invalidate()
    yield tmp_path
    cache.invalidate()


def test_each_reader_gets_its_own_answer_until_the_store_moves(store):
    computed = []

    def read(reader):
        def compute():
            computed.append(reader)
            return {"canonical_session": {"reader": reader}}
        return cache.cached_reader_session_fields(store, reader, compute)["canonical_session"]

    owner, other = ("uid:desk", True), ("uid:other", False)
    assert read(owner) == {"reader": owner}
    assert read(other) == {"reader": other}, "another reader was served the owner's answer"
    assert read(owner) == {"reader": owner}
    assert computed == [owner, other], "an unchanged store was read again"

    (store / "state.db").write_bytes(b"y" * 32)
    assert read(owner) == {"reader": owner}
    assert computed == [owner, other, owner], "a moved store was served from the memo"

    cache.invalidate(store)
    read(other)
    assert computed[-1] == other, "invalidate() kept the reader memo"
