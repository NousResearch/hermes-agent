"""A relay reply is an immutable delivery receipt, not a last-writer-wins file."""
import json

from tools.bot_relay import write_reply


def test_relay_reply_retry_cannot_replace_a_settled_delivery(tmp_path):
    key = 'd' * 32
    path = write_reply(tmp_path, key, reply='original')
    before = path.read_bytes()
    assert write_reply(tmp_path, key, reply='original') == path
    assert path.read_bytes() == before
    # A replayed envelope's second outcome (or a bookkeeping timeout) never displaces the first
    # settled reply — the waiter may already have read it (tests/tools/test_bot_relay.py pins the
    # same rule from the drain side).
    assert write_reply(tmp_path, key, reply='replacement', error='late', reason='delivery_timeout') == path
    assert path.read_bytes() == before
    assert json.loads(path.read_text())['reply'] == 'original'
