"""A relay reply is an immutable delivery receipt, not a last-writer-wins file."""
import json
import pytest
from tools.bot_relay import write_reply


def test_relay_reply_retry_cannot_replace_a_settled_delivery(tmp_path):
    key = 'd' * 32
    path = write_reply(tmp_path, key, reply='original')
    before = path.read_bytes()
    assert write_reply(tmp_path, key, reply='original') == path
    assert path.read_bytes() == before
    with pytest.raises(ValueError, match='different'):
        write_reply(tmp_path, key, reply='replacement')
    assert json.loads(path.read_text())['reply'] == 'original'
