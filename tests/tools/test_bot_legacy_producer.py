"""A stale owner advertisement must never create another execution queue."""
import pytest
from tools.bot_live_delivery import deliver_to_live_owner


def test_legacy_owner_requires_a_live_authority_before_admission(tmp_path):
    owner = dict(profile_home=str(tmp_path), session_id='legacy', lease_id='dead', live_session_id='old-ui')
    with pytest.raises(ValueError, match='authority'):
        deliver_to_live_owner(tmp_path, owner, 'message', delivery_id='e' * 32)
    assert not (tmp_path / 'runtime' / 'bot_live_delivery' / ('e' * 32 + '.json')).exists()
