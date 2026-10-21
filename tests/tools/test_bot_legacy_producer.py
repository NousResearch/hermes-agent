"""A stale owner advertisement must never create another execution queue."""
import pytest
from tools.bot_live_delivery import deliver_to_live_owner


def test_legacy_owner_requires_a_live_authority_before_admission(tmp_path):
    owner = dict(profile_home=str(tmp_path), session_id='legacy', lease_id='dead', live_session_id='old-ui')
    with pytest.raises(ValueError, match='authority'):
        deliver_to_live_owner(tmp_path, owner, 'message', delivery_id='e' * 32)
    assert not (tmp_path / 'runtime' / 'bot_live_delivery' / ('e' * 32 + '.json')).exists()


def test_legacy_poller_cannot_claim_an_unmigrated_input(tmp_path):
    from tools.bot_live_delivery import _locked, _write, claim_pending_delivery
    owner = dict(profile_home=str(tmp_path), session_id='legacy', lease_id='dead', live_session_id='old-ui')
    with _locked(tmp_path) as root:
        _write(root / ('f' * 32 + '.json'), dict(owner=owner, delivery_id='f' * 32,
            message='old queued input', status='queued', created_at=1))
    assert claim_pending_delivery(tmp_path, owner) is None
