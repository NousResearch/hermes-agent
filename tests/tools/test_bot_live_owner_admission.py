"""Canonical-owner admission after the authority migration (regression for #118635)."""
import asyncio

import pytest
import pytest_asyncio

from hermes_state_runtime import RuntimeStoreError, list_session_admissions
from tools import bot_live_delivery as mailbox

pytest_plugins = ['tests.gateway.test_session_bot_retry']


@pytest_asyncio.fixture
async def transport(bot, monkeypatch):
    """Keep the real tool and authority admission path; replace only the websocket hop."""
    from gateway.session_bot import deliver

    loop = asyncio.get_running_loop()
    monkeypatch.setattr(mailbox, 'authority_delivery', lambda home, params:
                        asyncio.run_coroutine_threadsafe(deliver(bot.connection, params), loop).result())


def _owner(bot, session_id):
    return dict(profile_home=str(bot.home.resolve()), session_id=session_id,
                lease_id=bot.authority.instance_id, live_session_id=session_id)


@pytest.mark.asyncio
async def test_live_noncanonical_lease_cannot_admit_but_bot_chat_can(bot, transport):
    from hermes_cli.active_sessions import try_acquire_active_session
    from tests.gateway.test_session_bot_retry import _settle_quietly

    bot.authority.db.create_session('other', 'cli')
    lease, refusal = try_acquire_active_session(
        session_id='other', surface='desktop', config={}, registry_home=bot.home,
        metadata={'live_session_id': 'live-other', 'bot_live_delivery_consumer': True})
    assert lease is not None and refusal is None
    key = 'a' * 32
    try:
        noncanonical = dict(_owner(bot, 'other'), lease_id=lease.lease_id, live_session_id='live-other')
        with pytest.raises(RuntimeStoreError, match='admission_conflict'):
            await asyncio.to_thread(mailbox.deliver_to_live_owner, bot.home, noncanonical,
                                    'dead drop', delivery_id=key)
        assert mailbox.read_delivery_result(bot.home, key) is None
        assert list_session_admissions(bot.authority.db, session_id='other', pending_only=False) == []
        assert list_session_admissions(bot.authority.db, session_id=bot.chat, pending_only=False) == []

        admitted = await asyncio.to_thread(mailbox.deliver_to_live_owner, bot.home, _owner(bot, bot.chat),
                                           'claimable', delivery_id=key)
        await _settle_quietly(bot)
        receipt = mailbox.read_delivery_result(bot.home, key)
        assert receipt['admission_id'] == admitted['admission_id']
        assert (receipt['session_id'], receipt['status'], receipt['reply']) == (bot.chat, 'settled', 'pong')
    finally:
        lease.release()


@pytest.mark.asyncio
async def test_existing_delivery_survives_canonical_target_handoff(bot, transport):
    from gateway.session_local import create_local_session
    from gateway.session_local_title import title_new_session
    from tests.gateway.test_session_bot_retry import _settle_quietly

    owner, key = _owner(bot, bot.chat), 'b' * 32
    first = await asyncio.to_thread(mailbox.deliver_to_live_owner, bot.home, owner, 'hello', delivery_id=key)
    await _settle_quietly(bot)
    bot.authority.db.set_session_title(bot.chat, 'Previous Bot Chat')
    replacement = create_local_session(bot.authority, bot.connection.actor, {
        'request_id': 'replacement', 'source': 'gui', 'cwd': str(bot.home),
        'model': 'fixture', 'toolsets': []})
    title_new_session(bot.authority, replacement, 'Bot Chat')
    assert replacement.session_id != bot.chat

    again = await asyncio.to_thread(mailbox.deliver_to_live_owner, bot.home, owner, 'hello', delivery_id=key)
    assert again['admission_id'] == first['admission_id']
    assert (again['session_id'], again['status'], again['reply']) == (bot.chat, 'settled', 'pong')
    with pytest.raises(RuntimeStoreError, match='admission_conflict'):
        await asyncio.to_thread(mailbox.deliver_to_live_owner, bot.home, owner, 'hello', delivery_id='c' * 32)
    assert mailbox.read_delivery_result(bot.home, 'c' * 32) is None
    assert len(list_session_admissions(bot.authority.db, session_id=bot.chat, pending_only=False)) == 1
    assert list_session_admissions(bot.authority.db, session_id=replacement.session_id, pending_only=False) == []
