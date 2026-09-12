"""SDK integers are already converted by the real adapter, not owner coercion."""
from datetime import datetime, timezone
import pytest
from telegram import Chat, Message, User
from gateway.config import PlatformConfig
from gateway.platforms.event import MessageType
from plugins.platforms.telegram.adapter import TelegramAdapter
from tools.approval_delegation import owner_for_source
from tests.tools.test_background_approval_routing import rig


@pytest.mark.parametrize('chat_id,user_id', [(101, 202), (-100123456789, 202)])
def test_sdk_integer_identity_converts_at_adapter_boundary(rig, chat_id, user_id):
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token='fixture'))
    message = Message(message_id=1, date=datetime(2026, 1, 1, tzinfo=timezone.utc),
                      chat=Chat(id=chat_id, type='private'),
                      from_user=User(id=user_id, first_name='Fixture', is_bot=False), text='/approve')
    event = adapter._build_message_event(message, MessageType.TEXT)
    assert type(event.source.user_id) is str
    assert type(event.source.chat_id) is str
    owner = owner_for_source(event.source, rig.key)
    assert owner.actor == str(user_id)
    assert owner.chat == str(chat_id)
    # Raw integers/booleans do not have permission to bypass that boundary.
    event.source.user_id = user_id
    assert owner_for_source(event.source, rig.key) is None
