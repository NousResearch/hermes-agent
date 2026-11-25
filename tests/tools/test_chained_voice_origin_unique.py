import pytest

from gateway.platforms.event import MessageEvent, MessageType
from gateway.session_context import clear_session_vars, set_session_vars
from tools.async_delegation import _capture_routing_origin


@pytest.mark.parametrize('internal,expected', [(True, 'voice'), (False, 'text')])
def test_dispatch_trusts_only_internal_presentation_origin(internal, expected):
    event = MessageEvent(text='completion', message_type=MessageType.TEXT, internal=internal,
                         metadata={'presentation_origin': 'voice'})
    tokens = set_session_vars(turn_event=event)
    try:
        assert _capture_routing_origin()['origin_message_type'] == expected
    finally:
        clear_session_vars(tokens)
