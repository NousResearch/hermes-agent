import json
from unittest.mock import Mock

from tools.employee_messaging import handle


def test_empty_destination_never_reaches_native_home_fallback(monkeypatch):
    sender = Mock(return_value='{"success":true}')
    monkeypatch.setattr('tools.send_message_tool._handle_send', sender)
    for target in ('telegram', 'telegram:', 'telegram: ', ':123', ' :123'):
        result = json.loads(handle({'target': target, 'message': 'private report'}))
        assert not result['success']
    sender.assert_not_called()
    args = {'target': 'telegram:123:45', 'message': 'private report'}
    assert json.loads(handle(args))['success']
    sender.assert_called_once_with(args)
