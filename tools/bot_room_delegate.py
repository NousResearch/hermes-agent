"""Service-gated Telegram DM delegation to one locally configured Bot Mode room."""
import json

import hermes_yaml as yaml

from hermes_constants import get_hermes_home
from tools.registry import registry


def check_requirements():
    """Opt-in only. Identity is checked on every invocation, never in the cached gate."""
    try:
        config = yaml.safe_load((get_hermes_home() / 'config.yaml').read_text()) or {}
        return config.get('telegram_room_relay', {}).get('enabled') is True
    except (OSError, ValueError, TypeError, AttributeError, yaml.YAMLError):
        return False


def handle(args):
    if (not isinstance(args, dict) or set(args) != {'recipient', 'message'}):
        return json.dumps({'status': 'denied', 'reason': 'invalid_arguments'})
    from gateway.run import _gateway_runner_ref
    from gateway.hosted_room_telegram_relay import delegate
    return json.dumps(delegate(_gateway_runner_ref(), recipient=args['recipient'], message=args['message']))


registry.register(
    name='bot_room_delegate', toolset='bot_room_relay', handler=handle,
    check_fn=check_requirements, emoji='🤝',
    schema={
        'name': 'bot_room_delegate',
        'description': 'Delegate an explicit task to the configured Bot Mode room from an authorized '
                       'Telegram private DM. Returns only this discussion’s committed replies. '
                       'Timeout/incomplete means work may still be pending; do not claim completion '
                       'or silently resubmit. If a request_id is returned, use bot_room_result. '
                       'If no request_id is present, authorization did not complete and no task was submitted. '
                       'A denied result may include a coarse check label for the configured sender, never identifiers. '
                       'Room member sessions retain their normal room history.',
        'parameters': {
            'type': 'object', 'additionalProperties': False,
            'properties': {
                'recipient': {'type': 'string', 'enum': ['server']},
                'message': {'type': 'string', 'minLength': 1, 'maxLength': 8000},
            },
            'required': ['recipient', 'message'],
        },
    },
)


def handle_result(args):
    if not isinstance(args, dict) or set(args) != {'request_id'}:
        return json.dumps({'status': 'denied', 'reason': 'invalid_arguments'})
    from gateway.run import _gateway_runner_ref
    from gateway.hosted_room_telegram_relay import room_result
    return json.dumps(room_result(_gateway_runner_ref(), request_id=args['request_id']))


registry.register(
    name='bot_room_result', toolset='bot_room_relay', handler=handle_result,
    check_fn=check_requirements, emoji='🤝',
    schema={
        'name': 'bot_room_result',
        'description': 'Read committed results for a retained bot_room_delegate request_id from the '
                       'same authorized Telegram DM. On timeout retain request_id and poll again; '
                       'never resubmit silently. Incomplete means pending or unfinished work.',
        'parameters': {
            'type': 'object', 'additionalProperties': False,
            'properties': {'request_id': {'type': 'string', 'pattern': '^[0-9a-f]{64}$'}},
            'required': ['request_id'],
        },
    },
)
