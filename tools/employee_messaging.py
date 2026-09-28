"""Employee message contract using native directory, authorization and delivery."""
import json


def list_targets():
    from gateway.channel_directory import load_directory
    from gateway.session_context import get_session_env
    targets = []
    for platform, entries in load_directory().get('platforms', {}).items():
        for entry in entries:
            target = f"{platform}:{entry['id']}"
            targets.append({'channel': platform, 'target': target, 'label': entry.get('name') or target})
    platform = get_session_env('HERMES_SESSION_PLATFORM', '')
    chat = get_session_env('HERMES_SESSION_CHAT_ID', '')
    thread = get_session_env('HERMES_SESSION_THREAD_ID', '')
    if platform and chat:
        target = f'{platform}:{chat}' + (f':{thread}' if thread else '')
        targets = [row for row in targets if row['target'] != target]
        targets.insert(0, {'channel': platform, 'target': target, 'label': 'Current conversation'})
    return json.dumps({'success': True, 'count': len(targets), 'targets': targets}, ensure_ascii=False)


def handle(args, **kwargs):
    from tools.send_message_tool import _handle_send
    action = args.get('action') or 'send'
    if action == 'list':
        return list_targets()
    message = str(args.get('message') or '')
    target = str(args.get('target') or '')
    error = None
    if action != 'send':
        error = "send_message supports only 'send' and 'list'"
    elif not message.strip():
        error = 'message is required'
    elif len(message) > 100000:
        error = 'message exceeds the size limit'
    elif 'MEDIA:' in message:
        error = 'send_message supports text only'
    elif ':' not in target or not all(part.strip() for part in target.split(':', 1)):
        error = "send_message requires one exact target; call send_message with action='list' — the current conversation is listed first."
    if error:
        return json.dumps({'success': False, 'error': error})
    return _handle_send(args)
