"""Bounded, read-only previews of retained child transcript messages."""
import json

from tools.delegation_redaction import redact_child_text


def preview_text(text, limit):
    """Keep both context and outcome, counting the omission marker in the cap."""
    if len(text) <= limit:
        return text
    # Reserve enough room even if the omitted count crosses a decimal boundary.
    reserve = len(f'\n[{len(text)} chars omitted]\n')
    if limit <= reserve:
        return text[:max(0, limit - 1)] + '…' if limit else ''
    keep = limit - reserve
    head = min(400, keep // 2)
    tail = keep - head
    return text[:head] + f'\n[{len(text) - keep} chars omitted]\n' + text[-tail:]


def _call_preview(call):
    function = call.get('function')
    if not isinstance(function, dict):
        return None
    name = function.get('name')
    if not isinstance(name, str):
        return None
    name = redact_child_text(name, force=False)
    result = {'name': preview_text(name, 80)}
    truncated = len(name) > 80
    arguments = function.get('arguments')
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except (ValueError, RecursionError):
            arguments = None
    if isinstance(arguments, dict):
        for source, target in (('path', 'target'), ('file_path', 'target'), ('command', 'command_preview')):
            value = arguments.get(source)
            if isinstance(value, str) and target not in result:
                value = redact_child_text(value, force=False)
                result[target] = preview_text(value, 200)
                truncated = truncated or len(value) > 200
    return result, truncated


def message_metadata(message):
    """Whitelist useful call fields; never copy full arguments or reasoning."""
    metadata = {}
    truncated = False
    if message['role'] == 'tool' and isinstance(message.get('tool_name'), str):
        name = redact_child_text(message['tool_name'], force=False)
        metadata['tool_name'] = preview_text(name, 80)
        truncated = len(name) > 80
    calls = message.get('tool_calls')
    if message['role'] == 'assistant' and isinstance(calls, list):
        previews = []
        for call in calls:
            preview = _call_preview(call) if isinstance(call, dict) else None
            if preview is None:
                truncated = True
                continue
            preview, shortened = preview
            truncated = truncated or shortened
            candidate = {'tool_calls': [*previews, preview]}
            if len(json.dumps(candidate, ensure_ascii=False)) > 1000:
                truncated = True
                break
            previews.append(preview)
        if previews:
            metadata['tool_calls'] = previews
    return metadata, truncated
