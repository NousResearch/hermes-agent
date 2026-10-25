"""Task-local provenance for memory staging; never shared with skill writes."""
from contextlib import contextmanager
from contextvars import ContextVar

_source = ContextVar('memory_review_source', default=None)


def source_context(agent, tool_call_id=None):
    from gateway.session_context import get_session_env
    def identity(name):
        value = getattr(agent, '_' + name, None)
        return str(value if value is not None else get_session_env('HERMES_SESSION_' + name.upper(), '') or '')
    return {'session_id': str(getattr(agent, 'session_id', '') or ''),
            'parent_session_id': str(getattr(agent, '_parent_session_id', '') or ''),
            'tool_call_id': str(tool_call_id or ''),
            'message_id': str(get_session_env('HERMES_SESSION_MESSAGE_ID', '') or ''),
            'platform': str(getattr(getattr(agent, 'platform', ''), 'value', getattr(agent, 'platform', '')) or ''),
            'model': str(getattr(agent, 'model', '') or ''),
            'provider': str(getattr(agent, 'provider', '') or ''),
            'user_id': identity('user_id'), 'chat_id': identity('chat_id'), 'thread_id': identity('thread_id')}


@contextmanager
def memory_source(context):
    token = _source.set(context)
    try:
        yield
    finally:
        _source.reset(token)


def current_source():
    return {k: v for k, v in (_source.get() or {}).items() if v}
