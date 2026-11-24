"""Protect output before spill/truncation, including fragmented synthetic values."""
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent import secure_input_broker as broker


@pytest.fixture(autouse=True)
def clean_broker():
    broker.clear_all()
    yield
    broker.clear_all()


def test_foreground_spill_is_masked_before_capture_even_across_chunks(monkeypatch):
    from tools.environments.base_output import _new_output_collector, _finalize_wait_result
    from tools import tool_output_limits
    value = 'synthetic-stream-sensitive-suffix'
    metadata = broker.register_secret(value, purpose='fixture')
    proc = SimpleNamespace(_hermes_secure_output_deadline=metadata['expires_at'])
    monkeypatch.setattr(tool_output_limits, 'get_max_bytes', lambda: 90)
    collector = _new_output_collector(proc, True)
    collector.append('p' * 120 + value[:12])
    collector.append(value[12:] + 'z' * 120)
    result = _finalize_wait_result(collector, collector.render(), 0)
    spill = Path(result['full_output_path']).read_text()
    assert value not in spill
    assert '[REDACTED SECURE INPUT]' in spill
    assert value[:12] not in result['output']
    assert value[12:] not in result['output']


def test_expiration_discards_partial_capture_instead_of_releasing_it(monkeypatch):
    from agent.secure_input_output import SecureOutputFilter
    from tools.environments.base_output import _BoundedOutputCollector
    clock = [1000.0]
    monkeypatch.setattr(broker, '_now', lambda: clock[0])
    value = 'synthetic-expired-partial-sensitive-suffix'
    metadata = broker.register_secret(value, purpose='fixture', ttl_seconds=10)
    output_filter = SecureOutputFilter(metadata['expires_at'])
    collector = _BoundedOutputCollector(300, output_filter=output_filter)
    collector.append(value[:15])
    clock[0] += 11
    collector.append(value[15:])
    collector.finish()
    assert value not in collector.render()
    assert 'synthetic-expired' not in collector.render()
    assert 'sensitive-suffix' not in collector.render()
    assert 'TTL expired' in collector.render()


def test_expiration_racing_broker_lookup_cannot_release_unmasked_chunk(monkeypatch):
    from agent.secure_input_output import SecureOutputFilter
    clock = [1000.0]
    monkeypatch.setattr(broker, '_now', lambda: clock[0])
    value = 'synthetic-expiry-race-sensitive-suffix'
    metadata = broker.register_secret(value, purpose='fixture', ttl_seconds=10)
    output_filter = SecureOutputFilter(metadata['expires_at'])
    original = broker._purge_expired_locked

    def expire_during_lookup(now=None):
        clock[0] = metadata['expires_at'] + 1
        original(now)

    monkeypatch.setattr(broker, '_purge_expired_locked', expire_during_lookup)
    assert value not in output_filter.feed(value, final=True)


def test_quiet_process_does_not_keep_pending_plaintext_past_ttl():
    from agent.secure_input_output import SecureOutputFilter
    from tests.waiting import wait_until
    value = 'synthetic-quiet-sensitive-suffix'
    metadata = broker.register_secret(value, purpose='fixture', ttl_seconds=1)
    output_filter = SecureOutputFilter(metadata['expires_at'])
    assert output_filter.feed(value[:12]) == ''
    try:
        wait_until(lambda: not output_filter._pending and not broker._records
                   and not broker._redaction_values, timeout=3, what='automatic TTL eviction without output or reads')
    finally:
        output_filter.feed('', final=True)


@pytest.mark.parametrize(('short', 'value'), [
    ('synthetic', 'synthetic-sensitive'), ('aba', 'abababa'), ('秘密', '秘密🚀'), ('x', 'x'),
])
def test_stream_redaction_is_invariant_at_every_chunk_boundary(short, value):
    from agent.secure_input_output import SecureOutputFilter
    broker.register_secret(short, purpose='fixture')
    metadata = broker.register_secret(value, purpose='fixture')
    data = '<<' + value + '>>'
    expected = broker.redact_known_secrets(data)
    for boundary in range(len(data) + 1):
        output_filter = SecureOutputFilter(metadata['expires_at'])
        actual = (output_filter.feed(data[:boundary]) + output_filter.feed(data[boundary:])
                  + output_filter.feed('', final=True))
        assert actual == expected, boundary


def test_terminal_dispatch_masks_real_pipe_before_spill(tmp_path, monkeypatch):
    import importlib
    importlib.import_module('tools.terminal_tool')
    import json
    import shlex
    import sys
    from tools.registry import registry
    from tools.environments import base
    from tools import tool_output_limits

    monkeypatch.setenv('TERMINAL_ENV', 'local')
    monkeypatch.setattr(tool_output_limits, 'get_max_bytes', lambda: 90)
    captured = []
    original = base._new_output_collector

    def observe(proc, bounded, **kwargs):  # v0.23.3 passes control_marker= (cwd fallback line)
        collector = original(proc, bounded, **kwargs)
        if bounded:
            captured.append(collector)
        return collector

    monkeypatch.setattr(base, '_new_output_collector', observe)
    value = 'synthetic-dispatch-pipe-sensitive-suffix'
    ref = broker.register_secret(value, purpose='fixture')['secret_ref']
    code = "import sys; print('p'*120 + sys.stdin.read() + 'z'*120)"
    command = shlex.quote(sys.executable) + ' -c ' + shlex.quote(code)
    result = json.loads(registry.dispatch('terminal', {'command': command, 'workdir': str(tmp_path),
                                                       'stdin_secret_ref': ref}))
    assert result['exit_code'] == 0
    assert captured
    spill = captured[-1]._spill_path.read_text()
    assert value not in spill
    assert '[REDACTED SECURE INPUT]' in spill
