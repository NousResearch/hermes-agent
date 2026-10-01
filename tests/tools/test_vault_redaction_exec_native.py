"""Exercise browser_exec's real output boundary with a local Python test CLI.

No browser, vault entry, login, network, or external service is used.
"""
import json
import sys

import pytest

from agent import redact
from hermes_constants import reset_hermes_home_override, set_hermes_home_override
from tools import browser_use_cli


@pytest.mark.parametrize('canary', ['x', 'xy', 'xyz', '\u0627', 'fixture-opaque-long-98'])
def test_native_process_output_keeps_strict_vault_redaction(tmp_path, monkeypatch, canary):
    token = set_hermes_home_override(str(tmp_path / 'boundary-profile'))
    payload = 'prefix' + canary + 'suffix'
    code = 'import sys; sys.stdin.read(); sys.stdout.write(' + repr(payload) + '); sys.stderr.write(' + repr(payload) + ')'
    monkeypatch.setattr(browser_use_cli, '_find_cli', lambda: [sys.executable, '-c', code])
    monkeypatch.setattr(browser_use_cli, '_route_backend', lambda *args: None)
    monkeypatch.setattr(browser_use_cli, '_attach_vault_supervisor', lambda *args: None)
    monkeypatch.setattr(browser_use_cli, '_workspace_dir', lambda *args: None)
    monkeypatch.setattr(browser_use_cli, '_read_browser_cfg', lambda: {})
    try:
        redact.register_vault_redaction_value(canary)
        result = json.loads(browser_use_cli.browser_exec('pass', timeout_s=20))
        assert result['success'] is True
        assert result['exit_code'] == 0
        expected = redact.redact_registered_vault_values(payload)
        assert result['output'] == expected
        assert result['stderr'] == expected
        assert payload not in result['output']
        assert '«redacted-vault-secret»' in result['output']
    finally:
        redact.clear_vault_redaction_values()
        reset_hermes_home_override(token)
