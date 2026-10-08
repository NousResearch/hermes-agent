"""The native Camofox vault path must never attach or fall back to CDP."""
import json
import pytest
from tools import browser_vault_tool as vault
from tools import browser_camofox as camo


def test_camofox_vault_transport_success_and_errors(monkeypatch, caplog):
    monkeypatch.setattr(camo, 'is_camofox_mode', lambda: True)
    monkeypatch.setattr(camo, 'get_camofox_url', lambda: 'http://127.0.0.1:9377')
    monkeypatch.setattr(camo, '_ensure_tab', lambda task: {'tab_id': 'test-tab', 'user_id': 'test-identity'})
    monkeypatch.setattr(vault, '_ensure_supervisor', lambda _: pytest.fail('Camofox must never attach CDP'))
    calls = []
    def post(path, body, *, allow_redirects):
        assert allow_redirects is False
        calls.append((path, body))
        return {'result': {'filled': 1}}
    monkeypatch.setattr(camo, '_post', post)
    assert vault._eval_js_secret('test', 'synthetic-expression') == {'success': True, 'result': {'filled': 1}}
    assert calls[0][0] == '/tabs/test-tab/evaluate'
    assert calls[0][1]['userId'] == 'test-identity'
    assert 'synthetic-expression' in calls[0][1]['expression']
    assert vault._focus_bound_origin('test', 'https://example.com', 'login') is None
    assert vault._eval_js('test', 'location.origin')['success']
    def fail(*_, **__): raise RuntimeError('synthetic-secret-that-must-not-escape')
    monkeypatch.setattr(camo, '_post', fail)
    result = vault._eval_js_secret('test', 'synthetic-secret-that-must-not-escape')
    assert result['success'] is False
    assert 'synthetic-secret' not in json.dumps(result) + caplog.text
