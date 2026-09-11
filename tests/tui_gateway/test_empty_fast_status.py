from types import SimpleNamespace
import pytest
from tui_gateway import methods_config_set as config


@pytest.mark.parametrize('tier,expected', [('', 'normal'), (None, 'normal'), ('priority', 'fast'), ('normal', 'normal'), ('auto', 'auto'), ('cold', 'cold')])
@pytest.mark.parametrize('built', [True, False])
def test_fast_status_preserves_tier_semantics_without_mutation(monkeypatch, tier, expected, built):
    session = {'agent': SimpleNamespace(service_tier=tier)} if built else {'create_service_tier_override': tier}
    replies = []
    monkeypatch.setattr(config, '_load_service_tier', lambda: tier, raising=False)
    monkeypatch.setattr(config, '_kv', lambda rid, key, value: replies.append((rid, key, value)))
    def forbidden(*args, **kwargs):
        pytest.fail('Status query must not write configuration')
    monkeypatch.setattr(config, '_write_config_key', forbidden, raising=False)
    config._set_fast(1, {'session_id': 'test'}, 'fast', 'status', session)
    assert replies == [(1, 'fast', expected)]
    if built:
        assert session['agent'].service_tier == tier
    else:
        assert session['create_service_tier_override'] == tier
