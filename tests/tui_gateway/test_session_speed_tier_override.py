"""The exact composer tier wins over the legacy Fast checkbox."""
import pytest
from tui_gateway.server import _create_overrides

@pytest.mark.parametrize('params, expected', [
    ({'fast': False, 'service_tier': 'ultrafast'}, 'ultrafast'),
    ({'fast': True, 'service_tier': 'normal'}, ''),
    ({'fast': True}, 'priority'),
])
def test_session_create_preserves_exact_speed(params, expected):
    assert _create_overrides(params)[2] == expected

def test_session_create_rejects_unknown_speed():
    with pytest.raises(ValueError):
        _create_overrides({'service_tier': 'turbo'})
