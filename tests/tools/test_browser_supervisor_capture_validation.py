import pytest
from tests.tools.test_browser_supervisor_capture_dispatch import pair  # noqa: F401

@pytest.mark.parametrize('timeout', [0, -1, float('inf'), float('nan'), 1e300, True, '5'])
def test_invalid_timeout(pair, timeout):
    server, _, _, cap = pair
    before = list(server.methods_on(1))
    with pytest.raises(ValueError):
        cap.call('Probe.invalid', timeout=timeout)
    assert server.methods_on(1) == before


def test_async_validator_refuses(pair):
    server, _, _, cap = pair
    before = list(server.methods_on(1))
    async def check():
        return None
    with pytest.raises(TypeError):
        cap.call('Probe.async', before_send=check)
    assert server.methods_on(1) == before
