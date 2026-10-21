"""A legacy ``paste.collapse`` on the shared owner writes under the ticket's profile home."""
from pathlib import Path
from types import SimpleNamespace

import tui_gateway.server as srv
from gateway.runtime_bootstrap import _PURPOSE_CAPABILITIES
from tui_gateway.ws_legacy_fallback import dispatch_legacy


def test_secondary_profile_paste_lands_in_its_own_home(tmp_path, monkeypatch):
    launch, secondary = tmp_path / 'launch', tmp_path / 'profiles' / 'coder'
    launch.mkdir()
    secondary.mkdir(parents=True)
    monkeypatch.setattr(srv, '_hermes_home', launch)
    actor = SimpleNamespace(profile_id=str(secondary), capabilities=_PURPOSE_CAPABILITIES['interactive'])
    reply = dispatch_legacy(srv, {'jsonrpc': '2.0', 'id': 1, 'method': 'paste.collapse',
                                  'params': {'text': 'secret\npaste'}}, None, actor)
    path = Path(reply['result']['path'])
    assert path.parent == secondary / 'pastes' and path.read_text() == 'secret\npaste'
    assert not (launch / 'pastes').exists()
