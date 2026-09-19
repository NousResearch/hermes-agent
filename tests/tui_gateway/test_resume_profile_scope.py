# session.resume must resolve stored runtime overrides under the launch-profile scope.
#
# _stored_session_runtime_overrides heals a stale persisted provider via the
# configured provider chain (canonical_custom_identity / is_routable_provider),
# which reads the CURRENT profile config. The eager resume path already wraps
# that call in _profile_build_scope; the deferred and cold paths must do the
# same or the heal resolves against the ambient launch environment and
# mis-routes the resumed chat. See #115607.

from __future__ import annotations

import contextlib
import types

from tui_gateway import server


def _install_scope_probe(monkeypatch):
    # Stub _profile_build_scope as a recording context manager; return call log.
    calls = {'scope_homes': [], 'overrides_in_scope': []}
    active = {'home': None}

    @contextlib.contextmanager
    def _scope(profile_home):
        calls['scope_homes'].append(profile_home)
        prev = active['home']
        active['home'] = profile_home
        try:
            yield
        finally:
            active['home'] = prev

    def _overrides(found):
        calls['overrides_in_scope'].append(active['home'])
        return {}

    monkeypatch.setattr(server, '_profile_build_scope', _scope)
    monkeypatch.setattr(server, '_stored_session_runtime_overrides', _overrides)
    monkeypatch.setattr(server, '_schedule_resume_hydration', lambda *a, **k: None)
    monkeypatch.setattr(server, '_schedule_session_cap_enforcement', lambda *a, **k: None)
    monkeypatch.setattr(server, '_schedule_agent_build', lambda *a, **k: None)
    monkeypatch.setattr(server, '_resume_response', lambda *a, **k: {'ok': True})
    return calls


def _deferred_ctx(home):
    return types.SimpleNamespace(
        found={'id': 's1'},
        profile_home=str(home),
        target='s1',
        db=None,
        owns_db=False,
        omit_messages=False,
        mint=lambda prompts=True: ('sid-d', 'tui', '/tmp'),
        record=lambda source, cwd, history, overrides=None, **kw: {},
        claim=lambda sid, record: None,
        info=lambda cwd, overrides: {},
    )


def _cold_ctx(home):
    return types.SimpleNamespace(
        found={'id': 's1'},
        profile_home=str(home),
        target='s1',
        rid='1',
        mint=lambda prompts=True: ('sid-c', 'tui', '/tmp'),
        restore=lambda: ([], [], []),
        display_prefix=lambda: [],
        record=lambda source, cwd, history, overrides=None, **kw: {},
        claim=lambda sid, record: None,
        info=lambda cwd, overrides: {},
    )


def test_deferred_resume_resolves_overrides_under_profile_scope(monkeypatch, tmp_path):
    home = tmp_path / 'work'
    home.mkdir()
    calls = _install_scope_probe(monkeypatch)
    ctx = _deferred_ctx(home)

    server._resume_deferred(ctx)

    assert calls['scope_homes'] == [str(home)]
    assert calls['overrides_in_scope'] == [str(home)]


def test_cold_resume_resolves_overrides_under_profile_scope(monkeypatch, tmp_path):
    home = tmp_path / 'work'
    home.mkdir()
    calls = _install_scope_probe(monkeypatch)
    ctx = _cold_ctx(home)

    server._resume_cold(ctx)

    assert calls['scope_homes'] == [str(home)]
    assert calls['overrides_in_scope'] == [str(home)]
