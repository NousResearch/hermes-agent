"""The sessionless roster resolves tmux only inside each requested profile."""


def test_roster_tmux_lookup_is_profile_scoped(two_profiles, monkeypatch):
    # Importing this fixture module registers the fixture via pytest_plugins below.
    from hermes_constants import get_hermes_home
    from tools import session_tmux

    _, roster = two_profiles
    seen = []

    def owners():
        home = get_hermes_home()
        seen.append(home.name)
        # Include a foreign ID: DB ownership, not discovery alone, is authoritative.
        return [('bot2-conv', 2), ('default-conv', 5)] if home.name == 'bot2' else []

    monkeypatch.setattr(session_tmux, 'running_pane_owners', owners, raising=False)
    assert roster() == {}
    assert roster('bot2') == {'bot2-conv': 2}
    assert 'bot2' in seen


pytest_plugins = ['tests.tui_gateway.test_subagent_roster_profiles']
