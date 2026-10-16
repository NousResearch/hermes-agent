"""Regression: ``_clear_setup_sessions`` resolved Paths without importing Path.

The onboarding-reset helper called ``Path(...)`` while the module only imported
``threading``, so the first setup reset on the TUI/Desktop surface died with
``NameError: name 'Path' is not defined`` instead of clearing the staged
sessions.
"""


def test_clear_setup_sessions_runs_on_empty_profile(tmp_path, monkeypatch):
    """Resetting an empty profile dir must complete — it resolves the target and
    walks the session store, both of which need ``Path`` imported."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from tui_gateway import methods_onboarding

    methods_onboarding._clear_setup_sessions(str(tmp_path))
