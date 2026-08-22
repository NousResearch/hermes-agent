"""Regression tests for delegate_task tool-schema description guidance."""


def test_top_level_description_discourages_polling_live_transcripts(monkeypatch):
    """The schema description must match the background dispatch payload guidance."""
    import tools.delegate_tool as dt

    monkeypatch.setattr(dt, "_get_max_concurrent_children", lambda: 3)
    monkeypatch.setattr(dt, "_get_max_spawn_depth", lambda: 1)
    monkeypatch.setattr(dt, "_get_orchestrator_enabled", lambda: True)

    description = dt._build_top_level_description()

    assert "Do NOT wait or poll" in description
    assert "live transcripts are append-only logs, not completion signals" in description
    assert "tail -f" not in description
    assert "Read or `tail -f`" not in description
