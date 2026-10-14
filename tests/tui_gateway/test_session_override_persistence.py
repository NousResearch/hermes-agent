"""Explicit session-scoped overrides survive idle reaping (PR #98901).

A `config.set model|reasoning --session` for a non-live session persists a
`session_override` marker into the row's model_config. On resume that marker
must beat the Bot Mode / follow-profile exemptions in
`_stored_session_runtime_overrides`, or the pin is silently dropped.
"""
import json

from tui_gateway import server


def _row(**kw):
    base = {"id": "s1", "title": "chat", "model": "old-model", "model_config": {}}
    base.update(kw)
    return base


def test_row_has_explicit_override():
    assert server._row_has_explicit_override(None) is False
    assert server._row_has_explicit_override(_row()) is False
    marked = _row(model_config=json.dumps({"session_override": True, "model": "m"}))
    assert server._row_has_explicit_override(marked) is True
    # dict form (not just JSON text) also counts
    assert server._row_has_explicit_override(
        _row(model_config={"session_override": True})) is True


def test_explicit_override_beats_room_plumbing_exemption():
    """A Bot Mode plumbing row with an explicit pin still restores the pin."""
    row = _row(model="pinned-model", model_config=json.dumps({
        "room_plumbing": True,
        "session_override": True,
        "model": "pinned-model",
        "provider": "nous",
    }))
    overrides = server._stored_session_runtime_overrides(row)
    assert overrides.get("model_override", {}).get("model") == "pinned-model"
    assert overrides.get("provider_override") == "nous"


def test_explicit_override_beats_follow_profile_exemption():
    """A follow-profile canonical row with an explicit pin still restores the pin."""
    row = _row(title="Bot Chat", model="pinned-model", model_config=json.dumps({
        "follow_profile_config": True,
        "session_override": True,
        "model": "pinned-model",
        "provider": "nous",
    }))
    overrides = server._stored_session_runtime_overrides(row)
    assert overrides.get("model_override", {}).get("model") == "pinned-model"


def test_no_marker_no_override_for_exempt_rows():
    """Without the marker the exemptions still apply (no behavior change)."""
    row = _row(model_config=json.dumps({"room_plumbing": True, "model": "m"}))
    assert server._stored_session_runtime_overrides(row) == {}
