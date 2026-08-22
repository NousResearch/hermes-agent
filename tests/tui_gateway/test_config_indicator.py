from tui_gateway import server


# ── config.get indicator normalization ───────────────────────────────


def test_config_get_indicator_returns_known_value_verbatim(monkeypatch):
    monkeypatch.setattr(
        server, "_load_cfg", lambda: {"display": {"tui_status_indicator": "emoji"}}
    )
    resp = server.handle_request(
        {"id": "1", "method": "config.get", "params": {"key": "indicator"}}
    )
    assert resp["result"] == {"value": "emoji"}


def test_config_get_indicator_normalizes_casing_and_whitespace(monkeypatch):
    """Hand-edited config.yaml stays consistent with what the TUI shows.

    Frontend's `normalizeIndicatorStyle` lowercases + trims, so config.get
    must do the same — otherwise `/indicator` prints 'EMOJI ' while the
    UI is actually rendering the kaomoji default."""
    monkeypatch.setattr(
        server, "_load_cfg", lambda: {"display": {"tui_status_indicator": " EMOJI "}}
    )
    resp = server.handle_request(
        {"id": "1", "method": "config.get", "params": {"key": "indicator"}}
    )
    assert resp["result"] == {"value": "emoji"}


def test_config_get_indicator_falls_back_to_default_for_unknown(monkeypatch):
    """An unknown value in config.yaml falls back to the same default
    the frontend uses (`_INDICATOR_DEFAULT`)."""
    monkeypatch.setattr(
        server, "_load_cfg", lambda: {"display": {"tui_status_indicator": "rainbow"}}
    )
    resp = server.handle_request(
        {"id": "1", "method": "config.get", "params": {"key": "indicator"}}
    )
    assert resp["result"] == {"value": "kaomoji"}


def test_config_get_indicator_falls_back_when_unset(monkeypatch):
    monkeypatch.setattr(server, "_load_cfg", lambda: {"display": {}})
    resp = server.handle_request(
        {"id": "1", "method": "config.get", "params": {"key": "indicator"}}
    )
    assert resp["result"] == {"value": "kaomoji"}


# ── config.set indicator validation ──────────────────────────────────


def test_config_set_indicator_accepts_known_value(monkeypatch):
    written: dict = {}
    monkeypatch.setattr(
        server,
        "_write_config_key",
        lambda k, v: written.update({k: v}),
    )
    resp = server.handle_request(
        {
            "id": "1",
            "method": "config.set",
            "params": {"key": "indicator", "value": "EMOJI"},
        }
    )
    assert resp["result"] == {"key": "indicator", "value": "emoji"}
    assert written == {"display.tui_status_indicator": "emoji"}


def test_config_set_indicator_falsy_non_string_surfaces_in_error(monkeypatch):
    """`0` / `False` / `[]` are not valid styles, but the error message
    must still tell the user what they sent — `value or ""` would have
    erased them to a blank string."""
    monkeypatch.setattr(server, "_write_config_key", lambda *a, **k: None)

    for bad in (0, False, []):
        resp = server.handle_request(
            {
                "id": "1",
                "method": "config.set",
                "params": {"key": "indicator", "value": bad},
            }
        )
        assert "error" in resp
        msg = resp["error"]["message"]
        assert "unknown indicator" in msg
        # The exact repr varies; `0`/`False` stringify with content,
        # `[]` becomes an empty list — what matters is the diagnostic
        # is no longer just `unknown indicator: ` with nothing after.
        assert msg.split("; ")[0] != "unknown indicator: ''"


def test_config_set_indicator_none_keeps_blank_repr(monkeypatch):
    """`None` is the genuine 'no value' case — empty raw is acceptable."""
    monkeypatch.setattr(server, "_write_config_key", lambda *a, **k: None)
    resp = server.handle_request(
        {
            "id": "1",
            "method": "config.set",
            "params": {"key": "indicator", "value": None},
        }
    )
    assert "error" in resp
    assert "unknown indicator: ''" in resp["error"]["message"]
