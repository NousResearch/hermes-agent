"""Runtime identity contract for TUI gateway machine protocol fields."""

from types import SimpleNamespace

from tui_gateway import server


def test_session_info_advertises_base_version_and_release_date(monkeypatch):
    monkeypatch.setattr(
        "hermes_cli.version_info.get_version_info",
        lambda: SimpleNamespace(base_version="1.2.3", derived_version="1.2.3+4.gabcdef0"),
    )
    monkeypatch.setattr("hermes_cli.version_info.get_release_date", lambda: "2026.9.23")

    info = server._session_info(None, {})

    assert info["version"] == "1.2.3"
    assert info["release_date"] == "2026.9.23"
