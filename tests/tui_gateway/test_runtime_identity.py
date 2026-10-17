"""Runtime identity contract for TUI gateway machine protocol fields."""

import hermes_cli
from hermes_cli.version_info import VersionInfo
from tui_gateway import server


def test_session_info_advertises_display_version_and_release_date(monkeypatch):
    monkeypatch.setattr(
        "hermes_cli.version_info.get_version_info",
        lambda: VersionInfo("1.2.3", "1.2.3+4.gabcdef0", 4, "abcdef0", "main", "git"),
    )
    monkeypatch.setattr(hermes_cli, "__release_date__", "2026.9.23")

    info = server._session_info(None, {})

    assert info["version"] == "1.2.3+4"
    assert info["release_date"] == "2026.9.23"


def test_session_info_display_version_falls_back_to_base_past_a_tag(monkeypatch):
    # distance=None is a checkout sitting exactly on the release tag: no "+0" suffix.
    monkeypatch.setattr(
        "hermes_cli.version_info.get_version_info",
        lambda: VersionInfo("1.2.3", "1.2.3", None, "abcdef0", "main", "git"),
    )

    info = server._session_info(None, {})

    assert info["version"] == "1.2.3"
