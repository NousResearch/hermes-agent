"""Runtime identity contract for TUI gateway machine protocol fields."""

from types import SimpleNamespace

import hermes_cli
from tui_gateway import server


def test_session_info_advertises_mithril_distribution_and_runtime_contracts(monkeypatch):
    monkeypatch.setattr(
        "hermes_cli.version_info.get_version_info",
        lambda: SimpleNamespace(base_version="1.2.3", derived_version="1.2.3+4.gabcdef0"),
    )
    monkeypatch.setattr(hermes_cli, "__release_date__", "2026.9.23")

    info = server._session_info(None, {})

    assert info["version"] == "1.2.3"
    assert info["release_date"] == "2026.9.23"
    assert info["distribution"] == "mithril-agent"
    assert info["action_contract"] == ""


def test_lazy_session_info_keeps_mithril_runtime_identity():
    info = server._fallback_session_info({"cwd": ""})

    assert info["distribution"] == "mithril-agent"
    assert info["action_contract"] == ""
