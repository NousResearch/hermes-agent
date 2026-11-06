"""Passive wake auto-arm must not probe optional voice dependencies (#86131)."""

import logging
import os
from pathlib import Path
from unittest.mock import Mock

import pytest

from tools import wake_word
from tui_gateway import server


@pytest.mark.parametrize(
    ("enabled", "configured_surface", "surface", "reason"),
    [(False, "auto", "tui", "disabled"),
     (False, "auto", "gui", "disabled"),
     (True, "gui", "tui", "disabled_for_surface"),
     (True, "tui", "gui", "disabled_for_surface")],
)
def test_passive_start_reads_config_before_probing(
    monkeypatch, caplog, enabled, configured_surface, surface, reason
):
    config_path = Path(os.environ["HERMES_HOME"]) / "config.yaml"
    config_path.write_text(
        f"wake_word:\n  enabled: {str(enabled).lower()}\n  surface: {configured_surface}\n"
        "  phrase: from-test-config\n",
        encoding="utf-8",
    )
    config = wake_word.load_wake_word_config()
    assert config["enabled"] is enabled and config["phrase"] == "from-test-config"
    resolve_capture = Mock(return_value="local")
    requirements = Mock(return_value={"available": True})
    persist = Mock()
    monkeypatch.setattr(wake_word, "resolve_capture_mode", resolve_capture)
    monkeypatch.setattr(wake_word, "check_wake_word_requirements", requirements)
    monkeypatch.setattr(server, "_persist_wake_enabled", persist)

    with caplog.at_level(logging.INFO):
        response = server.handle_request({
            "id": "wake-passive-config", "method": "wake.start", "params": {"surface": surface},
        })

    assert response["result"] == {"started": False, "reason": reason}
    assert f"wake.start({surface}): {reason}" in caplog.text
    resolve_capture.assert_not_called()
    requirements.assert_not_called()
    persist.assert_not_called()


def test_enable_gesture_checks_requirements_before_persisting(monkeypatch):
    config_path = Path(os.environ["HERMES_HOME"]) / "config.yaml"
    config_path.write_text("wake_word:\n  enabled: false\n", encoding="utf-8")
    requirements = Mock(return_value={"available": False, "hint": "voice unavailable"})
    persist = Mock()
    monkeypatch.setattr(wake_word, "resolve_capture_mode", lambda *_a, **_k: "local")
    monkeypatch.setattr(wake_word, "check_wake_word_requirements", requirements)
    monkeypatch.setattr(server, "_persist_wake_enabled", persist)

    response = server.handle_request({
        "id": "wake-enable", "method": "wake.start", "params": {"surface": "tui", "persist": True},
    })

    assert response["result"]["reason"] == "unavailable"
    requirements.assert_called_once()
    persist.assert_not_called()
