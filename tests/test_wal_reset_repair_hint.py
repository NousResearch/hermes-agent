"""Behavior tests for hermes_state._wal_reset_repair_hint.

The WAL-reset corruption warning's repair hint must match what the user's
install can actually do: source checkouts get the venv-setup step, packaged
installs get their steward's update command (#75153).
"""

from unittest.mock import patch

import hermes_state


def _hint_for(method):
    with patch("hermes_cli.config.detect_install_method", return_value=method), \
         patch("hermes_cli.config.recommended_update_command_for_method",
               return_value=f"CMD-{method}"):
        return hermes_state._wal_reset_repair_hint()


def test_source_checkout_gets_venv_setup_hint():
    assert _hint_for("source") == (
        "source checkouts can repair the runtime by re-running their venv setup"
    )


def test_git_install_gets_hermes_managed_hint():
    assert _hint_for("git") == (
        "Hermes-managed installs can repair the embedded runtime with `CMD-git`"
    )


def test_docker_gets_container_image_hint():
    assert _hint_for("docker") == "update the container image with `CMD-docker`"


def test_nix_passes_through_recommended_command():
    assert _hint_for("nix") == "CMD-nix"


def test_hint_falls_back_when_detection_fails():
    with patch("hermes_cli.config.detect_install_method",
               side_effect=RuntimeError("boom")):
        hint = hermes_state._wal_reset_repair_hint()
    assert "SQLite 3.51.3+" in hint
