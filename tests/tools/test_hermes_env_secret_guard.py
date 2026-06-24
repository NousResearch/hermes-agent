"""Regression tests for protecting Hermes .env secrets from agent erasure."""

from __future__ import annotations

import json
import os
from pathlib import Path

from tools.approval import (
    check_dangerous_command,
    disable_session_yolo,
    enable_session_yolo,
    reset_current_session_key,
    set_current_session_key,
)
from tools.file_tools import write_file_tool


def _hermes_env_path() -> Path:
    return Path(os.environ["HERMES_HOME"]) / ".env"


def _seed_env_file() -> Path:
    path = _hermes_env_path()
    path.write_text(
        "HASS_TOKEN=hass-placeholder\n"
        "HASS_URL=http://homeassistant.local:8123\n"
        "TELEGRAM_BOT_TOKEN=telegram-placeholder\n"
        "TELEGRAM_ALLOWED_USERS=12345\n",
        encoding="utf-8",
    )
    return path


def test_write_file_blocks_replacing_hermes_env_when_existing_secret_key_would_be_removed():
    env_path = _seed_env_file()
    original = env_path.read_text(encoding="utf-8")

    result = json.loads(
        write_file_tool(
            str(env_path),
            "HASS_TOKEN=hass-placeholder\nHASS_URL=http://homeassistant.local:8123\n",
        )
    )

    assert result.get("error")
    assert "would remove existing secret keys" in result["error"]
    assert "HASS_TOKEN" not in result["error"]
    assert "TELEGRAM_BOT_TOKEN" not in result["error"]
    assert env_path.read_text(encoding="utf-8") == original


def test_write_file_allows_replacing_hermes_env_when_existing_secret_keys_are_preserved():
    env_path = _seed_env_file()

    result = json.loads(
        write_file_tool(
            str(env_path),
            "# Hermes secrets\n"
            "HASS_TOKEN=hass-placeholder\n"
            "HASS_URL=http://homeassistant.local:8123\n"
            "TELEGRAM_BOT_TOKEN=telegram-placeholder\n"
            "TELEGRAM_ALLOWED_USERS=12345\n"
            "API_SERVER_ENABLED=true\n",
        )
    )

    assert result.get("error") is None
    written = env_path.read_text(encoding="utf-8")
    assert "HASS_TOKEN=" in written
    assert "TELEGRAM_BOT_TOKEN=" in written
    assert "API_SERVER_ENABLED=true" in written


def test_terminal_yolo_still_hardline_blocks_hermes_env_truncation_patterns():
    token = set_current_session_key("hermes-env-guard")
    try:
        enable_session_yolo("hermes-env-guard")
        hermes_home = os.environ["HERMES_HOME"]
        commands = [
            'printf "HASS_TOKEN=replacement\\n" > "$HERMES_HOME/.env"',
            f"cp /tmp/replacement-env {hermes_home}/.env",
            f"mv /tmp/replacement-env {hermes_home}/.env",
            'truncate -s 0 "$HERMES_HOME/.env"',
        ]

        for command in commands:
            result = check_dangerous_command(command, "local")
            assert result["approved"] is False, command
            assert result.get("hardline") is True, command
            assert "Hermes .env" in result["message"]
    finally:
        disable_session_yolo("hermes-env-guard")
        reset_current_session_key(token)
