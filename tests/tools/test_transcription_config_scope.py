"""STT background config reads must resolve only the owning profile's secrets."""
import logging
import os

import pytest

from agent import secret_scope
from tools.transcription_tools import _load_stt_config, _configured_idle_unload_seconds


@pytest.mark.parametrize("routed", [False, True])
def test_unscoped_stt_config_uses_profile_dotenv(tmp_path, monkeypatch, caplog, routed):
    from hermes_constants import set_hermes_home_override, reset_hermes_home_override

    launch = tmp_path / "launch"
    launch.mkdir()
    (launch / ".env").write_text("OPENROUTER_API_KEY=launch-key\n")
    monkeypatch.setenv("HERMES_HOME", str(launch if routed else tmp_path))
    home_token = set_hermes_home_override(tmp_path if routed else None)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    (tmp_path / ".env").write_text("OPENROUTER_API_KEY=own-key\n")
    (tmp_path / "config.yaml").write_text(
        "model:\n  api_key: ${env:OPENROUTER_API_KEY}\n"
        "stt:\n  provider: local\n  local:\n    unload_after_idle_seconds: 123\n"
        "  openai:\n    api_key: ${env:OPENROUTER_API_KEY}\n"
        "    base_url: https://openrouter.ai/api/v1\n"
    )
    token = secret_scope.set_secret_scope(None)
    mode = secret_scope.set_multiplex_context(True)
    try:
        with caplog.at_level(logging.WARNING):
            assert _configured_idle_unload_seconds() == 123
            cfg = _load_stt_config()
        assert cfg["openai"]["api_key"] == "own-key"
        assert cfg["openai"]["base_url"] == "https://openrouter.ai/api/v1"
        assert not [r for r in caplog.records if "Config ref" in r.message]
        assert secret_scope.current_secret_scope() is None
        assert "OPENROUTER_API_KEY" not in os.environ
    finally:
        reset_hermes_home_override(home_token)
        secret_scope.reset_multiplex_context(mode)
        secret_scope.reset_secret_scope(token)


def test_bound_stt_scope_is_authoritative(tmp_path, monkeypatch, caplog):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("OPENROUTER_API_KEY", "foreign-process-key")
    (tmp_path / ".env").write_text("OPENROUTER_API_KEY=foreign-file-key\n")
    (tmp_path / "config.yaml").write_text(
        "stt:\n  openai:\n    api_key: ${env:OPENROUTER_API_KEY}\n"
    )
    mode = secret_scope.set_multiplex_context(True)
    token = secret_scope.set_secret_scope({})
    try:
        with caplog.at_level(logging.WARNING):
            assert _load_stt_config()["openai"]["api_key"] == "${env:OPENROUTER_API_KEY}"
        assert any("Config ref" in r.message for r in caplog.records)
    finally:
        secret_scope.reset_secret_scope(token)
        secret_scope.reset_multiplex_context(mode)
