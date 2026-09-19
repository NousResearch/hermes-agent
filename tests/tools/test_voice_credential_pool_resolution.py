"""Tests for ``resolve_provider_secret`` — the single owner of STT/TTS
provider key resolution (#68003).

Keys added via ``hermes auth add <provider>`` live in the credential pool /
auth store and used to be invisible to the voice tools, which only read
``os.environ`` + ``~/.hermes/.env`` via ``get_env_value``. The shared
resolver falls back to the pool; env still wins when set; an explicit
config.yaml value wins over both; and under a multiplexed gateway turn the
profile secret scope stays authoritative (no pool borrow).
"""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from tools.tool_backend_helpers import resolve_provider_secret


def _fake_pool(key: str = "", *, has: bool = True):
    entry = SimpleNamespace(runtime_api_key=key, access_token=key) if key else None
    return SimpleNamespace(
        has_credentials=lambda: has and bool(key),
        peek=lambda: entry,
    )


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for var in (
        "GROQ_API_KEY", "MISTRAL_API_KEY", "ELEVENLABS_API_KEY",
        "DEEPINFRA_API_KEY", "MINIMAX_API_KEY", "GEMINI_API_KEY",
        "GOOGLE_API_KEY", "XAI_API_KEY", "OPENAI_API_KEY",
        "VOICE_TOOLS_OPENAI_KEY",
    ):
        monkeypatch.delenv(var, raising=False)
    yield


@pytest.fixture(autouse=True)
def _no_dotenv(monkeypatch):
    """Keep the developer's real ~/.hermes/.env out of these tests."""
    import hermes_cli.config as config_mod

    monkeypatch.setattr(config_mod, "load_env", lambda: {})
    yield


class TestPoolFallback:
    """Each provider's key resolves from the credential pool when env is empty."""

    @pytest.mark.parametrize(
        "env_var,provider_id",
        [
            ("GROQ_API_KEY", "groq"),
            ("MISTRAL_API_KEY", "mistral"),
            ("ELEVENLABS_API_KEY", "elevenlabs"),
            ("DEEPINFRA_API_KEY", "deepinfra"),
            ("MINIMAX_API_KEY", "minimax"),
            ("GEMINI_API_KEY", "gemini"),
            ("XAI_API_KEY", "xai"),
            ("OPENAI_API_KEY", "openai-api"),
        ],
    )
    def test_pool_entry_resolves_when_env_empty(self, env_var, provider_id):
        pool_key_seen = []

        def fake_load_pool(pid):
            pool_key_seen.append(pid)
            if pid == provider_id:
                return _fake_pool(f"pool-key-{provider_id}")
            return _fake_pool("")

        with patch("agent.credential_pool.load_pool", side_effect=fake_load_pool):
            assert resolve_provider_secret(env_var, provider_id) == (
                f"pool-key-{provider_id}"
            )
        assert provider_id in pool_key_seen


    def test_pool_read_failure_never_raises(self):
        with patch(
            "agent.credential_pool.load_pool", side_effect=Exception("disk error")
        ):
            assert resolve_provider_secret("MISTRAL_API_KEY", "mistral") == ""


class TestEnvPrecedence:
    """Env / .env still wins over the pool when set — unchanged behaviour."""

    def test_env_wins_over_pool(self, monkeypatch):
        monkeypatch.setenv("ELEVENLABS_API_KEY", "env-key")
        with patch(
            "agent.credential_pool.load_pool",
            return_value=_fake_pool("pool-key"),
        ) as lp:
            assert (
                resolve_provider_secret("ELEVENLABS_API_KEY", "elevenlabs")
                == "env-key"
            )
        lp.assert_not_called()

    def test_env_getter_is_consulted(self):
        """Callers can pass their module-level get_env_value wrapper."""
        with patch(
            "agent.credential_pool.load_pool", return_value=_fake_pool("")
        ):
            assert (
                resolve_provider_secret(
                    "GROQ_API_KEY",
                    "groq",
                    env_getter=lambda name: "dotenv-key",
                )
                == "dotenv-key"
            )


class TestConfigPrecedence:
    def test_config_value_wins_over_env_and_pool(self, monkeypatch):
        monkeypatch.setenv("MISTRAL_API_KEY", "env-key")
        with patch(
            "agent.credential_pool.load_pool",
            return_value=_fake_pool("pool-key"),
        ):
            assert (
                resolve_provider_secret(
                    "MISTRAL_API_KEY", "mistral", config_value="cfg-key"
                )
                == "cfg-key"
            )


class TestMultiplexScope:
    """Under multiplexing the profile scope is authoritative — no pool borrow."""

    @pytest.fixture(autouse=True)
    def _reset_multiplex(self):
        from agent import secret_scope as ss

        ss.set_multiplex_active(False)
        yield
        ss.set_multiplex_active(False)

    def test_scope_value_wins(self, monkeypatch):
        from agent import secret_scope as ss

        monkeypatch.setenv("MISTRAL_API_KEY", "sk-other-profile")
        ss.set_multiplex_active(True)
        token = ss.set_secret_scope({"MISTRAL_API_KEY": "sk-this-profile"})
        try:
            assert (
                resolve_provider_secret("MISTRAL_API_KEY", "mistral")
                == "sk-this-profile"
            )
        finally:
            ss.reset_secret_scope(token)

    def test_scope_miss_does_not_fall_through_to_pool(self):
        from agent import secret_scope as ss

        ss.set_multiplex_active(True)
        token = ss.set_secret_scope({"UNRELATED": "x"})
        try:
            with patch(
                "agent.credential_pool.load_pool",
                return_value=_fake_pool("pool-key"),
            ) as lp:
                assert resolve_provider_secret("MISTRAL_API_KEY", "mistral") == ""
            lp.assert_not_called()
        finally:
            ss.reset_secret_scope(token)


class TestToolWiring:
    """The tools' module-level helpers delegate to the shared resolver."""

    def test_transcription_tools_delegates(self):
        from tools import transcription_tools as tt

        def fake_load_pool(pid):
            return _fake_pool("stt-pool-key" if pid == "groq" else "")

        with patch("agent.credential_pool.load_pool", side_effect=fake_load_pool):
            assert tt._resolve_provider_key("GROQ_API_KEY", "groq") == "stt-pool-key"


    def test_openai_audio_key_falls_back_to_pool(self):
        from tools.tool_backend_helpers import resolve_openai_audio_api_key

        def fake_load_pool(pid):
            return _fake_pool("oai-pool-key" if pid == "openai-api" else "")

        with patch("agent.credential_pool.load_pool", side_effect=fake_load_pool):
            assert resolve_openai_audio_api_key() == "oai-pool-key"
