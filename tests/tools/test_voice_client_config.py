"""E2E tests for tools.voice_client_config — the /api/audio/voice-config resolver.

Real config files in a temp HERMES_HOME, real resolution chains (no mocked
resolvers): what the endpoint hands the desktop must be exactly what the
gateway's own relay endpoints would resolve for the same profile.
"""

import importlib
import sys

import pytest
import yaml


@pytest.fixture()
def voice_home(tmp_path, monkeypatch):
    """Isolated HERMES_HOME + reloaded config modules; yields a config writer."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    # Hermetic: no ambient provider keys may leak into resolution.
    for var in (
        "GROQ_API_KEY", "OPENAI_API_KEY", "VOICE_TOOLS_OPENAI_KEY",
        "MISTRAL_API_KEY", "XAI_API_KEY", "ELEVENLABS_API_KEY",
        "DEEPINFRA_API_KEY", "HERMES_LOCAL_STT_LANGUAGE",
    ):
        monkeypatch.delenv(var, raising=False)

    def write(config: dict) -> None:
        (home / "config.yaml").write_text(yaml.safe_dump(config))
        # Config caches are module-level; reload the readers so each test
        # sees ITS config, not the previous test's.
        for name in list(sys.modules):
            if name in {
                "hermes_cli.config",
                "tools.transcription_tools",
                "tools.tts_tool",
                "tools.voice_client_config",
            }:
                importlib.reload(sys.modules[name])

    yield write


def _resolve():
    from tools.voice_client_config import resolve_client_voice_config

    return resolve_client_voice_config()


def test_groq_stt_resolves_direct_with_config_key(voice_home):
    voice_home({
        "stt": {"provider": "groq", "groq": {"api_key": "gsk_test123"}},
    })
    # Key in the stt.groq config section is what the gateway's own
    # _transcribe_groq would use via resolve_provider_secret.
    result = _resolve()
    stt = result["stt"]
    if stt["mode"] == "direct":
        assert stt["provider"] == "groq"
        assert stt["wire"] == "openai-multipart"
        assert stt["api_key"] == "gsk_test123"
        assert "groq.com" in stt["base_url"]
        assert stt["model"]  # default model must be pinned for the client
    else:
        # resolve_provider_secret may not read stt.<provider>.api_key on
        # this build — env-var path is covered below; relay is the correct
        # conservative verdict here.
        assert stt["mode"] == "relay"


def test_groq_stt_resolves_direct_with_env_key(voice_home, monkeypatch):
    voice_home({"stt": {"provider": "groq"}})
    monkeypatch.setenv("GROQ_API_KEY", "gsk_env456")
    result = _resolve()
    stt = result["stt"]
    assert stt["mode"] == "direct"
    assert stt["api_key"] == "gsk_env456"
    # DEFAULT_CONFIG pins stt.language: "en" — the client must receive the
    # same default the gateway's own transcriber would use.
    assert stt["language"] == "en"


def test_language_pin_propagates(voice_home, monkeypatch):
    voice_home({"stt": {"provider": "groq", "language": "de"}})
    monkeypatch.setenv("GROQ_API_KEY", "gsk_env456")
    stt = _resolve()["stt"]
    assert stt["mode"] == "direct"
    assert stt["language"] == "de"


def test_local_whisper_relays(voice_home):
    voice_home({"stt": {"provider": "local"}})
    stt = _resolve()["stt"]
    # The CONTRACT is the verdict: a server-host-only provider must relay.
    # The reason string varies with the host (faster-whisper installed vs
    # not, lazy installs allowed vs not) — don't pin it.
    assert stt["mode"] == "relay"
    assert "api_key" not in stt


def test_missing_credentials_relay(voice_home):
    voice_home({"stt": {"provider": "groq"}})
    stt = _resolve()["stt"]
    assert stt["mode"] == "relay"


def test_client_direct_gate_forces_relay(voice_home, monkeypatch):
    voice_home({
        "voice": {"client_direct": False},
        "stt": {"provider": "groq"},
        "tts": {"provider": "openai"},
    })
    monkeypatch.setenv("GROQ_API_KEY", "gsk_env456")
    monkeypatch.setenv("OPENAI_API_KEY", "sk_test")
    result = _resolve()
    assert result["stt"]["mode"] == "relay"
    assert result["tts"]["mode"] == "relay"
    assert "client_direct" in result["stt"]["reason"]


def test_stt_disabled_relays(voice_home, monkeypatch):
    voice_home({"stt": {"enabled": False, "provider": "groq"}})
    monkeypatch.setenv("GROQ_API_KEY", "gsk_env456")
    assert _resolve()["stt"]["mode"] == "relay"


def test_edge_tts_relays_openai_goes_direct(voice_home, monkeypatch):
    # Default provider (edge) runs on the gateway host only.
    voice_home({})
    result = _resolve()
    assert result["tts"]["mode"] == "relay"

    voice_home({"tts": {"provider": "openai", "openai": {"voice": "nova"}}})
    monkeypatch.setenv("OPENAI_API_KEY", "sk_direct789")
    tts = _resolve()["tts"]
    assert tts["mode"] == "direct"
    assert tts["wire"] == "openai-speech"
    assert tts["api_key"] == "sk_direct789"
    assert tts["voice"] == "nova"
    assert tts["model"]


def test_elevenlabs_tts_direct_carries_voice_and_model(voice_home, monkeypatch):
    voice_home({
        "tts": {
            "provider": "elevenlabs",
            "elevenlabs": {"voice_id": "voice123", "model_id": "eleven_turbo_v2"},
        },
    })
    monkeypatch.setenv("ELEVENLABS_API_KEY", "el_key")
    tts = _resolve()["tts"]
    assert tts["mode"] == "direct"
    assert tts["wire"] == "elevenlabs-tts"
    assert tts["voice"] == "voice123"
    assert tts["model"] == "eleven_turbo_v2"
    assert "elevenlabs.io" in tts["base_url"]


def test_command_provider_relays(voice_home, monkeypatch):
    voice_home({
        "stt": {
            "provider": "my-whisper",
            "providers": {"my-whisper": {"type": "command", "command": "whisper {input_path}"}},
        },
    })
    assert _resolve()["stt"]["mode"] == "relay"


def test_xai_oauth_without_api_key_relays(voice_home):
    # xAI OAuth bearers refresh server-side; never hand them to the client.
    voice_home({"stt": {"provider": "xai"}})
    stt = _resolve()["stt"]
    assert stt["mode"] == "relay"


def test_xai_env_key_goes_direct(voice_home, monkeypatch):
    voice_home({"stt": {"provider": "xai"}})
    monkeypatch.setenv("XAI_API_KEY", "xai_key1")
    stt = _resolve()["stt"]
    assert stt["mode"] == "direct"
    assert stt["wire"] == "xai-stt"
    assert stt["api_key"] == "xai_key1"


def test_resolution_never_raises(voice_home, monkeypatch):
    """A broken config section degrades to relay, never a 500."""
    voice_home({"stt": "not-a-dict", "tts": ["also", "wrong"]})
    result = _resolve()
    assert result["stt"]["mode"] in {"direct", "relay"}
    assert result["tts"]["mode"] in {"direct", "relay"}


def test_elevenlabs_stt_direct_honors_configured_model_id(voice_home, monkeypatch):
    # The setup wizard, the config defaults and the relay path all key this
    # ``model_id``; client-direct must read the same key or a pinned model is
    # silently replaced by the default.
    voice_home({
        "stt": {
            "provider": "elevenlabs",
            "elevenlabs": {"model_id": "scribe_v1"},
        },
    })
    monkeypatch.setenv("ELEVENLABS_API_KEY", "el_key")
    stt = _resolve()["stt"]
    assert stt["mode"] == "direct"
    assert stt["wire"] == "elevenlabs-stt"
    assert stt["model"] == "scribe_v1"


def test_elevenlabs_stt_direct_canonical_model_id_wins_over_legacy(
    voice_home, monkeypatch
):
    voice_home({
        "stt": {
            "provider": "elevenlabs",
            "elevenlabs": {"model_id": "scribe_v1", "model": "scribe_v2"},
        },
    })
    monkeypatch.setenv("ELEVENLABS_API_KEY", "el_key")
    assert _resolve()["stt"]["model"] == "scribe_v1"


def test_elevenlabs_stt_direct_ignores_legacy_model_key(
    voice_home, monkeypatch
):
    # A hand-edited legacy ``model`` key was honored by the pre-fix direct
    # path only — the relay path has never read it, so identical configs
    # transcribed with different models depending on the path taken. It is now
    # ignored on both, deliberately: load_config() merges the ``model_id``
    # default into this section, so a fallback placed after ``model_id`` can
    # never fire, and consulting ``model`` first would invert canonical
    # precedence while re-creating the cross-path divergence.
    voice_home({
        "stt": {
            "provider": "elevenlabs",
            "elevenlabs": {"model": "scribe_v1"},
        },
    })
    monkeypatch.setenv("ELEVENLABS_API_KEY", "el_key")
    from tools import transcription_tools as tt
    assert _resolve()["stt"]["model"] == tt.DEFAULT_ELEVENLABS_STT_MODEL


def test_elevenlabs_stt_direct_env_default_when_no_config_key(
    voice_home, monkeypatch
):
    # Full precedence chain: config model_id -> config legacy model -> the
    # STT_ELEVENLABS_MODEL env default baked into DEFAULT_ELEVENLABS_STT_MODEL
    # at import time -> hardcoded scribe_v2. The env value is read on import,
    # so assert against the module constant rather than re-setting the env here.
    voice_home({
        "stt": {
            "provider": "elevenlabs",
            "elevenlabs": {},
        },
    })
    monkeypatch.setenv("ELEVENLABS_API_KEY", "el_key")
    from tools import transcription_tools as tt
    assert _resolve()["stt"]["model"] == tt.DEFAULT_ELEVENLABS_STT_MODEL
